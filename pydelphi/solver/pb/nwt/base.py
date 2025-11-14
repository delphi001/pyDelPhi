#!/usr/bin/env python
# coding: utf-8

# This file is part of pyDelPhi.
# Copyright (C) 2025 The pyDelPhi Project and contributors.
#
# pyDelPhi is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# pyDelPhi is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with pyDelPhi. If not, see <https://www.gnu.org/licenses/>.


"""
This module implements iterative relaxation methods for solving Poisson-like equations
on a 3D grid, optimized for performance using Numba and CUDA.

It supports single, double, and mixed precision based on configuration,
and provides both CPU and CUDA implementations for core iteration routines.
"""

import numpy as np
from numba import njit, prange, cuda, float64
import math
from math import sin, sqrt

from numba import njit, prange, cuda

from pydelphi.foundation.enums import Precision
from pydelphi.config.global_runtime import (
    PRECISION,
    delphi_bool,
    delphi_int,
    delphi_real,
)

from pydelphi.constants import (
    XYZ_COMPONENTS,
    HALF_GRID_OFFSET_LAGGING,
    HALF_GRID_OFFSET_LEADING,
    BOX_BOUNDARY,
    BOX_HOMO_EPSILON,
    BOX_ION_ACCESSIBLE,
)

# Import precision-specific utils
precision = PRECISION.value
if precision in (Precision.SINGLE.value,):
    from pydelphi.utils.prec.single import *

    try:
        from pydelphi.utils.cuda.single import *  # Optional CUDA utils for single precision
    except ImportError:
        pass

elif precision == Precision.DOUBLE.value:
    from pydelphi.utils.prec.double import *

    try:
        from pydelphi.utils.cuda.double import *  # Optional CUDA utils for double precision
    except ImportError:
        pass

BLOCK_SIZE = 1024  # safe across all GPUs
PHI_CLIP = 50.0
SINH_PHI_CLIP = math.sinh(PHI_CLIP)


@njit(inline="always")
def safe_sinh_cosh_unified(phi, phi_clip, sinh_phi_clip):
    """
    Stable sinh/cosh evaluation preserving cosh²−sinh²≈1
    and avoiding overflow beyond the clipping limit |phi| > clip.

    Args:
        phi (float): Input potential value.
        phi_clip (float): Maximum allowed |phi| before switching to linearized tail.
        sinh_phi_clip (float): Precomputed sinh(clip) ≈ 0.5 * exp(clip).

    Returns:
        (sinh_phi, cosh_phi): Tuple of stable sinh/cosh values.
    """
    if phi > phi_clip:
        delta = phi - phi_clip
        sinh_phi = sinh_phi_clip + sinh_phi_clip * delta
        cosh_phi = math.sqrt(1.0 + sinh_phi * sinh_phi)
    elif phi < -phi_clip:
        delta = phi + phi_clip
        sinh_phi = -sinh_phi_clip + sinh_phi_clip * delta
        cosh_phi = math.sqrt(1.0 + sinh_phi * sinh_phi)
    else:
        e = math.exp(phi)
        e_inv = 1.0 / e
        sinh_phi = 0.5 * (e - e_inv)
        cosh_phi = 0.5 * (e + e_inv)
    return sinh_phi, cosh_phi


# Step 2: CUDA device specialization reuses same source
@cuda.jit(device=True, inline=True)
def cu_safe_sinh_cosh_unified(phi, phi_clip, sinh_phi_clip):
    return safe_sinh_cosh_unified(phi, phi_clip, sinh_phi_clip)


# --- Core Iteration Functions ---


@cuda.jit(cache=True)
def _cuda_reset_rmsd_and_dphi(sum_sq_out, max_change_out):
    """
    Zero out global accumulators for RMSD (ΣΔφ²) and maximum potential change (Δφ_max)
    on the device before any SOR iteration block that performs atomic reductions.

    ⚠️ Must be called immediately before `_cuda_iterate_SOR_odd_with_dphi_rmsd`
    to ensure the global accumulators start from zero for each block.

    This kernel is single-threaded and operates entirely on device memory, avoiding
    PCIe round-trips. Typically launched as: `_cuda_reset_rmsd_and_dphi[1, 1](...)`.

    Args:
        sum_sq_out (float64[1]): Device scalar accumulator for RMSD (sum of squared Δφ).
        max_change_out (float64[1]): Device scalar accumulator for maximum Δφ magnitude.

    Mutates:
        sum_sq_out: Reset in place to 0.0.
        max_change_out: Reset in place to 0.0.
    """
    if cuda.threadIdx.x == 0:
        sum_sq_out[0] = 0.0
        max_change_out[0] = 0.0


@cuda.jit(cache=True)
def _cuda_iterate_nwt(
    vacuum: delphi_bool,
    even_odd: delphi_int,
    non_zero_salt: delphi_bool,
    approx_zero: delphi_real,
    omega_adaptive: delphi_real,
    grid_shape: np.ndarray[delphi_int],
    phi_map_current_half_1d: np.ndarray[delphi_real],
    phi_map_next_half_1d: np.ndarray[delphi_real],
    salt_ions_solvation_penalty_map_1d: np.ndarray[delphi_real],
    epsilon_map_midpoints_1d: np.ndarray[delphi_real],
    epsilon_sum_neighbors_only_1d: np.ndarray[delphi_real],
    boundary_flags_1d: np.ndarray[delphi_bool],
    charge_map_1d: np.ndarray[delphi_real],
    ion_exclusion_map_1d: np.ndarray[delphi_real],
) -> None:
    """
    CUDA kernel for one **half-iteration** of the nonlinear Newton (NWT) solver with
    in-kernel RMSD and Δφ tracking.

    Args:
        vacuum (delphi_bool): Whether this run corresponds to the vacuum phase.
        even_odd (delphi_int): 0 for even, 1 for odd half-grid iteration.
        non_zero_salt (delphi_bool): Whether ionic strength is non-zero.
        approx_zero (delphi_real): Threshold to neglect small charge densities.
        omega_adaptive (delphi_real): Adaptive damping factor to use in NWT.
        grid_shape (np.ndarray[delphi_int]): Grid dimensions (nx, ny, nz).
        phi_map_current_half_1d (np.ndarray[delphi_real]): Current φ half-map (read only).
        phi_map_next_half_1d (np.ndarray[delphi_real]): Updated φ half-map (in place).
        salt_ions_solvation_penalty_map_1d (np.ndarray[delphi_real]):
            Precomputed κ²h²ε₀ weighting term for ionic contribution.
        epsilon_map_midpoints_1d (np.ndarray[delphi_real]): Dielectric constants at midpoints.
        epsilon_sum_neighbors_only_1d (np.ndarray[delphi_real]):
            Sum of ε from 6 neighbors plus salt screening term.
        boundary_flags_1d (np.ndarray[delphi_bool]):
            Bitmask encoding BOX_BOUNDARY, BOX_HOMO_EPSILON, BOX_ION_ACCESSIBLE.
        charge_map_1d (np.ndarray[delphi_real]): Charge density map.
        ion_exclusion_map_1d (np.ndarray[delphi_real]): Accessibility map (used indirectly).
    """
    nx, ny, nz = grid_shape
    y_stride = nz
    x_stride = ny * y_stride

    y_stride_x_3 = y_stride * XYZ_COMPONENTS
    x_stride_x_3 = x_stride * XYZ_COMPONENTS
    y_stride_x_3_minus_1 = y_stride_x_3 - 1

    y_stride_half = y_stride // 2
    x_stride_half = x_stride // 2

    if even_odd.item() == 0:
        z_stride_half_leading_offset = HALF_GRID_OFFSET_LEADING
        z_stride_half_lagging_offset = HALF_GRID_OFFSET_LAGGING
        y_stride_half_leading_offset = y_stride_half
        y_stride_half_lagging_offset = y_stride_half + 1
        x_stride_half_leading_offset = x_stride_half
        x_stride_half_lagging_offset = x_stride_half + 1
    elif even_odd.item() == 1:
        z_stride_half_leading_offset = HALF_GRID_OFFSET_LAGGING
        z_stride_half_lagging_offset = HALF_GRID_OFFSET_LEADING
        y_stride_half_leading_offset = y_stride_half + 1
        y_stride_half_lagging_offset = y_stride_half
        x_stride_half_leading_offset = x_stride_half + 1
        x_stride_half_lagging_offset = x_stride_half

    num_grid_points = nx * x_stride
    num_grid_points_half = (num_grid_points + 1) // 2

    ijk1d_half = cuda.grid(1)  # Get CUDA grid index
    ijk1d = 2 * ijk1d_half + even_odd.item()  # Convert half index to full index
    ijk1d_x_3 = ijk1d * XYZ_COMPONENTS

    omega_old_weight = 1.0 - omega_adaptive

    if ijk1d < num_grid_points:
        if (boundary_flags_1d[ijk1d] & BOX_BOUNDARY) != BOX_BOUNDARY:
            epsilon_sum_local = epsilon_sum_neighbors_only_1d[
                ijk1d
            ]  # Sum of neighbor epsilons

            # Retrieve phi values from neighbor grid points from current_half phi map
            phi_k_minus_1, phi_k_plus_1 = (
                phi_map_current_half_1d[ijk1d_half - z_stride_half_lagging_offset],
                phi_map_current_half_1d[ijk1d_half + z_stride_half_leading_offset],
            )
            phi_j_minus_1, phi_j_plus_1 = (
                phi_map_current_half_1d[ijk1d_half - y_stride_half_lagging_offset],
                phi_map_current_half_1d[ijk1d_half + y_stride_half_leading_offset],
            )
            phi_i_minus_1, phi_i_plus_1 = (
                phi_map_current_half_1d[ijk1d_half - x_stride_half_lagging_offset],
                phi_map_current_half_1d[ijk1d_half + x_stride_half_leading_offset],
            )

            numerator = 0.0  # ensure defined regardless of branching
            # Calculate numerator for NWT update
            if (boundary_flags_1d[ijk1d] & BOX_HOMO_EPSILON) == BOX_HOMO_EPSILON:
                eps = epsilon_map_midpoints_1d[ijk1d_x_3]
                phi_sum = (
                    phi_k_minus_1
                    + phi_k_plus_1
                    + phi_j_minus_1
                    + phi_j_plus_1
                    + phi_i_minus_1
                    + phi_i_plus_1
                )
                numerator = eps * phi_sum
            else:
                # Retrieve epsilon values for neighbor midpoints
                eps_k_minus_half = epsilon_map_midpoints_1d[
                    ijk1d_x_3 - HALF_GRID_OFFSET_LAGGING
                ]
                eps_k_plus_half = epsilon_map_midpoints_1d[ijk1d_x_3 + 2]
                eps_j_minus_half = epsilon_map_midpoints_1d[
                    ijk1d_x_3 - y_stride_x_3_minus_1
                ]
                eps_j_plus_half = epsilon_map_midpoints_1d[ijk1d_x_3 + 1]
                eps_i_minus_half = epsilon_map_midpoints_1d[ijk1d_x_3 - x_stride_x_3]
                eps_i_plus_half = epsilon_map_midpoints_1d[ijk1d_x_3]

                # Calculate $\sigma_{p=1}^{6}{phi_ijk_neighbor_p*eps_ijk_midpoint_p}$ term of numerator
                numerator = (
                    phi_k_minus_1 * eps_k_minus_half
                    + phi_k_plus_1 * eps_k_plus_half
                    + phi_j_minus_1 * eps_j_minus_half
                    + phi_j_plus_1 * eps_j_plus_half
                    + phi_i_minus_1 * eps_i_minus_half
                    + phi_i_plus_1 * eps_i_plus_half
                )

            denominator = epsilon_sum_local
            charge_density = charge_map_1d[ijk1d]  # Local charge density

            phimap_ijk1d = phi_map_next_half_1d[ijk1d_half]

            # Non-linear screening due to salt
            if (
                (not vacuum)
                and non_zero_salt
                and (
                    (boundary_flags_1d[ijk1d] & BOX_ION_ACCESSIBLE)
                    == BOX_ION_ACCESSIBLE
                )
            ):
                sinh_phi, cosh_phi = cu_safe_sinh_cosh_unified(
                    phimap_ijk1d, PHI_CLIP, SINH_PHI_CLIP
                )

                salt_factor_numerator = salt_ions_solvation_penalty_map_1d[ijk1d] * (
                    phimap_ijk1d * cosh_phi - sinh_phi
                )
                salt_factor_denominator = (
                    salt_ions_solvation_penalty_map_1d[ijk1d] * cosh_phi
                )

                numerator += salt_factor_numerator
                denominator += salt_factor_denominator

            # Newton update with adaptive damping
            if abs(charge_density) > approx_zero:
                phi_candidate = (numerator + charge_density) / denominator
            else:
                phi_candidate = numerator / denominator

            # Adaptive relaxation
            updated_phi = (
                omega_old_weight * phimap_ijk1d + omega_adaptive * phi_candidate
            )

            phi_map_next_half_1d[ijk1d_half] = updated_phi


@cuda.jit(cache=True)
def _cuda_iterate_nwt_with_dphi_rmsd(
    vacuum: delphi_bool,
    even_odd: delphi_int,
    non_zero_salt: delphi_bool,
    approx_zero: delphi_real,
    omega_adaptive: delphi_real,
    grid_shape: np.ndarray[delphi_int],
    phi_map_current_half_1d: np.ndarray[delphi_real],
    phi_map_next_half_1d: np.ndarray[delphi_real],
    salt_ions_solvation_penalty_map_1d: np.ndarray[delphi_real],
    epsilon_map_midpoints_1d: np.ndarray[delphi_real],
    epsilon_sum_neighbors_only_1d: np.ndarray[delphi_real],
    boundary_flags_1d: np.ndarray[delphi_bool],
    charge_map_1d: np.ndarray[delphi_real],
    ion_exclusion_map_1d: np.ndarray[delphi_real],
    sum_sq_out,  # float64[1]
    max_change_out,  # float64[1]
):
    """
    CUDA kernel for one **half-iteration** of the nonlinear Newton (NWT) solver with
    in-kernel RMSD and Δφ tracking. This mirrors the structure of
    `_cuda_iterate_SOR_odd_with_dphi_rmsd`, including shared-memory accumulation
    and block-level reduction for ΣΔφ² and max|Δφ|.

    ⚠️ IMPORTANT:
        `_cuda_reset_rmsd_and_dphi()` **must be launched before** this kernel
        to zero the global accumulators `sum_sq_out` and `max_change_out`.

    Args:
        vacuum (delphi_bool): Whether this run corresponds to the vacuum phase.
        even_odd (delphi_int): 0 for even, 1 for odd half-grid iteration.
        non_zero_salt (delphi_bool): Whether ionic strength is non-zero.
        approx_zero (delphi_real): Threshold to neglect small charge densities.
        omega_adaptive (delphi_real): Adaptive damping factor to use in NWT.
        grid_shape (np.ndarray[delphi_int]): Grid dimensions (nx, ny, nz).
        phi_map_current_half_1d (np.ndarray[delphi_real]): Current φ half-map (read only).
        phi_map_next_half_1d (np.ndarray[delphi_real]): Updated φ half-map (in place).
        salt_ions_solvation_penalty_map_1d (np.ndarray[delphi_real]):
            Precomputed κ²h²ε₀ weighting term for ionic contribution.
        epsilon_map_midpoints_1d (np.ndarray[delphi_real]): Dielectric constants at midpoints.
        epsilon_sum_neighbors_only_1d (np.ndarray[delphi_real]):
            Sum of ε from 6 neighbors plus salt screening term.
        boundary_flags_1d (np.ndarray[delphi_bool]):
            Bitmask encoding BOX_BOUNDARY, BOX_HOMO_EPSILON, BOX_ION_ACCESSIBLE.
        charge_map_1d (np.ndarray[delphi_real]): Charge density map.
        ion_exclusion_map_1d (np.ndarray[delphi_real]): Accessibility map (used indirectly).
        sum_sq_out (float64[1]): Global accumulator for ΣΔφ² (RMSD).
        max_change_out (float64[1]): Global accumulator for max|Δφ|.
    """

    # Shared memory for block reductions
    shared_sq = cuda.shared.array(BLOCK_SIZE, float64)
    shared_max = cuda.shared.array(BLOCK_SIZE, float64)

    tid = cuda.threadIdx.x
    idx = cuda.grid(1)
    stride = cuda.gridsize(1)

    # Derived constants
    nx, ny, nz = grid_shape
    y_stride = nz
    x_stride = ny * y_stride
    num_grid_points = nx * x_stride
    num_grid_points_half = (num_grid_points + 1) // 2

    y_stride_x_3 = y_stride * XYZ_COMPONENTS
    x_stride_x_3 = x_stride * XYZ_COMPONENTS
    y_stride_x_3_minus_1 = y_stride_x_3 - 1

    # Even/odd dependent offsets
    if even_odd == 0:
        z_stride_half_leading_offset = HALF_GRID_OFFSET_LEADING
        z_stride_half_lagging_offset = HALF_GRID_OFFSET_LAGGING
        y_stride_half_leading_offset = y_stride // 2
        y_stride_half_lagging_offset = y_stride_half_leading_offset + 1
        x_stride_half_leading_offset = x_stride // 2
        x_stride_half_lagging_offset = x_stride_half_leading_offset + 1
    else:
        z_stride_half_leading_offset = HALF_GRID_OFFSET_LAGGING
        z_stride_half_lagging_offset = HALF_GRID_OFFSET_LEADING
        y_stride_half_leading_offset = y_stride // 2 + 1
        y_stride_half_lagging_offset = y_stride // 2
        x_stride_half_leading_offset = x_stride // 2 + 1
        x_stride_half_lagging_offset = x_stride // 2

    # Thread-local accumulators
    local_sq = 0.0
    local_max = 0.0

    omega_old_weight = 1.0 - omega_adaptive

    # -------------------------------
    # Main half-iteration loop
    # -------------------------------
    for ijk1d_half in range(idx, num_grid_points_half, stride):
        ijk1d = 2 * ijk1d_half + even_odd
        if ijk1d < num_grid_points:
            if (boundary_flags_1d[ijk1d] & BOX_BOUNDARY) != BOX_BOUNDARY:
                ijk1d_x_3 = ijk1d * XYZ_COMPONENTS
                eps_sum_local = epsilon_sum_neighbors_only_1d[ijk1d]

                # Neighbor φ values
                phi_k_minus_1 = phi_map_current_half_1d[
                    ijk1d_half - z_stride_half_lagging_offset
                ]
                phi_k_plus_1 = phi_map_current_half_1d[
                    ijk1d_half + z_stride_half_leading_offset
                ]
                phi_j_minus_1 = phi_map_current_half_1d[
                    ijk1d_half - y_stride_half_lagging_offset
                ]
                phi_j_plus_1 = phi_map_current_half_1d[
                    ijk1d_half + y_stride_half_leading_offset
                ]
                phi_i_minus_1 = phi_map_current_half_1d[
                    ijk1d_half - x_stride_half_lagging_offset
                ]
                phi_i_plus_1 = phi_map_current_half_1d[
                    ijk1d_half + x_stride_half_leading_offset
                ]

                # Numerator assembly
                if (boundary_flags_1d[ijk1d] & BOX_HOMO_EPSILON) == BOX_HOMO_EPSILON:
                    eps = epsilon_map_midpoints_1d[ijk1d_x_3]
                    phi_sum = (
                        phi_k_minus_1
                        + phi_k_plus_1
                        + phi_j_minus_1
                        + phi_j_plus_1
                        + phi_i_minus_1
                        + phi_i_plus_1
                    )
                    numerator = eps * phi_sum
                else:
                    eps_k_minus_half = epsilon_map_midpoints_1d[
                        ijk1d_x_3 - HALF_GRID_OFFSET_LAGGING
                    ]
                    eps_k_plus_half = epsilon_map_midpoints_1d[ijk1d_x_3 + 2]
                    eps_j_minus_half = epsilon_map_midpoints_1d[
                        ijk1d_x_3 - y_stride_x_3_minus_1
                    ]
                    eps_j_plus_half = epsilon_map_midpoints_1d[ijk1d_x_3 + 1]
                    eps_i_minus_half = epsilon_map_midpoints_1d[
                        ijk1d_x_3 - x_stride_x_3
                    ]
                    eps_i_plus_half = epsilon_map_midpoints_1d[ijk1d_x_3]
                    numerator = (
                        phi_k_minus_1 * eps_k_minus_half
                        + phi_k_plus_1 * eps_k_plus_half
                        + phi_j_minus_1 * eps_j_minus_half
                        + phi_j_plus_1 * eps_j_plus_half
                        + phi_i_minus_1 * eps_i_minus_half
                        + phi_i_plus_1 * eps_i_plus_half
                    )

                denominator = eps_sum_local
                charge_density = charge_map_1d[ijk1d]
                phi_ijk1d = phi_map_next_half_1d[ijk1d_half]
                updated_phi = phi_ijk1d

                # Nonlinear salt term (only if ion-accessible)
                if (
                    (not vacuum)
                    and non_zero_salt
                    and (
                        (boundary_flags_1d[ijk1d] & BOX_ION_ACCESSIBLE)
                        == BOX_ION_ACCESSIBLE
                    )
                ):
                    sinh_phi, cosh_phi = cu_safe_sinh_cosh_unified(
                        phi_ijk1d, PHI_CLIP, SINH_PHI_CLIP
                    )
                    salt_factor = salt_ions_solvation_penalty_map_1d[ijk1d]
                    numerator += salt_factor * (phi_ijk1d * cosh_phi - sinh_phi)
                    denominator += salt_factor * cosh_phi

                # Newton update with adaptive damping
                if abs(charge_density) > approx_zero:
                    phi_candidate = (numerator + charge_density) / denominator
                else:
                    phi_candidate = numerator / denominator

                # Adaptive relaxation
                updated_phi = (
                    omega_old_weight * phi_ijk1d + omega_adaptive * phi_candidate
                )

                phi_map_next_half_1d[ijk1d_half] = updated_phi

                # Δφ accumulation
                diff = updated_phi - phi_ijk1d
                local_sq += diff * diff
                abs_diff = math.fabs(diff)
                if abs_diff > local_max:
                    local_max = abs_diff

    # -------------------------------
    # Block-level reduction
    # -------------------------------
    shared_sq[tid] = local_sq
    shared_max[tid] = local_max
    cuda.syncthreads()

    s = cuda.blockDim.x // 2
    while s > 0:
        if tid < s:
            shared_sq[tid] += shared_sq[tid + s]
            if shared_max[tid + s] > shared_max[tid]:
                shared_max[tid] = shared_max[tid + s]
        s //= 2
        cuda.syncthreads()

    # Write block result to global atomics
    if tid == 0:
        cuda.atomic.add(sum_sq_out, 0, shared_sq[0])
        cuda.atomic.max(max_change_out, 0, shared_max[0])


@njit(nogil=True, boundscheck=False, parallel=True, cache=True)
def _cpu_iterate_nwt(
    vacuum: delphi_bool,
    even_odd: delphi_int,
    non_zero_salt: delphi_bool,
    approx_zero: delphi_real,
    omega_adaptive: delphi_real,
    grid_shape: np.ndarray[delphi_int],
    phi_map_current_half_1d: np.ndarray[delphi_real],
    phi_map_next_half_1d: np.ndarray[delphi_real],
    salt_ions_solvation_penalty_map_1d: np.ndarray[delphi_real],
    epsilon_map_midpoints_1d: np.ndarray[delphi_real],
    epsilon_sum_neighbors_only_1d: np.ndarray[delphi_real],
    boundary_flags_1d: np.ndarray[delphi_bool],
    charge_map_1d: np.ndarray[delphi_real],
    ion_exclusion_map_1d: np.ndarray[delphi_real],
) -> None:
    """
    CPU function for Successive Over-Relaxation (SOR) iteration, similar to CUDA version.

    Args:
        vacuum (delphi_bool): Whether is run corresponds to vacuum phase.
        even_odd (delphi_int): Flag for even/odd iterations.
        non_zero_salt (delphi_bool): Whether salt-concentration is non-zero.
        approx_zero (delphi_real): Threshold for negligible charge density.
        omega_adaptive (delphi_real): Adaptive damping factor to use in NWT.
        grid_shape (np.ndarray[delphi_int]): Shape of the 3D grid.
        phi_map_current_half_1d (np.ndarray[delphi_real]): Current phi values (odd or even half).
        phi_map_next_half_1d (np.ndarray[delphi_real]): Next phi values (even or odd half).
        epsilon_map_midpoints_1d (np.ndarray[delphi_real]): Epsilon values at midpoints.
        epsilon_sum_neighbors_only_1d (np.ndarray[delphi_real]): Sum of epsilons for neighbors.
        boundary_flags_1d (np.ndarray[delphi_bool]): Boundary grid point flags.
        charge_map_1d (np.ndarray[delphi_real]): Charge density map.
    """
    nx, ny, nz = grid_shape
    y_stride = nz
    x_stride = ny * y_stride

    y_stride_x_3 = y_stride * XYZ_COMPONENTS
    x_stride_x_3 = x_stride * XYZ_COMPONENTS
    y_stride_x_3_minus_1 = y_stride_x_3 - 1

    y_stride_half = y_stride // 2
    x_stride_half = x_stride // 2

    if even_odd == 0:
        z_stride_half_leading_offset = HALF_GRID_OFFSET_LEADING
        z_stride_half_lagging_offset = HALF_GRID_OFFSET_LAGGING
        y_stride_half_leading_offset = y_stride_half
        y_stride_half_lagging_offset = y_stride_half + 1
        x_stride_half_leading_offset = x_stride_half
        x_stride_half_lagging_offset = x_stride_half + 1
    elif even_odd == 1:
        z_stride_half_leading_offset = HALF_GRID_OFFSET_LAGGING
        z_stride_half_lagging_offset = HALF_GRID_OFFSET_LEADING
        y_stride_half_leading_offset = y_stride_half + 1
        y_stride_half_lagging_offset = y_stride_half
        x_stride_half_leading_offset = x_stride_half + 1
        x_stride_half_lagging_offset = x_stride_half

    num_grid_points = nx * x_stride
    num_grid_points_half = (num_grid_points + 1) // 2

    omega_old_weight = 1.0 - omega_adaptive

    # Parallel loop over half grid points
    for ijk1d_half in prange(num_grid_points_half):
        ijk1d = 2 * ijk1d_half + even_odd  # Convert half index to full index
        if ijk1d < num_grid_points:  # Boundary check
            ijk1d_x_3 = ijk1d * XYZ_COMPONENTS
            if (boundary_flags_1d[ijk1d] & BOX_BOUNDARY) != BOX_BOUNDARY:
                epsilon_sum_local = epsilon_sum_neighbors_only_1d[
                    ijk1d
                ]  # Sum of neighbor epsilons

                # Retrieve phi values from neighbor grid points from current_half phi map
                phi_k_minus_1, phi_k_plus_1 = (
                    phi_map_current_half_1d[ijk1d_half - z_stride_half_lagging_offset],
                    phi_map_current_half_1d[ijk1d_half + z_stride_half_leading_offset],
                )
                phi_j_minus_1, phi_j_plus_1 = (
                    phi_map_current_half_1d[ijk1d_half - y_stride_half_lagging_offset],
                    phi_map_current_half_1d[ijk1d_half + y_stride_half_leading_offset],
                )
                phi_i_minus_1, phi_i_plus_1 = (
                    phi_map_current_half_1d[ijk1d_half - x_stride_half_lagging_offset],
                    phi_map_current_half_1d[ijk1d_half + x_stride_half_leading_offset],
                )

                numerator = 0.0  # ensure defined regardless of branching
                # Calculate numerator for NWT update
                if (boundary_flags_1d[ijk1d] & BOX_HOMO_EPSILON) == BOX_HOMO_EPSILON:
                    eps = epsilon_map_midpoints_1d[ijk1d_x_3]
                    phi_sum = (
                        phi_k_minus_1
                        + phi_k_plus_1
                        + phi_j_minus_1
                        + phi_j_plus_1
                        + phi_i_minus_1
                        + phi_i_plus_1
                    )
                    numerator = eps * phi_sum
                else:
                    # Retrieve epsilon values for neighbor midpoints
                    eps_k_minus_half = epsilon_map_midpoints_1d[
                        ijk1d_x_3 - HALF_GRID_OFFSET_LAGGING
                    ]
                    eps_k_plus_half = epsilon_map_midpoints_1d[ijk1d_x_3 + 2]
                    eps_j_minus_half = epsilon_map_midpoints_1d[
                        ijk1d_x_3 - y_stride_x_3_minus_1
                    ]
                    eps_j_plus_half = epsilon_map_midpoints_1d[ijk1d_x_3 + 1]
                    eps_i_minus_half = epsilon_map_midpoints_1d[
                        ijk1d_x_3 - x_stride_x_3
                    ]
                    eps_i_plus_half = epsilon_map_midpoints_1d[ijk1d_x_3]

                    # Calculate $\sigma_{p=1}^{6}{phi_ijk_neighbor_p*eps_ijk_midpoint_p}$ term of numerator
                    numerator = (
                        phi_k_minus_1 * eps_k_minus_half
                        + phi_k_plus_1 * eps_k_plus_half
                        + phi_j_minus_1 * eps_j_minus_half
                        + phi_j_plus_1 * eps_j_plus_half
                        + phi_i_minus_1 * eps_i_minus_half
                        + phi_i_plus_1 * eps_i_plus_half
                    )

                denominator = epsilon_sum_local
                charge_density = charge_map_1d[ijk1d]
                phimap_ijk1d = phi_map_next_half_1d[ijk1d_half]

                if (
                    (not vacuum)
                    and non_zero_salt
                    and (
                        (boundary_flags_1d[ijk1d] & BOX_ION_ACCESSIBLE)
                        == BOX_ION_ACCESSIBLE
                    )
                ):
                    sinh_phi, cosh_phi = safe_sinh_cosh_unified(
                        phimap_ijk1d, PHI_CLIP, SINH_PHI_CLIP
                    )

                    salt_factor_numerator = salt_ions_solvation_penalty_map_1d[
                        ijk1d
                    ] * (phimap_ijk1d * cosh_phi - sinh_phi)
                    salt_factor_denominator = (
                        salt_ions_solvation_penalty_map_1d[ijk1d] * cosh_phi
                    )

                    numerator += salt_factor_numerator
                    denominator += salt_factor_denominator

                # Newton update with adaptive damping
                if abs(charge_density) > approx_zero:
                    phi_candidate = (numerator + charge_density) / denominator
                else:
                    phi_candidate = numerator / denominator

                # Adaptive relaxation
                updated_phi = (
                    omega_old_weight * phimap_ijk1d + omega_adaptive * phi_candidate
                )

                phi_map_next_half_1d[ijk1d_half] = updated_phi  # Update phi value


@njit(nogil=True, boundscheck=False, parallel=True, cache=True)
def _cpu_iterate_nwt_with_dphi_rmsd(
    vacuum: delphi_bool,
    even_odd: delphi_int,
    non_zero_salt: delphi_bool,
    approx_zero: delphi_real,
    omega_adaptive: delphi_real,
    grid_shape: np.ndarray[delphi_int],
    phi_map_current_half_1d: np.ndarray[delphi_real],
    phi_map_next_half_1d: np.ndarray[delphi_real],
    salt_ions_solvation_penalty_map_1d: np.ndarray[delphi_real],
    epsilon_map_midpoints_1d: np.ndarray[delphi_real],
    epsilon_sum_neighbors_only_1d: np.ndarray[delphi_real],
    is_boundary_gridpoint_1d: np.ndarray[delphi_bool],
    charge_map_1d: np.ndarray[delphi_real],
    ion_exclusion_map_1d: np.ndarray[delphi_real],
    num_cpu_threads: delphi_int,  # new explicit argument
) -> tuple[delphi_real, delphi_real]:
    """
    Fused CPU kernel for one Newton (NWT) half-iteration with concurrent RMSD/Δφ tracking.

    Performs a single NWT half-grid update (even or odd) while simultaneously computing
    the RMSD (ΣΔφ²) and maximum |Δφ| across all updated grid points.  This fused variant
    avoids a separate reduction pass and improves cache locality.

    Args:
        vacuum (delphi_bool): Whether this run corresponds to the vacuum phase.
        even_odd (delphi_int): Flag for even/odd half-grid iteration.
        non_zero_salt (delphi_bool): Whether ionic strength is non-zero.
        approx_zero (delphi_real): Threshold below which charge density is neglected.
        omega_adaptive (delphi_real): Adaptive damping factor to use in NWT.
        grid_shape (np.ndarray[delphi_int]): 3D grid dimensions (nx, ny, nz).
        phi_map_current_half_1d (np.ndarray[delphi_real]): Current φ half-map (read-only).
        phi_map_next_half_1d (np.ndarray[delphi_real]): Next φ half-map (updated in place).
        salt_ions_solvation_penalty_map_1d (np.ndarray[delphi_real]):
            Precomputed κ²h²ε₀ weighting term per grid point.
        epsilon_map_midpoints_1d (np.ndarray[delphi_real]):
            Dielectric constants at grid midpoints.
        epsilon_sum_neighbors_only_1d (np.ndarray[delphi_real]):
            Sum of neighboring ε values (Σεₙ) without salt contribution.
        is_boundary_gridpoint_1d (np.ndarray[delphi_bool]):
            Bitmask array encoding BOX_BOUNDARY, BOX_HOMO_EPSILON, and BOX_ION_ACCESSIBLE flags.
        charge_map_1d (np.ndarray[delphi_real]): Charge density per grid point.
        ion_exclusion_map_1d (np.ndarray[delphi_real]):
            Ion-accessibility map with values in the range [0.0, 1.0].
            Used for fractional accessibility in Gaussian models, but logical
            inclusion/exclusion is controlled by the BOX_ION_ACCESSIBLE flag.

    Mutates:
        phi_map_next_half_1d:
            Updated in place with the new φ values for the specified half-grid.

    Returns:
        tuple[delphi_real, delphi_real]:
            (sum_squared_dphi, max_abs_dphi)
            Sum of squared potential changes and the maximum absolute Δφ value.
    """
    nx, ny, nz = grid_shape
    y_stride = nz
    x_stride = ny * y_stride
    num_grid_points = nx * x_stride
    num_grid_points_half = (num_grid_points + 1) // 2

    y_stride_x_3 = y_stride * XYZ_COMPONENTS
    x_stride_x_3 = x_stride * XYZ_COMPONENTS
    y_stride_x_3_minus_1 = y_stride_x_3 - 1
    y_stride_half = y_stride // 2
    x_stride_half = x_stride // 2

    # Half-grid stride offsets
    if even_odd == 0:
        z_stride_half_leading_offset = HALF_GRID_OFFSET_LEADING
        z_stride_half_lagging_offset = HALF_GRID_OFFSET_LAGGING
        y_stride_half_leading_offset = y_stride_half
        y_stride_half_lagging_offset = y_stride_half + 1
        x_stride_half_leading_offset = x_stride_half
        x_stride_half_lagging_offset = x_stride_half + 1
    else:
        z_stride_half_leading_offset = HALF_GRID_OFFSET_LAGGING
        z_stride_half_lagging_offset = HALF_GRID_OFFSET_LEADING
        y_stride_half_leading_offset = y_stride_half + 1
        y_stride_half_lagging_offset = y_stride_half
        x_stride_half_leading_offset = x_stride_half + 1
        x_stride_half_lagging_offset = x_stride_half

    # Thread-local accumulators
    dphi_sq_partial = np.zeros(num_cpu_threads, dtype=delphi_real)
    max_dphi_partial = np.zeros(num_cpu_threads, dtype=delphi_real)

    # Divide total work evenly among threads
    chunk_size = (num_grid_points_half + num_cpu_threads - 1) // num_cpu_threads

    omega_old_weight = 1.0 - omega_adaptive

    for thread_id in prange(num_cpu_threads):
        start = thread_id * chunk_size
        end = min(start + chunk_size, num_grid_points_half)

        local_sq = 0.0
        local_max = 0.0

        for ijk1d_half in range(start, end):
            ijk1d = 2 * ijk1d_half + even_odd
            if ijk1d < num_grid_points:
                if (is_boundary_gridpoint_1d[ijk1d] & BOX_BOUNDARY) != BOX_BOUNDARY:
                    ijk1d_x_3 = ijk1d * XYZ_COMPONENTS
                    epsilon_sum_local = epsilon_sum_neighbors_only_1d[ijk1d]

                    # Neighbor potentials
                    phi_k_minus_1 = phi_map_current_half_1d[
                        ijk1d_half - z_stride_half_lagging_offset
                    ]
                    phi_k_plus_1 = phi_map_current_half_1d[
                        ijk1d_half + z_stride_half_leading_offset
                    ]
                    phi_j_minus_1 = phi_map_current_half_1d[
                        ijk1d_half - y_stride_half_lagging_offset
                    ]
                    phi_j_plus_1 = phi_map_current_half_1d[
                        ijk1d_half + y_stride_half_leading_offset
                    ]
                    phi_i_minus_1 = phi_map_current_half_1d[
                        ijk1d_half - x_stride_half_lagging_offset
                    ]
                    phi_i_plus_1 = phi_map_current_half_1d[
                        ijk1d_half + x_stride_half_leading_offset
                    ]

                    # Numerator
                    if (
                        is_boundary_gridpoint_1d[ijk1d] & BOX_HOMO_EPSILON
                    ) == BOX_HOMO_EPSILON:
                        eps = epsilon_map_midpoints_1d[ijk1d_x_3]
                        phi_sum = (
                            phi_k_minus_1
                            + phi_k_plus_1
                            + phi_j_minus_1
                            + phi_j_plus_1
                            + phi_i_minus_1
                            + phi_i_plus_1
                        )
                        numerator = eps * phi_sum
                    else:
                        eps_k_minus_half = epsilon_map_midpoints_1d[
                            ijk1d_x_3 - HALF_GRID_OFFSET_LAGGING
                        ]
                        eps_k_plus_half = epsilon_map_midpoints_1d[ijk1d_x_3 + 2]
                        eps_j_minus_half = epsilon_map_midpoints_1d[
                            ijk1d_x_3 - y_stride_x_3_minus_1
                        ]
                        eps_j_plus_half = epsilon_map_midpoints_1d[ijk1d_x_3 + 1]
                        eps_i_minus_half = epsilon_map_midpoints_1d[
                            ijk1d_x_3 - x_stride_x_3
                        ]
                        eps_i_plus_half = epsilon_map_midpoints_1d[ijk1d_x_3]
                        numerator = (
                            phi_k_minus_1 * eps_k_minus_half
                            + phi_k_plus_1 * eps_k_plus_half
                            + phi_j_minus_1 * eps_j_minus_half
                            + phi_j_plus_1 * eps_j_plus_half
                            + phi_i_minus_1 * eps_i_minus_half
                            + phi_i_plus_1 * eps_i_plus_half
                        )

                    denominator = epsilon_sum_local
                    charge_density = charge_map_1d[ijk1d]
                    old_phi = phi_map_next_half_1d[ijk1d_half]
                    updated_phi = old_phi

                    # Salt-coupled nonlinear correction
                    if (
                        (not vacuum)
                        and non_zero_salt
                        and (
                            (is_boundary_gridpoint_1d[ijk1d] & BOX_ION_ACCESSIBLE)
                            == BOX_ION_ACCESSIBLE
                        )
                    ):
                        sinh_phi, cosh_phi = safe_sinh_cosh_unified(
                            old_phi, PHI_CLIP, SINH_PHI_CLIP
                        )
                        salt_factor = salt_ions_solvation_penalty_map_1d[ijk1d]
                        numerator += salt_factor * (old_phi * cosh_phi - sinh_phi)
                        denominator += salt_factor * cosh_phi

                    # Newton update with adaptive damping
                    if abs(charge_density) > approx_zero:
                        phi_candidate = (numerator + charge_density) / denominator
                    else:
                        phi_candidate = numerator / denominator

                    # Adaptive relaxation
                    updated_phi = (
                        omega_old_weight * old_phi + omega_adaptive * phi_candidate
                    )

                    phi_map_next_half_1d[ijk1d_half] = updated_phi

                    # Δφ accumulation
                    dphi = updated_phi - old_phi
                    local_sq += dphi * dphi
                    adphi = abs(dphi)
                    if adphi > local_max:
                        local_max = adphi

        # Write thread-local results
        dphi_sq_partial[thread_id] = local_sq
        max_dphi_partial[thread_id] = local_max

    # Global reduction
    total_sq = 0.0
    max_dphi = 0.0
    for t in range(num_cpu_threads):
        total_sq += dphi_sq_partial[t]
        if max_dphi_partial[t] > max_dphi:
            max_dphi = max_dphi_partial[t]

    return total_sq, max_dphi
