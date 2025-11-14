#!/usr/bin/env python
# -*- coding: utf-8 -*-

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
This module provides Numba-optimized functions for manipulating and analyzing
1D representations of 3D grid data, particularly focusing on operations
related to grid sampling, copying between full and sampled grids, and
calculating Root Mean Square Deviation (RMSD) and maximum deviation between
a full grid and its sampled version. These operations are fundamental
in numerical methods for solving partial differential equations on grids,
such as multigrid methods where data is transferred between grids of
different resolutions.

The functions are designed for performance using Numba's `njit` decorator,
with `parallel=True` where appropriate for multi-threading.
Type hints (`delphi_real`, `delphi_int`) are used for clarity,
referring to `float` and `int` types as defined in `pydelphi.config.global_runtime`.

Functions:
- `_copy_to_sample`: Copies data from a full 1D array to a smaller 1D
  "sample" array, based on a given offset and skip interval.
- `_copy_to_full`: Copies data from a smaller 1D "sample" array back to
  a larger 1D "full" array, populating elements at specific offsets and strides.
- `_sum_of_product_sample`: Calculates the sum of products of elements
  between a full 1D array and a half-sized 1D array, respecting offset and stride.
- `_calculate_phi_map_sample_rmsd`: Computes the RMSD and maximum deviation
  between a full-resolution potential map and a sampled (half-resolution)
  potential map in parallel.
"""

import math
import numpy as np


from numba import njit, prange, float64

from pydelphi.config.global_runtime import (
    delphi_int,
    delphi_real,
)


@njit(nogil=True, boundscheck=False, fastmath=True, cache=True)
def _copy_to_sample(
    destination_array: np.ndarray[delphi_real],
    source_array: np.ndarray[delphi_real],
    offset: delphi_int,
    skip: delphi_int,
):
    """
    Copies a sample of elements from source array to destination array with given offset and skip.

    Args:
        destination_array (np.ndarray[delphi_real]): Array to copy to.
        source_array (np.ndarray[delphi_real]): Array to copy from.
        offset (delphi_int): Starting index offset in source array.
        skip (delphi_int): Skip interval in source array.
    """
    for i in range(destination_array.size):
        source_index = skip * i + offset  # Calculate source index with skip and offset
        if source_index < source_array.size:  # Boundary check for source array
            destination_array[i] = source_array[source_index]


@njit(nogil=True, boundscheck=False, fastmath=True, cache=True)
def _copy_to_full(
    destination_array: np.ndarray[delphi_real],
    source_array: np.ndarray[delphi_real],
    offset: delphi_int,
    skip: delphi_int,
):
    """
    Copies elements from source array to destination array to fill a full array,
    using offset and skip for indexing in the destination array.

    Args:
        destination_array (np.ndarray[delphi_real]): Array to copy to (full array).
        source_array (np.ndarray[delphi_real]): Array to copy from (sample array).
        offset (delphi_int): Starting index offset in destination array.
        skip (delphi_int): Skip interval in destination array.
    """
    for i in range(source_array.size):
        destination_index = (
            i * skip + offset
        )  # Calculate destination index with skip and offset
        if (
            destination_index < destination_array.size
        ):  # Boundary check for destination array
            destination_array[destination_index] = source_array[i]


@njit(nogil=True, boundscheck=False, cache=True)
def _sum_of_product_sample(
    full_array: np.ndarray[delphi_real],
    half_array: np.ndarray[delphi_real],
    offset: delphi_int,
    stride: delphi_int,
    num_grid_points: delphi_int,
):
    """
    Calculates the sum of products of elements from a full array and a half array,
    using offset and stride to align indices.

    Args:
        full_array (np.ndarray[delphi_real]): The full-sized array.
        half_array (np.ndarray[delphi_real]): The half-sized array (odds or evens).
        offset (delphi_int): Starting index offset in full array.
        stride (delphi_int): Stride interval in full array.
        num_grid_points (delphi_int): Total number of grid points in full array.

    Returns:
        delphi_real: Sum of products.
    """
    total_sum = 0.0
    # Iterate with offset and skip in full array
    for i in range(offset, num_grid_points, stride):
        # Calculate corresponding index in half array
        half_index = (i - offset) // stride
        total_sum += full_array[i] * half_array[half_index]
    return total_sum


@njit(nogil=True, boundscheck=False, fastmath=True, parallel=True, cache=True)
def _calculate_phi_map_sample_rmsd(
    phi_map_full_1d: np.ndarray,
    phi_map_half_1d: np.ndarray,
    offset: int,
    stride: int,
    num_cpu_threads: int,
    dtype: float,
) -> tuple:
    """
    Parallel RMSD and max deviation between full phi map and sampled phi map.

    Intermediate values are accumulated in float64 for stability,
    and final values are cast back to the input dtype.

    Args:
        phi_map_full_1d: Full-resolution phi map.
        phi_map_half_1d: Sampled phi map on the odd grid.
        offset: Offset from start of full map for sampling.
        stride: Step size between samples in full map.
        dtype: The precision to use for the final result (float32 or float64).

    Returns:
        (rmsd, max_deviation) as same dtype as phi_map_half_1d
    """
    num_grid_points_half = phi_map_half_1d.size

    # Use float64 accumulators regardless of input precision
    thread_sums = np.zeros(num_cpu_threads, dtype=np.float64)
    thread_maxes = np.zeros(num_cpu_threads, dtype=np.float64)

    for thread_id in prange(num_cpu_threads):
        items_per_thread = (
            num_grid_points_half + num_cpu_threads - 1
        ) // num_cpu_threads
        start = thread_id * items_per_thread
        end = min(start + items_per_thread, num_grid_points_half)

        local_sum = 0.0
        local_max = 0.0

        for i in range(start, end):
            full_index = i * stride + offset
            if full_index < phi_map_full_1d.size:
                deviation = float64(phi_map_half_1d[i]) - float64(
                    phi_map_full_1d[full_index]
                )
                local_sum += deviation * deviation
                abs_dev = abs(deviation)
                if abs_dev > local_max:
                    local_max = abs_dev

        thread_sums[thread_id] = local_sum
        thread_maxes[thread_id] = local_max

    # Serial reduction
    total_sum = 0.0
    max_deviation = 0.0
    for t in range(num_cpu_threads):
        total_sum += thread_sums[t]
        if thread_maxes[t] > max_deviation:
            max_deviation = thread_maxes[t]

    rmsd = math.sqrt(total_sum / (num_grid_points_half - 1))

    # Cast final result to match input dtype
    return dtype(rmsd), dtype(max_deviation)


@njit(nogil=True, boundscheck=False, cache=True)
def _iteration_control_check(
    rmsd_buf: np.ndarray,
    ptr: int,
    wrapped: bool,
    rmsd: float,
    dphi: float,
    rms_threshold: float,
    dphi_threshold: float,
    enforce_dphi: bool,
    max_iters: int,
    iteration: int,
    disable_stagnation_check: bool = False,
    flip_limit: int = 3,
):
    """
    Unified iteration control for linear and nonlinear solvers.

    Performs convergence, divergence, and *consecutive sign-flip* stagnation
    detection using a circular buffer of recent RMSD values.

    The stagnation detector identifies oscillatory RMSD behavior by counting
    consecutive changes in the sign of ΔRMSD = RMSD[i−1] − RMSD[i].
    It tolerates short same-sign sequences, capturing both single-step (↓↑)
    and multi-step (↓↓↑↑) oscillations typical of block-wise or nonlinear
    relaxation behavior.

    Args:
        rmsd_buf (np.ndarray): Circular buffer storing recent RMSD values.
        ptr (int): Current write index in buffer.
        wrapped (bool): Whether buffer has completed a full cycle.
        rmsd (float): Current RMSD value.
        dphi (float): Current ΔΦ (maximum potential change).
        rms_threshold (float): RMSD convergence tolerance.
        dphi_threshold (float): ΔΦ convergence tolerance.
        enforce_dphi (bool): Whether ΔΦ-based convergence is enforced.
        max_iters (int): Maximum iteration count allowed.
        iteration (int): Current global iteration number.
        disable_stagnation_check (bool): If True, disables stagnation detection.
        flip_limit (int): Number of *consecutive* ΔRMSD sign flips required to
                          declare stagnation (default=3).

    Returns:
        (stop, status, ptr_next, wrapped):
            stop (bool): True if iteration should stop.
            status (int):
                0 = continue
                1 = converged (threshold satisfied)
                2 = stagnation plateau (relaxed convergence)
                3 = divergence (monotonic RMSD increase or NaN detected)
            ptr_next (int): Updated buffer pointer.
            wrapped (bool): Updated wrapped flag.
    """
    # --- Divergence / NaN safety ---
    if not math.isfinite(rmsd) or (enforce_dphi and not math.isfinite(dphi)):
        return True, 3, ptr, wrapped

    # --- Primary convergence check ---
    if enforce_dphi:
        if dphi < dphi_threshold:
            return True, 1, ptr, wrapped
    else:
        if rmsd < rms_threshold:
            return True, 1, ptr, wrapped

    # --- Update circular buffer ---
    n = rmsd_buf.size
    rmsd_buf[ptr] = rmsd
    ptr_next = (ptr + 1) % n
    if not wrapped and ptr_next == 0:
        wrapped = True

    # --- Stagnation & divergence detection ---
    if (not disable_stagnation_check) and wrapped:
        flip_streak = 0
        same_sign_count = 0
        last_sign = 0
        increasing_count = 0
        decreasing_count = 0

        # Chronological scan (oldest → newest)
        for i in range(1, n):
            i0 = (ptr_next + i - 1) % n
            i1 = (ptr_next + i) % n
            dr = rmsd_buf[i0] - rmsd_buf[i1]
            sign = 1 if dr > 0 else (-1 if dr < 0 else 0)

            if sign == 0:
                continue

            # Track monotonic trends
            if sign > 0:
                decreasing_count += 1
            elif sign < 0:
                increasing_count += 1

            # Detect oscillatory sign alternations
            if last_sign != 0 and sign != last_sign:
                flip_streak += 1
                same_sign_count = 0
            else:
                same_sign_count += 1
                # Allow limited persistence of same-sign segments
                if same_sign_count > 2:
                    flip_streak = max(0, flip_streak - 1)
                    same_sign_count = 0

            last_sign = sign

            # Early stop on sufficient consecutive flips
            if flip_streak >= flip_limit:
                return True, 2, ptr_next, wrapped

        # --- Monotonic divergence detection ---
        if increasing_count >= int(0.6 * n):
            return True, 3, ptr_next, wrapped

    # --- Max iteration fallback ---
    if iteration >= max_iters:
        return True, 2, ptr_next, wrapped

    # Continue iterating
    return False, 0, ptr_next, wrapped
