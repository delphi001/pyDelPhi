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


import numpy as np
import math

from numba import njit

try:
    from numba import cuda

    # Check if a CUDA device is actually detected at runtime
    CUDA_AVAILABLE = cuda.is_available()
except Exception:
    # This handles ImportError if numba.cuda isn't installed,
    # or other issues if the environment is misconfigured.
    CUDA_AVAILABLE = False

# Assuming these are defined elsewhere in the user's code
# For demonstration, I'll define them here with example values
MAX_SINH_TAYLOR_TERMS = 7
SINH_TAYLOR_COEFFS = np.array(
    [
        1 / 6,  # for i=0, corresponds to phi^3 / 3!
        1 / 120,  # for i=1, corresponds to phi^5 / 5!
        1 / 5040,  # for i=2, corresponds to phi^7 / 7!
        1 / 362880,  # for i=3, corresponds to phi^9 / 9!
        1 / 39916800,  # for i=4, corresponds to phi^11 / 11!
        1 / 6227020800,  # for i=5, corresponds to phi^13 / 13!
        1 / 1307674368000,  # for i=6, corresponds to phi^15 / 15!
    ],
    dtype=np.float64,
)


@njit(inline="always")
def sinh_taylor_safe(
    phi: float,
    taylor_cutoff: float = 4.0,
    clip_cutoff: float = 5.0,
    n_terms: int = 5,
    linear_extension_slope: float = 0.2,
) -> float:
    """
    Safe approximation of sinh(phi) using Taylor expansion for intermediate values,
    native sinh for very small values, and a controlled linear extension for large values.

    Args:
        phi: Input value.
        taylor_cutoff: Threshold below which native sinh is used.
        clip_cutoff: Maximum |phi| value allowed for Taylor approximation.
        n_terms: Number of Taylor terms to use (up to MAX_SINH_TAYLOR_TERMS).
        linear_extension_slope: The slope for the linear extension beyond clip_cutoff.

    Returns:
        Approximated sinh(phi), safe for CPU.
    """
    abs_phi = abs(phi)
    final_result = 0.0

    # 1. Use native sinh for very small values
    if abs_phi < taylor_cutoff:
        final_result = np.sinh(phi)

    # 2. Taylor approximation or linear extension
    else:
        # Determine the argument for the Taylor calculation (min(|phi|, clip_cutoff))
        taylor_arg = min(abs_phi, clip_cutoff)

        # Calculate the Taylor series for sinh(taylor_arg)
        phi_sq = taylor_arg * taylor_arg
        term = taylor_arg
        result_taylor = taylor_arg

        # Cap number of terms
        max_terms = n_terms
        if max_terms > MAX_SINH_TAYLOR_TERMS:
            max_terms = MAX_SINH_TAYLOR_TERMS

        # Taylor series expansion sum up to given n_terms
        for i in range(max_terms):
            # term calculates phi^(2i+3), SINH_TAYLOR_COEFFS[i] is 1/(2i+3)!
            term *= phi_sq
            result_taylor += SINH_TAYLOR_COEFFS[i] * term

        if abs_phi < clip_cutoff:
            # Case 2a: abs_phi is in [taylor_cutoff, clip_cutoff). Apply sign.
            if phi >= 0.0:
                final_result = result_taylor
            else:
                final_result = -result_taylor
        else:
            # Case 2b: abs_phi >= clip_cutoff. Apply linear extension.

            value_at_clip_cutoff_positive = result_taylor

            # Calculate the linear extension
            delta = abs_phi - clip_cutoff
            positive_extension_value = (
                value_at_clip_cutoff_positive + linear_extension_slope * delta
            )

            # Apply the original sign of phi
            if phi >= 0.0:
                final_result = positive_extension_value
            else:
                final_result = -positive_extension_value

    return final_result


# Initialize the CUDA function pointer to None by default
cu_sinh_taylor_safe = None

if CUDA_AVAILABLE:

    @cuda.jit(device=True, inline="always")
    def cu_sinh_taylor_safe(
        phi: float,
        taylor_cutoff: float = 4.0,
        clip_cutoff: float = 5.0,
        n_terms: int = 5,
        linear_extension_slope: float = 0.2,
    ) -> float:
        """Explicit CUDA device implementation (cuda.jit(device=True))."""
        abs_phi = abs(phi)
        final_result = 0.0

        if abs_phi < taylor_cutoff:
            final_result = math.sinh(phi)  # Use math.sinh for CUDA
        else:
            taylor_arg = min(abs_phi, clip_cutoff)

            phi_sq = taylor_arg * taylor_arg
            term = taylor_arg
            result_taylor = taylor_arg

            max_terms = n_terms
            if max_terms > MAX_SINH_TAYLOR_TERMS:
                max_terms = MAX_SINH_TAYLOR_TERMS

            # Taylor series expansion
            for i in range(max_terms):
                term *= phi_sq
                result_taylor += SINH_TAYLOR_COEFFS[i] * term

            # Apply sign and linear extension logic
            value_at_clip_cutoff_positive = result_taylor
            if abs_phi >= clip_cutoff:
                delta = abs_phi - clip_cutoff
                value_at_clip_cutoff_positive += linear_extension_slope * delta

            final_result = (
                value_at_clip_cutoff_positive
                if phi >= 0.0
                else -value_at_clip_cutoff_positive
            )

        return final_result
