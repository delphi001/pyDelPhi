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


from pydelphi.foundation.enums import (
    PBSolver,
    BoundaryCondition,
)

from pydelphi.utils.io.inproc_helpers.param_definitions.parameters import (
    ParameterGroup,
    ParamStatement,
)


def get_group_definition():
    """Defines and returns the 'pb' ParameterGroup."""
    return ParameterGroup(
        "iterativesolver",
        "The set of parameters for specifying the numerical solver, iterations, block size of iterations and boundary condition.",
        "The set of parameters for specifying the numerical solver, iterations, block size of iterations and boundary condition.",
    )


def get_param_definitions():
    """Defines and returns PB-related ParamStatement objects."""
    params = {}

    params[("solver", "solver", "solver")] = ParamStatement(
        full_name="solver",
        long_name="solver",
        short_name="solver",
        units=None,
        dtype=PBSolver,
        default=PBSolver.SOR,
        min_value=None,
        max_value=None,
        override=True,
        desc_short="Select the iterative solver to use.",
        desc_long="Select the iterative solver to use.",
        required=False,
    )

    params[("max_linear_iteration", "linit", "linit")] = ParamStatement(
        full_name="max_linear_iteration",
        long_name="linit",
        short_name="linit",
        units=None,
        dtype=int,
        default=5000,
        min_value=10,
        max_value=50000,
        override=True,
        desc_short="Maximum number of linear PB iterations to be done to reach convergence.",
        desc_long="Maximum number of linear PB iterations to be done to reach convergence.",
        required=False,
    )

    params[("iteration_block_size", "itrblocksize", "itrbsz")] = ParamStatement(
        full_name="iteration_block_size",
        long_name="itrblocksize",
        short_name="itrbsz",
        units=None,
        dtype=int,
        default=50,
        min_value=1,
        max_value=1000,
        override=True,
        desc_short="Number of iterations performed in an iteration block and after which RMSD is calculated.",
        desc_long="Number of iterations performed in an iteration block and after which RMSD is calculated.",
        required=True,
    )

    params[("nonlinear_iteration_block_size", "nlitrblocksize", "nlitrbsz")] = (
        ParamStatement(
            full_name="nonlinear_iteration_block_size",
            long_name="nlitrblocksize",
            short_name="nlitrbsz",
            units=None,
            dtype=int,
            default=5,
            min_value=1,
            max_value=1000,
            override=True,
            desc_short="Number of iterations performed in an iteration block and after which RMSD is calculated during nonlinear coupling steps of SOR.",
            desc_long="Number of iterations performed in an iteration block and after which RMSD is calculated  during nonlinear coupling steps of SOR.",
            required=True,
        )
    )

    params[("max_nonlinear_iteration", "nonlinit", "nonit")] = ParamStatement(
        full_name="max_nonlinear_iteration",
        long_name="nonlinit",
        short_name="nonit",
        units=None,
        dtype=int,
        default=0,
        min_value=0,
        max_value=50000,
        override=True,
        desc_short="Maximum number of non-linear PB iterations to be done to reach convergence.",
        desc_long="Maximum number of non-linear PB iterations to be done to reach convergence.",
        required=False,
    )

    params[("nonlinear_coupling_steps", "nonlinearcouplingsteps", "nlcs")] = (
        ParamStatement(
            full_name="nonlinear_coupling_steps",
            long_name="nonlinearcouplingsteps",
            short_name="nlcs",
            units=None,
            dtype=int,
            default=20,
            min_value=1,
            max_value=100,
            override=True,
            desc_short="Maximum number of non-linear PB iterations to be done to reach convergence.",
            desc_long="Maximum number of non-linear PB iterations to be done to reach convergence.",
            required=False,
        )
    )

    params[("nonlinear_relaxation_param", "nonlinearrelaxationparam", "nlrelpar")] = (
        ParamStatement(
            full_name="nonlinear_relaxation_param",
            long_name="nonlinearrelaxationparam",
            short_name="nlrelpar",
            units=None,
            dtype=float,
            default=0.0,  # default: no damping
            min_value=0.0,
            max_value=2.0,
            override=True,
            desc_short="Relaxation factor ω applied to non-linear PB updates.",
            desc_long=(
                "Controls the relaxation applied to nonlinear PB updates:\n\n"
                "    φ_new = (1 - ω)·φ_old + ω·φ_GS\n\n"
                "Special values:\n"
                "  ω = 0.0 → use automatically computed ω_SOR (default)\n"
                "  ω = 1.0 → no damping (pure Gauss–Seidel)\n"
                "  ω < 1.0 → under-relaxation (more stable)\n"
                "  ω > 1.0 → over-relaxation (faster, may diverge)\n"
                "Valid range: 0–2. Recommended range: 0.7–1.2 depending on convergence stability."
            ),
            required=False,
        )
    )

    params[("nwt_adaptive_omega", "nwtomega", "nwtw")] = ParamStatement(
        full_name="nwt_adaptive_omega",
        long_name="nwtomega",
        short_name="nwtw",
        units=None,
        dtype=float,
        default=1.0,
        min_value=0.1,  # prevents frozen or near-zero update steps
        max_value=1.0,
        override=True,
        desc_short="Adaptive damping factor for NWT nonlinear iterations.",
        desc_long=(
            "Adaptive damping factor ω_adaptive for the Newton (NWT) nonlinear "
            "iteration scheme. Values below 1.0 under-relax updates for improved "
            "stability in strongly nonlinear or oscillatory cases. Default is 1.0 "
            "(pure Newton–Gauss–Seidel). Values below 0.1 are automatically capped "
            "to 0.1 to prevent stalled iterations. Values above 1.0 are capped to "
            "1.0 and can cause divergence. The value 0.0 is reserved for potential "
            "future use as an auto-adaptive sentinel."
        ),
        required=False,
    )

    params[("boundary_condition", "bndcon", "bc")] = ParamStatement(
        full_name="boundary_condition",
        long_name="bndcon",
        short_name="bc",
        units=None,
        dtype=BoundaryCondition,
        default=BoundaryCondition.COULOMBIC,
        min_value=None,
        max_value=None,
        override=False,
        desc_short="Boundary condition.",
        desc_long="Boundary condition.",
        required=True,
    )

    return params
