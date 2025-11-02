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

#
# pyDelPhi is free software: you can redistribute it and/or modify
# (at your option) any later version.
#
# pyDelPhi is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#

#
# PyDelphi is free software: you can redistribute it and/or modify
# (at your option) any later version.
#
# PyDelphi is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#

import numpy as np
import math
import struct

from pydelphi.utils.interpolation import interpolate
from pydelphi.site.siteexceptions import *


def expand(
    grid, mgrid, prgf_phi_map, phimap_in, phimap, b_biosystem_out, f_scale, interpl_func
):
    """
    Expands or contracts the potential map grid.

    Adjusts grid resolution of potential map (prgf_phi_map) from 'grid' to 'mgrid' points,
    storing result in 'phimap_in', and scales 'f_scale'.

    Args:
        grid (int): Original grid size.
        mgrid (int): Target grid size.
        prgf_phi_map (numpy.ndarray): Input potential map.
        phimap_in (list): List for expanded/contracted potential map.
        phimap (numpy.ndarray): Placeholder, seemingly unused.
        b_biosystem_out (bool): Unused flag.
        f_scale (float): Original scale factor.
        interpl_func (function): Interpolation function.

    Returns:
        float: New scale factor.
    """
    rscale = (grid - 1.0) / (mgrid - 1.0)

    if grid == mgrid:
        phimap_in[:] = prgf_phi_map
    else:
        if not phimap_in:
            phimap_in.extend([0.0] * (mgrid * mgrid * mgrid))

        if not b_biosystem_out:
            for iz in range(mgrid, 0, -1):
                gc_z = (iz - 1) * rscale + 1.0
                for iy in range(mgrid, 0, -1):
                    gc_y = (iy - 1) * rscale + 1.0
                    for ix in range(mgrid, 0, -1):
                        gc_x = (ix - 1) * rscale + 1.0
                        phiv = interpl_func(grid, prgf_phi_map, (gc_x, gc_y, gc_z))
                        iw = (iz - 1) * mgrid * mgrid + (iy - 1) * mgrid + (ix - 1)
                        phimap_in[iw] = phiv

    new_f_scale = f_scale / rscale
    return new_f_scale


def phicon(
    grid,
    dielectric_map_1d,
    phimap,
    phimap_out,
    f_taylor_coeff1,
    f_taylor_coeff2,
    f_taylor_coeff3,
    f_taylor_coeff4,
    f_taylor_coeff5,
    f_ion_strength,
    f_zero,
    i_non_iterate_num,
):
    """
    Converts potentials to net charge concentrations using Taylor coefficients.

    Modifies potential map (phimap) to represent net charge concentrations if
    ionic strength (f_ion_strength) > 0. Uses polynomial expansion defined by
    Taylor coefficients.

    Args:
        grid (int): Grid size.
        dielectric_map_1d (numpy.ndarray): 1D dielectric map.
        phimap (numpy.ndarray): Input potential map.
        phimap_out (numpy.ndarray): Output potential map (modified in place).
        f_taylor_coeff1 (float): Taylor coefficient 1.
        f_taylor_coeff2 (float): Taylor coefficient 2.
        f_taylor_coeff3 (float): Taylor coefficient 3.
        f_taylor_coeff4 (float): Taylor coefficient 4.
        f_taylor_coeff5 (float): Taylor coefficient 5.
        f_ion_strength (float): Ionic strength.
        f_zero (float): Zero value threshold.
        i_non_iterate_num (int): Non-iterate number flag.

    Returns:
        numpy.ndarray: Modified potential map (phimap_out).
    """
    tmp = abs(f_taylor_coeff2 * f_taylor_coeff4)
    phimap_out[:] = np.copy(phimap)

    if f_ion_strength > 0.0:
        print("\nconverting potentials to \n")
        print("net charge concentrations...\n\n")
        warning = CUntestedPhicon()

        if i_non_iterate_num != 0:
            if f_zero > tmp:
                for iz in range(grid):
                    for iy in range(grid):
                        for ix in range(grid):
                            iw = iz * grid * grid + iy * grid + ix
                            if dielectric_map_1d[iw]:
                                phi = phimap[iz][iy][ix]
                                phisq = phi * phi
                                temp = phisq * f_taylor_coeff5 + f_taylor_coeff3
                                temp = temp * phisq + f_taylor_coeff1
                                phimap_out[iz][iy][ix] = temp * phi
                            else:
                                phimap_out[iz][iy][ix] = 0.0
            else:
                for iz in range(grid):
                    for iy in range(grid):
                        for ix in range(grid):
                            iw = iz * grid * grid + iy * grid + ix
                            if dielectric_map_1d[iw]:
                                phi = phimap[iz][iy][ix]
                                temp = phi * f_taylor_coeff5 + f_taylor_coeff4
                                temp = phi * temp + f_taylor_coeff3
                                temp = phi * temp + f_taylor_coeff2
                                temp = phi * temp + f_taylor_coeff1
                                phimap_out[iz][iy][ix] = temp * phi
                            else:
                                phimap_out[iz][iy][ix] = 0.0
        else:
            for iz in range(grid):
                for iy in range(grid):
                    for ix in range(grid):
                        iw = iz * grid * grid + iy * grid + ix
                        if dielectric_map_1d[iw]:
                            phi = phimap[iz][iy][ix]
                            phimap_out[iz][iy][ix] = f_taylor_coeff1 * phi
                        else:
                            phimap_out[iz][iy][ix] = 0.0

    if f_zero > abs(f_ion_strength):
        warning = CNoPotential2CrgConcentrate()
    return phimap_out


def rforce(
    bndy_grid_num,
    crg_grid_num,
    prgf_surf_crg_e,
    prgfg_surf_crg_a,
    prgfg_crg_pose_a,
    prggv_atomic_crg,
    prgi_crg_at,
    prgi_at_surf,
):
    """
    Calculates reaction force on atoms due to surface and atomic charges.

    Computes electric field and reaction force based on surface charge density
    and atomic charges within the grid.

    Args:
        bndy_grid_num (int): Number of boundary grid points.
        crg_grid_num (int): Number of charge grid points.
        prgf_surf_crg_e (numpy.ndarray): Array of surface charge densities.
        prgfg_surf_crg_a (numpy.ndarray): Array of surface charge positions.
        prgfg_crg_pose_a (numpy.ndarray): Array of atomic charge positions.
        prggv_atomic_crg (numpy.ndarray): Array of atomic charge values (nValue attribute).
        prgi_crg_at (numpy.ndarray): Array mapping charge grid points to atom indices.
        prgi_at_surf (numpy.ndarray): Array mapping surface grid points to atom indices.

    Returns:
        numpy.ndarray: Array of reaction forces on atoms.

    Raises:
        CCrgatnError: If invalid atom index (iat == 0).
    """
    afield = np.zeros((bndy_grid_num, 3), dtype=float)
    sfield = np.zeros(3, dtype=float)

    for i in range(bndy_grid_num):
        sfield[:] = 0.0
        sc = prgf_surf_crg_e[i]

        for j in range(crg_grid_num):
            vtemp = prgfg_surf_crg_a[i] - prgfg_crg_pose_a[j]
            dist_sq = np.dot(vtemp, vtemp)
            dist = np.sqrt(dist_sq)
            temp = prggv_atomic_crg[j].nValue / (dist_sq * dist)
            sfield += temp * vtemp
            iat = prgi_crg_at[j]
            if iat < 0:
                continue
            if iat == 0:
                raise CCrgatnError()
            afield[iat - 1] -= (sc * temp) * vtemp

        sfield *= sc
        j = prgi_at_surf[i]
        afield[j - 1] += sfield

    return afield


def rforceeps1(
    bndy_grid_num,
    atom_num,
    crg_grid_num,
    prgfg_surf_crg_a,
    prgf_surf_crg_e,
    prgi_at_surf,
    prggv_atomic_crg,
    prgi_crg_at,
    f_scale,
    fg_box_center,
):
    """
    Calculates reaction force with epsilon=1 approximation, considering self-polarization.

    Similar to `rforce` but includes self-polarization approximation and uses
    different approach to calculate electric field, assuming epsilon=1.

    Args:
        bndy_grid_num (int): Number of boundary grid points.
        atom_num (int): Number of atoms.
        crg_grid_num (int): Number of charge grid points.
        prgfg_surf_crg_a (list): Surface charge positions.
        prgf_surf_crg_e (list): Surface charge densities.
        prgi_at_surf (list): Mapping surface grid points to atom indices.
        prggv_atomic_crg (list): Atomic charge values.
        prgi_crg_at (list): Mapping charge grid points to atom indices.
        f_scale (float): Scale factor.
        fg_box_center (numpy.ndarray): Center of the grid.

    Returns:
        list: Reaction forces on atoms.

    Raises:
        CCrgatnError: If invalid atom index (iat == 0).
    """
    afield = [np.array([0.0, 0.0, 0.0]) for _ in range(bndy_grid_num)]

    sfield = np.array([0.0, 0.0, 0.0])
    xyz = np.array([0.0, 0.0, 0.0])
    vtemp = np.array([0.0, 0.0, 0.0])
    ixyz = np.array([0, 0, 0])
    FACT_EPS1 = -2.0 * math.pi * 80.0 / 79.0
    REAL_DS_FACTOR = 0.0393
    MY_DS_FACTOR = 16.0 * math.pi
    cont = 0

    for p in range(bndy_grid_num):
        sfield[:] = 0.0

        ixyz[0] = round((prgfg_surf_crg_a[p][0] - fg_box_center[0]) * f_scale)
        ixyz[1] = round((prgfg_surf_crg_a[p][1] - fg_box_center[1]) * f_scale)
        ixyz[2] = round((prgfg_surf_crg_a[p][2] - fg_box_center[2]) * f_scale)

        xyz[0] = float(ixyz[0]) / f_scale + fg_box_center[0]
        xyz[1] = float(ixyz[1]) / f_scale + fg_box_center[1]
        xyz[2] = float(ixyz[2]) / f_scale + fg_box_center[2]

        sc = 0.5 * prgf_surf_crg_e[p]
        vtemp = prgfg_surf_crg_a[p] - xyz
        vvtemp = np.dot(vtemp, prgf_surf_crg_e[p])
        fact1 = 0.8 / (f_scale * f_scale) - vvtemp * vvtemp
        deltas = math.pi * fact1
        sigmap = prgf_surf_crg_e[p] / deltas
        realds = 0.0
        rmyds = 0.0
        realds += prgf_surf_crg_e[p] / REAL_DS_FACTOR
        rmyds += deltas
        trullo = FACT_EPS1 * sigmap * sigmap * deltas

        if 400 == (p + 1):
            print(
                f"normale {prgf_surf_crg_e[p][0]} {prgf_surf_crg_e[p][1]} {prgf_surf_crg_e[p][2]}"
            )
            print(f"P {vtemp}")
            print(
                f"{prgfg_surf_crg_a[p][0]} {prgfg_surf_crg_a[p][1]} {prgfg_surf_crg_a[p][2]}"
            )
            print(
                f"area {prgf_surf_crg_e[p] / REAL_DS_FACTOR} {prgf_surf_crg_e[p] * f_scale * f_scale / REAL_DS_FACTOR}"
            )

        j = prgi_at_surf[p]

        for iat in range(atom_num):
            if (j - 1) == iat:
                cont += 1
                afield[iat][0] += trullo

        for i in range(crg_grid_num):
            vtemp = prgfg_surf_crg_a[p] - prgfg_crg_pose_a[i]
            dist_sq = np.dot(vtemp, vtemp)
            dist = math.sqrt(dist_sq)
            temp = prggv_atomic_crg[j].nValue / (dist_sq * dist)
            sfield += temp * vtemp

            iat = prgi_crg_at[i]
            if iat < 0:
                continue
            if iat == 0:
                raise CCrgatnError()
            if j == iat:
                afield[iat - 1] = (
                    afield[iat - 1] - (sc * temp) * vtemp + fact1 * prgf_surf_crg_e[p]
                )

        sfield *= sc

    print(f"supcalc {realds / MY_DS_FACTOR} mia {rmyds / MY_DS_FACTOR}")

    return afield


def tops(xxo, xxu, crg, eps, flag):
    """
    Calculates potential at a point using trilinear interpolation.

    Reads potential map from "lkphi.dat" and calculates potential at
    given point (xxo, xxu coordinates) using trilinear interpolation.
    Can also calculate electric field components if flag > 1.

    Args:
        xxo (list): Lower corner coordinates [x, y, z].
        xxu (list): Upper corner coordinates [x, y, z].
        crg (float): Charge value (unused in potential calculation).
        eps (float): Dielectric constant.
        flag (int): Flag to control calculation type (1 for potential, >1 for field).

    Returns:
        float: Calculated potential at the given point.
    """
    pot = 0.0
    phi = np.zeros((65, 65, 65), dtype=float)

    phi_file_name = "lkphi.dat"
    try:
        with open(phi_file_name, "rb") as f:
            phi = np.fromfile(f, dtype=np.float64).reshape((65, 65, 65))
    except FileNotFoundError:
        print(f"Error: Potential map file '{phi_file_name}' not found.")
        return pot

    xo = xxo
    xu = xxu

    axo = [math.floor(coord) for coord in xo]
    bxo = [(a - 1.0) if coord < 0.0 else (a + 1.0) for a, coord in zip(axo, xo)]

    cr_grid_positions = np.zeros((3, 8), dtype=float)
    for i in range(4):
        for j in range(3):
            cr_grid_positions[j][i] = axo[j]

    for i in range(4, 8):
        cr_grid_positions[2][i] = bxo[2]

    cr_grid_positions[0][1] = bxo[0]
    cr_grid_positions[1][3] = bxo[1]
    cr_grid_positions[0][2] = bxo[0]
    cr_grid_positions[1][2] = bxo[1]

    for i in range(4, 8):
        for j in range(2):
            cr_grid_positions[j][i] = cr_grid_positions[j][i - 4]

    faxo = [abs(coord - a) for coord, a in zip(xo, axo)]
    fbxo = [abs(coord - b) for coord, b in zip(xo, bxo)]

    mfo = [
        fbxo[0] * fbxo[1] * fbxo[2],
        faxo[0] * fbxo[1] * fbxo[2],
        faxo[0] * faxo[1] * fbxo[2],
        fbxo[0] * faxo[1] * fbxo[2],
        fbxo[0] * fbxo[1] * faxo[2],
        faxo[0] * fbxo[1] * faxo[2],
        faxo[0] * faxo[1] * faxo[2],
        fbxo[0] * faxo[1] * faxo[2],
    ]

    crgrid = [crg * m for m in mfo]

    axu = [math.floor(coord) for coord in xu]
    bxu = [(a - 1.0) if coord < 0.0 else (a + 1.0) for a, coord in zip(axu, xu)]

    tr_grid_positions = np.zeros((3, 8), dtype=float)
    for i in range(4):
        for j in range(3):
            tr_grid_positions[j][i] = axu[j]

    for i in range(4, 8):
        tr_grid_positions[2][i] = bxu[2]

    tr_grid_positions[0][1] = bxu[0]
    tr_grid_positions[1][3] = bxu[1]
    tr_grid_positions[0][2] = bxu[0]
    tr_grid_positions[1][2] = bxu[1]

    for i in range(4, 8):
        for j in range(2):
            tr_grid_positions[j][i] = tr_grid_positions[j][i - 4]

    faxu = [abs(coord - a) for coord, a in zip(xu, axu)]
    fbxu = [abs(coord - b) for coord, b in zip(axu, bxu)]

    mfu = [
        fbxu[0] * fbxu[1] * fbxu[2],
        faxu[0] * fbxu[1] * fbxu[2],
        faxu[0] * faxu[1] * fbxu[2],
        fbxu[0] * faxu[1] * fbxu[2],
        fbxu[0] * fbxu[1] * faxu[2],
        faxu[0] * fbxu[1] * faxu[2],
        faxu[0] * faxu[1] * faxu[2],
        fbxu[0] * faxu[1] * faxu[2],
    ]

    potential_values = [0.0] * 8
    electric_field = np.zeros((3, 8), dtype=float)

    for i in range(8):
        e = [0.0, 0.0, 0.0]
        for j in range(8):
            if crgrid[j] != 0:
                vec = [
                    int(tr_grid_positions[k][i] - cr_grid_positions[k][j])
                    for k in range(3)
                ]
                if flag in (1, 3):
                    dummy = phi[abs(vec[0])][abs(vec[1])][abs(vec[2])]
                    dummy *= crgrid[j]
                    potential_values[i] += dummy / eps

                if flag > 1:
                    for k in range(3):
                        si = 1 if vec[k] >= 0 else -1
                        a = [abs(vec[l]) for l in range(3)]
                        b = [abs(vec[l]) for l in range(3)]
                        if a[k] < 64:
                            a[k] += 1
                        if b[k] > 0:
                            b[k] -= 1
                        dummy = (phi[a[0]][a[1]][a[2]] - phi[b[0]][b[1]][b[2]]) / eps
                        e[k] -= si * crgrid[j] * dummy

            if flag > 1:
                for j in range(3):
                    electric_field[j][i] = e[j]

    pot = sum(potential_values[i] * mfu[i] for i in range(8))

    return pot


def write_p_analysis(grid, radipz, f_scale, prgf_phi_map, xn2, natom, fg_box_center):
    """
    Analyze and write potential values along z-axis to "pz.txt".

    Calculates and writes average potential along the z-axis, excluding
    regions near atoms. Results are written to "pz.txt".

    Args:
        grid (int): Grid size.
        radipz (float): Exclusion radius for atoms.
        f_scale (float): Scale factor.
        prgf_phi_map (list): Potential map.
        xn2 (list): Atomic coordinates (objects with nX, nY, nZ attributes).
        natom (int): Number of atoms.
        fg_box_center (numpy.ndarray): Grid center.
    """
    psumarr = [0.0] * 10000
    npoint = [0] * 10000
    rmid = float((grid + 1) / 2)
    distance_sq = (radipz * f_scale) ** 2
    lbox = int(radipz * f_scale)

    for k in range(1, grid + 1):
        for i in range(1, grid + 1):
            for j in range(1, grid + 1):
                index = (k - 1) * grid * grid + (j - 1) * grid + (i - 1)
                psumarr[k] += prgf_phi_map[index]
                npoint[k] += 1

    for m in range(natom):
        atom_x, atom_y, atom_z = int(xn2[m][0]), int(xn2[m][1]), int(xn2[m][2])
        for i in range(atom_x - lbox, atom_x + lbox + 2):
            for j in range(atom_y - lbox, atom_y + lbox + 2):
                for k in range(atom_z - lbox, atom_z + lbox + 2):
                    if (i - atom_x) ** 2 + (j - atom_y) ** 2 + (
                        k - atom_z
                    ) ** 2 < distance_sq:
                        index = (k - 1) * grid * grid + (j - 1) * grid + (i - 1)
                        if (
                            abs(prgf_phi_map[index]) > 0.00001
                            and prgf_phi_map[index] > 0.1
                        ):
                            psumarr[k] -= prgf_phi_map[index]
                            npoint[k] -= 1
                            prgf_phi_map[index] = 0.0

    pz_file_name = "pz.txt"
    with open(pz_file_name, "w") as pzfile:
        for k in range(1, grid + 1):
            average_potential = psumarr[k] / npoint[k] if npoint[k] > 0 else 0.0

            pzfile.write(
                f"z: {k:5d}  "
                f"{(k - rmid) / f_scale + fg_box_center[2]:8.3f}   "
                f"n: {npoint[k]:10d}   "
                f"Pz: {average_potential:12.4f} kt/e or  "
                f"{average_potential * 25.85:12.4f} mv\n"
            )


def write_potential_ccp4(phimap_in, grid, f_scale, phi_file):
    """
    Writes potential map in CCP4 format to a file.

    Writes potential map data to a file formatted according to CCP4
    crystallographic map format.

    Args:
        phimap_in (list): Potential map data.
        grid (int): Grid size.
        f_scale (float): Scale factor.
        phi_file (str): Output file path.
    """
    print("\nWriting potential map in CCP4 format\n")
    print(f"Potential map written to file {phi_file}\n")

    CCP4_MODE = 2
    CCP4_NCSTART = 1
    CCP4_NRSTART = 1
    CCP4_NSSTART = 1
    CCP4_ALPHA = 90.0
    CCP4_BETA = 90.0
    CCP4_GAMMA = 90.0
    CCP4_MAPC = 1
    CCP4_MAPR = 2
    CCP4_MAPS = 3
    CCP4_ISPG = 1
    CCP4_NSYMBT = 1
    CCP4_LSKFLG = 0
    CCP4_ARMS = 0.0
    CCP4_MACHST = 1
    CCP4_NLABL = 0
    CCP4_FZERO = 0.0
    CCP4_IZERO = 0
    CCP4_MAP_STR = b"MAP "
    CCP4_RESERVED_SPACE_INT = 9
    CCP4_RESERVED_SPACE_FLOAT = 3
    CCP4_RESERVED_ZEROS_INT = 15
    CCP4_RESERVED_ZEROS_LABEL_INT = 200

    nx = grid - 1
    ny = grid - 1
    nz = grid - 1
    xlen = (grid - 1) / f_scale
    ylen = (grid - 1) / f_scale
    zlen = (grid - 1) / f_scale
    minim = min(phimap_in)
    maxim = max(phimap_in)
    somma = sum(phimap_in)
    average = somma / (grid * grid * grid)

    with open(phi_file, "wb") as of_phi_stream:
        of_phi_stream.write(struct.pack("i", grid))
        of_phi_stream.write(struct.pack("i", grid))
        of_phi_stream.write(struct.pack("i", grid))
        of_phi_stream.write(struct.pack("i", CCP4_MODE))
        of_phi_stream.write(struct.pack("i", CCP4_NCSTART))
        of_phi_stream.write(struct.pack("i", CCP4_NRSTART))
        of_phi_stream.write(struct.pack("i", CCP4_NSSTART))
        of_phi_stream.write(struct.pack("i", nx))
        of_phi_stream.write(struct.pack("i", ny))
        of_phi_stream.write(struct.pack("i", nz))
        of_phi_stream.write(struct.pack("f", xlen))
        of_phi_stream.write(struct.pack("f", ylen))
        of_phi_stream.write(struct.pack("f", zlen))
        of_phi_stream.write(struct.pack("f", CCP4_ALPHA))
        of_phi_stream.write(struct.pack("f", CCP4_BETA))
        of_phi_stream.write(struct.pack("f", CCP4_GAMMA))
        of_phi_stream.write(struct.pack("i", CCP4_MAPC))
        of_phi_stream.write(struct.pack("i", CCP4_MAPR))
        of_phi_stream.write(struct.pack("i", CCP4_MAPS))
        of_phi_stream.write(struct.pack("f", minim))
        of_phi_stream.write(struct.pack("f", maxim))
        of_phi_stream.write(struct.pack("f", average))
        of_phi_stream.write(struct.pack("i", CCP4_ISPG))
        of_phi_stream.write(struct.pack("i", CCP4_NSYMBT))
        of_phi_stream.write(struct.pack("i", CCP4_LSKFLG))

        for _ in range(CCP4_RESERVED_SPACE_INT):
            of_phi_stream.write(struct.pack("i", CCP4_IZERO))
        for _ in range(CCP4_RESERVED_SPACE_FLOAT):
            of_phi_stream.write(struct.pack("f", CCP4_FZERO))
        for _ in range(CCP4_RESERVED_ZEROS_INT):
            of_phi_stream.write(struct.pack("i", CCP4_IZERO))

        of_phi_stream.write(CCP4_MAP_STR)
        of_phi_stream.write(struct.pack("i", CCP4_MACHST))
        of_phi_stream.write(struct.pack("f", CCP4_ARMS))
        of_phi_stream.write(struct.pack("i", CCP4_NLABL))
        for _ in range(CCP4_RESERVED_ZEROS_LABEL_INT):
            of_phi_stream.write(struct.pack("i", CCP4_IZERO))

        for value in phimap_in:
            of_phi_stream.write(struct.pack("f", value))


def write_potential_grasp(
    phimap_in,
    f_scale,
    fg_box_center,
    phi_file,
    b_out_crg_density=False,
    f_ion_strength=0,
):
    """
    Writes potential map in GRASP format to a binary file.

    Writes potential map data to a binary file formatted according to
    GRASP potential map format. Uses placeholder functions
    `expand_grasp` and `write_phi_map_grasp`.

    Args:
        phimap_in (list): Potential map data.
        f_scale (float): Scale factor.
        fg_box_center (tuple): Grid center (x, y, z).
        phi_file (str): Output file path.
        b_out_crg_density (bool): Output charge density flag.
        f_ion_strength (float): Ionic strength.
    """
    print("\nWriting potential map in GRASP format\n")
    print(f"Potential map written to file {phi_file}\n")

    GRASP_HEAD_LABEL = b"now starting phimap "
    GRASP_TAIL_LABEL = b" end of phimap"
    GRASP_CONCENTRATION_LABEL = b"concentrat"
    GRASP_POTENTIAL_LABEL = b"potential  "

    nxtlbl = (
        GRASP_CONCENTRATION_LABEL
        if b_out_crg_density and f_ion_strength != 0
        else GRASP_POTENTIAL_LABEL
    )

    phimap_expanded = expand_grasp(65, phimap_in)

    with open(phi_file, "wb") as of_phi_stream:
        of_phi_stream.write(GRASP_HEAD_LABEL.ljust(21, b"\0"))
        of_phi_stream.write(nxtlbl.ljust(10, b"\0"))
        of_phi_stream.write(b"\0" * 60)

        write_phi_map_grasp(0, phimap_expanded, of_phi_stream)

        of_phi_stream.write(GRASP_TAIL_LABEL.ljust(16, b"\0"))

        fscale = float(f_scale)
        foldmid_x, foldmid_y, foldmid_z = map(float, fg_box_center)
        of_phi_stream.write(struct.pack("f", fscale))
        of_phi_stream.write(struct.pack("f", foldmid_x))
        of_phi_stream.write(struct.pack("f", foldmid_y))
        of_phi_stream.write(struct.pack("f", foldmid_z))

    phimap_in.clear()


def expand_grasp(mgrid, phimap_in):
    """
    Placeholder function for expanding potential map for GRASP format.
    """
    return phimap_in


def write_phi_map_grasp(formatflag, phimap_expanded, of_phi_stream):
    """
    Writes potential map data to file in binary format for GRASP format.
    """
    for value in phimap_expanded:
        datavalue = float(value)
        of_phi_stream.write(struct.pack("f", datavalue))


def write_potential_insight(
    phimap4, phi_file, fg_box_center, grid, f_scale, rgc_file_map
):
    """
    Writes potential map in INSIGHT format to a formatted text file.

    Writes potential map data to a text file formatted according to
    INSIGHT potential map format.

    Args:
        phimap4 (list): Potential map data.
        phi_file (str): Output file path.
        fg_box_center (tuple): Grid center (x, y, z).
        grid (int): Grid size.
        f_scale (float): Scale factor.
        rgc_file_map (str): Metadata string for output file header.
    """
    print(f"Potential map written in INSIGHT format to file {phi_file}\n")

    INSIGHT_IVARY = 0
    INSIGHT_NBYTE = 4
    INSIGHT_INTDAT = 0
    INSIGHT_XANG = 90.0
    INSIGHT_YANG = 90.0
    INSIGHT_ZANG = 90.0

    intx = inty = intz = grid - 1

    xmax = max(fg_box_center)
    range_val = (grid - 1.0) / (2.0 * f_scale)
    extent = range_val + xmax
    xyzstart = [(center - range_val) / extent for center in fg_box_center]
    xyzend = xyzstart

    with open(phi_file, "w") as of_phi_stream:
        of_phi_stream.write(f"{rgc_file_map}\n")

        of_phi_stream.write(
            f"{INSIGHT_IVARY} {INSIGHT_NBYTE} {INSIGHT_INTDAT} {extent} {extent} {extent} "
            f"{INSIGHT_XANG} {INSIGHT_YANG} {INSIGHT_ZANG} "
            f"{xyzstart[0]} {xyzend[0]} {xyzstart[1]} {xyzend[1]} {xyzstart[2]} {xyzend[2]} "
            f"{intx} {inty} {intz}\n"
        )

        write_phi_map_insight(phimap4, of_phi_stream)


def write_phi_map_insight(phimap, of_phi_stream):
    """
    Writes potential map in formatted text format for INSIGHT.
    """
    for i, value in enumerate(phimap):
        of_phi_stream.write(f"{value:8.5f} ")
        if (i + 1) % 10 == 0:
            of_phi_stream.write("\n")
    of_phi_stream.write("\n")
