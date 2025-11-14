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


class CException(Exception):
    pass


class CWarning(Warning):
    pass


class CUnknownGridEngFile(CException):
    def __init__(self, strFileName):
        print(f"data file {strFileName} for analytic grid energy not present")


class CCrgatnError(CException):
    def __init__(self):
        print("PROBLEM WITH prgiCrgAt (crgatn)")


class CUnknownPreviousPhiFile(CException):
    def __init__(self, strPreviousPhiFile):
        print(f"THE INPUT PHI FILE {strPreviousPhiFile} DOES NOT EXIST")


class CUnmatchPotentialMap(CException):
    def __init__(self):
        print("THE TWO POTENTIAL MAPS DO NOT MATCH")


class CUnknownInFrcFile(CWarning):
    def __init__(self, strFrciFile):
        print(f"THE INPUT FRC FILE {strFrciFile} DOES NOT EXIST (EXITING...)")


class CNoAtomInfo(CWarning):
    def __init__(self, strFrciFile):
        print(
            f"THIS UNFORMATTED FILE {strFrciFile} DOES NOT CONTAIN ATOM INFO (ATOM INFO FLAG TRUNED OFF)"
        )


class CCalcReactForceError(CWarning):
    def __init__(self):
        print(
            "CANNOT CALCULATE REACTION FORCES W/O USING INTERNAL (SELF) COORDINATES (EXITING...)"
        )


class CSitePhiError(CWarning):
    def __init__(self):
        print("Something unclear with sitephi array (will be fixed soon...)")


class CNoIDebMap(CWarning):
    def __init__(self):
        print(
            "WRTSIT: THESE SALT CONCENTRATIONS DO NOT HAVE THE BENEFIT OF IDEBMAP (AS YET)"
        )


class CUntestedPhicon(CWarning):
    def __init__(self):
        print("PHICON: this option has not been tested yet")


class CNoPotential2CrgConcentrate(CWarning):
    def __init__(self):
        print(
            "CANNOT CONVERT FROM POTENTIALS TO CONCENTRATIONS IF THE IONIC STRENTH IS ZERO"
        )


class CEmptyPhiMap(CWarning):
    def __init__(self):
        print("THE REQUESTED OUPUT PHIMAP IS EMPTY.")
