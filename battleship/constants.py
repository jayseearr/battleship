#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 20:53:24 2023

@author: jason

This file defines package-level constants that are used in more than one
module. Access these constant using:
    
import battleship as bs
bs.constants.XX (where XX is the constant's name)

Individual module files may still define their own constants for use in that
file only; recommend using an underscore prefix.

"""

SUNK_OFFSET = 0.1

import enum

### Enums ###

class ShipType(enum.IntEnum):
    PATROL = 1
    DESTROYER = 2
    SUBMARINE = 3
    BATTLESHIP = 4
    CARRIER = 5
    
class TargetValue(enum.IntEnum):
    """An Enum used to track hits/misses on a board's target grid.
    Can be compared for equality to integers."""
    UNKNOWN = -2
    MISS = -1
    HIT = 0
    
class Align(enum.Enum):
    """Enum used to identify the directional alignment of ships; either 
    VERTICAL or HORIZONTAL (or any, which allows either direction).
    """
    ANY = 0
    VERTICAL = 1
    HORIZONTAL = 2
