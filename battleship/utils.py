#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 21:13:24 2023

@author: jason

Functions that are used across multiple files/class definitions in the
Battleship package. These functions should not be consistered public.

"""

#%%

### Imports ###

import numpy as np


# Imports from this package
from battleship.constants import Align



#%% 

### Coordinate Selection & Math Functions ###


def is_on_board(x, size):
    """
    Returns True if the input coordinate is on the board with specified size,
    and False otherwise. This is true when the row and column contained in 
    x are between 0 and size - 1.

    Parameters
    ----------
    x : Coord or tuple
        A tuple with two elements specifying row and column of the coordinate,
        or a Coord instance.
    size : int
        The number of coordinates along one dimension of a Board.

    Returns
    -------
    Bool
        True if x is on the board, False otherwise.

    """
    x = tuple(x)    # convert from Coord or list
    return (x[0] >= 0 and x[0] < size and x[1] >= 0 and x[1] < size)


def random_choice(x, p=None):
    """
    Returns a single element from the input array (x) randomly sampled with
    probability distribution p.

    Parameters
    ----------
    x : list or numpy array
        The items from which to sample.
    p : numpy array, optional
        An array with the same length as x. The probability of picking x[i]
        is given by p[i]. p should be normalized so that sum(p) == 1.
        The default is None, which samples from a uniform probability 
        distribution.

    Returns
    -------
        A single element from x. If x has length 1, the sole element is 
        returned. If x is empty, an empty numpy array is returned.

    """
    return x[np.random.choice(range(len(x)), p=p)]


def rotate_coords(coords, degrees, board_size):
    """
    Rotate the input coordinates by an angle about the center of the board.

    Parameters
    ----------
    coords : List of tuples
        A list of 2-element coordinate tuples to rotate.
    degrees : Float
        The angle to rotate the coordinates. Positive cooresponds to clockwise.
    board_size : Int
        The number of spaces along one edge of the board on which coords are
        placed. 

    Returns
    -------
    coords : List of tuples
        Same size as the input coord. The ith element is the rotated ith 
        element of the input list.

    """
    # for 90 degrees, (0,0) --> (0,9); (0,9) --> (9,9); (9,9) --> (9,0); (9,0) --> (0,0)
    # for 180 degrees, (0,0) --> (9,9); (0,9) --> (9,0); (9,9) --> (0,0); (9,0) --> (0,9)    
    # for -90 degrees, (0,0) --> (9,0); (0,9) --> (0,0); (9,9) --> (0,9); (9,0) --> (9,9)    
    angle = np.pi * degrees / 180
    sine = np.sin(angle)
    cosine = np.cos(angle)
    center = (board_size - 1) / 2
    if len(coords[0]) == 1:
        raise TypeError("coords should be a list (even for a single coord).")
    coords = [np.array(pt) - center * np.ones(2) for pt in coords]
    T = np.array(([cosine, sine], [-sine, cosine]))
    coords = [tuple(T @ pt + center * np.ones(2)) for pt in coords]
    return coords
    
def mirror_coords(coords, axis, board_size):
    """
    Flips the input board coordinates along the input axis. Axis=0 flips 
    vertically (up/down), and axis=1 flips horizontally (left/right).

    Parameters
    ----------
    coords : List of tuples
        A list of 2-element coordinate tuples to mirror.
    axis : Int (0 or 1)
        The axis along which to mirror the coordinates. This should be 0 for
        vertical or up/down mirroring (i.e., flipping across a horizontal line)
        and 1 for horizontal or left/right mirroring (i.e., flipping across
        a vertical line).
    board_size : Int
        The number of spaces along one edge of the board on which coords are
        placed. 

    Returns
    -------
    list
        List of tuples
            Same size as the input coord. The ith element is the mirrored ith 
            element of the input list.

    """
    center = (board_size - 1) / 2
    if len(coords[0]) == 1:
        raise TypeError("coords should be a list (even for a single coord).")
    coords = np.array(coords) - np.tile(center, (len(coords),2))
    coords[:,axis] = -coords[:,axis]
    coords = coords + np.tile(center, (len(coords),2))
    return [tuple(pt) for pt in coords]


#%%

### Parameter Parsing Functions ###
def parse_alignment(alignment):
    """
    Returns an Align enum based on the input string or enum.

    Parameters
    ----------
    alignment : Enum or str (case insensitive)
        One of the following strings:
            "North", "South", "N/S", "N", "S": returns Align.VERTICAL
            "East", "West", "E/W", "W", "E": returns Align.HORIZONTAL
            "Any", "All": returns Align.ANY

    Returns
    -------
    An Align object consistent with the input str (or the same Align object).

    """
    if alignment == None:
        return Align.ANY
    if isinstance(alignment, Align):
        return alignment
    if isinstance(alignment, str):
        alignment = alignment.upper()
    else:
        raise TypeError("alignment parameter must be a string or Align value.")
    if alignment in ["ANY", "ALL", "BOTH"]:
        alignment = Align.ANY
    elif alignment in ["N", "NS", "N/S", "VERTICAL", "V", "NORTH", "SOUTH", 
                       "NORTH/SOUTH", "S", "SN", "S/N", "N-S", "S-N"]:
        alignment = Align.VERTICAL    
    elif alignment in ["E", "EW", "E/W", "HORIZONTAL", "H", "EAST", "WEST", 
                       "EAST/WEST", "W", "WE", "W/E", "W-E", "E-W"]:
        alignment = Align.HORIZONTAL   
    else:
        raise ValueError("String input must indicate a direction "
                         "('north', 'east', 'horizontal', 'any', etc.)")
    return alignment