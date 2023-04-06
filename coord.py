#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 21:05:05 2023

@author: jason

Class definition for Coord object

"""

#%%

### Imports ###

import numpy as np

# Imports from this package
# None

#%%

### Class Definition ###
class Coord:
    
    __LETTER_OFFSET = 65
    __TOLERANCE__ = 1e-6
    
    def __init__(self, a, b=None):
        """
        A representation of a single space on the Battleship board. 
        A coord instance may be represented as row and column indices, or as
        a string with a letter and a number.

        A Coord may be initialized in any of the following formats:
        Row/Col format:
            Coord(row,col)      Row and col are integers between 0 and the
                                size of the board minus 1.
        Tuple format:
            Coord((row,col))    The tuple (row,col) is passed as a single 
                                parameter. Row and col are again integers.
        String format:
            Coord("X#")         X is a letter between A and the letter 
                                corresponding to the board size (typically J,
                                for a 10 x 10 board), and # is an integer 
                                between 1 and the board size (typically 10).
        Parameters
        ----------
        a : str or tuple or int
            See input formats, above.
        b : int, optional
            The column index. Only used in Row/Col input format, described 
            above. The default is None.

        Raises
        ------
        ValueError
            DESCRIPTION.
        TypeError
            When the inputs are not str, tuple, or int.

        Returns
        -------
        An instance of a Coord object. Important properties are:
            rowcol      A two-element tuple that gives the row a column of 
                        the Coord.
            lbl         The string representation of the row and column 
                        (e.g., "C6").

        """
        if b is not None:
            r = a
            c = b
            if not (isinstance(a,(int,np.integer)) and 
                    isinstance(b,(int,np.integer))):
                raise ValueError("When specifying a Coord with 2 arguments, they "
                                 "must both be integers.")
        elif isinstance(a, list):
            if isinstance(a[0], int) and len(a) == 2:
                r,c = a[0], a[1]
            else:
                raise ValueError("Cannot create a Coord out of a list unless "
                                 "it has only 2 elements.")
        elif isinstance(a, tuple):
            r,c = a
        elif isinstance(a, str):
            r,c = self._str_to_rc(a)
        elif isinstance(a, np.ndarray) and len(a) == 2:
            r,c = a[0], a[1]
        else:
            raise TypeError("Coord argument types are: tuple, list, str, int.")
        if (np.abs(r - np.round(r)) > self.__TOLERANCE__ or 
            np.abs(c - np.round(c)) > self.__TOLERANCE__):
                raise ValueError("Row/column values must be integers.")
                
        self._rowcol = int(np.round(r)), int(np.round(c))
        self._value = None
        
    @classmethod    
    def _str_to_rc(cls, s):
        """Returns a string with a letter and number (1 or 2 digits) that
        correspond to the Coord's row and column values.
        """        
        r = ord(s[0].upper()) - cls.__LETTER_OFFSET
        c = int(s[1:]) - 1 
        return r,c
    
    @property
    def rowcol(self):
        """Returns a tuple with the row and column values describing the 
        Coord's location."""
        return self._rowcol
    
    @property
    def lbl(self):
        """Returns a string description of the Coord's location, such as 
        A1, C5, J3, etc.
        """
        return f"{chr(self._rowcol[0] + self.__LETTER_OFFSET)}" \
            f"{self._rowcol[1]+1}"
    
    @property
    def value(self):
        """Returns the current value of the Coord's value variable (a scalar).
        """
        return self._value
    
    @value.setter
    def value(self, x):
        """Set the Coord's value variable to input scalar x."""
        self._value = x
        
    def __str__(self):
        """Returns the label strings for the Coord's location."""
        return self.lbl
    
    def __len__(self):
        return len(self.rowcol)
    
    def __eq__(self, other):
        """Compares the Coord's row/column and value (if any) variables. 
        If they are the same as the second item in the comparison, the two 
        instances are equal.
        """
        if isinstance(other, (tuple, list)):
            return (len(other) == 2 
                    and self.rowcol[0] == other[0] 
                    and self.rowcol[1] == other[1])
        return self._rowcol == other.rowcol and self._value == other.value
    
    def __getitem__(self, key):
        """Returns the row or column value of the Coord for key = 0 and key = 1,
        respectively.
        """
        return self._rowcol[key]
    
    def __setitem__(self, key, value):
        """Set the row or column value of the Coord to the input value."""
        if key < 0 or key > 1:
            raise ValueError("Index must be 0 or 1.")
        if value < 0 or not isinstance(value, (np.integer, int)):
            raise ValueError("Coord row/column cannot be set to a non-integer type.")
        rowcol = list(self._rowcol)
        rowcol[key] = value
        self._rowcol = rowcol[0], rowcol[1]
    
    def __add__(self, other):
        """Adds another Coord or a 2-element tuple to the Coord object."""
        other = self.check_other(other)
        return Coord((self[0] + other[0], self[1] + other[1]))
    
    def __sub__(self, other):
        """Adds another Coord or a 2-element tuple to the Coord object."""
        other = self.check_other(other)
        return Coord((self[0] - other[0], self[1] - other[1]))
    
    def __mul__(self, other):
        """Multiplies another Coord or a 2-element tuple to the Coord object
        in an element-by-element manner (not matrix multiplication).
        
        If the input is a scalar, multiplies both row and column values by
        that scalar."""
        other = self.check_other(other)
        return Coord((self[0] * other[0], self[1] * other[1]))
    
    def check_other(self, other):
        """Returns a properly-formatted 'other' parameter, which can be input
        as any of the following:
            scalar (integer-valued)
            Coord
            tuple or list (two-element, both integer-valued).
            
        The returned value will be a two-element tuple (even for Coord input).
        """
        if isinstance(other, (int,np.integer,float,np.floating)):
            other = (other, other)
        if len(other) > 2:
            raise ValueError("other must be a scalar, Coord, or two-element "
                             "list or tuple.")
        if not (int(other[0] == other[0]) and int(other[1] == other[1])):
            raise ValueError("other must be integer-valued.")
        return other
    
    def __repr__(self):
        return f'Coord({self.rowcol})'
    
    ### Comparison methods ###
    
    def next_to(self, other):
        """
        Determines if the Coord is directly adjacent to another Coord.
        See also 'diagonal_to'.
        
        Parameters
        ----------
        other : Coord
            An instance of Coord, or a 2-element list, tuple, or numpy.ndarray.

        Returns
        -------
        Bool
            True if the input touches this Coord, False otherwise.

        """
        return (self[0] - other[0])**2 + (self[1] - other[1])**2 == 1
    
    def diagonal_to(self, other):
        """
        Determines if the Coord is diagonally adjacent to another Coord.
        See also 'next_to'.
        
        Parameters
        ----------
        other : Coord
            An instance of Coord, or a 2-element list, tuple, or numpy.ndarray.

        Returns
        -------
        Bool
            True if the input is diagonally adjacent to this Coord, and
            False otherwise.

        """
        return (np.abs(self[0] - other[0]) == 1 and 
                np.abs(self[1] - other[1]) == 1)
    
    def dist_to(self, other, method="dist"):
        """
        Returns the distance to the input coordinate.

        Parameters
        ----------
        other : Coord
            Another Coord object.
        method : str
            String describing the method to use for calculating and adding 
            coordinate-to-coordinate distances. Options are:
                'manhattan' : row and column distances for each pair of 
                placements are calculated independently, then the absolute 
                values are added together.
                'dist' : The standard euclidean distance between the two
                coordinates (i.e., sqrt((x2-x1)^2 + (y2-y1)^2)).
                'dist2' : The euclidean distance squared.
            The default is 'dist'.

        Returns
        -------
        dist : float
            The distance to the input coordinate.

        """
        if isinstance(other, tuple):
            other = Coord(other)
        delta = other - self
        if method == "manhattan":
            dist = np.abs(delta[0]) + np.abs(delta[1])
        else:
            dist = np.sum(delta[0]**2 + delta[1]**2)
            if method == "dist":
                dist = np.sqrt(dist)
        return dist
    
    def dists_to(self, others, method="dist"):
        """
        Returns the distance to the input coordinate.

        Parameters
        ----------
        others : list
            A list of Coords or tuple coordinates.
        method : str
            String describing the method to use for calculating and adding 
            coordinate-to-coordinate distances. Options are:
                'manhattan' : row and column distances for each pair of 
                placements are calculated independently, then the absolute 
                values are added together.
                'dist' : The standard euclidean distance between the two
                coordinates (i.e., sqrt((x2-x1)^2 + (y2-y1)^2)).
                'dist2' : The euclidean distance squared.
            The default is 'dist'.

        Returns
        -------
        dists : list
            A list of floats containing the distance between each element of
            others and this Coord.

        """
        if isinstance(others, (Coord, tuple)):
            return self.dist_to(others, method)
        else: 
            return [self.dist_to(coord, method) for coord in others]