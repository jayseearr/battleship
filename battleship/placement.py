#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 21:07:16 2023

@author: jason

Class definition for Placement object
"""

#%%

### Imports ###

import numpy as np

# Import from this package
from battleship.coord import Coord


#%%

### Class Definition ###

class Placement:
    
    def __init__(self, coord=None, heading=None, length=None):
        """
        Creates a Placement instance representing a ship's location on 
        a board. This location consists of a coordinate, heading (facing),
        and ship length.

        Parameters
        ----------
        coord : tuple
            Two-element (row,col) tuple.
        heading : str
            A single character indicating which way the ship is facing. 'N' 
            for North, 'S' for South, 'E' for East, 'W' for West. Typically
            only 'N' and 'W' are used since the other two are redundant for 
            describing the spaces a ship occupies.
        length : int
            The number of consecutive coordinates a ship occupies.

        Returns
        -------
        None.

        """
        if isinstance(coord, dict):
            self.__init__(coord['coord'], coord['heading'], coord['length'])
            return
        self._coord = None
        self._heading = None
        self._length = None
        # set properties using setters
        self.coord = coord
        self.heading = heading
        self.length = length
      
    def __repr__(self):
        return (f"Placement({self._coord!r}, {self._heading!r}, "
                f"{self._length!r})")
    ### Properties ###
    
    @property
    def coord(self):
        return self._coord
    
    @coord.setter
    def coord(self, value):
        if isinstance(value, tuple):
            if len(value) != 2:
                raise ValueError("coord must be a two-element tuple.")
        elif isinstance(value, Coord):
            value = (value[0], value[1])
        self._coord = value
        
    @property
    def heading(self):
        return self._heading
    
    @heading.setter
    def heading(self, value):
        if value is None:
            self._heading = None
            return
        value = value.upper()
        if value in ["N", "S", "E", "W"]:
            self._heading = value
        elif value in ["NORTH", "SOUTH", "EAST", "WEST"]:
            self._heading = value[0]
        else:
            raise ValueError("Heading must be 'North', 'South', "
                             "'East', 'West' (or abbreviation).")
        
    @property
    def length(self):
        return self._length
    
    @length.setter
    def length(self, value):
        if isinstance(value, int):
            self._length = value
        else:
            raise ValueError("length must be an integer.")
        
    ### Computed Properties ###
    
    def coords(self):
        """
        Returns a list of the coordinates occupied by the placement.

        Returns
        -------
        coords : list
            List of (row,col) tuples, with length equal to the placement's
            length.

        """
        if self.heading == "N":
            ds = (1,0)
        elif self.heading == "S":
            ds = (-1,0)
        elif self.heading == "E":
            ds = (0,-1)
        elif self.heading == "W":
            ds = (0,1)
            
        ds = np.vstack(([0,0], 
                        np.cumsum(np.tile(ds, (self.length-1,1)), axis=0)))
        return [(delta[0] + self.coord[0], delta[1] + self.coord[1])
                for delta in ds]
        
    def rows(self):
        """
        Returns the rows occupied by the placement as a numpy array.

        Returns
        -------
        numpy array.

        """
        return np.array([coord[0] for coord in self.coords()])
    
    def cols(self):
        """
        Returns the cols occupied by the placement as a numpy array.

        Returns
        -------
        numpy array.

        """
        return np.array([coord[1] for coord in self.coords()])
    
    ### Overloaded methods ###
    
    def __len__(self):
        """
        The length of the ship in this placement.

        Returns
        -------
        int

        """
        return self.length
    
    def __getitem__(self, i):
        """
        For input parameter i, returns the coordinate at the ith position in
        the ship. The first (i=0) coordinate is the one at the placement's
        coord property.

        Parameters
        ----------
        i : int
            The index of the placement coordinate to return.

        Returns
        -------
        coord : tuple
            Two-element (row,col) tuple of the coordinate at index i.

        """
        return self.coords()[i]
    
    def __eq__(self, other):
        """
        Tests for equality between the placement and another placement object
        by checking that the two placements occupy exactly the same 
        coordinates.
        
        Note that placements with different coord properties and opposite 
        headings could still be equal because they occupy an identical set of
        coordinates. For example, the following two placements are defined
        differently, but are equivalent: 
            {'coord':(1,2), 'heading':'N', 'length':3} --> (1,2), (2,2), (3,2)
            {'coord':(3,2), 'heading':'S', 'length':3} --> (3,2), (2,2), (1,2)

        Parameters
        ----------
        other : placement
            The placement to compare against this placement.

        Returns
        -------
        bool 
            True if the two placements share identical coordinates. 
            False otherwise.

        """
        return set(self.coords()) == set(other.coords())
    
    ### Other methods ###
    
    def dist_to_coord(self, coord, method="dist"):
        """
        Returns an array of the  distances between each coordinate in the 
        placement and the input coordinate.

        Parameters
        ----------
        coord : tuple
            A two-element tuple
            
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
        dist : numpy array
            The distance between each coordinate in the placement
            and the input coord parameter.

        """
        rows, cols = zip(*self.coords())
        dr = np.array(rows) - coord[0]
        dc = np.array(cols) - coord[1]
        if method == "manhattan":
            dist = np.abs(dr) + np.abs(dc)
        else:
            dist = dr**2 + dc**2
            if method == "dist":
                dist = np.sqrt(dist)
        return dist
    
    def total_dist_to_coord(self, coord, method="dist"):
        """
        Returns the sum of distances between each coordinate in the 
        placement and the input coordinate.

        Parameters
        ----------
        coord : tuple
            A two-element tuple
            
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
        dist : numpy array
            The distance between each coordinate in the placement
            and the input coord parameter.

        """
        return np.sum(self.dist_to_coord(coord, method))
    
    def total_dist_to_placement(self, other, method="dist"):
        """
        Returns the sum of the absolute distances between each pair of 
        coordinates in two placements. Useful for determining how "far apart"
        the placements are, in a qualitative sense.

        Parameters
        ----------
        other : Placement
            Another placement instance.
        
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
            The sum of all distances between each pair of coordinates in 
            placement1 and placement2.


        """
        dist = [np.sum(self.dist_to_coord(coord, method)) 
                for coord in other.coords()]
        return sum([x for x in dist])
    
    # def dist_stats_to_placement(self, other):
    #     """
    #     Returns the minimum and maximum distances between pairs of coordinates 
    #     in this placement and the input placement.

    #     Parameters
    #     ----------
    #     other : Placement
    #         A placement object.
            

    #     Returns
    #     -------
    #     stats : tuple
    #         Two-element tuple with the min and max distances between 
    #         coordinates in the placements.

    #     """
    #     coords1 = np.array(self.coords())
    #     coords2 = np.array(other.coords())
    #     dR = (np.tile(coords1[:,0], (len(coords2),1)) - 
    #           np.tile(coords2[:,0], (len(coords1),1)).T)
    #     dC = (np.tile(coords1[:,1], (len(coords2),1)) - 
    #           np.tile(coords2[:,1], (len(coords1),1)).T)
    #     D = np.sqrt(dR**2 + dC**2)
    #     return (D.min(), D.max())
        
    def placement_dist_matrix(self, other, method="dist"):
        """
        Returns a matrix of the distances between coordinates in this placement
        and another placement.

        Parameters
        ----------
        other : Placement
            A placement object.
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
        dist_matrix : N x M numpy array
            If this placement is N coordinates long and the other placement is
            M coordinates long, the returned matrix will be N x M.
            Element dist_matrix[i,j] gives the distance between coordinate i
            of this placement and coordinate j of the other placement.
            dist_matrix.min() - 1 is the "buffer" between the two placements 
            (i.e., the minimum number of spaces separating the two).


        """
        coords1 = np.array(self.coords())
        coords2 = np.array(other.coords())
        dR = (np.tile(coords1[:,0], (len(coords2),1)) - 
              np.tile(coords2[:,0], (len(coords1),1)).T)
        dC = (np.tile(coords1[:,1], (len(coords2),1)) - 
              np.tile(coords2[:,1], (len(coords1),1)).T)
        if method == "manhattan":
            D = np.abs(dR) + np.abs(dC)
        else:
            D = dR**2 + dC**2
            if method == "dist":
                D = np.sqrt(D)
            elif method == "dist2":
                pass
            else:
                raise ValueError("Invalid method (try 'dist', 'dist2', or "
                                 "'manhattan'.")
        return D
        
    def dists_to_placements(self, 
                            others, 
                            method="dist", 
                            output_as_buffer=False):
        """
        

        Parameters
        ----------
        others : list
            List of placement objects.
        method : str, optional
            String describing the method to use for calculating and adding 
            coordinate-to-coordinate distances. Options are:
                'manhattan' : row and column distances for each pair of 
                placements are calculated independently, then the absolute 
                values are added together.
                'dist' : The standard euclidean distance between the two
                coordinates (i.e., sqrt((x2-x1)^2 + (y2-y1)^2)).
                'dist2' : The euclidean distance squared.
            The default is 'dist'.
        output_as_buffer : bool, optional
            If True, the output list will contain the buffer (rather than the
            distance) between each other placement and this one. "Buffer" is
            just the number of spaces separating any two coordinates, rather
            than the number distance between coordinates (this definition is
            unclear if the coordinates are not on the same row or column, so
            it's an approximation in those cases; consider use cases carefully).
            

        Returns
        -------
        A list with the same length as others. Element i of this list is the
        minimum distance separating this placement with the ith other placement.

        """
        if not isinstance(others, (list,tuple)):
            raise TypeError("others input must be a list/tuple; wrap in [] if "
                            "it is a single Placement.")
        dists = [self.placement_dist_matrix(other, method).min()
                 for other in others]
        if output_as_buffer:
            dists = [d-1 for d in dists]
        return dists
        
    def total_dist_grid(self, board_size, method="dist"):
        """
        Returns an N x N array with element (r,c) equal to the total distance
        from (r,c) to the coordinates in the placement. The minimum distance
        will always be at the middle of the placement coordinates.
        
        This is useful for computing how isolated points on a board are from a
        given placement.

        Parameters
        ----------
        board_size : int
            The size of the board on which the placement is assumed to be
            located. The returned array will be a square array of this size.
        method : str, optional
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
        total_dist : numpy array
            A square array (board_size x board_size). Each element is the 
            sum of the distances between that element's location and all of 
            the coordinates in the placement.

        """
        x = np.arange(board_size)
        R,C = np.meshgrid(x, x, indexing='ij')
        dist_grid = np.zeros((board_size, board_size))
        for coord in self.coords():
            dr = R - coord[0]
            dc = C - coord[1]
            if method == "manhattan":
                next_dist = np.abs(dr) + np.abs(dc)
            else:
                next_dist = dr**2 + dc**2
                if method == "dist":
                    next_dist = np.sqrt(next_dist)
            dist_grid += next_dist
        return dist_grid