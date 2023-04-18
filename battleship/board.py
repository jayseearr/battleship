#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 21:08:49 2023

@author: jason

Class definition for Board object
"""

#%%

### Imports ###

import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt

# Imports from this package
from battleship import constants
from battleship import utils

from battleship.ship import Ship
from battleship.coord import Coord
from battleship.placement import Placement
from battleship.viz import Viz



#%%

### Class Definition ###

class Board:
    
    def __init__(self, size=10):
        """
        Creates an instance of a Battleship game Board. The board has a 
        fleet of 5 ships, an ocean grid upon which ships are arranged,
        and a target grid where hits/misses on opponent ships are tracked.

        Parameters
        ----------
        size : int, optional
            The number of spaces along each dimension of the board.
            The default is 10.

        Returns
        -------
        Board instance.
        
        Properties:
            - size : int
            - ocean_grid : numpy array
            - target_grid : numpy array
            - fleet : dict
            - ship_placements : dict
            
        Useful methods that should be called by a Player instance include the
        following:
            - reset
            - all_valid_target_placements
            - add_fleet
            - is_ready_to_play
            - incoming_at_coord
            - update_target_grid
            - possible_ship_types_at_coord
            - possible_targets_grid
            - find_hits
            - targets_around_all_hits
            - colinear_targets_along
            
        Factory methods:
            - copy
            - outcome
            - test
            

        """
        if size < 1:
            raise ValueError("size must be a positive integer.")
        self._size = size
        self._fleet = {}
        self._ship_placements = {}
        self._ocean_grid = np.zeros((size, size), dtype=np.int8)      
        self._target_grid = (np.ones((size, size)) 
                             * constants.TargetValue.UNKNOWN)
        
    def __str__(self):
        """Returns a string that uses text to represent the target map
        and grid (map on top, grid on the bottom). On the map, an 'O' represents
        a miss, an 'X' represents a hit, and '-' means no shot has been fired
        at that space. 
        On the grid, a '-' means no ship, an integer 1-5 means a ship slot with
        no damage, and a letter means a slot with damage (the letter matches
        the ship name, so P means a damaged patrol boat slot).
        """
        return self.color_str()
    
    def __repr__(self):
        return f"Board({self.size})"
    
    # # Re-initalization
    
    # def reset(self):
    #     """
    #     Removes all ships and pegs (i.e., ship damage and hit/miss shots) from
    #     the target and ocean grids.

    #     Returns
    #     -------
    #     None.

    #     """
    #     board = Board(self.size)
    #     # self._fleet = board.fleet
    #     # self._target_grid = board.target_grid
    #     # self._ocean_grid = board.ocean_grid
        
    # Factory methods
    
    # @classmethod
    # def copy(cls, other_board):
    #     """Creates a copy of the input board. The copy can be manipulated
    #     independently of the original board (i.e., all attributes are
    #     copies, not the same object)."""
        
    #     board = cls(other_board.size)
    #     placements = dict([(k,other_board.placement_for_ship(k)) 
    #                   for k in other_board.fleet])
    #     board.add_fleet(placements)
    #     board.target_grid = other_board.target_grid.copy()
    #     return board
    
    def copy(self):
        """
        Creates a copy of the board.
        
        Returns
        -------
        copied_board : Board
            A new copy of the board.

        """
        new_board = Board(self.size)
        new_board.add_fleet(self.ship_placements)
        new_board._target_grid = self.target_grid.copy()
        return new_board
        
    @classmethod
    def outcome(cls, coord, hit, sunk=False, sunk_ship_type=None):
        """
        Returns an outcome dictionary for the input shot information.

        Parameters
        ----------
        coord : 
            Two-element tuple indicating the (row,column) of the shot.
        hit : Bool
            True if a ship was hit at coord.
        sunk : Bool
            True if a ship was sunk.
        sunk_ship_type : Bool, optional
            The ShipType value (int 1-5) of the ship that was sunk.
            The default is None, which means no ship sunk.

        Returns
        -------
        outcome : dict
            A dictionary with the following key/value pairs:
                'coord': (row,col) tuple of the shot location
                'hit': Bool; True if shot was a hit
                'sunk' : Bool; True if ship was sunk
                'sunk_ship_type' : ShipType enum (or int 1-5) indicating 
                                 the type of ship sunk (if one was sunk).

        """
        if sunk_ship_type is not None:
            if not sunk:
                raise ValueError("'sunk' must be True when "
                                 "'sunk_ship_type' is not None.")
        if sunk and not hit:
            raise ValueError("'hit' must be True when 'sunk' is True.")
        return {'coord': coord, 
                'hit': hit,
                'sunk': sunk,
                'sunk_ship_type': sunk_ship_type}
        
    @classmethod
    def test(cls,layout=0,targets=True):
        """
        Returns a test board with ships at arbitrary (consistent) locations.

        Parameters
        ----------
        targets : Bool, optional
            Adds some arbitrary targets and outcomes on the target board,
            as well as hits to the fleet if True. The default is True.

        Returns
        -------
        board : instance of Board.

        """
        board = cls(10)
        
        if layout == 0:
            board.add_ship(1, Coord("J9"), "W") 
            board.add_ship(2, Coord("E8"), "N") 
            board.add_ship(4, Coord("G2"), "W") 
            board.add_ship(3, Coord("B6"), "W") 
            board.add_ship(5, Coord("A5"), "N")
            
            if targets:
                board.update_target_grid(Board.outcome((0,4), True))
                board.update_target_grid(Board.outcome((0,3), True))
                board.update_target_grid(Board.outcome((0,5), True, True, 
                                                       constants.ShipType(3)))
                board.update_target_grid(Board.outcome((2,8), False))
                board.update_target_grid(Board.outcome((9,7), False))
                board.update_target_grid(Board.outcome((6,7), False))
                board.update_target_grid(Board.outcome((4,4), False))
                board.update_target_grid(Board.outcome((3,2), False))
                board.update_target_grid(Board.outcome((1,6), False))
                
                board.fleet[constants.ShipType(2)].hit(0)
                board.fleet[constants.ShipType(2)].hit(1)
                board.fleet[constants.ShipType(2)].hit(2)
                board.fleet[constants.ShipType(4)].hit(3)
                board.fleet[constants.ShipType(5)].hit(2)
        
        elif layout == 1:
            board.update_target_grid(Board.outcome((1,1), False))
            board.update_target_grid(Board.outcome((1,2), True))
            board.update_target_grid(Board.outcome((1,3), True))
            board.update_target_grid(Board.outcome((1,4), True, True, 
                                                   constants.ShipType(3)))
            board.update_target_grid(Board.outcome((2,4), True))
            board.update_target_grid(Board.outcome((3,4), True))
            
        elif layout == 2:
            # - 0 0 - -
            # 0 X X - 0
            # - 0 0 - 0
            # - X X 3 -
            board.update_target_grid(Board.outcome((0,1), False))
            board.update_target_grid(Board.outcome((0,2), False))
            
            board.update_target_grid(Board.outcome((1,0), False))
            board.update_target_grid(Board.outcome((1,1), True))
            board.update_target_grid(Board.outcome((1,2), True))
            board.update_target_grid(Board.outcome((1,4), False))
            
            board.update_target_grid(Board.outcome((2,1), False))
            board.update_target_grid(Board.outcome((2,2), False))
            board.update_target_grid(Board.outcome((2,4), False))
            
            board.update_target_grid(Board.outcome((3,1), True))
            board.update_target_grid(Board.outcome((3,2), True))
            board.update_target_grid(Board.outcome((3,3), True, True, 3))
            
        return board

    # Properties
    
    @property
    def size(self):
        return self._size
    
    @property
    def ocean_grid(self):
        return self._ocean_grid

    @property
    def target_grid(self):
        return self._target_grid
    
    @property
    def fleet(self):
        return self._fleet
    # fleet does not need a setter because it is a dictionary, and items are
    # only added by key/value access.
    
    @property
    def ship_placements(self):
        return self._ship_placements
    
    # General Coordinate Methods
    
    def rel_coords_for_heading(self, heading, length):
        """
        Returns arrays containing the row and column indices that, when
        added to an arbitrary coordinate, equal the coordinates that would be
        occuppied by a ship with the input heading and length placed at 
        that coordinate. 
        
        For example, heading = "N" and length = 3 will return 
        ([0,1,2], [0,0,0]). If a ship is placed at a coordinate (R,C) with 
        these relative coordinates and with a heading of North and length 3, 
        the ship will occupy coordinates (R,C), (R+1,C), and (R+2,C).
        
        rel_coords_for_heading("E", 4) returns ([0,0,0,0], [0,-1,-2,-3]),
        so placing this ship at (R,C) means it needs to span coordinates
        (R,C), (R,C-1), (R,C-2), and (R,C-3).
        
        The returned row/column arrays will always both begin at 0. One of
        them will be all zeros (depending on whether heading is N/S or E/W), 
        and the other one will be range(0,length).
        Since the returned

        Parameters
        ----------
        heading : str
            A string describing the desired heading of the ship. This can be
            a cardinal direction or single-letter abbreviation ("North" or 
            "N"), or Up/Down/Left/Right, or Vertical/Horizontal. By convention,
            Vertical is the same as North and Horizontal is the same as West;
            i.e., vertical and horizontal headings default toward the origin 
            of the board.
            
        length : int
            The length of a ship.

        Raises
        ------
        ValueError
            Occurs when an invalid heading string is used.

        Returns
        -------
        Tuple of two numpy arrays (each with length equal to length parameter)
            out[0] is the array of row values and out[1] is the array of
            column values that should be added to an arbitrary coordinate 
            to get all coordinates for a ship.

        """
        heading = heading[0].upper()
        if heading == "N" or heading == "D" or heading == "V":
            ds = (1,0)
        elif heading == "S" or heading == "U":
            ds = (-1,0)
        elif heading == "E" or heading == "L":
            ds = (0,-1)
        elif heading == "W" or heading == "R" or heading == "H":
            ds = (0,1)
        else:
            raise ValueError(("heading input must be North, South, East, \
                              or West (first letter is okay, and everything \
                              is case insensitive."))
        if length == 1:
            return (np.array([0]), np.array([0]))
        else:
            ds = np.vstack(([0,0], 
                            np.cumsum(np.tile(ds, (length-1,1)), axis=0)))
            return (ds[:,0], ds[:,1])
        
    # def coords_for_placement(self, placement):
    #     """
    #     Returns the coordinates that correspond to the input placement
    #     (coordinate, heading) and ship length.

    #     Parameters
    #     ----------
    #     placement : dict 
    #         placement['coord']: A coordinate (a tuple with two elements
    #         corresponding to a row and a column) 
    #         placement['heading']: A string denoting a heading 
    #         ("N", "S", "E", or "W").
    #         placement['length']: An int equal to the length of the ship
    #         at this placement. The output will have a number of elements 
    #         equal to length.

    #     Returns
    #     -------
    #     List of coordinates that span the input starting coordinate/heading
    #     and the length.

    #     """
    #     return placement.coords()
    
    def is_valid_coord(self, coords):
        """
        Checks if the input coordinates are valid or not. Returns an array
        of bools with length equal to the input list of coordinates.

        Parameters
        ----------
        coords : list
            A list of coordinates (two-element tuples).

        Returns
        -------
        numpy.ndarray
            Same length as the input coords list. The ith element is True
            if coords[i] is a coordinate that falls on the instance of 
            Board, and False otherwise.

        """
        # rows, cols = zip(*coords)
        # rows = np.array(rows)
        # cols = np.array(cols)
        # return np.array((rows >= 0) * (rows < self.size) * (cols >= 0) *
        #                  (cols < self.size))
        return np.array([True if (rc[0] >= 0 and rc[0] < self.size 
                         and rc[1] >= 0 and rc[1] < self.size)
                         else False for rc in coords])
    
    
    # @classmethod
    # def coords_to_coords_distances(cls, coords1, coords2):
    #     """
    #     Returns a 2d array containing the distances between all pairs of 
    #     coordinates in the two input arrays.
        
    #     Parameters
    #     ----------
    #     coords1 : list
    #         A list of (row,col) tuples.
    #     coords2 : list
    #         A list of (row,col) tuples.
        
    #     Returns
    #     -------
    #     distances : numpy array (2d).
    #         distances[i,j] is the distance between coords1[i] and coords2[j]

    #     """
    #     if isinstance(coords1, list):
    #         coords1 = np.array(coords1)
    #     if isinstance(coords2, list):
    #         coords2 = np.array(coords2)
    #     n2 = coords2.shape[0]
    #     distances = np.zeros((coords1.shape[0], coords2.shape[0]))
    #     for row in range(coords1.shape[0]):
    #         coord1 = np.tile(coords1[row,:], (n2,1))
    #         distances[row,:] = np.sqrt(np.sum((coord1 - coords2)**2, axis=1))
    #     return distances
    
    # Random Coordinates
    
    def random_coord(self, unoccuppied=False, untargeted=False):
        """
        Returns a randomly selected coordinate from the board.
        If the unoccuppied parameter is True, the coordinate will chosen
        from the spaces on the ocean grid that do not contain ships.
        If the untargeted parameter is True, the coordinate will be chosen
        from the spaces on the target grid that do not have hits or misses.

        Parameters
        ----------
        unoccuppied : Bool, optional
            If True, only coordinates on the ocean grid with no ships will 
            be sampled. The default is False.
        untargeted : Bool, optional
            If True, only coordinates on the target grid that have not been
            fired at will be sampled. The default is False.

        Returns
        -------
        Tuple
            Contains the (row,col) of a randomly selected coordinate that
            is consisted with the untargeted/unoccuppied parameters.

        """
        if unoccuppied and untargeted:
            raise ValueError("Inputs unoccuppied and untargeted cannot both \
                             be true.")
        if unoccuppied:
            exclude = self.all_coords(occuppied=True)
        elif untargeted:
            exclude = self.all_targets(targeted=True)
        else:
            exclude = []
        indices = [(i,j) for i in range(self.size) 
                   for j in range(self.size) 
                   if (i,j) not in exclude]
        return indices[np.random.choice(range(len(indices)))]
    
    def random_heading(self, alignment=None):
        """
        Returns a randomly selected heading from "N", "S", "E", or "W".
        The alignment parameter may be used to restrict the possible heading
        values.

        Parameters
        ----------
        alignment : str or Align (Enum), optional
            A string or Align value that constrains the allowable headings. 
            Possible values and their effect on the random selection are:
                - None or "Any" or Align.ANY
                    Any of the four directions are possible.
                - "NS", "N/S", "N|S" or Align.VERTICAL:
                    "N" and "S" are the possible values.
                - "EW", "E/W", "E|W" or Align.HORIZONTAL:
                    "E" and "W" are the possible values.
                - "NW" or "standard":
                    "N" and "W" are the possible values. This is useful for
                    enumerating over all possible placements on a board,
                    since it will not double count equivalent placements that
                    face opposite directions but have matching coordinates.
            The default is None.

        Returns
        -------
        str : N", "S", "E", or "W"

        """
        if alignment == None:
            alignment = "any"
        if isinstance(alignment, constants.Align):
            if alignment == constants.Align.VERTICAL:
                alignment = "NS"
            elif alignment == constants.Align.HORIZONTAL:
                alignment = "EW"
            else:
                alignment = "any"
        if not isinstance(alignment, str):
            raise TypeError("alignment must be an Align object or str.")
            
        alignment = alignment.upper()
        if alignment in ["NORTH","SOUTH","EAST","WEST"]:
            alignment = alignment[0]
        for punc in ["/", "-", "|"]:
            alignment = alignment.replace(punc, "")
            
        if alignment == "ANY":
            values = ['N','S','E','W']
        elif alignment in ["STANDARD", "NW"]:
            values = ['N', 'W']
        elif alignment in ["NS", "SN", "V", "VERTICAL", "VERT"]:
            values = ['N', 'S']
        elif alignment in ["EW", "WE", "H", "HORIZONTAL", "HORIZ"]:
            values = ['E', 'W']
        else:
            raise ValueError("alignment must be 'NS', 'EW', 'any', 'NW', "
                             "or equivalent.")
        
        return(np.random.choice(values))
        
    
    def random_placement(self, length, unoccuppied=False, untargeted=False, 
                         alignment=None):
        """
        Returns a random placement dict (coordinate, heading, length).

        Parameters
        ----------
        length : int
            The length of the desired ship that will occupy the placement.
        unoccuppied : Bool, optional
            If True, the returned placement will be one that does not overlap
            any of the ships on the OCEAN GRID. The default is False.
        untargeted : Bool, optional
            If True, the returned placement will be one that does not overlap
            any of the misses on the TARGET GRID. The default is False.
        alignment : str or Align enum, optional
            A string describing the allowed alignments. The default is None.

        Returns
        -------
        None.

        """
        if unoccuppied and untargeted:
            raise ValueError("unoccuppied and untargeted cannot both be True.")
            
        if unoccuppied:
            placements = self.all_valid_ship_placements(ship_len=length,
                                                        alignment=alignment)
        elif untargeted:
            placements = self.all_valid_target_placements(ship_len=length)
        else:
            placements = self.all_valid_ship_placements(ship_len=length,
                                                        alignment=alignment)
        if len(placements) == 0:
            raise ValueError("No valid placements found for input parameters.")
        return utils.random_choice(placements)
    
    # Target Grid Methods
    
    def all_targets(self, untargeted=False, targeted=False):
        """
        Returns a list of all coordinates (row/column indices)on the target 
        grid. Returns only previously targeted or untargeted coordinates
        if either the 'targeted' or 'untargeted' parameters are set to True.
        Only one of these parameters may be True at a time.

        Parameters
        ----------
        untargeted : Bool, optional
            When True, only untargeted coordinates are returned.
            The default is False.
        targeted : Bool, optional
            When True, only previously targeted coordinates are returned.
            DESCRIPTION. The default is False. Targeted and untargeted
            cannot both be True.

        Returns
        -------
        List of tuples.
            A list of 2-element tuples, each of which corresponds to the row/
            column indices of a coordinate on the target grid that is 
            untargeted or targeted, as set by the corresponding input 
            parameters.

        """
        if untargeted and targeted:
            raise ValueError("Only one of 'untargeted' and 'targeted' may"
                             " be True.")
        exclude = []
        if untargeted:
            xr,xc = np.where(self.target_grid != constants.TargetValue.UNKNOWN)
            exclude = list(zip(xr, xc))
        elif targeted:
            r,c = np.where(self.target_grid != constants.TargetValue.UNKNOWN)
            indices = list(zip(r,c))
            return indices
        if exclude:
            indices = [(r,c) for r in range(self.size) 
                       for c in range(self.size) 
                       if (r,c) not in exclude]
        else:
            indices = [(r,c) for r in range(self.size) 
                       for c in range(self.size)]
        return indices
    
    
    def all_targets_where(self, 
                          values=None,
                          ship_type=None, 
                          hit=None, 
                          miss=None):
        """
        Returns a list of coordinates where the input parameter condition
        is true. The parameter ship_type, hit, or miss must be provided with
        a valid value (and int for ship_type, and True/False for hit & miss). 
        The other two of these parameters must be omitted.
      
        
        Parameters
        ----------
        values : list / tuple, optional
            A list of TargetValues that will have coordinates returned by 
            this method. Acceptable values are any TargetValue int (-2 to 0),
            or a ship_type in (1-5), 'hit', 'miss', 'ship'.
            Currently the list can be omitted for a single entry, but this
            usage is not futureproof.
        ship_type : ShipType, optional
            Type of ship (and int 1-5)
        hit : bool, optional
            If True, the output will include coordinates where hits occurred.
            If False, the output will contain coordinates that are not hits 
            (includes all untargeted spaces).
            The default is None.
        miss : bool, optional
            If True, the output will include coordinates where misses occurred.
            If False, the output will contain coordinates that are not misses 
            (includes all untargeted spaces).

        Returns
        -------
        targets : list
            A list of coordinate tuples that match the input parameter 
            condition.

        """
        if sum([not x for x in [ship_type==None, hit==None, miss==None]]) != 1:
            raise ValueError("Only one of ship_type, hit, miss, or values can "
                             "be passed as a parameter. The others must be "
                             "None.")
            
        if values is not None:
            targets = list(self.target_grid_is(values = values, 
                                               output = 'coords'))
        if ship_type is not None:
            targets = self.target_coords_with_type(ship_type)
        elif hit is not None:
            values = ([constants.TargetValue.HIT] 
                      + list(range(1,len(Ship.data)+1)))
            rows,cols = np.where(self.target_grid_is(values) == hit)
            targets = list(zip(rows,cols))
        elif miss is not None:
            rows,cols = np.where(self.target_grid_is(
                constants.TargetValue.MISS) == miss)
            targets = list(zip(rows,cols))
        return targets
    
    def targets_around(self, coord, diagonal=False, 
                       targeted=False, untargeted=False, values=None):
        """
        Returns a list of coordinates on the target grid that surround
        the input coordinate. See also coords_around.

        Parameters
        ----------
        coord : tuple or Coord
            A tuple that contains (row,col) indices, or a Coord instance.
        diagonal : Bool, optional
            True if the method should return the (up to) four coordinates
            that are diagonal to the input coordinate. The default is False.
        targeted : Bool, optional
            If True, only the coordinates around coord that have previously
            been targeted will be returned. The default is False.
        untargeted : Bool, optional
            If True, only the coordinates around coord that have not yet
            been targeted will be returned. The default is False.
        values : list/tuple, optional
            The returned list will only contain coordinates around the input
            coord that have target grid values equal to one of the values
            in this set. 
            
            Recognized values are:
                TargetValue.UNKNOWN (-2)
                TargetValue.MISS    (-1)
                TargetValue.HIT     (0)
                ShipType value      (1-5)
                'ship', 'hit', or 'miss'.
                
            If this parameter is not None, it overrides the 
            targeted and untargeted parameters. Default is None.
            

        Returns
        -------
        list of tuples
            A list of tuples that correspond to the coordinates that are
            immediately adjacent (and diagonally adjacent, if diagonal is True)
            to the input coordinate.

        """
        if untargeted and targeted:
            raise ValueError("untargeted and targeted inputs cannot "
                             "both be True.")
        if values and (untargeted or targeted):
            Warning("The 'values' parameter will override the "
                    "targeted/untargeted parameter.")
            
        if diagonal:
            rows = coord[0] + np.array([0, -1, -1, -1, 0, 1, 1, 1])
            cols = coord[1] + np.array([1, 1, 0, -1, -1, -1, 0, 1])
        else:
            rows = coord[0] + np.array([0, -1, 0, 1])
            cols = coord[1] + np.array([1, 0, -1, 0])
        ivalid = ((rows >= 0) * (rows < self.size) 
                  * (cols >= 0) * (cols < self.size))
        rows = rows[ivalid]
        cols = cols[ivalid]
        
        if targeted:
            values = (constants.TargetValue.MISS, constants.TargetValue.HIT)
            for t in Ship.types:
                values += (t,)
        elif untargeted:
            values = (constants.TargetValue.UNKNOWN, )
            
        if values is not None:
            coords_with_values = self.target_grid_is(values, 
                                                     exact_ship_type = False, 
                                                     output = 'coords')
            coords = [(r,c) for (r,c) in zip(rows,cols)  
                      if (r,c) in coords_with_values]
        else:
            coords = list(zip(rows, cols))
        
        return coords
        
        # if untargeted and targeted:
        #     raise ValueError("untargeted and targeted inputs cannot "
        #                      "both be True.")
        # if values:
        #     if isinstance(values, (int, np.integer)):
        #         values = (values,)
        #     rows,cols = np.where(~self.target_grid_is(values))
        #     exclude = list(zip(rows,cols))
        # elif untargeted:
        #     exclude = self.all_targets(targeted=True)
        # elif targeted:
        #     exclude = self.all_targets(untargeted=True)
        # else:
        #     exclude = []
        
        # if diagonal:
        #     rows = coord[0] + np.array([0, -1, -1, -1, 0, 1, 1, 1])
        #     cols = coord[1] + np.array([1, 1, 0, -1, -1, -1, 0, 1])
        # else:
        #     rows = coord[0] + np.array([0, -1, 0, 1])
        #     cols = coord[1] + np.array([1, 0, -1, 0])
        # ivalid = ((rows >= 0) * (rows < self.size) * 
        #           (cols >= 0) * (cols < self.size))
        # rows = rows[ivalid]
        # cols = cols[ivalid]
        # return [(rows[i], cols[i]) for i in range(len(rows)) if 
        #            (rows[i], cols[i]) not in exclude]
    
    def is_valid_target_ship_placement(self, placement, ship=None):
        """
        Returns True if the target grid allows for the input ship and placement 
        (starting coord and heading). That is, if there are only unknowns or 
        hits at the coordinates spanned by the ship's placement, and if any 
        spaces that are known to be hits on the input ship type lie in these
        coordinates.

        Parameters
        ----------
        placement : dict
            A dict with 'coord' and 'heading' keys that specify the coordinate
            and facing of a ship.
        ship : Ship, optional
            The ship instance that is being checked for consistency with the
            target grid's state. If no ship is provided, the exact ship type
            will not be checked, but the length of the input placement must 
            still be consistent with a placement on the target grid.

        Returns
        -------
        Bool
            True if the placement for a target ship lies on the target grid and
            is consistent with the known hits & misses.

        """
        if ship == None:
            ship_type = 0
        else:
            if ship.length != placement.length:
                raise ValueError("placement and ship have different lengths.")
            ship_type = ship.ship_id

        coords = placement.coords()
        on_board = self.is_valid_coord(coords)
        if np.any(on_board == False):
            return False
        rows,cols = zip(*coords)
        rows = np.array(rows)
        cols = np.array(cols)
        grid_vals = self.target_grid[rows, cols]
        if ship_type == 0:
            fits_target_grid = np.all((grid_vals == constants.TargetValue.HIT)
                                      + (grid_vals == constants.TargetValue.UNKNOWN)
                                      + (np.round(grid_vals) >= ship_type))
            known_ship_types = np.round(grid_vals[grid_vals > 0])
            if len(known_ship_types) > 0:
                fits_target_grid = (fits_target_grid and 
                                    np.all(known_ship_types 
                                           == known_ship_types[0]))
        else:
            fits_target_grid = np.all((grid_vals == constants.TargetValue.HIT)
                                      + (grid_vals == constants.TargetValue.UNKNOWN)
                                      + (np.round(grid_vals) == ship_type))
        if ship_type > 0:
            matches = self.all_targets_where(ship_type=ship_type)
            fits_target_grid = (fits_target_grid and
                                all([match in coords for match in matches]))
        return fits_target_grid
    
    def all_valid_target_placements(self, ship_type=None, ship_len=None):
        """
        Returns all of the possible placement dictionaries on the target grid
        for the input ship_type parameter. A placement is possible/valid for
        a given ship_type if it is fully on the board, does not overlap another known ship,
        does not contain any known misses, and contains ALL known instances
        of hits to that type of ship. 

        Parameters
        ----------
        ship_type : ShipType (EnumInt) or int 1-5
            The type of ship for which to determine possible placements.
        ship_len : int, optional
            The length of the ship for which to determine possible placments.
            Alternative to ship_type input (which just uses the length of
            a ship of that type to determine placement).

        Returns
        -------
        placements : list
            List of placement dictionaries that contain all possible 
            coordinates and headings for the input ship_type.

        """     
        if ship_type:
            if ship_len:
                raise ValueError("ship_len and ship_type should not both "
                                 "be provided.")
            ship_len = Ship(ship_type).length
            must_include_coords = set(self.target_coords_with_type(ship_type))
        else:
            must_include_coords = []
        if not ship_len:
            raise ValueError("Either ship_type or ship_len must be provided.")
        #must_include_coords = self.target_coords_with_type(ship_type)
        sink_coord = self.target_coords_with_type(ship_type, sink=True)
        if sink_coord:
            sink_coord = sink_coord.pop()
            
        if len(must_include_coords) > 0:
            allowed_coords = np.zeros((self.size, self.size))
            for coord in must_include_coords:
                row0 = np.max((coord[0]-ship_len+1, 0))
                row1 = np.min((coord[0]+ship_len-1, self.size-1))
                col0 = np.max((coord[1]-ship_len+1, 0))
                col1 = np.min((coord[1]+ship_len-1, self.size-1))
                allowed_coords[row0:(row1+1),coord[1]] = 1
                allowed_coords[coord[0],col0:(col1+1)] = 1      
        else:
            allowed_coords = np.ones((self.size, self.size))
        allowed_coords[self.target_grid == constants.TargetValue.MISS] = 0
        allowed_coords[(self.target_grid > 0) 
                       * (np.round(self.target_grid) != ship_type)] = 0
        rows,cols = np.where(allowed_coords)
        coords = list(zip(rows,cols))
        placements = []
        for heading in ["N","W"]:
            for coord in coords:
                p = Placement(coord, heading, ship_len)
                if self.is_valid_placement(p):
                    pcoords = p.coords()
                    pr,pc = zip(*pcoords)
                    values = self.target_grid[pr,pc]
                    # if values are all HIT, the placement cannot be valid 
                    # because one of the HITS would have to be a SINK 
                    # (i.e., a ShipType value).
                    if np.all(values == constants.TargetValue.HIT):
                        pass
                    elif sink_coord:
                        # if there is a miss/unknown in the placement, or a 
                        # ship that is not ship_type, the placement is not 
                        # valid since the ship has been sunk.
                        if np.all((values == constants.TargetValue.HIT) +
                                  (np.floor(values) == ship_type)):
                            if must_include_coords:
                                if must_include_coords.issubset(set(pcoords)):
                                    placements += [p]
                            else:
                                placements += [p]
                    elif np.all((values == constants.TargetValue.UNKNOWN) 
                              + (values == constants.TargetValue.HIT) 
                              + (np.floor(values) == ship_type)):
                        if must_include_coords:
                            if must_include_coords.issubset(set(pcoords)):
                                placements += [p]
                        else:
                            placements += [p]
        return placements
        
    def target_grid_is(self, values, exact_ship_type=False, output='array'):
        """
        Returns a boolen array with same size as target grid that is True
        anywhere the target grid matches the input value(s), and False anywhere
        it does not match the input values.

        Parameters
        ----------
        values : float or list
            The values to match the target grid against. Typically, this is one
            of the following:
                TargetValue.UNKNOWN (-2)
                TargetValue.MISS    (-1)
                TargetValue.HIT     (0)
                ShipType value      (1-5)
                'ship', 'hit', or 'miss'.
                
        exact_ship_type : Bool, optional
            If False (the default), the returned array will be True as long as
            the target grid has a ship of known type, whether it is inferred 
            or known because the ship was sunk at a particular coordinate. 
            
            If True, the returned array will be True only if the corresponding
            element in target_grid matches one of the input values exactly. 
            Set to True only in rare cases where you need to know where a 
            ship was sunk vs. where the non-sinking hits were located.
            
        output : str, optional
            Controls the output of the method:
                For output='array', the output will be a 2d numpy array of
                Boolean values. The array is True where target grid matches
                any of the values in the values input.
                
                For output='coords', the output will be a set of (row,col) 
                coordinate tuples that correspond to the row/columns of the
                target grid that match any of the input values. 

        Returns
        -------
        match_grid : numpy 2d array or set of tuples
            For output = 'array' (default):
            Boolean array with the same size as the board's target grid. 
            The value of match_grid[i,j] is True if target_grid[i,j] equals
            one of the input values, and False otherwise.
            
            For output = 'coords':
            Set of (row,col) coordinate tuples at which target_grid[row,col]
            match one of the input values.

        """
        if isinstance(values, (list,tuple,set)):
            match_grid = self.target_grid_is(values[0], exact_ship_type,
                                             output = 'array')
            for val in values[1:]:
                match_grid += self.target_grid_is(val, exact_ship_type,
                                                  output = 'array')
        elif isinstance(values, str):
            if values.lower() == 'ship':
                match_grid = (self.target_grid >= constants.TargetValue.HIT)
            elif values.lower() == 'hit':
                if exact_ship_type:
                    match_grid = (self.target_grid == constants.TargetValue.HIT)
                else:
                    match_grid = (self.target_grid >= constants.TargetValue.HIT)
            elif values.lower() == 'miss':
                match_grid = (self.target_grid == constants.TargetValue.MISS)
            elif values.lower() == 'unknown':
                match_grid = (self.target_grid == constants.TargetValue.UNKNOWN)
            else:
                raise ValueError("A string input for the 'values' parameter "
                                 "must be one of the following: " 
                                 "'ship', 'hit', 'miss', 'unknown'.")
        else:   # values is a scalar
            if not exact_ship_type:
                match_grid = (np.floor(self.target_grid) == values)
            else:
                match_grid = (self.target_grid == values)
        if output.lower() == 'coords':
            rows,cols = np.where(match_grid != 0)
            out = set(zip(rows,cols))
        elif output.lower() == 'array':
            out = (match_grid != 0)
        else:
            raise ValueError("output parameter must be 'array' (default) or "
                             "'coords'.")
        return out
        
    def target_coords_with_type(self, ship_type, sink=False):
        """
        Returns the coordinates (list of (row,col) tuples) where the 
        target grid contains the input ship type. The 'sink' input can be
        set to True if only the coordinate where the ship received its last
        point of damage (resulting in it sinking) is desired.

        Parameters
        ----------
        ship_type : ShipType enum (int 1-5)
            The type of ship to search the target grid for.
        sink : Bool, optional
            If True, only the coordinate where the input ship_type was
            sunk will be returned. If that ship has not yet been sunk,
            an empty list will be returned. The default is False.

        Returns
        -------
        coords : set
            Set of (row,col) tuples containing the coordinates on the target
            grid that are known to be occupied by a ship of ship_type.
            Since the ship type is only revealed on a hit that causes a ship
            to sink, oftentimes only one spot on the target grid (i.e., the
            hit that caused the sink) will be known to have a certain type.

        """
        if ship_type == None:
            return []
        if sink:
            rows,cols = np.where(self.target_grid == ship_type 
                                 + constants.SUNK_OFFSET)
        else:
            rows,cols = np.where((self.target_grid == ship_type 
                                  + constants.SUNK_OFFSET)
                                 + (self.target_grid == ship_type))
        return list(zip(rows,cols))
    
    # Ocean Grid Methods
    
    def all_coords(self, unoccuppied=False, occuppied=False):
        """
        Returns a list of all coordinates (row/column indices)on the ocean 
        grid. Returns only coordinates that are occuppied or unoccuppied by a
        ship if either the 'occuppied' or 'unoccuppied' parameters are set to 
        True. Only one of these parameters may be True at a time.

        Parameters
        ----------
        unoccuppied : Bool, optional
            When True, only coordinates that are not occuppied by a ship are
            returned.
            The default is False.
        occuppied : Bool, optional
            When True, only coordinates that are occuppied by a ship are 
            returned.
            The default is False. Occuppied and unoccuppied cannot both be 
            True.

        Returns
        -------
        List of tuples.
            A list of 2-element tuples, each of which corresponds to the row/
            column indices of a coordinate on the ocean grid that are 
            unoccuppied or occuppied, as set by the corresponding input 
            parameters.

        """
        if unoccuppied and occuppied:
            raise ValueError("Only one of 'unoccuppied' and 'occuppied' may"
                             " be True.")
        if unoccuppied:
            r,c = np.where(self.ocean_grid == 0)
            indices = list(zip(r,c))
        elif occuppied:
            r,c = np.where(self.ocean_grid != 0)
            indices = list(zip(r,c))
        else:
            indices = [(r,c) for r in range(self.size) 
                       for c in range(self.size)]
        return indices
    
    def coords_around(self, coord, diagonal=False, 
                     occuppied=False, unoccuppied=False):
        """
        Returns a list of coordinates on the ocean grid that surround
        the input coordinate. See also targets_around.

        Parameters
        ----------
        coord : tuple or Coord
            A tuple that contains (row,col) indices, or a Coord instance.
        diagonal : Bool, optional
            True if the method should return the (up to) four coordinates
            that are diagonal to the input coordinate. The default is False.
        occuppied : Bool, optional
            If True, only the coordinates around coord that have a ship in
            them will be returned. The default is False.
        unoccuppied : Bool, optional
            If True, only the coordinates around coord that DO NOT have a ship
            in them will be returned. The default is False.

        Returns
        -------
        list of tuples
            A list of tuples that correspond to the coordinates that are
            immediately adjacent (and diagonally adjacent, if diagonal is True)
            to the input coordinate.

        """

        if unoccuppied:
            exclude = self.all_coords(occuppied=True)
        elif occuppied:
            exclude = self.all_coords(unoccuppied=True)
        else:
            exclude = []
        
        if diagonal:
            rows = coord[0] + np.array([0, -1, -1, -1, 0, 1, 1, 1])
            cols = coord[1] + np.array([1, 1, 0, -1, -1, -1, 0, 1])
        else:
            rows = coord[0] + np.array([0, -1, 0, 1])
            cols = coord[1] + np.array([1, 0, -1, 0])
        i_valid = ((rows >= 0) * (rows < self.size) * 
                  (cols >= 0) * (cols < self.size))
        rows = rows[i_valid]
        cols = cols[i_valid]
        return [(rows[i], cols[i]) for i in range(len(rows)) if 
                   (rows[i], cols[i]) not in exclude]
    
    def is_valid_placement(self, placement, unoccuppied=False):
        """
        Returns True if a ship with the specified length placed on the ocean 
        grid at a coordinate and heading contained in placement lies entirely 
        on the board.
        If the unoccuppied parameter is set to True, the placement is only
        valid if it does not overlap another ship.

        Parameters
        ----------
        placement : dict
            A dict with 'coord', 'heading', and 'length' keys that specify the
            coordinate, length, and facing of a ship.
        unoccuppied : TYPE
            If True, the input placement and length must not intersect an
            existing ship on the ocean grid. If False, the method just checks
            whether the placement lies entirely on the board.

        Returns
        -------
        Bool
            True if the placement for a ship with specified length lies on the
            board (and, if unoccuppied == True, if the ship would not overlap
            another existing ship).

        """
        coords = placement.coords()
        if not all(self.is_valid_coord(coords)):
            return False
        if unoccuppied:
            rows,cols = zip(*coords)
            if any(self.ocean_grid[rows,cols] > 0):
                return False
        return all(self.is_valid_coord(coords))
    
    def all_valid_ship_placements(self,
                                  ship_type=None,
                                  ship_len=None,
                                  ship_buffer=0,
                                  edge_buffer=0,
                                  alignment=constants.Align.ANY,
                                  diagonal=True):
        if ship_type != None and ship_len != None:
            raise ValueError("Both ship_type and ship_len cannot be provided.")
        if ship_type is None and ship_len is None:
            raise ValueError("Either ship_type or ship_len must be provided.")
        if ship_type:
            ship_len = Ship(ship_type).length

        if diagonal:
            dist_method = "manhattan"
        else:
            dist_method = "dist"
        rows = range(edge_buffer, self.size-edge_buffer)
        coords = [(r,c) for r in rows for c in rows]
        placements = []
        ship_placements = [p for p in self.ship_placements.values()]
        
        if utils.parse_alignment(alignment) in [constants.Align.ANY, 
                                                constants.Align.VERTICAL]:
            placements += [Placement(c, "N", length=ship_len) for c in coords
                           if c[0] + ship_len - 1 < self.size - edge_buffer]    
        if utils.parse_alignment(alignment) in [constants.Align.ANY, 
                                                constants.Align.HORIZONTAL]:
            placements += [Placement(c, "W", length=ship_len) for c in coords
                           if c[1] + ship_len - 1 < self.size - edge_buffer]
        if utils.parse_alignment(alignment) not in constants.Align:
            raise ValueError("Alignment must be an Align object or appropriate "
                             "string (N/S, NS, Vertical, etc.")
        valid_placements = []
        for place in placements:
            if self.is_valid_placement(place, unoccuppied=True):
                if ship_buffer > 0:
                    buffers = place.dists_to_placements(ship_placements, 
                                                        method=dist_method,
                                                        output_as_buffer=True)
                    if len(buffers) == 0:
                        valid_placements += [place]     # no ships on board
                    elif min([b for b in buffers]) >= ship_buffer:
                        valid_placements += [place]
                else:
                    valid_placements += [place]
            else:
                pass
        return valid_placements
    
    
    # Ship Methods
    
    def ship_for_type(self, ship_type):
        """
        Returns the Ship instance on the board that corresponds to the 
        input ship_type.

        Parameters
        ----------
        ship_type : int, ShipType, or string.
            An integer or string that describes the desired ship. If an 
            integer or ShipType enum, ship_type should be 1-5. If a
            string, it should be 'patrol', 'sub(marine)', 'destroyer',
            'battleship' or 'carrier'.

        Returns
        -------
        Ship
            The instance of Ship on the board that corresponds to the 
            identifier ship_type. If no 

        """
        if isinstance(ship_type, (int, np.integer)):
            if ship_type not in Ship.data:
                raise ValueError(f"Integer {ship_type} does not correspond "
                                 f" to a valid ship type.")
        elif isinstance(ship_type, str):
            ship_type = Ship.type_for_name(ship_type)
        return self.fleet[ship_type]
    
    def ship_at_coord(self, coord):
        
        """
        Get the ship at the input coordinate.

        Parameters
        ----------
        coord : Tuple containing (row,col) or Coord
            A coordinate on the board.

        Returns
        -------
        Ship or None
            If the input coordinate is occuppied by a ship, the ship instance.
            Otherwise, if coordinate is unoccuppied, None.

        """
        if not utils.is_on_board(coord, self.size):
            raise ValueError("coord is not on the board.")
        # indexing coord allows for both tuple and Coord input.
        val = self.ocean_grid[(coord[0], coord[1])]
        if val == 0:
            return None
        else:
            return self.fleet[val]
        
    def damage_at_coord(self, coord):
        """
        Returns the damage at the slot corresponding to the input coordinate.
        If no ship is present or if the ship is not damaged at that particular
        slot, returns 0. Otherwise, if that spot has been hit, returns 1.

        Parameters
        ----------
        coord : tuple
            A tuple containing a (row,col) index.

        Returns
        -------
        int
            The number of times the ship at the input coordinate has been hit.
            0 if no ship is present at the coordinate, or if there is a ship
            but it has not been damaged yet.

        """
        ship = self.ship_at_coord(coord)
        if not ship:
            return 0
        coords = self.coords_for_ship(ship)
        return ship.damage[coords.index(coord)]
    
    def is_fleet_afloat(self):
        """
        Returns an array of Bools, where element i is True if the ship with
        ShipType i+1 is afloat, and False if it is sunk.

        Returns
        -------
        numpy.ndarray
            A 5-element boolean array.
        """
        keys = sorted(self.fleet.keys())
        return [self.fleet[k].is_afloat() for k in keys]
            
    def coords_for_ship(self, ship):
        """
        Returns a list of the row/col coordinates occuppied by the input Ship 
        instance.

        Parameters
        ----------
        ship : Ship
            The ship instance to be coordinated.

        Returns
        -------
        list
            List of coordinate tuples (row,col).

        """
        ship_type = ship.ship_id
        if ship_type not in self.ocean_grid:
            raise ValueError("ship type not found on ocean grid.")
        r,c = np.where(self.ocean_grid == ship_type)
        if len(r) != Ship.data[ship_type]["length"]:
            raise ValueError("The number of spaces on the ocean grid occuppied"
                             " by the ship do not match the ship's length.")
        return list(zip(r,c))
    
    def placement_for_ship(self, ship):
        """
        Returns a tuple with the coordinate and heading of the input ship
        or ShipType value.

        Parameters
        ----------
        ship : Ship
            The ship to locate.

        Returns
        -------
        dict
            Dictionary containing values for 'coord' and 'heading' keys.
            dict['coord'] is a tuple containing the first (row,col) of the 
            placement. 
            dict['heading'] is a direction character ("N"/"S"/"E"/"W") 
            indicating which way the ship is facing.

        """
        rows,cols = zip(*self.coords_for_ship(ship))
        if rows[0] == rows[-1]:
            # horizontal
            heading = "W"
            coord = (rows[0], min(cols))
        elif cols[0] == cols[-1]:
            # vertical
            heading = "N"
            coord = (min(rows), cols[0])
        else:
            raise Exception(f"Invalid coordinates for ship {ship}.")
        return Placement(coord, heading, ship.length)
    
    def placements_containing_coord(self, 
                                    coord, 
                                    ship_types=None):
        """
        Returns a list of all possible placmeents that include the input
        coord for the input ship types.

        Parameters
        ----------
        coord : tuple or Coord
            A single coordinate tuple (or Coord object). The returned list
            will contain all valid placements that include this coordinate.
        ship_types : list
            The list of ships to consider for the desired placement list.

        Returns
        -------
        placements : list
            A list of all valid placements that contain coord.

        """
        if not ship_types:
            ship_types = Ship.types
        
        placements = []
        for ship_type in ship_types:
            ship_len = Ship.data[ship_type]["length"]
            for shift in range(ship_len):
                p1 = Placement((coord[0] - shift, coord[1]), "N", ship_len)
                p2 = Placement((coord[0], coord[1] - shift), "W", ship_len)
                if self.is_valid_placement(p1):
                    placements += [p1]
                if self.is_valid_placement(p2):
                    placements += [p2]
        return set(placements)
    
    def placements_containing_all_coords(self, 
                                         coords,
                                         ship_types = None):
        """
        Returns a list of all of the placements that span the input 
        coordinates. Coordinates may be input as 

        Parameters
        ----------
        coords : tuple or list
            A tuple or list of two-element row/col tuples
        ship_types : list, optional
            A list of ship types to consider when determining if a placement
            should be included in the returned list. 
            The default is None, which implies that all ship types should be
            considered.

        
        Returns
        -------
        valid_placements : list
            A list of all of the valid placements that include all of the
            input coordinates.

        """

        placements = {}
        for coord in coords:
            placements[coord] = self.placements_containing_coord(coord, 
                                                                 ship_types)
        valid_placements = placements[coords[0]]
        for coord in coords[1:]:
            valid_placements = valid_placements.intersection(placements[coord])
        return valid_placements 
    
    # Fleet Placement Methods
    
    def add_fleet(self, placements):
        """
        Places the ships with locations described by the placements dict onto
        the ocean grid.

        Parameters
        ----------
        placements : dict
            A dictionary with keys equal to the ShipTypes (int/EnumInt 1-5) 
            that are to be placed on the board. The value for each ShipType
            key should be a placement object, with format:
                Placement(coord, heading, length)
                    'coord' is a tuple containing the (row,col) of the
                        first coordinate of the ship.
                    'heading' is an uppercase character specifying the cardinal
                        direction ("N"/"S"/"E"/"W") that the ship should point.
                    
            An example placements dictionary would be as follows:
                placements = {
                    ShipType(1): Placement((r1,c1), h1, 2),
                    ShipType(2): Placement((r2,c2), h2, 3),
                    ShipType(3): Placement((r3,c3), h3, 3),
                    ShipType(4): Placement((r4,c4), h4, 4),
                    ShipType(5): Placement((r5,c5), h5, 5)}
                }
            where rX and cX are integers between 0 and 9 (for a 10 x 10 board)
            and hX is a direction-specifying character "N", "S", "E", or "W".
                

        Returns
        -------
        None.

        """
        for (k,placement) in placements.items():
            self.add_ship(k, placement.coord, placement.heading)
    
    def add_ship(self, ship_type, coord=None, heading=None, placement=None):
        """
        Add a ship to the board of the specified type at the input coordinate
        and heading.

        Parameters
        ----------
        ship_type : ShipType (EnumInt) or int 1-5.
            A ShipType value indicating the type of ship to place on the board.
        coord : Tuple continaing (row,col)
            The row and column where the front of the ship should be placed.
        heading : chr
            A character indicating the direction the ship should face. Valid
            values are "N", "S", "E", and "W". The ship will be positioned
            such that its front is at the input coord, and its front will 
            face toward heading. All other slots on the ship will be located
            at coordinates behind coord, when facing toward this heading.
        placement : Placement, optional
            A placement object, which can be used instead of coord/heading
            values to specify ship location and orientation. If placement is
            provided, coord and heading should be omitted.

        Raises
        ------
        Exception
            Raised if the input ship_type already exists on the board or if
            the placement would cause the ship to hang off the edge of the
            board.

        Returns
        -------
             None.

        """
        if isinstance(ship_type, str):
            ship_type = Ship.type_for_name(ship_type)
        ship = Ship(ship_type)
        if coord or heading:
            if placement:
                raise ValueError("If placement is provided, coord/heading "
                                 "should not be.")
            placement = Placement(coord, heading, ship.length)
        if ship.length != placement.length:
            raise ValueError("Placement length does not match ship length.")
        if not self.is_valid_placement(placement, 
                                       unoccuppied=True):
            raise Exception(f"Cannot place {ship.name} at {placement.coord}.")
        else:
            self.fleet[ship.ship_id] = ship
            self.ship_placements[ship.ship_id] = placement
            drows,dcols = self.rel_coords_for_heading(placement.heading, 
                                                      ship.length)
            self.ocean_grid[placement.coord[0] + drows, 
                            placement.coord[1] + dcols] = ship.ship_id
        
    def is_ready_to_play(self):
        """
        Returns True if the board is ready to play a new game. This is defined
        as: 
            1. All 5 ships have been placed on the board.
            2. None of the ships are damage.
            3. No shots have been fired (i.e., both the ocean and target grids 
                are empty).
        
        Returns
        -------
        bool
            True if the board has all ship.

        """
        if not np.all(self.target_grid == constants.TargetValue.UNKNOWN):
            return False
        if not (np.sum(self.ocean_grid > 0) == 
                sum([ship.length for ship in self.fleet.values()])):
            return False
        for ship in self.fleet.values():
            if np.any(ship.damage > 0):
                return False
        ship_ids = [ship.ship_id for ship in self.fleet.values()]
        if sorted(ship_ids) != list(range(1, len(Ship.data)+1)):
            return False
        return True
    
    # Gameplay Methods
                                  
    def incoming_at_coord(self, coord):
        """
        Determines the outcome of an opponent's shot landing at the input
        coord. If there is a ship at that location, the outcome is a hit and
        the ship's slot corresponding to the input space is damaged. 
        If the hit means that all slots in a ship are damaged, the ship is sunk.
        If there is no ship at the location, the outcome is a miss.

        Parameters
        ----------
        coord : tuple
            A two-element tuple specifying the (row,column) of the targeted
            coordinate.

        Returns
        -------
        outcome : dict
            A dictionary containing information about the results of the
            shot at coord. It will contain the following key/value pairs:
                "hit": Bool
                "coord": equal to the coord parameter
                "sunk": Bool
                "sunk_ship_type": ShipType enum
                "message": string

        """
        if isinstance(coord, Coord):
            coord = coord.rowcol
        ship = self.ship_at_coord(coord)
        msg = []
        if not ship:
            outcome = {"hit": False, "coord": coord, 
                       "sunk": False, "sunk_ship_type": None, "message": []}
        else:
            outcome = {"hit": True, "coord": coord, "sunk": False, 
                       "sunk_ship_type": None, "message": []}
            sunk, damage = self.damage_ship_at_coord(ship, coord)
            if damage > 1:
                msg += [f"Repeat damage: {coord}."]
            if sunk:
                msg += [f"{ship.name} sunk."]
                outcome["sunk"] = True
                outcome["sunk_ship_type"] = ship.ship_id
        if msg:
            [a.strip() for a in msg]
        outcome['message'] = msg
        return outcome
    
    def damage_ship_at_coord(self, ship, coord):
        """
        Adds a point of damage to the ship at the input coord. 
        Raises an exception if the coordinate does not lie on the ship in 
        question.

        Parameters
        ----------
        ship : Ship
            The ship that will be damaged.
        coord : tuple
            Two-element tuple with row/col on the board where a hit occurred.

        Raises
        ------
        ValueError
            Raised if ship is not in the board's fleet.
        Exception
            Raised if ship does not sit on coord.

        Returns
        -------
        tuple : (Bool, int)
            The bool indicates whether ship has been sunk, and the int gives
            the number of times coord has been hit (usually 1).

        """
        
        if ship not in self.fleet.values():
            raise ValueError("Input ship instance must point to one of "
                             "the object in this Board's fleet.")
        ship_coords = self.coords_for_ship(ship)
        if coord not in ship_coords:
            raise ValueError("Input ship is not located at input coord.")
        dmg_slot = [i for i in range(len(ship_coords)) 
                    if ship_coords[i] == coord]
        if len(dmg_slot) != 1:
            raise ValueError(f"Could not determine unique damage location for "
                             f"target {coord}")
        damage = ship.hit(dmg_slot[0])
        return ship.is_sunk(), damage
        
    def update_target_grid(self, outcome):
        """
        Updates the targets grid with the results of the outcome dictionary.

        Parameters
        ----------
        outcome : dict
            Dictionary that contains the following key/value pairs:
                
            "coord" : Tuple with the (row,col) where the shot was fired.
            "hit" : Bool - True if a ship was hit, False if shot was a miss.
            "sunk" : Bool - True if a ship was sunk by the outcome's shot.
            "sunk_ship_type" : int corresponding to the ShipType of a ship if
                a ship was sunk by the outcome's shot..

        Returns
        -------
        None.

        """
        coord = outcome["coord"]
        ship_type = outcome["sunk_ship_type"]
        if isinstance(coord, Coord):
            coord = coord.rowcol
        prev_val = self.target_grid[coord]
        if outcome["hit"]:
            if ship_type:
                self.target_grid[coord] = ship_type + constants.SUNK_OFFSET
            else:
                self.target_grid[coord] = constants.TargetValue.HIT
        else:
            self.target_grid[coord] = constants.TargetValue.MISS
        if prev_val != constants.TargetValue.UNKNOWN:
            print(f"Repeat target: {Coord(coord).lbl}")
        
        # Update target_grid if any ship placements can now be unambiguously
        # determined:
        out_of_sync = True
        while out_of_sync:
            out_of_sync = False
            for tp in Ship.types:
                placements = self.all_valid_target_placements(tp)
                if len(placements) == 1:
                    coords = placements[0].coords()
                    for coord in coords:
                        if (self.target_grid[coord[0], coord[1]] 
                                == constants.TargetValue.HIT):
                            # if any updates are made, repeat the search
                            # process to see if any other ships can be 
                            # identified.
                            out_of_sync = True
                            self.target_grid[coord[0], coord[1]] = tp
                            
    
    
    # Visualization functions
    
    def ocean_grid_image(self):
        """
        Returns a matrix corresponding to the board's ocean grid, with
        blank spaces as 0, unhit ship slots as 1, and hit slots as 2.

        Returns
        -------
        im : 2d numpy array
        
        """
        im = np.zeros((self.size, self.size))
        for ship_id in self.fleet:
            ship = self.fleet[ship_id]
            dmg = ship.damage
            rows,cols = zip(*self.coords_for_ship(ship))
            im[rows,cols] = dmg + 1
        return im
    

    def target_grid_image(self):
        """
        Returns a matrix corresponding to the board's vertical grid (where 
        the player keeps track of their own shots). 
        The matrix has a -2 for no shot, -1 for miss (white peg), 0 for a hit, 
        and shipId (1-5) if the spot is a hit on a known ship type.

        Returns
        -------
        im : 2d numpy array

        """
        return self.target_grid
        
    
    def ship_rects(self):
        """
        Returns a dictionary with keys equal to ShipId and values equal to
        the rectangles that bound each ship on the grid. The rectangles have
        format [x, y, width, height].

        Returns
        -------
        rects : dict
            Dictionary with the following keys/values:
                rects[ship_type] = numpy array with (x,y,width,height) for
                                    each ship_type.

        """
        rects = {}
        for ship in list(self.fleet.values()):
            rows, cols = zip(*self.coords_for_ship(ship))
            rects[ship.ship_id] = np.array((np.min(cols)+0.5, 
                                            np.min(rows)+0.5,
                                            np.max(cols) - np.min(cols) + 1, 
                                            np.max(rows) - np.min(rows) + 1))
        return rects
        
    
    def color_str(self, grid='both'):
        """
        Returns a string in color that represents the state of the board.
        to a human player.

        Returns
        -------
        s : str
            A string with background and foreground text color codes encoded
            to display a color text representation of the board.

        """  
        def format_target_id(x):
            s = f"{str(round(x))}"
            if x - round(x) == constants.SUNK_OFFSET:
                s += "*"
            return s
        
        if grid.lower() == 'both':
            grid = ('target', 'ocean')
        else:
            grid = (grid.lower(), )
            
        ocean = "\x1b[1;44;31m "
        ship_hit = "\x1b[1;41;37m"
        ship_no_hit = "\x1b[2;47;30m"
        red_peg = "\x1b[1;44;31mX"
        red_peg_sunk = "\x1b[1;44;31m"
        white_peg = "\x1b[1;44;37m0"
        
        s = "\n  "
        s += (" ".join([str(x) for x in np.arange(self.size) + 1]) 
              + "\n")
        if 'target' in grid:
            for (r,row) in enumerate(self.target_grid):
                s += (Coord(r,0).lbl[0]
                      + " "
                      + " ".join([ocean if x == constants.TargetValue.UNKNOWN else 
                                  white_peg if x == constants.TargetValue.MISS else
                                  red_peg if x == constants.TargetValue.HIT else 
                                  (red_peg_sunk + format_target_id(x)) 
                                  for x in row]) 
                      + "\033[0;0m\n")
            s += ("\n  " 
                  + " ".join([str(x) for x in np.arange(self.size) + 1]) 
                  + "\n")
            
        if 'ocean' in grid:
            for (r,row) in enumerate(self.ocean_grid):
                s += (Coord(r,0).lbl[0]
                      + " "
                      + " ".join([ocean if x == 0 else 
                                  (ship_hit + str(x)) if 
                                  self.damage_at_coord((r,c)) > 0 else
                                  (ship_no_hit + str(x)) 
                                  for (c,x) in enumerate(row)]) 
                      + "\033[0;0m\n")
        return s
    
    
    def show(self, grid="both"):
        """
        Shows images of the target map and grid on a figure with two 
        subaxes. Colors the images according to hit/miss (for target map) and
        damage (for the grid). Returns the figure containing the images.

        Parameters
        ----------
        grid : str, optional
            The grid to display; "ocean", "target", or "both". 
            The default is "both".

        Raises
        ------
        ValueError
            Raised if the grid paramater is not "ocean", "liquid" or "both".

        Returns
        -------
        fig : pyplot figure
            A figure containing axes with the image of the board's grid(s) 
            that have been selected using the grid parameter.

        """
        if grid.lower() == "both":
            grid = ["target", "ocean"]
        else:
            grid = [grid.lower()]
        if (grid[0] not in ["target", "ocean"] or 
                grid[-1] not in ["target", "ocean"]):
            raise ValueError("grid must be 'target', 'ocean', "
                             "or 'both' (default).")
            
        label_color = "lightgray"
        grid_color = "gray"
        axis_color = "lightgray"
        bg_color = "#202020"
        ship_outline_color = "#202020"
        peg_outline_color = "#202020"
        ship_color = "darkgray"

        ocean_color = 'tab:blue' #cadetblue
        miss_color = "whitesmoke"
        hit_color = "tab:red" #"firebrick"

        peg_radius = 0.3
        
        cmap_flat = mpl.colors.ListedColormap([ocean_color, ship_color, ship_color])
        cmap_vert = mpl.colors.ListedColormap([ocean_color, miss_color, hit_color])

        if len(grid) == 2:
            fig, axs = plt.subplots(2, 1, figsize = (10,6))
        else:
            fig, axs = plt.subplots(1, 1, figsize = (6,6))
            axs = np.array([axs])

        flat_grid = self.ocean_grid_image()
        vert_grid = self.target_grid_image()
        
        # grid lines
        grid_extent = [1-0.5, self.size+0.5, self.size+0.5, 1-0.5]
        for ax in axs:
            ax.set_xticks(np.arange(1,11), minor=False)
            ax.set_xticks(np.arange(0.5,11), minor=True)        
            ax.set_yticks(np.arange(1,11), minor=False)            
            ax.set_yticks(np.arange(0.5,11), minor=True)
            for spine in ax.spines.values():
                spine.set_color(axis_color)
            ax.xaxis.grid(False, which='major')
            ax.xaxis.grid(True, which='minor', color=grid_color, 
                          linestyle=':', linewidth=0.5, )
            ax.yaxis.grid(False, which='major')
            ax.yaxis.grid(True, which='minor', color=grid_color, 
                          linestyle=':', linewidth=0.5)
            ax.set_xticklabels([""]*(self.size+1), minor=True)
            ax.set_xticklabels([str(n) for n in range(1,self.size+1)], 
                               minor=False, color=label_color)            
            ax.set_yticklabels([""]*(self.size+1), minor=True)
            ax.set_yticklabels([chr(65+i) for i in range(self.size)], 
                               minor=False, color=label_color) 
            ax.xaxis.tick_top()
            ax.tick_params(color = 'lightgray')
            ax.tick_params(which='minor', length=0)

        if "target" in grid:
            axs[0].imshow(np.zeros(vert_grid.shape), cmap=cmap_vert, 
                                 extent=grid_extent)
            # add pegs to target grid
            rows,cols = np.where(vert_grid == constants.TargetValue.MISS)
            white_pegs = [plt.Circle((x,y), radius=peg_radius, 
                                     facecolor = miss_color, 
                                     edgecolor = peg_outline_color) 
                        for (x,y) in zip(cols+1,rows+1)]
            for peg in white_pegs:
                axs[0].add_patch(peg) 
            rows,cols = np.where(vert_grid >= constants.TargetValue.HIT)
            red_pegs = [plt.Circle((x,y), radius=peg_radius, 
                                   facecolor = hit_color, 
                                   edgecolor = peg_outline_color) 
                        for (x,y) in zip(cols+1,rows+1)]
            for peg in red_pegs:
                axs[0].add_patch(peg) 
                
        if "ocean" in grid:
            axs[-1].imshow(np.zeros(flat_grid.shape), cmap=cmap_flat, 
                           extent=grid_extent)
            # add pegs to ships
            rows,cols = np.where(flat_grid == 2)
            red_pegs = [plt.Circle((x,y), radius=peg_radius, 
                                   facecolor = hit_color, 
                                   edgecolor = peg_outline_color) 
                        for (x,y) in zip(cols+1,rows+1)]
            for peg in red_pegs:
                axs[-1].add_patch(peg)
            # ships
            ship_boxes = self.ship_rects()
            for ship_type in ship_boxes:
                box = ship_boxes[ship_type]
                axs[-1].add_patch( 
                    mpl.patches.Rectangle((box[0], box[1]), box[2], box[3], \
                                          edgecolor = ship_outline_color,
                                          fill = False,                                       
                                          linewidth = 2))
                axs[-1].text(box[0]+0.12, box[1]+0.65, 
                             Ship.data[ship_type]["name"][0])

        fig.set_facecolor(bg_color)
        return fig
    
    
    def show2(self, grid="both"):
        """
        Shows images of the target map and grid on a figure with two 
        subaxes. Colors the images according to hit/miss (for target map) and
        damage (for the grid). Returns the figure containing the images.

        Parameters
        ----------
        grid : str, optional
            The grid to display; "ocean", "target", or "both". 
            The default is "both".

        Raises
        ------
        ValueError
            Raised if the grid paramater is not "ocean", "liquid" or "both".

        Returns
        -------
        fig : pyplot figure
            A figure containing axes with the image of the board's grid(s) 
            that have been selected using the grid parameter.

        """
        return Viz.show_board(self, grid)
    
    
    def status_str(self):
        """
        Returns a string describing the status of the board, including number
        of shot fired, hits, sinks both for and against the board.

        Returns
        -------
        s : str

        """
        tg = self.target_grid.flatten()
        shots_fired = sum(tg != constants.TargetValue.UNKNOWN)
        hits = sum(tg >= constants.TargetValue.HIT)
        misses = sum(tg == constants.TargetValue.MISS)
        
        dmg_by_ship = [sum(s.damage) for s in self.fleet.values()]
        
        s = (f"Shots fired: {shots_fired} ({hits} hits, {misses} misses)\n"
             f"Ships:       {sum([d for d in dmg_by_ship])} hits "
             f"on {sum(d>0 for d in dmg_by_ship)} ships, "
             f"{sum(self.is_fleet_afloat()==False)} ships sunk")
        return s
    
    # Targeting Methods
    
    def possible_ship_types_at_coord(self, coord, output="dict"):
        """
        Determines which ships are possible at the input coord, and how many
        possible placements (for each ship type) would result in a ship at 
        that coord.

        Parameters
        ----------
        coord : tuple 
            The (row, col) tuple for the coordinate of interest.
        output : str (optional)
            If output is "dict", the return value is a dictionary that
            contains information about the number of ways a ship may be placed
            in order to occupy the input coordinate.
            If output is "list", the return value is a list that contains the
            ShipType integers for the ships that are possible at coordinate, 
            without any information about the number of ways each ship may
            be placed.
            The default is "dict".
            
        Returns
        -------
        out : dictionary or list
            If output is "dict":
                out[n] where n is a ShipType (integer 1-5) is equal
                to the number of ways a ship of type n could be placed on the
                board and occupy the input coordinate.
            If output is "list":
                out is a list that contains the ShipType values for each 
                ship that may potentially occupy the input coordinate. If coord
                is at a miss site, out will be empty. If it is at a site with 
                a known ship type, it will have length 1. Otherwise it could 
                be up to 5 elements long.

        """
        counts = {}
        ship_types = [constants.ShipType(k) for k in 
                      range(min(Ship.data), max(Ship.data)+1)]
        for tp in ship_types:
            counts[tp] = 0
        for tp in ship_types:
            placements = self.all_valid_target_placements(tp)
            for placement in placements:
                place_coords = placement.coords()
                counts[tp] += int(coord in place_coords)
        if output == "dict":
            return counts
        elif output == "list":
            return [k for k in ship_types if counts[k] > 0]
        
        
    def possible_targets_grid(self, by_type=False):
        """
        Determines the possible ships at each space on the board, and the
        number of ways each ship could be placed in order to occupy that 
        coordinte. In effect, the value at each point is the number of ways
        ANY ship could be positioned and occupy that point. It is therefore
        something like a target probability density.

        Parameters
        ----------
        by_type : Bool, optional
            If False, the method returns a 2d array, where element [i,j] is
            the number of ways any ship can be positioned and overlap 
            row i and column j.
            If by_type is True, the output grid_by_ship[n] will return
            a 2d N x N array (N is board size) where element [i,j] is the 
            number of ways a ship of type n can be positioned and overlap 
            row i and column j.
            The default is False.

        Returns
        -------
        grid_by_ship : numpy array (2d or 3d)
            The number of possible placements at each row/col that will allow
            a ship to overlap that row/col. See by_type for details.

        """
        grid_by_ship = {}
        ship_types = [constants.ShipType(k) for k in range(min(Ship.data), 
                                                           max(Ship.data)+1)]
        for k in ship_types:
            grid_by_ship[k] = None
        for tp in ship_types:
            grid = np.zeros(self.target_grid.shape)
            placements = self.all_valid_target_placements(tp)
            for p in placements:
                for coord in p.coords():
                    grid[coord] += 1
            grid_by_ship[tp] = grid
        if by_type == False:
            out = np.zeros(self.target_grid.shape)
            for k in grid_by_ship:
                out += grid_by_ship[k]
            return out
        return grid_by_ship
    
    
    # More targeting methods (in development)
    
    def find_hits(self, unresolved=False):
        """
        Returns the set of all coordinates on the target grid that have a hit 

        Parameters
        ----------
        unresolved : Bool
            If True, hit spaces that can be assigned to a definite ship type
            will not be returned; only those hits that are on an unknown
            type of ship will be returned. Since ship type is revealed when
            a ship is sunk (and adjacent hits can then often--though not 
            always--be logically assigned to the same type of ship), setting
            unresolved to True will usually get hits that still have adjacent
            spaces with ship slots that have not yet been targeted.
        
        Returns
        -------
        hits : set
            All coordinates with a hit on the target grid.

        """
        if unresolved:
            rows,cols = np.where(self.target_grid 
                                 == constants.TargetValue.HIT)
        else:
            rows,cols = np.where(np.round(self.target_grid) 
                                 >= constants.TargetValue.HIT)
        return set(zip(rows,cols))
    
    
    # def colinear_targets_along(self, hit):
    #     """
    #     Finds the targets (unknown spaces) at the end of horizontal and 
    #     vertical line of hits starting at the input hit.

    #     Parameters
    #     ----------
    #     hit : tuple (row,col)
    #         A hit that lies in a suspected line of 2 or more hits.

    #     Returns
    #     -------
    #     targets : set
    #         The set of all targets (of length 0 to 4) that lie at the end of
    #         any horizontal and vertical lines of hits that extend from the
    #         input hit.

    #     """
    #     hit_grid = self.target_grid_is('hit')
    #     horizontal_labels = label_array(hit_grid[hit[0],:])
    #     vertical_labels = label_array(hit_grid[:,hit[1]])
    #     hlbl = horizontal_labels[hit[1]]
    #     vlbl = vertical_labels[hit[0]]
    #     hidx = np.flatnonzero(horizontal_labels == hlbl)
    #     vidx = np.flatnonzero(vertical_labels == vlbl)
    #     if len(hidx) == 1 and len(vidx) == 1:
    #         targets = [(hit[0]-1,hit[1]), (hit[0]+1,hit[1]),
    #                    (hit[0],hit[1]-1), (hit[0],hit[1]+1)]
    #     else:            
    #         if len(hidx) > 1:
    #             targets = [(hit[0],hit[1]-1), (hit[0],hit[1]+1)]
    #         elif len(hidx) > 1:
    #             targets = [(hit[0]-1,hit[1]), (hit[0]+1,hit[1])]
    #         else:
    #             # len(hidx) and len(vidx) are both 0.
    #             raise Exception("hit was not properly labeled by label_array.")
    #     targets = [t for t in targets if 
    #                (self.is_valid_coord([t]) and 
    #                 self.target_grid[t[0],t[1]] == TargetValue.UNKNOWN)]
    #     return targets
    
    
    # def targets_around_all_hits(self, unresolved=False, prefer_colinear=False):
    #     """
    #     Returns a set of all untargeted coordinates around spaces with hits.

    #     Parameters
    #     ----------
    #     unresolved : Bool, optional
    #         If True, hit spaces that can be assigned to a definite ship type
    #         will not be included when searching for viable target spaces.
    #         Only those untargeted spaces around hits that are on an unknown 
    #         type of ship will be returned. The default is False.

    #     prefer_colinear : Bool, optional
    #         If True, when a hit has any other hits adjacent to it, only
    #         untargeted spaces that are in-line with two or more hits will be
    #         returned.
            
    #         For example, in the following scenario, let "-" mean untargeted,
    #             "X" mean hit, and "O" mean miss. 
                
    #                 1 2 3 4 5 
    #               A - - - - - 
    #               B - X X X - 
    #               C - - - - - 
            
    #         Since all of the hits (X's) at B2 through B4 have at least one 
    #         adjacent hit, a target space will only be returned if it forms a 
    #         line with an adjacent hit and one of that hit's neighbors. 
    #         B1 will be included because it lines up with the hits at B2 and B3,
    #         but A2 will not be because the hit at B2 forms a line with B3,
    #         but not with C2 (at least, not a known line). Therefore
    #         targets_around_hits(prefer_colinear=True) will return B1 and B5
    #         for this board.
                
    #         If prefer_colinear is False, then every "-" space surrounding each
    #         of the hits ("X") will be returned: A2, A3, A4, B1, B5, C2, C3, C5.
            
    #         If a hit forms multiple lines, as in the following layout, then 
    #         spaces that line-up with either line are valid:
                
    #                 1 2 3 4  
    #               A - - - - 
    #               B - X X - 
    #               C - X - - 
    #               D - - - - 
            
    #         In this case, A2, B1, B4, and D2 would be returned.
              
    #     Returns
    #     -------
    #     targets : set
    #         The set of target coordinates that are adjacent to all of the
    #         hits on the board and have not yet been targeted.

    #     """
    #     hits = self.find_hits(unresolved=unresolved)
    #     targets = set()
    #     for hit in hits:
    #         for coord in self.targets_around(hit, untargeted=True):
    #             if prefer_colinear:
    #                 if self.is_colinear_with_hits(coord, hit):
    #                     targets.add(coord)
    #                 else:
    #                     pass
    #             else:
    #                 targets.add(coord)
    #     return targets
        
    
    # def is_colinear_with_hits(self, target, hit):
    #     """
    #     Determines if the input target coordinate is colinear with the input
    #     hit and and other hits that are adjacent to that hit.
        
    #     Colinear in this case means that if the input hit has one or more
    #     adjacent spaces with a hit, then the target is considered colinear
    #     if it falls in the same column/row as the column/row of the hit and
    #     adjacent hit, and is also adjacent to the input hit.
        
    #     As an example, if ? is the target space, X is the input hit, and 
    #     x are other hits (- is unknown), then ? is colinear in the scenario
    #     on the left, but not on the right.
        
    #         COLINEAR        NOT COLINEAR
    #         - ? X x -       - ? X - 
    #         - - - - -       - - x - 
            
    #     If the input hit has no adjacent hits, then the target coordinate is
    #     always considered colinear (i.e., it forms a line with the hit of 
    #     length 2).

    #     Parameters
    #     ----------
    #     target : tuple (row,col)
    #         The target coordinate. Method returns True if this coordinate is
    #         colinear with the input hit and any adjacent hits.
    #     hit : tuple (row,col)
    #         The coordinates of a hit that is adjacent to target.

    #     Returns
    #     -------
    #     colinear : Bool
    #         True if target is in-line with hit and at least one of hit's 
    #         adjacent hits, or if hit has no adjacent hits.
    #         False if target does not form a line with hit and any of hit's 
    #         adjacent hits.

    #     """
    #     adjacents = self.coords_around(hit)
    #     if target not in adjacents:
    #         raise ValueError("Target coordinate is not adjacent to hit.")
    #     adjacent_hits = [a for a in adjacents 
    #                      if self.target_grid[a[0],a[1]] >= TargetValue.HIT]
    #     ds = np.array(adjacent_hits) - np.tile(np.array(hit), 
    #                                            (len(adjacent_hits),1))
    #     dtarget = np.array((target[0] - hit[0], target[1] - hit[1]))
    #     colinear = np.any((ds == -dtarget).all(axis=1))
    #     return colinear
