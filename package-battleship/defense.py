#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 23:21:42 2022

@author: jason
"""

#%% Imports

import abc
import numpy as np

from battleship import (SHIP_DATA, MAX_ITER, Align, ShipType,
                        Ship, Board, Coord, random_choice)
    
#%% Constants
_RANDOMIZE_SHIP_PLACEMENT_ORDER = True

#%% Defensive Strategy
class Defense(metaclass=abc.ABCMeta):    
    """
    All Defense instances must implement the following attributes and methods:
    
    Abstract Properties/Methods:
    - fleet_placements
    
    Properties
    - fleet_alignment
    - edges
    
    Other methods
    - __init__
    
    Class methods
    - from_strs
    """
    
    ### Initializer ###
    
    def __init__(self, alignment=None, edges=True):
        """
        An instance of a Defense subclass should not actually place ships
        on the board; that is the role of a Player object.
        Likewise, an Offense should not fire a shot or update the board,
        it should just provide targets; the Player instance is the object
        that actually interacts with the Board object.

        Parameters
        ----------
        alignment : str, optional
            A string indicating whether ships should be aligned along a partiular
            direction: "Vertical" (same as "North-South" or "N-S"),
            "Horizontal" (same as "East-West" or "E-W"), or "Any" (same as None).
            
            The default is None.
            
        edges : bool, optional
            When True, ships may be placed in the first and last rows and 
            columns of the board. When False, they may not.
            The default is True.
    

        Returns
        -------
        None.

        """
        self.fleet_alignment = alignment
        self.edges = edges
        
    def __repr__(self):
        return f"Defense({self.fleet_alignment}, {self.edges})"
    
    ### Abstract Methods ###
    
    @abc.abstractmethod
    def placement_for_ship(self, board, ship_type):
        pass
    
    ### Properties ###
    @property
    def fleet_alignment(self):
        return self._fleet_alignment
    
    @fleet_alignment.setter
    def fleet_alignment(self, align):
        if align == None or align.upper() == "ANY":
            self._fleet_alignment = Align.ANY
        elif align.upper() in ["VERTICAL", "NS", "N-S", "N/S", 
                       "NORTH-SOUTH", "NORTH/SOUTH"]:
            self._fleet_alignment = Align.VERTICAL
        elif align.upper() in ["HORIZONTAL", "EW", "E-W", "E/W", 
                       "EAST-WEST", "EAST/WEST"]:
            self._fleet_alignment = Align.HORIZONTAL
        else:
            raise ValueError("Alignment must be 'horizontal' or 'vertical', "
                             "or equivalent directions (NS, EW, etc.)")
        
    @property
    def edges(self):
        return self._edges
    
    @edges.setter
    def edges(self, value):
        self._edges = (value == True)
        
    ### Class factory method ###
    
    # @classmethod
    # def from_strs(cls, def_type, **kwargs):
    #     """
    #     Allowable def_types are:
    #         "random"
            
    #     Other parameters can be entered by keyword:
    #         align = 'horizontal' or 'vertical' (or 'any', the default)
    #         diagonal = True / False (Isolated Strategy only)
    #         max_sep = True / False (Isolated Strategy only)
    #         separation = integer 
            
    #     """
    #     if def_type == None or def_type.lower() == "random":
    #         return RandomDefense(**kwargs)
    #     else:
    #         raise ValueError("Invalid method string.")
            
    @classmethod
    def random(cls, weight=None, alignment=None, edges=True, **kwargs):
        return RandomDefense(weight, alignment, edges, **kwargs)
        
    ### Other methods ###
    
    def fleet_placements(self, board):
        """Return a dictionary of placements for each ship type. Each value 
        contains a tuple with (coordinate, heading) for the respective 
        ship_type key.
        """
        
        placements = {}
        # Since the Defense doesn't actually place ships on board, 
        # create a temporary copy of board to record placements as they are
        # are generated (to avoid invalid positions).
        newboard = Board()
        nships = len(SHIP_DATA)
        ship_order = range(1, nships+1)
        
        # ok = False
        # count = 0
        # while not ok and count <= MAX_ITER:
        #     newboard = Board(board.size)
        #     if _RANDOMIZE_SHIP_PLACEMENT_ORDER:
        #         ship_order = np.random.choice(ship_order, nships, replace=False)
        #     for ship_type in ship_order:
        #         coord, heading = self.placement_for_ship(newboard, ship_type)
        #         ok = (coord != None and heading != None)
        #         if ok:
        #             placements[ship_type] = {"coord": coord, "heading": heading}
        #             newboard.place_ship(ship_type, coord, heading)
        #     count += 1
        # if count == MAX_ITER:
        #     raise Exception("Maximum iterations reached while finding ship "
        #                     "placements.")
        # return placements
            
            
        if _RANDOMIZE_SHIP_PLACEMENT_ORDER:
            ship_order = [int(x) for x in 
                          np.random.choice(ship_order, nships, replace=False)]
            
        for ship_type in ship_order:
            coord, heading = self.placement_for_ship(newboard, ship_type)
            count = 0
            while coord == None and heading == None and count <= MAX_ITER:
                coord, heading = self.placement_for_ship(newboard, ship_type) 
                count += 1
            if coord == None or heading == None:
                raise Exception(f"No valid locations found for "
                                f"{SHIP_DATA[ShipType(ship_type)]['name']}")                    
            if not newboard.is_valid_ship_placement(ship_type, coord, heading):
                raise Exception(f"Cannot place "
                                f"{SHIP_DATA[ShipType(ship_type)]['name']} "
                                f"at {coord} with heading {heading}.")
            placements[ship_type] = {"coord": coord, "heading": heading}
            newboard.place_ship(ship_type, coord, heading)
        return placements
    
#%%
class RandomDefense(Defense):
    """
    Abstract Properties/Methods:
    - fleet_placements
    
    Properties
    - fleet_alignment
    
    Other methods
    - __init__
    - placement_for_ship
    """
    
    def __init__(self, weight=None, alignment=None, edges=True, **kwargs):
        """
        Creates an instance of RandomDefense with the indicated or default
        parameters.

        Parameters
        ----------
        weight : TYPE, optional
            One of the following strings which determines how to weight 
            different coordinates when choosing ship placement.
            'flat' or 'random'      All coordinates are equally likely
            'clustered'             Favor coordinates that are close to other 
                                    ships.
            'isolated'              Favor coordinates that are not adjacent
                                    to other ships, and as far as possible
                                    from other ships (if the 'maxsep' 
                                    parameter is set to True)
            The default is 'flat'.
            
        alignment : str, optional
            The alignment of all ships in the fleet. The alignment may be
            'horizontal'/'E-W', 'vertical'/'N-S', or 'any' (default).
        edges : Bool, optional
        
        Keyword Arguments
        -----------------
        buffer : int, optional
            Applies only when weight = 'isolated'. The minimum spacing between
            ships. The default is 1, meaning that there must be at least one
            space between two ships in order for a ship placement to be 
            considered valid (see also 'diagonal').
        diagonal : Bool, optional
            Applies only when weight = 'isolated'. If True, a ship placement 
            is allowed if it touches another ship on a diagonal, but not on
            an edge. If False, they cannot be adjacent even
            on a diagonal. The default is True.
        maxsep : Bool, optional 
            Applies only when weight = 'isolated'. If True, the defense will
            favor ship placements that are as far as possible from all other
            ships on the board. The default is False. 
        

        Returns
        -------
        None.

        """
        super().__init__(alignment, edges)
        if weight == None:
            self.weight = "flat"
        else:
            self.weight = weight
        self.buffer = kwargs.get('buffer')
        self.buffer = (0 if self.buffer == None else self.buffer)
        self.diagonal = kwargs.get('diagonal')
        self.diagonal = (True if self.diagonal == None else self.diagonal)
        self.max_sep = kwargs.get('maxsep')
        self.max_sep = (False if self.max_sep == None else self.max_sep)        
        
    def __repr__(self):
        r = (f"RandomDefense({self.weight!r}, {self.fleet_alignment!r}, "
             f"{self.edges!r}")
        if self.buffer:
            r += f", buffer={self.buffer!r}"
        if not self.diagonal:
            r += f", diagonal={self.diagonal!r}"
        if self.max_sep:
            r += f", max_sep={self.max_sep!r}"
        r += ")"
        return r
        
    
    @property
    def weight(self):
        return self._weight
    
    @weight.setter
    def weight(self, value):
        value = value.lower()
        value = ("flat" if value in ["flat", "random"] else
                 "isolated" if value.startswith("isolate") else
                 "clustered" if value.startswith("cluster") else
                 None)
        if value == None:
            raise ValueError("weight must be 'flat', 'isolated', 'clustered'.")
        self._weight = value
                 
    #@abc.abstractmethod
    def placement_for_ship(self, board, ship_type):
        """Returns a (coord, heading) tuple for the input ship type. The
        placement is based on the algorithm for this particular subclass of
        Defense. In this case, the algorithm favors placements that
        minimize the sum of distances between all coordinates of the input
        ship to all coordinates of all other ships already on the board.
        """
        
        new_ship = Ship(ship_type)
        
        # completely random / flat
        if self.weight == "flat":
            all_placements = board.all_valid_ship_placements(new_ship.length,
                alignment = self.fleet_alignment, edges = self.edges)
            prob = np.ones(len(all_placements))
        
        # cluster
        elif self.weight == "clustered":
            all_placements = board.all_valid_ship_placements(new_ship,
                alignment = self.fleet_alignment, edges = self.edges)
            prob = np.ones(len(all_placements))
            for (i, place) in enumerate(all_placements):
                if len(board.fleet) == 0:
                    pass
                else:
                    ds = board.relative_coords_for_heading(place[1], new_ship.length)
                    rows = place[0][0] + ds[0]
                    cols = place[0][1] + ds[1]
                    d2 = 0
                    for (r,c) in zip(rows, cols):
                        for other_ship in list(board.fleet.values()):
                            other_coords = np.array(board.coord_for_ship(other_ship))
                            delta = other_coords - np.tile((r,c), 
                                                           (other_coords.shape[0],1))
                            d2 += np.sum(delta**2)
                    prob[i] = 1 / d2
                    
        # isolated
        elif self.weight == "isolated":
            all_placements = board.all_valid_ship_placements(new_ship,
                distance = self.buffer,diagonal = self.diagonal,
                alignment = self.fleet_alignment, edges = self.edges)
            prob = np.ones(len(all_placements))
            if self.max_sep:
                for (i, place) in enumerate(all_placements):
                    ds = board.relative_coords_for_heading(place[1], new_ship.length)
                    rows = place[0][0] + ds[0]
                    cols = place[0][1] + ds[1]
                    for other_ship in list(board.fleet.values()):
                        other_coords = np.array(board.coord_for_ship(other_ship))
                        d2 = 0
                        for (r,c) in zip(rows, cols):
                            delta = (np.tile((r,c), (other_coords.shape[0],1))
                                     - other_coords)
                            d2 += np.sum(delta**2)
                        prob[i] *= d2
                if np.all(prob == 0):
                    prob = np.ones(prob.shape)
            
        if len(all_placements) == 0:
            # raise ValueError("No valid ship placements are possible. Relax"
            #                  " buffer, diagonal, or edges.")
            all_placements = [(None,None)]
            prob = np.array([1.])
        return random_choice(all_placements, p=prob/np.sum(prob))