#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 17:06:57 2023

@author: jason
"""

"""
A Defense object takes the following inputs:
    formation       A string description of the preferred ship arrangement.
                    'cluster(ed)', 'isolate(d)', or 'random'.
                    'clutered' prefers ship placements that are as close as 
                    possible, without violating the buffer & alignment settings
                    (below).
                    'isolated' prefers placements that are as far as possible,
                    without violating buffer & alignment settings.
                    'random' choses ship placements that satisfy alignment and
                    buffer settings, but are otherwise equally likely.
    method          A string that describes how placements are chosen; 
                    'optimize' selects that placement that best satisfies the 
                    formation (farthest away from existing ships for 'isolated'
                    formation, closest for 'clustered' formation, random for
                    'random' formation). 
                    'weighted' selects a placement for each ship based on how
                    far it is from existing ships (closer is better for 
                    'clustered' formation, farther is better for 'isolated').
                    'any' selects one of the valid placements with equal 
                    probability. For 'clustered', ony placements that are 
                    exactly ship_buffer away from another ship are considered 
                    valid.
                    The default is 'weighted'.
    edge_buffer     The number of spaces that must be left between a ship and
                    the edge of the board. Default is 0.
    ship_buffer     The minimum number of spaces that must be left between any
                    two slots on any two ships. Default is 0 (i.e., ships can 
                    be right against one another).
    fleet_alignment An IntENum that indicates whether ships must be aligned 
                    vertically, horizontally, or both.
    
It should implement these methods:
    fleet_placements (returns dict of dicts)
"""

#%% Imports
import abc
import numpy as np

# Imports from this package
from battleship import constants
from battleship import utils

from battleship.ship import Ship
from battleship.board import Board
from battleship.placement import Placement

#%% Constants

# Place ships on the board randomly; otherwise, go in ascending ShipType order.
_RANDOMIZE_SHIP_PLACEMENT_ORDER = True

# The number of times a defense will attempt to place a ship given the input
# parameters before timing out.
_MAX_ITER = 1000

#%% Class Definition

class Defense(metaclass=abc.ABCMeta):   
    
    ### Initializer ###
    
    def __init__(self, alignment=None, edges=True):
        """

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
        pass
        
    def __repr__(self):
        return f"Defense({self.fleet_alignment}, {self.edges})"
    
    ### Abstract Methods ###
    
    @abc.abstractmethod
    def placement_for_ship(self, board, ship_type):
        """
        This method is called by fleet_placements (a method of Defense,
        not a subclass) and should return that placement of the input ship
        on the input board.
        """
        pass
    
    ### Concrete Methods ###
    
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
        nships = len(Ship.data)
        ship_order = range(1, nships+1)
        
        if _RANDOMIZE_SHIP_PLACEMENT_ORDER:
            ship_order = [int(x) for x in 
                          np.random.choice(ship_order, nships, replace=False)]
            
        for ship_type in ship_order:
            placement = self.placement_for_ship(newboard, ship_type)
            count = 0
            while (placement.coord == None 
                   and placement.heading == None 
                   and count <= constants.MAX_ITER):
                placement = self.placement_for_ship(newboard, ship_type) 
                count += 1
            if placement.coord == None or placement.heading == None:
                if count > constants.MAX_ITER:
                    print("Maximum placement attempts exceeded.")
                    ship_name = Ship.data[constants.ShipType(ship_type)]['name']
                raise Exception(f"No valid locations found for "
                                f"{ship_name}")                    
            if not newboard.is_valid_placement(placement):
                ship_name = Ship.data[constants.ShipType(ship_type)]['name']
                raise Exception(f"Cannot place "
                                f"{ship_name} "
                                f"at {placement.coord} with heading "
                                f"{placement.heading}.")
            placements[ship_type] = placement
            newboard.add_ship(ship_type, placement.coord, placement.heading)
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
    
    def __init__(self, 
                 formation = "random", 
                 method = "weighted",
                 edge_buffer = 0, 
                 ship_buffer = 0, 
                 alignment = constants.Align.ANY):
        """
        RandomDefense implements an algorithm for placing ships on a Battleship
        board. Fleet arrangement can be controlled used the input parameters,
        which weight unoccuppied spots differently when randomly selecting a 
        placement for each ship. For example, if formation = 'cluster', ships
        will be preferentially placed close to other ships (i.e., spaces close
        to already-placed ships will be weighted more heavily when randomly
        chosing a space for the next ship).

        Parameters
        ----------
        formation : str
            A string description of how the ships should be arranged on the 
            Battleship board. All formations must adhere to the edge_buffer,
            ship_buffer, and alignment parameters detailed below.
            Allowable formations are:
                'cluster' or 'clustered'
                    Prefer ship placements that are as close as possible. 
                    
                    If method is 'weighted', closer placements are favored but
                    not guaranteed. 
                    If method is 'optimize', the closest placement will be 
                    chosen ("closest" means the smallest sum of space-to-space 
                    distances between all pairs of spaces on both ships). 
                    Note that this still leads to random fleet formations 
                    because the first ship is always placed randomly.
                    If method is 'any', a ship placement will be chosen 
                    randomly from among those placements that are exactly 
                    ship_buffer away from already-placed ships (similar to 
                    'weighted', but where weights for any placements more
                    than ship_buffer away from existing ships are set to 0).
                'isolate' or 'isolated'
                    Prefer placements that are as far as possible from all
                    already-placed ships.
                    
                    If method is 'weighted', farther placements are favored but
                    not guaranteed. 
                    If method is 'optimize', the placement that is farthest 
                    from all existing ships will be chosen ("farthest" means 
                    the largest sum of space-to-space distances between all 
                    pairs of spaces on both ships). 
                    If method is 'any', a ship placement will be chosen 
                    randomly from among those placements that are exactly 
                    ship_buffer away from already-placed ships (similar to 
                    'weighted', but where weights for any placements more
                    than ship_buffer away from existing ships are set to 0).
                            without violating buffer & alignment settings.
                            'random' choses ship placements that satisfy alignment and
                            buffer settings, but are otherwise equally likely.
                'random'
                    Placements are randomly chosen from possible placements.
                    
        method : str, optional
            A string that describes how placements are chosen. Allowable values
            are:
                'optimize' : Select the placement that best satisfies the 
                            formation condition (farthest away from existing 
                            ships for 'isolated' formation, closest for 
                            'clustered' formation, random for 'random' 
                            formation).  
                            Note that this still leads to random fleet 
                            formations because the first ship is always placed 
                            randomly.
                'weighted' : Select placement for each ship based on distance 
                            to existing ships (closer is better for 'clustered' 
                            formation, farther is better for 'isolated').   
                            No effect for 'random' formation.
                'any' :     Select one of the valid placements with equal 
                            probability. 
                            For 'clustered', ony placements that are exactly 
                            ship_buffer away from another ship are considered 
                            valid. 
            The default is 'weighted'.
            
        edge_buffer : int, optional
            The number of spaces that must be left between a ship and the edge 
            of the board. 
            The default is 0.
            
        ship_buffer : int, optional
            The number of spaces that must be left between a ship and all other
            ships. 
            The default is 0.
            
        alignment : Align (IntEnum), optional
            If this parameter is Align.VERTICAL or Align.HORIZONTAL, all ships
            must be pointed in a vertical or horizontal heading, respectively
            (i.e., a ship must occupy a single column or row, respectively). 
            If alignment is Align.ANY, either heading is acceptable. 
            The default is Align.ANY.

        Abstract Methods implemented
        -------
        fleet_placements : returns dict
            DESCRIPTION: Returns a dictionary of placements (one for each
                ship type) based on the Defense's parameters.
            
        Returns
        -------
        None.

        """
        
        super().__init__()
        self.formation = formation
        self.method = method
        self.edge_buffer = edge_buffer
        self.ship_buffer = ship_buffer
        self.alignment = alignment     
        
    def __repr__(self):
        r = (f"RandomDefense({self.formation!r}, {self.method!r}, "
             f"{self.edge_buffer!r}, {self.ship_buffer!r}, "
             f"{self.alignment})")
        return r
    
    def __str__(self):
        s = (f"RandomDefense(formation={self.formation!r}, "
             f"method={self.method!r}, "
             f"edge_buffer={self.edge_buffer!r}, "
             f"ship_buffer={self.ship_buffer!r}, "
             f"alignment={self.alignment})")
        return s
    
    ### Properties ###
    
    @property
    def formation(self):
        return self._formation
    
    @formation.setter
    def formation(self, value):
        value = value.lower()
        value = ("random" if value in ["flat", "random"] else
                 "isolated" if value.startswith("isolate") else
                 "clustered" if value.startswith("cluster") else
                 None)
        if value == None:
            raise ValueError("weight must be 'random', 'isolated'/'isolate', "
                             "or 'clustered'/'cluster'.")
        self._formation = value
      
    @property
    def method(self):
        return self._method
    
    @method.setter
    def method(self, value):
        value = value.lower()
        value = ("optimize" if value.startswith("optimize") else
                 "weighted" if value in ["weight", "weighted"] else
                 "any" if value in ["any"] else
                 None)
        if value == None:
            raise ValueError("method must be 'optimize', 'weighted', or"
                             " 'any'.")
        self._method = value
        
    @property
    def edge_buffer(self):
        return self._edge_buffer
    
    @edge_buffer.setter
    def edge_buffer(self, value):
        if not isinstance(value, (int, np.integer)):
            raise TypeError("Buffer must be a non-negative integer.")
        if value < 0:
            raise ValueError("Buffer must be >= 0.")
        self._edge_buffer = value
        
    @property
    def ship_buffer(self):
        return self._ship_buffer
    
    @ship_buffer.setter
    def ship_buffer(self, value):
        if not isinstance(value, (int, np.integer)):
            raise TypeError("Buffer must be a non-negative integer.")
        if value < 0:
            raise ValueError("Buffer must be >= 0.")
        self._ship_buffer = value
        
    @property
    def alignment(self):
        return self._alignment
    
    @alignment.setter
    def alignment(self, alignment):
        if isinstance(alignment, constants.Align):
            self._alignment = alignment
        elif alignment == None or alignment.upper() == "ANY":
            self._alignment = constants.Align.ANY
        elif alignment.upper() in ["VERTICAL", "NS", "N-S", "N/S", 
                                   "NORTH-SOUTH", "NORTH/SOUTH"]:
            self._alignment = constants.Align.VERTICAL
        elif alignment.upper() in ["HORIZONTAL", "EW", "E-W", "E/W", 
                                   "EAST-WEST", "EAST/WEST"]:
            self._alignment = constants.Align.HORIZONTAL
        else:
            raise ValueError("Alignment must be 'horizontal', 'vertical', "
                             "or equivalent directions (NS, EW, etc.),"
                             "or 'any'. Or an instance of Align.")
            
    ### Abstract method from Defense superclass ###
    
    def placement_for_ship(self, board, ship_type):
        """
        Returns a placement dictionary for the input board and ship type.
        The placement is based on the algorithm for the particular subclass
        of Defense and the instance's parameters.

        Parameters
        ----------
        board : Board
            An instance of a Board, which may already have ships placed.
        ship_type : ShipType (IntEnum or Int 1-5)
            The type of ship that will be placed on board.

        Returns
        -------
        dict
            A dictionary with the following key/values:
                'coord' : a (row, col) tuple
                'heading' : 'N' (north) or 'W' (west)
                'length' : the number of spaces the ship occupies                

        """
        possible_placements, probs = self.placement_probs(board, ship_type)
        
        if len(possible_placements) == 0:
            # If no possible placements were found, return a null placement
            possible_placements = [Placement(None, 
                                             None, 
                                             Ship.data[ship_type]['length'])]
            probs = np.ones((1)) * 1.
            
        return utils.random_choice(possible_placements, p=probs/np.sum(probs))
    
    ### Other Methods ###
    
    def placement_probs(self, board, ship_type):
        """
        Returns the viable placements and associated probabilities for placing
        a ship of type ship_type on board.

        Parameters
        ----------
        board : Board
            The board on which the ship of type ship_type is to be placed.
            This board may be empty or have some ships already placed.
        ship_type : ShipType or IntEnum (1-5)
            The numeric description of the type of ship to be placed.

        Returns
        -------
        possible_placements, probs : tuple
            possible_placements is a list of Placement objects.
            probs is a list of probabilities associated with each of the
            possible_placements, in order. 
            These two lists are the same length.

        """
        # Placement will be based on formation and method parameters
        if self.formation == "random" or len(board.fleet) == 0:
            possible_placements = board.all_valid_ship_placements(
                ship_type, 
                ship_buffer = self.ship_buffer, 
                edge_buffer = self.edge_buffer,
                alignment = self.alignment
                )
            probs = np.ones(len(possible_placements))
        elif self.formation == "isolated":
            possible_placements, probs = self.isolated_placements(ship_type,
                                                                  board)
        elif self.formation == "clustered":
            possible_placements, probs = self.clustered_placements(ship_type,
                                                                   board)
        
        return possible_placements, probs
    
    def isolated_placements(self, ship_type, board):
        
        """
        Returns all of the possible placements that conform to the defense's
        method, edge_buffer, ship_buffer, and alignment properties, as well as
        the corresponding probability that each placement should be chosen for 
        the location of the input type of ship. 

        Parameters
        ----------
        ship_type : ShipType (IntEnum or int 1-5)
            The type of ship for which placements are desired.
        board : Board
            The board on which the ship is to be placed.

        Returns
        -------
        (placements, probs) : tuple
            A 2-element tuple. First element is a list of placement 
            dictionaries, and the second element is a list of probabilities
            corresponding to each placement (probs[i] is the probability 
            corresponds to placements[i]).

        """
        placements = board.all_valid_ship_placements(
            ship_type, 
            ship_buffer = self.ship_buffer, 
            edge_buffer = self.edge_buffer,
            alignment = self.alignment
            )
        probs = np.ones(len(placements))
        
        for (idx,place) in enumerate(placements):
            for ship_placement in board.ship_placements.values():
                probs[idx] = place.total_dist_to_placement(ship_placement, 
                                                           method='dist')
        
        if self.method == "optimize":
            # choose the placements that are farthest away from all ships
            idx = np.flatnonzero(probs == probs.max())
            probs = probs[idx]
            placements = [placements[i] for i in idx]
        elif self.method == "any":
            # all valid placements are equally valid, so set all probs equal
            probs = np.ones(len(placements))
        elif self.method == "weighted":
            # select according to probs
            pass
        else:
            raise ValueError(f"Invalid method: {self.method}")
        
        return placements, probs
        
    def clustered_placements(self, ship_type, board):
        """
        Returns all of the possible placements that conform to the defense's
        method, edge_buffer, ship_buffer, and alignment properties, as well as
        the corresponding probability that each placement should be chosen for 
        the location of the input type of ship. 

        Parameters
        ----------
        ship_type : ShipType (IntEnum or int 1-5)
            The type of ship for which placements are desired.
        board : Board
            The board on which the ship is to be placed.

        Returns
        -------
        (placements, probs) : tuple
            A 2-element tuple. First element is a list of placement 
            dictionaries, and the second element is a list of probabilities
            corresponding to each placement (probs[i] is the probability 
            that corresponds to placements[i]).

        """
        placements = board.all_valid_ship_placements(
            ship_type, 
            ship_buffer = self.ship_buffer, 
            edge_buffer = self.edge_buffer,
            alignment = self.alignment
            )
        probs = np.ones(len(placements))
        
        for (idx,place) in enumerate(placements):
            for ship_placement in board.ship_placements.values():
                probs[idx] = 1. / place.total_dist_to_placement(ship_placement, 
                                                                method='dist2')
        if self.method == "optimize":
            # choose the placements that are closest to other ships
            idx = np.flatnonzero(probs == probs.max())
            probs = probs[idx]
            placements = [placements[i] for i in idx]
        elif self.method == "any":
            # select from those placements are are exactly one ship_buffer 
            # away from the nearest ship.
            print(self.ship_buffer)
            keep = []
            for (place, p) in zip(placements, probs):
                for ship_place in board.ship_placements.values():
                    D = place.placement_dist_matrix(ship_place)
                    if (D.min() - 1) == self.ship_buffer:
                        keep += [True]
            placements = [p for (p, k) in zip(placements, keep) if k]
            probs = [pr for (pr, k) in zip(probs, keep) if k]
            
        elif self.method == "weighted":
            pass
        else:
            raise ValueError(f"Invalid method: {self.method}")
        
        return placements, probs