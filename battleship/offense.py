#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 09:24:00 2023

@author: jason
"""

"""
The philosophy behind an Offense is that an instance should be able to be 
given a Battleship board and a shot outocme history, and return a suggested 
next target. 
An Offense does not own a board or a history, nor does it have a "parent"
board or history. In fact, a single Offense could be shared by multiple
Players and it could serve both of them, giving appropriate targets for their
differing boards and histories.
An Offense should never modify a board or a Player; it should use the state of
a board and player history to calculate a target, and return that target when
the appropriate methods are called.

"""

#%% Imports

import abc
from enum import Enum
import numpy as np
from collections import Counter

# Imports from this package
from battleship import constants
from battleship import utils

from battleship.coord import Coord
from battleship.ship import Ship
from battleship.placement import Placement


#%%
### Enums ###

class Mode(Enum):
    HUNT = 0
    KILL = 1
    

#%%
#########################        
### Class Definitions ###
#########################    

class Offense(metaclass=abc.ABCMeta):
    
    ### Initializer ###
    
    def __init__(self):
        pass
    
    ### Basic Methods ###
    
    def __repr__(self):
        return "Offense()"
    
    ### Abstract Methods ###
    
    @abc.abstractmethod
    def target(self, board, outcome_history):
        """
        Returns a target coordinate based on the input board and the shot/
        outcome history in outcome_history.

        Parameters
        ----------
        board : Board
            An instance of a Board on which the offense is to select a target
            coordinate.
        outcome_history : An ordered list of outcomes to previous shots at the
            board. The first outcome is at index 0, and the most recent outcome
            is at index len(outcome_history) - 1. 
            
            Each element of outcome_history should be a dictionary with the 
            following key/value pairs:
                'coord': tuple (row, col); the targeted coordinate.
                'hit': bool; True if shot hit a ship.
                'sunk': bool; True if shot resulted in sinking a ship.
                'sunk_ship_type': ShipType; the reported type of the hit ship, 
                    if a ship was sunk.
                'inferred_ship_type': ShipType; for future use.
            

        Returns
        -------
        target : tuple
            A two-element (row,col) tuple which can be used as the target for
            a player's next shot.

        """
        pass
    
    ### Factory Methods ###
    
    def from_name(cls, name, style, **kwargs):
        """
        Returns an instance of a subclass of Offense. The specific subclass is
        based on the input name. The other parameters are passed to that 
        subclass to set its properties.

        Parameters
        ----------
        name : str
            One of the following:
                'hunter' : returns an instance of HunterOffense
                .
        style : str
            The style .
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
    ### Other Methods ###
    

###################################
### CONCRETE OFFENSE SUBCLASSES ###
###################################

class HunterOffense(Offense):
    
    def __init__(self, 
                 hunt_style, 
                 hunt_pattern=None, 
                 hunt_options={}, 
                 kill_options={}):
        """
        Initializes a HunterOffense ("Hunter") instance. A Hunter is 
        characterized by two modes of operation: Hunt and Kill.
        In Hunt mode, the Offense searches for ships by firing shots at
        untargeted spaces. Once a hit is scored, the Hunter enters Kill
        mode, in which it fires only at spaces around the most recent hit
        until a ship is sunk. Once a sink occurs (or no potential targets
        remain), the Hunter returns to Hunt mode.
        
        The algorithm by which the Hunter select targets while in hunt mode
        is determined by the hunt_style, hunt_pattern, and pattern_options
        parameters.
        
        When no valid targets remain that conform to the hunt_style, 
        hunt_patten, and pattern_options values, a target is selected 
        from all remaining targets. 
        
        If pattern_options['no_valid_targets'] is 'random' (default), a target
        will be selected randomly from the untargeted spaces (still weighted
        according to hunt_pattern, if applicable). If it is 'ordered', the
        target will be selected in ascending order from the untargeted spaces,
        moving from column to column along a single row, then on to the next 
        row.
        

        Parameters
        ----------
        hunt_style : str
            A string that sets the behavior of the hunter offense when 
            searching for a ship to sink. Allowable options are:
                'random' :  Fire randomly at untargeted spaces. The probability
                            of firing at a particular space may be weighted
                            differently using the hunt_patten options, below.
                'pattern' : Fire at spaces on a fixed pattern. The shape of the
                            pattern is specified in hunt_pattern, and the 
                            details of how spaces are selected are set using
                            pattern_options.
        hunt_pattern : str, optional
            Describes the shape of the target pattern. The hunt_pattern must
            be specified for all hunt_styles except for 'random'. Allowable
            values of hunt_pattern vary for different hunt_styles as shown
            below:
                
            'random' hunt_style:
                'random' (default): All untargeted spaces are equally likely.
                'maxprob': Targets are weighted by the number of possible
                            placements at each space.
                'isolate(d)' : Weight spaces by how far they are from known
                            hits or misses (see pattern_options)
                'cluster(ed)' : Weight spaces by how close they are to known
                            hits or misses (see pattern_options)
            'pattern' hunt_style:
                'grid' : Shots are fired on a square or rectangular grid.
                'diagonals' : Shots are fired along diagonal lines.
                'spiral' : Shots are fired on a spiral that starts in a 
                            corner or in the center.
                pattern_options is used to set the specifics of these patterns.
                
            When hunt_style == 'random', the default hunt_pattern is 'random'.
            When hunt_style == 'pattern', the default hunt_pattern is 'grid'.
                        
        hunt_options : dict, optional
            A dictionary containing key/value pairs that control the details
            of the shot pattern. Some keys are available to all hunt_styles
            and hunt_patterns, while others are only available to specific 
            ones.
                'edge_buffer': (int) Do not target spaces within this many 
                                spaces of the edge of the board. Default is 0.
                'ship_buffer': (int) Do not target spaces within this many 
                                spaces of a known ship. Default is 0.
                'weight' : (str) 'hits' 'misses', 'shots'/'any'
                                'weight' is applicable to the 'random' 
                                hunt_style only.
                                
                                If 'hits', weighting will occur based on
                                a space's distance to all hits (i.e.,
                                spaces that are far from all hits will be 
                                likely for 'clustered' hunt_pattern, and 
                                unlikely for 'isolated' hunt_pattern).
                                If 'misses', weighting will occur based on
                                a space's distance to all missed shots.
                                If 'shots' or 'any', weighting will occur
                                based on a space's distance to all shots
                                taken, regardless of whether a shot is a 
                                hit or a miss.
                'rotate' : (int) 0, +/-90, +/-180, 270, 360
                                'pattern' hunt_stlye only.
                                The degrees to rotate a pattern relative to 
                                a starting position of coordinate A1.
                'mirror' : (str) 'horizontal' or 'vertical'
                                'pattern' hunt_style only.
                                If 'horizontal', the pattern will be flipped
                                across a horizontal (row).
                                If 'vertical', it will be flipped across a 
                                vertical (column).
                'spacing' : (int) 'pattern' hunt_style only.
                                The separation between subsequent targets
                                in a pattern. 
                'secondary_spacing' : (int) 'pattern' hunt_style only.
                                Some hunt_pattern types accept a spacing
                                along a second direction.
                                For 'diagonal', This is the horizontal 
                                separation between each diagonal line.
                                For 'grid', it is the horizontal separation 
                                between each vertical line.
                                For 'spiral', it has no effect.
                'no_valid_targets' : (str) 'random' or 'ordered'
                                Sets the Offense's behavior once the pattern
                                list has no more target coordinates.
                                If 'random', a target will be selected randomly
                                from the untargeted spaces (still weighted
                                according to hunt_pattern, if applicable). 
                                If 'ordered', the target will be selected in 
                                ascending order from the untargeted spaces,
                                moving from column to column along a single 
                                row, then on to the next row.
                            
        kill_options : dict, optional
            A dictionary containing key/value pairs that control the details
            of target selection in Kill mode. 
            The default is {'method' : 'advanced'}
                'method' : (str) 'dumb', 'basic' or 'advanced'
                                'basic' moves along rows/columns of hits
                                to guess where the next one might be
                                'advanced' takes possible placements of every
                                remaining ship into account to generate a 
                                probability around the most recent hit.

        Returns
        -------
        None.

        """
        
        super().__init__()
        
        # Parse inputs
        hunt_style = hunt_style.lower()
        self._hunt_style = hunt_style
        if hunt_pattern is None:
            if hunt_style == "random":
                hunt_pattern = "random" 
            elif hunt_style == "pattern":
                hunt_pattern = "grid" 
        hunt_style = hunt_style.lower()
        self._hunt_pattern = hunt_pattern
        self._original_hunt_options = hunt_options.copy()
        self._original_kill_options = kill_options.copy()
        self._hunt_options = HunterOffense.parse_hunt_options(hunt_style,
                                                              hunt_pattern,
                                                              hunt_options)
        self._kill_options = HunterOffense.parse_kill_options(kill_options)
        
        self._state = Mode.HUNT
        self._first_hit = None
        self._target_probs = None
        
        #*** Currently for debug only
        self._board = None
        self._outcome_history = None
        
        
    def __repr__(self):
        return (f"HunterOffense({self._hunt_style!r}, "
                f"{self._hunt_pattern!r}, "
                f"{self._original_hunt_options!r}, "
                f"{self._original_kill_options!r})")
    
    def __str__(self):
        return (f"{repr(self)}"
                f"  state: {self.state}"
                f"  first_hit: {self.first_hit}"
                )
    
    ### Properties ###
    
    @property
    def state(self):
        return self._state
    
    @state.setter
    def state(self, value):
        if value == Mode.HUNT or value == Mode.KILL:
            self._state = value
        else:
            raise ValueError("state must be a valid Mode value.")
            
    @property
    def first_hit(self):
        return self._first_hit
    
    @first_hit.setter
    def first_hit(self, value):
        if isinstance(value, (tuple, Coord)):
            self._first_hit = value
        else:
            raise TypeError("first_hit coord needs to be a tuple or Coord "
                            "object.")
            
    @property
    def hunt_options(self):
        return self._hunt_options
    
    @hunt_options.setter
    def hunt_options(self, value):
        self.hunt_options = HunterOffense.parse_hunt_options(self.hunt_style,
                                                             self.hunt_pattern, 
                                                             value)
        
    @property
    def kill_options(self):
        return self._kill_options
    
    @kill_options.setter
    def kill_options(self, value):
        self.kill_options = HunterOffense.parse_kill_options(value)
        
    @property
    def board(self):
        return self._board
    
    @board.setter
    def board(self, value):
        self._board = value
        
    @property
    def outcome_history(self):
        return self._outcome_history
    
    @outcome_history.setter
    def outcome_history(self, value):
        self._outcome_history = value
       
        
    ### Abstract Method Definitions ###
    
    def target(self, board=None, outcome_history=None):
        """
        Returns a target coordinate based on the input board and the shot/
        outcome history in outcome_history.

        Parameters
        ----------
        board : Board
            An instance of a Board on which the offense is to select a target
            coordinate.
        outcome_history : An ordered list of outcomes to previous shots at the
            board. The first outcome is at index 0, and the most recent outcome
            is at index len(outcome_history) - 1. 
            
            Each element of outcome_history should be a dictionary with the 
            following key/value pairs:
                'coord': tuple (row, col); the targeted coordinate.
                'hit': bool; True if shot hit a ship.
                'sunk': bool; True if shot resulted in sinking a ship.
                'sunk_ship_type': ShipType; the reported type of the hit ship, 
                    if a ship was sunk.
                'inferred_ship_type': ShipType; for future use.
            

        Returns
        -------
        target : tuple
            A two-element (row,col) tuple which can be used as the target for
            a player's next shot.

        """
        #** For debug purposes only **
        if board == None:
            board = self.board
        if outcome_history == None:
            outcome_history = self.outcome_history
        # *** ^^^^^^^^ ***
            
        if self.state == Mode.HUNT:
            self._target_probs = self.hunt_targets(board, outcome_history)
        elif self.state == Mode.KILL:
            self._target_probs = self.kill_targets(board, outcome_history)
        targets, probs = zip(*self._target_probs.items())
        target = utils.random_choice(targets, probs)
        return target
    
    
    ### Other Methods ###
    
    def hunt_targets(self, board=None, outcome_history=None):
        """
        Returns possible target coordinates and associated probabilities for 
        choosing each target coordinate.

        Parameters
        ----------
        board : Board
            The board whose target grid (vertical grid) should be used to
            determinate a hunt target coordinate.
        outcome_history : list
            A list of outcome dictionaries, with most recent outcome last.
            
            Each element of outcome_history should be a dictionary with the 
            following key/value pairs:
                'coord': tuple (row, col); the targeted coordinate.
                'hit': bool; True if shot hit a ship.
                'sunk': bool; True if shot resulted in sinking a ship.
                'sunk_ship_type': ShipType; the reported type of the hit ship, 
                    if a ship was sunk. None if no ship was sunk.
                'inferred_ship_type': ShipType; for future use.

        Returns
        -------
        targets_with_probs : dict
            A dictionary with keys equal to each possible target tuple (i.e.,
            a (row,col) tuple) and value equal to the probability that the 
            Hunter should choose that coordinate as its target.

        """
        
        #** For debug purposes only **
        if board == None:
            board = self.board
        if outcome_history == None:
            outcome_history = self.outcome_history
        # *** ^^^^^^^^ ***
        
        if self._hunt_style == "random":
            targets_with_probs = self.random_hunt_targets(board, outcome_history)
        elif self._hunt_style == "pattern":
            targets_with_probs = self.pattern_hunt_targets(board, outcome_history)
        else:
            raise ValueError(f"{self._hunt_style} is not a valid value for "
                             "the hunt_style parameter.")
            
        return targets_with_probs
    
    def random_hunt_targets(self, board=None, outcome_history=None):
        """
        Returns possible target coordinates and associated probabilities for 
        choosing each target coordinate for a randomly chosen target in 
        hunt mode.

        Parameters
        ----------
        board : Board
            The board whose target grid (vertical grid) should be used to
            determinate a hunt target coordinate.
        outcome_history : list
            A list of outcome dictionaries, with most recent outcome last.
            
            Each element of outcome_history should be a dictionary with the 
            following key/value pairs:
                'coord': tuple (row, col); the targeted coordinate.
                'hit': bool; True if shot hit a ship.
                'sunk': bool; True if shot resulted in sinking a ship.
                'sunk_ship_type': ShipType; the reported type of the hit ship, 
                    if a ship was sunk. None if no ship was sunk.
                'inferred_ship_type': ShipType; for future use.

        Returns
        -------
        targets_with_probs : dict
            A dictionary with keys equal to each possible target tuple (i.e.,
            a (row,col) tuple) and value equal to the probability that the 
            Hunter should choose that coordinate as its target.

        """
        #** For debug purposes only **
        if board == None:
            board = self.board
        if outcome_history == None:
            outcome_history = self.outcome_history
        # *** ^^^^^^^^ ***
        
        targets = [Coord(t) for t in board.all_targets(untargeted=True)]
        if self._hunt_pattern == "random":
            probs = np.ones(len(targets))
            
        elif self._hunt_pattern == "maxprob":
            # Probability is proportional to number of possible ways any ship
            # could be placed so that it sits on a given coordinate
            targets = [Coord((r,c)) for r in range(board.size) 
                       for c in range(board.size)]
            probs = board.possible_targets_grid().flatten('C')
            targets = [t for (idx,t) in enumerate(targets) if probs[idx] > 0]
            probs = probs[probs > 0]
            
        
        elif self._hunt_pattern in ["isolate", "isolated", 
                                    "cluster", "clustered"]:
            if self._hunt_options['weight'] == "hits":
                marks = board.all_targets_where(hit=True)
            elif self._hunt_options['weight'] == "misses":
                marks = board.all_targets_where(miss=True)
            elif self._hunt_options['weight'] in ["any", "shots"]:
                marks = board.all_targets(targeted=True)
            marks = [Coord(m) for m in marks]
            
            # Calculate distances to all marks (hits and/or misses):
            probs = np.zeros(len(targets))
            dist_method = self._hunt_options['dist_method']
            for (idx,target) in enumerate(targets):
                distances = np.array([target.dists_to(mark, dist_method) 
                                      for mark in marks])
                probs[idx] = np.sum(distances)
                
            # Compute a target's probability based on summed distance to each 
            # mark and the hunt_pattern settings
            if (self._hunt_pattern == "isolate" or 
                    self._hunt_pattern == "isolated"):
                # favor far distance from marks
                pass
            elif (self._hunt_pattern == "cluster" or 
                      self._hunt_pattern == "clustered"):
                # favor close distances to marks
                probs = 1. / probs
            
        else:
            raise ValueError(f"{self._hunt_pattern} is not a valid value for "
                             "the hunt_pattern parameter.")
        probs = probs / np.sum(probs)
        targets = [target.rowcol for target in targets]
        return dict(zip(targets, probs))
                    
    def kill_targets(self, board=None, outcome_history=None):
        """
        Returns possible target coordinates and associated probabilities for 
        choosing each target coordinate. Probabilities are based on the goal
        of sinking the most recently hit ship.

        Parameters
        ----------
        board : Board
            The board whose target grid (vertical grid) should be used to
            determinate a kill target coordinate.
        outcome_history : list
            A list of outcome dictionaries, with most recent outcome last.
            
            Each element of outcome_history should be a dictionary with the 
            following key/value pairs:
                'coord': tuple (row, col); the targeted coordinate.
                'hit': bool; True if shot hit a ship.
                'sunk': bool; True if shot resulted in sinking a ship.
                'sunk_ship_type': ShipType; the reported type of the hit ship, 
                    if a ship was sunk. None if no ship was sunk.
                'inferred_ship_type': ShipType; for future use.

        Returns
        -------
        targets_with_probs : dict
            A dictionary with keys equal to each possible target tuple (i.e.,
            a (row,col) tuple) and value equal to the probability that the 
            Hunter should choose that coordinate as its target.

        """
        #** For debug purposes only **
        if board == None:
            board = self.board
        if outcome_history == None:
            outcome_history = self.outcome_history
        # *** ^^^^^^^^ ***
        
        initial_hit = self.first_hit
        
        if self.kill_options['method'] == 'dumb':
            print("kill_targets_around_last_hit")
            targets_with_probs = self.kill_targets_around_last_hit(board,
                                                            outcome_history)
        elif (self.kill_options['method'] == 'advanced' 
                and initial_hit == outcome_history[-1]['coord']):
            print("kill_targets_from_placements")
            targets_with_probs = self.kill_targets_from_placements(board, 
                                                            outcome_history,
                                                            True,
                                                            True)
        else:   
            # method == 'basic' OR method == 'advanced' on the second 
            # (or more) shot.
            print("kill_targets_from_shape")
            targets_with_probs = self.kill_targets_from_shape(board, 
                                                              outcome_history)
        return targets_with_probs
    
    
    def kill_targets_around_last_hit(self, board=None, outcome_history=None):
        """
        Returns a target around the last successful hit

        Parameters
        ----------
        board : Board
            The board whose target grid (vertical grid) should be used to
            determinate a kill target coordinate.
        outcome_history : list
            A list of outcome dictionaries, with most recent outcome last.
            
            Each element of outcome_history should be a dictionary with the 
            following key/value pairs:
                'coord': tuple (row, col); the targeted coordinate.
                'hit': bool; True if shot hit a ship.
                'sunk': bool; True if shot resulted in sinking a ship.
                'sunk_ship_type': ShipType; the reported type of the hit ship, 
                    if a ship was sunk. None if no ship was sunk.
                'inferred_ship_type': ShipType; for future use.

        Returns
        -------
        targets_with_probs : dict
            A dictionary with keys equal to each possible target tuple (i.e.,
            a (row,col) tuple) and value equal to the probability that the 
            Hunter should choose that coordinate as its target.

        """
        #** For debug purposes only **
        if board == None:
            board = self.board
        if outcome_history == None:
            outcome_history = self.outcome_history
        # *** ^^^^^^^^ ***
        
        targets = board.targets_around(self.last_hit(outcome_history), 
                                       untargeted=True)
        return dict(zip(targets, [1]*len(targets)))
    
    
    def kill_targets_from_shape(self, board=None, outcome_history=None):
        """
        Returns a dict of target coordinates and probabilities (which are
        uniform, in this case) based on logically determining the likely
        location of a target ship based on the board's target grid.

        Parameters
        ----------
        board : Board
            The board whose target grid (vertical grid) should be used to
            determinate a kill target coordinate.
        outcome_history : list
            A list of outcome dictionaries, with most recent outcome last.
            
            Each element of outcome_history should be a dictionary with the 
            following key/value pairs:
                'coord': tuple (row, col); the targeted coordinate.
                'hit': bool; True if shot hit a ship.
                'sunk': bool; True if shot resulted in sinking a ship.
                'sunk_ship_type': ShipType; the reported type of the hit ship, 
                    if a ship was sunk. None if no ship was sunk.
                'inferred_ship_type': ShipType; for future use.

        Returns
        -------
        targets_with_probs : dict
            A dictionary with keys equal to each possible target tuple (i.e.,
            a (row,col) tuple) and value equal to the probability that the 
            Hunter should choose that coordinate as its target.

        """
        #** For debug purposes only **
        if board == None:
            board = self.board
        if outcome_history == None:
            outcome_history = self.outcome_history
        # *** ^^^^^^^^ ***
        
        initial_hit = self.first_hit
        last_hit = self.last_hit(outcome_history)
        last_target = outcome_history[-1]["coord"]
        
        targets = []
        if last_target == initial_hit:
            # if this is the first hit, choose an adjacent target
            targets = board.targets_around(initial_hit, untargeted=True)
        else:
            d = (last_hit[0] - initial_hit[0], last_hit[1] - initial_hit[1])
            targets = self.targets_on_line(board, initial_hit, d, True)
            if targets:
                # keep moving along the same direction as previous shot
                targets = [targets[0]]  
            else:
                d = (last_hit[1] - initial_hit[1], 
                     (initial_hit[0] - last_hit[0])) # rotated by 90 degrees
                targets = self.targets_on_line(board, initial_hit, d, True)
                # don't need to take first target because we have not yet moved
                # along this row/col
                
        if not targets:
            return {}
        else:
            return dict(zip(targets, [1]*len(targets)))
        
        
    def targets_on_line(self, 
                        board, 
                        init_hit, 
                        direction, 
                        both_directions=True):
        """
        Returns viable untargeted spaces at the end of a line of hits starting
        at init_hit and extending along the vector in direction (optionally 
        both along direction and opposite to direction)

        A viable space is one that is untargeted (no hit or miss) and connected
        to init_hit by unattributed hits only.
        
        Parameters
        ----------
        board : Board
            The board with a populated target_grid that will be searched for
            viable untargeted spaces along the row/column defined by init_hit
            and direction.
        init_coord : tuple
            A (row,col) tuple.
        direction : tuple
            A two-element tuple indicating which direction to search along. 
            Must be one of the following:
                (1,0)  : search below init_hit (along the column containing 
                         init_hit, at higher row numbers)
                (-1,0) : search above init_hit (along the column containing
                         init_hit, at lower row numbers)
                (0,1)  : search to the right of init_hit (along the row 
                         containing init_hit, at higher column numbers)
                (0,-1) : search to the left of init_hit (along the row
                         containing init_hit, at lower column numbers)
        both_directions: bool
            If True, up to two endpoints will be returned; the one closest
            to init_hit along direction, and the one closest to init_hit along
            -direction. The default is True

        Returns
        -------
        targets : list
            A zero-, one-, or two-element list containing the coordinates at
            the endpoints of the line of hits starting at init_hit and 
            extending along the row/column defined by direction.
            
            If both_directions is True (default), up to two targets can be
            returned. If it is False, up to one target will be returned. In
            either case, an empty list may be returned if there are no valid
            targets found.

        """
        # normalize direction
        direction = (int(abs(direction[0]) > 1e-9), 
                   int(abs(direction[1]) > 1e-9))
        
        # find the first non-hit space starting at init_hit and moving along
        # direction
        next_coord = (init_hit[0] + direction[0], init_hit[1] + direction[1])
        while (utils.is_on_board(next_coord, board.size) 
               and board.target_grid[next_coord] == constants.TargetValue.HIT):
            next_coord = (next_coord[0] + direction[0], 
                          next_coord[1] + direction[1])
        if not utils.is_on_board(next_coord, board.size):
            targets = []
        elif board.target_grid[next_coord] == constants.TargetValue.UNKNOWN:
            targets = [next_coord]
        else:
            targets = []
            
        if both_directions:
            targets += self.targets_on_line(board, 
                                            init_hit, 
                                            (-direction[0],-direction[1]), 
                                            both_directions=False)
        return targets
        
        
    def kill_targets_from_placements(self, 
                                     board=None, 
                                     outcome_history=None, 
                                     variable_prob=True,
                                     last_hit_adjacent=True):
        """
        Determines kill target based on possible placements around the hit
        that set the Hunter into kill mode. 

        Parameters
        ----------
        board : Board
            The board whose target grid (vertical grid) should be used to
            determinate a kill target coordinate.
        outcome_history : list
            A list of outcome dictionaries, with most recent outcome last.
            
            Each element of outcome_history should be a dictionary with the 
            following key/value pairs:
                'coord': tuple (row, col); the targeted coordinate.
                'hit': bool; True if shot hit a ship.
                'sunk': bool; True if shot resulted in sinking a ship.
                'sunk_ship_type': ShipType; the reported type of the hit ship, 
                    if a ship was sunk. None if no ship was sunk.
                'inferred_ship_type': ShipType; for future use.
        variable_prob : bool, optional
            If True, the probability of chosing a target will be proportional
            to the number of possible placements at that target. If False, 
            the probability will be uniform for all potential target 
            coordinates.
            The default is True.
        last_hit_adjacent : bool, optional
            If True, only the coordinates that are adjacent to the most recent
            hit will be returned. If False, any coordinates within a ship's 
            length with be returned (remaining ships only).

        Returns
        -------
        targets_with_probs : dict
            A dictionary with keys equal to each possible target tuple (i.e.,
            a (row,col) tuple) and value equal to the probability that the 
            Hunter should choose that coordinate as its target.

        """
        #** For debug purposes only **
        if board == None:
            board = self.board
        if outcome_history == None:
            outcome_history = self.outcome_history
        # *** ^^^^^^^^ ***
        
        initial_hit = self.first_hit
        sunk_ships = [outcome['sunk_ship_type'] for outcome in outcome_history
                      if outcome['sunk_ship_type'] is not None]
        afloat_ships = [i for i in range(1,len(Ship.data)+1) 
                        if not i in sunk_ships]
        coords = []
        for ship_type in afloat_ships:
            ship_len = Ship.data[ship_type]["length"]
            for shift in range(ship_len):
                place1 = Placement((initial_hit[0] - shift, initial_hit[1]),
                                   "N", ship_len)
                place2 = Placement((initial_hit[0], initial_hit[1] - shift),
                                   "W", ship_len)
                if board.is_valid_target_ship_placement(place1):
                    coords += place1.coords()
                if board.is_valid_target_ship_placement(place2):
                    coords += place2.coords()
        
        # get rid of coords that already have hits/misses
        if last_hit_adjacent:
            last_hit = self.last_hit(outcome_history)
            if not last_hit:
                raise ValueError("Could not find last hit in history.")
            last_hit_coord = Coord(last_hit)
            coords = [coord for coord in coords 
                      if ((board.target_grid[coord] 
                          == constants.TargetValue.UNKNOWN) 
                          and last_hit_coord.next_to(coord))]
        
        if not coords:
            return {}
        if variable_prob:
            return dict(Counter(coords))
        else:
            coords = set(coords)
            return dict(zip(coords, [1]*len(coords)))
        
    def outcomes_in_current_state(self, outcome_history=None):
        """
        Returns a list of outcomes that have occurred since the Hunter 
        was put into its current state. This is useful for determining where
        shots have been fired while the Hunter has been in kill mode.

        Parameters
        ----------
        outcome_history : list
            List of outcomes in order from earliest to most recent.
            
        Returns
        -------
        outcomes : list
            A list of outcome dictionaries for shots that have been taken
            since the Hunter was put into its current state. The outcome
            when the Hunter was first put into its current state is first,
            and the most recent outcome is last.

        """
        #** For debug purposes only **
        if outcome_history == None:
            outcome_history = self.outcome_history
        # *** ^^^^^^^^ ***
        
        outcomes = []
        if not outcome_history:
            return []
        
        targets = [outcome['coord'] for outcome in outcome_history]
        if self.first_hit in targets:
            idx = targets.index(self.first_hit)
            outcomes = outcome_history[idx:]
        else:
            raise Exception("Target for current state cannot be found "
                            "in outcome history. That shouldn't be "
                            "possible for a self-consistent offense.")
        return outcomes
    
    def update_state(self, outcome):
        """
        Changes the state from hunt to kill or kill to hunt based on the 
        current state and the outcome of the last shot. This logic is as 
        follows:
            
            In HUNT mode:
                If outcome was a HIT:
                    If outcome was a SINK --> stay in HUNT mode
                    Otherwise --> go to KILL mode
                If outcome was a MISS --> stay in HUNT mode
            
            In KILL mode:
                If outcome was a HIT:
                    If outcome was a SINK --> go to HUNT mode
                    Otherwise --> stay in KILL mode
                If outcome was a MISS --> stay in KILL mode

        Parameters
        ----------
        outcome : dict
            An outcome dictionary that contains the following key/value pairs:
                'coord': tuple (row, col); the targeted coordinate.
                'hit': bool; True if shot hit a ship.
                'sunk': bool; True if shot resulted in sinking a ship.
                'sunk_ship_type': ShipType; the reported type of the hit ship, 
                    if a ship was sunk.
                'inferred_ship_type': ShipType; for future use.                

        Returns
        -------
        None.

        """
        if self.state == Mode.HUNT:
            if outcome['hit'] and not outcome['sunk']:
                self.set_to_kill(outcome)
        else:
            if outcome['hit'] and outcome['sunk']:
                self.set_to_hunt(outcome)
                
        
    def last_hit(self, outcome_history=None):
        """
        Returns the coordinate of the most recent shot that resulted in a hit.

        Parameters
        ----------
        outcome_history : list
            Ordered list of outcome dictionaries.

        Returns
        -------
        hit_coord : tuple
            Two-element (row,col) tuple where the Hunter last hit a target on
            the opponent's board. If no hits have been scored, returns None.

        """
        #** For debug purposes only **
        if outcome_history == None:
            outcome_history = self.outcome_history
        # *** ^^^^^^^^ ***
        
        last_hit = None
        for outcome in reversed(outcome_history):
            if outcome['hit']:
                last_hit = outcome['coord']
                break
        return last_hit
    
    
    def set_to_hunt(self, outcome):
        """
        Sets the Hunter to hunt mode. Throws a warning if already in hunt mode.

        Parameters
        ----------
        outcome : dict
            An outcome dict that contains, at minimum, a value for key 'coord'.
        
        Returns
        -------
        None.

        """
        if self.state == Mode.HUNT:
            print("Attempting to set state to Hunt while in Hunt mode. "
                  "No change.")
        else:
            self.state = Mode.HUNT
            self.first_hit = outcome['coord']
            
        
    def set_to_kill(self, outcome):
        """
        Sets the Hunter to hunt mode. Throws a warning if already in hunt mode.

        Parameters
        ----------
        outcome : dict
            An outcome dict that contains, at minimum, a value for key 'coord'.
            
        Returns
        -------
        None.

        """
        if self.state == Mode.KILL:
            print("Attempting to set state to Kill while in Kill mode. "
                  "No change.")
        else:
            self.state = Mode.KILL
            self.first_hit = outcome['coord']
            
    
    ### Class Methods ###
    
    @classmethod
    def parse_hunt_options(cls, 
                           hunt_style,
                           hunt_pattern,
                           options):
        """
        Translates input options dict into a standard format, and adds default
        values to the options dict.

        Parameters
        ----------
        hunt_style : str
            The desired hunt_style
        hunt_pattern : str
            The desired hunt_pattern.
        options : dict
            Dictionary of option key/value pairs applicable to hunt pattern
            generation. See Returns for allowable keys/values.
            Note that any keys beyond those noted below will also be included
            in the returned options dict.
            

        Returns
        -------
        options : dict.
            Dictionary with the following keys/value options:
               
        'edge_buffer': (int) Do not target spaces within this many 
                        spaces of the edge of the board. Default is 0.
        'ship_buffer': (int) Do not target spaces within this many 
                        spaces of a known ship. Default is 0.
        'weight' : (str) 'hits' 'misses', 'shots'/'any'
                        'weight' is applicable to the 'random' 
                        hunt_style only.
                        
                        If 'hits', weighting will occur based on
                        a space's distance to all hits (i.e.,
                        spaces that are far from all hits will be 
                        likely for 'clustered' hunt_pattern, and 
                        unlikely for 'isolated' hunt_pattern).
                        If 'misses', weighting will occur based on
                        a space's distance to all missed shots.
                        If 'shots' or 'any', weighting will occur
                        based on a space's distance to all shots
                        taken, regardless of whether a shot is a 
                        hit or a miss.
        'rotate' : (int) 0, +/-90, +/-180, 270, 360
                        'pattern' hunt_stlye only.
                        The degrees to rotate a pattern relative to 
                        a starting position of coordinate A1.
        'mirror' : (str) 'horizontal' or 'vertical'
                        'pattern' hunt_style only.
                        If 'horizontal', the pattern will be flipped
                        across a horizontal (row).
                        If 'vertical', it will be flipped across a 
                        vertical (column).
        'spacing' : (int) 'pattern' hunt_style only.
                        The separation between subsequent targets
                        in a pattern. 
        'secondary_spacing' : (int) 'pattern' hunt_style only.
                        Some hunt_pattern types accept a spacing
                        along a second direction.
                        For 'diagonal', This is the horizontal 
                        separation between each diagonal line.
                        For 'grid', it is the horizontal separation 
                        between each vertical line.
                        For 'spiral', it has no effect.
        'no_valid_targets' : (str) 'random' or 'ordered'
                        Sets the Offense's behavior once the pattern
                        list has no more target coordinates.
                        If 'random', a target will be selected randomly
                        from the untargeted spaces (still weighted
                        according to hunt_pattern, if applicable). 
                        If 'ordered', the target will be selected in 
                        ascending order from the untargeted spaces,
                        moving from column to column along a single 
                        row, then on to the next row.
        'dist_method' : (str) 'dist2' (default), 'dist', or 'manhattan'
                        Sets the method used to calculate distances between
                        a coordinate and ships. See Placement and Coord 
                        dist_to methods for specifics.
                        
        """
        new_options = {
            'edge_buffer': 0,
            'ship_buffer': 0,
            'weight': 'any',
            'rotate': 0,
            'mirror': False,
            'spacing': 0,
            'secondary_spacing': 0,
            'no_valid_targets': 'random',
            'dist_method': 'dist2',
            'user_data': None,
            'note': None,
            }
        for k in new_options:
            if k not in options:
                options[k] = new_options[k]
            if isinstance(options[k], str):
                options[k] = options[k].lower()
        return options
    
    @classmethod
    def parse_kill_options(cls, options):
        """
        Translates input options dict into a standard format, and adds default
        values to the options dict.

        Parameters
        ----------
        options : dict
            Dictionary of option key/value pairs applicable to kill target
            selection. See Returns for allowable keys/values.
            Note that any keys beyond those noted below will also be included
            in the returned options dict.
            

        Returns
        -------
        options : dict.
            A dictionary containing the following key/value pairs:
                'method' : (str) 'dumb', 'basic' or 'advanced'
                                'basic' moves along rows/columns of hits
                                to guess where the next one might be
                                'advanced' takes possible placements of every
                                remaining ship into account to generate a 
                                probability around the most recent hit.

        """
        new_options = {
            'method': 'advanced',
            'user_data': None,
            'note': None,
            }
        for k in new_options:
            if k not in options:
                options[k] = new_options[k]
            if isinstance(options[k], str):
                options[k] = options[k].lower()
        return options