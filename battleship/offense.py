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
#from collections import Counter

# Imports from this package
from battleship import constants
from battleship import utils

from battleship.coord import Coord
from battleship.ship import Ship
#from battleship.placement import Placement


#%%
### Enums ###

class Mode(Enum):
    HUNT = 0
    KILL = 1
    
    def __eq__(self, other):
        return 
    

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
    
    
    @abc.abstractmethod
    def update_state(self, outcome):
        """
        Updates any instance variables based on the outcome dict of the most
        recent turn taken by the player who owns the offense.

        Parameters
        ----------
        outcome : dict
            An outcome dictionary.

        Returns
        -------
        None.

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
    @abc.abstractmethod
    def reset(self):
        """
        Resets any properties that are common to all Offense subclasses.
        
        Currently does not take any action.

        Returns
        -------
        None.

        """
        pass
    
    # Static Utility Methods
    
    @staticmethod
    def normalized_probs(targets_with_probs):
        """
        Normalizes the probabilities in the input dictionary so that the 
        sum of all probabilities associated with all possible targets is 1.

        Parameters
        ----------
        targets_with_probs : dict
            A dictionary with length N. Each key/value pair is a coordinate
            tuple (key) and the probability that coordinate should be targeted
            (value).

        Returns
        -------
        targets_with_normalized_probs : list
            Dictionary with the same keys as the input targets_with_probs, 
            and probability values scaled such that the sum of the 
            probabilities is 1. Namely,
            sum(.

        """
        targets_with_normalized_probs = targets_with_probs.copy()
        
        c = sum([x for x in targets_with_probs.values()])
        for k in targets_with_normalized_probs.keys():
            targets_with_normalized_probs[k] *= (1./c)
            
        return targets_with_normalized_probs
    
        
###################################
### CONCRETE OFFENSE SUBCLASSES ###
###################################

class RandomOffense(Offense):
    
    def __init__(self):
        """
        Initializes an offense that fires at random untargeted spaces.

        Returns
        -------
        None.

        """
        super().__init__()
        
        self._state = Mode.HUNT
        self._initial_hit = None  
        self._target_probs = None
        
        #*** Currently for debug only
        self._board = None
        self._outcome_history = None
        
    def __repr__(self):
        return ("RandomOffense()")
                
    def __str__(self):
        return (f"{repr(self)}")
    
    def reset(self):
        """
        Resets the offense's outcome_history (if applicable) property.

        Returns
        -------
        None.

        """
        super().reset()
        if self.outcome_history:
            self.outcome_history = None
            
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
        
    @property
    def target_probs(self):
        return self._target_probs
       
        
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
        
        targets = board.all_targets(untargeted=True)
        probs = np.ones(len(targets)) / len(targets)
        self._target_probs = self.normalized_probs(dict(zip(targets, probs)))
        target = utils.random_choice(targets, probs)
        return target
    
    
    def update_state(self, outcome):
        """
        Updates the RandomOffense state; since this offense chooses targets
        randomly, no state variables are required; this method does nothing.

        Parameters
        ----------
        outcome : dict
            Outcome dictionary.

        Returns
        -------
        None.

        """
        pass
    
        
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
                                hunt_style only, and then only when
                                hunt_pattern is 'clustered' or 'isolated'
                                
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
            else:
                hunt_pattern = "random"
        hunt_pattern = hunt_pattern.lower()
        if hunt_pattern == "isolate":
            hunt_pattern = "isolated"
        elif hunt_pattern == "cluster":
            hunt_pattern = "clustered"
        self._hunt_pattern = hunt_pattern
        self._original_hunt_options = hunt_options.copy()
        self._original_kill_options = kill_options.copy()
        self._hunt_options = HunterOffense.parse_hunt_options(hunt_style,
                                                              hunt_pattern,
                                                              hunt_options)
        self._kill_options = HunterOffense.parse_kill_options(kill_options)
        
        #self._state = Offense.Mode.HUNT
        self._initial_hit = None  
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
                f"  initial_hit: {self.initial_hit}"
                )
    
    def reset(self):
        """
        Resets the hunter's state, initial_hit, and outcome_history (if 
        applicable) properties.

        Returns
        -------
        None.

        """
        super().reset()
        temp_offense = HunterOffense(self._hunt_style)
        #self._state = temp_offense.state
        #self._first_initial_hit = temp_offense.initial_hit
        self._outcome_history = temp_offense.outcome_history
        
        
    ### Properties ###
    
    # @property
    # def state(self):
    #     return self._state
    
    # @state.setter
    # def state(self, value):
    #     if value is Mode.HUNT or value is Mode.KILL:
    #         self._state = value
    #     else:
    #         raise ValueError("state must be a valid Mode value.")
            
    # @property
    # def initial_hit(self):
    #     return self._initial_hit
    
    # @initial_hit.setter
    # def initial_hit(self, value):
    #     if value == None:
    #         self._initial_hit = None
    #     elif isinstance(value, (tuple, Coord)):
    #         self._initial_hit = value
    #     else:
    #         raise TypeError("initial_hit coord needs to be a tuple or Coord.")
            
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
        
    @property
    def target_probs(self):
        return self._target_probs
    
    
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
            
        open_target = self.last_open_hit(board, outcome_history)
        if open_target:
            self._target_probs = self.kill_targets(board, outcome_history)
        else:
            self._target_probs = self.hunt_targets(board, outcome_history)
        
        # As a final failsafe, if not targets have been selected return one
        # selected with uniform probability from the untargeted spaces.
        if self._target_probs:
            targets, probs = zip(*self._target_probs.items())
        else:
            targets = board.all_targets(untargeted=True)
            probs = np.ones(len(targets)) / len(targets)
            Warning("No targets were identified; choosing one randomly.")
        target = utils.random_choice(targets, probs)
        return target
    
    
    def update_state(self, outcome):
        """
        Updates the HunterOffense state; since this offense chooses targets
        based on history, no state variables are required; 
        this method does nothing.

        Parameters
        ----------
        outcome : dict
            Outcome dictionary.

        Returns
        -------
        None.

        """
        pass
    
    
    ### Hunt Methods ###
    
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
            
        # Get target probabilities based on pattern:
        if self._hunt_style == "random":
            targets_with_probs = self.random_hunt_targets(board, outcome_history)
        elif self._hunt_style == "pattern":
            targets_with_probs = self.pattern_hunt_targets(board, outcome_history)
        else:
            raise ValueError(f"{self._hunt_style} is not a valid value for "
                             "the hunt_style parameter.")
            
        return self.normalized_probs(targets_with_probs)
    
    
    # Pattern Hunt Methods
    
    def pattern_hunt_targets(self, board=None, outcome_history=None):
        """
        Returns target coordinates for the next coordinate along the hunt
        pattern in hunt mode.

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

            Since the target is chosen from a list of coordinates that form
            the hunt pattern, the targets_with_probs dict contains a single 
            coordinate with probability 1.
        """
        raise NotImplementedError("This method is not yet implemented.")
        
        
    # Random Hunt Methods
    
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
    
    
    ### Kill Methods ###
    
    def kill_targets(self, board=None, outcome_history=None):
        
        #** For debug purposes only **
        if board == None:
            board = self.board
        if outcome_history == None:
            outcome_history = self.outcome_history
        # *** ^^^^^^^^ ***
        
        if self.kill_options['method'] == 'dumb':
            targets_with_probs = self.kill_targets_around_last_hit(board,
                                                            outcome_history)
        elif (self.kill_options['method'] == 'advanced' ):
            targets_with_probs = self.kill_targets_from_placements(board, 
                                                            outcome_history,
                                                            True,
                                                            True)
        else: # method == 'basic'
            targets_with_probs = self.kill_targets_from_shape(board, 
                                                              outcome_history)
            
        # If no kill targets are found, revert to HUNT mode and get a hunt
        # target.
        if not targets_with_probs:
            self.set_to_hunt(None)
            targets_with_probs = self.hunt_targets(board, outcome_history)
            
        return self.normalized_probs(targets_with_probs)
    
    
    # Dumb Method
    
    def kill_targets_around_last_hit(self, board=None, outcome_history=None):
        """
        Returns a target around the last successful hit that has untargeted
        spaces around it.

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
            
        # Go backward through hits in the history and find adjacent untargeted
        # coords.
        open_hits = self.open_hits(board, outcome_history)
        targets = []
        for hit in reversed(open_hits):
            targets = board.targets_around(hit, 
                                           values=constants.TargetValue.UNKNOWN)
            if targets:
                break
            
        if targets:
             targets_with_probs = dict(zip(targets, 
                                           [1./len(targets)] * len(targets)))
        else:
            targets_with_probs = {}
        return targets_with_probs
    
    
    # Advanced Method
    
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
        
        
        last_open_hit = self.last_open_hit()
        if last_open_hit == None:
            raise ValueError("last_open_hit is None; "
                             "should not be in kill methods.")
        elif board.target_grid[last_open_hit] < constants.TargetValue.HIT:
            raise ValueError("last_open_hit is not a hit.")
            
        sunk_ships = [outcome['sunk_ship_type'] for outcome in outcome_history
                      if outcome['sunk_ship_type'] is not None]
        afloat_ships = [i for i in range(1,len(Ship.data)+1) 
                        if not i in sunk_ships]
        
        # find placements that overlap the last_hit
        placements = board.placements_containing_coord(last_open_hit, 
                                                       ship_types=afloat_ships)
        # Eliminate placements that do not contain only unknowns 
        # and unresolved hits
        placements = [p for p in placements 
                      if board.is_valid_target_ship_placement(p)]
        
        # find all untargeted coords that are in the valid placements
        coords = []
        for p in placements:
            valid_coords = [coord for coord in p.coords() 
                            if (board.target_grid[coord] 
                                == constants.TargetValue.UNKNOWN)]
            coords.extend(valid_coords)
        coords = list(set(coords))
        
        # if variable probability is not required, just return the coords 
        # with uniform probability
        if not variable_prob:
            return dict(zip(coords, [1/len(coords)] * len(coords)))
        
        # score each coord by the number of placements it shows up in, scaled
        # by the hits in that placement (or the damage % of the ship in that
        # placement)
        scores = {}
        for p in placements:
            if set(p.coords()).intersection(coords):
                # number of hits on the placement
                nhits = sum([(board.target_grid[rc] == constants.TargetValue.HIT) 
                             for rc in p.coords()])
                # percent of the placement that is damaged
                damage_percent = nhits / p.length
                if damage_percent == 1:     # A ship of this length would have been
                    damage_percent = 0.     #  sunk already with 100% damage.
                    
                # number of afloat ship types with length equal to placement length
                n_possible_afloats = sum([Ship(t).length == p.length 
                                          for t in afloat_ships])
                scores[p] = damage_percent * n_possible_afloats
            else:
                pass    # only score the placement p if it contains at least
                        # one valid coord.
                        
        targets_with_probs = {}
        for coord in coords:
            score = 0.
            for p in placements:
                if coord in p.coords():
                    score += scores[p]
            targets_with_probs[coord] = np.round(score, 3)
            
        # return only the coords with the highest score (multiples if there
        # is a tie)
        max_score = max(targets_with_probs.values())
        coords = [coord for coord in coords 
                  if targets_with_probs[coord] == max_score]
        return dict(zip(coords, [1/len(coords)] * len(coords)))
    

    # Basic Method
    
    def kill_targets_from_shape(self, 
                                board=None, 
                                outcome_history=None):
        
        #** For debug purposes only **
        if board == None:
            board = self.board
        if outcome_history == None:
            outcome_history = self.outcome_history
        # *** ^^^^^^^^ ***
        
        open_hits = self.open_hits(board, outcome_history)
        if not open_hits:
            raise ValueError("last_open_hit is empty; "
                             "should not be in kill methods.")
            
        # If there is only one open hit, find targets around it.
        if len(open_hits) == 1:
            targets = board.targets_around(open_hits[-1], untargeted=True)
        
        # Otherwise, look for the most recent open hit that has an adjacent
        # open hit.
        else:
            for hit in reversed(open_hits):
                hit = Coord(hit)
                adjacent_hit = None
                for other_hit in reversed(open_hits):
                    if hit.next_to(other_hit):
                        adjacent_hit = other_hit
                        break
                if adjacent_hit:
                    break
            
            # If two adjacent open hits were found, find targets at the ends
            # of the line defined by the hits.
            if adjacent_hit:
                targets = self.targets_along(adjacent_hit, 
                                             tuple(hit),    # Coord does not play nice with target_grid (np.array) 
                                             board, 
                                             outcome_history)
                # Keep the target closer to the more recent hit, if more
                # than one was located
                if len(targets) == 2:
                    targets = [targets[-1]]
                
            # If two adjacent hits were NOT found, find an arbitrary target
            # around any of the open hits (prefer the more recent ones).
            else:
                for hit in reversed(open_hits):
                    targets = board.targets_around(hit, untargeted=True)
                    if targets:
                        break
                
        if targets:
             targets_with_probs = dict(zip(targets, 
                                           [1./len(targets)] * len(targets)))
        else:
            targets_with_probs = {}
        return targets_with_probs
    
    def targets_along(self, 
                      first_hit, 
                      second_hit, 
                      board=None, 
                      outcome_history=None):
        """
        Returns a list of targets at the ends of the line defined by first_hit
        and second_hit. In this case, a "line" is a series of adjacent hits
        on unknown ship types. 
        
        The targets will be a list of 0, 1, or 2 coordinates that are at the
        end of the line defined by the hits and have unknown target grid 
        values (that is, they have not been targeted yet). If the first spaces
        outside of the line are misses or known hits, they will not be 
        included in the targets list. If the line has no possible targets at
        its ends, and empty list will be returned.
        
        If the line has targets at both ends, the first one will be the one
        closer to first_hit.
        

        Parameters
        ----------
        first_hit : tuple
            Row/col tuple (or Coord) of the first hit in the line.
        second_hit : tuple
            Row/col tuple (or Coord) of the second hit in the line.
        board : TYPE, optional
            DESCRIPTION. The default is None.
        outcome_history : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        targets : list
            A list with zero, one, or two coordinate tuples. Each tuple is 
            an untargeted coordinate at the end of the line defined by the
            two input hit coordinates.

        """
        
        #** For debug purposes only **
        if board == None:
            board = self.board
        if outcome_history == None:
            outcome_history = self.outcome_history
        # *** ^^^^^^^^ ***
    
        vector = np.array((second_hit[0] - first_hit[0], 
                           second_hit[1] - first_hit[1]))
        
        #assert board.target_grid[first_hit] == constants.TargetValue.HIT
        assert board.target_grid[second_hit] == constants.TargetValue.HIT
        assert vector[0] == 0 or vector[1] == 0
        
        if vector[0] == 0:
            line = board.target_grid[second_hit[0],:]
            # i1 = first_hit[1]
            i2 = second_hit[1]
        else:
            line = board.target_grid[:, second_hit[1]]
            # i1 = first_hit[0]
            i2 = second_hit[0]
        
        targets = []
        for idx in range(i2 + 1, board.size):
            if line[idx] != constants.TargetValue.HIT:
                if line[idx] == constants.TargetValue.UNKNOWN:
                    targets += [idx]
                break
            
        for idx in range(i2 - 1, -1, -1):
            if line[idx] != constants.TargetValue.HIT:
                if line[idx] == constants.TargetValue.UNKNOWN:
                    targets = [idx] + targets
                break
        
        if vector[0] == 0:
            targets = [(second_hit[0], t) for t in targets]
        else:
            targets = [(t, second_hit[1]) for t in targets]
            
        if sum(vector) < 0 and len(targets) == 2:
            targets = [targets[1], targets[0]]
            
        return targets
        
    ### Other Methods (Computated Properties) ###
    
    def open_hits(self, board=None, outcome_history=None):
        """
        Returns the list of hits on the board's target grid that have not
        been assigned to a particular ship type. The list is ordered such
        that the most recent hit is last.

        Returns
        -------
        hits : list
            A list of target coordinates (tuples) for hits that cannot be
            assigned to a particular type of ship.

        """
        #** For debug purposes only **
        if board == None:
            board = self.board
        if outcome_history == None:
            outcome_history = self.outcome_history
        # *** ^^^^^^^^ ***
        open_hits = [outcome['coord'] for outcome in reversed(outcome_history)
                     if (board.target_grid[outcome['coord']] 
                         == constants.TargetValue.HIT)]
        return open_hits
    
    
    def last_open_hit(self, board=None, outcome_history=None):
        """
        Returns a tuple with the coordinates of the most recent open hit--the
        hit that cannot be assigned to a particular ship type.
        
        If there are no open hits on the board, None is returned.

        Parameters
        ----------
        board : Board, optional
            The board on which targets are recorded. The default is None,
            in which case the Hunter's board property is used (if it exists).
        outcome_history : list, optional
            The history list in which outcomes are recorded. The default is 
            None, in which case the Hunter's outcome_history property is used 
            (if it exists).

        Returns
        -------
        hit : tuple (row,col)
            The (row,col) of the most recent open hit, or None if no such
            hit exists.

        """
        #** For debug purposes only **
        if board == None:
            board = self.board
        if outcome_history == None:
            outcome_history = self.outcome_history
        # *** ^^^^^^^^ ***
        
        last_open_hit = None
        for outcome in reversed(outcome_history):
            if (outcome['hit'] 
                and (board.target_grid[outcome['coord']] 
                     == constants.TargetValue.HIT)): 
                last_open_hit = outcome['coord']
                break
        return last_open_hit
    
    
    def last_target(self, outcome_history=None):
        """
        Returns the coordinate of the last targeted spot on the board.

        Parameters
        ----------
        outcome_history : list, optional
            The history list in which outcomes are recorded. The default is 
            None, in which case the Hunter's outcome_history property is used 
            (if it exists).

        Returns
        -------
        target : tuple (row,col)
            The (row,col) of the most recent targeted coordinate.

        """
        #** For debug purposes only **
        if outcome_history == None:
            outcome_history = self.outcome_history
        # *** ^^^^^^^^ ***
        if outcome_history:
            target = outcome_history[-1]['coord']
        else:
            target = None
        return target
    
    
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