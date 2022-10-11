#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 23:21:42 2022

@author: jason
"""

from battleship import (Coord, Board, random_choice, mirror_coords, 
                        rotate_coords, TargetValue)
import offense
import abc
import numpy as np
from enum import Enum
from matplotlib import pyplot as plt

#%% Constants

DEFAULT_GRID_STEP_SIZE = 2
DEFAULT_GRID_OFFSET = 1
DEFAULT_GRID_DIRECTION = 1
DEFAULT_GRID_STARTING_CORNER = 'NW'
DEFAULT_GRID_SKIP = 0
DEFAULT_GRID_EDGE_BUFFER = 0
DEFAULT_PATTERN_ALTERNATE = True
DEFAULT_RANDOM_WEIGHT = "maxprob"

#%% Offense Abstract Base Class
### Must be subclassed in order to make instances

class Offense(metaclass=abc.ABCMeta):
    
    ### Initializer ###
    
    def __init__(self):
        """
        Initializes attributes of the abstract class. Concrete subclasses
        may also need initialization.

        Parameters
        ----------
        None (see Subclasses)

        Returns
        -------
        None.

        """

    def __repr__(self):
        return "Offense()"
    
    ### Abstract Methods ###
    
    @abc.abstractmethod
    def target(self, board, history):
        """
        Returns a target location based on the input board, history, offense 
        method and state.

        Parameters
        ----------
        board : Board
            Instance of Board class.
        history : list
            List of outcome dictionaries.

        Returns
        -------
        An instance of Coord.

        """
        pass
    
    @abc.abstractmethod
    def update(self, outcome):
        """
        Updates any attributes that the Offense needs to keep track of its
        state. Specific updates depends on the details of concrete subclasses.

        Parameters
        ----------
        outcome : dict
            Dictionary containing items that describe the results of a shot.
            Keys and value types should include:
                - hit: Bool
                - coord: two-element tuple
                - sunk: Bool
                - sunk_ship_id: Int
                - message: string

        Returns
        -------
        None.

        """
        pass
    
    @abc.abstractmethod
    def reset(self):
        """
        Puts the offense back into its initial state. This typically will
        involve setting any state variables back to initial values, and 
        resetting any histories or logs that track targets and outcomes.

        Returns
        -------
        None.

        """
        pass
        
    # @abc.abstractmethod
    # def __str__(self):
    #     """
    #     String describing the Offense.

    #     Returns
    #     -------
    #     String.

    #     """
    #     pass
    
#%% 

class HunterOffense(Offense):
    
    class Mode(Enum):
        HUNT = 1
        KILL = 2
        BARRAGE = 3
        
    def __init__(self, pattern, board=None, **kwargs):
        """
        Initializes attributes and state of a HunterOffense instance.

        Parameters
        ----------
        pattern : string
            String describing the pattern the offense will use to search for
            targets to sink. The pattern should have 1-3 components separated by 
            '-' or '.':
                SEARCH_PATTERN - SINK - 
            The SEARCH component can be:
                - Random: Pick a random untargeted space.
                - Grid: Fire on a periodic grid with period and direction 
                    specified by a 'step' parameter.
                - Spiral: Fire on a spiral starting either from a corner or 
                    the center. The start location can be set explicitly
                    by setting the 'start_at' parameter to a tuple or coordinate
                    string (A1), or with the strings 'corner' or 'center'. 
                    The space between shots is set by the 'step' argument. 
                    The direction of the spiral is set by the 'direction' 
                    parameter, and can be 'cw'/'clockwise' or 'ccw'/'counterclockwise'.
                - Open: Fire at spots that are farthest from already targeted
                    spots. Can either fire at the most isolated location,
                    or randomly weighted by isolation. 
                - Diagonals: like grid, but at 45 degrees. Usually start at 
                    the longest diagonal, then skip+1 spaces above the diagonal,
                    skip+1 below the diagonal, etc.
                - Center: focus more shots near the center of the grid and 
                    fewer near the edges. (this could be a keyword for Random).
                    
            The SINK component can be:
                - 'SinkFirst'
                - 'SearchFirst': If a hit is scored, keep searching until the
                    search method has reached the end of the board, then go
                    back to the hits and try to sink the ships. 
                    
            Once the search has reached the end of the board, 
            The default value is "random".
            
        board : Board, optional
            If a non-standard Board is to be used by this Offense, it can be 
            specified using the optional 'board' argument.
            A standard board is 10 x 10 spaces with 5 ships.
            
        **kwargs : Key/value pairs.
            Valid keys and types are:
                - step: int or tuple (valid for Grid and Spiral methods)
                - smart: Bool. When sinking a ship, should the player look for
                    two hits in a row (smart == True), or just randomly shoot
                    adjacent to each hit (smart == False)? The latter has the 
                    benefit of possibly uncovering adjacent ships and does not
                    risk confusion if adjacent hits are on different ships,
                    but is generally less efficient at sinking a located ship.
                - start_at: int or tuple (valid for Grid and Spiral methods). 
                    Sets the starting point of a grid or spiral pattern 
                    relative to the A1 corner (not valid for a center start
                    spiral).
                - wrap: str (valid for Grid). How to target after a row has 
                    been completed. Options are:
                    - align
                    - offset
                    - wrap
                - barrage: Int. At each search location, fire this many
                    shots before moving on to the next location.
                - barrage_pattern: str. How to select the cluster of barrage
                    targets at each search location. Options are:
                    - 'cross': 4 shots directly adjacent to each search location.
                    - 'random': shots randomly selected from the 8 spots
                        adjacent and diagonal to the search location.
                    - 'X': 4 shots diagonal to the search location.
                    - 'tight': shots placed on a spiral around the search location.
                        Clockwise unless the 'direction' parameter is also 
                        provided, and set to 'ccw'.
                - repeat: Bool (valid for Grid, Spiral). If True, once the 
                    search method has finished searching the board, it will 
                    be repeated with an offset (Grid) or a reversed spiral
                    (Spiral). For Grid, repeat will continue until the 
                    grid overlaps a previous grid spot. For spiral, only one
                    reverse spiral will be performed. After that, searching
                    resorts to the behavior described by last_resort.
                - last_resort: str (valid for Grid, Spiral). Once the pattern
                    has been exhaused, targets will be chosen randomly if
                    last_resort is 'random' (default) or by going one at a 
                    time if last_resort is 'grid'

        Returns
        -------
        None.

        """
        
        super().__init__()
        
        board = Board() if board == None else board
        
        self._hunt_method = pattern
        self._weight = (kwargs['weight'] if 'weight' in kwargs else None)
        self._target_pattern = self.generate_pattern(pattern, board, **kwargs)
        self._smart = True
        self.params = kwargs
        
        # state variables
        self._state = self.Mode.HUNT
        self.barrage_count = 0
        
        self._outcome_history = []
        self._possible_targets = None
        self._target_probabilities = None
        
        self._barrage = (kwargs['barrage'] if 'barrage' in kwargs else 0)
        self._barrage_method = (kwargs['barrage_method'] if '_barrage_method' 
                                in kwargs else None)
        if self.barrage > 0 and self.barrage_method == None:
            self.barrage_method = "random"
    
    def __repr__(self):
        r = f"HunterOffense({self.hunt_method!r}" #"{self.board}"
        if self.weight:
            r += f", weight={self.weight!r}"
        if self.barrage:
            r += f", barrage={self.barrage!r}"
        if self.barrage_method:
            r += f", barrage_method={self.barrage_method!r}"
        if self.params.get("edge_buffer"):
            r += f", edge_buffer={self.params.get('edge_buffer')!r}"
        if self.params.get("starting_corner"):
            r += f", starting_corner={self.params.get('starting_corner')!r}"
        if self.params.get("step_size"):
            r += f", step_size={self.params.get('step_size')!r}"
        if self.params.get("skip"):
            r += f", skip={self.params.get('skip')!r}"
        if self.params.get("offset"):
            r += f", offset={self.params.get('offset')!r}"
        if self.params.get("alternate"):
            r += f", alternate={self.params.get('alternate')!r}"
        if self.params.get("reverse_lines"):
            r += f", reverse_lines={self.params.get('reverse_lines')!r}"
        if self.params.get("start_at"):
            r += f", start_at={self.params.get('start_at')!r}"
        if self.params.get("expand"):
            r += f", expand={self.params.get('expand')!r}"
        r += ")"
        
        return r
    
    def __str__(self):
        """
        String describing the Offense.

        Returns
        -------
        String.

        """
        s = (f"Hunt method: {self.hunt_method!r}\n"
             f"Weighting: {self._weight!r}\n"
             f"Current state: {self.state}\n"
             f"Smart: {self.smart}\n")
        if self.barrage:
            s += (f"Barrage: {self.barrage}\n"
                  f"Barrage method: {self.barrage_method}\n"
                  f"Current barrage count: {self.barrage_count}\n")
        s += (f"Potential targets: {self.possible_targets}\n")
              #f"Potential target probabilities: {self.target_probabilities}\n")
        return s
        
    
    # Properties
    @property
    def hunt_method(self):
        return self._hunt_method
    @hunt_method.setter
    def hunt_method(self, value):
        self._hunt_method = value
        
    @property
    def target_pattern(self):
        return self._target_pattern
    
    @property
    def smart(self):
        return self._smart
    
    @smart.setter
    def smart(self, value):
        self._smart = (value == True)
        
    @property
    def weight(self):
        return self._weight
    @weight.setter
    def weight(self, value):
        self._weight = value
        
    @property
    def state(self):
        return self._state
    @state.setter
    def state(self, value):
        if value == self.Mode.HUNT or value == self.Mode.KILL:
            self._state = value
        else:
            raise ValueError(f"Value must be {self.Mode.HUNT} or "
                             f"{self.Mode.KILL}.")

    @property
    def outcome_history(self):
        return self._outcome_history
    def add_to_outcome_history(self, outcome):
        self._outcome_history.append(outcome)
     
    @property
    def barrage_count(self):
        return self._barrage_count
    @barrage_count.setter
    def barrage_count(self, value):
        if value < 0:
            raise ValueError("barrage_count cannot be less than 0.")
        self._barrage_count = value
        
    @property
    def barrage(self):
        return self._barrage
    @barrage.setter
    def barrage(self, value):
        if value < 0:
            raise ValueError("barrage cannot be less than 0.")
        self._barrage = value
        
    @property
    def barrage_method(self):
        return self._barrage_method
    @barrage_method.setter
    def barrage_method(self, value):
        self._barrage_method = value
       
    @property 
    def possible_targets(self):
        return self._possible_targets
    @possible_targets.setter
    def possible_targets(self, value):
        self._possible_targets = value
        
    ### Computed Attributes ### 
    
    @property
    def last_hit(self):
        hit = None
        if self._outcome_history:
            for out in self._outcome_history[::-1]:
                if out["hit"]:
                    hit = out["coord"]
                    break
        return hit
    
    @property
    def last_miss(self):
        hit = None
        if self._outcome_history:
            for out in self._outcome_history[::-1]:
                if not out["hit"]:
                    hit = out["coord"]
                    break
        return hit
    
    @property
    def last_sink(self):
        hit = None
        if self._outcome_history:
            for out in self._outcome_history[::-1]:
                if out["sunk_ship_id"] != None:
                    hit = out["coord"]
                    break
        return hit
        
    @property  
    def last_target(self):
        if self._outcome_history:
            return self._outcome_history[-1]["coord"]
        else:
            return None
        
    @property
    def last_pattern_target(self):
        target = None
        if self.hunt_method == "random":
            target = self.last_target
        elif self._outcome_history:
            for out in self._outcome_history[::-1]:
                if out["coord"] in self.target_pattern:
                    target = out["coord"]
                    break
        return target
    
    @property
    def target_history(self):
        return [outcome["coord"] for outcome in self.outcome_history]
    
    @property
    def hits_since_last_sink(self):
        shot_did_sink = np.array([outcome["sunk"] for outcome in 
                                  self.outcome_history])        
        idx_last_sink = np.where(shot_did_sink)[0]
        if len(idx_last_sink) == 0:
            idx = 0
        else:
            idx = idx_last_sink[0] + 1
        hits = [outcome["coord"] for outcome in self.outcome_history[idx:] if
                outcome["hit"] == True]
        return hits
        
    @property
    def barrage_target_history(self):
        """List of coords that were targeted in the current barrage (i.e.,
        since the turn when the state was changed to BARRAGE).
        """
        targets = []
        for outcome in self.outcome_history[::-1]:
            if outcome["state"] == self.Mode.BARRAGE:
                targets.insert(0, outcome["coord"])
        return targets
    
    @property
    def hunt_history(self):
        """List of outcomes when the Offense has been in HUNT mode.
        """
        return [outcome for outcome in self.outcome_history if 
                outcome["state"] == self.Mode.HUNT]
        
    ### Abstract Methods ###
    
    def target(self, board, history):
        """
        Returns a target location based on the input board, history, offense 
        method and state.

        Parameters
        ----------
        board : Board
            Instance of Board class.
        history : list
            List of outcome dictionaries.

        Returns
        -------
        None.

        """
        if self.state == self.Mode.HUNT:
            targets, probs = self.hunt_targets(board, history)
        elif self.state == self.Mode.KILL:
            targets, probs = self.kill_targets(board, history)
        elif self.state == self.Mode.BARRAGE:
            targets, probs = self.barrage_targets(board, history)
        else:
            raise AttributeError("HunterOffense state is invalid.") 
        self.possible_targets = dict(zip(targets, probs))
        return random_choice(targets, probs)
        
    def update(self, outcome):
        """
        Updates any attributes that the Offense needs to keep track of its
        state. Specific updates depends on the details of concrete subclasses.

        Parameters
        ----------
        outcome : dict
            Dictionary containing items that describe the results of a shot.
            Keys and value types should include:
                - hit: Bool
                - coord: two-element tuple
                - sunk: Bool
                - sunk_ship_id: Int
                - message: string

        Returns
        -------
        None.

        """
        # last_target = outcome["coord"]
        outcome["mode"] = self.state
        self.add_to_outcome_history(outcome)
        if outcome["sunk_ship_id"]:
            self.state = self.Mode.HUNT
            self._hits_since_last_sink = []
            self.barrage_count = 0
        elif outcome["hit"]:
            self.state = self.Mode.KILL
            # self.add_to_hits_since_last_sink(last_target)
            self.barrage_count = 0
        else:
            if self.barrage:
                self.barrage_count += 1   
            # self.last_pattern_target = last_target
                  
        self.message = None 
    
    def reset(self):
        """
        Sets the state variable to initial value.

        Returns
        -------
        None.

        
        """
        self._state = self.Mode.HUNT
        self.barrage_count = 0
        
        self._outcome_history = []
        self._possible_targets = None
        self._target_probabilities = None
        
    
    def hunt_targets(self, board, history):
        """
        Determine potential targets and associated probabilities of choosing
        those targets while the Offense is in HUNT mode.
        
        Parameters
        ----------
        board : Instance of Board
            The board owned by the Offense's player.
        history : List
            A list of outcome dictionaries, with most recent outcome last.

        Returns
        -------
        targets : list
            A list of coordinates; potential hunt targets.
        probs : list
            A numpy array of probabilities. probs[k] is the probability that
            targets[k] should be chosen as the actual target.

        """
        # if open hits are present, shoot at the coordinate most likely to 
        # have a target
        probs = None
        open_hits = self.open_hits(board)
        if open_hits:
            targets = []
            for hit in open_hits:
               targets += board.coords_around(hit, untargeted=True, diagonal=False)
            probs = [np.sum(list(board.possible_targets_at_coord(t).values()))
                     for t in targets]
            probs = probs / np.sum(probs) if probs else np.ones(len(targets))
        # if no open hits, search according to hunt method
        elif self.hunt_method == "random":    
            targets, probs = self.pattern_random(board, self.weight)
        else:
            targets = self.next_step(self.last_pattern_target, self.pattern)
        if np.all(probs == None):
            probs = np.ones(len(targets)) / len(targets)
        #print(open_hits)
        #print(targets,probs)
        return targets, probs
    
    def kill_targets(self, board, history):
        """
        Determine a target coord while in kill mode (i.e., trying to sink a
        ship around the most recent hit). 
        
        Parameters
        ----------
        board : Instance of Board
            The board owned by the Offense's player.
        history : List
            A list of outcome dictionaries, with most recent outcome last.

        Returns
        -------
        targets : list
            A list of coordinates; potential hunt targets.
        probs : list
            A numpy array of probabilities. probs[k] is the probability that
            targets[k] should be chosen as the actual target.


        """
        hits = self.hits_since_last_sink
        self.message = "kill mode."
        if len(hits) == 1 or not self.smart:
            targets = board.coords_around(hits[-1], untargeted=True)
            self.message += " Searching around last hit."
        elif len(hits) > 1:
            targets = board.colinear_target_coords(hits, untargeted=True)
            self.message += " Searching along colinear hits."                    
        else:
            raise Exception("hits_since_last_sink should not be empty.")
        if not targets:
            self.state = self.Mode.HUNT
            self.message += " No viable target found; " \
                "reverting to hunt mode. "
            targets, probs = self.hunt_targets(board, history)
        else:
            probs = np.ones(len(targets)) / len(targets)
        return targets, probs
    
    def barrage_targets(self, board, history):
        """
        Return a list of potential targets for the current barrage. 

        Parameters
        ----------
        board : Instance of Board
            The board owned by the Offense's player.
        history : List
            A list of outcome dictionaries, with most recent outcome last.

        Returns
        -------
        targets : list
            A list of coordinates; potential hunt targets.
        probs : list
            A numpy array of probabilities. Barrage targets are either random
            with equal probability or sequential as determined by 
            barrage_method, so this array has all elements equal.

        """
        current_target = self.last_pattern_target
        method = self.barrage_method.lower()
        
        if method == "x":
            targets = board.coords_around(current_target, diagonal=True, 
                                          untargeted=True)
            exclude = board.coords_around(current_target, diagonal=False, 
                                          untargeted=True)
            targets = [t for t in targets if t not in exclude]
            
        elif method == "+" or method == "cross":
            targets = board.coords_around(current_target, untargeted=True)
            
        elif isinstance(method, (int, np.integer)):
            targets = board.coords_around(current_target, diagonal=True, 
                                          untargeted=True)
        
        elif method == "tight":
            current_barrage = self.barrage_target_history
            if current_barrage:
                v = (0,1)
            elif len(current_barrage) == 1:
                v = (current_barrage[0][0] - current_target[0],
                     current_barrage[0][1] - current_target[1])
            else:
                v = (current_barrage[-1][0] - current_barrage[-2][0],
                     current_barrage[-1][1] - current_barrage[-2][1])
            target = (current_barrage[-1][0] + v[0], 
                      current_barrage[-1][1] + v[1])
            targets = [target]
        
        probs = np.ones(len(targets)) / len(targets)
        return targets, probs
    
    ### Other methods ###
    
    def open_hits(self, board):
        ## This needs work!!!
        """
        Returns any hits that cannot be attributed to a sunken ship.

        Parameters
        ----------
        board : Board instance
            The board with a target grid that tracks hits, misses, and sinks.

        Returns
        -------
        List of known hits that cannot be attributed to a ship that is 
        known to be sunk.

        """
        #open_hits = []
        rows,cols = np.where(board.target_grid == TargetValue.HIT)
        return list(zip(rows,cols))
        # hits = list(zip(rows,cols))
        # for hit in hits:
        #     if len(board.coords_around(hit, diagonal=False, untargeted=True)):
        #         open_hits += [hit]
        # return open_hits
        
    def sync_state_to_history(self, history):
        """
        Updates the state variables to be consistent with the turn outcomes
        in 'history'.

        Parameters
        ----------
        history : List
            List of outcome dictionaries.

        Returns
        -------
        None.

        """
        self.state = self.Mode.HUNT
        self.barrage_count = 0
        self._outcome_history = []
        
        for h in history:
            self.update(h)
            
        return self.outcome_history == history
    
    ### Pattern Generation Methods ###
    
    def generate_pattern(self, pattern_name, board, **kwargs):

        # eliminate already-used key/values in keyword arguments:
        for key in ['weight', 'barrage', 'barrage_method']:
            if key in kwargs:
                del(kwargs[key])
        if pattern_name == "random":
            # this includes patterns where weight = 'center', 'isolated', and numbers > 0.
            pattern = None #self.pattern_random(board, self.weight)
        elif pattern_name == "grid":
            pattern = self.pattern_grid(board.size, **kwargs)
        elif pattern_name == "spiral":
            pattern = self.pattern_spiral(board.size, **kwargs)
        elif pattern_name == "diagonals":
            pattern = self.pattern_diagonals(board.size, **kwargs)
        else:
            raise ValueError(f"Invalid pattern_name: {pattern_name}.\n"
                             f"  Must be one of: 'random', 'grid', 'spiral',"
                             f"'diagonals'.\n"
                             "Check that 'weight' parameter is not being used"
                             "as 'pattern_name'.")
        return pattern
    
    def pattern_random(self, board, weight=None, **kwargs):
        weight = weight.lower() if isinstance(weight, str) else weight
        if weight == None or weight == 0:    # equal weight to all untargeted coords
            weight = 'flat'
            
        if weight == 'flat':
            targets = board.all_coords(untargeted=True)
            probs = np.ones(len(targets)) / len(targets)
            
        elif isinstance(weight, (int, float)):
            # weigh spaces that are not adjacent to ships (or edge, if edge_buffer > 0)
            # weight times greater than those that are.
            targets = [Coord(t) for t in board.all_coords(untargeted=True)]
            probs = weight * np.ones(len(targets))
            edge_buffer = kwargs['edge_buffer'] if 'edge_buffer' in kwargs else 0
            for (i, target) in enumerate(targets):
                if (edge_buffer > 0 and 
                    (target[0] < edge_buffer or target[1] < edge_buffer or 
                     target[0] >= board.size - edge_buffer or 
                     target[1] >= board.size - edge_buffer)):
                        probs[i] = 1
                for ship in board.fleet_list():
                    if any([target.next_to(coord) for 
                            coord in board.coords_for_ship(ship)]):
                        probs[i] = 1
            
        elif weight == "isolated":
            # target coords that are far from those that have already been targeted.
            targets = board.all_coords(untargeted=True)
            targeted = np.array(board.all_coords(targeted=True))
            probs = np.ones(len(targets))
            if np.any(targeted):
                for (i,new_target) in enumerate(targets):
                    d2 = np.sum((np.tile(np.array(new_target), 
                                        (targeted.shape[0], 1)) - targeted)**2)
                    probs[i] = d2
            if np.all(probs == 0):
                probs = np.ones(len(probs))
            probs = probs / np.sum(probs)
            
        elif weight == "center":
            targets = board.all_coords(untargeted=True)
            r0 = c0 = (board.size - 1) / 2
            weight = [np.sqrt(1 / np.abs(r - r0) / np.abs(c - c0)) 
                      for (r,c) in targets]
            probs = np.array(weight) / np.sum(weight)
            
        elif weight == "maxprob":
            # target coords that have the maximum number of possible ships.
            targets = board.all_coords(untargeted=True)
            likelihood = board.possible_ships_grid()
            probs = [likelihood[coord[0], coord[1]] for coord in targets]
            probs = np.array(probs)
            probs = probs / np.sum(probs)
                
        else:
            raise ValueError("'weight' parameter must be a number or one" 
                             " of the following: 'center', 'isolated',"
                             " 'maxprob'.")
            
        return targets, probs
    
    def pattern_grid(self, board_size, direction=1, at_end=None, 
                     max_reps=None, **kwargs):
                     #starting_corner, step_size, skip, 
                 #edge_buffer, offset, reverse_lines):
        """
        
        
        Parameters
        ----------
        starting_corner : TYPE
            DESCRIPTION.
        step_size : TYPE
            DESCRIPTION.
        skip : TYPE
            DESCRIPTION.
        edge_buffer : TYPE
            DESCRIPTION.
        reverse_lines : TYPE
            DESCRIPTION.
        offset : TYPE
            DESCRIPTION.
        at_end : TYPE
            DESCRIPTION.
        max_reps : TYPE
            DESCRIPTION.
        board_size : TYPE
            DESCRIPTION.

        Returns
        -------
        G : TYPE
            DESCRIPTION.

        """
        
        # Set default parameters here:
        starting_corner = kwargs.get('starting_corner')
        step_size = kwargs.get('step_size')
        skip = kwargs.get('skip')
        edge_buffer = kwargs.get('edge_buffer')
        reverse_lines = kwargs.get('reverse_lines')
        offset = kwargs.get('offset')
        
        starting_corner = (DEFAULT_GRID_STARTING_CORNER if 
                           starting_corner == None else starting_corner)
        step_size = (DEFAULT_GRID_STEP_SIZE if step_size == None else step_size)
        skip = (DEFAULT_GRID_SKIP if skip == None else skip)
        edge_buffer = (DEFAULT_GRID_EDGE_BUFFER if 
                       edge_buffer == None else edge_buffer)
        reverse_lines = (False if reverse_lines == None else reverse_lines)
        offset = (DEFAULT_GRID_OFFSET if offset == None else offset)
        max_reps = (10 if max_reps == None else max_reps)
        
        G = HunterOffense.grid(board_size, starting_corner, step_size, skip, 
                               edge_buffer, reverse_lines, offset)
        end_count = 1
        ok = end_count < max_reps
        if at_end == 'restart' or at_end == 'repeat':
            while ok:
                rows = np.sort(np.unique(np.array(G)[:,0]))
                available_rows = [r for r in range(board_size) if r not in rows]
                if len(available_rows) < 1:
                    ok = False
                else:
                    target_rows = [(rows[i]+rows[i+1])/2 for i in range(len(rows)-1)]
                    idx = np.abs(available_rows - target_rows[0]).argmin()
                    cols = np.sort(np.unique(np.array(G)[:,1]))
                    new_corner = Coord((available_rows[idx], cols[0]))
                    new_offset = int(np.floor(np.mean(cols[:2])))
                    G += HunterOffense.grid(board_size, new_corner, step_size, 
                                            skip, edge_buffer, reverse_lines, 
                                            new_offset)
                    end_count += 1
                ok = ok * (end_count <= max_reps)
        elif at_end == 'reverse':
            while ok:
                Grot = rotate_coords(G, 180, board_size)
                rows = np.sort(np.unique(np.array(Grot)[:,0]))
                available_rows = [r for r in range(board_size) if r not in rows]
                if len(available_rows) < 1:
                    ok = False
                else:
                    target_rows = [(rows[i]+rows[i+1])/2 for i in range(len(rows)-1)]
                    idx = np.abs(available_rows - target_rows[0]).argmin()
                    cols = np.sort(np.unique(np.array(G)[:,1]))
                    new_corner = Coord((available_rows[idx], cols[0]))
                    new_offset = int(np.floor(np.mean(cols[:2])))
                    Gnew = HunterOffense.grid(board_size, new_corner, 
                                              step_size, skip, edge_buffer, 
                                              reverse_lines, new_offset)
                    G += rotate_coords(Gnew, 180, board_size)
                ok = ok * (end_count <= max_reps)
        
        return G
    
    def pattern_diagonals(self, board_size, step_vec, at_end, max_reps=None,
                          **kwargs):
        
        # Set default parameters here:
        starting_corner = kwargs.get('starting_corner')
        skip = kwargs.get('skip')
        edge_buffer = kwargs.get('edge_buffer')
        reverse_lines = kwargs.get('reverse_lines')
        alternate = kwargs.get('alternate')
        
        starting_corner = (DEFAULT_GRID_STARTING_CORNER if 
                           starting_corner == None else starting_corner)
        skip = (DEFAULT_GRID_SKIP if skip == None else skip)
        edge_buffer = (DEFAULT_GRID_EDGE_BUFFER if 
                       edge_buffer == None else edge_buffer)
        reverse_lines = (False if reverse_lines == None else reverse_lines)
        alternate = (DEFAULT_PATTERN_ALTERNATE if alternate == None else alternate)
        max_reps = (10 if max_reps == None else max_reps)
        
        if any(x < 0 for x in step_vec):
            raise ValueError("step_vec cannot have negative elements.")
            
        diagonals = HunterOffense.diagonals(board_size, step_vec, skip, 
                                            edge_buffer, starting_corner, 
                                            alternate, reverse_lines)
        pattern = diagonals
        end_count = 1
        ok = end_count < max_reps
        if at_end == 'restart' or at_end == 'repeat':
            if skip == 0:
                pass
            else:
                next_diag = [(p[0],p[1]+1) for p in diagonals]
                next_diag = [p for p in next_diag if 
                             HunterOffense.coord_ok(p, pattern, 
                                              board_size, edge_buffer)
                             ]
                if any([p in pattern for p in next_diag]):
                    ok = False
                else:
                    pattern += next_diag
                ok = ok * (end_count <= max_reps)
        elif at_end == 'reverse':
            if skip == 0:
                pass
            else:
                next_diag = [(p[0],p[1]+1) for p in diagonals]
                next_diag = [p for p in next_diag if 
                             HunterOffense.coord_ok(p, pattern, 
                                                    board_size, edge_buffer)
                             ]
                if any([p in pattern for p in next_diag]):
                    ok = False
                else:
                    pattern += rotate_coords(next_diag, 180, board_size)
                ok = ok * (end_count <= max_reps)

        return pattern
            
    @staticmethod
    def diagonals(board_size, step_vec, skip, edge_buffer, 
                  starting_corner, alternate, reverse_lines):
        starting_corner = HunterOffense.coord_for_corner(starting_corner, 
                                                         board_size)        
        basic = HunterOffense.diagonals_basic(board_size, step_vec, skip, 
                                              edge_buffer, alternate, 
                                              reverse_lines)
        
        if starting_corner == Coord((0,0)):
            pattern = basic
        elif starting_corner == Coord((0,board_size-1)):
            pattern = mirror_coords(basic, axis=1, board_size=board_size)
        elif starting_corner == Coord((board_size-1, 0)):
            pattern = mirror_coords(basic, axis=0, board_size=board_size) 
        elif starting_corner == Coord((board_size-1, board_size-1)):
            pattern = rotate_coords(basic, 180, board_size)
        return pattern
    
    @staticmethod
    def diagonals_basic(board_size, step_vec, skip, edge_buffer, 
                        alternate, reverse_lines):
        
        step_vec = np.array(step_vec)
        istep = np.arange(0, board_size+1)
        iskip = np.arange(-board_size, board_size+1)
        Steps, Skips = np.meshgrid(istep, iskip, indexing='ij')
        # for basic, skip direction is always along columns
        R = Steps.transpose() * step_vec[0] + Skips.transpose() * 0     
        C = Steps.transpose() * step_vec[1] + Skips.transpose() * 2 * skip
        coords = list(zip(R.flatten(),C.flatten()))
        coords = [r for r in coords if 
                  HunterOffense.coord_ok(r, [], board_size, edge_buffer)]
        # arrange so that the first diagonal is the one starting at (0,0)
        i_new_diag = [i+1 for (i,delta) in 
                      enumerate(np.diff(np.array(coords),axis=0)) 
                      if delta[0] < 0]
        diag_ranges = ([range(0,i_new_diag[0])] 
                       + [range(i_new_diag[j], i_new_diag[j+1]) for
                          j in range(len(i_new_diag)-1)] 
                       + [range(i_new_diag[-1],len(coords))])
        start_range = [r for r in diag_ranges if coords.index((0,0)) in r][0]
        i_start_range = diag_ranges.index(start_range)
        nr = len(diag_ranges)
        # offset_vec = (int(np.floor(step_vec[0]/2)), 
        #               int(np.floor(step_vec[1]/2)))
        offset_vec = (0,0)
        if alternate:
            # put diags in alternating order
            alternating = np.arange(-nr, nr+1)
            range_order = np.vstack((alternating[(nr+1):],
                                     alternating[:nr][::-1])).flatten(order='F')
            range_order = np.hstack((0, range_order)) + i_start_range
            diag_ranges_ordered = [diag_ranges[ro] for ro in range_order if 
                                   ro >= 0 and ro < len(diag_ranges)]
            diagonals = []
            for (j,diag_range) in enumerate(diag_ranges_ordered):
                diagonals += [(coords[idx][0] + (j%2)*offset_vec[0], 
                               coords[idx][1] + (j%2)*offset_vec[1])
                              for idx in diag_range]
        
        else:
            range_order = (list(range(i_start_range, nr)) 
                           + list(range(i_start_range)))
            diagonals = []
            for (j,irange) in enumerate(range_order):
                next_coords = [(coords[idx][0] + (j%2)*offset_vec[0], 
                               coords[idx][1] + (j%2)*offset_vec[1])
                              for idx in diag_ranges[irange]]
                if reverse_lines and j % 2 == 1:
                    # reverse every other range
                    next_coords = next_coords[::-1]
                diagonals += next_coords
        return diagonals
    
    def pattern_spiral(self, board_size, direction='cw', at_end=None,
                       max_reps=None, **kwargs):
        # board_size, step_size, skip, start_at, 
        #                repeat, edge_buffer, direction, expand, last_resort):
        """
        
        Returns
        -------
        A list of target coordinates that conform to the spiral pattern.

        """
        
        # Set default parameters here:
        start_at = kwargs.get('start_at')
        step_size = kwargs.get('step_size')
        #skip = kwargs.get('skip')
        edge_buffer = kwargs.get('edge_buffer')
        offset = kwargs.get('offset')
        expand = kwargs.get('expand')
        
        start_at = (DEFAULT_GRID_STARTING_CORNER if 
                    start_at == None else start_at)
        step_size = (DEFAULT_GRID_STEP_SIZE if step_size == None else step_size)
        #skip = (DEFAULT_GRID_SKIP if skip == None else skip)
        edge_buffer = (DEFAULT_GRID_EDGE_BUFFER if 
                       edge_buffer == None else edge_buffer)
        offset = (DEFAULT_GRID_OFFSET if offset == None else offset)
        expand = (False if expand == None else expand)
        max_reps = (10 if max_reps == None else max_reps)
        
        spiral = HunterOffense.spiral(board_size, step_size, edge_buffer, 
                                      expand, start_at, direction)
        rotate90 = np.array(([0, 1], [-1, 0]))
        if at_end == 'repeat' or at_end == 'restart':
            if step_size == 1:
                pass
            else:
                if expand:
                    pass
                else:
                    dr = ((np.array(spiral[1]) - np.array(spiral[0])) 
                          / step_size)
                    # 90 degree rotation
                    dr1 = (rotate90 @ dr if direction.lower() == 'cw' else
                           -rotate90 @ dr if direction.lower() == 'ccw' else
                           None)
                    dr2 = dr * offset
                    new_start_at = self.coord_for_corner(start_at, board_size)
                    new_start_at = new_start_at + dr1 + dr2
                
                # use edge_buffer to offset the spiral from the corner
                new_spiral = self.spiral(board_size, step_size, 
                                         new_start_at[0], 
                                         expand, start_at, direction)
                spiral.extend(new_spiral)
                
        elif at_end == 'reverse':
            if step_size == 1:
                pass
            else:
                new_spiral = self.rotate_coords(spiral[::-1], 180, board_size)
                spiral.extend(new_spiral)
            
        elif at_end == None:
            pass
        else:
            raise ValueError("parameter at_end must be 'repeat', 'reverse', "
                             "or None (default).")
        return spiral
        return [(round(p[0]),round(p[1])) for p in spiral]
                     
    def spiral(self, board_size, step_size, edge_buffer, expand, 
               start_at=None, direction="CW"):
        """
        Returns an ordered list of coordinates that constitute a spiral 
        pattern. The direction of the spiral, its starting location, spacing
        between points, and distance to the edge are controlled by parameters.

        Parameters
        ----------
        board_size : int
            Size of the board where the spiral will be generated.
        step_size : int
            Number of coordinates to step between each point in the spiral.
        edge_buffer : int
            The pattern will not come within this many spaces of the edges.
        expand : bool
            If True, the spiral starts in the center and winds outward.
            If False (default), the spiral starts at a corner and winds inward.
        start_at : tuple, Coord, or str, optional
            For an expanding spiral, this must be "center".
            For a contracting spiral (default), start_at can be any of:
                - A str abbreviation for the direction of the desired starting
                    corner (NE, NW, SE, SW).
                - A str label for the coordinate at the desired corner 
                    (A1, A10, J1, J10).
                - A tuple or coordinate corresponding to the desired corner
                    ((0,0), (0,9), (9,0), (9,9) or Coord((0,0)), etc.)
            The default is "NW".
        direction : str, optional
            "CW" for clockwise, "CCW" or "ACW" for coutnerclockwise.
            The default is "CW".

        Returns
        -------
        List of coordinates that constitute a spiral pattern.

        """
        if expand:
            if start_at == None:
                start_at = "center"
            if start_at.lower() != "center":
                raise ValueError("start_at must be 'center' for an expanding"
                                 " spiral.")
        else:
            start_at = self.coord_for_corner(start_at, board_size)
            
        pattern = self.spiral_basic(board_size, step_size, 
                                    edge_buffer, expand)
        direction = direction.upper()
        if direction == "ACW":
            direction = "CCW" 
        
        if expand:
            if direction == "CCW":
                pattern = self.mirror_coords(pattern, 1, board_size)
        else:
            if direction == "CW":
                n_rotations = (0 if start_at == Coord((0,0)) else
                               1 if start_at == Coord((0,board_size-1)) else
                               2 if start_at == Coord((board_size-1,
                                                       board_size-1)) else
                               3 if start_at == Coord((board_size-1,0)) else
                               None)
            elif direction == "CCW":
                pattern = self.mirror_coords(pattern, 0, board_size)
                n_rotations = (1 if start_at == Coord((0,0)) else
                               2 if start_at == Coord((0,board_size-1)) else
                               3 if start_at == Coord((board_size-1,
                                                       board_size-1)) else
                               0 if start_at == Coord((board_size-1,0)) else
                               None)
                
            for _ in range(n_rotations):
                pattern = self.rotate_coords(pattern, 90, board_size)
                
        return pattern
        
    @staticmethod
    def spiral_basic(board_size, step_size, edge_buffer, expand):
        """A basic spiral that starts in the upper left corner and spirals in 
        toward the center of the board (or starts and the center and spirals
        out, if expand is True).
        """
        
        rot_angle = np.pi/2
        rotate = np.round(np.array(([np.cos(rot_angle), np.sin(rot_angle)], 
                                    [-np.sin(rot_angle), np.cos(rot_angle)])))
        dr = np.array((0, int(step_size * np.sin(rot_angle))))
        if expand:
            r = (int(board_size/2) - 1, int(board_size/2) - 1)
            dr = -rotate @ dr
            transform_for_ok = rotate
            transform_for_not_ok = np.identity(len(dr))   
        else:
            r = (edge_buffer, edge_buffer)
            transform_for_ok = np.identity(len(dr))   
            transform_for_not_ok = rotate
        
        pattern = [r]
        go = True
        while go:
            trial = tuple(np.array(pattern[-1]) + transform_for_ok @ dr)
            if HunterOffense.coord_ok(trial, pattern, 
                                      board_size, edge_buffer):
                dr = transform_for_ok @ dr
                pattern.append(trial)
            else:
                trial = tuple(np.array(pattern[-1]) + transform_for_not_ok @ dr)
                if HunterOffense.coord_ok(trial, pattern, 
                                          board_size, edge_buffer):
                    dr = transform_for_not_ok @ dr
                    pattern.append(trial)
                else:
                    go = False
        return [tuple(pt) for pt in pattern]
     
    @staticmethod
    def grid(board_size, starting_corner, step_size, skip, 
             edge_buffer, reverse_lines, offset):
        """Returns an ordered list of coordinates that constitute a single grid 
        pattern. The grid may start in any corner and proceed along either rows
        or columns first.
        """
        starting_corner = HunterOffense.coord_for_corner(starting_corner, 
                                                         board_size)        
        G0 = HunterOffense.grid_basic(board_size, step_size, skip, edge_buffer, 
                                      reverse_lines, offset)
        
        if starting_corner == Coord((0,0)):
            G = G0
        elif starting_corner == Coord((0,board_size-1)):
            G = mirror_coords(G0, axis=1, board_size=board_size)
        elif starting_corner == Coord((board_size-1, 0)):
            G = mirror_coords(G0, axis=0, board_size=board_size) 
        elif starting_corner == Coord((board_size-1, board_size-1)):
            G = rotate_coords(G0, 180, board_size)
        return G
     
    @staticmethod
    def grid_basic(board_size, step_size, skip, edge_buffer, 
                   reverse_lines, offset):
        """Returns an ordered list of coordinates that constitute a grid 
        pattern. The basic grid always starts in the upper left corner and 
        steps from column to column across a row, then goes to the next
        row once it reaches the end of the column.
        """
        # step_vector = np.array((0, step_size))
        nlines = int(np.ceil((board_size - 2 * edge_buffer) / (1 + skip)))
        nsteps_per_line = int(np.ceil((board_size - 2 * edge_buffer) / step_size))  # included for possible future decoupling of row and col edge_buffers.
        
        coords = []
        for r in range(nlines):
            row = edge_buffer + r * (skip + 1)
            cols = (edge_buffer + np.arange(nsteps_per_line) * step_size + 
                    (offset * (r % 2)))
            cols = cols[cols < board_size]                
            cols = cols[::-1] if (reverse_lines and r % 2 == 1) else cols     
            coords += zip(np.full_like(cols, row), cols)
        return coords
    
    def next_step(self, current_coord, pattern):
        """
        Returns the next coordinate in the pattern of coordinates. Returns None
        if the input coordinate is the last one in the pattern.

        Parameters
        ----------
        current_coord : Coord or tuple (2 elements)
            The current coordinate in the pattern. The return value will be the
            coorindate that follows this one (each coordinate should be unique).
        pattern : List
            A list of coordinates (2-element tuples) that make up a pattern of
            coordinates, such as a grid or spiral.

        Returns
        -------
        Tuple (2-element)
            The next coordinate in the pattern.

        """
        if current_coord == None:
            return pattern[0]
        
        if isinstance(current_coord, Coord):
            current_coord = current_coord.rowcol
        idx = pattern.index(current_coord)
        if idx == len(pattern) - 1:
            return None
        return pattern[idx+1]
    
    ### Utility Methods ###
    
    @staticmethod
    def coord_for_corner(corner, board_size):
        """
        Returns the Coord that corresponds to the input corner string and 
        board size. The string may be a coordinate label such as A1, A10, J1, 
        etc., or it may be a direction: NW (upper left), NE (upper right),
        SW (lower left), SE (lower right).

        Parameters
        ----------
        corner : str
            A label for the corner coordinate on board of size board_size,
            or a directional string:: NW (upper left), NE (upper right),
            SW (lower left), SE (lower right). The label must be one of A1, 
            A10, J1, J10 (for a board of size 10; the two "10" coordinates 
            change with board size.)
        board_size : int
            The size of the board for which a corner coordinate is desired.

        Returns
        -------
        Coord
            The Coord object corresponding to the specified corner string.

        """
        corner_labels = ["A1", f"A{board_size}", 
                         f"{chr(64+board_size)}1", 
                         f"{chr(64+board_size)}{board_size}"]
        if isinstance(corner, Coord):
            lbl = corner.lbl
            if lbl not in corner_labels:
                lbl = None
        else:
            corner = corner.upper()
            if corner in corner_labels:
                lbl = corner
            else:
                lbl = ("A1" if corner == "NW" else 
                       f"A{board_size}" if corner == "NE" else
                       f"{chr(64+board_size)}1" if corner == "SW" else 
                       f"{chr(64+board_size)}{board_size}" if corner == "SE" else
                       None)
        if lbl == None:
            raise ValueError("Coordinate is not at a corner of the board.")
        return Coord(lbl)
    
    @staticmethod
    def coord_ok(coord, pattern, board_size, edge_buffer):
        """
        Returns True if the input coordinate is not already listed in the
        pattern parameter AND if it lies on a valid space on the input board
        (and not on one of the edge_buffer rows/columns at the edge of the
        board, if edge_buffer is greater than 0).

        Parameters
        ----------
        coord : Coord or tuple
            A single Coord instance or two-element tuple.
        pattern : list
            A list of coordinates.
        board_size : Int 
            Number of spaces on the board on which coord is to be validated. 
            If coord lies past the edges of the board, False is returned.
        edge_buffer : Int, optional
            An integer. If 0, Coord may lie anywhere on the board and True will
            be returned. If edge_buffer is an integer greater than 0, when
            coord lies in the first or last edge_buffer rows or columns of the
            board, False will be returned. The default is 0.

        Returns
        -------
        Bool
            True if coord is not contained in pattern, lies on a valid space
            on the board, and is not on one of the rows or columns that is
            edge_buffer from the board's edges.

        """
        return (not (coord in pattern)
                and coord[0] >= edge_buffer
                and coord[0] < board_size - edge_buffer
                and coord[1] >= edge_buffer
                and coord[1] < board_size - edge_buffer)
        
     
#%% Unincorporated stuff

def coord_interp(R):
    """
    Interpolates the input list of coordinates to single coordinate steps.
    

    Parameters
    ----------
    R : List
        A list of coordinates.

    Returns
    -------
    Ri : List
        List of all coordinates between the points in the input pattern
        (including those points actually in the input list)..

    """
    # start at first point, step toward second point, then toward third, etc.
    R = [Coord(r) if not isinstance(r, Coord) else r for r in R]
    current_coord = R.pop(0)
    next_coord = R.pop(0)
    dr = np.array((next_coord - current_coord).rowcol)
    step = dr / np.sqrt(np.sum(dr**2))
    Ri = [current_coord]
    while next_coord:
        ri = Ri[-1] + step
        Ri += [ri]
        if ri == next_coord:
            current_coord = next_coord
            if R:
                next_coord = R.pop(0)
                dr = np.array((next_coord - current_coord).rowcol)
                step = dr / np.abs(dr).max()
            else:
                next_coord = None
    return Ri

def plot_pattern(coords):
    y,x = zip(*coords)
    fig,ax = plt.subplots(1,1)
    ax.set_xlim((-0.5,9.5))
    ax.set_ylim((-0.5,9.5))
    ax.set_xticks(range(10))
    ax.set_xticks(np.arange(-0.5,9.5), minor=True)
    ax.set_xticklabels(range(1,11))
    ax.set_yticks(range(10))
    ax.set_yticklabels([chr(64+x) for x in range(1,11)])
    ax.set_yticks(np.arange(-0.5,9.5), minor=True)
    ax.grid(which='minor')
    for (i,coord) in enumerate(coords):
        ax.text(coord[1],coord[0],str(i),horizontalalignment='center',
                verticalalignment='center', color='black',fontsize=14)
    ax.invert_yaxis()
    return ax
        
def targets_str(board, targets):
    board_str = str(board)
    board_str = board_str[1:int(len(board_str)/2)]
    while board_str[0] == "\n":
        board_str = board_str[1:]
    chars_per_row = board_str.index("\n")
    for target in targets:
        row = target[0]
        col = target[1]
        idx = (1 + row) * chars_per_row + 3 + 2 * col
        board_str = board_str[:idx] + "+" + board_str[(idx+1):]
    return board_str
    
       

        
            