#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 12:47:00 2022

@author: jason
"""


#%% Imports
import abc
import numpy as np
from matplotlib import pyplot as plt

# Imports from this package
from battleship import constants
from battleship import utils

from battleship.ship import Ship
from battleship.board import Board
from battleship.coord import Coord


# %% Player Class

class Player(abc.ABC):
    """An abstract class that can play a game of Battleship.
    This class provides the superclass for two concrete subclasses:
    AIPlayer and HumanPlayer.
    
    All Player instances must implement the following attributes and methods:
    
    
    
    Properties
    - name
    - offense (optional for HumanPlayer)
    - defense (optional for HumanPlayer)
    - shot_history
    - outcome_history
    - board
    - opponent
    - possible_targets
    - remaining_targets
    - set_opponent
    
    Other methods
    - __init__
    - __str__
    - reset
    - copy_history_from
    - last_target
    - last_outcome
    - isalive
    - fire_at_target
    - place_fleet_using_defense
    - show_possible_targets
    """
    
    ### Initializers and Such ###
    
    def __init__(self, name="", board=None):
        """
        An abstract player that controls a board in a game of Battleship.
        This class is an abstract base class for subclasses of Player that
        can be instantiated.
        
        Concrete subclasses of Player must implement the following:
        
        - player_type (property)
        - take_turn (method)
        - prepare_fleet (method)
        
        All subclasses of Player will also include the following properties,
        which are defined in the Player class:
            
        - player_type : str
        - name : str
        - shot_history : list of coordinates that the player has fired at.
        - outcome_history : list of dictionaries detailing the outcome of
                            each shot.
        - remaining_targets : set of coordinates that has not yet been shot at.
        - board : Board instance on which the Player plays.
        - opponent : Player subclass instance. Only the incoming_at_coord 
                     method should be accessed by this instance of player
                     (anything else could be cheating!).
        
        Parameters
        ----------
        name : str, optional
            An identifying string used to refer to this player instance.
            The default is "".
        board : Board, optional
            An instance of a Battleship Board. If not provided during 
            initialization, an empty board will be created when prepare_fleet
            is called. The default is None.

        Returns
        -------
        None.

        """
        self._player_type = None
        self._name = name
        self._shot_history = []
        self._outcome_history = []
        self._remaining_targets = []
        self._possible_targets = None
        self._board = board
        self._opponent = None
        self._verbose = False
        
        self._game_history = None    # for future use
        self._notes = None           # for future use
        
    def __str__(self):
        """
        Returns a user-friendly string representation of the instance.

        Returns
        -------
        str
            String containing player type, strategy, and any strategy
            data.

        """
        
        return (f"Player {self.name}\n" 
                f"  Type: {self.player_type}\n" 
                f"  Defense: {self.defense}\n" 
                f"  Offense: {self.offense}")
            
    def __repr__(self):
        return f"Player({self._name!r}, {self._board!r})"
            
    def turn_str(self):
        """
        Returns a string describing the actions taken and outcome of the
        player's most recent turn.

        Returns
        -------
        s : str
            String indicating the player's name, most recent target coordinate,
            potential targets, offense details, and outcome of last shot.

        """
        s = (f"Player {self.name}" 
             f"  Fired at coordinate {self.last_target()}")
        if len(self.possible_targets) > 6:
            s += (f"    Selected from {len(self.possible_targets)} "
                  f"    potential targets.")
        else:
            s += (f"    Selected from potential targets:\n"
                  f"    {self.potential_targets}")
        if self.offense:
            s += f"  Offense:\n{self.offense}"
        outcome = self.last_outcome()
        res = (Ship.data[constants.ShipType(outcome["sunk_ship_id"])]["name"] 
               + " sunk" if outcome["sunk"] else 
               "hit" if outcome["hit"] else "miss")
        s += f"Result: {res}"
        s += f"Outcome:\n{self.last_outcome()}"
        return s
        
    ### Abstract Property ###
    @property
    @abc.abstractmethod
    def player_type(self):
        """
        This method should return a string describing the player type
        (such as 'AI' or 'Human').
        """
        pass
    
    ### Abstract Method ###
    @abc.abstractmethod
    def take_turn(self): 
        """
        This method should choose a target coord, then call the player's 
        fire_at_target method (which updates offense data), and perform
        any desired book-keeping for the Player subclass.
        """
        pass
    
    @abc.abstractmethod
    def prepare_fleet(self):
        """This method should determine the desired placement of each ship,
        and place it on the board using player.board.add_fleet.
        Typical use is to call the method place_fleet_using_defense, 
        which gets placements from the player's defense then passes those to
        player.board.add_fleet. Or, for a human player, get placements
        from user input then pass the placements to player.board.add_fleet.
        """
        pass
    
    ### Properties ###
    
    @property
    def name(self):
        """
        The player's identifying name.

        Returns
        -------
        str
            A string that uniquely identifies the player instance.

        """
        
        return self._name
    
    @name.setter
    def name(self, name):
        self._name = name
       
    @property
    def offense(self):
        """
        A instance of a subclass of the Offense abstract metaclass.
        Optional for instances of HumanPlayer, required for instances of 
        AIPlayer.

        Returns
        -------
        Offense subclass instance
            
        """
        return self._offense
    
    @offense.setter
    def offense(self, offense):
        self._offense = offense

    @property    
    def defense(self):
        """
        A instance of a subclass of the Defense abstract metaclass.
        Optional for instances of HumanPlayer, required for instances of 
        AIPlayer.

        Returns
        -------
        Defense subclass instance
            
        """
        return self._defense
    
    @defense.setter
    def defense(self, defense):
        self._defense = defense
    
    @property
    def outcome_history(self):
        """A list of outcome dictionaries describing the result of a turn.
        The ith element corresponds to the outcome of the ith turn taken by 
        the player. Reset before the start of a new game using the reset() 
        method.
        """
        return self._outcome_history
    
    def add_outcome_to_history(self, outcome):
        """Adds the input outcome dictionary to the outcome history list.
        The outcome input should have keys 'hit', 'coord', 'sunk',
        'sunk_ship_id', and 'message'.
        """
        if len(self._outcome_history) == 0 and self.offense:
            self.offense.outcome_history = self._outcome_history
        self._outcome_history += [outcome]
        
    @property
    def board(self):
        """The instance of Board used by the player to hold their fleet,
        track damage, and track shot target locations.
        """
        return self._board
    
    @board.setter
    def board(self, board):
        self._board = board
        if self.offense:
            self.offense.board = board
        self._remaining_targets = self.board.all_targets(untargeted=True)
        
    @property
    def opponent(self):
        """An instance of Player that corresponds to the current opponent.
        This is usually set by an instance of Game.
        """
        return self._opponent
    
    @opponent.setter
    def opponent(self, opponent):
        """Sets the opponent variable to the input Player.
        """
        self._opponent = opponent
        if opponent.opponent != self:
            opponent.opponent = self
    
    @property
    def remaining_targets(self):
        """Returns a list of coordinates that have not yet been targeted during
        the current game.
        """        
        return self._remaining_targets
    
    @property
    def verbose(self):
        return self._verbose
    
    @verbose.setter
    def verbose(self, value):
        self._verbose = value
            
    
    # Computed Properties
    
    @property
    def shot_history(self):
        """A list of coordinates (2-element tuples or instances of Coord) 
        with ith element corresponding to the coordiate targeted in the ith
        shot taken by the player. Reset before the start of a new game using
        the reset() method.
        """
        return [outcome["coord"] for outcome in self.outcome_history]
    
    @property
    def last_target(self):  
        """Returns the most recently targeted coordinate, or None if no shot
        has been taken yet.
        """
        if not self.outcome_history:
            return None
        else:
            return self.outcome_history[-1]["coord"]
        
    @property
    def last_outcome(self): 
        """Returns the outcome of the last shot, or None if no shot has been 
        taken yet.
        """
        if not self.outcome_history:
            return None
        else:
            return self.outcome_history[-1]
        
    ### General Methods ###
    
    def reset(self):
        """Removes all ships and hit/miss indicators from the player's board,
        and empties out the shot and outome logs. The opponent remains the 
        same until changed usint set_opponent.
        """
        #self._shot_history = []
        self._outcome_history = []
        if self.board:
            self.board = Board(self.board.size)
            self._remaining_targets = self.board.all_coords()
        if self.offense:
            self.offense.reset()
        
    def copy_history_from(self, other):
        """Copies shot and outcome histories from the input Player instance
        into this Player's histories (overwriting any existing histories).
        Also copies over the remaining_targets, game_history, and notes 
        properties, and sets the possible_targets list to None.
        """
        self._outcome_history = other.outcome_history
        self._remaining_targets = other.remaining_targets
        self._game_history = other.game_history
        self._notes += other.notes
        
    def is_alive(self): 
        """Returns true if any of this players ships are still afloat."""
        return any(self.board.is_fleet_afloat())
    
    def fire_at_target(self, target_coord): 
        """Fires a shot at the input target space and handles the resulting
        outcome. If the shot results in a hit, the opponent's ship is damaged.
        The Player's board is updated with a hit or miss value at the 
        target_space.
        Returns an outcome dictionary with the following keys:
            - hit
            - coord
            - sunk
            - sunk_ship_id
            - message
        """
        target_coord = (target_coord[0], target_coord[1]) # enforce tuple
        outcome = self.opponent.board.incoming_at_coord(target_coord)
        self.board.update_target_grid(outcome)
        self.add_outcome_to_history(outcome)
        if target_coord in self.remaining_targets:
            self.remaining_targets.pop(
                self.remaining_targets.index(target_coord)
                )
        if self.offense:
            self.offense.update_state(outcome)    # update is impelmented in subclass
        
    def place_fleet_using_defense(self, defense):
        """Place the fleet ship-by-ship according to the input defense."""
        placements = defense.fleet_placements(self.board)
        self.board.add_fleet(placements)
        
    def show_possible_targets(self):
        """Shows the board for the player, along with possible targets."""
        fig = self.board.show()
        axs = fig.axes
        ax = axs[0]
        rowcol = self.last_target
        target_circ = plt.Circle((rowcol[1]+1, rowcol[0]+1),
                                 radius = 0.3*1.4, 
                                 fill = False,
                                 edgecolor = "yellow") 
        possibilities = [plt.Circle((rowcol[1]+1, rowcol[0]+1), 
                                    radius = 0.3,
                                    fill = False,
                                    edgecolor = "cyan",
                                    linestyle = ":") 
                         for rowcol in self.possible_targets]
        ax.add_patch(target_circ)
        for p in possibilities:
            ax.add_patch(p)
            
    def show_possible_targets_new(self):
        #Viz.show_possibilities_grid()
        pass
            
    def stats(self):
        history = self.outcome_history
        nshots = len(history)
        shots_rc = np.array([h['coord'] for h in history])
        hits = np.array([h['hit'] for h in history])
        ships_sank = np.array([h['sunk_ship_id'] for h in history if 
                               h['sunk_ship_id'] != None])
        shots_mean = np.mean(shots_rc,axis=0)
        shots_var = np.var(shots_rc - np.tile(shots_mean, (nshots,1)), axis=0)
        # ships
        ships_rc = np.empty((0,2), int)
        for ship in self.board.fleet.values():
            ships_rc = np.append(ships_rc, self.board.coords_for_ship(ship), 
                                 axis=0)
        ships_mean = np.mean(ships_rc, axis=0)
        ships_var = np.var(ships_rc - np.tile(ships_mean, 
                                             (ships_rc.shape[0],1)), axis=0)
        return dict(nshots = nshots,
                    hits = hits, 
                    ships_sank = ships_sank,
                    shots_mean = shots_mean,
                    shots_var = shots_var,
                    ships_mean = ships_mean,
                    ships_var = ships_var
                    )
        
        
        
#%% Concrete Player Subclasses

class HumanPlayer(Player):
    """An concrete subclass of Player provides an interface so a Human user
    can play a game of Battleship.
    
    All Player instances must implement the following attributes and methods:
    
    Abstract Properties/Methods:
    - player_type (property)
    - take_turn (method)
    - prepare_fleet (method)
    
    This particular subclass has the following properties and methods:
        
    Properties
    - target_select_mode
    - show_targets
    
    Other methods
    - __init__
    """
    
    ### Initializers and Such ###
    
    def __init__(self, offense=None, defense=None, name=""):
        """
        A subclass of Player that allows a Human user to interface with and
        play a game of Battleship against an opponent (either Human or AI).

        Parameters
        ----------
        offense : Offense subclass, optional
            An instance of a subclass of Offense. Valid options include:
                - RandomOffense
                - HunterOffense(smart, spacing)
            The default is None. 
            
        defense : Defense subclass, optional
            An instance of a subclass of Offense. Valid options include:
                - RandomDefense(alignment)
                - ClusterDefense(alignment)
                - IsolateDefense(alignment, diagonal, max_sep, separation)
            The default is None. When it is the HumanPlayer's turn to place
            their fleet, the player may either enter all ship locations and
            headings manually, or allow the Defense subclass to place the ships.
            
        name : str, optional
            A string used to idenfify the HumanPlayer instance. The default is "".

        Returns
        -------
        None.

        """
        super().__init__(name)
        
        self.offense = offense
        self.defense = defense
            
        self.show_targets = False
        self._target_select_mode = "text" 
        
    def __repr__(self):
        return f"HumanPlayer({self.offense!r}, {self.defense!r}, {self.name})"
    
    ### Abstract Properties and Methods
    @property
    def player_type(self):
        return "Human"
    
    def take_turn(self):
        """Get a target from the human, and fire at that target on the opponent's
        board.
        """
        if self.target_select_mode == "text":
            print(self.board.color_str())
            if self.outcome_history:
                outcome = self.outcome_history[-1]
                hitmiss = "hit" if outcome['hit'] else "miss"
                print(f"Last shot: {outcome['coord']} -->> "
                      f"{hitmiss}.")
                outcome = self.opponent.outcome_history[-1]
                hitmiss = "hit" if outcome['hit'] else "miss"
                print(f"Opponent's last shot: {outcome['coord']} -->> "
                      f"{hitmiss}.")
                
            target = Coord(input("Enter target: ").upper())
            if target in self.shot_history:
                target = input((f"*** Space {target} has already been \
                                targeted.\nEnter again to confirm, or select \
                                another target: ")) 
        else:
            raise AttributeError("Invalid target_select_mode.")
        self.fire_at_target(target)
        
    def prepare_fleet(self):
        """If a defense strategy (i.e., fleet placement) is provided, let
        if determine how to set up the fleet. Otherwise, have the player
        enter it manually.
        """
        
        if not self.board:
            self.board = Board()
            
        if self.defense:
            if isinstance(self.defense, dict):
                placements = self.defense
            else:
                placements = self.defense.fleet_placements(self.board)
            return
        elif self.defense is None:
            placements = self.input_placements()
        else:
            raise ValueError("The HumanPlayer defense must be an instance of "
                             "Defense, a dictionary, or None (for "
                             "interactive input).")
            
        self.board.place_fleet(placements)
        if self.verbose:
            print("Fleet placed successfully.")  
    
    ### Properties ###
    
    @property
    def target_select_mode(self):
        return self._target_select_mode
    
    @target_select_mode.setter
    def target_select_mode(self, value):
        if value.lower() == "text":
            self._target_select_mode = "text"
        else:
            raise ValueError("target_select_mode only supports 'text'.")
        
    ### Other Methods ###
    
    def input_placements(self):
        """
        Get ship placements interactively from the user via the command line.

        Returns
        -------
        Dictionary of ship placements.

        """
        board = Board(self.board.size)  # temporary board
        for i in range(1,len(Ship.data)+1):
            ok = False
            name = Ship.data[constants.ShipType(i)]["name"]
            while not ok:
                space = Coord(input("Space for {name.upper())} ('r' for random): "))
                if space.lower() == 'r':
                    coord, heading = utils.random_choice(
                        board.random_placement(
                            Ship.data[constants.ShipType(i)]["length"])
                        )
                else:
                    coord = Coord(space)
                    heading = input("Heading for {name.upper())}: ")                
                if not board.is_valid_ship_placement(i, coord, heading):
                    ok = False
                    print("Invalid placement.")
                else:
                    board.place_ship(i, coord, heading)
                    print(name + " placed successfully (" 
                          + str(i) + " of " + str(len(Ship.data)) + ").")
    
    
#%% AIPlayer
class AIPlayer(Player):
    
    ### Initializers and Such ###
    
    def __init__(self, offense=None, defense=None, name=""):
        """
        A subclass of Player that generates ship placements and targets for
        playing a game of Battleship against an opponent (either Human or AI).
        The AIPlayer makes use of two input objects--a subclass of Offense and
        a subclass of Defense--in order to make targeting and ship placement
        decisions, respectively.
        
        See the Parameters below for supported Offense and Defense subclasses.
        New subclasses of Offense/Defense may also be used, as long as they
        implement the proper methods for targeting and ship placement (see
        documentation for Offense and Defense).

        Parameters
        ----------
        offense : Offense subclass, optional
            An instance of a subclass of Offense. Valid options include:
                - RandomOffense
                - HunterOffense(smart, spacing)
                
            The default is None. 
            
        defense : Defense subclass, optional
            An instance of a subclass of Offense. Valid options include:
                - RandomDefense(alignment)
                - ClusterDefense(alignment)
                - IsolateDefense(alignment, diagonal, max_sep, separation)
                
            The default is None. When it is the HumanPlayer's turn to place
            their fleet, the player may either enter all ship locations and
            headings manually, or allow the Defense subclass to place the ships.
            
        name : str, optional
            A string used to idenfify the HumanPlayer instance. 
            The default is "".

        Examples
        --------
        p1 = AIPlayer(HunterOffense('random',weight='maxprob'), 
                      RandomDefense('isolated', maxsep=True))
        p1.prepare_fleet()
        Returns
        -------
        None.
        """
        super().__init__(name)
        
        self.offense = offense
        self.defense = defense
            
        self.show_targets = False
        self._target_select_mode = "text" 
        
    def __repr__(self):
        return f"AIPlayer({self.offense!r}, {self.defense!r}, {self.name!r})"
    
    ### Abstract Properties and Methods
    @property
    def player_type(self):
        return "AI"
    
    def take_turn(self):
        """Get a target from the AI Player's strategy and fire at that 
        coordinate.
        """
        self.fire_at_target(self.offense.target(self.board, 
                                                self.outcome_history))
        self.offense.update_state(self.last_outcome)
        
        # self.offense.possible_targets contains the list of all potential
        # targets from which one actual target was chosen randomly.
        
    def prepare_fleet(self):
        """Place fleet according to the defense strategy."""
        if not self.board:
            self.board = Board()
            
        if isinstance(self.defense, dict):
            placements = self.defense
        elif self.defense is None:
            raise ValueError("Defense cannot be None for an AIPlayer.")
        else:
            placements = self.defense.fleet_placements(self.board)
            
        self.board.add_fleet(placements)
        if self.verbose:
            print("Fleet placed successfully.") 
            
#%% Pattern functions

def search_pattern(board, method, **kwargs):
    """
    Returns an ordered list of targets based on the search method, board size, 
    and options entered as key/value pairs. Each type of method has different
    required and optional parameters.

    Parameters
    ----------
    board : Board 
        Instance of board on which to create a search pattern.
    method : str
        One of the following:
            "grid"      Search along a periodic grid of coordinates.
            "spiral"    Search along an inward/outward spiral of coordinates.
            "diagonal"  Similar to grid, but proceed along diagonals rather
                        than rows/columns.
    **kwargs : key/value pairs (e.g., spacing = 2)
        Named options for tuning the search pattern. Keys for each of the 
        search methods are listed below.
        
        'Grid' pattern
        --------------
        start_at        (row,col) tuple where the first search coordinate in
                        the pattern will be located.
        axis            0 to proceed along a column to the end, then move to 
                        the next row. 1 to proceed along a row to the end, then
                        move to the next column. Defaults to 1.
        spacing         The step size between subsequent points along each row
                        and column. There will be spacing-1 untargeted rows/
                        columns between subsequent targeted rows/columns.
        end_of_line     str, 'reset' or 'reverse'. If 'reset', go to the 
                        start of the next row and proceed with the same step
                        direction as the current row/col. If 'reverse', go to
                        the end of the next row/col and proceed with a step
                        that is in the opposite direction to the step in the
                        current row/col. Defaults to 'reset'.
        edge_offset     Leave at least this many spaces between a space in the
                        pattern and the edge of the board. If the value of 
                        start_at would put the first coordinate within 
                        edge_offset of the edge, the edge_offset is respected
                        and all points that are too close to the edge are 
                        not included in the pattern.
                        Defaults to 0 (i.e., the pattern can go up to the 
                        first/last row and columns.)
                        
        'Spiral' pattern
        --------------
        start_at        (row,col) tuple where the first search coordinate in
                        the pattern will be located.
        direction       str : 'cw' for a clockwise spiral, 'ccw' for a counter-
                        clockwise spiral.
        spiral_outward  If True, the spiral is run in reverse, starting near
                        the middle of the grid and expanding outward.
                        The default it False.
        spacing         The step size between subsequent points along the 
                        pattern. spacing-1 points will be left between each
                        turn in the spiral. This means that the corner of a 
                        turn does not necessarily have a target at it.
        at_end          str, 'reset' or 'reverse'. If 'reset', go to the 
                        start of the spiral, move to the next row (or column,
                        if the pattern first proceeds along rows), and move
                        through a similar spiral from the new start point.
                        If 'reverse', change the direction from 'cw' to 'ccw' 
                        or vice-versa, and spiral_outward to its inverse.
                        Defaults to 'reset'.
        edge_offset     Leave at least this many spaces between a space in the
                        pattern and the edge of the board. If the value of 
                        start_at would put the first coordinate within 
                        edge_offset of the edge, the edge_offset is respected
                        and all points that are too close to the edge are 
                        not included in the pattern.
                        Defaults to 0 (i.e., the pattern can go up to the 
                        first/last row and columns.)
        
        'Diagonal' pattern
        --------------
        start_at        (row,col) tuple where the first search coordinate in
                        the pattern will be located.
        row_step        int : The number of rows to move between points in the
                        pattern. Can be negative.
        col_step        int : The number of columns to move between points
                        in the pattern. Can be negative.
        wrap            Bool : if True, when an edge (or within edge_offset
                        of an edge) is encountered, treat the board as if it
                        is periodic. Otherwise go back to the start of the
                        previous diagonal, move over by 'spacing', and proceed.
                        The default it False.
        spacing         The step size between subsequent diagonals. If wrap
                        is False, this has no effect.
        at_end          str, 'reset' or 'reverse'. If 'reset', go to the 
                        start of the pattern, move to the next row (or column,
                        if the pattern first proceeds along rows), and move
                        through a similar pattern from the new start point.
                        If 'reverse', move to the next row or column, reverse
                        the row_step and col_steps, and start a new diagonal
                        pattern.
                        Defaults to 'reset'.
        edge_offset     Leave at least this many spaces between a space in the
                        pattern and the edge of the board. If the value of 
                        start_at would put the first coordinate within 
                        edge_offset of the edge, the edge_offset is respected
                        and all points that are too close to the edge are 
                        not included in the pattern.
                        Defaults to 0 (i.e., the pattern can go up to the 
                        first/last row and columns.)

    Returns
    -------
    coords : list of coordinate tuples (row,col). 
        The element at position i in the list is the ith target that a player
        should fire at in order to follow the desired search pattern.


    """
    
def next_matching_coord_along_vector(board, matches, start_coord, delta,
                                     stop_at = 'match'):
    """
    Returns the coordinate that is along the direction of a vector (delta) 
    beginning at an input start_coord that has a target grid value matching 
    one of an input set of values.
    

    Parameters
    ----------
    board : Board
        The board on which target grid values will be checked.
    matches : list
        The first target grid coordinate (starting from start_coord, proceeding
        along delta) that equals one of these values will be returned.
    start_coord : tuple
        (row,col) tuple. The search begins at this coordinate and proceeds 
        along delta.
    delta : tuple
        A 2-element tuple indicating a direction along the board. One of the
        elements must be zero. If delta does not have length 1, it will be 
        normalized to 1. 
    stop_at : str
        If 'match', the method returns the coordinate of the first target grid
        value that equals one of the values in matches.
        If 'nonmatch', the method returns the coordinate of the first target 
        grid value that DOES NOT equal one of the values in matches.
        
    Returns
    -------
    coord : tuple
        The (row,col) of the first coordinate on the target grid that equals
        one of the values in the matches input, when starting at start_coord
        and proceeding along delta. If no matches are found before the edge
        of the board is encountered, None is returned.

    """
    start_coord = np.array(start_coord)
    delta = np.array(delta)
    if isinstance(matches, str):
        if matches.lower() == 'hit':
            matches = [constants.TargetValue.HIT]
        elif matches.lower() == 'miss':
            matches = [constants.TargetValue.MISS]
        elif matches.lower() == 'unknown':
            matches = [constants.TargetValue.UNKNOWN]
        elif matches.lower() == 'ship':
            matches = ([constants.TargetValue.HIT] 
                       + list(range(1,len(Ship.data)+1)))
    elif not isinstance(matches, list):
        matches = [matches]
    if not (delta == 0).any():
        raise ValueError("Input 'delta' must be a unit vector along either "
                         "the row or column direction.")
    if stop_at.lower() in {'match', 'is'}:
        # continue search while target grid is NOT on of matches
        should_continue_search = lambda x,values: x not in values
    elif stop_at.lower() in {'non-match','nonmatch', 'nomatch', 
                             'isnot', 'is not'}:
        # continue search while target grid IS one of matches
        should_continue_search = lambda x,values: x in values
        
    delta = np.array(delta / np.abs(delta).max(), dtype = int)
    coord = start_coord + delta
    while should_continue_search(
            board.target_grid[coord[0],coord[1]], matches
            ):
        coord += delta
        if (coord < 0).any() or (coord >= board.size).any():
            coord = None
            break
    return coord

def kill_target(board, hit):
    """
    Returns a list of coordinates that are either next to the input hit or next
    to a line of hits adjacent to the input hit, and which is possibly a spot
    on the same ship as the input hit.

    Parameters
    ----------
    board : Board
        The board firing at the target. Target will be selected based on 
        this board's target_grid property.
    hit : tuple
        A (row,col) tuple that was a hit on the opponent's board.

    Returns
    -------
    targets : list
        A list of coordinates that are equally likely to contain a spot on
        the ship at the input 'hit' coordinate. If no targets are found, an
        empty list is returned.

    """
    # find any hits adjacent to hit
    targets = []
    adjacent_hits = board.targets_around(hit, values='hit')
    
        
    return targets
