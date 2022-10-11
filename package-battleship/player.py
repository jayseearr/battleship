#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 23:18:02 2022

@author: jason
"""

import abc
from matplotlib import pyplot as plt
import numpy as np
from battleship import Coord, Board, SHIP_DATA, ShipType, random_choice

# %% Player Class

class Player(abc.ABC):
    """An abstract class that can play a game of Battleship.
    This class provides the superclass for two concrete subclasses:
    AIPlayer and HumanPlayer.
    
    All Player instances must implement the following attributes and methods:
    
    Abstract Properties/Methods:
    - player_type (property)
    - take_turn (method)
    - prepare_fleet (method)
    
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
        """Initialize an instance of the class with the input name.
        This should be called by the __init__ method of a concrete subclass.
        """
        self._player_type = None
        self._name = name
        self._shot_history = []
        self._outcome_history = []
        self._remaining_targets = []
        self._possible_targets = None
        self._board = board
        self._opponent = None
        
        self.game_history = None    # for future use
        self.notes = None           # for future use
        
    def __str__(self):
        """Returns a string indicating player type, strategy, and any strategy
        data.
        """
        return (f"Player {self.name}\n" 
                f"  Type: {self.player_type}\n" 
                f"  Defense: {self.defense}\n" 
                f"  Offense: {self.offense}")
            
    def __repr__(self):
        return f"Player({self._name!r}, {self._board!r})"
            
    def turn_str(self):
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
        res = (SHIP_DATA[ShipType(outcome["sunk_ship_id"])]["name"] + " sunk" 
               if outcome["sunk"] else 
               "hit" if outcome["hit"] else "miss")
        s += f"Result: {res}"
        s += f"Outcome:\n{self.last_outcome()}"
        return s
        
    ### Abstract Property ###
    @property
    @abc.abstractmethod
    def player_type(self):
        """A string describing the player type: AI or Human."""
        pass
    
    ### Abstract Method ###
    @abc.abstractmethod
    def take_turn(self): 
        """Fire a shot at the opponent, get the outcome, and update any
        strategy-related variables."""
        pass
    
    @abc.abstractmethod
    def prepare_fleet(self):
        """Put each ship onto the board."""
        pass
    
    ### Properties ###
    @property
    def name(self):
        """The name of the player (string)."""
        return self._name
    
    @name.setter
    def name(self, name):
        self._name = name
       
    @property
    def offense(self):
        """An offense strategy (subclass of Offense). Optional for 
        instances of HumanPlayer, required for instances of AIPlayer.
        """
        return self._offense
    
    @offense.setter
    def offense(self, offense):
        self._offense = offense

    @property    
    def defense(self):
        """A defense strategy (subclass of Defense). Optional for 
        instances of HumanPlayer, required for instances of AIPlayer.
        """
        return self._defense
    
    @defense.setter
    def defense(self, defense):
        self._defense = defense
    
    @property
    def shot_history(self):
        """A list of coordinates (2-element tuples or instances of Coord) 
        with ith element corresponding to the coordiate targeted in the ith
        shot taken by the player. Reset before the start of a new game using
        the reset() method.
        """
        return [outcome["coord"] for outcome in self.outcome_history]
    
    # def add_shot_to_history(self, coord):
    #     """Adds the input coordinate to the shot history list."""
    #     self._shot_history += [coord]
        
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
        self._remaining_targets = self.board.all_coords(untargeted=True)
        
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
            
    # @property
    # def possible_targets(self):
    #     """The list of potential targets, as determined by the offense
    #     strategy. The chosen target will be randomly selected from this list.
    #     """
    #     return self._possible_targets
    
    @property
    def remaining_targets(self):
        """Returns a list of coordinates that have not yet been targeted during
        the current game.
        """        
        return self._remaining_targets
    
    ### General Methods ###
    
    def reset(self):
        """Removes all ships and hit/miss indicators from the player's board,
        and empties out the shot and outome logs. The opponent remains the 
        same until changed usint set_opponent.
        """
        #self._shot_history = []
        self._outcome_history = []
        if self.board:
            self.board.reset()
            self._remaining_targets = self.board.all_coords()
        if self.offense:
            self.offense.reset()
        
    def copy_history_from(self, other):
        """Copies shot and outcome histories from the input Player instance
        into this Player's histories (overwriting any existing histories).
        Also copies over the remaining_targets, game_history, and notes 
        properties, and sets the possible_targets list to None.
        """
        # self._shot_history = other.shot_history
        self._outcome_history = other.outcome_history
        self._remaining_targets = other.remaining_targets
        # self.possible_targets = None
        self.game_history = other.game_history
        self.notes += other.notes
    
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
        
    def isalive(self): 
        """Returns true if any of this players ships are still afloat."""
        return any(self.board.afloat_ships())
    
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
        if isinstance(target_coord, Coord):
            target_coord = tuple(target_coord)
        outcome = self.opponent.board.incoming_at_coord(target_coord)
        self.board.update_target_grid(target_coord, 
                                     outcome["hit"], 
                                     outcome["sunk_ship_id"])
        # Uncomment if shot_history is not a computed attribute:
        # self.add_shot_to_history(target_coord)    
        self.add_outcome_to_history(outcome)
        if target_coord in self.remaining_targets:
            self.remaining_targets.pop(
                self.remaining_targets.index(target_coord)
                )
        if self.offense:
            self.offense.update(outcome)    # update is impelmented in subclass
        
        return outcome
   
    def place_fleet_using_defense(self, defense):
        """Place the fleet ship-by-ship according to the input defense."""
        placements = defense.fleet_placements(self.board)
        self.board.place_fleet(placements)
        
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
                self.board.place_fleet(self.defense)
            else:
                self.place_fleet_using_defense(self.defense)
            return
        
        for i in range(1,len(SHIP_DATA)+1):
            ok = False
            name = SHIP_DATA[ShipType(i)]["name"]
            while not ok:
                space = Coord(input("Space for {name.upper())} ('r' for random): "))
                if space.lower == 'r':
                    coord, heading = random_choice(
                        self.board.random_placement(
                            SHIP_DATA[ShipType(i)]["length"])
                        )
                else:
                    coord = Coord(space)
                    heading = input("Heading for {name.upper())}: ")                
                if not self.board.is_valid_ship_placement(i, coord, heading):
                    ok = False
                    print("Invalid placement.")
                else:
                    self.board.place_ship(i, coord, heading)
                    print(name + " placed successfully (" 
                          + str(i) + " of " + str(len(SHIP_DATA)) + ").")
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
        # self.offense.possible_targets contains the list of all potential
        # targets from which one actual target was chosen randomly.
        
    def prepare_fleet(self):
        """Place fleet according to the defense strategy."""
        if not self.board:
            self.board = Board()
        if isinstance(self.defense, dict):
            self.board.place_fleet(self.defense)
        else:
            self.place_fleet_using_defense(self.defense)