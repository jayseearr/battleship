#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 12:11:29 2022

@author: jason
"""

#%%
# Stuff to do:
    # * Figure out how to organize Strategies, how to combine them, and how
    #   to instantiate them. Like, how would a user call the API to use existing
    #   classes, and make their own classes?
    #
    # * Separate Player class into AIPlayer and HumanPlayer subclasses.
    
#%%
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from enum import Enum, IntEnum
import abc
import re

#%% Constants

# Hard-coded Constants
# LETTER_OFFSET = 65
# DEFAULT_BOARD_SIZE = 10
MAX_ITER = 1000
    
### Enums ###

class ShipType(IntEnum):
    PATROL = 1
    DESTROYER = 2
    SUBMARINE = 3
    BATTLESHIP = 4
    CARRIER = 5
    
class TargetValue(IntEnum):
    """An Enum used to track hits/misses on a board's target grid.
    Can be compared for equality to integers."""
    UNKNOWN = -2
    MISS = -1
    HIT = 0
    
class Align(Enum):
    """Enum used to identify the directional alignment of ships; either 
    VERTICAL or HORIZONTAL (or any, which allows either direction).
    """
    ANY = 0
    VERTICAL = 1
    HORIZONTAL = 2
    
### Data ###

SHIP_DATA = {ShipType.PATROL: {"name": "Patrol".title(), "length": 2},
             ShipType.DESTROYER: {"name": "Destroyer".title(), "length": 3},
             ShipType.SUBMARINE: {"name": "Submarine".title(), "length": 3},
             ShipType.BATTLESHIP: {"name": "Battleship".title(), "length": 4},
             ShipType.CARRIER: {"name": "Carrier".title(), "length": 5}
             }

#%% Graphics classes

### Text color escapes ###
class fg:
    """Foreground text colors."""
    Esc = "\033"
    White = Esc + "[1;37m"
    Yellow = Esc + "[1;33m"
    Green = Esc + "[1;32m"
    Blue = Esc + "[1;34m"
    Cyan = Esc + "[1;36m"
    Red = Esc + "[1;31m"
    Magenta = Esc + "[1;35m"
    Black = Esc + "[1;30m"
    DarkWhite = Esc + "[0;37m"
    DarkYellow = Esc + "[0;33m"
    DarkGreen = Esc + "[0;32m"
    DarkBlue = Esc + "[0;34m"
    DarkCyan = Esc + "[0;36m"
    DarkRed = Esc + "[0;31m"
    DarkMagenta = Esc + "[0;35m"
    DarkBlack = Esc + "[0;30m"
    Off = Esc + "[0;0m"
    
    @staticmethod
    def rgb(rgb):
        """Returns a text sequence that can be used to set foreground text
        color to the input RGB color (0-255).
        """
        return f"\033[38;2;{rgb[0]};{rgb[1]};{rgb[2]}m"
    
class bg:
    """Background text colors."""
    Esc = "\033"
    White = Esc + "[1;47m"
    Yellow = Esc + "[1;43m"
    Green = Esc + "[1;42m"
    Blue = Esc + "[1;44m"
    Cyan = Esc + "[1;46m"
    Red = Esc + "[1;41m"
    Magenta = Esc + "[1;45m"
    Black = Esc + "[1;40m"
    DarkWhite = Esc + "[0;47m"
    DarkYellow = Esc + "[0;43m"
    DarkGreen = Esc + "[0;42m"
    DarkBlue = Esc + "[0;44m"
    DarkCyan = Esc + "[0;46m"
    DarkRed = Esc + "[0;41m"
    DarkMagenta = Esc + "[0;45m"
    DarkBlack = Esc + "[0;40m"
    Off = Esc + "[0;0m"
    
    @staticmethod
    def rgb(rgb):
        """Returns a text sequence that can be used to set background text
        color to the input RGB color (0-255).
        """
        return f"\033[48;2;{rgb[0]};{rgb[1]};{rgb[2]}m"    

#%% Utility Functions

def random_choice(x, p=None):
    """Returns a single randomly sampled element from the input list, tuple,
    or array."""
    return x[np.random.choice(range(len(x)), p=p)]

def on_board(x, size):
    """Returns True if x is a coordinate on a board with input size.
    Practically, occurs when x is a coordinate or tuple with both elements
    between 0 and size - 1.
    """
    x = tuple(x)    # convert from Coord or list
    return (x[0] >= 0 and x[0] < size and x[1] >= 0 and x[1] < size)

#%% Gameplay Functions

def sim_games(ngames):
    p1 = Player("ai", ("random", "random.hunt"), name="Hunt-1")
    p2 = Player("ai", ("random", "random"), name="Rando-2")
    winner = []
    nturns = []
    for n in range(ngames):
        g = Game(p1,p2,verbose=(ngames==1));
        g.setup() 
        g.play()
        #g.print_outcome()
        if g.winner.name==p1.name:
            winner += [1]
        else:
            winner += [2]
        nturns += [g.turn_count]
        p1.reset()
        p2.reset()
    return np.array(winner), np.array(nturns)

def play_test(strat1, strat2):
    """Plays a single game with players using the two input strategy 
    descriptions. Each input is a tuple containing the player's placement
    and offense strategies, respectively.
    """
    p1 = Player("ai", strat1, name="#1 - " + "/".join(strat1))
    p2 = Player("ai", strat2, name="#2 - " + "/".join(strat2))
    g = Game(p1,p2,verbose=True)
    g.setup()
    g.play()
    return p1.board, p2.board


#%% Coord Class

class Coord:
    """A class that represents a single space on the Battleship board. A
    coord instance may be represented as a pair of row/column indices or as
    a letter/number string.
    
    A new instance of Coord can be created using either row/column list or
    tuple, or a string with a letter and digit.
    
    Instances of this class can be compared for equality to one another 
    (i.e., Coord([9,4]) == Coord("J5") is True).
    """
    
    __LETTER_OFFSET = 65
    
    def __init__(self, a, b=None):
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

        self._rowcol = r,c
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
    
    def __eq__(self, other):
        """Compares the Coord's row/column and value (if any) variables. 
        If they are the same as the second item in the comparison, the two 
        instances are equal.
        """
        return self._rowcol == other.rowcol and self._value == other.value
    
    def __getitem__(self, key):
        """Returns the row or column value of the Coord for key = 0 and key = 1,
        respectively.
        """
        return self._rowcol[key]
    
    def __add__(self, other):
        """Adds another Coord or a 2-element tuple to the Coord object."""
        return Coord((self[0] + other[0], self[1] + other[1]))
    
    def __repr__(self):
        return f'Coord({self.rowcol})'
    
#%% Game Class

class Game:
    
    def __init__(self, player1, player2, 
                 game_id=None, 
                 first_move=1,
                 verbose=True):
        """Creates a Battleship game with two players described by the 
        player1 and player2 inputs. These inputs can be instances of the Player
        class, the string "human" for human players, or a string describing
        the strategy of an AI player. In the latter case, the string should 
        describe both the placement and offensive strategies of the player, 
        separated by a - or / or |.
        Currently supported AI strategies are:
            Offensive Strategies
            'random'    Random placement and offense (no need for separate
                        placement and offensive descriptions if both are to
                        be random; 'random' is the same as 'random/random')
            'random.hunt'   Randomly fire at spaces until a ship is hit, then
                            search around the hit until the ship is found and 
                            sunk.
            'random.hunt.dumb'  As above, but with no smarts about ships being
                                linear.
                                
            Placements
            'cluster'   Place first ship randomly, then each subsequent ship
                        with a probability that descreases as you move away 
                        from the previous ship.
            'edges'     Prefer placement around the edges.
            'isolated'  Randomly placed, but no two ships can touch (on a 
                        diagonal is okay)
            'very.isolated'     As above, but on a diagonal is not okay.
            
        An optional ID string 'game_id' can be added to identify the game.
        
        The 'verbose' input controls whether the outcome of the game is printed
        to the console or not. This defaults to True, but can be turned off
        if many robot vs. robot games are going to be simulated.
        
        The game should not care whether a player is human or robot (so don't
        have it access self.player1.type, for example).
        
        A game is played by creating the Game instance, calling game.setup(),
        then game.play().
        """
        
        self._game_id = game_id
        self._player1 = player1
        self._player2 = player2
        # if isinstance(player1, (HumanPlayer, AIPlayer)):
        #     self.player1 = player1
        # else:
        #     self.player1 = Player(player1)
            
        # if isinstance(player2, (HumanPlayer, AIPlayer)):
        #     self.player2 = player2
        # else:
        #     self.player2 = Player(player2)
            
        self.ready = False
        self.winner = None
        self.loser = None
        self.turn_count = 0
        self.verbose = verbose
        
    @property
    def game_id(self):
        return self._game_id
    
    @game_id.setter
    def game_id(self, value):
        self._game_id = value
        
    @property
    def player1(self):
        return self._player1
    
    @property
    def player2(self):
        return self._player2
    
    def setup(self):
        """Set up a game by placing both fleets."""
        self.player1.opponent = self.player2
        self.player2.opponent = self.player1
        self.player1.prepare_fleet()
        self.player2.prepare_fleet()
        self.ready = (self.player1.board.ready_to_play() and 
                      self.player2.board.ready_to_play())
            
    def play(self, first_move=1):
        """Play one game of Battleship. Each player takes a turn until one
        of them has no ships remaining.
        Returns a tuple containing (winner, loser) Player instances.
        """
        
        if not self.ready:
            print("Game setup not complete.")
            return (None, None)
        
        # Choose which player goes first
        if (isinstance(first_move, str) and 
                first_move == "?" or first_move == "random"):
            first_move = np.random.randint(1,3)
        if first_move == 1:
            first_player, second_player = self.player1, self.player2
        elif first_move == 2:
            first_player, second_player = self.player2, self.player1
            
        # Play until one player has lost all ships
        game_on = True
        self.turn_count = 0
        while game_on:
            first_player.take_turn()
            if self.verbose:
                self.report_turn_outcome(first_player)
            second_player.take_turn()
            if self.verbose: 
                self.report_turn_outcome(second_player)
            self.turn_count += 1
            game_on = first_player.isalive() and second_player.isalive()
            
        # See who won
        if first_player.isalive():
            self.winner, self.loser = first_player, second_player
        else:
            self.winner, self.loser = second_player, first_player
        
        if self.verbose:
            self.print_outcome()
            
        return (self.winner, self.loser)
            
    def report_turn_outcome(self, player):
        """Displays text reporting the target and outcome on the most recent
        turn for the input player.
        """
        outcome = player.last_outcome()
        target = player.last_target()
        name = player.name
        sink = ""
        if outcome["hit"]:
            hit_or_miss = "Hit!"
        else:
            hit_or_miss = "Miss."
        if outcome["sunk_ship_id"]:
            sink = SHIP_DATA[outcome["sunk_ship_id"]]["name"] + " sunk!"
        print(f"Turn {len(player.shot_history)}: Player {name} fired a shot at "
              f"{target}...{hit_or_miss} {sink}")
        
    def print_outcome(self):
        """Prints the results of the game for the input winner/loser players 
        (which include their respective boards)."""
        
        print("")
        print("GAME OVER!")
        print("Player " + self.winner.name + " wins.")
        print("  Player " + self.winner.name + " took " \
              + str(len(self.winner.shot_history)) + " shots, and sank " \
              + str(sum(1 - self.loser.board.afloat_ships())) + " ships.")
        print("  Player " + self.loser.name + " took " \
              + str(len(self.loser.shot_history)) + " shots, and sank " \
              + str(sum(1 - self.winner.board.afloat_ships())) + " ships.")
        print("Game length: " + str(self.turn_count) + " turns.")
        print("(Game ID: " + str(self.game_id) + ")")
        print("")
            
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
        
        self.game_history = None    # for future use
        self.notes = None           # for future use
        
    def __str__(self):
        """Returns a string indicating player type, strategy, and any strategy
        data.
        """
        return f"Player {self.name}\n" \
            f"  Type: {self.player_type}\n" \
            f"  Placement strategy: {type(self.defense)}\n" \
            f"  Offensive strategy: {type(self.offense)}"
            
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
        return self._shot_history
    
    def add_shot_to_history(self, coord):
        """Adds the input coordinate to the shot history list."""
        self._shot_history += [coord]
        
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
        if isinstance(opponent, (HumanPlayer, AIPlayer)):
            self._opponent = opponent
        else:
            raise TypeError("Input must be a Player object.")
            
    @property
    def possible_targets(self):
        """The list of potential targets, as determined by the offense
        strategy. The chosen target will be randomly selected from this list.
        """
        return self._possible_targets
    
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
        self._shot_history = []
        self._outcome_history = []
        self.board.reset()
        self._remaining_targets = self.board.all_coords()
        
    def copy_history_from(self, other):
        """Copies shot and outcome histories from the input Player instance
        into this Player's histories (overwriting any existing histories).
        Also copies over the remaining_targets, game_history, and notes 
        properties, and sets the possible_targets list to None.
        """
        self._shot_history = other.shot_history
        self._outcome_history = other.outcome_history
        self._remaining_targets = other.remaining_targets
        self.possible_targets = None
        self.game_history = other.game_history
        self.notes += other.notes
    
    def last_target(self):  
        """Returns the most recently targeted coordinate, or None if no shot
        has been taken yet.
        """
        if not self.shot_history:
            return None
        else:
            return self.shot_history[-1]
        
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
        outcome = self.opponent.board.incoming_at_coord(target_coord)
        self.board.update_target_grid(target_coord, 
                                     outcome["hit"], 
                                     outcome["sunk_ship_id"])
        self.add_shot_to_history(target_coord)
        self.add_outcome_to_history(outcome)
        if target_coord in self.remaining_targets:
            self.remaining_targets.pop(self.remaining_targets.index(target_coord))

        ### This has to be impelmented in subclass!
        if self.offense:
            self.offense.update(outcome)
        ###
        
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
        rowcol = self.last_target()
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
            return self.place_fleet_using_defense(self.defense)
        
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
        return ok
    
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

        Returns
        -------
        None.
        """
        super().__init__(name)
        
        self.offense = offense
        self.defense = defense
            
        self.show_targets = False
        self._target_select_mode = "text" 
        
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
        return self.place_fleet_using_defense(self.defense)
        
#%% Offense

class Offense(metaclass=abc.ABCMeta):
    """
    All Offense instances must implement the following attributes and methods:
    
    Abstract Properties/Methods:
    - targets
    - update
    - __str__
    
    Properties
    - possible_targets
    
    Other methods
    - __init__
    
    Class methods
    - from_strs
    """
    
    ### Initializer ###
    
    def __init__(self, method=None, **kwargs):
        if method:
            self = Offense.from_strs(method, kwargs)
        self._possible_targets = None
        
    ### Abstract Methods ###
    
    @abc.abstractmethod
    def target(self, board, history):
        """Returns a target coordinate based on the input board and outcome 
        history. This is where the smarts are implemented.
        """
        pass
    
    @abc.abstractmethod
    def update(self, outcome):
        pass
        
    @abc.abstractmethod
    def __str__(self):
        pass
    
    ### Class factory method ###
    
    @classmethod
    def from_strs(cls, method, **kwargs):
        """
        Allowable methods are:
            "random"
            "random.hunter"
            "random.finisher" (not supported)
            "random.dumb.hunter" or "random.hunter.dumb"
            "grid.hunter" (needs a separate spacing = N argument, where N is an integer)          
        """
        method = method.lower().replace(".", "")
        if "dumb" in method:
            smart = False
            method = method.replace("dumb", "")
        else:
            smart = True
            
        if method == "random":
            return RandomOffense() 
        elif method == "randomhunter" or method == "hunter":
            return HunterOffense("random", smart=smart)
        elif method == "gridhunter":
            return HunterOffense("grid", spacing=kwargs["spacing"])
        
  
#%% RandomOffense: Concrete Offense subclass

class RandomOffense(Offense):
    
    """
    All Offense instances must implement the following attributes and methods:
    
    Abstract Properties/Methods:
    - target
    - update
    
    Properties
    - possible_targets
    
    Other methods
    - __init__
    
    Class methods
    - from_strs
    """
    
    ### Initializer ###
    
    def __init__(self):
        self._possible_targets = None
        
    def __str__(self):
        return "Random Offense"
    
    ### Abstract Methods ###
    
    def target(self, board, history):
        """Returns a target coordinate based on the input board and outcome 
        history. This is where the smarts are implemented.
        """
        self.possible_targets = board.all_coords(untargeted=True)
        return random_choice(self.possible_targets)
    
    def update(self, outcome):
        pass
        
    
        
#%% HunterOffense subclass of Offense

class HunterOffense(Offense):
    
    """
    All Offense instances must implement the following attributes and methods:
    
    Abstract Properties/Methods:
    - target
    - update
    
    Properties
    - possible_targets
    
    Other methods
    - __init__
    
    Class methods
    - from_strs
    """
    
    class Mode(Enum):
        HUNT = 1
        KILL = 2
        
    ### Initializer ###
    
    def __init__(self, smart=True, spacing=None, barrage=0):
        """
        Hunter offensive class that hunts for ships using various methods and,
        once a ship is hit, switches over to kill mode and searches the 
        immediate vicinity of the hit until the location of a ship can
        be determined and the ship sunk.

        Parameters
        ----------
        smart : TYPE, optional
            DESCRIPTION. The default is True.
        spacing : TYPE, optional
            DESCRIPTION. The default is None.
        barrage: int, optional
            When hunting for ships, this many shots will be fired in a cluster
            before choosing another random coordinate to target.

        Returns
        -------
        None.

        """
        self._possible_targets = None
        self.smart = smart
        self._mode = self.Mode.HUNT
        self._hits_since_last_sink = []
        self._message = None
        self.hunt_method = "random"   # random, empty, grid
        
        # barrage mode
        self._barrage = None
        self._barrage_count = 0
        
        # grid mode
        self.grid_spacing = spacing
        self.gird_first_target = None
        self.grid_direction = np.array((1,1))   # direction of step
        self.grid_wrap = None
        
    ### Abstract Methods ###
    
    def target(self, board, history):
        """Returns a target coordinate based on the input board and outcome 
        history. This is where the smarts are implemented.
        """
        if self.mode == self.Mode.HUNT:
            targets = self.hunt_targets(board)
            self.message = "hunt mode."
        else:
            hits = self.hits_since_last_sink
            self.message = "kill mode."
            if len(hits) == 1 or not self.smart:
                targets = board.coords_around(hits[-1], untargeted=True)
                self.message += " Searching around last hit."
            elif len(hits) > 1:
                targets = board.colinear_target_coords(hits)
                self.message += " Searching along colinear hits."                    
            else:
                raise Exception("hits_since_last_sink should not be empty.")
             
        if not targets:
            self.mode = self.Mode.HUNT
            self.message += " No viable target found; " \
                "reverting to hunt mode. "
            targets = board.all_coords(untargeted=True)
        
        self.possible_targets = board.all_coords(untargeted=True)
        return random_choice(self.possible_targets)
    
    def update(self, outcome):
        """Updates instance state (i.e., attributes) with the results of the
        input outcome.
        """
        last_target = outcome["coord"]
        self.last_target = last_target
        if outcome["sunk_ship_id"]:
            self.mode = self.Mode.HUNT
            self.hits_since_last_sink = []
            self.barrage_count = 0
        elif outcome["hit"]:
            self.mode = self.Mode.KILL
            self.hits_since_last_sink += [last_target]
            self.barrage_count = 0
        else:
            if self.barrage:
                self.barrage_count += 1           
            
        self.message = None 
        
    def __str__(self):
        s = (f"Hunter Offense\n  Smart: {self.smart}\n"
             f"  Hunt method: {self.hunt_method}\n"
             f"  Current mode: {self.mode}\n")
        if self.barrage:
            s += (f"  Barrate shot {self.barrage_count} of "
                  "{self.barrage}.")
            
    ### Properties ###
    @property
    def mode(self):
        return self._mode
    
    @mode.setter
    def mode(self, mode):
        if mode == self.Mode.HUNT or mode == self.Mode.KILL:
            self._mode = mode
        else:
            raise ValueError("mode must be Mode.HUNT or Mode.KILL.")
            
    @property
    def hits_since_last_sink(self):
        return self._hits_since_last_sink
    
    @hits_since_last_sink.setter
    def hits_since_last_sink(self, hits):
        self._hits_since_last_sink = hits
        
    @property
    def message(self):
        return self._message
    
    @message.setter
    def message(self, message):
        self._message = message
        
    @property
    def barrage(self):
        return self._barrage
    
    @barrage.setter
    def barrage(self, n):
        self._barrage = n
    
    @property
    def barrage_count(self):
        return self._barrage_count
    
    @barrage_count.setter
    def barrage_count(self, n):
        self._barrage_count = n
        
    ### Methods ###
    def barrage_targets(self, board, prev_barrage_targets):
        """Returns a list of potential targets based on the location of targets
        in the current barrage.
        """
        if len(prev_barrage_targets) > 1:
            t1 = board.coords_around(prev_barrage_targets[-1],
                                     diagonal = True)
            t2 = board.coords_around(prev_barrage_targets[-2],
                                     diagonal = True)
            targets = [t for t in t1 if t in t2]
        else:
            targets = board.coords_around(self.shot_history[-1],
                                          diagonal = False)  
        return targets
    
    def open_ocean_targets(self, board):
        """Returns a list of potential targets and relative probabilities
        based on how 'isolated' each target is. A target is more isolated if
        it is far from its closest previously targeted coordinate.
        """
        untargeted = board.all_coords(untargeted=True)
        target_rows, target_cols =  \
            np.where(board.target_grid > TargetValue.UNKNOWN)
        prob = np.zeros(len(untargeted))
        for (i,coord) in enumerate(untargeted):
            dr = target_rows - coord[0]
            dc = target_cols - coord[1]
            prob[i] = (np.sqrt(dr**2 + dc**2)).min()
        
        prob = prob**2
        
        return (untargeted, prob)
            
    def hunt_targets(self, board):
        """Returns a list of potential targets based on the hunt_method and
        the state of the board.
        """
        mid_barrage = (self.barrage > 0 and
                       self.barrage_count > 0)
        if mid_barrage:
            prev_barrage_targets = self.shot_history[-(self.barrage_count):]
            return self.barrage_targets(board, prev_barrage_targets)
        if self.hunt_method == "random":
            targets = board.all_coords(untargeted=True)
        elif self.hunt_method == "empty":
            # find coord with max distance to nearest shot
            targets, prob = self.open_ocean_targets(board)
            
        elif self.hunt_method == "grid":
            # self.grid_spacing = spacing
            # self.gird_first_target = None
            # self.grid_direction = np.array((1,1))   # direction of step
            # self.grid_wrap = None
            if not self.shot_history:
                targets = [self.grid_first_target]
            else:
                ds = self.grid_direction
                step = self.grid_spacing * ds
                t = self.last_target() + step
                if not on_board(t, board.size):
                    if self.grid_wrap == "shift":
                        t = self.last_target() + np.array(ds[1],ds[0])
                        self.grid_direction = -self.grid_direction
                    elif self.grid_wrap == "offset":
                        pass
                    elif self.grid_wrap == "wrap":
                        RC = [(r,c) for r in range(board.size) 
                              for c in range(board.size)]
                        idx = RC.index(self.last_target())
                        idx += board.size * ds[0] + ds[1]
                        if idx > board.size**2:
                            t = None
                        else:
                            t = RC[idx]
                if t and on_board(t, board.size):
                    targets = [t]
                    prob = np.array((1))
                else:
                    targets = []
                    prob = []
                
        return targets, prob
    
    def heatmap(self):
        """Returns a 2d array the same size as the board with
        the relative probability of targeting each coordinate.
        """
        pass
        

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
    
    @classmethod
    def from_strs(cls, method, **kwargs):
        """
        Allowable methods are:
            "random"
            "cluster[ed]"
            "isolated"
        
        Other parameters can be entered by keyword:
            align = 'horizontal' or 'vertical' (or 'any', the default)
            diagonal = True / False (Isolated Strategy only)
            max_sep = True / False (Isolated Strategy only)
            
        """
        if "alignment" in kwargs:
            alignment = kwargs["alignment"]
        else:
            alignment = None
            
        if "diagonal" in kwargs:
            diagonal = kwargs["diagonal"]
        else:
            diagonal = False
            
        if "max_sep" in kwargs:
            max_sep = kwargs["max_sep"]
        else:
            max_sep = False
            
        if "separation" in kwargs:
            sep = kwargs["separation"]
        else:
            sep = None
            
        # Choose defense and pass arguments
        if re.search("isolate", method, re.IGNORECASE):
            if re.search("max", method):
                max_sep = True
            return IsolatedDefense(alignment=alignment, diagonal=diagonal,
                                   max_sep=max_sep, separation=sep)
        
        elif re.search("cluster", method, re.IGNORECASE):
            return ClusterDefense(alignment=alignment)
        elif re.search("random", method, re.IGNORECASE):
            return RandomDefense()
        
    ### Other methods ###
    def fleet_placements(self, board):
        """Return a dictionary of placements for each ship type. Each value 
        contains a tuple with (coordinate, heading) for the respective 
        ship_type key.
        """
        
        placements = {}
        newboard = Board(board.size)
        for ship_type in range(1,len(SHIP_DATA)+1):
            coord, heading = self.placement_for_ship(newboard, ship_type)
            count = 0
            while coord == None and heading == None and count <= MAX_ITER:
                coord, heading = self.placement_for_ship(newboard, ship_type) 
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
            
    
#%% RandomDefense: Concrete Defense subclass

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
    
    def __init__(self, alignment=None, edges=True):
        """
        

        Parameters
        ----------
        alignment : str, optional
            The direction of all ships in the fleet; can by "Any", "Vertical", 
            or "Horizontal". The default is "Any".

        Returns
        -------
        None.

        """
        super().__init__(alignment, edges)
        
    #@abc.abstractmethod
    def placement_for_ship(self, board, ship_type):
        """Returns a (coord, heading) tuple for the input ship type. The
        placement is based on the algorithm for this particular subclass of
        Defense. In this case, the algorithm favors placements that
        minimize the sum of distances between all coordinates of the input
        ship to all coordinates of all other ships already on the board.
        """
        ok = False
        count = 0
        while not ok:
            coord = random_choice(board.all_coords(unoccuppied=True))
            heading = board.random_heading(self.fleet_alignment)
            ok = (board.is_valid_ship_placement(ship_type, coord, heading) and 
                  count <= MAX_ITER)
            if not self.edges:
                dr,dc = board.relative_coords_for_heading(
                            heading, SHIP_DATA[ShipType(ship_type)]["length"])
                rows, cols = dr + coord[0], dc + coord[1]
                ok = ok and not (np.any(rows < 1) or np.any(cols < 1) or 
                          np.any(rows >= board.size - 1) or 
                          np.any(cols >= board.size - 1))
            count += 1
        if count >= MAX_ITER:
            raise Exception("Max iterations encountered when trying to place"
                            " ship randomly.")
        return (coord, heading)
    
#%% ClusterDefense: Concrete Defense subclass
    
class ClusterDefense(Defense):
    """
    Abstract Properties/Methods:
    - fleet_placements
    
    Properties
    - fleet_alignment
    
    Other methods
    - __init__
    - placement_for_ship
    """
    
    def __init__(self, alignment=None, edges=True):
        """

        Parameters
        ----------
        alignment : str, optional
            The direction of all ships in the fleet; can by "Any", "Vertical", 
            or "Horizontal". The default is "Any".

        Returns
        -------
        None.

        """
        super().__init__(alignment, edges)
        
    #@abc.abstractmethod
    def placement_for_ship(self, board, ship_type):
        """Returns a (coord, heading) tuple for the input ship type. The
        placement is based on the algorithm for this particular subclass of
        Defense. In this case, the algorithm favors placements that
        minimize the sum of distances between all coordinates of the input
        ship to all coordinates of all other ships already on the board.
        """
        new_ship = Ship(ship_type)
        all_placements = board.all_valid_placements(new_ship.length,
                                                    alignment = self.fleet_alignment,
                                                    edges = self.edges)
        # if self.fleet_alignment == Align.VERTICAL:
        #     all_placements = [p for p in all_placements if 
        #                       p[1] == "N" or p[1] == "S"]
        # elif self.fleet_alignment == Align.HORIZONTAL:
        #     all_placements = [p for p in all_placements if 
        #                       p[1] == "E" or p[1] == "W"]
        prob = np.zeros(len(all_placements))
        for (i, place) in enumerate(all_placements):
            if len(board.fleet) == 0:
                prob[i] = 1.
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
                    
        return random_choice(all_placements, p=prob/np.sum(prob))
        
#%% IsolatedDefense: Concrete Defense subclass
        
class IsolatedDefense(Defense):
    """
    Abstract Properties/Methods:
    - fleet_placements
    
    Properties
    - fleet_alignment
    - diagonal
    - maximize_separation
    
    Other methods
    - __init__
    - placement_for_ship
    """
    
    ### Initializer ###
    def __init__(self, alignment=None, edges=True, diagonal=True, max_sep=False,
                 separation=1, ):
        """
        Parameters
        ----------
        alignment : str, optional
            The direction of all ships in the fleet; can by "Any", "Vertical", 
            or "Horizontal". The default is "Any".
        diagonal : bool, optional
            True if ships are allowed to occupy diagonally adjacent coordinates
            (such as C5 and D6). This parameter is the difference between
            'Isolated' and 'Very Isolated' strategies; it is True for 'Isolated',
            and False for 'Very Isolated'. The default is True.
        max_sep : bool, optional
            Maximize separation between ships. When True, ships are placed in
            a way that favors maximal separation between all other ships that
            have already been placed on the board. This parameter is True for
            'Max Isolated' mode, and False otherwise. Setting this parameter
            to True overrides the diagonal parameter. The default is False.
        separation : int, optional
            The minimum number of spaces between ships. When diagonal is True,
            ships can still be diagonally adjacent by 'separation' spaces.

        Returns
        -------
        None.

        """
        super().__init__(alignment, edges)
        self.diagonal = diagonal
        self.maximize_separation = max_sep
        
    ### Abstract Methods ###
    
    #@abc.abstractmethod
    def placement_for_ship(self, board, ship_type):
        """Returns a (coord, heading) tuple for the input ship type. The
        placement is based on the algorithm for this particular subclass of
        Defense. In this case, the algorithm favors placements that either
        forbids ships from touching (basic isolated), forbids them from 
        touching even on diagonal spaces (very isolated), or maximizes the
        distances between ships (max isolated).
        In max isolated, the algorithm is more likely to choose placements 
        where the sum of distances between the placement in question and the
        placement of all ships already on the board is large.
        """       
        new_ship = Ship(ship_type)
        all_placements = board.all_valid_placements(new_ship.length,
                                                    distance = 1,
                                                    diagonal = self.diagonal,
                                                    alignment = self.fleet_alignment,
                                                    edges = self.edges)
        if len(all_placements) == 0:
            return None,None
        if self.maximize_separation:
            prob = np.zeros(len(all_placements))
            for (i, place) in enumerate(all_placements):
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
                prob[i] = d2
            if np.all(prob == 0):
                prob = np.ones(prob.shape)
            
            return random_choice(all_placements, p=prob/np.sum(prob))
        
        else:
            return random_choice(all_placements)
    
    ### Properties ###
    
    @property
    def diagonal(self):
        """Diagonal is True if ships are allowed to touch on diagonally-
        connected spaces (for example, A1 and B2), and False otherwise.
        """
        return self._diagonal
    
    @diagonal.setter
    def diagonal(self, value):
        # Force _diagonal to be True or False
        self._diagonal = (value == True)
        
    @property
    def maximize_separation(self):
        return self._maximize_separation
    
    @maximize_separation.setter
    def maximize_separation(self, value):
        # Force _maximize_separation to be True or False
        self._maximize_separation = (value == True)    
        
#%%
class Board:
    """An instance of a Battleship game board. Each board has a fleet of 5 
    ships, a 10 x 10 ocean grid upon which ships are arranged, and a 10 x 10 
    target grid where hit/miss indicators are placed to keep track of 
    opponent ships.
    """

    def __init__(self, size=10):
        self.size = size
        self.fleet = {}
        self.ocean_grid = np.zeros((size, size), dtype=np.int16)      
        self.target_grid = np.ones((size, size), 
                                   dtype=np.int8) * TargetValue.UNKNOWN
        
    def __str__(self):
        """Returns a string that uses text to represent the target map
        and grid (map on top, grid on the bottom). On the map, an 'O' represents
        a miss, an 'X' represents a hit, and '-' means no shot has been fired
        at that space. 
        On the grid, a '-' means no ship, an integer 1-5 means a ship slot with
        no damage, and a letter means a slot with damage (the letter matches
        the ship name, so P means a damaged patrol boat slot).
        """
        s = "\n  "
        s += (" ".join([str(x) for x in np.arange(self.size) + 1]) 
              + "\n")
        for (r,row) in enumerate(self.target_grid):
            s += (Coord(r,0).lbl[0]
                  + " "
                  + " ".join(["-" if x == TargetValue.UNKNOWN else 
                              'O' if x == TargetValue.MISS else
                              'X' if x == TargetValue.HIT else 
                              str(x) for x in row]) 
                  + "\n")
        s += ("\n  " 
              + " ".join([str(x) for x in np.arange(self.size) + 1]) 
              + "\n")
        for (r,row) in enumerate(self.ocean_grid):
            s += (Coord(r,0).lbl[0]
                  + " "
                  + " ".join(["-" if x == 0 else 
                              str(x) for x in row]) 
                  + "\n")
        return s
    
    def reset(self):
        """Removes all ships and pegs from the board."""
        board = Board(self.size)
        self.fleet = board.fleet
        self.target_grid = board.target_grid
        self.ocean_grid = board.ocean_grid
        
    # Coordinate functions        
    def all_coords(self, untargeted=False, unoccuppied=False,
                   targeted=False, occuppied=False):
        """Returns a list of all coordinates on the board.
        If untargeted is True, only coords that have not been targeted are 
        returned. If unoccuppied is True, only coords with no ships on the
        board's ocean grid are returned."""
        if np.sum((untargeted, unoccuppied, targeted, occuppied)) > 1:
            raise ValueError("Only one of the (un)occuppied or (un)targeted" \
                             " inputs can be set to True.")
        indices = None
        exclude = []
        if untargeted:
            xr,xc = np.where(self.target_grid != TargetValue.UNKNOWN)
            exclude = list(zip(xr, xc))
        elif unoccuppied:
            xr,xc = np.where(self.ocean_grid != 0)
            exclude = list(zip(xr, xc))
        elif targeted:
            r,c = np.where(self.target_grid != TargetValue.UNKNOWN)
            indices = list(zip(r,c))
        elif occuppied:
            r,c = np.where(self.ocean_grid != 0)
            indices = list(zip(r,c))

        if indices == None:
            indices = [(r,c) for r in range(self.size) 
                       for c in range(self.size) 
                       if (r,c) not in exclude]
        return indices
        #return [Coord(rc) for rc in indices]
    
    def unoccuppied_coords(self):
        """Returns the row/column for all coordinates that are not occuppied by
        a ship."""
        return self.all_coords(unoccuppied=True)

    def occuppied_coords(self):
        """Returns the indices of all ship-occuppied coordinates on the board 
        (regardless of damage/sunk status).
        """
        return self.all_coords(occuppied=True)
    
    def targeted_coords(self):
        """Returns the indices of all previously targeted indices on the target
        grid (i.e., the location of all pegs, white or red).
        """
        return self.all_coords(targeted=True)
    
    def untargeted_coords(self):
        """Returns the indices of all non targeted indices on the target
        grid (i.e., the locations with no pegs).
        """
        return self.all_coords(untargeted=True)
    
    def relative_coords_for_heading(self, heading, length=1):
        """Returns row and column arrays with displacements corresponding to
        a ship facing the input heading and with input length. The first slot 
        on the ship will be at relative position (0,0), the second at (-1,0) 
        for north heading, (0,-1) for east, (1,0) for south, and (0,1) for
        west. The third and subsequent positions will be at 2x this vector,
        3x, etc.
        Useful for determining which rows/cols a ship will occupy based on
        a certain heading.
        """
        if (heading.upper() in ["NORTH", "SOUTH", "EAST", "WEST"] or
            heading.upper() in ["UP", "DOWN", "LEFT", "RIGHT"]):
            heading = heading[0].upper()
            
        if heading == "N" or heading == "D":
            ds = (1,0)
        elif heading == "S" or heading == "U":
            ds = (-1,0)
        elif heading == "E" or heading == "L":
            ds = (0,-1)
        elif heading == "W" or heading == "R":
            ds = (0,1)
        else:
            raise ValueError(("heading input must be North, South, East, \
                              or West (first letter is okay, and everything \
                              is case insensitive."))
        if length == 1:
            return ds
        else:
            ds = np.vstack(([0,0], 
                            np.cumsum(np.tile(ds, (length-1,1)), axis=0)))
            return (ds[:,0], ds[:,1])
       
    def coords_around(self, coord, diagonal=False, 
                     untargeted=False, unoccuppied=False):
        """Returns all coordinates that surrounds the input coordinate.
        By default, the random coordinate will be touching the input coordinate
        (so input C5 could result in one of B5, C4, C6, or D5). If
        diagonal is True, the 4 spaces that touch the input coordinate diagaonally
        (the ones NW, NE, SW, SE of the target) are also possible.
        If untargeted is True, then only untargeted coordinates will be chosen.
        """
        if unoccuppied:
            exclude = self.all_coords(occuppied=True)
        if untargeted:
            exclude = self.all_coords(targeted=True)
        else:
            exclude = []
        
        if diagonal:
            rows = coord[0] + np.array([0, -1, -1, -1, 0, 1, 1, 1])
            cols = coord[1] + np.array([1, 1, 0, -1, -1, -1, 0, 1])
        else:
            rows = coord[0] + np.array([0, -1, 0, 1])
            cols = coord[1] + np.array([1, 0, -1, 0])
        ivalid = ((rows >= 0) * (rows < self.size) * 
                  (cols >= 0) * (cols < self.size))
        rows = rows[ivalid]
        cols = cols[ivalid]
        return [(rows[i], cols[i]) for i in range(len(rows)) if 
                   (rows[i], cols[i]) not in exclude]
    
    def colinear_target_coords(self, coords, 
                               untargeted=False, unoccuppied=False):
        """Returns the two coordinates at the end of the line made
        by the input coordinates.
        """
        if unoccuppied:
            exclude = self.occuppied_indices()
        if untargeted:
            exclude = self.targeted_indices()
        else:
            exclude = []
            
        ds = (np.diff(np.array(coords), axis=0))
        ia,ib = sorted([coords[0], coords[-1]])
        choices = []
        
        # If vertical, find the next miss or unknown N/S of the line
        if np.any(ds[:,0] != 0):
            
            while (ia[0] >= 0 and ia[0] < self.size and 
                   self.target_grid[ia] != TargetValue.UNKNOWN):
                ia = (ia[0] - 1, ia[1])
            while (ib[0] >= 0 and ib[0] < self.size and 
                   self.target_grid[ib] != TargetValue.UNKNOWN):
                ib = (ib[0] + 1, ib[1])
            if ia[0] >= 0:
                choices += [ia]
            if ib[0] < self.size:
                choices += [ib]
                
        # If horizontal, find the next miss or unknown E/W of the line
        elif np.any(ds[:,1] != 0):
            while (ia[1] >= 0 and ia[1] < self.size and 
                   self.target_grid[ia] == TargetValue.HIT):
                ia = (ia[0], ia[1] - 1)
            while (ib[1] >= 0 and ib[1] < self.size and 
                   self.target_grid[ib] == TargetValue.HIT):
                ib = (ib[0], ib[1] + 1)
            if (ia[1] >= 0 and 
                    self.target_grid[ia] == TargetValue.UNKNOWN):
                choices += [ia]
            if (ib[1] < self.size and 
                    self.target_grid[ib] == TargetValue.UNKNOWN):
                choices += [ib]
        
        return [c for c in choices if c not in exclude]
                
    # Ship access functions
    def afloat_ships(self):
        """Returns a list of the Ship instances that are still afloat."""
        return np.array([ship.afloat() for ship in list(self.fleet.values())])
            
    def coord_for_ship(self, ship):
        """Returns the row/col coordinates occuppied by the input Ship instance."""
        r,c = np.where(self.ocean_grid == ship.ship_id)
        return list(zip(r,c))
    
    def ship_at_coord(self, coord):
        """Returns the Ship object placed at the input coordinate.
        If no ship is present at the input space, returns None.
        """        
        ship_id = self.ocean_grid[coord[0],coord[1]]
        if ship_id == 0:
            return None
        return self.fleet[ShipType(ship_id)]
    
    def damage_at_coord(self, coord):
        """Returns the damage at the slot corresponding to the input coordinate.
        If no ship is present or if the ship is not damaged at that particular
        slot, returns 0. Otherwise, if that spot has been hit, returns 1.
        """
        ship = self.ship_at_coord(coord)
        if not ship:
            return 0
        coords = self.coord_for_ship(ship)
        return ship.damage[coords.index(coord)]
    
    # Ship placement functions        
    def all_valid_placements(self, length, distance=0, diagonal=True,
                             edges=True, alignment=Align.ANY):
        """Returns a list of all possible placements (consisting of row, 
        column, and heading) on the board. The distance parameter is the 
        minimum separation between ships.
        If the diagonal input is True, the four coordinates 
        diagonally touching existing ships are considered valid (and invalid
        if diagonal is False).
        The returned list of placements is in the form of a list of tuples
        ((row, column), heading).
        """
        headings = {"N": (1,0), "S": (-1,0), "E": (0,-1), "W": (0,1)}
        
        if alignment == Align.VERTICAL:
            allowed_headings = ["N", "S"]
        elif alignment == Align.HORIZONTAL:
            allowed_headings = ["E", "W"]
        else:
            allowed_headings = ["N","S","E","W"]
        # Determine whether edge coordinates can be occuppied
        rcmin = 1 - edges
        rcmax = self.size - (1 - edges) - 1
        
        unoccuppied = self.all_coords(unoccuppied=True)
        placements = []
        for unocc in unoccuppied:
            for h in allowed_headings:
                ds = np.vstack(([0,0], np.cumsum(np.tile(
                    headings[h], (length-1,1)), axis=0)))
                rows,cols = unocc[0] + ds[:,0], unocc[1] + ds[:,1]
                ok = True
                if (np.any(rows < rcmin) or np.any(rows > rcmax) or 
                    np.any(cols < rcmin) or np.any(cols > rcmax) or
                    np.any(self.ocean_grid[rows,cols])):
                        ok = False
                if distance > 0 and ok:
                    dmin = self.size
                    for (r,c) in zip(rows, cols):
                        for other_ship in list(self.fleet.values()):
                            other_coords = np.array(self.coord_for_ship(other_ship))
                            delta = other_coords - np.tile((r,c), 
                                                           (other_coords.shape[0],1))
                            d = np.sqrt(np.sum(delta**2, axis=1))
                            dmin = np.min((np.min(d), dmin))
                    if diagonal:
                        dmin = np.ceil(dmin)
                    else:
                        dmin = np.floor(dmin)
                    ok = (dmin > distance)
                if ok:
                    placements += [(unocc, h)]
        return placements
    
    def place_fleet(self, placements):
        """Places all ships according to the data in the input dictionary,
        which has the following format:
        """
        for (k,place) in placements.items():
            self.place_ship(k, place["coord"], place["heading"])
            
    def place_ship(self, ship_desc, coord, heading):
        """Places a ship corresponding to the input ship_desc (which may be an
        integer 1-5 or a string like "Patrol", "Carrier", etc.)
        on the grid at the input indices and the input heading 
        (a string, "N", "S", "E", or "W"). The ship is added to the board's fleet.
        Returns True if the placement was successful (i.e., if the input 
        placement is valid) and False otherwise, in which case the ship is
        NOT added to the fleet.
        """
        if isinstance(ship_desc, str):
            ship_id = [k.value for k in SHIP_DATA.keys() if 
                       SHIP_DATA[k]["name"] == ship_desc.title()][0]
        else:
            ship_id = ship_desc
            
        ship = Ship(ship_id)
        if not self.is_valid_ship_placement(ship_id, coord, heading):
            raise Exception(f"Cannot place {ship.name} at {coord}.")
            return False
        else:
            self.fleet[ShipType(ship_id)] = ship
            drows,dcols = self.relative_coords_for_heading(heading, ship.length)
            self.ocean_grid[coord[0] + drows, coord[1] + dcols] = ship_id
            return True
        
    def is_valid_ship_placement(self, ship_id, coord, heading):
        """Returns True if the input ship can be placed at the input spot with
        the input heading, and False otherwise.
        The ship/space/heading is valid if the ship:
            1. Is not a duplicate of an existing ship
            2. Does not hang off the edge of the grid
            3. Does not overlap with an existing ship on the grid
        """
        length = SHIP_DATA[ShipType(ship_id)]["length"]
        if ShipType(ship_id) in self.fleet.keys():
            return False
        drows,dcols = self.relative_coords_for_heading(heading, length)
        rows,cols = coord[0] + drows, coord[1] + dcols
        if (np.any(rows < 0) or np.any(rows >= self.size) or 
            np.any(cols < 0) or np.any(cols >= self.size)):
                return False
        occuppied = self.all_coords(occuppied=True)
        for i in range(len(rows)):
            if (rows[i],cols[i]) in occuppied:
                return False
        return True
    
    def ready_to_play(self):
            """Returns True if the full fleet is placed on the grid, all ships are
            undamaged, and the target map is empty. Returns False otherwise (which
            indicates that either setup is not complete, or the game has already
            started).
            """
            if not np.all(self.target_grid == TargetValue.UNKNOWN):
                return False
            for ship in self.fleet.values():
                if np.any(ship.damage > 0):
                    return False
            ship_ids = [ship.ship_id for ship in self.fleet.values()]
            if sorted(ship_ids) != list(range(1, len(SHIP_DATA)+1)):
                return False
            return True
        
    # Random coordinate functions
    def random_coord(self, unoccuppied=False, untargeted=False):
        """Returns a random row/col coordinate from the board.
        
        If the unoccuppied input is True, the coordinate will be one that does 
        not contain a ship on this board's ocean grid (note that if this is 
        used publically, this could reveal ship locations).
                    
        If the untargeted input is True, the coordinate will be one that has
        not been shot at from this board (i.e., no peg on the target grid).
        """
        if unoccuppied and untargeted:
            raise ValueError("Inputs unoccuppied and untargeted cannot both \
                             be true.")
        if unoccuppied:
            exclude = self.all_coords(occuppied=True)
        elif untargeted:
            exclude = self.all_coords(targeted=True)
        else:
            exclude = []
        indices = [(i,j) for i in range(self.size) 
                   for j in range(self.size) 
                   if (i,j) not in exclude]
        return indices[np.random.choice(range(len(indices)))]
    
    def random_heading(self, alignment=None):
        """Returns a random heading: N/S/E/W."""
        if alignment == None or alignment == Align.ANY:
            return(np.random.choice(['N','S','E','W']))
        
        if isinstance(alignment, str):
            alignment = alignment.upper()
            for a in ["NORTH","SOUTH","EAST","WEST"]:
                alignment = alignment.replace(a, a[0])
                for a in ["/", "-", "|"]:
                    alignment = alignment.replace(a, "")
        if (alignment == "NS" or alignment == "SN" or alignment == "UD" or 
            alignment == "DU" or alignment == Align.VERTICAL):
            return(np.random.choice(['N','S']))
        elif (alignment == "EW" or alignment == "WE" or alignment == "LR" or 
              alignment == "RL" or Align.HORIZONTAL):
            return(np.random.choice(['E','W']))
    
    # Combat functions
    def update_target_grid(self, target_coord, hit, ship_id):
        """Updates the target map at the input space with the hit informaton
        (hit and ship_id, if a ship was sunk) provided as inputs.
        Prints a message to the console if the target space has already been
        targeted.
        """
        if isinstance(target_coord, Coord):
            target_coord = target_coord.rowcol
            
        prev_val = self.target_grid[target_coord]
        if hit:
            if ship_id:
                self.target_grid[target_coord] = ship_id
            else:
                self.target_grid[target_coord] = TargetValue.HIT   
        else:
            self.target_grid[target_coord] = TargetValue.MISS
        if prev_val != TargetValue.UNKNOWN:
            print(f"Repeat target: {Coord(target_coord).lbl}")
            
    def incoming_at_coord(self, coord):
        """Determines the outcome of an opponent's shot landing at the input
        coord. If there is a ship at that location, the outcome is a hit and
        the ship's slot corresponding to the input space is damaged. 
        If the hit means that all slots in a ship are damaged, the ship is sunk.
        If there is no ship at the location, the outcome is a miss.
        In either case, if the space has previously been targeted, the outcome
        will return a 'repeat target' message.
        The returned outcome is a dict with the following key/value pairs:
            hit: True/False
            sunk: True/False
            sunk_ship_id: ShipId Enum
            repeat_target: True/False
            message: list containing the following, with optional parts in ():
                ['hit'/'miss', ('[shipname] sunk'), ('repeat target')]
        """
        if isinstance(coord, Coord):
            coord = coord.rowcol
        ship = self.ship_at_coord(coord)
        if not ship:
            outcome = {"hit": False, "coord": coord, 
                       "sunk": False, "sunk_ship_id": None, "message": []}
        else:
            outcome = {"hit": True, "coord": coord, "sunk": False, 
                       "sunk_ship_id": None, "message": []}
            ship_coords = self.coord_for_ship(ship)
            dmg_slot = [i for i in range(len(ship_coords)) 
                      if ship_coords[i] == coord]
            if len(dmg_slot) != 1:
                raise Exception(f"Could not determine damage location for \
                                target {coord}")
            dmg = self.fleet[ship.ship_id].hit(dmg_slot[0])
            msg = []
            if dmg > 1:
                msg += [f"Repeat target: {coord}."]
            if self.fleet[ship.ship_id].sunk():
                msg += [f"{ship.name} sunk."]
                outcome["sunk"] = True
                outcome["sunk_ship_id"] = ship.ship_id
        return outcome
    
    # Test function
    def test(self, targets=True):
        self.place_ship(1, "J9", "R") 
        self.place_ship(2, "E8", "N") 
        self.place_ship(4, "G2", "W") 
        self.place_ship(3, "B6", "W") 
        self.place_ship(5, "A5", "N")
        
        if targets:
            self.update_target_grid((0,4), True, None)
            self.update_target_grid((0,3), True, None)
            self.update_target_grid((0,5), True, ShipType(3))
            self.update_target_grid((2,8), False, None)
            self.update_target_grid((9,7), False, None)
            self.update_target_grid((6,7), False, None)
            self.update_target_grid((4,4), False, None)
            self.update_target_grid((3,2), False, None)
            self.update_target_grid((1,6), False, None)
            
            self.fleet[ShipType(2)].hit(0)
            self.fleet[ShipType(2)].hit(1)
            self.fleet[ShipType(2)].hit(2)
            self.fleet[ShipType(4)].hit(3)
            self.fleet[ShipType(5)].hit(2)
        
    # Visualization functions
    def ocean_grid_image(self):
        """Returns a matrix corresponding to the board's ocean grid, with
        blank spaces as 0, unhit ship slots as 1, and hit slots as 2.
        """
        im = np.zeros((self.size, self.size))
        for ship_id in self.fleet:
            ship = self.fleet[ship_id]
            dmg = ship.damage
            rows,cols = zip(*self.coord_for_ship(ship))
            im[rows,cols] = dmg + 1
        return im

    def target_grid_image(self):
        """Returns a matrix corresponding to the board's vertical grid (where 
        the player keeps track of their own shots). 
        The matrix has a 0 for no shot, -1 for miss (white peg), 1 for a hit, 
        and 10 + shipId (11-15) if the spot is a hit on a known ship type."""
        return self.target_grid
        
    def ship_rects(self):
        """Returns a dictionary with keys equal to ShipId and values equal to
        the rectangles that bound each ship on the grid. The rectangles have
        format [x, y, width, height].
        """
        rects = {}
        for ship in list(self.fleet.values()):
            rows, cols = zip(*self.coord_for_ship(ship))
            rects[ship.ship_id] = np.array((np.min(cols)+0.5, 
                                            np.min(rows)+0.5,
                                            np.max(cols) - np.min(cols) + 1, 
                                            np.max(rows) - np.min(rows) + 1))
        return rects
        
    def color_str(self):
        """Returns a string in color that represents the state of the board.
        to a human player.
        """
        
        ocean = "\x1b[1;44;31m "
        ship_hit = "\x1b[1;41;37m"
        ship_no_hit = "\x1b[2;47;30m"
        red_peg = "\x1b[1;44;31mX"
        red_peg_sunk = "\x1b[1;44;31m"
        white_peg = "\x1b[1;44;37m0"
        
        s = "\n  "
        s += (" ".join([str(x) for x in np.arange(self.size) + 1]) 
              + "\n")
        for (r,row) in enumerate(self.target_grid):
            s += (Coord(r,0).lbl[0]
                  + " "
                  + " ".join([ocean if x == TargetValue.UNKNOWN else 
                              white_peg if x == TargetValue.MISS else
                              red_peg if x == TargetValue.HIT else 
                              (red_peg_sunk + str(x)) for x in row]) 
                  + "\033[0;0m\n")
        s += ("\n  " 
              + " ".join([str(x) for x in np.arange(self.size) + 1]) 
              + "\n")
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
        """Shows images of the target map and grid on a figure with two 
        subaxes. Colors the images according to hit/miss (for target map) and
        damage (for the grid). Returns the figure containing the images.
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
            rows,cols = np.where(vert_grid == TargetValue.MISS)
            white_pegs = [plt.Circle((x,y), radius=peg_radius, 
                                     facecolor = miss_color, 
                                     edgecolor = peg_outline_color) 
                        for (x,y) in zip(cols+1,rows+1)]
            for peg in white_pegs:
                axs[0].add_patch(peg) 
            rows,cols = np.where(vert_grid >= TargetValue.HIT)
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
            for shipId in ship_boxes:
                box = ship_boxes[shipId]
                axs[-1].add_patch( 
                    mpl.patches.Rectangle((box[0], box[1]), box[2], box[3], \
                                          edgecolor = ship_outline_color,
                                          fill = False,                                       
                                          linewidth = 2))
                axs[-1].text(box[0]+0.12, box[1]+0.65, 
                             SHIP_DATA[shipId]["name"][0])

        fig.set_facecolor(bg_color)
        return fig
    
# -----------------------------------------------------------------------------
class Ship:
    """A ship that can be placed on a Battleship board, with
    slots for tracking damage.
    Initialize using one of the following:
        Ship(ShipId.CARRIER)    (ShipId.CARRIER is the same as ShipId(5))
        Ship(5) 
        Ship("Carrier")         (case insensitive)
        """
    
    def __init__(self, ship_type):
        """Returns an instance of class Ship described by the input 
        ship_type, which may be an integer 1-5, an instance of the
        Enum ShipId, or one of the following (case insensitive) strings:
            Patrol, Destroyer, Submarine, Battleship, Carrier
        """
        if isinstance(ship_type, str):
            id_match = [k for k in SHIP_DATA if 
                        SHIP_DATA[k]["name"].lower() == ship_type.lower()]
            if id_match:
                self._ship_id = id_match[0]
            else:
                ValueError("For string input, ship_type must be one of: " 
                           + ", ".join([SHIP_DATA[k]["name"] for k in SHIP_DATA])) 
        else:
            self._ship_id = ship_type
            
        self.name = SHIP_DATA[self._ship_id]["name"]
        self.length = SHIP_DATA[self._ship_id]["length"]
        self.damage = np.zeros(self.length, dtype=np.int8)
        
    def __len__(self):
        """Returns an integer corresponding to the number of spaces
        the ship occupies, which is also the number of damage slots
        for the ship.
        """
        return self.length
    
    def __str__(self):
        """Returns a string listing the Ship instance name, length, damage,
        and whether the ship is sunk.
        """
        s = (f"{self.name} (id = {self._ship_id}) - {self.length} slots, " 
             f"damage: {str(self.damage)}")
        if self.sunk():
            s += " (SUNK)"
        return(s)
    
    @property
    def ship_id(self):
        return self._ship_id
    
    def hit(self, slot):
        """Adds damage to the input slot position. Returns the damage value
        at the hit position. Raises an exception if the input slot is invalid 
        (i.e., if it is non-integer, less than 0, or greater than the ship
         length minus 1).
        """
        if slot < 0 or slot >= self.length:
            raise ValueError("Hit at slot #" + str(slot) + " is not valid; "
                             "must be 0 to " + str(self.length-1))
        if self.damage[slot] > 0:
            Warning("Slot " + str(slot) + " is already damaged.")
        self.damage[slot] += 1
        return(self.damage[slot])
    
    def afloat(self):
        """Returns True if the ship has not taken damage at each slot."""
        return(any(dmg < 1 for dmg in self.damage))

    def sunk(self):
        """Returns True if the ship has taken damage at each slot."""
        return(all(dmg > 0 for dmg in self.damage))
