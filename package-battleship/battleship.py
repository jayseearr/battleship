#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 22:57:54 2022

@author: jason
"""

#%%
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from enum import Enum, IntEnum
from collections import Counter

#%% Constants

### Hard-Coded ###

MAX_ITER = 1000
SUNK_OFFSET = 0.1

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
    
#%% Data

SHIP_DATA = {ShipType.PATROL: {"name": "Patrol".title(), "length": 2},
             ShipType.DESTROYER: {"name": "Destroyer".title(), "length": 3},
             ShipType.SUBMARINE: {"name": "Submarine".title(), "length": 3},
             ShipType.BATTLESHIP: {"name": "Battleship".title(), "length": 4},
             ShipType.CARRIER: {"name": "Carrier".title(), "length": 5}
             }

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

def rotate_coords(coords, degrees, board_size):
    """
    Rotate the input coordinates by an angle about the center of the board.

    Parameters
    ----------
    coords : List of tuples
        A list of 2-element coordinate tuples to rotate.
    degrees : Float
        The angle to rotate the coordinates. Positive cooresponds to clockwise.
    board_size : Int
        The number of spaces along one edge of the board on which coords are
        placed. 

    Returns
    -------
    coords : List of tuples
        Same size as the input coord. The ith element is the rotated ith 
        element of the input list.

    """
    # for 90 degrees, (0,0) --> (0,9); (0,9) --> (9,9); (9,9) --> (9,0); (9,0) --> (0,0)
    # for 180 degrees, (0,0) --> (9,9); (0,9) --> (9,0); (9,9) --> (0,0); (9,0) --> (0,9)    
    # for -90 degrees, (0,0) --> (9,0); (0,9) --> (0,0); (9,9) --> (0,9); (9,0) --> (9,9)    
    angle = np.pi * degrees / 180
    sine = np.sin(angle)
    cosine = np.cos(angle)
    center = (board_size - 1) / 2
    coords = [np.array(pt) - center * np.ones(2) for pt in coords]
    T = np.array(([cosine, sine], [-sine, cosine]))
    coords = [tuple(T @ pt + center * np.ones(2)) for pt in coords]
    return coords
    
def mirror_coords(coords, axis, board_size):
    """
    Flips the input board coordinates along the input axis. Axis=0 flips 
    vertically (up/down), and axis=1 flips horizontally (left/right).

    Parameters
    ----------
    coords : List of tuples
        A list of 2-element coordinate tuples to mirror.
    axis : Int (0 or 1)
        The axis along which to mirror the coordinates. This should be 0 for
        vertical or up/down mirroring (i.e., flipping across a horizontal line)
        and 1 for horizontal or left/right mirroring (i.e., flipping across
        a vertical line).
    board_size : Int
        The number of spaces along one edge of the board on which coords are
        placed. 

    Returns
    -------
    list
        List of tuples
            Same size as the input coord. The ith element is the mirrored ith 
            element of the input list.

    """
    center = (board_size - 1) / 2
    coords = np.array(coords) - np.tile(center, (len(coords),2))
    coords[:,axis] = -coords[:,axis]
    coords = coords + np.tile(center, (len(coords),2))
    return [tuple(pt) for pt in coords]

def label_array(x):
    """
    Labels connected elements of input array of dimension 1. Any adjacent 
    elements that are >= 0 are considered to be part of the same region and
    will have the same label.
    
    Parameters
    ----------
    x : numpy.ndarray
        A one-dimensional numpy array.

    Returns
    -------
    lbl : numpy.ndarray
        An array of the same size as the input x, with 0 for regions in x that
        are 0, and an integer label for elements taht are in a contiguous
        region.

    """
    x = (x == True)
    lbl = np.empty(len(x), dtype=int)
    current_label = 1
    for (i,el) in enumerate(x):
        if el:
            lbl[i] = current_label
        else:
            lbl[i] = 0
            if i > 0 and lbl[i-1] != 0:
                current_label += 1
    return lbl

#%% ###################
###                 ###
### GENERAL CLASSES ###
###                 ###
### ###################

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
    __TOLERANCE__ = 1e-6
    
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
        else:
            raise TypeError("Coord argument types are: tuple, list, str, int.")
        if (np.abs(r - np.round(r)) > self.__TOLERANCE__ or 
            np.abs(c - np.round(c)) > self.__TOLERANCE__):
                raise ValueError("Row/column values must be integers.")
                
        self._rowcol = int(np.round(r)), int(np.round(c))
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
    
    def __setitem__(self, key, value):
        """Set the row or column value of the Coord to the input value."""
        if key < 0 or key > 1:
            raise ValueError("Index must be 0 or 1.")
        if value < 0 or not isinstance(value, (np.integer, int)):
            raise ValueError("Coord row/column cannot be set to a non-integer type.")
        rowcol = list(self._rowcol)
        rowcol[key] = value
        self._rowcol = rowcol
    
    def __add__(self, other):
        """Adds another Coord or a 2-element tuple to the Coord object."""
        return Coord((self[0] + other[0], self[1] + other[1]))
    
    def __sub__(self, other):
        """Adds another Coord or a 2-element tuple to the Coord object."""
        return Coord((self[0] - other[0], self[1] - other[1]))
    
    def __mul__(self, other):
        """Multiplies another Coord or a 2-element tuple to the Coord object
        in an element-by-element manner (not matrix multiplication)."""
        return Coord((self[0] * other[0], self[1] * other[1]))
    
    def __repr__(self):
        return f'Coord({self.rowcol})'
    
    ### Comparison methods ###
    
    def next_to(self, other):
        """
        Determines if the Coord is directly adjacent to another Coord.
        See also 'diagonal_to'.
        
        Parameters
        ----------
        other : Coord
            An instance of Coord, or a 2-element list, tuple, or numpy.ndarray.

        Returns
        -------
        Bool
            True if the input touches this Coord, False otherwise.

        """
        return (self[0] - other[0])**2 + (self[1] - other[1])**2 <= 1
    
    def diagonal_to(self, other):
        """
        Determines if the Coord is diagonally adjacent to another Coord.
        See also 'next_to'.
        
        Parameters
        ----------
        other : Coord
            An instance of Coord, or a 2-element list, tuple, or numpy.ndarray.

        Returns
        -------
        Bool
            True if the input is diagonally adjacent to this Coord, and
            False otherwise.

        """
        return (np.abs(self[0] - other[0]) == 1 and 
                np.abs(self[1] - other[1]) == 1)
    
#%% Game Class

class Game:
    
    def __init__(self, player1, player2, 
                 game_id=None, 
                 verbose=True,
                 show=False):
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
        self.show = False
        
    def __repr__(self):
        return (f"Game({self._player1!r},\n {self._player2!r},\n "
                f"game_id={self.game_id!r}, verbose={self.verbose!r})")
    
    def __str__(self):
        if self.winner:
            s = f"Completed Game - {self.turn_count} turns.\n"
            if self.game_id:
                s += f"Game ID: {self.game_id}\n"
            s += f"Winner: {self.winner!r}\n"
            s += f"Loser: {self.loser!r}\n"
        else:
            s = "Unplayed Game\n"
            if self.game_id:
                s += f"Game ID: {self.game_id}\n"
            s += f"Player 1: {self.player1!r}\n"
            s += f"Player 2: {self.player2!r}\n"
        return s
        
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
    
    # Other methods 
    
    def setup(self):
        """Set up a game by placing both fleets."""
        if ((self.player1.board and self.player1.board.fleet) or 
            (self.player2.board and self.player2.board.fleet)):
            raise Exception("Boards need to be reset; call reset() method.")
        self.player1.opponent = self.player2
        #self.player2.opponent = self.player1
        self.player1.prepare_fleet()
        self.player2.prepare_fleet()
        self.ready = (self.player1.board.ready_to_play() and 
                      self.player2.board.ready_to_play())
    
    def reset(self):
        self.player1.reset()
        self.player2.reset()
        self.ready = False
        self.winner = None
        self.loser = None
        self.turn_count = 0
     
    # Gameplay methods 
    
    def play_one_turn(self, first_player, second_player):
        """
        

        Parameters
        ----------
        first_player : Player subclass
            DESCRIPTION.
        second_player : Player subclass
            DESCRIPTION.

        Returns
        -------
        Bool
            True if the game should continue, False if the game should end
            (i.e., False if one player no longer has any ships afloat).

        """
        first_player.take_turn()
        if self.verbose:
            self.report_turn_outcome(first_player)
        second_player.take_turn()
        if self.verbose:
            self.report_turn_outcome(second_player)
        self.turn_count += 1
        if self.show:
            first_player.board.show()
            second_player.board.show()
        return (first_player.isalive() and 
                second_player.isalive())
        
    def play(self, first_move=1, max_turns=None):
        """Play one game of Battleship. Each player takes a turn until one
        of them has no ships remaining.
        Returns a tuple containing (winner, loser) Player instances.
        """
        
        if max_turns == None:
            max_turns = 1e6
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
            still_playing = self.play_one_turn(first_player, second_player)
            game_on = still_playing and self.turn_count < max_turns
            
        # See who won
        if first_player.isalive() and not second_player.isalive():
            self.winner, self.loser = first_player, second_player
        elif second_player.isalive() and not first_player.isalive():
            self.winner, self.loser = second_player, first_player
        elif self.turn_count >= max_turns:
            pass
        else:
            raise Exception("Winner could not be determined.")
        
        if self.verbose:
            self.print_outcome()
            
        return (self.winner, self.loser)
         
    def report_turn_outcome(self, player):
        """Displays text reporting the target and outcome on the most recent
        turn for the input player.
        """
        outcome = player.last_outcome
        target = player.last_target
        name = player.name
        sink = ""
        if outcome["hit"]:
            hit_or_miss = "Hit!"
        else:
            hit_or_miss = "Miss."
        if outcome["sunk_ship_id"]:
            sink = SHIP_DATA[outcome["sunk_ship_id"]]["name"] + " sunk!"
        print(f"Turn {len(player.shot_history)}: Player {name} fired a shot at "
              f"{Coord(target)}...{hit_or_miss} {sink}")
        
    def print_outcome(self):
        """Prints the results of the game for the input winner/loser players 
        (which include their respective boards)."""

        if self.winner:
            print("")
            print("GAME OVER!")
            print(f"Player {self.winner.name} wins.")
            print(f"  Player {self.winner.name} took "
                  f"{len(self.winner.shot_history)} shots, and sank " 
                  f"{sum(1 - self.loser.board.afloat_ships())} ships.")
            print(f"  Player {self.loser.name} took " 
                  f"{len(self.loser.shot_history)} shots, and sank " 
                  f"{sum(1 - self.winner.board.afloat_ships())} ships.")
            print(f"Game length: {self.turn_count} turns.")
            print(f"(Game ID: {self.game_id})")
            print("")
        else:
            print("")
            print("GAME TERMINATED - NO WINNER.")
            print(f"  Player {self.player1.name} (player 1) took " 
                  f"{len(self.player1.shot_history)} shots, and sank "
                  f"{sum(1 - self.player2.board.afloat_ships())} ships.")
            print(f"  Player {self.player2.name} (player 2) took " 
                  f"{len(self.player2.shot_history)} shots, and sank "
                  f"{sum(1 - self.player1.board.afloat_ships())} ships.")
            print(f"Game length: {self.turn_count} turns.")
            print(f"(Game ID: {self.game_id})")
            print("")
            
    # Factory method
    @classmethod
    def random(cls, seed=None, game_id=None, verbose=True, show=False):
        from player import AIPlayer
        from offense import HunterOffense
        from defense import RandomDefense
        np.random.seed(seed)
        off_weights = ['isolated', 'flat', 'center', 'maxprob']
        def_weights = ['flat', 'clustered', 'isolated']
        p1 = AIPlayer(HunterOffense('random', 
                                    weight=random_choice(off_weights)),
                      RandomDefense(random_choice(def_weights)), 
                      name = "Gwen")
        p2 = AIPlayer(HunterOffense('random', 
                                    weight=random_choice(off_weights)),
                      RandomDefense(random_choice(def_weights)), 
                      name = "Sunny")
        return Game(p1, p2, game_id, verbose, show)
        
#%% Board Class
class Board:
    """An instance of a Battleship game board. Each board has a fleet of 5 
    ships, a 10 x 10 ocean grid upon which ships are arranged, and a 10 x 10 
    target grid where hit/miss indicators are placed to keep track of 
    opponent ships.
    """

    def __init__(self, size=10):
        self.size = size
        self.fleet = {}
        self.ocean_grid = np.zeros((size, size), dtype=np.int8)      
        self.target_grid = np.ones((size, size)) * TargetValue.UNKNOWN
        
    def __str__(self):
        """Returns a string that uses text to represent the target map
        and grid (map on top, grid on the bottom). On the map, an 'O' represents
        a miss, an 'X' represents a hit, and '-' means no shot has been fired
        at that space. 
        On the grid, a '-' means no ship, an integer 1-5 means a ship slot with
        no damage, and a letter means a slot with damage (the letter matches
        the ship name, so P means a damaged patrol boat slot).
        """
        # s = "\n  "
        # s += (" ".join([str(x) for x in np.arange(self.size) + 1]) 
        #       + "\n")
        # for (r,row) in enumerate(self.target_grid):
        #     s += (Coord(r,0).lbl[0]
        #           + " "
        #           + " ".join(["-" if x == TargetValue.UNKNOWN else 
        #                       'O' if x == TargetValue.MISS else
        #                       'X' if x == TargetValue.HIT else 
        #                       str(x) for x in row]) 
        #           + "\n")
        # s += ("\n  " 
        #       + " ".join([str(x) for x in np.arange(self.size) + 1]) 
        #       + "\n")
        # for (r,row) in enumerate(self.ocean_grid):
        #     s += (Coord(r,0).lbl[0]
        #           + " "
        #           + " ".join(["-" if x == 0 else 
        #                       str(x) for x in row]) 
        #           + "\n")
        # return s
        return self.color_str()
    
    def __repr__(self):
        return f"Board({self.size})"
    
    def reset(self):
        """Removes all ships and pegs from the board."""
        board = Board(self.size)
        self.fleet = board.fleet
        self.target_grid = board.target_grid
        self.ocean_grid = board.ocean_grid
        
    # Factory method
    
    @classmethod
    def copy(cls, other_board):
        """Creates a copy of the input board. The copy can be manipulated
        independently of the original board (i.e., all attributes are
        copies, not the same object)."""
        
        board = cls(other_board.size)
        placements = dict([(k,other_board.placement_for_ship(k)) 
                      for k in other_board.fleet])
        board.place_fleet(placements)
        board.target_grid = other_board.target_grid.copy()
        return board
    
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
                heading.upper() in ["UP", "DOWN", "LEFT", "RIGHT"] or
                heading.upper() in ["VERTICAL", "HORIZONTAL"]):
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
            return ds
        else:
            ds = np.vstack(([0,0], 
                            np.cumsum(np.tile(ds, (length-1,1)), axis=0)))
            return (ds[:,0], ds[:,1])
       
    def coords_for_placement(self, placement, length):
        """
        

        Parameters
        ----------
        placement : tuple 
            First element of tuple is a coordinate (a tuple with two elements
            corresponding to a row and a column) and the second is a chr 
            denoting a heading ("N", "S", "E", or "W").
        length : int
            The length of a ship placed at the input coordinate facing the
            input heading,
            The output will have a number of elements equal to length.

        Returns
        -------
        List of coordinates that span the input starting coordinate/heading
        and the length.

        """
        coord, heading = placement
        ds = self.relative_coords_for_heading(heading, length)
        return list(zip(coord[0] + ds[0], coord[1] + ds[1]))
        
    def valid_coord(self, coords):
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
        rows, cols = zip(*coords)
        rows = np.array(rows)
        cols = np.array(cols)
        return np.array((rows >= 0) * (rows < self.size) * (cols > 0) *
                         (cols < self.size))
    
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
    
    def colinear_target_coords(self, coords, untargeted):
        """Returns the two coordinates at the end of the line made
        by the input coordinates.
        """
        # find the first unknown or edge
        ds = (np.diff(np.array(coords), axis=0))
        ia,ib = sorted([coords[0], coords[-1]])
        choices = []
        
        if np.any(ds[:,0] != 0):
            while (ia[0] >= 0 and ia[0] < self.size and 
                   self.target_grid[ia] != TargetValue.UNKNOWN):
                ia = (ia[0] - 1, ia[1])
            while (ib[0] >= 0 and ib[0] < self.size and 
                   self.target_grid[ib] != TargetValue.UNKNOWN):
                ib = (ib[0] + 1, ib[1])
            if (ia[0] >= 0 and 
                    (self.target_grid[ia[0]+1,ia[1]] != TargetValue.MISS or
                     untargeted == False)):
                choices += [ia]
            if (ib[0] < self.size and 
                    (self.target_grid[ib[0]-1,ib[1]] != TargetValue.MISS or
                     untargeted == False)):
                choices += [ib]
                
        elif np.any(ds[:,1] != 0):
            while (ia[1] >= 0 and ia[1] < self.size and 
                   self.target_grid[ia] != TargetValue.UNKNOWN):
                ia = (ia[0], ia[1] - 1)
            while (ib[1] >= 0 and ib[1] < self.size and 
                   self.target_grid[ib] != TargetValue.UNKNOWN):
                ib = (ib[0], ib[1] + 1)
            if (ia[1] >= 0 and 
                    (self.target_grid[ia[0],ia[1]+1] != TargetValue.MISS or
                     untargeted == False)):
                choices += [ia]
            if (ib[1] < self.size and 
                    (self.target_grid[ib[0],ib[1]-1] != TargetValue.MISS or
                     untargeted == False)):
                choices += [ib]
        
        return [rc for rc in choices if rc not in coords]
          
    def possible_ships_grid(self, simplify=False, by_type=False):
        """
        Returns the number of possible ship placements at each coordinate on 
        the target grid. P[r,c] is equal to the number of uniqe positions and 
        headings that would cause a ship to have a space located at [r,c], 
        added together for each ship type.

        Parameters
        ----------
        simplify : Bool, optional
            If True, the returned grid contains the number of possible ships
            that can be at location (r,c), rather than the number of possible
            placements for each ship. The default is False.

        Returns
        -------
        psg : numpy.ndarray
            A two-dimensional array with shape equal to the board's target_grid
            shape. The value at row r and column c is equal to the number of
            ways any ship could be placed so that it has a space at (r,c).
            The maximum value is 34 (2x5 + 2x4 + 2x3 + 2x3 + 2x2).
            
            If the simplify parameter is True, the value at (r,c) is the number
            of possible ships that could be at (r,c). In this case, the value
            is between 0 and 5.

        """
        untargeted = self.all_coords(untargeted=True)
        # possible ship grid
        psg = np.zeros((len(SHIP_DATA), self.size, self.size))
        for n in range(1,len(SHIP_DATA)+1):
            ship = Ship(n)
            placements = self.all_valid_placements('target', ship.ship_id,
                                                   headings='NW')
            placement_coords = []
            for p in placements:
                ds = self.relative_coords_for_heading(p[1], ship.length)
                placement_coords += [tuple(x) for x in 
                                     list(np.tile(np.array(p[0]), 
                                                  (ship.length,1)) 
                                          + np.array(ds).transpose())]
            coord_freq = Counter(placement_coords)
            for coord in untargeted:
                psg[n-1, coord[0], coord[1]] = coord_freq[coord]
        if simplify:
            psg = np.sum(psg > 0, axis=0)
        elif not by_type:
            psg = np.sum(psg, axis=0)
        return psg
    
    def possible_targets_at_coord(self, coord, out='dict'):
        """
        An dictionary with the number of possible placements for each ship type
        at the input coordinate.

        Parameters
        ----------
        coord : tuple
            A coordinate on the board (two elements, row and column).
        out : str, optional
            String describing what type of output to return. 
            If 'dict', a dictionary with ship type as key will be returned.
            If 'array', an array ordered by ship type. The default is 'dict'.

        Returns
        -------
        nplacements_by_ship : dict or array (see parameter 'out')
            The number of possible placements at the input coord
            for each ship type.

        """
        nplacements_by_ship = {}
        target_grid = self.target_grid
        value_at_coord = target_grid[coord]
        
        def valid_target_placement(placement, length, ship_id=None):
            print(placement)
            coords = self.coords_for_placement(placement, length)
            print(coords)
            print(self.valid_coord(coords))
            if np.any(~self.valid_coord(coords)):
                return False
            target_grid_values = np.array(
                [self.target_grid[c] for c in coords]
                )
            return np.all(target_grid_values != TargetValue.MISS)
        
        def nhits_on_ship(ship):
            return np.sum(np.round(self.target_grid) == ship.length) 
        
        for ship_id in sorted(SHIP_DATA):
            ship = Ship(ship_id)
            # If ship is sunk and overlaps coord, there is only 1 possible
            #   target at that spot. If it does not overlap coord, there are
            #   0 targets (of this ship type).
            if (np.any(self.target_grid == ship.ship_id + SUNK_OFFSET) or
                    nhits_on_ship(ship) == ship.length):
                # ship is sunk.
                nplacements_by_ship[ship_id] = int(coord in 
                                                self.coords_for_ship(ship))
                
            # If coord has a ship_id, nplacements should be 0 for all other 
            #   ship_ids
            elif (value_at_coord > TargetValue.HIT 
                      and value_at_coord != ship_id):
                nplacements_by_ship[ship_id] = 0
                
            else:
                dn = self.relative_coords_for_heading("N", length=ship.length)
                dw = self.relative_coords_for_heading("W", length=ship.length)
                placements = [((coord[0] - dr, coord[1]), "N") for dr in dn[0]]
                placements += [((coord[0], coord[1] - dr), "W") for dr in dw[1]]
                placements = [p for p in placements if 
                              valid_target_placement(p, ship.length)]
                # Eliminate placements that do not contain ALL instances of
                #   ship_id on the target grid.
                rows, cols = np.where(np.round(self.target_grid) == 
                                      ship.ship_id)
                hits = list(zip(rows, cols))
                if hits:
                    ok_placements = []
                    for p in placements:
                        ship_coords = self.coords_for_placement(p, ship.length)
                        if all([h in ship_coords for h in hits]):
                            ok_placements += [p]
                    placements = ok_placements
                    
                # ----> Should be able to modify nplacement values if certain
                # ships can only be in certain locations ([4, -2, 4] means that
                # the -2 should be a 4, although this should be take care of
                # by some other logic in the 'update' methods.)
                
                nplacements_by_ship[ship_id] = len(placements)
            
        return nplacements_by_ship
            
        # nplacements_by_ship = {}
        # for ship in list(self.fleet.values()):
        #     offset = np.arange(ship.length)
        #     count = 0
        #     for shift in offset:
        #         placement = ((coord[0] + shift, coord[1]), "S")
        #         coords = self.coords_for_placement(placement, ship.length)
        #         rows,cols = zip(*coords)
        #         if all(r>0 and r<self.size and c>0 and c<self.size 
        #                for (r,c) in coords):
        #             values = self.target_grid[rows,cols]
        #             count += np.all((values == TargetValue.HIT) 
        #                             + (values == TargetValue.UNKNOWN))
        #         placement = ((coord[0], coord[1] + shift), "E")
        #         coords = self.coords_for_placement(placement, ship.length)
        #         if all(r>0 and r<self.size and c>0 and c<self.size 
        #                for (r,c) in coords):
        #             rows,cols = zip(*coords)
        #             values = self.target_grid[rows,cols]
        #             count += np.all((values == TargetValue.HIT) 
        #                             + (values == TargetValue.UNKNOWN))
        #     nplacements_by_ship[ship.ship_id] = count
        # return nplacements_by_ship
    
    # Ship access functions
    def afloat_ships(self):
        """Returns a list of the Ship instances that are still afloat."""
        return np.array([ship.afloat() for ship in list(self.fleet.values())])
            
    def coords_for_ship(self, ship):
        """Returns the row/col coordinates occuppied by the input Ship instance."""
        if isinstance(ship, Ship):
            ship = int(ship.ship_id)
        if not isinstance(ship, int):
            raise TypeError("ship must be an instance of Ship or an int.")
        r,c = np.where(self.ocean_grid == ship)
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
        coords = self.coords_for_ship(ship)
        return ship.damage[coords.index(coord)]
    
    def placement_for_ship(self, ship):
        """Returns a tuple with the coordinate and heading of the input ship
        or ship_id integer."""
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
        return {"coord": coord, "heading": heading}
            
    # Ship placement functions        
    def all_valid_ship_placements(self, ship_id, distance=0, diagonal=True,
                                  edges=True, alignment=Align.ANY,
                                  headings='all'):
        """Returns a list of all possible placements (consisting of row, 
        column, and heading) on the board. The distance parameter is the 
        minimum separation between ships.
        If the diagonal input is True, the four coordinates 
        diagonally touching existing ships are considered valid (and invalid
        if diagonal is False).
        The returned list of placements is in the form of a list of tuples
        ((row, column), heading).
        """
        headings_dict = {"N": (1,0), "S": (-1,0), "E": (0,-1), "W": (0,1)}
        ship = Ship(ship_id)
        if headings.lower() == 'all':
            allowed_headings = ["N","S","E","W"]
        else:
            allowed_headings = [h.upper() for h in headings]
        if alignment == Align.VERTICAL:
            allowed_headings = [h for h in allowed_headings if h in ["N", "S"]]
        elif alignment == Align.HORIZONTAL:
            allowed_headings = [h for h in allowed_headings if h in ["E", "W"]]
        else:
            pass
            
        grid = self.ocean_grid
        
        # Determine whether edge coordinates can be occuppied
        rcmin = 1 - edges
        rcmax = self.size - (1 - edges) - 1
        unoccuppied = self.all_coords(unoccuppied=True)
        placements = []
        for unocc in unoccuppied:
            for h in allowed_headings:
                ds = np.vstack(([0,0], np.cumsum(np.tile(
                    headings_dict[h], (ship.length-1,1)), axis=0)))
                rows,cols = unocc[0] + ds[:,0], unocc[1] + ds[:,1]
                ok = True
                if (np.any(rows < rcmin) or np.any(rows > rcmax) or 
                    np.any(cols < rcmin) or np.any(cols > rcmax) or
                    np.any(_grid[rows,cols])):
                        ok = False
                if distance > 0 and ok:
                    dmin = self.size
                    for (r,c) in zip(rows, cols):
                        other_coords = []
                        for other_ship in list(self.fleet.values()):
                            other_coords += self.coords_for_ship(other_ship)
                        if ok and len(other_coords):
                            other_coords = np.array(other_coords)
                            delta = (other_coords 
                                     - np.tile((r,c), (other_coords.shape[0],1)))
                            d = np.sqrt(np.sum(delta**2, axis=1))
                            dmin = np.min((np.min(d), dmin))
                            if diagonal:
                                dmin = np.ceil(dmin)
                            else:
                                dmin = np.floor(dmin)
                            ok = ok * (dmin > distance)
                if ok:
                    placements += [(unocc, h)]
        return placements
        
    def all_valid_target_placements(self, ship_id):
        """Returns a list of valid target placements based on the current 
        state of the target grid. A placements is valid for a given ship if
        that ship is fully on the board, does not overlap another known ship,
        does not contain any known misses, and contains ALL known instances
        of hits to that type of ship."""
        ship = ship_id if isinstance(ship_id, Ship) else ship(ship_id)
        coords = self.all_coords()
        valid_rows,valid_cols = np.where(np.round(self.target_grid) 
                                         == ship.ship_id)
        must_include_coords = list(zip(valid_rows,valid_cols))
        rows,cols = np.where(np.round(self.target_grid) == TargetValue.MISS)
        must_exclude_coords = list(zip(rows,cols))
        if must_include_coords:
            coords = [(r,c) for (r,c) in coords if 
                      (r in valid_rows) or (c in valid_cols)]
            coords = [(r,c) for (r,c) in coords if 
                      (np.max(np.abs(r - valid_rows)) +  
                       np.max(np.abs(c - valid_cols)) < ship.length)]
        coords = [c for c in coords if self.valid_coord(c)]
        
    
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
        for (r,c) in zip(rows,cols):
            if (r,c) in occuppied:
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
    def update_target_grid_with_outcome(self, outcome):
        """Updates the targets grid with the results of the 'outcome' dict,
        which contains the target coordinate, hit indicator, and identity
        of a sunk ship (if applicable).
        """
        self.update_target_grid(outcome["coord"], outcome["hit"],
                                outcome["sunk_ship_id"])
        
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
                self.target_grid[target_coord] = ship_id + SUNK_OFFSET
                self.update_grid_for_sink(target_coord, ship_id)
            else:
                self.target_grid[target_coord] = TargetValue.HIT   
        else:
            self.target_grid[target_coord] = TargetValue.MISS
        if prev_val != TargetValue.UNKNOWN:
            print(f"Repeat target: {Coord(target_coord).lbl}")
           
    def update_grid_for_sink(self, coord, sunk_ship_id):
        """If there are hits on the target grid that can be unambiguously 
        assigned to the ship sunk at the input coord, convert those hits 
        to ship_id for book-keeping purposes. This helps with determining 
        which target ships may be where on the grid."""
        print("Updating for sink: ", coord,sunk_ship_id)
        ship_length = SHIP_DATA[ShipType(sunk_ship_id)]["length"]
        
        # Can any existing hits be attributed to the ship that was just sunk?
        target_cols = self.target_grid[coord[0],:]
        target_rows = self.target_grid[:,coord[1]]
        hit_cols = label_array((target_cols == TargetValue.HIT) +  
                               (target_cols == sunk_ship_id))
        hit_rows = label_array((target_rows == TargetValue.HIT) +  
                               (target_rows == sunk_ship_id))
        contiguous_cols = np.where(hit_cols == hit_cols[coord[1]])[0]
        contiguous_rows = np.where(hit_rows == hit_rows[coord[0]])[0]
        
        if not (len(contiguous_cols) >= ship_length or 
                len(contiguous_rows) >= ship_length):
            raise Exception("Failed to find ship location in target grid.")
        if len(contiguous_cols) < ship_length:
            # ship must be oriented vertically
            rows = []
            if contiguous_cols[0] == coord[0]:
                rows = contiguous_rows[:ship_length]
                self.target_grid[rows, coord[1]] = sunk_ship_id
                self.target_grid[coord] = sunk_ship_id + SUNK_OFFSET
            elif contiguous_rows[-1] == coord[0]:
                rows = contiguous_rows[-ship_length:]
                self.target_grid[rows, coord[1]] = sunk_ship_id
                self.target_grid[coord] = sunk_ship_id + SUNK_OFFSET
            print(f"Updating target grid with sink #{sunk_ship_id} for "
                  f"rows = {rows}, col = {coord[1]}.")
            # otherwise, could not determine location uniquely.
        elif len(contiguous_rows) < ship_length:
            # ship must be oriented horizontally
            if contiguous_cols[0] == coord[1]:
                cols = contiguous_cols[:ship_length]
                self.target_grid[coord[0], cols] = sunk_ship_id
                self.target_grid[coord] = sunk_ship_id + SUNK_OFFSET
            elif contiguous_cols[-1] == coord[1]:
                cols = contiguous_cols[-ship_length:]
                self.target_grid[coord[0], cols] = sunk_ship_id
                self.target_grid[coord] = sunk_ship_id + SUNK_OFFSET
            print(f"Updating target grid with sink #{sunk_ship_id} for "
                  f"rows = {coord[0]}, col = {cols}.")
            # otherwise, could not determine location uniquely.
        else:
            # cannot determine orientation.
            pass        
                                    
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
            sunk, damage = self.damage_ship_at_coord(ship, coord)
            msg = []
            if damage > 1:
                msg += [f"Repeat target: {coord}."]
            if sunk:
                msg += [f"{ship.name} sunk."]
                outcome["sunk"] = True
                outcome["sunk_ship_id"] = ship.ship_id
        return outcome
    
    def damage_ship_at_coord(self, ship, coord):
        """Adds a point of damage to the ship at the input coord. 
        Raises an exception if the coordinate does not lie on the ship in 
        question.
        Returns a tuple with a bool indicating whether the ship has been
        sunk, and the number of times coord has been hit.
        """
        if ship not in self.fleet_list:
            raise ValueError("Input ship instance must point to one of "
                             "the object in this Board's fleet.")
        ship_coords = self.coords_for_ship(ship)
        dmg_slot = [i for i in range(len(ship_coords)) 
                    if ship_coords[i] == coord]
        if len(dmg_slot) != 1:
            raise Exception(f"Could not determine unique damage location for "
                            f"target {coord}")
        damage = ship.hit(dmg_slot[0])
        return ship.sunk(), damage
        
    @property
    def fleet_list(self):
        """Returns the ship objects in the fleet attribute as a list."""
        return list(self.fleet.values())
        
    # Test function
    def test(self, targets=True):
        self.place_ship(1, Coord("J9"), "R") 
        self.place_ship(2, Coord("E8"), "N") 
        self.place_ship(4, Coord("G2"), "W") 
        self.place_ship(3, Coord("B6"), "W") 
        self.place_ship(5, Coord("A5"), "N")
        
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
            rows,cols = zip(*self.coords_for_ship(ship))
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
            rows, cols = zip(*self.coords_for_ship(ship))
            rects[ship.ship_id] = np.array((np.min(cols)+0.5, 
                                            np.min(rows)+0.5,
                                            np.max(cols) - np.min(cols) + 1, 
                                            np.max(rows) - np.min(rows) + 1))
        return rects
        
    def color_str(self):
        """Returns a string in color that represents the state of the board.
        to a human player.
        """
            
        def format_target_id(x):
            s = f"{str(round(x))}"
            if x - round(x) == SUNK_OFFSET:
                s += "*"
            return s
        
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
                              (red_peg_sunk + format_target_id(x)) 
                              for x in row]) 
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
    
#%% Ship Class
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
    
    def __repr__(self):
        return f"Ship({self.ship_id})"        
    
    # @classproperty
    # def data(cls, ship_id=None):
    #     """
    #     Returns a dictionary with keys for each of the possible ship types
    #     (see ShipType) and values equal to a dictionary of properties for 
    #     the type of ship corresponding to the key.
    #     The properties dictionary has keys 'name' and 'length'.

    #     An ShipType value (or integer 1-5) can be provided as an input, in 
    #     which case only the properties dictionary for that type of ship
    #     will be returned.
        
    #     Parameters
    #     ----------
    #     ship_id : TYPE, optional
    #         A ShipType value or integer between 1 and 5. If provided, the
    #         properties dictionary for only the ship type corresponding to the
    #         input value will be returned (1 = patrol, 2 = destroyer, 
    #         3 = submarine, 4 = battleship, 5 = carrier).
    #         The default is None.

    #     Returns
    #     -------
    #     dict
    #         A dictionary with dictionary values (for no input parameters) or
    #         a dictionary with 'name' and 'length' keys (for int input 
    #         parameter).

    #     """
    #     if ship_id == None:
    #         return SHIP_DATA
    #     else:
    #         return SHIP_DATA[ship_id]
        
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