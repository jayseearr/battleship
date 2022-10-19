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

# Ship.data = {ShipType.PATROL: {"name": "Patrol".title(), "length": 2},
#              ShipType.DESTROYER: {"name": "Destroyer".title(), "length": 3},
#              ShipType.SUBMARINE: {"name": "Submarine".title(), "length": 3},
#              ShipType.BATTLESHIP: {"name": "Battleship".title(), "length": 4},
#              ShipType.CARRIER: {"name": "Carrier".title(), "length": 5}
#              }

#%% Utility Functions

def random_choice(x, p=None):
    """
    Returns a single element from the input array (x) randomly sampled with
    probability distribution p.

    Parameters
    ----------
    x : list or numpy array
        The items from which to sample.
    p : numpy array, optional
        An array with the same length as x. The probability of picking x[i]
        is given by p[i]. p should be normalized so that sum(p) == 1.
        The default is None, which samples from a uniform probability 
        distribution.

    Returns
    -------
        A single element from x. If x has length 1, the sole element is 
        returned. If x is empty, an empty numpy array is returned.

    """
    return x[np.random.choice(range(len(x)), p=p)]

def is_on_board(x, size):
    """
    Returns True if the input coordinate is on the board with specified size,
    and False otherwise. This is true when the row and column contained in 
    x are between 0 and size - 1.

    Parameters
    ----------
    x : Coord or tuple
        A tuple with two elements specifying row and column of the coordinate,
        or a Coord instance.
    size : int
        The number of coordinates along one dimension of a Board.

    Returns
    -------
    Bool
        True if x is on the board, False otherwise.

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
        """
        A representation of a single space on the Battleship board. 
        A coord instance may be represented as row and column indices, or as
        a string with a letter and a number.

        A Coord may be initialized in any of the following formats:
        Row/Col format:
            Coord(row,col)      Row and col are integers between 0 and the
                                size of the board minus 1.
        Tuple format:
            Coord((row,col))    The tuple (row,col) is passed as a single 
                                parameter. Row and col are again integers.
        String format:
            Coord("X#")         X is a letter between A and the letter 
                                corresponding to the board size (typically J,
                                for a 10 x 10 board), and # is an integer 
                                between 1 and the board size (typically 10).
        Parameters
        ----------
        a : str or tuple or int
            See input formats, above.
        b : int, optional
            The column index. Only used in Row/Col input format, described 
            above. The default is None.

        Raises
        ------
        ValueError
            DESCRIPTION.
        TypeError
            When the inputs are not str, tuple, or int.

        Returns
        -------
        An instance of a Coord object. Important properties are:
            rowcol      A two-element tuple that gives the row a column of 
                        the Coord.
            lbl         The string representation of the row and column 
                        (e.g., "C6").

        """
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
            sink = Ship.data[outcome["sunk_ship_id"]]["name"] + " sunk!"
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
                  f"{sum(1 - self.loser.board.is_fleet_afloat())} ships.")
            print(f"  Player {self.loser.name} took " 
                  f"{len(self.loser.shot_history)} shots, and sank " 
                  f"{sum(1 - self.winner.board.is_fleet_afloat())} ships.")
            print(f"Game length: {self.turn_count} turns.")
            print(f"(Game ID: {self.game_id})")
            print("")
        else:
            print("")
            print("GAME TERMINATED - NO WINNER.")
            print(f"  Player {self.player1.name} (player 1) took " 
                  f"{len(self.player1.shot_history)} shots, and sank "
                  f"{sum(1 - self.player2.board.is_fleet_afloat())} ships.")
            print(f"  Player {self.player2.name} (player 2) took " 
                  f"{len(self.player2.shot_history)} shots, and sank "
                  f"{sum(1 - self.player1.board.is_fleet_afloat())} ships.")
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

        """
        self._size = size
        self._fleet = {}
        self._ship_placements = {}
        self._ocean_grid = np.zeros((size, size), dtype=np.int8)      
        self._target_grid = np.ones((size, size)) * TargetValue.UNKNOWN
        
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
    
    # Re-initalization
    
    def reset(self):
        """Removes all ships and pegs from the board."""
        board = Board(self.size)
        self.fleet = board.fleet
        self.target_grid = board.target_grid
        self.ocean_grid = board.ocean_grid
        
    # Factory methods
    
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
    
    @classmethod
    def outcome(cls, coord, hit, sunk=False, sunk_ship_id=None):
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
        sunk_ship_id : Bool, optional
            The ShipType value (int 1-5) of the ship that was sunk.
            The default is None, which means no ship sunk.

        Returns
        -------
        outcome : dict
            A dictionary with the following key/value pairs:
                'coord': (row,col) tuple of the shot location
                'hit': Bool; True if shot was a hit
                'sunk' : Bool; True if ship was sunk
                'sunk_ship_id' : ShipType enum (or int 1-5) indicating 
                                 the type of ship sunk (if one was sunk).

        """
        if sunk_ship_id is not None:
            if not sunk:
                raise ValueError("'sunk' must be True when "
                                 "'sunk_ship_id' is not None.")
        if sunk and not hit:
            raise ValueError("'hit' must be True when 'sunk' is True.")
        return {'coord': coord, 
                'hit': hit,
                'sunk': sunk,
                'sunk_ship_id': sunk_ship_id}
        
    @classmethod
    def test(cls,targets=True):
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
        board.add_ship(1, Coord("J9"), "R") 
        board.add_ship(2, Coord("E8"), "N") 
        board.add_ship(4, Coord("G2"), "W") 
        board.add_ship(3, Coord("B6"), "W") 
        board.add_ship(5, Coord("A5"), "N")
        
        if targets:
            board.update_target_grid(Board.outcome((0,4), True))
            board.update_target_grid(Board.outcome((0,3), True))
            board.update_target_grid(Board.outcome((0,5), True, True, ShipType(3)))
            board.update_target_grid(Board.outcome((2,8), False))
            board.update_target_grid(Board.outcome((9,7), False))
            board.update_target_grid(Board.outcome((6,7), False))
            board.update_target_grid(Board.outcome((4,4), False))
            board.update_target_grid(Board.outcome((3,2), False))
            board.update_target_grid(Board.outcome((1,6), False))
            
            board.fleet[ShipType(2)].hit(0)
            board.fleet[ShipType(2)].hit(1)
            board.fleet[ShipType(2)].hit(2)
            board.fleet[ShipType(4)].hit(3)
            board.fleet[ShipType(5)].hit(2)
        
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
            return ds
        else:
            ds = np.vstack(([0,0], 
                            np.cumsum(np.tile(ds, (length-1,1)), axis=0)))
            return (ds[:,0], ds[:,1])
        
    def coords_for_placement(self, placement, length):
        """
        Returns the coordinates that correspond to the input placement
        (coordinate, heading) and ship length.

        Parameters
        ----------
        placement : dict 
            placement['coord']: A coordinate (a tuple with two elements
            corresponding to a row and a column) 
            placement['heading']: A string denoting a heading 
            ("N", "S", "E", or "W").
        length : int
            The length of a ship placed at the input coordinate facing the
            input heading.
            The output will have a number of elements equal to length.

        Returns
        -------
        List of coordinates that span the input starting coordinate/heading
        and the length.

        """
        coord = placement['coord']
        heading = placement['heading']
        ds = self.rel_coords_for_heading(heading, length)
        return list(zip(coord[0] + ds[0], coord[1] + ds[1]))
    
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
        rows, cols = zip(*coords)
        rows = np.array(rows)
        cols = np.array(cols)
        return np.array((rows >= 0) * (rows < self.size) * (cols >= 0) *
                         (cols < self.size))
    
    @classmethod
    def coords_to_coords_distances(cls, coords1, coords2):
        """
        Returns a 2d array containing the distances between all pairs of 
        coordinates in the two input arrays.
        
        Parameters
        ----------
        coords1 : list
            A list of (row,col) tuples.
        coords2 : list
            A list of (row,col) tuples.
        
        Returns
        -------
        distances : numpy array (2d).
            distances[i,j] is the distance between coords1[i] and coords2[j]

        """
        if isinstance(coords1, list):
            coords1 = np.array(coords1)
        if isinstance(coords2, list):
            coords2 = np.array(coords2)
        n2 = coords2.shape[0]
        distances = np.zeros((coords1.shape[0], coords2.shape[0]))
        for row in range(coords1.shape[0]):
            coord1 = np.tile(coords1[row,:], (n2,1))
            distances[row,:] = np.sqrt(np.sum((coord1 - coords2)**2, axis=1))
        return distances
    
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
        untargeted : TYPE, optional
            If True, only coordinates on the target grid that have not been
            fired at will be sampled. The default is False.

        Raises
        ------
        ValueError
            DESCRIPTION.

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
        if (alignment == None or 
                alignment.lower() == "any" or 
                alignment == Align.ANY):
            return(np.random.choice(['N','S','E','W']))
        if (alignment == "NW" or alignment.lower() == "standard"):
            return(np.random.choice(["N", "W"]))
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
            xr,xc = np.where(self.target_grid != TargetValue.UNKNOWN)
            exclude = list(zip(xr, xc))
        elif targeted:
            r,c = np.where(self.target_grid != TargetValue.UNKNOWN)
            indices = list(zip(r,c))
        if exclude:
            indices = [(r,c) for r in range(self.size) 
                       for c in range(self.size) 
                       if (r,c) not in exclude]
        else:
            indices = [(r,c) for r in range(self.size) 
                       for c in range(self.size)]
        return indices
    
    def colinear_targets(self, coords, include_misses=False):
        """
        Returns the two untargeted coordinates at the end of the line made 
        by the input coordinates (which are typically hits).
        
        If the include_misses parameter is True, the returned targets will be
        the first untargeted space at the end of the line of hits and/or 
        misses made by the input coords. 
        
        If include_misses is False (default), the returned targets will be
        the first untargeted space at the end of the line of hits only formed
        by coords. Any misses encountered at the end of the line will 
        terminate the search in that direction
        
        As an example, consider the following row (where - = untargeted, 
        O = miss, X = hit):
        
            - - 0 X 0 X X X - - 
        
        If coords = [(0,5), (0,6)] and include_misses == True, the output 
        will be [(0,1), (0,8)]. However, if include_misses == False, the
        output will be [(0,8)] because there is no untarget spot to the left
        of the row of contiguous hits (i.e., there is a miss at (0,4) that
        ends the search).
        

        Parameters
        ----------
        coords : list of tuples
            A list of at least 2 row/col tuples. These coordinates should lie
            along a single row or column on the board.
        include_misses : Bool
            If True, the coord.

        Returns
        -------
        list
            A list containing the row/col tuples of the coordinates at the
            end of the line formed by the input coordinates.

        
        """
        # find the first unknown or edge
        ds = (np.diff(np.array(coords), axis=0))
        ia,ib = sorted([coords[0], coords[-1]])
        choices = []
        
        if np.any(ds[:,0] != 0):
            while (ia[0] >= 0 and 
                   self.target_grid[ia] != TargetValue.UNKNOWN):
                ia = (ia[0] - 1, ia[1])
            while (ib[0] < self.size and 
                   self.target_grid[ib] != TargetValue.UNKNOWN):
                ib = (ib[0] + 1, ib[1])
            if (ia[0] >= 0 and 
                    (self.target_grid[ia[0]+1,ia[1]] != TargetValue.MISS or
                     include_misses == True)):
                choices += [ia]
            if (ib[0] < self.size and 
                    (self.target_grid[ib[0]-1,ib[1]] != TargetValue.MISS or
                     include_misses == True)):
                choices += [ib]
                
        elif np.any(ds[:,1] != 0):
            while (ia[1] >= 0 and 
                   self.target_grid[ia] != TargetValue.UNKNOWN):
                ia = (ia[0], ia[1] - 1)
            while (ib[1] < self.size and 
                   self.target_grid[ib] != TargetValue.UNKNOWN):
                ib = (ib[0], ib[1] + 1)
            if (ia[1] >= 0 and 
                    (self.target_grid[ia[0],ia[1]+1] != TargetValue.MISS or
                     include_misses == False)):
                choices += [ia]
            if (ib[1] < self.size and 
                    (self.target_grid[ib[0],ib[1]-1] != TargetValue.MISS or
                     include_misses == False)):
                choices += [ib]
        
        return [rc for rc in choices if rc not in coords]
    
    def targets_around(self, coord, diagonal=False, 
                       targeted=False, untargeted=False):
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

        Returns
        -------
        list of tuples
            A list of tuples that correspond to the coordinates that are
            immediately adjacent (and diagonally adjacent, if diagonal is True)
            to the input coordinate.

        """

        if untargeted:
            exclude = self.all_targets(targeted=True)
        if targeted:
            exclude = self.all_targets(untargeted=True)
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
    
    def is_valid_target_ship_placement(self, placement, ship):
        """
        Returns True if the target grid allows for the input ship and placement 
        (starting coord and heading). That is, if there are only unknowns or 
        hits at the coordinates spanned by the ship's placement, or if any such
        spaces are known to be hits on the input ship type.

        Parameters
        ----------
        placement : dict
            A dict with 'coord' and 'heading' keys that specify the coordinate
            and facing of a ship.
        ship : Ship
            The ship instance that is being checked for consistency with the
            target grid's state.

        Returns
        -------
        Bool
            True if the placement for a target ship lies on the target grid and
            is consistent with the known hits & misses.

        """
        coords = self.coords_for_placement(placement, ship.length)
        on_board = self.is_valid_coord(coords)
        if np.any(on_board == False):
            return False
        rows,cols = zip(*coords)
        rows = np.array(rows)
        cols = np.array(cols)
        grid_vals = self.target_grid[rows, cols]
        fits_target_grid = np.all((grid_vals == TargetValue.HIT)
                                  + (grid_vals == TargetValue.UNKNOWN)
                                  + (np.round(grid_vals) == ship.ship_id))
        return fits_target_grid
    
    def all_valid_target_placements(self, ship_type):
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

        Returns
        -------
        placements : list
            List of placement dictionaries that contain all possible 
            coordinates and headings for the input ship_type.

        """
        
        ship = Ship(ship_type)
        ship_len = ship.length
        if np.any(self.target_grid == ship_type):
            allowed_coords = np.zeros((self.size, self.size))
            rows,cols = np.where(self.target_grid == ship_type)
            for (r,c) in zip(rows,cols):
                allowed_coords[(r-ship_len+1):(r+ship_len-1),c] = 1
                allowed_coords[r,(c-ship_len+1):(c+ship_len-1)] = 1      
            pass
        else:
            allowed_coords = np.ones((self.size, self.size))
        allowed_coords[self.target_grid == TargetValue.MISS] = 0
        allowed_coords[(self.target_grid > 0) 
                       * (self.target_grid != ship_type)] = 0
        rows,cols = np.where(allowed_coords)
        coords = list(zip(rows,cols))
        placements = []
        for heading in ["N","W"]:
            for coord in coords:
                p = {'coord': coord, 'heading': heading}
                values = self.target_grid[coord]
                if np.all((values == TargetValue.UNKNOWN) 
                          + (values == TargetValue.HIT) 
                          + (np.round(values) == ship_type)):
                    placements += [p]
        return placements
        
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
            r,c = np.where(self.target_grid != 0)
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
        if occuppied:
            exclude = self.all_coords(unoccuppied=True)
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
    
    def is_valid_placement(self, placement, length, unoccuppied=False):
        """
        Returns True if a ship with the specified length placed at a coordinate
        and heading contained in placement lies entirely on the board.
        If the unoccuppied parameter is set to True, the placement is only
        valid if it does not overlap another ship.

        Parameters
        ----------
        placement : dict
            A dict with 'coord' and 'heading' keys that specify the coordinate
            and facing of a ship.
        length : int
            The length of a ship placed at the input placement parameter.
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
        coords = self.coords_for_placement(placement, length)
        if unoccuppied:
            rows,cols = zip(*coords)
            if np.any(self.ocean_grid[rows,cols] > 0):
                return False
        return np.all(self.is_valid_coord(coords))
    
    def all_valid_ship_placements(self, ship_type, distance=0, diagonal=True,
                                  edge_buffer=0, alignment=Align.ANY):
        """
        Return all placements that would allow the input ship_type to be placed
        on the ocean grid.

        Parameters
        ----------
        ship_type : hipType (EnumInt) or int 1-5.
            The type of ship that is allowed at the returned placements.
        distance : int, optional
            Only include placements in which the ship is at least this many
            spaces from any part of any other ship. A distance of 0 means
            that the placements can be adjacent to another ship.
            The default is 0.
        diagonal : Bool, optional
            If True, and if the distance input is 0, placements that would put
            ship diagonally adjacent to another ship are allowed (whereas they 
            are not allowed if diagonal is False).
            If distance is >1, diagonal = True causes the calculated separation
            between ship coordinates to be rounded down. Otherwise the 
            separation will be rounded to the nearest integer.
            diagonal has no effect if distance is 0. 
            The default is True.
        edge_buffer : int, optional
            If 0, placements next to the board edges are allowed. If
            a positive integer, any placements must allow for at least 
            edge_buffer spaces between any part of the ship and the board's 
            edge. 
            The default is 0.
        alignment : Align (Enum) value, optional
            The ship placements must be aligned horizontally if alignment is
            Align.HORIZONTAL or vertically if alignment is Align.VERTICAL. 
            Ship alignment can be either horizontal or vertical if 
            alignemnt is Align.ANY. The default is Align.ANY.

        Returns
        -------
        placements : list of placement dictionaries ("coord" and "heading" 
                                                     keys).

        """
        headings_dict = {"N": (1,0), "S": (-1,0), "E": (0,-1), "W": (0,1)}
        ship = Ship(ship_type)
        round_fcn = np.round if diagonal else np.floor
        ship_len = ship.length
        
        if alignment == Align.VERTICAL:
            allowed_headings = ["N"]
        elif alignment == Align.HORIZONTAL:
            allowed_headings = ["W"]
        else:
            allowed_headings = ["N", "W"]
        placements = []
        
        rows,cols = np.where(self.ocean_grid > 0)
        occuppied_coords = list(zip(rows,cols))
        for heading in allowed_headings:
            allowed_spaces = np.ones((self.size, self.size))
            if edge_buffer:
                allowed_spaces[edge_buffer,:] = 0.
                allowed_spaces[-edge_buffer,:] = 0.
                allowed_spaces[:,edge_buffer] = 0.
                allowed_spaces[0,-edge_buffer] = 0.                
            ds = headings_dict[heading]
            if ds[0] > 0:
                allowed_spaces[-(ds[0]*ship_len-1):,:] = 0.
            elif ds[1] > 0:
                allowed_spaces[:,-(ds[1]*ship_len-1):] = 0.
            rows,cols = np.where(allowed_spaces)
            coords = list(zip(rows,cols))
            for coord in coords:
                if self.ocean_grid[coord] != 0:     # coord already occuppied
                    allowed_spaces[coord] = 0
                # check if placement starting at coord is within distance
                # of any existing ships (also checks if there is overlap)
                else:   
                    p = {'coord': coord, 'heading': heading}
                    current_coords = self.coords_for_placement(p, ship_len)
                    min_separation = np.min(Board.coords_to_coords_distances(
                        current_coords, occuppied_coords))
                    if round_fcn(min_separation) <= distance + 1:
                        allowed_spaces = 0
            rows,cols = np.where(allowed_spaces)
            coords = list(zip(rows,cols))
            placements += [{'coord': c, 'heading': heading} for c in coords]
        return placements
    
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
        # indexing coord allows for both tuple and Coord input.
        val = self.ocean_grid[(coord[0], coord[1])]
        if val == 0:
            return None
        else:
            return self.ship(val)
        
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
        return np.array([self.fleet[k].is_afloat() for k in keys])
            
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
        if isinstance(ship, Ship):
            ship_type = int(ship.ship_id)
        else:
            ship_type = ship
        if not isinstance(ship_type, int):
            raise TypeError("ship must be an instance of Ship or an int.")
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
        return {"coord": coord, "heading": heading}
    
    # Fleet Placement Methods
    
    def place_fleet(self, placements):
        """
        Places the ships with locations described by the placements dict onto
        the ocean grid.

        Parameters
        ----------
        placements : dict
            A dictionary with keys equal to the ShipTypes (int/EnumInt 1-5) 
            that are to be placed on the board. The value for each ShipType
            key should be a placement dictionary, with format:
                placement['coord']: Tuple containing the (row,col) of the
                    first coordinate of the ship.
                placement['heading']: Character specifying the cardinal
                    direction ("N"/"S"/"E"/"W") that the ship should point.
                    
            An example placements dictionary would be as follows:
                placements = {
                    ShipType(1): {'coord': (r1,c1), 'heading': h1}},
                    ShipType(1): {'coord': (r2,c2), 'heading': h2}},
                    ShipType(1): {'coord': (r3,c3), 'heading': h3}},
                    ShipType(1): {'coord': (r4,c4), 'heading': h4}}
                    ShipType(1): {'coord': (r5,c5), 'heading': h5}}
                }
            where rX and cX are integers between 0 and 9 (for a 10 x 10 board)
            and hX is a direction-specifying character "N", "S", "E", or "W".
                

        Returns
        -------
        None.

        """
        for (k,placement) in placements.items():
            self.add_ship(k, placement["coord"], placement["heading"])
    
    def add_ship(self, ship_type, coord, heading):
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
        elif isinstance(ship_type, Ship):
            ship_type = ship_type.ship_id            
        ship = Ship(ship_type)
        if not self.is_valid_placement({'coord':coord, 'heading':heading}, 
                                       ship.length, unoccuppied=True):
            raise Exception(f"Cannot place {ship.name} at {coord}.")
        else:
            self.fleet[ship.ship_id] = ship
            drows,dcols = self.rel_coords_for_heading(heading, ship.length)
            self.ocean_grid[coord[0] + drows, coord[1] + dcols] = ship.ship_id
        
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
                "sunk_ship_id": ShipType enum
                "message": string

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
        dmg_slot = [i for i in range(len(ship_coords)) 
                    if ship_coords[i] == coord]
        if len(dmg_slot) != 1:
            raise Exception(f"Could not determine unique damage location for "
                            f"target {coord}")
        damage = ship.hit(dmg_slot[0])
        return ship.sunk(), damage
        
    def update_target_grid(self, outcome):
        """
        Updates the targets grid with the results of the outcome dictionary.
        which contains the following key/value pairs:
            

        Parameters
        ----------
        outcome : dict
            Dictionary that contains the following key/value pairs:
                
            "coord" : Tuple with the (row,col) where the shot was fired.
            "hit" : Bool - True if a ship was hit, False if shot was a miss.
            "sunk" : Bool - True if a ship was sunk by the outcome's shot.
            "sunk_ship_id" : int corresponding to the ShipType of a ship if
                a ship was sunk by the outcome's shot..

        Returns
        -------
        None.

        """
        coord = outcome["coord"]
        ship_type = outcome["sunk_ship_id"]
        if isinstance(coord, Coord):
            coord = coord.rowcol
        prev_val = self.target_grid[coord]
        if outcome["hit"]:
            if ship_type:
                self.target_grid[coord] = ship_type + SUNK_OFFSET
                self.update_target_grid_for_sink(coord, ship_type)
            else:
                self.target_grid[coord] = TargetValue.HIT   
        else:
            self.target_grid[coord] = TargetValue.MISS
        if prev_val != TargetValue.UNKNOWN:
            print(f"Repeat target: {Coord(coord).lbl}")
    
    def update_target_grid_for_sink(self, coord, sunk_ship_id):
        """
        Updates the target grid based on knowing the identity of the ship
        sunk at the coord parameter. Updates will be made if there are hits 
        on the target grid that can be unambiguously assigned to a ship type
        based on the knowledge that sunk_ship_id is at coord.

        Parameters
        ----------
        coord : Tuple (row,col)
            The (row,col) coordinate of the shot that sunk ship with type
            sunk_ship_id.
        sunk_ship_id : ShipType (EnumInt)
            The type of the ship that was sunk by the shot that landed at
            coord.

        Raises
        ------
        ValueError
            Raised if the target grid is not consistent with the input ship 
            type being located at coord.

        Returns
        -------
        None.

        """
        pass
        print("Updating for sink: ", coord,sunk_ship_id)
        ship_length = Ship.data[ShipType(sunk_ship_id)]["length"]
        
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
            raise ValueError("Failed to find ship location in target grid.")
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
        if not np.all(self.target_grid == TargetValue.UNKNOWN):
            return False
        if not (np.sum(self.ocean_grid) == 
                sum([ship.length for ship in self.fleet.values()])):
            return False
        for ship in self.fleet.values():
            if np.any(ship.damage > 0):
                return False
        ship_ids = [ship.ship_id for ship in self.fleet.values()]
        if sorted(ship_ids) != list(range(1, len(Ship.data)+1)):
            return False
        return True
    
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
        
    def color_str(self):
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
    
    # Targeting Methods
    
    def all_target_placements_for_ship_type(self, ship_type):
        """
        Return a list of placement dictionaries that enumerate all possible
        placements where the input ship can be located based on the hits, 
        misses, and known ship types (based on sink events) on the target grid.

        Parameters
        ----------
        ship_type : ShipType (EnumInt) or int 1-5.
            The type of ship that can be placed at all returned placements.

        Returns
        -------
        placements : list
            List of dictionaries, each of which contains placement info
            for a possible ship location and heading. Each dict has keys
            "coord" and "heading", which have values that are a (row,col) tuple
            and a direction character ("N","S","E","W"), respectively.

        """
        placements = []
        ship = Ship(ship_type)
        bad_targets = (self.target_grid == TargetValue.MISS)
        for tp in range(min(Ship.data),max(Ship.data.keys())+1):
            if (tp != ship_type):
                bad_targets += (np.round(self.target_grid) == tp)
        rows,cols = np.where(~bad_targets)
        for coord in list(zip(rows,cols)):
            placement = {"coord":coord, "heading":"N"}
            if self.is_valid_target_ship_placement(placement, ship):
                placements += [placement]
            placement = {"coord":coord, "heading":"W"}
            if self.is_valid_target_ship_placement(placement, ship):
                placements += [placement]
        return placements
    
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
        ship_types = [ShipType(k) for k in range(min(Ship.data), max(Ship.data)+1)]
        for tp in ship_types:
            counts[tp] = 0
        for tp in ship_types:
            placements = self.all_target_placements_for_ship_type(tp)
            ship_len = Ship.data[tp]["length"]
            for placement in placements:
                place_coords = self.coords_for_placement(placement, ship_len)
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
        ship_types = [ShipType(k) for k in range(min(Ship.data), 
                                                 max(Ship.data)+1)]
        for k in ship_types:
            grid_by_ship[k] = None
            
        for tp in ship_types:
            ship_len = Ship.data[tp]["length"]
            grid = np.zeros(self.target_grid.shape)
            placements = self.all_target_placements_for_ship_type(tp)
            for p in placements:
                for coord in self.coords_for_placement(p, ship_len):
                    grid[coord] += 1
            grid_by_ship[tp] = grid
        if by_type == False:
            out = np.zeros(self.target_grid.shape)
            for k in grid_by_ship:
                out += grid_by_ship[k]
            return out
        return grid_by_ship

    
#%% Ship Class
class Ship:
    
    _data = {ShipType.PATROL: {"name": "Patrol".title(), "length": 2},
             ShipType.DESTROYER: {"name": "Destroyer".title(), "length": 3},
             ShipType.SUBMARINE: {"name": "Submarine".title(), "length": 3},
             ShipType.BATTLESHIP: {"name": "Battleship".title(), "length": 4},
             ShipType.CARRIER: {"name": "Carrier".title(), "length": 5}
             }
    
    def __init__(self, ship_type):
        """
        Initialize a ship object with type ship_type, 

        Parameters
        ----------
        ship_type : int, ShipType (EnumInt), or str
            An integer value (or equivalent EnumInt) between 1 and 5, or
            a string that describes the type of ship desired. Valid strings
            and their corresponding integer values are:
                Patrol (1)
                Destroyer (2)
                Submarine (3)
                Battleship (4)
                Carrier (5)

        Returns
        -------
        An instance of Ship with properties corresponding the the ship_type
        parameter. Each instance has properties:
            ship_id (ShipType enum)
            name (str)
            length (int)
            damage (numpy 1d array with length equal to the length property)
            

        """
        if isinstance(ship_type, str):
            ship_type = Ship.type_for_name(ship_type)
        self._ship_id = ShipType(ship_type)
            
        self.name = Ship.data[self._ship_id]["name"]
        self.length = Ship.data[self._ship_id]["length"]
        self.damage = np.zeros(self.length, dtype=np.int8)
        
    def __len__(self):
        """
        Return the length of the ship, which is equal to both the number
        of adjacent coordinates on the board that the ship occuppies, as well
        as the number of damage slots on the ship.
        
        Returns
        -------
        int
            The number of slots on the ship insance.
        """
        return self.length
    
    def __str__(self):
        """
        Returns a string listing the Ship instance name, length, damage,
        and whether the ship is sunk.

        Returns
        -------
        str
        
        """
        s = (f"{self.name} (id = {self._ship_id}) - {self.length} slots, " 
              f"damage: {str(self.damage)}")
        if self.is_sunk():
            s += " (SUNK)"
        return(s)
    
    def __repr__(self):
        """
        Returns a string representation that can be evaluated to recreate
        the object

        Returns
        -------
        str
        
        """
        return f"Ship({self.ship_id})"        
        
    @property
    def ship_id(self):
        return self._ship_id
    
    @classmethod
    def type_for_name(self, name):
        """
        Returns the ship type integer that corresponds to the input ship name
        string. The output is a value of the ShipType enum, which is equivalent
        to an integer.

        Parameters
        ----------
        name : str
            A string description of the desired ship type. One of 'patrol',
            'destroyer', 'submarine' (or 'sub'), 'battleship', or 'carrier'.
            Case insensitive.

        Returns
        -------
        ShipType (EnumInt)
            An integer value 1-5 that can be used as a key for accessing
            info about the particular ship type from the 'data' class
            method.

        """
        name = name.title()
        try:
            idx = [ship_dict["name"] for ship_dict in 
                   Ship.data.values()].index(name)
        except ValueError:
            if name == "Sub":
                name += "marine"
            elif name == "Patrol Boat":
                name = "Patrol"
            idx = [ship_dict["name"] for ship_dict in 
                   Ship.data.values()].index(name)
        return list(Ship.data.keys())[idx]
    
    @classmethod
    @property
    def data(cls):
        """
        Returns a dictionary containing name and length data for each type
        of ship. The dictionary's keys are the ShipType Enum values.

        Parameters
        ----------
        None

        Returns
        -------
        dict
            A dictionary with the following keys. The corresponding value
            gives the value of property described by the key for the ship type
            given by the ship_type input (1 for Patrol, 2 for Destroyer, etc.)
            'name':     The name of the ship type.
            'length':   The number of coordinates the ship occupied on a board
                        (equal to number of times it can be hit before sinking).

        """
        return cls._data
        
    # Combat Methods
    
    def hit(self, slot):
        """
        Adds a point of damage to the ship at the input slot position.

        Parameters
        ----------
        slot : int
            The slot at which the ship will be damaged. Conforms to standard
            list indexing, so slot should be on the interval [0, ship.length].

        Raises
        ------
        ValueError
            If the slot parameter falls outside of the valid index range for
            the ship's length.

        Returns
        -------
        int     The total amage at the input slot.

        """
        if slot < 0 or slot >= self.length:
            raise ValueError("Hit at slot #" + str(slot) + " is not valid; "
                             "must be 0 to " + str(self.length-1))
        self.damage[slot] += 1
        if self.damage[slot] > 1:
            Warning("Slot " + str(slot) + " is already damaged.")
        return(self.damage[slot])
    
    def is_afloat(self):
        """
        Returns True if the ship has not been sunk (i.e., if it has at least
        one slot that has not taken damage).

        Returns
        -------
        Bool    True if the ship is still afloat, False if it is not.

        """
        
        return(any(dmg < 1 for dmg in self.damage))

    def is_sunk(self):
        """
        Returns True if the ship has been sunk (i.e., if it has at least
        one damage level at each slot).

        Returns
        -------
        Bool    True if the ship is sunk, False if it is afloat.

        """
        return(all(dmg > 0 for dmg in self.damage))
