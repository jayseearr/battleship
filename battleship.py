#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 12:11:29 2022

@author: jason
"""

#%%
# Stuff to do:
    # * Implement a Strategy class that can be given to a Player. The class
    #   should implement methods 'target' and 'placement' to give a 
    #   target when Player calls .pick_target, and .place_fleet (or whatever) 
    #   for a given board and shot/outcome history.
    #
    # * Separate Player class into AIPlayer and HumanPlayer subclasses.
    
#%%
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from enum import IntEnum
import abc

#%%
# Constants
# LETTER_OFFSET = 65
# DEFAULT_BOARD_SIZE = 10
MAX_ITER = 1000
    
# Enums
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
    
# Data
SHIP_DATA = {ShipType.PATROL: {"name": "Patrol".title(), "length": 2},
             ShipType.DESTROYER: {"name": "Destroyer".title(), "length": 3},
             ShipType.SUBMARINE: {"name": "Submarine".title(), "length": 3},
             ShipType.BATTLESHIP: {"name": "Battleship".title(), "length": 4},
             ShipType.CARRIER: {"name": "Carrier".title(), "length": 5}
             }

#%%
# Text color escapes
class fg:
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

#%% Functions

def random_choice(x, p=None):
    """Returns a single randomly sampled element from the input list, tuple,
    or array."""
    return x[np.random.choice(range(len(x)), p=p)]

def battleship_sim(ngames):
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
        
        self.game_id = game_id
        
        if isinstance(player1, Player):
            self.player1 = player1
        else:
            self.player1 = Player(player1)
            
        if isinstance(player2, Player):
            self.player2 = player2
        else:
            self.player2 = Player(player2)
            
        self.ready = False
        self.winner = None
        self.loser = None
        self.turn_count = 0
        self.verbose = verbose
        
    def setup(self):
        """Set up a game by placing both fleets."""
        self.player1.set_opponent(self.player2)
        self.player2.set_opponent(self.player1)
        self.player1.set_up_fleet()
        self.player2.set_up_fleet()
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
        print(f"Turn {len(player.shot_log)}: Player {name} fired a shot at "
              f"{target}...{hit_or_miss} {sink}")
        
    def print_outcome(self):
        """Prints the results of the game for the input winner/loser players 
        (which include their respective boards)."""
        
        print("")
        print("GAME OVER!")
        print("Player " + self.winner.name + " wins.")
        print("  Player " + self.winner.name + " took " \
              + str(len(self.winner.shot_log)) + " shots, and sank " \
              + str(sum(1 - self.loser.board.afloat_ships())) + " ships.")
        print("  Player " + self.loser.name + " took " \
              + str(len(self.loser.shot_log)) + " shots, and sank " \
              + str(sum(1 - self.winner.board.afloat_ships())) + " ships.")
        print("Game length: " + str(self.turn_count) + " turns.")
        print("(Game ID: " + str(self.game_id) + ")")
        print("")
            
# %% Player Class
class Player:
    """A Battleship player that has access to a board on which a game
    can be played. The player can be human or AI. In the case of human,
    this class provides an interface for selecting targets. In the case of
    an AI, the class has a 'strategy' that automatically selects targets.
    
    Valid placement strategies are:
        random, cluster, isolated, very.isolated
        
    Valid offense strategies are:
        random, random.hunt.dumb, random.hunt
    """
    
    def __init__(self, player_type, strategy, name=None):
        if player_type.lower() == "human":
            self.type = "Human"
        elif player_type.lower() == "robot" or player_type.lower() == "ai":
            self.type = "Robot"     
            
        if isinstance(strategy, str):
            strategy = (strategy, strategy)
        
        self.strategy = {'placement': strategy[0], 
                         'offense': strategy[1]}
        if self.type.lower() == "human":
            self.strategy['offense'] = "manual"
            
        self.strategy_data = {"state": "hunt",
                              "last_hit": None,
                              "last_miss": None,
                              "last_target": None,
                              "current_target_id": None,
                              "last_sink_id": None,
                              "hits_since_last_sink": [],
                              "message": None}

        self.name = name
        
        # the following variables may change with each game:
        self.shot_log = []
        self.outcome_log = []
        self.board = Board()
        self.opponent = None
        self.remaining_targets = self.board.all_coords()
        self.possible_targets = []
        
        self.game_history = None    # for future use
        self.notes = None           # for future use
        
    def __str__(self):
        """Returns a string indicating player type, strategy, and any strategy
        data.
        """
        strat1 = self.strategy["placement"]
        strat2 = self.strategy["offense"]
        return f"Player {self.name}\n" \
            f"  Type: {self.type}\n" \
            f"  Placement strategy: {strat1}\n" \
            f"  Offensive strategy: {strat2}\n" \
            f"  Strategy data: {self.strategy_data}"
            
    def reset(self):
        """Removes all ships and hit/miss indicators from the player's board,
        and empties out the shot and outome logs. The opponent remains the 
        same until changed usint set_opponent.
        """
        self.shot_log = []
        self.outcome_log = []
        self.board.reset()
        self.remaining_targets = self.board.all_coords()
        self.strategy_data = Player(self.type, (self.strategy["placement"],
                                                self.strategy["offense"])).strategy_data
        
    def opponents_board(self):
        """Returns the opponent's Board."""
        return self.opponent.board
        
    def random_ship_placement(self, method, ship_id=None):
        """Returns a random ship placement (Coord and heading) for the input 
        method and ship type.
        """ 
        if method == "random":
            coord = self.board.random_coord(unoccuppied=True)
            heading = self.board.random_heading()
            
        elif method in ["cluster", "clustered"]:
            X,Y = np.meshgrid(range(self.board.size), range(self.board.size))
            D = np.zeros(X.shape)
            for ship in list(self.board.fleet.values()):
                coords = self.board.coord_for_ship(ship)
                rows,cols = zip(*coords)
                for (r,c) in zip(rows,cols): 
                    D = D + (X - c)**2 + (Y - r)**2
            coords = self.board.all_coords(unoccuppied=True)
            prob = np.zeros(len(coords))
            for (i,rc) in enumerate(coords):
                prob[i] = 1 / (1 + D[rc[0],rc[1]])
            prob = np.array(prob)
            prob = prob / np.sum(prob)
            coord = random_choice(coords, p=prob)
            heading = self.board.random_heading()
            
        elif (method == "isolated" or method in 
              ["veryisolated", "very.isolated", "very isolated", 
               "very_isolated", "isolated+"]):
            # find a placement that is not touching another ship
            sites = self.board.all_valid_placements(
                SHIP_DATA[ShipType(ship_id)]["length"],
                distance=1, diagonal=(method != "isolated"))
            coord, heading = random_choice(sites)
        
        return coord, heading
    
    def place_fleet_randomly(self, method):
        """Places all 5 ships randomly on the board, with randomization method
        set by the 'method' input. This input can be 'random', 'cluster', 
        'isolated', or 'very.isolated'.
        """
        method = method.lower()
        for ship_id in SHIP_DATA:
            ok = False
            counter = 0
            while not ok and counter < MAX_ITER:
                coord, heading = self.random_ship_placement(method, ship_id)
                ok = self.board.is_valid_ship_placement(ship_id, coord, heading)
                counter += 1
            if not ok:
                raise Exception("Max ship placement repetions has been reached.")
            self.board.place_ship(ship_id, coord, heading)
        return ok
            
    def set_up_fleet(self): 
        """Place ships in the fleet either manually via text input (for Human
        player) or according to the placement strategy.
        """
        if self.type == "Human" and self.strategy["placement"] != "random":
            self.place_fleet_manually()
        else:
            self.place_fleet_randomly(self.strategy["placement"])
                
    def place_fleet_manually(self):
        """Provides a text-based interface on the console for fleet placement.
        """
        for i in range(1,len(SHIP_DATA)+1):
            ok = False
            name = SHIP_DATA[ShipType(i)]["name"]
            while not ok:
                coord = Coord(input("Space for {name.upper())}: "))
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
        
    def last_target(self, as_lettnum=False):  
        """Returns the most recently targeted space. Defaults to row/column 
        indices for output, but returns a letter/number string if the as_lettnum
        input is set to True.
        """
        if not self.shot_log:
            return None
        else:
            return self.shot_log[-1]
        
    def last_outcome(self): 
        """Returns the outcome of the last shot."""
        if not self.outcome_log:
            return None
        else:
            return self.outcome_log[-1]
    
    def set_opponent(self, other_player): 
        """Sets the opponent variable to the input Player.
        Should also do the same for the other player (this does not happen 
        automatically so as to avoid circular referencing).
        """
        if isinstance(other_player, Player):
            self.opponent = other_player
        else:
            raise TypeError("other_player must be an instance of Player.")
            
    def isalive(self): 
        """Returns true if any of this players ships are still afloat."""
        return any(self.board.afloat_ships())

    def take_turn(self): 
        """Fire a shot at the opponent, get the outcome, and update any
        strategy-related variables."""
        
        if self.type == "Human":
            target = input("Enter target: ").upper()
            if target in self.shot_log:
                target = input((f"*** Space {target} has already been \
                                targeted.\nEnter again to confirm, or select \
                                another target: "))           
        else:
            target = self.pick_target()
        return self.fire_at_target(target) 
    
    def update_state(self, last_target, last_outcome):  
        """Updates strategy_data based on the last targeted space and the
        resulting outcome.
        """
        strat_data = self.strategy_data
        strat_data["last_target"] = last_target
        if last_outcome["sunk_ship_id"]:
            strat_data["state"] = "hunt"
            strat_data["last_hit"] = last_target
            strat_data["current_target_id"] = None
            strat_data["last_sink_id"] = last_outcome["sunk_ship_id"]
            strat_data["hits_since_last_sink"] = []
        elif last_outcome["hit"]:
            strat_data["state"] = "kill"
            strat_data["last_hit"] = last_target
            strat_data["hits_since_last_sink"] += [last_target]
        else:
            strat_data["last_miss"] = last_target
        strat_data["message"] = None
        self.strategy_data = strat_data       
            
    def pick_target(self): 
        """Returns a target coordinate string based on the player's strategy 
        and shot/outcome history.
        This is where the strategy is implemented.
        """
        if self.strategy["offense"] == "random":
            targets = self.board.all_coords(untargeted=True)
        
        elif self.strategy["offense"] == "random.hunt.dumb":
            # If in kill mode, fire around the most recent hit.
            if self.strategy_data["state"] == "hunt":
                targets = self.board.all_coords(untargeted=True)
                self.strategy_data["message"] = "hunt mode."
            else:
                last_hit = self.strategy_data["last_hit"]
                targets = self.board.coords_around(last_hit, untargeted=True)
                self.strategy_data["message"] = "kill mode."
            if not targets:
                self.strategy_data["state"] = "hunt"
                self.strategy_data["message"] += f" No viable target found " \
                    f"around last hit {last_hit}; reverting to hunt mode. "
                targets = self.board.all_coords(untargeted=True)

        elif self.strategy["offense"] == "random.hunt":
            # If in kill mode, find two hits in a row. Then fire in a line
            # until a ship is sunk.
            if self.strategy_data["state"] == "hunt":
                targets = self.board.all_coords(untargeted=True)
                self.strategy_data["message"] = "hunt mode."
            else:
                #last_hit = self.strategy_data["last_hit"]
                hits = self.strategy_data["hits_since_last_sink"]
                self.strategy_data["message"] = "kill mode."
                if len(hits) == 1:
                    targets = self.board.coords_around(hits[-1], untargeted=True)
                    self.strategy_data["message"] += " Searching around last hit."
                elif len(hits) > 1:
                    targets = self.board.colinear_target_coords(hits)
                    self.strategy_data["message"] += " Searching along colinear hits."                    
                else:
                    raise Exception("hits_since_last_sink should not be empty.")
            if not targets:
                self.strategy_data["state"] = "hunt"
                self.strategy_data["message"] += " No viable target found; " \
                    "reverting to hunt mode. "
                targets = self.board.all_coords(untargeted=True)
            
        else:
            s = self.strategy["offense"]
            raise ValueError(f"Invalid offensive strategy: {s}")
            
        self.possible_targets = targets
        return random_choice(targets)  
            
        
    def fire_at_target(self, target_coord): 
        """Fires a shot at the input target space and handles the resulting
        outcome. If the shot results in a hit, the opponent's ship is damaged.
        The Player's board is updated with a hit or miss value at the 
        target_space.
        Returns a Space enum value (MISS or HIT).
        """
        outcome = self.opponent.board.incoming_at_coord(target_coord)
        self.board.update_target_grid(target_coord, 
                                     outcome["hit"], 
                                     outcome["sunk_ship_id"])
        self.shot_log += [target_coord]
        if target_coord in self.remaining_targets:
            self.remaining_targets.pop(self.remaining_targets.index(target_coord))
        self.outcome_log += [outcome]
        self.update_state(target_coord, outcome)
        return outcome
    
    def show_attack(self):
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
     
#%%
class Strategy(metaclass=abc.ABCMeta):
    """A set of instructions that determines how an AI player places ships on
    a board and chooses targets. 
    An Strategy instance knows about the state of the board as well as the
    history of shot coordinates, history of shot outcomes, and the turn number.
    
    An instance of Strategy should implement two methods:
        .pick_target(board, outcome_history)
        .placement_for_ship(board, ship)
    """
      
    @abc.abstractmethod
    def pick_target(self, board, history):
        """Returns a target coordinate based on the input board and outcome 
        history. This is where the smarts are implemented.
        """
        pass
     
    @abc.abstractmethod
    def placement_for_ship(self, board, ship):
        """Returns a tuple with a coordinate and heading where the input 
        ship should be placed. The placement should be valid for the input
        board (not intersecting other ships and not over the board edge).
        """
        pass
    
class IsolatedRandomHunter(Strategy):
    """Placement strategy is Isolated, Offensive strategy is Random Hunter.
    """
        
    def __init__(self):
        self._data = {"state": "hunt",
                      "last_hit": None,
                      "last_miss": None,
                      "last_target": None,
                      "current_target_id": None,
                      "last_sink_id": None,
                      "hits_since_last_sink": [],
                      "message": None}
        
    def pick_target(self, board, history):
        """Hunt randomly until a ship is hit. Then search around the hit to 
        find two adjacent hits. Then fire along that line until a ship is sunk 
        or no other options remain (which may mean that the adjacent hits
        were on two different ships).
        """
        # own_damage = [np.sum(ship.damage) for ship in list(board.fleet.values())]
        # own_sunk = ~board.afloat_ships()
        # nturns = len(history)
        
        if self._data["state"] == "hunt":
            targets = board.all_coords(untargeted=True)
            self._data["message"] = "hunt mode."
        else:
            hits = self._data["hits_since_last_sink"]
            self._data["message"] = "kill mode."
            if len(hits) == 1:
                targets = board.coords_around(hits[-1], untargeted=True)
                self._data["message"] += " Searching around last hit."
            elif len(hits) > 1:
                targets = board.colinear_target_coords(hits)
                self._data["message"] += " Searching along colinear hits."                    
            else:
                raise Exception("hits_since_last_sink should not be empty.")
        if not targets:
            self._data["state"] = "hunt"
            self._data["message"] += " No viable target found; " \
                "reverting to hunt mode. "
            targets = board.all_coords(untargeted=True)
        return targets
            
    def placement_for_ship(self, board, ship):
        """Returns a coordinate and heading that is not touching another ship.
        """
        if not isinstance(ship, Ship):
            ship = Ship(ship)
        sites = board.all_valid_placements(ship.length,
                                           distance = 1, 
                                           diagonal = False)
        coord, heading = random_choice(sites)
        return coord, heading
    
    def update_data(self, outcome):
        """Updates strategy_data based on the last targeted space and the
        resulting outcome.
        """
        last_target = outcome["coord"]
        
        self._data["last_target"] = last_target
        if outcome["sunk_ship_id"]:
            self._data["state"] = "hunt"
            self._data["last_hit"] = last_target
            self._data["current_target_id"] = None
            self._data["last_sink_id"] = outcome["sunk_ship_id"]
            self._data["hits_since_last_sink"] = []
        elif outcome["hit"]:
            self._data["state"] = "kill"
            self._data["last_hit"] = last_target
            self._data["hits_since_last_sink"] += [last_target]
        else:
            self._data["last_miss"] = last_target
        self._data["message"] = None
        
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
    def all_valid_placements(self, length, distance=0, diagonal=False):
        """Returns a list of all possible placements (consisting of row, 
        column, and heading) on the board. The distance parameter controls
        how close the placements can be to another existing ship.
        The returned list of placements is in the form of a list of tuples
        ((row, column), heading).
        """
        headings = {"N": (1,0), "S": (-1,0), "E": (0,-1), "W": (0,1)}
        unoccuppied = self.all_coords(unoccuppied=True)
        placements = []
        for unocc in unoccuppied:
            for h in ["N","S","E","W"]:
                ds = np.vstack(([0,0], np.cumsum(np.tile(
                    headings[h], (length-1,1)), axis=0)))
                rows,cols = unocc[0] + ds[:,0], unocc[1] + ds[:,1]
                ok = True
                if (np.any(rows < 0) or np.any(rows >= self.size) or 
                    np.any(cols < 0) or np.any(cols >= self.size) or
                    np.any(self.ocean_grid[rows,cols])):
                        ok = False
                if distance > 0 and ok:
                    for (r,c) in zip(rows, cols):
                        surrounding = zip(*self.coords_around((r,c), diagonal))
                        if np.any(self.ocean_grid[tuple(surrounding)]):
                            ok = False
                if ok:
                    placements += [(unocc, h)]
        return placements
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
            print(f"Cannot place {ship.name} at {coord}.")
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
            ship_ids = [ship.ship_id.value for ship in self.fleet.values()]
            if sorted(ship_ids) != list(range(1, len(SHIP_DATA.keys())+1)):
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
    
    def random_heading(self):
        """Returns a random heading: N/S/E/W."""
        return(np.random.choice(['N','S','E','W']))
    
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
    
    def show(self):
        """Shows images of the target map and grid on a figure with two 
        subaxes. Colors the images according to hit/miss (for target map) and
        damage (for the grid). Returns the figure containing the images.
        """
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

        fig, axs = plt.subplots(2, 1, figsize = (10,6))

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

        axs[0].imshow(np.zeros(vert_grid.shape), cmap=cmap_vert, 
                      extent=grid_extent)
        axs[1].imshow(flat_grid, cmap=cmap_flat, extent=grid_extent)

        # add pegs
        rows,cols = np.where(flat_grid == 2)
        red_pegs = [plt.Circle((x,y), radius=peg_radius, 
                               facecolor = hit_color, 
                               edgecolor = peg_outline_color) 
                    for (x,y) in zip(cols+1,rows+1)]
        for peg in red_pegs:
            axs[1].add_patch(peg)
            
        # vertical grid
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
        
        # ships
        ship_boxes = self.ship_rects()
        for shipId in ship_boxes:
            box = ship_boxes[shipId]
            axs[1].add_patch( 
                mpl.patches.Rectangle((box[0], box[1]), box[2], box[3], \
                                      edgecolor = ship_outline_color,
                                      fill = False,                                       
                                      linewidth = 2))
            axs[1].text(box[0]+0.12, box[1]+0.65, SHIP_DATA[shipId]["name"][0])

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
        s = (f"{self.name} (id = {self._ship_id.value}) - {self.length} slots, " 
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
