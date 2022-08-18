#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 12:11:29 2022

@author: jason
"""

import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import re
from enum import Enum

# Constants
LETTER_OFFSET = 65
DEFAULT_BOARD_SIZE = 10

class ShipId(Enum):
    PATROL = 1
    DESTROYER = 2
    SUBMARINE = 3
    BATTLESHIP = 4
    CARRIER = 5
    
# Enums
class Space(Enum):
    """An Enum used to track hits/misses on a board's target map.
    Can be compared for equality to integers."""
    UNKNOWN = -2
    MISS = -1
    HIT = 0
    def __eq__(self, other):
         if self.__class__ is other.__class__:
             return self.value == other.value
         elif isinstance(other,int):
             return self.value == other
         return NotImplemented
    
### Data
SHIP_DATA = {ShipId.PATROL: {"name": "Patrol".title(), "length": 2},
             ShipId.DESTROYER: {"name": "Destroyer".title(), "length": 3},
             ShipId.SUBMARINE: {"name": "Submarine".title(), "length": 3},
             ShipId.BATTLESHIP: {"name": "Battleship".title(), "length": 4},
             ShipId.CARRIER: {"name": "Carrier".title(), "length": 5}
             }

# -----------------------------------------------------------------------------
def random_choice(x, p=None):
    """Returns a single randomly sampled element from the input list, tuple,
    or array."""
    return x[np.random.choice(range(len(x)), p=p)]

# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
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
        self.turnCount = 0
        while game_on:
            first_player.take_turn()
            if self.verbose:
                self.report_turn_outcome(first_player)
            second_player.take_turn()
            if self.verbose: 
                self.report_turn_outcome(second_player)
            self.turn_count += 1
            game_on = first_player.still_alive() and second_player.still_alive()
            
        # See who won
        if first_player.still_alive():
            self.winner = first_player
            self.loser = second_player
        else:
            self.winner = second_player
            self.loser = first_player
        
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
            sink = SHIP_DATA[ShipId(outcome["sunk_ship_id"])]["name"] + " sunk!"
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
        
# -----------------------------------------------------------------------------
class Player:
    """A Battleship player that has access to a board on which a game
    can be played. The player can be human or AI. In the case of human,
    this class provides an interface for selecting targets. In the case of
    an AI, the class has a 'strategy' that automatically selects targets.
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
        self.return_all_possibilities = False
        
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
        
    def place_fleet_randomly(self, method):
        """Places all 5 ships randomly on the board, with randomization method
        set by the 'method' input. This input can be 'random', 'cluster', 
        'isolated', or 'very.isolated'.
        """
        method = method.lower()
        for ship_id in range(1,len(SHIP_DATA)+1):
            ok = False
            while not ok:
                
                heading = self.board.random_heading()
                if method == "random":
                    # all placements are equally likely
                    index = self.board.random_index(unoccuppied=True)
                    ok = self.board.is_valid_ship_placement(ship_id, index, heading)
                    
                elif method == "cluster":
                    # closer to an existing ship is more likely
                    X,Y = np.meshgrid(range(self.board.size), range(self.board.size))
                    D = np.zeros(X.shape)
                    for ship in list(self.board.fleet.values()):
                        rows,cols = zip(*self.board.index_for_ship(ship))
                        for (r,c) in zip(rows,cols): 
                            D = D + (X - c)**2 + (Y - r)**2
                    indices = self.board.unoccuppied_indices()
                    prob = np.zeros(len(indices))
                    for (i,rc) in enumerate(indices):
                        prob[i] = 1 / (1 + D[rc[0],rc[1]])
                    prob = np.array(prob)
                    prob = prob / np.sum(prob)
                    index = indices[np.random.choice(range(len(indices)), 
                                                     size=1, 
                                                     p = prob)[0]]
                    ok = self.board.is_valid_ship_placement(ship_id, index, heading)
                    
                elif method == "isolated": 
                    # find a placement that is not touching another ship
                    sites = self.board.all_valid_placements(
                        SHIP_DATA[ShipId(ship_id)]["length"],
                        distance=1, diagonal=False)
                    index, heading = sites[np.random.choice(range(len(sites)))]
                    ok = True
                    
                elif method == "very.isolated":
                    # find a placement that is not touching another ship, 
                    # even on a diagonal
                    sites = self.board.all_valid_placements(
                        SHIP_DATA[ShipId(ship_id)]["length"],
                        distance=1, diagonal=True)
                    index, heading = sites[np.random.choice(range(len(sites)))]
                    ok = True
                    
                else:
                    raise ValueError(f"Invalid placement strategy: {method}.")
                #ok = self.board.is_valid_ship_placement(i, index, heading)
            self.board.place_ship(ship_id, index, heading)
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
            name = SHIP_DATA[ShipId(i)]["name"]
            while not ok:
                space = input("Space for {name.upper())}: ")
                heading = input("Heading for {name.upper())}: ")                
                if not self.board.is_valid_ship_placement(i, space, heading):
                    ok = False
                    print("Invalid placement.")
                else:
                    self.board.place_ship(i, space, heading)
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
            
    def still_alive(self): 
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
        # print(last_target)
        # print(last_outcome)
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
            if self.return_all_possibilities:
                return self.board.untargeted_
            return self.board.random_coord(untargeted=True)
        
        elif self.strategy["offense"] == "random.hunt.dumb":
            # If in kill mode, fire around the most recent hit.
            if self.strategy_data["state"] == "hunt":
                target = self.board.random_coord(untargeted=True)
                self.strategy_data["message"] = "hunt mode."
            else:
                last_hit = self.strategy_data["last_hit"]
                target = self.board.random_coord_around(last_hit, 
                                                        untargeted=True)
                self.strategy_data["message"] = "kill mode."
            if not target:
                self.strategy_data["state"] = "hunt"
                self.strategy_data["message"] += f" No viable target found " \
                    f"around last hit {last_hit}; reverting to hunt mode. "
                target = self.board.random_coord(untargeted=True)
            return target

        elif self.strategy["offense"] == "random.hunt":
            # If in kill mode, find two hits in a row. Then fire in a line
            # until a ship is sunk.
            if self.strategy_data["state"] == "hunt":
                target = self.board.random_coord(untargeted=True)
                self.strategy_data["message"] = "hunt mode."
            else:
                last_hit = self.strategy_data["last_hit"]
                hits = self.strategy_data["hits_since_last_sink"]
                self.strategy_data["message"] = "kill mode."
                if len(hits) == 1:
                    target = self.board.random_coord_around(hits[-1], 
                                                          untargeted=True)
                    self.strategy_data["message"] += " Searching around last hit."
                elif len(hits) > 1:
                    target = self.board.random_colinear_target_coords(hits)
                    self.strategy_data["message"] += " Searching along colinear hits."                    
                else:
                    raise Exception("hits_since_last_sink should not be empty.")
                if not target:
                    self.strategy_data["state"] = "hunt"
                    self.strategy_data["message"] += " No viable target found; " \
                        "reverting to hunt mode. "
                    target = self.board.random_coord(untargeted=True)
            return target
            
        else:
            s = self.strategy["offense"]
            raise ValueError(f"Invalid offensive strategy: {s}")
            
        
    def fire_at_target(self, target_coord): 
        """Fires a shot at the input target space and handles the resulting
        outcome. If the shot results in a hit, the opponent's ship is damaged.
        The Player's board is updated with a hit or miss value at the 
        target_space.
        Returns a Space enum value (MISS or HIT).
        """
        outcome = self.opponent.board.incoming_at_coord(target_coord)
        self.board.update_target_grid(self.board.index_for_coord(target_coord), 
                                     outcome["hit"], 
                                     outcome["sunk_ship_id"])
        self.shot_log += [target_coord]
        if target_coord in self.remaining_targets:
            self.remaining_targets.pop(self.remaining_targets.index(target_coord))
        self.outcome_log += [outcome]
        # print(target_coord)
        # print(outcome)
        self.update_state(target_coord, outcome)
        return outcome
    
    def show_attack(self):
        """Shows the board for the player, along with possible targets."""
        fig = self.board.show_board()
        axs = fig.axes
        ax = axs[0]
        target_circ = plt.Circle(self.board.index_for_coord(self.last_target()), 
                                 radius=0.3*1.4, 
                                 fill = False,
                                 edgecolor = "yellow") 
        ax.add_patch(target_circ)
        
       
# -----------------------------------------------------------------------------
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
                                   dtype=np.int8) * Space.UNKNOWN.value
        
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
            s += (self.row_label(r) 
                  + " "
                  + " ".join(["-" if x == Space.UNKNOWN else 
                              'O' if x == Space.MISS else
                              'X' if x == Space.HIT else 
                              str(x) for x in row]) 
                  + "\n")
        s += ("\n  " 
              + " ".join([str(x) for x in np.arange(self.size) + 1]) 
              + "\n")
        for (r,row) in enumerate(self.ocean_grid):
            s += (self.row_label(r) 
                  + " "
                  + " ".join(["-" if x == 0 else 
                              str(x) for x in row]) 
                  + "\n")
        return s
    
    # Indexing & coordinate functions
    def index_for_coord(self, coord):
        """Returns a tuple containing the row/column indices that correspond
        to the input coordinate.
        For example, index_for_coord('B4') returns (1,3).
        """
        max_letter = chr(LETTER_OFFSET + self.size - 1)
        m = re.search(f"[A-{max_letter}][1-9][0-9]?", coord)
        if m:
            row = ord(m.string[0].upper()) - LETTER_OFFSET
            col = int(m.string[1:]) - 1
        else:
            raise ValueError(f"Input coordinate {coord} must be a string consisting"
                             f" of a letter A-J followed by a number 1-10.")
        if row < 0 or row >= self.size or col < 0 or col >= self.size:
            raise ValueError(f"Input space '{coord}' did not "
                             f"produce row and column values between 1 and "
                             f"{self.size}.")
        return (row, col)
        
    def coord_for_index(self, index):
        """Returns a list of Space strings corresponding to the input row/column
        tuple."""
        if isinstance(index, list):
            return [self.coord_for_index(i) for i in index]
        
        row,col = index
        if ((row < 0) or (row >= self.size) or 
                (col < 0) or (col >= self.size)):
            raise ValueError(f"Rows and cols must be between 0 and " 
                             f"{self.size}.")
        return chr(row + LETTER_OFFSET) + str(col+1)
        
    def row_label(self, row):
        """Returns the label of the input row index. Rows 0-9 correspond to
        labels A-J (or however big the board is)."""
        if row < 0 or row >= self.size:
            raise ValueError(f"Input row must be between 0 and {self.size}.")
        return chr(LETTER_OFFSET + row)
    
    def all_indices(self, untargeted=False, unoccuppied=False):
        """Returns a list of all row/column index pairs on the board. If 
        untargeted is True, only indices that have not been targeted are 
        returned. If unoccuppied is True, only indices with no ships on the
        board's ocean grid are returned."""
        all_indices = [(r,c) for r in range(self.size) 
                       for c in range(self.size)]
        if untargeted and unoccuppied:
            raise ValueError("untargeted and unoccuppied cannot both be True.")
        if untargeted:
            exclude = list(zip(
                np.where(self.target_grid != Space.UNKNOWN.value)))
        elif unoccuppied:
            exclude = list(zip(np.where(self.ocean_grid != 0)))
        else:
            return all_indices
        return [i for i in all_indices if i not in exclude]
        
    def all_coords(self, untargeted=False, unoccuppied=False):
        """Returns a list of all coordinates on the board.
        If untargeted is True, only coords that have not been targeted are 
        returned. If unoccuppied is True, only coords with no ships on the
        board's ocean grid are returned."""
        if not untargeted and not unoccuppied:
            rows = [chr(LETTER_OFFSET + x) for x in range(self.size)]
            cols = [str(x+1) for x in range(self.size)]
            return [r + c for r in rows for c in cols]
        else:
            return [self.coord_for_index(i) for i in 
                    self.all_indices(untargeted, unoccuppied)]
    
    def unoccuppied_indices(self):
        """Returns the indicides of all coordinates that are not occuppied by
        a ship."""
        occ = self.coord_for_index(self.occuppied_indices())
        unocc = [c for c in self.all_coords() if c not in occ]
        return [self.index_for_coord(c) for c in unocc]

    def occuppied_indices(self):
        """Returns the indices of all ship-occuppied coordinates on the board 
        (regardless of damage/sunk status).
        """
        r,c = np.where(self.ocean_grid)
        return list(zip(r,c))
    
    def targeted_indices(self):
        """Returns the indices of all previously targeted indices on the target
        grid (i.e., the location of all pegs, white or red).
        """
        r,c = np.where(self.target_grid != Space.UNKNOWN.value)
        return list(zip(r,c))
    
    def relative_indices_for_heading(self, heading, length=1):
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
    
    def indices_around(self, index, diagonal=False):
        """Returns a list of all indices surrounding the input index. That is, 
        the row/column pairs for the spaces directly north, south, east, and
        west of the input index. 
        If the diagonal input is True, the spaces northwest, northeast, south-
        west, and southeast are also returned.
        """
        if diagonal:
            rows = np.array([-1, -1, 0, 1, 1, 1, 0, -1])
            cols = np.array([0, -1, -1, -1, 0, 1, 1, 1])
        else:
            rows = np.array([-1, 0, 1, 0])
            cols = np.array([0, -1, 0, 1])
        r = rows + index[0]
        c = cols + index[1]
        ivalid = (r >= 0) * (c >= 0) * (r < self.size) * (c < self.size)
        return list(zip(r[ivalid], c[ivalid]))
    
    def all_valid_placements(self, length, distance=0, diagonal=False):
        """Returns a list of all possible placements (consisting of row, 
        column, and heading) on the board. The distance parameter controls
        how close the placements can be to another existing ship.
        The returned list of placements is in the form of a list of tuples
        ((row, column), heading).
        """
        headings = {"N": (1,0), "S": (-1,0), "E": (0,-1), "W": (0,1)}
        unoccuppied = self.unoccuppied_indices()
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
                        surrounding = zip(*self.indices_around((r,c), diagonal))
                        if np.any(self.ocean_grid[tuple(surrounding)]):
                            ok = False
                if ok:
                    placements += [(unocc, h)]
        return placements
                
    # Ship access functions
    def afloat_ships(self):
        """Returns a list of the Ship instances that are still afloat."""
        return np.array([ship.afloat() for ship in list(self.fleet.values())])
    
    def index_for_ship(self, ship):
        """Returns the indices of the ocean grid that contain the input ship.
        """
        r,c = np.where(self.ocean_grid == ship.ship_id.value)
        return list(zip(r,c))
        
    def coord_for_ship(self, ship):
        """Returns the coordinates occuppied by the input Ship instance."""
        return [self.coord_for_index(i) for i in self.index_for_ship(ship)]
    
    def ship_at_index(self, index):
        """Returns the Ship object placed at the input coordinate.
        If no ship is present at the input space, returns None.
        """
        ship_id = self.ocean_grid[index]
        if ship_id == 0:
            return None
        return self.fleet[ShipId(ship_id)]
    
    # Ship placement functions
    def reset(self):
        """Removes all ships and pegs from the board."""
        board = Board(self.size)
        self.fleet = board.fleet
        self.target_grid = board.target_grid
        self.ocean_grid = board.ocean_grid
        
    def place_ship(self, ship_desc, index, heading):
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
        if not self.is_valid_ship_placement(ship_id, index, heading):
            print(f"Cannot place {ship.name} at {index}.")
            return False
        else:
            self.fleet[ShipId(ship_id)] = ship
            drows,dcols = self.relative_indices_for_heading(heading, ship.length)
            self.ocean_grid[index[0] + drows, index[1] + dcols] = ship_id
            return True
        
    def is_valid_ship_placement(self, ship_id, index, heading):
        """Returns True if the input ship can be placed at the input spot with
        the input heading, and False otherwise.
        The ship/space/heading is valid if the ship:
            1. Is not a duplicate of an existing ship
            2. Does not hang off the edge of the grid
            3. Does not overlap with an existing ship on the grid
        """
        length = SHIP_DATA[ShipId(ship_id)]["length"]
        if ShipId(ship_id) in self.fleet.keys():
            return False
        drows,dcols = self.relative_indices_for_heading(heading, length)
        rows,cols = index[0] + drows, index[1] + dcols
        if (np.any(rows < 0) or np.any(rows >= self.size) or 
            np.any(cols < 0) or np.any(cols >= self.size)):
                return False
        r,c = np.where(self.ocean_grid)
        occuppied = list(zip(r,c))
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
            if not np.all(self.target_grid == Space.UNKNOWN.value):
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
        """Returns a random coordinate string from the board.
        
        If the unoccuppied input is True, the coordinate will be one that does 
        not contain a ship on this board's ocean grid (note that if this is 
        used publically, this could reveal ship locations).
                               
        If the untargeted input is True, the coordinate will be one that has
        not been shot at from this board (i.e., no peg on the target grid).
        """
        return self.coord_for_index(self.random_index(unoccuppied, untargeted))
        
    def random_index(self, unoccuppied=False, untargeted=False):
        """Returns a random index tuple from the board.
        
        If the unoccuppied input is True, the index will be one that does 
        not contain a ship on this board's ocean grid (note that if this is 
        used publically, this could reveal ship locations).
                    
        If the untargeted input is True, the index will be one that has
        not been shot at from this board (i.e., no peg on the target grid).
        """
        if unoccuppied and untargeted:
            raise ValueError("Inputs unoccuppied and untargeted cannot both \
                             be true.")
        if unoccuppied:
            exclude = self.occuppied_indices()
        elif untargeted:
            exclude = self.targeted_indices()
        else:
            exclude = []
        indices = [(i,j) for i in range(self.size) 
                   for j in range(self.size) 
                   if (i,j) not in exclude]
        return indices[np.random.choice(range(len(indices)))]
    
    def random_heading(self):
        """Returns a random heading: N/S/E/W."""
        return(np.random.choice(['N','S','E','W']))
    
    def random_coord_around(self, coord, untargeted=False, diagonal=False,):
        """Returns a random coordinate that surrounds the input coordinate.
        By default, the random coordinate will be touching the input coordinate
        (so input C5 could result in one of B5, C4, C6, or D5). If
        diagonal is True, the 4 spaces that touch the input coordinate diagaonally
        (the ones NW, NE, SW, SE of the target) are also possible.
        If untargeted is True, then only untargeted coordinates will be chosen.
        """
        # unoccuppied = False
        # if unoccuppied:
        #     exclude = self.occuppied_indices()
        if untargeted:
            exclude = self.targeted_indices()
        else:
            exclude = []
        
        r,c = self.index_for_coord(coord)
        if diagonal:
            rows = r + np.array([0, -1, -1, -1, 0, 1, 1, 1])
            cols = c + np.array([1, 1, 0, -1, -1, -1, 0, 1])
        else:
            rows = r + np.array([0, -1, 0, 1])
            cols = c + np.array([1, 0, -1, 0])
        ivalid = ((rows >= 0) * (rows < self.size) * 
                  (cols >= 0) * (cols < self.size))
        rows = rows[ivalid]
        cols = cols[ivalid]
        indices = [(rows[i], cols[i]) for i in range(len(rows)) if 
                   (rows[i], cols[i]) not in exclude]
        if not indices:
            return []
        return self.coord_for_index(
            indices[np.random.choice(range(len(indices)))])
    
    def random_colinear_target_coords(self, coords):
        """Returns one of the two coordinates at the end of the line made
        by the input coordinates.
        """
        indices = [self.index_for_coord(c) for c in sorted(coords)]
        ds = (np.diff(np.array(indices), axis=0))
        ia,ib = indices[0],indices[-1]
        choices = []
        
        # If vertical, find the next miss or unknown N/S of the line
        if np.any(ds[:,0] != 0):
            
            while (ia[0] >= 0 and ia[0] < self.size and 
                   self.target_grid[ia] != Space.UNKNOWN.value):
                ia = (ia[0] - 1, ia[1])
            while (ib[0] >= 0 and ib[0] < self.size and 
                   self.target_grid[ib] != Space.UNKNOWN.value):
                ib = (ib[0] + 1, ib[1])
            if ia[0] >= 0:
                choices += [ia]
            if ib[0] < self.size:
                choices += [ib]
                
        # If horizontal, find the next miss or unknown E/W of the line
        elif np.any(ds[:,1] != 0):
            while (ia[1] >= 0 and ia[1] < self.size and 
                   self.target_grid[ia] != Space.UNKNOWN.value):
                ia = (ia[0], ia[1] - 1)
            while (ib[1] >= 0 and ib[1] < self.size and 
                   self.target_grid[ib] != Space.UNKNOWN.value):
                ib = (ib[0], ib[1] + 1)
            if ia[1] >= 0:
                choices += [ia]
            if ib[1] < self.size:
                choices += [ib]
                
        if not choices:
            return None
        return self.coord_for_index(
            choices[np.random.choice(range(len(choices)))])
    
    # Combat functions
    def update_target_grid(self, target_index, hit, ship_id):
        """Updates the target map at the input space with the hit informaton
        (hit and ship_id, if a ship was sunk) provided as inputs.
        Prints a message to the console if the target space has already been
        targeted.
        """
        #print(target_index, hit, ship_id)
        prev_val = self.target_grid[target_index]
        if hit:
            if ship_id:
                self.target_grid[target_index] = ship_id.value
            else:
                self.target_grid[target_index] = Space.HIT.value   
        else:
            self.target_grid[target_index] = Space.MISS.value
        if prev_val != Space.UNKNOWN.value:
            print(f"Repeat target: {self.coord_for_index(target_index)}")
            
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
        target_index = self.index_for_coord(coord)
        ship = self.ship_at_index(target_index)
        if not ship:
            outcome = {"hit": False, "sunk": False, "sunk_ship_id": None,
                       "message": []}
        else:
            outcome = {"hit": True, "sunk": False, "sunk_ship_id": None,
                       "message": []}
            ship_indices = self.index_for_ship(ship)
            dmg_slot = [i for i in range(len(ship_indices)) 
                      if ship_indices[i] == target_index]
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
            self.update_target_grid((0,5), True, ShipId(3))
            self.update_target_grid((2,8), False, None)
            self.update_target_grid((9,7), False, None)
            self.update_target_grid((6,7), False, None)
            self.update_target_grid((4,4), False, None)
            self.update_target_grid((3,2), False, None)
            self.update_target_grid((1,6), False, None)
            
            self.fleet[ShipId(2)].hit(0)
            self.fleet[ShipId(2)].hit(1)
            self.fleet[ShipId(2)].hit(2)
            self.fleet[ShipId(4)].hit(3)
            self.fleet[ShipId(5)].hit(2)
        
    # Visualization functions
    def ocean_grid_image(self):
        """Returns a matrix corresponding to the board's ocean grid, with
        blank spaces as 0, unhit ship slots as 1, and hit slots as 2.
        """
        im = np.zeros((self.size, self.size))
        for ship_id in self.fleet:
            ship = self.fleet[ShipId(ship_id)]
            dmg = ship.damage
            rows,cols = zip(*self.index_for_ship(ship))
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
            rows, cols = zip(*self.index_for_ship(ship))
            rects[ship.ship_id] = np.array((np.min(cols)+0.5, 
                                            np.min(rows)+0.5,
                                            np.max(cols) - np.min(cols) + 1, 
                                            np.max(rows) - np.min(rows) + 1))
        return rects
        
    def show_board(self):
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
                               color = hit_color, 
                               edgecolor = peg_outline_color) 
                    for (x,y) in zip(cols+1,rows+1)]
        for peg in red_pegs:
            axs[1].add_patch(peg)
            
        # vertical grid
        rows,cols = np.where(vert_grid == Space.MISS.value)
        white_pegs = [plt.Circle((x,y), radius=peg_radius, 
                                 color = miss_color, 
                                 edgecolor = peg_outline_color) 
                    for (x,y) in zip(cols+1,rows+1)]
        for peg in white_pegs:
            axs[0].add_patch(peg) 
        rows,cols = np.where(vert_grid >= Space.HIT.value)
        red_pegs = [plt.Circle((x,y), radius=peg_radius, 
                               color = hit_color, 
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
            
        if isinstance(ship_type, ShipId):
            self._ship_id = ship_type
        elif isinstance(ship_type, int):
            try:
                self._ship_id = ShipId(ship_type)
            except:
                raise ValueError(f"For integer input, ship_type must be between" \
                                 f" 1 and {len(SHIP_DATA.keys())}.")
        elif isinstance(ship_type, str):
            self._ship_id = [k for k in SHIP_DATA if 
                            SHIP_DATA[k]["name"].lower() == ship_type.lower()]
            if not self.ship_id:
                ValueError("For string input, ship_type must be one of: " 
                           + ", ".join([SHIP_DATA[k]["name"] for k in SHIP_DATA])) 
            else:
                self._ship_id = self.ship_id[0]
        else:
            raise TypeError("Input ship_type must be a string, integer 1-5, "
                            "or ShipId Enum; not " + type(ship_type))
            
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
        if not isinstance(slot, int):
            raise TypeError("slot must be an integer between 0 and the ship" 
                            "length (exclusive).")
        if slot < 0 or slot >= self.length:
            raise ValueError("Hit at slot #" + str(slot) + " is not valid; "
                             "must be 0 to " + str(self.length-1))
        if self.damage[slot] > 0:
            Warning("Splot " + str(slot) + " is already damaged.")
        self.damage[slot] += 1
        return(self.damage[slot])
    
    def afloat(self):
        return(any(dmg < 1 for dmg in self.damage))

    def sunk(self):
        return(all(dmg > 0 for dmg in self.damage))
