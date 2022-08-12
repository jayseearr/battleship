#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 09:23:07 2022

@author: jason
"""

"""
Definitions for the Battleship module:
Board               The parts of the game corresponding to one player.
                    This includes the flat grid where a player's ships
                    are placed, the ships themselves, and the vertical
                    grid where shots are tracked.
Shot                A player's attempt to hit the other player's ships.
                    Each shot consists of a targeted space and an
                    outcome.
Target              The space onto which a shot was fired. 
Outcome             The result of a shot. If a ship was hit but not sunk,
                    the outcome is HIT. If a ship was hit and sunk,
                    the outcome is HIT, SUNK, and contains a ship ID
                    indicating which ship was sunk. If a ship was not
                    hit, the outcome is a MISS.
Damage              An array for each ship indicating whether each
                    slot on the ship has been damaged or not. The array
                    has one element per slot on the ship, and is equal
                    to 1 if that slot has been hit, and 0 otherwise.
                    If all slots are damaged, the ship is considered sunk.
Grid                The horizontal part of the board where a player 
                    places their own ships and keeps track of ship
                    damage. The grid consists of 10 x 10 spaces. Rows
                    are labeled by letters A-J, and columns are labeled
                    by numbers 1-10. The upper left is space A1, and 
                    lower right os space J10.
Space               A single position on the grid where a shot can be 
                    fired. Ships are placed over multiple contiguous spaces.
                    A Space is identified by a letter/number string (e.g., C2)
                    that indicate row and column, respectively. Internally,
                    Spaces are identified by row and column indices ranging
                    from 0-9.
(Ship) Slot         The positions on a ship that can be hit and damaged.
                    Different types of ships can have 2, 3, 4, or 5 slots.
(Ship) Position     The grid location (row/column) of the front of 
                    a ship. The rest of the ship will be aft of its
                    position, with direction determined by its heading.
Heading             The cardinal direction that a ship is facing 
                    (N/S/E/W).
Target Grid         The vertical part of the board where a player keeps
                    track of which spaces they have fired shots at, and
                    what the outcome of each shot was (hit or miss).
"""

import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import re
from enum import Enum

# Constants
BOARD_SIZE = 10
LETTER_OFFSET = 65

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
    
class ShipId(Enum):
    PATROL = 1
    DESTROYER = 2
    SUBMARINE = 3
    BATTLESHIP = 4
    CARRIER = 5
    
### Data
shipData = {ShipId.PATROL: {"name": "Patrol", "spaces": 2},
            ShipId.DESTROYER: {"name": "Destroyer", "spaces": 3},
            ShipId.SUBMARINE: {"name": "Submarine", "spaces": 3},
            ShipId.BATTLESHIP: {"name": "Battleship", "spaces": 4},
            ShipId.CARRIER: {"name": "Carrier", "spaces": 5}
            }

### Utility functions
def indices_for_space(space):
    """Returns a tuple containing the row/column indices that correspond
    to the input space.
    For example, indices_for_space('B4') returns (1,3).
    """
    
    m = re.search("[A-J][1-9]0?", space)
    if m:
        row = ord(m.string[0].upper()) - LETTER_OFFSET
        col = int(m.string[1:]) - 1
    else:
        raise ValueError("Input space '" + space + "' must be a string consisting"
                         " of a letter A-J followed by a number 1-10.")
    if row < 0 or row >= BOARD_SIZE or col < 0 or col >= BOARD_SIZE:
        raise ValueError("Input space '" + space + "' did not "
                         "produce row and column values between 1 and 10.")
    return (row-1, col-1)

def spaces_for_indices(rows,cols):
    """Returns a list of Space strings corresponding to the input row/column
    indices."""
    if np.issubdtype(type(rows), np.integer):
        if ((rows < 0) or (rows >= BOARD_SIZE) or 
                (cols < 0) or (cols >= BOARD_SIZE)):
            raise ValueError("Rows and cols must be between 0 and " 
                             + str(BOARD_SIZE-1) + ".")
        return chr(rows + LETTER_OFFSET) + str(cols+1)
    else:
        return [spaces_for_indices(r,c) for (r,c) in zip(rows,cols)]
    
# def is_valid_row_col(row,col):
#     """Returns TRUE if the input row and column correspond to a valid
#     space on the board, and False otherwise."""
#     if isinstance(row, int) and isinstance(col, int):
#         return row >= 0 and row < BOARD_SIZE and col >= 0 and col < BOARD_SIZE
#     else:
#         return False

###############################################################################
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
            self.ship_id = ship_type
        elif isinstance(ship_type, int):
            try:
                self.ship_id = ShipId(ship_type)
            except:
                raise ValueError("For integer input, ship_type must be between"
                                 " 1 and " + str(len(shipData)))
        elif isinstance(ship_type, str):
            self.ship_id = [k for k in shipData if 
                            shipData[k]["name"].lower() == ship_type.lower()]
            if not self.ship_id:
                ValueError("For string input, ship_type must be one of: " 
                           + ", ".join([shipData[k]["name"] for k in shipData])) 
            else:
                self.ship_id = self.ship_id[0]
        else:
            raise TypeError("Input ship_type must be a string, integer 1-5, "
                            "or ShipId Enum; not " + type(ship_type))
            
        self.name = shipData[self.ship_id]["name"]
        self.length = shipData[self.ship_id]["spaces"]
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
        s = (self.name + ", " + str(self.length) + 
             " slots, damage: " + str(self.damage))
        if self.sunk():
            s += " (SUNK)"
        return(s)
    
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
    
###############################################################################
class Board:
    """An instance of a Battleship game board. Each board has a fleet of 5 
    ships, a 10 x 10 grid upon which ships are arranged, and a 10 x 10 target
    map where hit/miss indicators are placed to keep track of opponent ships.
    """
    def __init__(self):
        self.fleet = {}
        self.grid = np.zeros((BOARD_SIZE, BOARD_SIZE), 
                             dtype=np.int16)      
        self.target_map = np.ones((BOARD_SIZE, BOARD_SIZE), 
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
        s = "\n" + "  "
        s += (" ".join([str(x) for x in np.arange(BOARD_SIZE) + 1]) 
              + "\n")
        for (r,row) in enumerate(self.target_map):
            s += (self.row_label(r) 
                  + " "
                  + " ".join(["-" if x == Space.UNKNOWN else 
                              'O' if x == Space.MISS else
                              'X' if x == Space.HIT else '?' for x in row]) 
                  + "\n")
        s += ("\n  " 
              + " ".join([str(x) for x in np.arange(BOARD_SIZE) + 1]) 
              + "\n")
        for (r,row) in enumerate(self.grid):
            s += (self.row_label(r) 
                  + " "
                  + " ".join(["-" if x == 0 else 
                              str(x) for x in row]) 
                  + "\n")
        return(s)
    
    # Internal indexing/labeling functions
    def row_label(self, row):
        """Returns the label of the input row index. Rows 0-9 correspond to
        labels A-J."""
        if not isinstance(row,int):
            raise TypeError("Input row must be an integer")
        if row < 0 or row >= BOARD_SIZE:
            raise ValueError("Input row must be between 0 and " + str(BOARD_SIZE))
        return chr(LETTER_OFFSET + row)
    
    # Fleet setup functions
    def is_valid_ship_placement(self, ship, space, heading):
        """Returns True if the input ship can be placed at the input spot with
        the input heading, and False otherwise.
        The ship/space/heading is valid if the ship:
            1. Is not a duplicate of an existing ship
            2. Does not hang off the edge of the grid
            3. Does not overlap with an existing ship on the grid
        """
    
    def is_fleet_placement_valid(self):
        """Returns True if the board contains 5 distince ships in valid 
        spots on the board. The ships cannot overlap, hang over the edge of
        the grid, and exactly one of each of the 5 ship types must be placed.
        If that is the case, True is returned (False otherwise).
        """
        
    def place_fleet(self, method="random"):
        """Places the 5 ships automatically using a method set by the 'method'
        input. The default method is 'random', where the ship positions and
        headings are chosen randomly.
        """
        
    def place_ship(self, ship_type, space, heading):
        """Places a ship corresponding to the input ship_type (which may be an
        integer 1-5, a ShipId enum, or a string like "Patrol", "Carrier", etc.)
        on the grid at the input space (which may also be a tuple or list of
        row/column indices) and the input heading (a string, "N", "S", "E", or
        "W"). The ship is added to the board's fleet.
        Returns True if the placement was successful (i.e., if the input 
        placement is valid) and False otherwise, in which case the ship is
        NOT added to the fleet.
        """
    
    def ready_to_play(self):
        """Returns True if the full fleet is placed on the grid, all ships are
        undamaged, and the target map is empty. Returns False otherwise (which
        indicates that either setup is not complete, or the game has already
        started).
        """
    
    # Fleet accessing functions       
    def fleet_list(self):
        """Returns a list of the ship objects in the fleet."""
        return list(self.ships.values())
        
    def ship_at_space(self, space):
        """Returns the Ship object placed at the input space, which may be
        either a space string ("D2") or a tuple of row/column indices.
        If no ship is present at the input space, returns None.
        """
        if isinstance(space, str):
            row, col = indices_for_space(space)
        else:
            row, col = space
        ship_id = self.grid[row, col]
        return self.ships[ShipId(ship_id)]
    
    def afloat_ships(self):
        """Returns a list of the Ship instances that are still afloat."""
        return np.array([ship.is_afloat() for ship in list(self.ships.values())])
    
    def sunk_ships(self):
        """Returns a list of the Ship instances that are sunk."""
        return np.array([ship.is_sunk() for ship in list(self.ships.values())])    
    
    def ship_for_id(self, ship_id):
        """Returns the ship object in the board's fleet that has the input
        id number (the input may be an integer 1-5 or a ShipId enum).
        """
        if isinstance(ship_id, int):
            ship_id = ShipId(ship_id)
        return self.ships[ship_id]
        
    # Space finding functions
    def encode_as_indices(self, space_desc):
        """Ensures that the input space description is in row/column format.
        For an input letter/number string or row/column tuple, a row/column
        tuple is returned.
        If a list of spaces is provided, the returned rows and columns will be
        arrays.
        """
        if isinstance(space_desc, tuple):
            return space_desc
        if isinstance(space_desc, str):
            return indices_for_space(space_desc)
        elif isinstance(space_desc, list) and isinstance(space_desc[0], str):
            rows, cols = zip(*[indices_for_space(sp) for sp in space_desc])
            return (np.array(rows), np.array(cols))
        
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
        if heading.upper() in ["NORTH", "SOUTH", "EAST", "WEST"]:
            heading = heading[0].upper()
            
        if heading == "N":
            ds = (-1,0)
        elif heading == "S":
            ds = (1,0)
        elif heading == "E":
            ds = (0,-1)
        elif heading == "W":
            ds = (0,1)
        else:
            raise ValueError(("heading input must be North, South, East, \
                              or West (first letter is okay, and everything \
                              is case insensitive."))
        if length == 1:
            return ds
        else:
            return np.vstack(([0,0], 
                              np.cumsum(np.tile(ds, (length-1,1)), axis=0)))
        
    def indices_for_ship_placement(self, space, heading, length):
        """Returns arrays of the rows and columns occuppied by the input ship
        placement info (i.e., the ship's starting space, its heading, and its
        length). 
        Returns None for both rows and cols arrays if the placement is invalid.
        """
        if isinstance(space, str):
            row, col = indices_for_space(space)
        else:
            row, col = space
        ds = self.relative_indices_for_heading(heading, length)
        return row + ds[:,0], cols + ds[:,1]
        
    def spaces_around_space(self, space):
        """Returns a list all spaces adjacent to the input space. The elements
        of the list are in letter/number string format.
        """
        if isinstance(space, str):
            row, col = indices_for_space(space)
        else:
            row, col = space
        rows = row + np.array((1, 0, -1, 0))
        cols = col + np.array((0, -1, 0, 1))
        ivalid = ((rows > 0) * (rows < BOARD_SIZE) * 
                  (cols > 0) * (cols < BOARD_SIZE))
        return spaces_for_indices(rows[ivalid], cols[ivalid])
    
    def indices_for_ship(self, ship_desc):
        """Returns row and column arrays with the indices that contain the 
        input ship description. The description may be any of the following:
            A Ship instance
            An integer 1-5
            A ShipId Enum
            A string with the name of a ship ("Patrol", "Battleship", etc.)
        If a matching ship is not on the board, returns None.
        """
        if isinstance(ship_desc, int):
            if ship_desc < 1:
                ValueError(("Idenfifying a ship by its integer ID value \
                           requires a value between 1 and " 
                           + str(len(self.ships)) + "."))
            ship_id = ship_desc
        elif isinstance(ship_desc, str):
            ids = [i for (i,s) in enumerate(shipData.values()) 
                       if s["name"] == ship_desc]
            if len(ids) == 0:
                raise ValueError(f"Invalid ship name: {ship_desc}")
            ship_id = ids[0]
        elif isinstance(ship_desc, ShipId):
            ship_id = ship_desc.value
        return np.where(self.grid == ship_id)
        
    def hit_or_miss_at_target_space(self, space):
        """Returns a Space enum corresponding the the input space.
        If the space has not been targets, Space.UNKNOWN is returned.
        If a ship was present at the space, Space.HIT is returned.
        If no ship was present, Space.MISS is returned.
        Space can be either a letter/digit string ("C5") or a tuple
        of row/column indices.
        """
        if isinstance(space, str):
            row, col = indices_for_space(space)
        else:
            row, col = space
        return Space(self.target_map[row, col])
    
    def all_spaces(self, as_indices=False):
        """Returns a list of all of the spaces on the board. Defaults to 
        letter/number format, but as_indices can be set to true to provide
        a two-array tuple as output. 
        """
        rows = [chr(LETTER_OFFSET + x) for x in range(BOARD_SIZE)]
        cols = [str(x+1) for x in range(BOARD_SIZE)]
        spaces = [r + c for r in rows for c in cols]
        if as_indices:
            return zip(*[indices_for_space(sp) for sp in spaces])
        else:
            return spaces
        
    def occuppied_spaces(self, as_indices=False):
        """Returns the indices of all ship-occuppied spaces on the board 
        (regardless of damage/sunk status).
        """
        rows, cols = np.where(self.grid)
        if as_indices:
            return (rows, cols)
        else:
            return spaces_for_indices(rows, cols)
        
    # Gameplay (combat) related functions
    
    def damage_by_ship(self):
        """Returns an array with the damage on each of the 5 ships, ordered
        by increasing ShipId. Note that each ship can sustain a different 
        number of damage before sinking.
        """
        return np.array([self.ships[ShipId(i)] 
                         for i in range(1,len(self.ships)+1)])
        
    def random_space(self, unoccuppied=False):
        """Returns a string for a random space from the grid (e.g., D2).
        If the unoccuppied input is True, the space will not have a ship on it.
        See also random_target and random_indices.
        """
        if not unoccuppied:
            return spaces_for_indices(
                (np.random.randint(1, BOARD_SIZE + 1), 
                 np.random.randint(1, BOARD_SIZE + 1)))
        else:
            unocc = [space for space in self.all_spaces() 
                     if space not in self.occuppied_spaces()]
            return np.random.choice(unocc)
        
    def random_target(self, untargeted=True):
        """Returns a untargeted space from the target map.
        See also random_space and random_indices.
        """
        targeted_spaces = spaces_for_indices(np.where(self.target_map()))
        untargeted = [space for space in self.all_spaces() 
                 if space not in targeted_spaces]
        return np.random.choice(untargeted)
    
    def random_indices(self):
        """Returns a random row and column from the grid. Analagous to 
        random_space, but useful when row/column indices are needed instead
        of a space string.
        See also random_space and random_target.
        """
        return (np.random.randint(1, BOARD_SIZE + 1), 
                np.random.randint(1, BOARD_SIZE + 1))
        
    def random_heading(self):
        """Returns one of four heading directions (N/S/E/W) randomly.
        """
        return np.random.choice(["N", "S", "E", "W"])
        
    def incoming_at_space(self, space):
        """Determines the outcome of an opponent's shot landing at the input
        space. If there is a ship at that location, the outcome is a hit and
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
        
    def update_target_map(self, target_space, hit, ship_id):
        """Updates the target map at the input space with the hit informaton
        (hit and ship_id, if a ship was sunk) provided as inputs.
        Prints a message to the console if the target space has already been
        targeted.
        """
        row, col = indices_for_space(target_space)
        prev_val = self.target_map[row,col]
        if hit:
            if ship_id:
                self.target_map[row,col] = ship_id
            else:
                self.target_map[row,col] = Space.HIT.value   
        else:
            self.target_map[row,col] = Space.MISS.value
        if prev_val is not Space.UNKNOWN.value:
            print("Repeat target: " + str(target_space))
            
    
    # Visualization functions
    def grid_image(self):
        """Returns a matrix that can be used to show the grid."""
        
    def target_map_image(self):
        """Returns a matrix that can be used to show the target map."""
        
    def ship_rects(self):
        """Returns a dictionary with keys equal to ShipId and values equal to
        the rectangles that bound each ship on the grid. The rectangles have
        format [x, y, width, height].
        """
        
    def show_board(self):
        """Shows images of the target map and grid on a figure with two 
        subaxes. Colors the images according to hit/miss (for target map) and
        damage (for the grid). Returns the figure containing the images.
        """

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
            
        self.strategy_data = {}
        self.strategy = {'placement': None, 
                         'offense': None}
        self.shot_log = []
        self.outcome_log = []
        self.board = Board()
        self.opponent = None
        self.remaining_targets = self.board.all_spaces()
        self.game_history = None    # for future use
        self.notes = None           # for future use
        
    def set_up_fleet(self):
        """Place ships in the fleet either manually via text input (for Human
        player) or according to the placement strategy.
        """
        if self.type == "Human":
            self.place_fleet_manually()
        else:
            if self.strategy["placement"] == "random":
                for i in range(1,len(shipData)+1):
                    ok = False
                    while not ok:
                        space = self.board.random_space(unoccuppied=True)
                        heading = self.board.random_heading()
                        ok = self.board.is_valid_ship_placement(i, space, heading)
                    self.board.place_ship(i, space, heading)
                            
            elif self.strategy["placement"] == "cluster": 
                for i in range(1,len(shipData)+1):
                    ok = False
                    while not ok:
                        heading = self.board.random_heading()
                        # find a placement that is close to another ship
                        ok = self.board.is_valid_ship_placement(i, space, heading)
                    self.board.place_ship(i, space, heading)
                    
            elif self.strategy["placement"] == "isolated":                
                for i in range(1,len(shipData)+1):
                    ok = False
                    while not ok:
                        heading = self.board.random_heading()
                        # find a placement that not touching another ship
                        ok = self.board.is_valid_ship_placement(i, space, heading)
                    self.board.place_ship(i, space, heading)
                    
            elif self.strategy["placement"] == "very.isolated": 
                for i in range(1,len(shipData)+1):
                    ok = False
                    while not ok:
                        heading = self.board.random_heading()
                        # find a placement that not touching another ship,
                        # even on a diagonal.
                        ok = self.board.is_valid_ship_placement(i, space, heading)
                    self.board.place_ship(i, space, heading)
                
                
            else:
                raise ValueError("Invalid placement strategy value: " + self.strategy["placement"])
            
            
    def place_fleet_manually(self):
        for i in range(1,len(shipData)+1):
            ok = False
            while not ok:
                space = input("Space for " 
                              + shipData[ShipId(i)]["name"] + ": ").upper()
                heading = input("Heading for " 
                                + shipData[ShipId(i)]["name"] + ": ")
                if not self.board.is_valid_ship_placement(i, space, heading):
                    ok = False
                    print("Invalid placement.")
                else:
                    self.board.place_ship(i, space, heading)
                    print(shipData[ShipId(i)] + " placed successfully (" 
                          + str(i) + " of " + str(len(shipData)) + ").")
        print("Fleet placed successfully.")
                                                   
    def opponents_board(self):
        """Returns the opponent's Board."""
        return self.opponent.board
        
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
        
    def update_state(self, last_target, last_outcome):
        """Updates strategy_data based on the last targeted space and the
        resulting outcome.
        """
        
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
            
    def pick_target(self):
        """Returns a target space based on the player's strategy and shot/
        outcome history.
        This is where the strategy is implemented.
        """
        
    def fire_at_target(self, target_space):
        """Fires a shot at the input target space and handles the resulting
        outcome. If the shot results in a hit, the opponent's ship is damaged.
        The Player's board is updated with a hit or miss value at the 
        target_space.
        Returns a Space enum value (MISS or HIT).
        """
        outcome = self.opponent.board.incoming_at_space(target_space)
        self.board.update_target_map(target_space, 
                                     outcome["hit"], 
                                     outcome["sunk_ship_id"])
        self.shot_log += [target_space]
        self.remaining_targets.pop(self.remaining_targets.index(target_space))
        self.outcome_log += [outcome]
        self.update_state(target_space, outcome)
        return outcome
        
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
            
        An optional ID string can be added to identify the game.
        The verbose input controls whether the outcome of the game is printed
        to the console or not. This defaults to True, but can be turned off
        if many robot vs. robot games are going to be simulated.
        
        The game should not care whether a player is human or robot (so don't
        have it access self.player1.type).
        """
        
        self.game_id = game_id
        self.player1 = Player(player1)
        self.player2 = Player(player2)
        
        self.winner = None
        self.loser = None
        self.turn_count = 0
        self.verbose = verbose
        
    def prep_game(self):
        """Set up a game by placing both fleets."""
        self.player1.set_opponent(self.player2)
        self.player2.set_opponent(self.player1)
        self.player1.set_up_fleet()
        self.player2.set_up_fleet()
            
    def play(self, first_move=1):
        """Play one game of Battleship. Each player takes a turn until one
        of them has no ships remaining.
        """
        
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
            outcome1 = first_player.take_turn()
            if self.verbose:
                self.report_turn_outcome(outcome1)
            outcome2 = second_player.take_turn()
            if self.verbose: 
                self.report_turn_outcome(outcome2)
            self.turn_count += 1
            game_on = first_player.still_alive() and second_player.still_alive()
            
        # See who won
        if second_player.still_alive():
            self.winner = first_player
            self.loser = second_player
        else:
            self.winner = second_player
            self.loser = first_player
        
        if self.verbose:
            self.print_outcome()
            
    def report_turn_outcome(self, player):
        """Displays text reporting the target and outcome on the most recent
        turn for the input player.
        """
        outcome = player.last_outcome()
        target = player.last_target()
        name = player.name
        if outcome["hit"]:
            hit_or_miss = "Hit!"
        else:
            hit_or_miss = "Miss."
        if outcome["sunk_ship_id"]:
            sink = shipData[ShipId(outcome["sunk_ship_id"]["name"])] + " sunk!"
        print(f"Player {name} fired a shot at {target}...{hit_or_miss} {sink}")
        
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
    
        
