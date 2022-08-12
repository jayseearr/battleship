import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import re


BOARD_SIZE = 10

kHit = 0
kMiss = -1
kUnknown = -2
    
shipInfo = {1: {"name": "Patrol", "spaces": 2},
            2: {"name": "Destroyer", "spaces": 3},
            3: {"name": "Submarine", "spaces": 3},
            4: {"name": "Battleship", "spaces": 4},
            5: {"name": "Carrier", "spaces": 5}
            }

def simulate(ngames, nreps=1, strategy1="random kill", strategy2="random"):
    if nreps > 1:
        P1,P2 = [],[]
        for rep in range(nreps):
            p1,p2 = simulate(ngames, 1, strategy1, strategy2)
            P1 += [p1]
            P2 += [p2]
        P1 = np.array(P1)
        P2 = np.array(P2)
        return (P1, P2)
            
    winner = []
    for n in range(ngames):
        game = BSGame("robot","robot",strategy1,strategy2); 
        game.play(); 
        winner += [game.winner.name]
    p1 = sum(np.array(winner) == 'player1')
    p2 = sum(np.array(winner) == 'player2')
    print("Player 1: " + str(p1/(p1+p2)*100) + "%")
    print("Player 2: " + str(p2/(p1+p2)*100) + "%")    
    return (p1,p2)

def rowColForSpot(position):
    m = re.search("[A-J][1-9]0?", position)
    if m:
        row = ord(m.string[0].upper()) - 64
        col = int(m.string[1:])
    else:
        ValueError("Input position '" + position + "' must be a letter A-J followed by a number 1-10.")
    if row < 1 or row > 10 or col < 1 or col > 10:
        ValueError("Input position '" + position + "' did not produce row and column values between 1 and 10.")
    return (row, col)

def spotsForRowCol(rows,cols):
    if np.issubdtype(type(rows), np.integer):
        if (rows < 1) or (rows > BOARD_SIZE) or (cols < 1) or (cols > BOARD_SIZE):
            ValueError("Rows and cols must be between 1 and " + str(BOARD_SIZE) + ".")
        return chr(rows + 64) + str(cols)
    else:
        return [spotsForRowCol(r,c) for (r,c) in zip(rows,cols)]

class BSShip:
    """A ship object"""

    def __init__(self, ship_type):
        if np.issubdtype(type(ship_type), np.integer):
            if ship_type < 1 or ship_type > 5:
                ValueError("Integer input must be 1-5 ('" + str(ship_type) + "' was input).")
            shipId = ship_type
        elif type(ship_type) == str:
            shipId = [k for k in shipInfo if shipInfo[k]["name"].lower() == ship_type.lower()]
            if len(shipId) == 1:
                shipId = shipId[0]
            elif len(shipId) > 1:
                ValueError("Multiple ship matches found for string input '" + ship_type + "'.")
            else:
                ValueError("ship_type input must be one of: " + ", ".join([shipInfo[k]["name"] for k in shipInfo]))                
        self.shipId = shipId
        self.shipName = shipInfo[shipId]["name"]
        self.length = shipInfo[shipId]["spaces"]
        self.damage = np.zeros(self.length)

    def __str__(self):
        s = self.shipName + ", " + str(self.length) + " spaces, damage: " + str(self.damage)
        if self.sunk():
            s += " (SUNK)"
        return(s)

    def __len__(self):
        return self.length
    
    def hit(self, position):
        if position < 1 or position > self.length:
            ValueError("Hit at position " + str(position) + " is not valid; must be 1 to " + str(self.length))
        if self.damage[position - 1] > 0:
            Warning("Position " + str(position) + " is already damaged.")
        self.damage[position - 1] += 1
        return(self.damage[position - 1])

    def damage_at_position(self, position):
        if position < 1 or position > self.length:
            ValueError("Ship position " + str(position) + " is not valid; must be 1 to " + str(self.length))
        return self.damage[position - 1]
    def afloat(self):
        return(any(dmg < 1 for dmg in self.damage))

    def sunk(self):
        return(not self.afloat())
    
class BSBoard:
    """A 10 x 10 board for playing Battleship.
        The 'grid' variable is a 10 x 10 matrix with values as follows:
        0    No ship
        1-5  Ship with ID # 1-5.
        The 'targetGrid' variable is a 10 x 10 matrix with 0 where no shot has fired, -1 where
        there is a miss, and 1 where there is a hit.
        """

    def __init__(self):

        self.ships = {}
        self.grid = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int16)      # 0 for no ship, 1-5 for ship at that location."
        self.targetGrid = kUnknown * np.ones((BOARD_SIZE, BOARD_SIZE), dtype=np.int16)

    def __str__(self):
        s = ""
        for row in self.grid:
            s += " ".join(["-" if x == 0 else str(x)[0] for x in row]) + "\n"
        return(s)

    def valid_ship_placement(self, newship, spot, facing):
        """Returns True if the proposed spot is valid (i.e., does not overlap with another ship,
            is not a duplicate of another ship, and does not go over the edge of the board."""

        #print("Checking ship placement: newship = " + str(newship) + ", spot = " + str(spot) + ", facing = " + facing)
        if type(newship) == int:
            shipId = newship
            length = shipInfo[shipId]["spaces"]
        else:
            shipId = newship.shipId
            length = newship.length
        rows, cols = self.rows_cols_for_coords(spot, facing, length)
        ok = True
        # does ship already exist?
        shipIds = list(self.ships.keys())
        if shipId in shipIds:
            ok = False
        # does ship overlap with another ship?
        rows = np.array(rows)
        cols = np.array(cols)
        valids = (rows > 0) & (rows <= BOARD_SIZE) & (cols > 0) & (cols <= BOARD_SIZE)
        idx = [i for (i,_) in enumerate(valids) if _ == True]
        rows = rows[idx] - 1
        cols = cols[idx] - 1
        newspots = self.grid[rows, cols]
        if any(newspots) > 0:
            ok = False
        # is the ship over the edge of the board?
        if any(rows < 1) or any(rows > BOARD_SIZE) or any(cols < 1) or any(cols > BOARD_SIZE):
            ok = False
        return ok

    def rows_cols_for_coords(self, spot, facing, length):
        if not type(facing) == str:
            TypeError("The input 'facing' must be a string ('N', 'S', 'E', or 'W').")

        facing = facing.upper()
        if facing == "N":
            increment = (1, 0)
        elif facing == "E":
            increment = (0, -1)
        elif facing == "S":
            increment = (-1, 0)
        elif facing == "W":
            increment = (0, 1)
        else:
            ValueError("Facing must be 'N', 'S', 'E', or 'W' ('" + facing + "' was input).")
        (r0,c0) = rowColForSpot(spot)
        rows = r0 + increment[0] * np.arange(length)
        cols = c0 + increment[1] * np.arange(length)
        return (rows, cols)
        
    def random_spot(self):
        return spotsForRowCol(np.random.randint(low=1, high=BOARD_SIZE, size=1), \
                              np.random.randint(low=1, high=BOARD_SIZE, size=1))[0]
    
    def place_fleet(self, method = "random"):
        facings = ["N", "S", "E", "W"]
        if method == "random":
            for shipId in np.random.permutation(range(1,6)):
                count = 0
                while not self.place_ship(shipId, self.random_spot(), facings[np.random.randint(4)], verbose=False) and count < 1000:
                    count += 1
                if count >= 1000:
                    ValueError("Could not find a valid spot after 1000 attempts. Aborting.")
            print("All ships placed randomly.")
                
        else:
            ValueError("Only the 'random' method is currently supported.")
            
    def place_ship(self, shipId, spot, facing, verbose=True):
        newship = BSShip(shipId)
        rows, cols = self.rows_cols_for_coords(spot, facing, newship.length)
        
        # check that ship does not already exist on grid
        if newship.shipId in self.ships.keys():
            ValueError("Ship with type " + newship.shipName + " is already on the board.")

        # determine location
        
        # check if any part of the ship is off board
        if any(rows < 1) or any(rows > BOARD_SIZE) or any(cols < 1) or any(cols > BOARD_SIZE):
            if verbose:
                print("Ship placement at spot = " + spot + ", facing = " + facing + \
                        " is over the edge of the board. Occupied rows/cols = " + ", ".join([str(s) for s in zip(rows,cols)]))
            return False

        # check if any part of the ship overlaps an existing ship
        spots = self.grid[rows-1,cols-1]
        if any(spots != 0):
            if verbose:
                print("A ship already exists on " + str(spotsForRowCol(rows, cols)) + ": ")
            occRows = rows[spots != 0]
            occCols = cols[spots != 0]
            for (r,c) in zip(occRows, occCols):
                occid = self.grid[r-1,c-1]
                if verbose:
                    print("  " + spotsForRowCol(r,c)[0] + ": " + shipInfo[occid]["name"] + " (id = " + str(occid) + ").")
            return False
            
        self.grid[rows-1,cols-1] = newship.shipId
        self.ships[newship.shipId] = newship
        return True
            
    def check(self):
        ok = True
        message = ""
        
        # check that all 5 ships exist
        shipIds = list(self.ships.keys())
        shipIds.sort()
        missing = [x in shipIds for x in range(1,6)]
        if any(x == True for x in missing):
            message += "Missing ships with IDs: " + str(", ".join([i+1 for i,m in missing if m == 0])) + "\n"
            ok = False

        shipList = list(self.ships.values())        
        # check that no ships are over the edge
        for ship in shipList:
            if sum(sum(self.grid == ship.shipId)) != ship.length:
                message += "Ship over edge (or wrong length, possibly due to overlap), id = " + str(ship.shipId) + "\n"
                ok = False
            
        if not ok:
            return (False, message)
        else:
            return (True, "Board is OK")
        
    def ship_at_spot(self, spot):
        r,c = rowColForSpot(spot)
        if self.grid[r-1,c-1] == 0:
            return 0
        return self.ship_with_id(self.grid[r-1,c-1])

    def row_col_with_shipId(self, shipId):
        """Returns np.arrays"""
        r,c = np.where(self.grid == shipId)
        return (r+1, c+1)
        
    def ship_with_id(self, shipId):
        return self.ships[shipId]
        
    def plot(self):
        None

    def fire(self, board, spot):
        r,c = rowColForSpot(spot)
        (hit, sunk, targetId, msg) = board.incoming(spot)
        if hit > 0:
            #print("Hit!")
            self.targetGrid[r-1,c-1] = kHit
            if sunk:
                #print(shipInfo[targetId]["name"] + " sunk.")
                self.targetGrid[r-1,c-1] = targetId
        else:
            self.targetGrid[r-1,c-1] = kMiss
            #print("Miss.")
        return (hit, sunk, targetId)
        
    def incoming(self, spot, hideId=True):
        r,c = rowColForSpot(spot)
        shipId = self.grid[r-1,c-1]
        if shipId > 0:
            hit = True
        else:
            hit = False
        sunk = False
        
        if hit is False:
            msg = "miss"
        else:
            ship = self.ship_at_spot(spot)
            rows, cols = self.row_col_with_shipId(shipId)
            msg = ""
            if all(np.diff(rows) == 0):
                hit_position = np.where(cols - c == 0)[0] + 1
            elif all(np.diff(cols) == 0):
                hit_position = np.where(rows - r == 0)[0] + 1
            else:
                hit_position = np.array(())
            if ship.damage_at_position(hit_position) > 1:
                msg = str(ship.damage_at_position[hit_position]) + " hits"
            else:
                msg = "hit"
                
            # record hit on ship
            already_sunk = self.ships[shipId].sunk()
            self.ships[shipId].hit(hit_position)
            sunk = self.ships[shipId].sunk()
            # check for sinking
            if sunk and not already_sunk:
                if len(msg) == 0:
                    msg = ship.shipName + " sunk"
                else:
                    msg += "; " + ship.shipName + " sunk"
            elif hideId:
                shipId = 0
            
        return (hit, sunk, shipId, msg)
    
    def damage_at_row_col(self, r, c):
        """Returns the damage (either 0 or 1) at the input row/col coordinate,
        which ranges from 1-10."""
        shipId = self.grid[r-1, c-1]
        if shipId > 0:
            rows, cols = self.row_col_with_shipId(shipId)
            if all(np.diff(rows) == 0):
                hit_position = np.where(cols - c == 0)[0] + 1
            elif all(np.diff(cols) == 0):
                hit_position = np.where(rows - r == 0)[0] + 1
            else:
                ValueError("Could not determine ship position for row/col = " + str((r,c)))
            return(np.min((1, self.ships[shipId].damage_at_position(hit_position))))
        else:
            return 0
        
    def ship_list(self):
        return [self.ships[shipId] for shipId in range(1, len(self.ships)+1)]
    
    def targets_near(self, targetSpot):
        """Returns all spots touching the input spot that have not yet
        been targeted."""
        r,c = rowColForSpot(targetSpot)
        rows = np.array((0,1,0,-1)) + r
        cols = np.array((1,0,-1,0)) + c
  
        ivalid = (cols <= BOARD_SIZE) * (cols > 1) * (rows <= BOARD_SIZE) * (rows > 1)
        rows = rows[ivalid]
        cols = cols[ivalid]
        ivalid = self.targetGrid[rows-1, cols-1] == kUnknown
        rows = rows[ivalid]
        cols = cols[ivalid]
        return spotsForRowCol(rows,cols)
        
    def spots_near(self, spot):
        """Returns all spots touching the input spot."""
        r,c = rowColForSpot(spot)
        rows = np.array((0,1,0,-1)) + r
        cols = np.array((1,0,-1,0)) + c
        ivalid = (cols <= 10) * (cols > 1) * (rows <= 10) * (rows > 1)
        rows = rows[ivalid]
        cols = cols[ivalid]
        return spotsForRowCol(rows,cols)
         
    def afloat_ships(self):
        return np.array([ship.afloat() for ship in self.ship_list()])
    
    def sunk_ships(self):
        return np.array([ship.sunk() for ship in self.ship_list()])
          
    def damage_by_ship(self):
        """Returns a vector with the damage on each of the 5 ships. Note that 
        each ship can sustain a different number of damage points before sinking.
        """
        return np.array([np.sum(ship.damage) for ship in self.ship_list()])
    
    def grid_image(self):
        """Returns a matrix corresponding to the board's flat grid (where the
        ships are). The matrix has a 0 for no shot and 1 for a hit. 
        Use display_ships to get ship bounding boxes."""
        # R = np.reshape(np.tile(np.arange(BOARD_SIZE) + 1, BOARD_SIZE), \
        #                (BOARD_SIZE, BOARD_SIZE))
        # C = np.transpose(R)
        # f = lambda r,c : self.damage_at_row_col(r, c)
        # return f(R, C)
        im = np.zeros((BOARD_SIZE, BOARD_SIZE))
        for shipId in self.ships:
            dmg = self.ships[shipId].damage
            if any(dmg > 0):
                rows, cols = self.row_col_with_shipId(shipId)
                im[rows-1, cols-1] = dmg
        return im
            
    def ship_rects(self):
        """Returns bounding boxes for all ships. Boxes run from 0.5 to 10.5."""
        rects = {}
        for shipId in self.ships:
            rows, cols = self.row_col_with_shipId(shipId)
            rects[shipId] = np.array((np.min(cols)-0.5, np.min(rows)-0.5, \
                                      np.max(cols) - np.min(cols) + 1, \
                                      np.max(rows) - np.min(rows) + 1))
        return rects
        
    def target_grid_image(self):
        """Returns a matrix corresponding to the board's vertical grid (where 
        the player keeps track of their own shots). 
        The matrix has a 0 for no shot, -1 for miss (white peg), 1 for a hit, 
        and 10 + shipId (11-15) if the spot is a hit on a known ship type."""
        return self.targetGrid
    
    def show_grids(self):
        
        # 'tab:blue', 'tab:red', 'steelblue', 'cadetblue'
        
        # figure background to black
        # axes ticks/tick labels off
        # grid
        #cmap = (mpl.colors.ListedColormap(['cadetblue', 'snow', 'firebrick' ]) \
        #        .with_extremes(over='0.25', under='0.75'))
        label_color = "lightgray"
        grid_color = "gray"
        axis_color = "lightgray"
        bg_color = "#202020"
        ship_outline_color = "#202020"
        ship_color = "dimgray"
        
        ocean_color = 'tab:blue' #cadetblue
        miss_color = "whitesmoke"
        hit_color = "tab:red" #"firebrick"
        
        cmap_flat = mpl.colors.ListedColormap([ocean_color, miss_color, hit_color])
        cmap_vert = mpl.colors.ListedColormap([ocean_color, miss_color, hit_color])
        
        fig, axs = plt.subplots(2, 1, figsize = (10,6))
        
        flat_grid = self.grid_image()
        vert_grid = self.target_grid_image()
        
        # grid lines
        grid_extent = [1-0.5, BOARD_SIZE+0.5, BOARD_SIZE+0.5, 1-0.5]
        for ax in axs:
            ax.set_xticks(np.arange(1,11), minor=False)
            ax.set_xticks(np.arange(0.5,11), minor=True)        
            ax.set_yticks(np.arange(1,11), minor=False)            
            ax.set_yticks(np.arange(0.5,11), minor=True)
            for spine in ax.spines.values():
                spine.set_color(axis_color)
            ax.xaxis.grid(False, which='major')
            ax.xaxis.grid(True, which='minor', color=grid_color, linestyle=':', linewidth=0.5, )
            ax.yaxis.grid(False, which='major')
            ax.yaxis.grid(True, which='minor', color=grid_color, linestyle=':', linewidth=0.5)
            ax.set_xticklabels([""]*(BOARD_SIZE+1), minor=True)
            ax.set_xticklabels([str(n) for n in range(1,BOARD_SIZE+1)], minor=False, color=label_color)            
            ax.set_yticklabels([""]*(BOARD_SIZE+1), minor=True)
            ax.set_yticklabels([chr(65+i) for i in range(BOARD_SIZE)], minor=False, color=label_color) 
            ax.xaxis.tick_top()
            ax.tick_params(color = 'lightgray')
            ax.tick_params(which='minor', length=0)
            
        axs[0].imshow(vert_grid, cmap=cmap_vert, extent=grid_extent)
        axs[1].imshow(flat_grid, cmap=cmap_flat, extent=grid_extent)
        
        # ships
        ship_boxes = self.ship_rects()
        for shipId in ship_boxes:
            box = ship_boxes[shipId]
            axs[1].add_patch( 
                mpl.patches.Rectangle((box[0], box[1]), box[2], box[3], \
                                      edgecolor = ship_outline_color,
                                      fill = False,                                       
                                      linewidth = 2))
            axs[1].text(box[0]+0.12, box[1]+0.65, shipInfo[shipId]["name"][0])
            
        fig.set_facecolor(bg_color)
        return fig
                     
class BSPlayer:
    
    """This class implements a 'placement strategy' and an 'offense strategy'
    in playing a game of Battleship. A player can also be Human, in which
    case this class implements a text-based interface for displaying the state of
    the game and asking for a target on each turn.
    
    Possible AI strategies include are discussed below.
    
    Placement Strategies
    - Random        
        Ships are randomly placed on the board.
        
    - Random with spacing   
        Ships are randomly placed with at least 1 space between them.
        
    - Random with clustering
        Ships are randomly placed but with greater likelihood of being close 
        to one another.
        
    - Random with distance
        Ships are randomly placed but distance between them is maximized
        
    Offense Strategies
    - Random
        Each shot is completely random
        
    - Random kill
        Shots are random until a hit, then clustered around the hit until a 
        ship is sunk.
        
    - Grid search N
        Systematic shots across the board with N spaces between shots. After
        the grid is complete, kill any open hits (i.e., hits that do not have
        a possible sunk ship connected to them). After that, switch to 
        'Random kill' strategy.
        
    - Grid search N kill
        As above, but any hits during the grid search immediately trigger
        'kill' strategy until a ship is sunk, then return to grid search.
        
    - X smart kill
        Any of the above 'kill' strategies, but with a smarter kill algorithm.
        
    - Neural Net
        Spots are chosen based on the output of a trained Neural Network 
        algorithm.
        """
        
    def __init__(self, playerType, strategy=None, name="?"):
        self.name = name
        self.board = BSBoard()
        self.playerType = playerType
        self.strategy = None
        self.strategyData = {}
        #self.offenseMode = "hunt"       # hunt (search for ships) or kill (sink most recent hit ship)
        self.shotHistory = []
        self.remainingTargets = [""] * BOARD_SIZE * BOARD_SIZE
        
        n = 0
        for r in [chr(x) for x in range(65,65+BOARD_SIZE)]:
            for c in range(1,11):
                self.remainingTargets[n] = r + str(c)
                n += 1                
        
        self.strategyData = {"lastTarget": None,
                             "lastHit": None,
                             "lastMiss": None,
                             "shipsSunk": [],
                             "state": ""}
        
        if self.playerType == "robot":
            if strategy == "random":
                self.strategy = {"setup": "random",
                                 "offense": "random"}
            elif strategy == "hunt kill" or strategy == "random kill":
                self.strategy = {"setup": "random",
                                 "offense": "random kill"}
                self.strategyData["state"] = "hunt"
            else:
                ValueError("Non-random strategy not supported for robot player type.")
        elif self.playerType == "human":
            if strategy == "random":
                self.strategy = {"setup": "random",
                                 "offsense": "manual"}
            else:
                self.strategy = {"setup": "manual",
                                 "offsense": "manual"}
        else:
            ValueError("playerType must be 'robot' or 'human'.")
        
    def setup_board(self):
        if self.strategy["setup"] == "random":
            self.board.place_fleet("random")
        elif self.strategy["setup"] == "manual":
            # this should probably be moved into BSBoard...
            for shipId in range(1,6):
                ok = True
                while not ok:
                    spot = input("Spot for " + shipInfo[shipId]["name"] + ": ")
                    facing = input("Facing for " + shipInfo[shipId]["name"] + ": ")                    
                    ok = self.board.place_ship(shipId, spot, facing)
                    if not ok:
                        print("Invalid placement. Try again.")
                self.board.place_ship(shipId, spot, facing)
            if self.board.check():
                print("Fleet placed by hand successfully!")
                print(self.board)
            else:
                print("Something is wrong with fleet placement:")
                print(self.board)
        else:
            ValueError("setup strategy must either be manual or random currently.") 
            
    def update_data(self, lastTarget, lastOutcome):
        hit, sunk, shipId = lastOutcome
        self.strategyData["lastTarget"] = lastTarget
        if hit:
            self.strategyData["lastHit"] = lastTarget
        else:
            self.strategyData["lastMiss"] = lastTarget
        if sunk:
            self.strategyData["shipsSunk"] += [shipId]
            
        # Any strategy updates should be done here:
        if self.strategy["offense"] == "random kill":
            if sunk:
                self.strategyData["state"] = "hunt"
            elif hit:
                self.strategyData["state"] = "kill"
        
        
    def take_turn(self, otherPlayer):
        if self.playerType == "human":
            print(self.board.targetGrid)
            print(self.board)
            target = input("Target spot: ")
            _,_ = rowColForSpot(target)
            self.board.fire(otherPlayer.board, target)
        elif self.playerType == "robot":
            target = self.pick_target()
            outcome = self.board.fire(otherPlayer.board, target)
            self.remainingTargets.pop(self.remainingTargets.index(target))
            self.update_data(target, outcome)

        self.shotHistory += [target]
            
    def last_shot(self):
        """Returns a value equal to kMiss if most recent shot was a miss,
        kHit if a hit, and equal to shipId if a known ship was sunk."""
        if len(self.shotHistory) == 0:
            return None
        r,c = rowColForSpot(self.shotHistory[-1])
        return self.board.targetGrid[r-1,c-1]
    
    def pick_target(self):
        """This is where the offensive strategy is implemented to choose a 
        target spot."""
        
        # Random strategy
        if self.strategy["offense"] == "random":
            target = self.random_target()
            
        # Random Kill strategy
        elif self.strategy["offense"] == "random kill":
            # If in hunt mode:
            #   If last shot was a hit and a sink: select a new random target.               
            #   If last shot was a hit but not a sink: should not happen.
            #   If last shot was a miss: select a new random target.
            # If in kill mode:
            #   If last shot was a hit and a sink: Go to hunt mode, select a new random target               
            #   If last shot was a hit but not a sink: Select a new target near last hit
            #   If last shot was a miss: Select a new target near last hit
            # 
            # STILL NEED TO:
                # If no valid target is around the last hit, go to the previous hit with a nearby valid target.
                # If in hunt mode, two hits in a row are found (but not sunk), continue along the line, then go opposite along the line.
            lastShot = self.last_shot()
            if self.strategyData["state"] == "hunt":
                if lastShot == None:
                    target = self.random_target()
                elif lastShot > kHit:     # got lucky and sank a ship in hunt mode last turn.
                    target = self.random_target()
                elif lastShot == kHit:
                    print("Warning: last shot was a hit but not a sink, and we're still in HUNT mode. That should not happen. Going into KILL mode.")
                    targets = self.board.targets_near(self.strategyData["lastHit"])
                    if len(targets) == 0:
                        print("Could not find a valid target near the last hit. Selecting randomly, and staying in HUNT mode.")
                        return self.random_target()
                    target = np.random.choice(targets)
                    self.strategyData["state"] = "kill"
                else:   # miss
                    target = self.random_target()
            elif self.strategyData["state"] == "kill":
                if lastShot == None:
                    print("In KILL mode on first shot of game. That should not happen. Selecting random target, going to HUNT mode.")
                    target = self.random_target()
                    self.strategyData["state"] = "hunt"
                elif lastShot > kHit:  
                    print("In KILL mode despite sinking a ship last turn. That should not happen. Going to HUNT mode and selecting random target.")
                    self.strategyData["state"] = "hunt"
                    target = self.random_target()
                elif lastShot == kHit:
                    targets = self.board.targets_near(self.strategyData["lastHit"])
                    if len(targets) == 0:
                        print("Could not find a valid target near the last hit. Selecting randomly, and staying in HUNT mode.")
                        return self.random_target()
                    target = np.random.choice(targets)
                else:   # miss
                    targets = self.board.targets_near(self.strategyData["lastHit"])
                    if len(targets) == 0:
                        print("Could not find a valid target near the last hit. Selecting randomly, and staying in HUNT mode.")
                        return self.random_target()
                    target = np.random.choice(targets)
        else:
            ValueError("Only 'random' and 'random kill' offensive strategies are currently supported.")
        return target       
            
    def random_target(self):
        return self.remainingTargets[np.random.randint(0,len(self.remainingTargets))]
        
    def still_alive(self):
        return any(self.board.afloat_ships())
    
class BSGame:
    
    def __init__(self, player1type="human", player2type="human", 
                 player1strategy="random", player2strategy="random", 
                 player1name = "player1", player2name = "player2", gameId = None):
        
        self.gameId = gameId
        self.player1 = self.player_for_type(player1type, player1strategy, player1name)
        self.player2 = self.player_for_type(player2type, player2strategy, player2name)
        
        self.player1.setup_board()
        self.player2.setup_board()
        
        self.winner = None
        self.loser = None
        self.turnCount = 0
        
    def player_for_type(self, playerType, strategy, name):
        return BSPlayer(playerType, strategy, name = name)
                
    def play(self):
        """Play one game of Battleship. Each player takes a turn until one
        of them has no ships remaining."""
        
        gameOn = True
        self.turnCount = 0
        while gameOn:
            self.player1.take_turn(self.player2)
            self.player2.take_turn(self.player1)
            self.turnCount += 1
            gameOn = self.player1.still_alive() and self.player2.still_alive()
            
        if self.player1.still_alive():
            self.winner = self.player1
            self.loser = self.player2
        else:
            self.winner = self.player2
            self.loser = self.player1
            
        self.print_outcome()
            
    def print_outcome(self):
        """Prints the results of the game for the input winner/loser players 
        (which include their respective boards)."""
        
        print("")
        print("GAME OVER!")
        print("Player " + self.winner.name + " wins.")
        print("  Player " + self.winner.name + " took " \
              + str(len(self.winner.shotHistory)) + " shots, and sank " \
              + str(sum(1 - self.loser.board.afloat_ships())) + " ships.")
        print("  Player " + self.loser.name + " took " \
              + str(len(self.loser.shotHistory)) + " shots, and sank " \
              + str(sum(1 - self.winner.board.afloat_ships())) + " ships.")
        print("Game length: " + str(self.turnCount) + " turns.")
        print("(Game ID: " + str(self.gameId) + ")")
        print("")
        
        
        
        
    
