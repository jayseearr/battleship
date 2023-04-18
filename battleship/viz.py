#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 21:31:10 2022

@author: jason
"""

from collections import Counter
import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
import time

from battleship import constants
#from battleship.board import Board
#from battleship.player import AIPlayer
from battleship.ship import Ship

#%% Viz Class Definition

class Viz:
    
    bg_color = "#202020"
    ship_outline_color = "#202020"
    #peg_outline_color = "#202020"
    peg_outline_color = None
    ship_color = "darkgray"
    target_color = "yellow"
    ocean_color = 'tab:blue' #cadetblue
    miss_color = "whitesmoke"
    hit_color = "tab:red" #"firebrick"
    peg_radius = 0.3
    
    
### "Private" Methods ###
    
    @staticmethod
    def _pegs(xy, color, edgecolor=None):
        """
        Get pyplot objects that represent pegs to show on the ocean or
        target grid.

        Parameters
        ----------
        xy : tuple
            The x and y coordinates of the peg.
        color : str
            String describing the desired fill color of the peg.
        edgecolor : str, optional
            String describing the desired edge color of the peg. 
            The default is None.

        Returns
        -------
        list
            List of pyplot.Circle objects representing hit/miss pegs.

        """
        if edgecolor == None:
            edgecolor = Viz.peg_outline_color
        return [plt.Circle(col_row, radius=Viz.peg_radius, facecolor=color,
                    edgecolor=edgecolor) for col_row in xy]
    
    @staticmethod
    def _format_axes(ax, board_size):
        """
        Applies standardized formatting to the input axes object.

        Parameters
        ----------
        ax : pyplot axes
            An axes to be formatted.
        board_size : int
            The size of the board that is plotted on ax.

        Returns
        -------
        None.

        """
        label_color = "lightgray"
        grid_color = "gray"
        axis_color = "lightgray"
        ocean_color = 'tab:blue' #cadetblue
        
        # grid lines
        ax.set_facecolor(ocean_color)
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
        ax.set_xticklabels([""]*(board_size+1), minor=True)
        ax.set_xticklabels([str(n) for n in range(1,board_size+1)], 
                           minor=False, color=label_color)            
        ax.set_yticklabels([""]*(board_size+1), minor=True)
        ax.set_yticklabels([chr(65+i) for i in range(board_size)], 
                           minor=False, color=label_color) 
        ax.xaxis.tick_top()
        ax.tick_params(color = 'lightgray')
        ax.tick_params(which='minor', length=0)
        ax.set_aspect('equal')
        
        
### "Public" Methods ###
    
    @staticmethod
    def show_player(player, show_possible_targets=True):
        """
        Shows the player's board (both target and ocean grids).
        Optionally shows the possible targets that the player can select from.

        Parameters
        ----------
        player : Player subclass 
            An instance of one of the Player subclasses (i.e., AIPlayer,
            HumanPlayer). This player's current board and target options
            will be plotted.
        show_possible_targets : TYPE, optional
            If True, the coordinates identified as possible targets will
            be plotted. Even though these targets may have different 
            probabilities of being selected as the next target, they are all
            plotted with the same formatting. 
            The default is True.

        Returns
        -------
        fig : pyplot figure
            The figure that contains the plot axes.

        """
        fig, axs = plt.subplots(2, 1, figsize = (6,10), squeeze=True)
        board_size = player.board.size
        grid_extent = [1-0.5, board_size+0.5, board_size+0.5, 1-0.5]
        colmap = mpl.colors.ListedColormap([Viz.ocean_color, Viz.ocean_color])
        
        for ax in axs.flatten():
            Viz._format_axes(ax, board_size)

        axs[0].imshow(np.zeros(player.board.target_grid.shape), cmap=colmap, 
                      extent=grid_extent)        
        # add pegs to target grid
        rows,cols = np.where(player.board.target_grid 
                             == constants.TargetValue.MISS)
        for p in Viz._pegs(zip(cols+1,rows+1), Viz.miss_color):
            axs[0].add_patch(p)
        rows,cols = np.where((player.board.target_grid 
                              >= constants.TargetValue.HIT))
        for p in Viz._pegs(zip(cols+1,rows+1), Viz.hit_color):
            axs[0].add_patch(p)    
            
        # Add the possible target locations to the target grid
        if show_possible_targets:
            if player.offense.target_probs:
                cols_rows = [(c+1,r+1) for (r,c) in player.offense.target_probs]
                for p in Viz._pegs(cols_rows, None, Viz.target_color):
                    axs[0].add_patch(p)
            col_row = (player.last_target[1]+1, player.last_target[0]+1)
            shot_color = (Viz.hit_color if 
                          player.board.target_grid[player.last_target] >= 0.
                          else Viz.miss_color)
            # add a special circle over the actual shot location
            axs[0].add_patch(plt.Circle(col_row,
                                        radius=Viz.peg_radius,
                                        facecolor = shot_color,
                                        edgecolor = "magenta",
                                        linewidth = 3.0))
            
        axs[1].imshow(np.zeros(player.board.target_grid.shape), cmap=colmap, 
                      extent=grid_extent)        
        # add pegs to ships
        rows,cols = np.where(player.board.ocean_grid_image() == 2)
        for p in Viz._pegs(zip(cols+1,rows+1), Viz.hit_color):
            axs[1].add_patch(p)
        
        # ships
        ship_boxes = player.board.ship_rects()
        for shipId in ship_boxes:
            box = ship_boxes[shipId]
            axs[1].add_patch( 
                mpl.patches.Rectangle((box[0], box[1]), box[2], box[3], \
                                      edgecolor = Viz.ship_outline_color,
                                      fill = False,                                       
                                      linewidth = 2))
            axs[1].text(box[0]+0.12, box[1]+0.65, 
                           Ship.data[shipId]["name"][0])

        fig.set_facecolor(Viz.bg_color)
        return fig
    
    
    @staticmethod
    def show_board(board, grid="both", possible_targets=None):
        """
        Shows a board as a pyplot figure. Displays either the ocean grid, 
        the target grid, or both (default).
        
        Can also show possible targets as unfilled circles.

        Parameters
        ----------
        board : Board
            The board to display on a figure.
        grid : str, optional
            Either "ocean" for ocean grid, "target" for target grid, or "both"
            for both ocean and target grids. The default is "both".
        possible_targets : list, optional
            A list of possible targets (row/col tuples). The default is None.

        Returns
        -------
        pyplot fig containing the 1-2 axes.

        """
        
        if grid.lower() == "both":
            grid = ["target", "ocean"]
        else:
            grid = [grid.lower()]
        if grid[0] == "t":
            grid[0] = "target"
        elif grid[0] == "o":
            grid[0] = "ocean"
        if grid[-1] == "t":
            grid[-1] = "target"
        elif grid[-1] == "o":
            grid[-1] = "ocean"
        if (grid[
                0] not in ["target", "ocean"] or 
                grid[-1] not in ["target", "ocean"]):
            raise ValueError("grid must be 'target', 'ocean', "
                             "or 'both' (default).")    
        if len(grid) == 2:
            fig, axs = plt.subplots(2, 1, figsize = (10,6), squeeze=False)
        else:
            fig, axs = plt.subplots(1, 1, figsize = (5,6), squeeze=False)
        
        for ax in axs.flatten():
            Viz._format_axes(ax, board.size)

        if "target" in grid:
            Viz.add_target_grid(axs[0][0], board)
            if possible_targets:
                for t in possible_targets:
                    axs[0].add_patch(
                        plt.Circle((t[0]+1,t[1]+1), 
                                   radius=Viz.peg_radius*1.2,
                                   edgecolor = Viz.target_color)
                        )
                        
        if "ocean" in grid:
            Viz.add_ocean_grid(axs[-1][0], board)
                
        fig.set_facecolor(Viz.bg_color)
        return fig
    
    
    @staticmethod
    def show_boards(board1, board2, grid="both", 
                    possible_targets1=None, 
                    possible_targets2=None):
        """
        Shows the current state of two boards side-by-side.

        Parameters
        ----------
        board1 : Board
            The first player's board.
        board2 : Board
            The second player's board.
        grid : str, optional
            String describing which grids to plot. Options are:
                'ocean', 'target', or 'both'. The default is "both".
        possible_targets1 : list, optional
            An optional list of possible target coordinates for board 1. The
            possible targets will be shown on board 1's target grid, if it
            is being shown. The default is None.
        possible_targets2 : list, optional
            An optional list of possible target coordinates for board 2. The
            possible targets will be shown on board 2's target grid, if it
            is being shown. The default is None.
            
        Returns
        -------
        fig : pyplot figure
            The figure that contains the plot axes.

        """
        boards = (board1, board2)
        possible_targets = (possible_targets1, possible_targets2)
        if grid.lower() == "both":
            grid = ["target", "ocean"]
        else:
            grid = [grid.lower()]
        if (grid[0] not in ["target", "ocean"] or 
                grid[-1] not in ["target", "ocean"]):
            raise ValueError("grid must be 'target', 'ocean', "
                             "or 'both' (default).")    
        if len(grid) == 2:
            fig, axs = plt.subplots(2, 2, figsize = (10,12), squeeze=False)
        else:
            fig, axs = plt.subplots(1, 2, figsize = (10,6), squeeze=False)
        board_size = np.max((board1.size, board2.size))
        
        for ax in axs.flatten():
            Viz._format_axes(ax, board_size)

        if "target" in grid:
            for c in [0,1]:
                Viz.add_target_grid(axs[0,c], boards[c])
                if possible_targets[c]:
                    for t in possible_targets[c]:
                        axs[0,c].add_patch(
                            plt.Circle((t[0]+1,t[1]+1), 
                                       radius=Viz.peg_radius*1.2,
                                       edgecolor = Viz.target_color)
                            )
                        
        if "ocean" in grid:
            for c in [0,1]:
                Viz.add_ocean_grid(axs[-1,c], boards[c])
                
        fig.set_facecolor(Viz.bg_color)
        return fig
    
    @staticmethod
    def show_replay(game):
        history1 = game.player1.outcome_history
        history2 = game.player2.outcome_history
        board1 = game.player1.board
        board2 = game.player2.board
        
        # This is what we'd really like to do, but it creates circular refs
        # because of the call to Viz in Board.show2:
        # new_board1 = Board(board1.size)
        # new_board2 = Board(board2.size)
        # Here is a workaround:
        new_board1 = board1.copy()
        new_board2 = board2.copy()
        new_board1.__init__(board1.size)
        new_board2.__init__(board2.size)
        
        new_board1.add_fleet(board1.ship_placements)
        new_board2.add_fleet(board2.ship_placements)
        
        # Another workaround for the commented-out code:
        # new_player1 = AIPlayer(name=game.player2.name)
        # new_player1.board = new_board1
        # new_player2 = AIPlayer(name=game.player1.name)
        # new_player2.board = new_board2
        new_player1 = game.player1.dumb_copy()
        new_player1.__init__(name=game.player1.name, board_size=new_board1.size)
        new_player2 = game.player2.dumb_copy()
        new_player2.__init__(name=game.player2.name, board_size=new_board2.size)
        
        new_player1.opponent = new_player2
        
        if game.game_result['first_move'] == 1:
            first_player = new_player1
            second_player = new_player2
            first_history = history1
            second_history = history2
        elif game.game_result['first_move'] == 2:
            first_player = new_player2
            second_player = new_player1
            first_history = history2
            second_history = history1
        else:
            raise ValueError("first_move should be 1 or 2.")
            
        for turn in range(game.game_result['turn_count']):
            first_player.fire_at_target(first_history[turn]['coord'])
            second_player.fire_at_target(second_history[turn]['coord'])
            Viz.show_boards(first_player.board, 
                            second_player.board, 
                            grid="both")
            plt.draw()
            #input(f"Turn {turn+1} of {game.game_result['turn_count']}. "
            #      f"Press Enter to continue.")
            time.sleep(0.5)
        plt.show()
       
        
    @staticmethod
    def show_player_history(player, annotate=None):
        """
        Shows the incoming and outgoing shot history of the player, with 
        each hit/miss on the target board annotated with the turn in which it 
        occurred.

        Parameters
        ----------
        player : Player
            The player whose history will be shown.
        annotate : str
            A string indicating how shots are annotated. Options are:
                'all' : The turn number will be drawn on each shot, hit or miss.
                'hits' : Turn number will be drawn only on hits.
                'smart' : Turn number will be drawn on shots taken once a ship
                            has been found until a ship is sunk
                'none' : No turn number annotation (default)

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure to display.

        """
        fig = Viz.show_board(player.board)
        axs = fig.axes
        ax = axs[0]
        text_color = {'hit': 'white', 'miss': 'black', 'sink': 'black'}
        
        if annotate and annotate.lower() != 'none':
            annotate = annotate.lower()
            kill_mode = False  # for 'smart' annotation
            for (turn, outcome) in enumerate(player.outcome_history):
                coord = outcome['coord']
                x = coord[1] + 1.
                y = coord[0] + 1.
                result = ('sink' if outcome['sunk'] 
                          else 'hit' if outcome['hit']
                          else 'miss')
                if annotate == 'all':
                    pass
                elif annotate == 'hits':
                    if result == 'miss':
                        result = None
                elif annotate == 'smart':
                    if kill_mode:
                        #if result == 'hit':
                        #    pass
                        #elif result == 'miss':
                        #    pass
                        if result == 'sink':
                            kill_mode = False
                    else:
                        if result == 'hit':
                            kill_mode = True
                        elif result == 'miss':
                            result = None
                        #elif result == 'sink':
                        #    kill_mode = False
                else:
                    raise ValueError('annotate parameter must be "all", '
                                     ' "hits", "smart", or "none".')
                if result:
                    ax.text(x, y + 0.02, f"{turn + 1}", 
                            color = text_color[result],
                            fontsize = 'x-small',
                            fontweight = 'normal',
                            horizontalalignment = 'center',
                            verticalalignment = 'center')
        
        return fig
        
        
    @staticmethod
    def show_outcome(outcome, board_size):
        """
        Show the location of the shot in the input outcome on a grid. 
        The shot is represented by a circle at its target location color-coded
        according to whether it was a miss (white), hit (light red), or hit
        plus sink (darker red).

        Parameters
        ----------
        outcome : dict
            Outcome dictionary with the following key/values:
                'coord' : (row,col) tuple
                'hit' : bool; True if the shot was a hit
                'sunk_ship_id' : the ShipType of the ship that was sunk, if any
        board_size : int
            The number of spaces along one edge of the board on which the
            outcome occurred.

        Returns
        -------
        fig : pyplot figure
            The figure that contains the plot axes.

        """
        fig, axs = plt.subplots(1, 1, figsize = (6,6), squeeze=True)
        axs = [axs]
        for ax in axs:
            Viz._format_axes(ax, board_size)
            
        peg_color = (Viz.sink_color if outcome['sunk_ship_id'] else
                     Viz.hit_color if outcome['hit'] else
                     Viz.miss_color)
        x,y = outcome['coord']
        axs[0].add_patch(plt.Circle((x+1,y+1), 
                                    radius = Viz.peg_radius,
                                    facecolor = peg_color,
                                    edgecolor = Viz.peg_outline_color))
        fig.set_facecolor(Viz.bg_color)
        return fig
         
    
    @staticmethod
    def shot_history(player):
        """
        Show the location of the shot in the input outcome on a grid. 
        The shot is represented by a circle at its target location color-coded
        according to whether it was a miss (white), hit (light red), or hit
        plus sink (darker red).

        Parameters
        ----------
        player : Player subclass instance
            An instance of one of the subclasses of Player (for example,
            AIPlayer or HumanPlayer) that has a outcome_history property.
        
        Returns
        -------
        fig : pyplot figure
            The figure that contains the plot axes.
        """
        fig, axs = plt.subplots(2, 1, figsize = (6,10), squeeze=True)
        for ax in axs:
            Viz._format_axes(ax, player.board.size)
            
        for outcome in player.outcome_history:
            peg_color = (Viz.sink_color if outcome['sunk_ship_id'] else
                         Viz.hit_color if outcome['hit'] else Viz.miss_color)
            x,y = outcome['coord']
            axs[0].add_patch(plt.Circle((x+1,y+1), 
                                        radius = Viz.peg_radius,
                                        facecolor = peg_color,
                                        edgecolor = Viz.peg_outline_color))
        return fig
        
    
    @staticmethod
    def show_probs(targets, probs=None, board_size=10):
        """
        Plot the relative probabilities of selecting a target coordinate
        from an input list of targets and probabilities.

        Parameters
        ----------
        targets : list or dict
            If a list, a list of coordinate tuples (row,col).
            If a dict, the keys should be coordinate tuples and the 
            values should be the probability of selecting the corresponding
            coordinate.
        probs : list, optional
            A list of probability values (floats) corresponding to the 
            probability of selecting each coordinate in the targets input.
            If targets is a dict, this input should be None.
            The default is None.
        board_size : int, optional
            The number of spaces along one edge of the board. 
            The default is 10.

        Returns
        -------
        fig : pyplot figure
            The figure that contains the plot axes.

        """
        if probs is None and isinstance(targets, dict):
            targets, probs = zip(*targets.items())
            targets = list(targets)
            probs = list(probs)

        image = np.zeros((board_size,board_size))
        if probs is None or len(probs) == 1:
            probs = np.ones(len(targets))
        for (i,target) in enumerate(targets):
            image[target] = probs[i]
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
        ax.imshow(image)
        ax.invert_yaxis()
        return fig
    
    
    @staticmethod
    def add_target_grid(ax, board):
        """
        Adds a plot of the board's target grid to the input axes object.'

        Parameters
        ----------
        ax : matplotlib.axes._subplots.AxesSubplot
            The axes onto which the target grid will be plotted.
        board : Board
            board's target grid (i.e., its ships) will be plotted on ax.

        Returns
        -------
        None.

        """
        vert_grid = board.target_grid_image()
        cmap = mpl.colors.ListedColormap([Viz.ocean_color, Viz.miss_color, 
                                          Viz.hit_color])
        grid_extent = [1-0.5, board.size+0.5, board.size+0.5, 1-0.5]
        ax.imshow(np.zeros(vert_grid.shape), cmap=cmap, 
                  extent=grid_extent)
        # add pegs to target grid
        rows,cols = np.where(vert_grid == constants.TargetValue.MISS)
        for (x,y) in zip(cols+1,rows+1):
            ax.add_patch(
                plt.Circle((x,y), radius = Viz.peg_radius, 
                           facecolor = Viz.miss_color, 
                           edgecolor = Viz.peg_outline_color) 
                ) 
        # red pegs    
        rows,cols = np.where(vert_grid >= constants.TargetValue.HIT)
        for (x,y) in zip(cols+1,rows+1):
            ax.add_patch(
                plt.Circle((x,y), radius = Viz.peg_radius, 
                           facecolor = Viz.hit_color, 
                           edgecolor = Viz.peg_outline_color) 
                ) 
    
    
    @staticmethod
    def add_ocean_grid(ax, board):
        """
        Adds a plot of the board's ocean grid to the input axes object.'

        Parameters
        ----------
        ax : matplotlib.axes._subplots.AxesSubplot
            The axes onto which the ocean grid will be plotted.
        board : Board
            board's ocean grid (i.e., its ships) will be plotted on ax.

        Returns
        -------
        None.

        """
        flat_grid = board.ocean_grid_image()
        cmap = mpl.colors.ListedColormap([Viz.ocean_color, Viz.ship_color, 
                                          Viz.ship_color])
        grid_extent = [1-0.5, board.size+0.5, board.size+0.5, 1-0.5]
        ax.imshow(np.zeros(flat_grid.shape), cmap=cmap, 
                  extent=grid_extent)
        # add pegs to ships
        rows,cols = np.where(flat_grid == 2)
        red_pegs = [plt.Circle((x,y), radius = Viz.peg_radius, 
                               facecolor = Viz.hit_color, 
                               edgecolor = Viz.peg_outline_color) 
                    for (x,y) in zip(cols+1,rows+1)]
        for peg in red_pegs:
            ax.add_patch(peg)
        # ships
        ship_boxes = board.ship_rects()
        for ship_type in ship_boxes:
            box = ship_boxes[ship_type]
            ax.add_patch( 
                mpl.patches.Rectangle((box[0], box[1]), box[2], box[3], \
                                      edgecolor = Viz.ship_outline_color,
                                      fill = False,                                       
                                      linewidth = 2))
            ax.text(box[0]+0.12, box[1]+0.65, 
                    Ship.data[ship_type]["name"][0])
            
            
    @staticmethod        
    def show_coords_on_board(board, coords, grid, 
                             fill=False, color='orange', alpha=None,
                             other_coords = None,
                             other_color = 'indigo',
                             other_fill = None,
                             show_order=False):
        """
        Shows the input board with the input coordinates overlaid as circles.
        Useful for visualizing where coordinates are in relation to the 
        board's ships or target pegs.

        Parameters
        ----------
        board : Board
            Instance of Board with (typically) a populated target and ocean 
            grid.
        coords : list
            List of coordinate tuples (row,col).
        grid : str
            String describing the board grid to show. Options are 'target' or
            'ocean'.
        fill : Bool, optional
            If True, the circles will be filled with the color specified by
            the color input.
        color : str, optional
            A string describing the color of the circles (both edge and fill
            color). Defaults to 'orange'.
        alpha : float, optional
            The alpha value of the circles overlaid on the grid(s). Should be
            between 0.0 and 1.0. Defaults to 1.0 if fill is False (default), 
            and 0.5 if fill is True.
        other_coords : list, optional
            A second list of coordinates that can be plotted using a second
            color.
        other_color : str, optional
            Same behavior as the 'color' input, but applies to the other_coords
            list of coordinates. Defaults to 'indigo'.
        other_fill : bool, optional
            Same behavior as the 'bool' input, but applies to the other_coords
            list of coordinates. Defaults to the same value as fill.
        show_order : bool, optional
            If True, a label indicating each coordinate's order in the coords
            parameter is placed next to each coordinat's circle. Default is 
            False.

        Returns
        -------
        None.

        """
        line_width = 2.0
        peg_radius = 1.25 * Viz.peg_radius
        if alpha == None:
            alpha = 0.5 if fill else 1.0
        if not other_fill:
            other_fill = fill
        fig, axs = plt.subplots(1, 1, figsize = (10,6), squeeze=False)
        board_size = board.size
        ax = axs.flatten()[0]
        Viz._format_axes(ax, board_size)

        if grid.lower() == "target":
            Viz.add_target_grid(ax, board)
        elif grid.lower() == "ocean":
            Viz.add_ocean_grid(ax, board)

        fig.set_facecolor(Viz.bg_color)
        for (idx, coord) in enumerate(coords):
            x = coord[1] + 1    # coord is stored as 0-indexed (row,col)
            y = coord[0] + 1
            ax.add_patch(plt.Circle(xy = (x,y), 
                                    radius = peg_radius, 
                                    fill = fill,
                                    facecolor = color,
                                    alpha = alpha,
                                    edgecolor = color,
                                    linestyle = '-',
                                    linewidth = line_width)
                         )
            if show_order:
                ax.text(x, y, idx+1, color = color,
                        verticalalignment = 'center',
                        horizontalalignment = 'center')
        if other_coords:
            for coord in other_coords:
                x = coord[1] + 1    # coord is stored as 0-indexed (row,col)
                y = coord[0] + 1
                ax.add_patch(plt.Circle(xy = (x,y), 
                                        radius = peg_radius, 
                                        fill = other_fill,
                                        facecolor = other_color,
                                        alpha = alpha,
                                        edgecolor = other_color,
                                        linestyle = '-',
                                        linewidth = line_width)
                             )
                if show_order:
                    ax.text(x, y, idx+1, color = other_color,
                            verticalalignment = 'center',
                            horizontalalignment = 'center')
        return fig
        
    
    @staticmethod
    def plot_placements(placements, board_size=10):
        """
        Shows an image with the placements highlighted.

        Parameters
        ----------
        placements : dict
            Dictionary with keys equal to ShipType (or int) values. The
            corresponding value for each key is a Placement object with
            properties coord, heading, and length.
            
        board_size : TYPE, optional
            DESCRIPTION. The default is 10.

        Returns
        -------
        None.

        """
        X = np.zeros((board_size, board_size))
        for (k,p) in placements.items():
            rows, cols = zip(*p.coords())
            X[np.array(rows),np.array(cols)] = k
        plt.imshow(X)
        
        
    @staticmethod
    def show_placements(board, placements, grid="target",
                        fill=False):
        """
        Shows all of the ship positions for the input placements 
        (coordinate, heading, and ship length).

        Parameters
        ----------
        board : Board
            The board on which the ship positions will be drawn. Positions
            can be drawn either on the ocean grid or target grid (default)
            using the 'grid' parameter. 
        placements : list
            A list of placement dictionaries, each of which must have
            the following key/value pairs:
                'coord': (row, col) tuple
                'heading': str ("N", "S", "E", "W")
                'length': int, number of spaces occuppied by the ship.
        grid : str (optional)
            The board grid on which to draw the ships; either 'target' 
            (default) or 'ocean'. Target indicators (for 'target') and existing 
            ships (for 'ocean') will be drawn under the ship positons defined 
            by the inputs.
        fill : bool (optional)
            The representations of the ships at each placement are unfilled
            ellipses, by default. Setting fill to True makes the fill color
            the same as the edge color, resulting in a much more colorful 
            board.

        Returns
        -------
        None.

        """
        
        oval_scale = 0.7
        oval_alpha = 0.6
        #oval_colors = mpl.colors.TABLEAU_COLORS
        #color_names = [color_names[k] for k in [1,3,4,5,6,8]]
        oval_colors = mpl.colors.CSS4_COLORS
        color_names = list(oval_colors.keys())
        color_names = ['blueviolet', 'lightcoral', 'tomato', 'lightgreen',
                       'rosybrown', 'plum', 'turquoise', 'lemonchiffon', 
                       'lavender']
        fig, axs = plt.subplots(1, 1, figsize = (10,6), squeeze=False)
        board_size = board.size
        ax = axs.flatten()[0]
        Viz._format_axes(ax, board_size)

        if grid.lower() == "target":
            Viz.add_target_grid(ax, board)
        elif grid.lower() == "ocean":
            Viz.add_ocean_grid(ax, board)

        fig.set_facecolor(Viz.bg_color)
        
        # generate ovals for each possible ship placement
        count = 0
        for placement in placements:
            coord = placement.coord
            heading = placement.heading
            length = placement.length
            if heading == "N":
                yc = coord[0] + 0.5 + length / 2.
                xc = coord[1] + 1
                w = 1 * oval_scale
                h = 2 * oval_scale + length - 2
            elif heading == "W":
                xc = coord[1] + 0.5 + length / 2.
                yc = coord[0] + 1
                w = 2 * oval_scale + length - 2
                h = 1 * oval_scale
            else:
                raise ValueError("placement heading should be 'N' or 'W'.")
            pc =  oval_colors[color_names[count % len(color_names)]]
            oval = mpl.patches.Ellipse((xc, yc), width = w, height = h, 
                                       fill = True,
                                       facecolor = pc,
                                       alpha = oval_alpha,
                                       edgecolor = pc,
                                       linestyle = '-'
                                       )
            ax.add_patch(oval)
            count += 1
        return fig
    
    
    @staticmethod
    def show_possibilities_grid(grid):
        """
        Plots the input grid containing the number of possible ship placements
        at each (row,col) coordinate.

        Parameters
        ----------
        grid : numpy 2d array
            A 2d array in which the value at each row/col pair is the number
            of possible ways a ship could be placed so that it overlaps that
            row/col. 
            The grid can be specific to a particular ship (e.g., Submarine)
            or the sum of possibilities for all ships, depending on how it
            was fetched from a Board instance.

        Returns
        -------
        None.

        """
        fig, axs = plt.subplots(1, 1, figsize = (10,6), squeeze=False)
        board_size = grid.shape[0]
        grid_extent = [1-0.5, board_size+0.5, board_size+0.5, 1-0.5]
        ax = axs.flatten()[0]
        Viz._format_axes(ax, board_size)
        ax.imshow(grid, extent=grid_extent)
        fig.set_facecolor(Viz.bg_color)
        
        
    @staticmethod
    def print_player_history(player, turns=None, show_board=False):
        """
        Prints a list of player open hits, target probabilities, target 
        selection, and outcomes.

        Parameters
        ----------
        player : Player
            A player with board and outcome_history properties.
        turns : int
            The number of turns to show (defaults to all turns)
        show_board : bool
            If True, the target grid will be shown at every turn.
            
        Returns
        -------
        None.

        """
        if turns is None:
            turns = len(player.outcome_history)
            
        board = player.board.copy()
        board._target_grid = (constants.TargetValue.UNKNOWN 
                              * np.ones((board.size, board.size)))
        
        for (turn, outcome) in enumerate(player.outcome_history):
            if outcome['hit']:
                if outcome['sunk_ship_type']:
                    result = (f"HIT - "
                              f"{Ship.data[outcome['sunk_ship_type']]['name']} "
                              f"SANK")
                result = 'HIT'
            else:
                result = 'MISS'
            print(f"\nTurn {turn+1}")
            if show_board:
                print(outcome)
                board.update_target_grid(outcome)
                print(board.color_str('target'))
            print(f"Open hits: {player.offense.open_hits()}")
            # The below call to target sets the target_probs property up to turn
            player.offense.target(board, player.outcome_history[:turn])
            print(f"Possible targets: {player.offense.target_probs}")
            print(f"Target: {outcome['coord']}")
            print(f"Outcome: {result}")
            
    @staticmethod
    def plot_results_summary(results):
        """
        Shows a 3x3 grid of plots with statistical summaries of the game 
        results (dicts) contained in the results parameter.

        Parameters
        ----------
        results : list
            A list of dictionaries, each of which should contain the following
            keys:
                'winner' :              Player instance (or None if a tie)
                'loser' :               Player instance (or None if a tie)
                'tie' :                 Bool, true if game was a tie
                'first_move':           int, 1 or 2
                'player1' :             The game's first player (not 
                                        necessarily the one that went first)
                'player2' :             The game's second player (not 
                                        necessarily the one that went second)
                'turn_count' :          int, the number of turns played
                'max_turns' :           int, the max # of turns allowed
                'game_id' :             An int or string identifying the game
                'duration' :            The time in seconds that elapsed while
                                        the game was played
                'player2_stats' :       A dictionary with statistics for 
                                        player 2. Keys are 'nshots', 
                                        'hits', 'ships_sank', 'shots_mean', 
                                        'shots_var', 'ships_mean', 'ships_var'.
                'player2_stats' :       See above, but for player 2.
                
            Properly formatted dictionaries are stored in a Game instance's
            'game_results' property.

        Returns
        -------
        None
        
        """
        plot_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        p1_color = plot_colors[0]
        p2_color = plot_colors[1]
        tie_color = plot_colors[4]
        
        winner = []         # the winner (1 for player1, 2 for player2, 0 for tie)
        nshots = []         # the # of shots taken
        hits_percent = []   # the % of shots that were a hit
        ships_var = []      # the variance in distance between ships
        loser_sinks = []    # the ships sunk by the loser (ngames x 5; [n,i] is True if ship i in game n was sunk, False otherwise)
        p1_sinks = []
        p2_sinks = []
        
        for result in results:
            stats1 = result['player1_stats']
            stats2 = result['player2_stats']
            
            winner += [1 if result['winner'] == result['player1']
                       else 2 if result['winner'] == result['player2']
                       else 0]
            nshots += [(stats1['nshots'],
                        stats2['nshots'])]
            hits_percent += [(np.mean(stats1['hits']),
                              np.mean(stats2['hits']))]
            ships_var += [(stats1['ships_var'],
                           stats2['ships_var'])]
            loser_stats = (stats1 if result['loser'] is result['player1']
                           else stats2 if result['loser'] is result['player2']
                           else None)
            sinks = np.zeros(5, dtype=bool)
            
            if loser_stats:
                if (len(loser_stats['ships_sank']) > 0):
                    sinks[loser_stats['ships_sank'] - 1] = True
            loser_sinks += [sinks]
            
            sinks = np.zeros(5, dtype=bool)
            if len(stats1['ships_sank']) > 0:
                sinks[stats1['ships_sank'] - 1] = True
            p1_sinks += [sinks]
            
            sinks = np.zeros(5, dtype=bool)
            if len(stats2['ships_sank']) > 0:
                sinks[stats2['ships_sank'] - 1] = True
            p2_sinks += [sinks]
            
            
        winner = np.array(winner)
        nshots = np.array(nshots)
        hits_percent = np.array(hits_percent)
        ships_var = np.array(ships_var)
        loser_sinks = np.array(loser_sinks)
        p1_sinks = np.array(p1_sinks)
        p2_sinks = np.array(p2_sinks)
        
        turn_count = np.array([r['turn_count'] for r in results])
        duration = np.array([r['duration'] for r in results])
        
        fig, axs = plt.subplots(4, 3, figsize = (12,12))
        
        # Upper Left
        pos = (0,0)
        #axs[pos].plot(winner)
        p1_wins = sum(winner == 1)
        p2_wins = sum(winner == 2)
        ties = sum(winner == 0)
        player_names = [results[-1]['player1'].name, 
                        'Tie', 
                        results[-1]['player2'].name]
        offset = max((p1_wins, p2_wins, ties)) * 0.02
        axs[pos].bar(player_names, 
                     [p1_wins, ties, p2_wins],
                     color = [p1_color, tie_color, p2_color])
        axs[pos].set_title('Winner')
        axs[pos].text(-0.35, p1_wins - offset,
                      f"{p1_wins / len(results) * 100:.1f}%\n",
                      verticalalignment='top', color='white')
        axs[pos].text(1.65, p2_wins - offset,
                      f"{p2_wins / len(results) * 100:.1f}%\n",
                      verticalalignment='top', color='white')
        axs[pos].text(0.65, ties,
                      f"{ties / len(results) * 100:.1f}%",
                      verticalalignment='bottom')
        
        # Upper Middle
        pos = (0,1)
        round_to = 10
        nbins = 20
        bins = (np.floor(turn_count.min()/round_to) * round_to,
                np.ceil(turn_count.max()/round_to) * round_to)
        bins = np.arange(bins[0], bins[1], (bins[1]-bins[0])/(nbins+1))
                         
        axs[pos].hist(turn_count, bins, color='gray', alpha=0.7)
        axs[pos].hist(turn_count[winner==1], bins, color=p1_color, alpha=0.7)
        axs[pos].hist(turn_count[winner==2], bins, color=p2_color, alpha=0.7)
        axs[pos].set_title('Turn Count')
        xlim = axs[pos].get_xlim()
        ylim = axs[pos].get_ylim()
        axs[pos].text(min(xlim) + np.diff(xlim) * 0.02, 
                      max(ylim) - np.diff(ylim) * 0.02,
                      f"{np.mean(turn_count):.1f} +/- {np.std(turn_count):.1f}\n"
                      f"Min: {turn_count.min()}\n"
                      f"Median: {np.median(turn_count):.1f}\n"
                      f"Max: {turn_count.max()}",
                      verticalalignment = "top")
        
        # Upper Right
        pos = (0,2)
        axs[pos].hist(duration)
        axs[pos].set_title('Duration (seconds)')
        xlim = axs[pos].get_xlim()
        ylim = axs[pos].get_ylim()
        axs[pos].text(min(xlim) + np.diff(xlim) * 0.02, 
                      max(ylim) - np.diff(ylim) * 0.02,
                      f"{np.mean(duration):.2f} +/- {np.std(duration):.2f}\n"
                      f"Min: {duration.min():.2f}\n"
                      f"Median: {np.median(duration):.2f}\n"
                      f"Max: {duration.max():.2f}",
                      verticalalignment = "top")
        
    
        
        # Middle Left
        pos = (1,0)
        h1 = axs[pos].plot(np.arange(1,6), np.mean(p1_sinks, axis=0),
                           'o')[0]
        axs[pos].plot(np.arange(1,6), np.mean(p1_sinks[winner==2,:],axis=0),
                           'v', color=h1.get_c())
        h2 = axs[pos].plot(np.arange(1,6), np.mean(p2_sinks, axis=0),
                           'o')[0]
        axs[pos].plot(np.arange(1,6), np.mean(p2_sinks[winner==1,:],axis=0),
                           'v', color=h2.get_c())
                      
        axs[pos].set_title('Ships Sank')
        axs[pos].set_xticklabels(['None', 'Patrol', 'Dest', 
                                  'Sub', 'Bship', 'Carrier'],
                                 rotation = 45, fontsize=8)
        axs[pos].legend(['All games', 'Losses only'], fontsize=10)
        
        # Middle Middle
        pos = (1,1)
        axs[pos].hist(np.sum(loser_sinks, axis=1), 
                      bins = np.array((0,1,2,3,4,5,6)) - 0.5)
        axs[pos].set_xticks((0,1,2,3,4,5))
        axs[pos].set_title('# Ships Sunk by Loser')
        xlim = axs[pos].get_xlim()
        ylim = axs[pos].get_ylim()
        # axs[pos].text(min(xlim) + np.diff(xlim) * 0.02, 
        #               max(ylim) - np.diff(ylim) * 0.02,
        #               f"{np.mean(duration):.2f} +/- {np.std(duration):.2f}\n"
        #               f"Min: {duration.min():.2f}\n"
        #               f"Median: {np.median(duration):.2f}\n"
        #               f"Max: {duration.max():.2f}",
        #               verticalalignment = "top")
        
        # Middle Right
        games_per_group = 10
        # number of experiments, games per experiment
        grouping = (int(len(results) / games_per_group), games_per_group)     
        
        win_matrix = winner.reshape(grouping)
        p1_wins = np.sum(win_matrix == 1, axis=1)
        p2_wins = np.sum(win_matrix == 2, axis=1)
        pos = (1,2)
        axs[pos].hist(p1_wins, bins = np.arange(grouping[1]+1), alpha=0.7)
        axs[pos].hist(p2_wins, bins = np.arange(grouping[1]+1), alpha=0.7)
        axs[pos].set_title(f"Wins per {grouping[1]} games")
        xlim = axs[pos].get_xlim()
        ylim = axs[pos].get_ylim()
        
        # Lower Left
        pos = (2,0)
        turns_btwn_hits_1 = []
        turns_btwn_hits_2 = []
        for result in results:
            hits1 = result['player1_stats']['hits']
            hits2 = result['player2_stats']['hits']
            turns_btwn_hits_1 += list(np.diff(np.flatnonzero(hits1 == True)))
            turns_btwn_hits_2 += list(np.diff(np.flatnonzero(hits2 == True)))
           
        axs[pos].hist(turns_btwn_hits_1, 
                      bins = np.arange(grouping[1]+1), alpha=0.7, density=True)
        axs[pos].hist(turns_btwn_hits_2, 
                      bins = np.arange(grouping[1]+1), alpha=0.7, density=True)
        axs[pos].set_title("Turns between hits")
        
        xlim = axs[pos].get_xlim()
        ylim = axs[pos].get_ylim()
        axs[pos].text(max(xlim) - np.diff(xlim) * 0.02, 
                      max(ylim) - np.diff(ylim) * 0.02,
                      f"{results[-1]['player1'].name}\n"
                      f"{np.mean(turns_btwn_hits_1):.1f} +/- "
                      f"{np.std(turns_btwn_hits_1):.1f}\n"
                      f"Min: {min(turns_btwn_hits_1)}\n"
                      f"Median: {np.median(turns_btwn_hits_1):.1f}\n"
                      f"Max: {max(turns_btwn_hits_1)}\n"
                      "\n"
                      f"{results[-1]['player2'].name}\n"
                      f"{np.mean(turns_btwn_hits_2):.1f} +/- "
                      f"{np.std(turns_btwn_hits_2):.1f}\n"
                      f"Min: {min(turns_btwn_hits_2)}\n"
                      f"Median: {np.median(turns_btwn_hits_2):.1f}\n"
                      f"Max: {max(turns_btwn_hits_2)}",
                      verticalalignment = "top",
                      horizontalalignment = "right")
        
        # Lower Middle
        pos = (2,1)
        turns_hit_to_sink_1 = []
        turns_hit_to_sink_2 = []
        for result in results:
            hits1 = np.flatnonzero(result['player1_stats']['hits'])
            hits2 = np.flatnonzero(result['player2_stats']['hits'])
            sinks1 = result['player1_stats']['turn_ship_sank']
            sinks2 = result['player2_stats']['turn_ship_sank']
            sinks1 = sorted([s for s in sinks1 if s >= 0])
            sinks2 = sorted([s for s in sinks2 if s >= 0])
            while sinks1:
                sink_turn = sinks1.pop(0)
                turns_hit_to_sink_1 += [sink_turn - hits1[0]]
                hits1 = hits1[hits1 > sink_turn]
            while sinks2:
                sink_turn = sinks2.pop(0)
                turns_hit_to_sink_2 += [sink_turn - hits2[0]]
                hits2 = hits2[hits2 > sink_turn]
           
        axs[pos].hist(turns_hit_to_sink_1, 
                      bins = np.arange(grouping[1]+1), alpha=0.7, density=True)
        axs[pos].hist(turns_hit_to_sink_2, 
                      bins = np.arange(grouping[1]+1), alpha=0.7, density=True)
        axs[pos].set_title("Turns from hit to sink")
        
        xlim = axs[pos].get_xlim()
        ylim = axs[pos].get_ylim()
        axs[pos].text(max(xlim) - np.diff(xlim) * 0.02, 
                      max(ylim) - np.diff(ylim) * 0.02,
                      f"{results[-1]['player1'].name}\n"
                      f"{np.mean(turns_hit_to_sink_1):.1f} +/- "
                      f"{np.std(turns_hit_to_sink_1):.1f}\n"
                      f"Min: {min(turns_hit_to_sink_1)}\n"
                      f"Median: {np.median(turns_hit_to_sink_1):.1f}\n"
                      f"Max: {max(turns_hit_to_sink_1)}\n"
                      "\n"
                      f"{results[-1]['player2'].name}\n"
                      f"{np.mean(turns_hit_to_sink_2):.1f} +/- "
                      f"{np.std(turns_hit_to_sink_2):.1f}\n"
                      f"Min: {min(turns_hit_to_sink_2)}\n"
                      f"Median: {np.median(turns_hit_to_sink_2):.1f}\n"
                      f"Max: {max(turns_hit_to_sink_2)}",
                      verticalalignment = "top",
                      horizontalalignment = "right")
        
        # Lower Right
        pos = (2,2)
        last_winner = -1
        streak_length = -1
        streaks = [(last_winner, streak_length)]
        for win in winner:
            if win == streaks[-1][0]:
                streak_length += 1
            else:
                streaks += [(last_winner, streak_length)]
                last_winner = win
                streak_length = 1
                
        streaks1 = np.array([x[1] for x in streaks if x[0] == 1])
        streaks2 = np.array([x[1] for x in streaks if x[0] == 2])
        axs[pos].hist(streaks1, color=p1_color, alpha=0.7)
        axs[pos].hist(streaks2, color=p2_color, alpha=0.7)
        
        axs[pos].set_title("Winning Streaks")
        
        xlim = axs[pos].get_xlim()
        ylim = axs[pos].get_ylim()
        axs[pos].text(max(xlim) - np.diff(xlim) * 0.02, 
                      max(ylim) - np.diff(ylim) * 0.02,
                      f"{results[-1]['player1'].name}\n"
                      f"{np.mean(streaks1):.1f} +/- "
                      f"{np.std(streaks1):.1f}\n"
                      f"Min: {min(streaks1)}\n"
                      f"Median: {np.median(streaks1):.1f}\n"
                      f"Max: {max(streaks1)}\n"
                      "\n"
                      f"{results[-1]['player2'].name}\n"
                      f"{np.mean(streaks2):.1f} +/- "
                      f"{np.std(streaks2):.1f}\n"
                      f"Min: {min(streaks2)}\n"
                      f"Median: {np.median(streaks2):.1f}\n"
                      f"Max: {max(streaks2)}",
                      verticalalignment = "top",
                      horizontalalignment = "right")
        
        # Lower Left
        pos = (3,0)
        # correlation between first hit and winning
        first_hit_1 = []
        first_hit_2 = []
        for result in results:
            hits = result['player1_stats']['hits']
            first_hit = np.flatnonzero(hits)[0] if any(hits) else -1
            first_hit_1 += [first_hit]
            hits = result['player2_stats']['hits']
            first_hit = np.flatnonzero(hits)[0] if any(hits) else -1
            first_hit_2 += [first_hit]
        first_hit_1 = np.array(first_hit_1)
        first_hit_2 = np.array(first_hit_2)
        p1_wins = Counter(first_hit_1[winner==1])
        p1_losses = Counter(first_hit_1[winner==2])
        p2_wins = Counter(first_hit_2[winner==2])
        p2_losses = Counter(first_hit_2[winner==1])
        
        bins = (min((min(p1_wins.keys()), min(p1_losses.keys()),
                     min(p2_wins.keys()), min(p2_losses.keys()))),
                max((max(p1_wins.keys()), max(p1_losses.keys()),
                             max(p2_wins.keys()), max(p2_losses.keys()))))
        bins = np.arange(bins[0], bins[1] + 1)
        p1_total = p1_wins.copy()
        p2_total = p2_wins.copy()
        for first_hit in bins:
            p1_total[first_hit] += p1_losses[first_hit]
            p2_total[first_hit] += p2_losses[first_hit]
        
        p1_percent = [p1_wins[k]/p1_total[k] if p1_total[k] > 0 else 0
                      for k in p1_total.keys()]
        p2_percent = [p2_wins[k]/p2_total[k] if p2_total[k] > 0 else 0
                      for k in p2_total.keys()]
        axs[pos].bar(bins, p1_percent, color=p1_color, alpha=0.7)
        axs[pos].bar(bins, p2_percent, color=p2_color, alpha=0.7)
