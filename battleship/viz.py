#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 21:31:10 2022

@author: jason
"""

import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt

from battleship import constants
#from battleship.board import Board
from battleship.ship import Ship

class Viz:
    
    bg_color = "#202020"
    ship_outline_color = "#202020"
    peg_outline_color = "#202020"
    ship_color = "darkgray"
    target_color = "yellow"
    ocean_color = 'tab:blue' #cadetblue
    miss_color = "whitesmoke"
    hit_color = "tab:red" #"firebrick"
    peg_radius = 0.3
    
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
        rows,cols = np.where(player.board.target_grid 
                             == constants.TargetValue.HIT)
        for p in Viz._pegs(zip(cols+1,rows+1), Viz.hit_color):
            axs[0].add_patch(p)           
        if (show_possible_targets and 
                getattr(player.offense,'possible_targets', False)):
            cols_rows = [(c+1,r+1) for (r,c) in player.offense.possible_targets]
            for p in Viz._pegs(cols_rows, None, Viz.target_color):
                axs[0].add_patch(p)
            col_row = (player.last_target[1]+1, player.last_target[0]+1)
            axs[0].add_patch(Viz._pegs([col_row], None, "purple")[0])
                
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
        if (grid[0] not in ["target", "ocean"] or 
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
            Viz.add_target_grid(axs[0], board)
            if possible_targets:
                for t in possible_targets:
                    axs[0].add_patch(
                        plt.Circle((t[0]+1,t[1]+1), 
                                   radius=Viz.peg_radius*1.2,
                                   edgecolor = Viz.target_color)
                        )
                        
        if "ocean" in grid:
            Viz.add_ocean_grid(axs[-1], board)
                
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
                             other_fill = None):
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
        for coord in coords:
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