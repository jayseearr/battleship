#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 21:31:10 2022

@author: jason
"""

import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt

from battleship import TargetValue, SHIP_DATA, Board, Ship

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
    def pegs(xy, color, edgecolor=None):
        if edgecolor == None:
            edgecolor = Viz.peg_outline_color
        return [plt.Circle(col_row, radius=Viz.peg_radius, facecolor=color,
                    edgecolor=edgecolor) for col_row in xy]
    
    @staticmethod
    def show_player(player, show_possible_targets=True):
        
        fig, axs = plt.subplots(2, 1, figsize = (6,10), squeeze=True)
        board_size = player.board.size
        grid_extent = [1-0.5, board_size+0.5, board_size+0.5, 1-0.5]
        colmap = mpl.colors.ListedColormap([Viz.ocean_color, Viz.ocean_color])
        
        for ax in axs.flatten():
            Viz.format_axes(ax, board_size)

        axs[0].imshow(np.zeros(player.board.target_grid.shape), cmap=colmap, 
                      extent=grid_extent)        
        # add pegs to target grid
        rows,cols = np.where(player.board.target_grid == TargetValue.MISS)
        for p in Viz.pegs(zip(cols+1,rows+1), Viz.miss_color):
            axs[0].add_patch(p)
        rows,cols = np.where(player.board.target_grid == TargetValue.HIT)
        for p in Viz.pegs(zip(cols+1,rows+1), Viz.hit_color):
            axs[0].add_patch(p)           
        if (show_possible_targets and 
                getattr(player.offense,'possible_targets', False)):
            cols_rows = [(c+1,r+1) for (r,c) in player.offense.possible_targets]
            for p in Viz.pegs(cols_rows, None, Viz.target_color):
                axs[0].add_patch(p)
            col_row = (player.last_target[1]+1, player.last_target[0]+1)
            axs[0].add_patch(Viz.pegs([col_row], None, "purple")[0])
                
        axs[1].imshow(np.zeros(player.board.target_grid.shape), cmap=colmap, 
                      extent=grid_extent)        
        # add pegs to ships
        rows,cols = np.where(player.board.ocean_grid_image() == 2)
        for p in Viz.pegs(zip(cols+1,rows+1), Viz.hit_color):
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
                           SHIP_DATA[shipId]["name"][0])

        fig.set_facecolor(Viz.bg_color)
        return fig
    
    @staticmethod
    def show_boards(board1, board2, grid="both", possible_targets=None):
        
        if grid.lower() == "both":
            grid = ["target", "ocean"]
        else:
            grid = [grid.lower()]
        if (grid[0] not in ["target", "ocean"] or 
                grid[-1] not in ["target", "ocean"]):
            raise ValueError("grid must be 'target', 'ocean', "
                             "or 'both' (default).")
            
        bg_color = "#202020"
        ship_outline_color = "#202020"
        peg_outline_color = "#202020"
        ship_color = "darkgray"
        target_color = "yellow"

        ocean_color = 'tab:blue' #cadetblue
        miss_color = "whitesmoke"
        hit_color = "tab:red" #"firebrick"

        peg_radius = 0.3
        
        cmap_flat = mpl.colors.ListedColormap([ocean_color, ship_color, 
                                               ship_color])
        cmap_vert = mpl.colors.ListedColormap([ocean_color, miss_color, 
                                               hit_color])

        if len(grid) == 2:
            fig, axs = plt.subplots(2, 2, figsize = (10,12), squeeze=False)
        else:
            fig, axs = plt.subplots(1, 2, figsize = (10,6), squeeze=False)

        flat_grids = board1.ocean_grid_image(), board2.ocean_grid_image()
        vert_grids = board1.target_grid_image(), board2.target_grid_image()
        board_size = np.max((board1.size, board2.size))
        grid_extent = [1-0.5, board_size+0.5, board_size+0.5, 1-0.5]
        
        for ax in axs.flatten():
            Viz.format_axes(ax, board_size)

        if "target" in grid:
            print(axs)
            for c in [0,1]:
                axs[0,c].imshow(np.zeros(vert_grids[c].shape), cmap=cmap_vert, 
                                     extent=grid_extent)
                # add pegs to target grid
                rows,cols = np.where(vert_grids[c] == TargetValue.MISS)
                for (x,y) in zip(cols+1,rows+1):
                    axs[0,c].add_patch(
                        plt.Circle((x,y), radius=peg_radius, 
                                   facecolor = miss_color, 
                                   edgecolor = peg_outline_color) 
                        ) 
                # red pegs    
                rows,cols = np.where(vert_grids[c] >= TargetValue.HIT)
                for (x,y) in zip(cols+1,rows+1):
                    axs[0,c].add_patch(
                        plt.Circle((x,y), radius=peg_radius, 
                                   facecolor = hit_color, 
                                   edgecolor = peg_outline_color) 
                        ) 
                    
                if possible_targets:
                    for t in possible_targets:
                        print(t[0])
                        axs[0,c].add_patch(
                            plt.Circle((t[0]+1,t[1]+1), radius=peg_radius*1.2,
                                       edgecolor = target_color)
                            )
                
        if "ocean" in grid:
            for c in [0,1]:
                board = board1 if c==0 else board2 if c==1 else None
                axs[-1,c].imshow(np.zeros(flat_grids[c].shape), cmap=cmap_flat, 
                                 extent=grid_extent)
                # add pegs to ships
                rows,cols = np.where(flat_grids[c] == 2)
                red_pegs = [plt.Circle((x,y), radius=peg_radius, 
                                       facecolor = hit_color, 
                                       edgecolor = peg_outline_color) 
                            for (x,y) in zip(cols+1,rows+1)]
                for peg in red_pegs:
                    axs[-1,c].add_patch(peg)
                # ships
                ship_boxes = board.ship_rects()
                for shipId in ship_boxes:
                    box = ship_boxes[shipId]
                    axs[-1,c].add_patch( 
                        mpl.patches.Rectangle((box[0], box[1]), box[2], box[3], \
                                              edgecolor = ship_outline_color,
                                              fill = False,                                       
                                              linewidth = 2))
                    axs[-1,c].text(box[0]+0.12, box[1]+0.65, 
                                   SHIP_DATA[shipId]["name"][0])

        fig.set_facecolor(bg_color)
        return fig
    
    @staticmethod
    def format_axes(ax, board_size):
        
        label_color = "lightgray"
        grid_color = "gray"
        axis_color = "lightgray"
        #bg_color = "#202020"
        #ship_outline_color = "#202020"
        #peg_outline_color = "#202020"
        #ship_color = "darkgray"
        #target_color = "yellow"

        ocean_color = 'tab:blue' #cadetblue
        #miss_color = "whitesmoke"
        #hit_color = "tab:red" #"firebrick"
        
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

        peg_outline_color = "#202020"
        miss_color = "whitesmoke"
        hit_color = "tab:red" #"firebrick"
        sink_color = "red"
        bg_color = "#202020"
        peg_radius = 0.3
        
        fig, axs = plt.subplots(1, 1, figsize = (6,6), squeeze=True)
        axs = [axs]
        for ax in axs:
            Viz.format_axes(ax, board_size)
            
        peg_color = (sink_color if outcome['sunk_ship_id'] else
                     hit_color if outcome['hit'] else miss_color)
        x,y = outcome['coord']
        axs[0].add_patch(plt.Circle((x+1,y+1), 
                                    radius = peg_radius,
                                    facecolor = peg_color,
                                    edgecolor = peg_outline_color))
        fig.set_facecolor(bg_color)
        return fig
            
    @staticmethod
    def shot_history(player):
        #from time import sleep
        
        #ship_outline_color = "#202020"
        peg_outline_color = "#202020"
        #ship_color = "darkgray"
        #target_color = "yellow"

        #ocean_color = 'tab:blue' #cadetblue
        miss_color = "whitesmoke"
        hit_color = "tab:red" #"firebrick"
        sink_color = "red"
        peg_radius = 0.3
        
        fig, axs = plt.subplots(2, 1, figsize = (6,10), squeeze=True)
        for ax in axs:
            Viz.format_axes(ax, player.board.size)
            
        for outcome in player.outcome_history:
            peg_color = (sink_color if outcome['sunk_ship_id'] else
                         hit_color if outcome['hit'] else miss_color)
            x,y = outcome['coord']
            axs[0].add_patch(plt.Circle((x+1,y+1), 
                                        radius = peg_radius,
                                        facecolor = peg_color,
                                        edgecolor = peg_outline_color))
            #sleep(0.5)
        return fig
        
    @staticmethod
    def show_probs(targets, probs=None, board_size=10):
        if probs == None and isinstance(targets, dict):
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
    def plot_placements(placements, board_size=10):
        """
        Shows an image with the placements highlighted.

        Parameters
        ----------
        placements : dict
            Dictionary with keys equal to ShipType (or int) values. The
            corresponding value for each key is a dict with keys 'coord' and
            'heading' that represent the location and direction of the
            ship of corresponding ShipType.
            
        board_size : TYPE, optional
            DESCRIPTION. The default is 10.

        Returns
        -------
        None.

        """
        b = Board(board_size)
        X = np.zeros((b.size,b.size))
        for (k,p) in placements.items():
            ship = Ship(k)
            ri = b.relative_coords_for_heading(p['heading'], ship.length)
            X[p['coord'][0] + ri[0], p['coord'][1] + ri[1]] = k
        plt.imshow(X)
        
    def plot_possible_placements(board, ship_type):
        if isinstance(ship_type, Ship):
            ship = ship_type
        else:
            ship = Ship(ship_type)
            
        pass
        
    