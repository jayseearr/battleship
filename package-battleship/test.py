#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 09:20:23 2022

@author: jason
"""
import numpy as np
from battleship import Board, Ship
# b=Board(10); 
# b.target_grid[np.array((2,3,4,5)),3]=0; 
# b.target_grid[4,np.array((3,4,5))]=0; 
# b.target_grid[4,3]=4
# b.possible_ships_grid()

pa = AIPlayer(HunterOffense("random",weight="flat"), RandomDefense("cluster"), 
            name="A"); 
pb = AIPlayer(HunterOffense("random",weight="flat"), RandomDefense("cluster"), 
            name="B"); 
pa.board = Board.copy(p.board)
pb.board = Board.copy(p.board)
pa.opponent = pb
pa.fire_at_target((6,4))
pa.fire_at_target((7,4))
pa.fire_at_target((7,5))
pa.fire_at_target((6,5))
