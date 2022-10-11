#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 22:18:16 2022

@author: jason
"""

import numpy as np
from matplotlib import pyplot as plt
from battleship import Board, Ship, Game
from offense import HunterOffense
from defense import RandomDefense
from player import AIPlayer

def plot_placements(placements):
    b = Board()
    X = np.zeros((b.size,b.size))
    for (k,p) in placements.items():
        ship = Ship(k)
        ri = b.relative_coords_for_heading(p['heading'], ship.length)
        X[p['coord'][0] + ri[0], p['coord'][1] + ri[1]] = k
    plt.imshow(X)

def1 = RandomDefense('random'); 
off1 = HunterOffense('random', weight='flat');
off2 = HunterOffense('random', weight='isolated'); 
#off3 = HunterOffense('random', weight='maxprob'); 
p1=AIPlayer(off1, def1, name="A"); 
p2=AIPlayer(off2, def1, name="B"); 
g = Game(p1, p2, verbose=False)
g.setup()
g.play()