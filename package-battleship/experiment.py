#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 23:55:48 2022

@author: jason
"""

import numpy as np
from battleship import Game

class Experiment:
    
    def __init__(self, lineup, ngames):
        
        self.lineup = lineup
        self.ngames = ngames
        self.game_history = []
        
    def __repr__(self):
        return f"Experiment({self.lineup!r}, {self.ngames})"
    
    def play_next_match(self):
        nplayed = len(self.game_history)
        I,J = np.meshgrid(range(len(self.lineup)),range(len(self.lineup)),
                          indexing='ij')
        I = I.flatten()
        J = J.flatten()
        i1 = I[I <= J]
        i2 = J[I <= J]
        player1 = self.lineup[i1[nplayed + 1]]
        player2 = self.lineup[i2[nplayed + 1]] 

        game = Game(player1, player2, verbose=False)
        game.setup()
        game.play()
        
    def game_outcome(self):
        pass
    
    def save_progress(self):
        pass