#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 21:00:32 2023

@author: jason

Class definition for Game object

"""

#%%

### Imports ###

import numpy as np

# Import from this package
from .coord import Coord
from .ship import Ship


#%%

### Class Definition ###


class Game:
    
    def __init__(self, player1, player2, 
                 salvo=False,
                 game_id=None, 
                 verbose=True,
                 show=False):
        """
        Creates a Battleship game with two players described by the 
        player1 and player2 parameters, and options set by the following
        parameters.

        The game should not care whether a player is human or robot (so don't
        have it access self.player1.type, for example).
        
        A game is played by creating the Game instance, calling game.setup(),
        then game.play().
        
        Parameters
        ----------
        player1 : Player
            An instance of a subclass of Player.
        player2 : Player
            A second instance of a subclass of Player.
        salvo : bool, optional
            If True, salvo rules are used. If false, standard single-shot rules
            are used for the game. The default is False.
        game_id : str, optional
            A string used to identify this particular game. 
            The default is None.
        verbose : bool, optional
            If True, the output of the game is printed to the console.
            No output is printed otherwise. Verbose should be False if many 
            robot vs. robot games are going to be simulated.
            The default is True.
        show : bool, optional
            If True, the results of each turn are shown on a Pyplot plot. 
            The default is False.

        Returns
        -------
        None.

        """
        
        self._game_id = game_id
        self._player1 = player1
        self._player2 = player2
            
        self.ready = False
        self.winner = None
        self.loser = None
        self.turn_count = 0
        self.verbose = verbose
        self.show = False
        
    def __repr__(self):
        return (f"Game({self._player1!r},\n {self._player2!r},\n "
                f"game_id={self.game_id!r}, verbose={self.verbose!r})")
    
    def __str__(self):
        if self.winner:
            s = f"Completed Game - {self.turn_count} turns.\n"
            if self.game_id:
                s += f"Game ID: {self.game_id}\n"
            s += f"Winner: {self.winner!r}\n"
            s += f"Loser: {self.loser!r}\n"
        else:
            s = "Unplayed Game\n"
            if self.game_id:
                s += f"Game ID: {self.game_id}\n"
            s += f"Player 1: {self.player1!r}\n"
            s += f"Player 2: {self.player2!r}\n"
        return s
        
    @property
    def game_id(self):
        return self._game_id
    
    @game_id.setter
    def game_id(self, value):
        self._game_id = value
        
    @property
    def player1(self):
        return self._player1
    
    @property
    def player2(self):
        return self._player2
    
    # Other methods 
    
    def setup(self):
        """Set up a game by placing both fleets."""
        if ((self.player1.board and self.player1.board.fleet) or 
            (self.player2.board and self.player2.board.fleet)):
            raise Exception("Boards need to be reset; call reset() method.")
        self.player1.opponent = self.player2
        #self.player2.opponent = self.player1
        self.player1.prepare_fleet()
        self.player2.prepare_fleet()
        self.ready = (self.player1.board.ready_to_play() and 
                      self.player2.board.ready_to_play())
    
    def reset(self):
        self.player1.reset()
        self.player2.reset()
        self.ready = False
        self.winner = None
        self.loser = None
        self.turn_count = 0
     
    # Gameplay methods 
    
    def play_one_turn(self, first_player, second_player):
        """
        

        Parameters
        ----------
        first_player : Player subclass
            DESCRIPTION.
        second_player : Player subclass
            DESCRIPTION.

        Returns
        -------
        Bool
            True if the game should continue, False if the game should end
            (i.e., False if one player no longer has any ships afloat).

        """
        first_player.take_turn()
        if self.verbose:
            self.report_turn_outcome(first_player)
        second_player.take_turn()
        if self.verbose:
            self.report_turn_outcome(second_player)
        self.turn_count += 1
        if self.show:
            first_player.board.show()
            second_player.board.show()
        return (first_player.isalive() and 
                second_player.isalive())
        
    def play(self, first_move=1, max_turns=None):
        """Play one game of Battleship. Each player takes a turn until one
        of them has no ships remaining.
        Returns a tuple containing (winner, loser) Player instances.
        """
        
        if max_turns == None:
            max_turns = 1e6
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
            still_playing = self.play_one_turn(first_player, second_player)
            game_on = still_playing and self.turn_count < max_turns
            
        # See who won
        if first_player.isalive() and not second_player.isalive():
            self.winner, self.loser = first_player, second_player
        elif second_player.isalive() and not first_player.isalive():
            self.winner, self.loser = second_player, first_player
        elif self.turn_count >= max_turns:
            pass
        else:
            raise Exception("Winner could not be determined.")
        
        if self.verbose:
            self.print_outcome()
            
        return (self.winner, self.loser)
         
    def report_turn_outcome(self, player):
        """Displays text reporting the target and outcome on the most recent
        turn for the input player.
        """
        outcome = player.last_outcome
        target = player.last_target
        name = player.name
        sink = ""
        if outcome["hit"]:
            hit_or_miss = "Hit!"
        else:
            hit_or_miss = "Miss."
        if outcome["sunk_ship_type"]:
            sink = Ship.data[outcome["sunk_ship_type"]]["name"] + " sunk!"
        print(f"Turn {len(player.shot_history)}: Player {name} fired a shot at "
              f"{Coord(target)}...{hit_or_miss} {sink}")
        
    def print_outcome(self):
        """Prints the results of the game for the input winner/loser players 
        (which include their respective boards)."""

        if self.winner:
            print("")
            print("GAME OVER!")
            print(f"Player {self.winner.name} wins.")
            print(f"  Player {self.winner.name} took "
                  f"{len(self.winner.shot_history)} shots, and sank " 
                  f"{sum(1 - self.loser.board.is_fleet_afloat())} ships.")
            print(f"  Player {self.loser.name} took " 
                  f"{len(self.loser.shot_history)} shots, and sank " 
                  f"{sum(1 - self.winner.board.is_fleet_afloat())} ships.")
            print(f"Game length: {self.turn_count} turns.")
            print(f"(Game ID: {self.game_id})")
            print("")
        else:
            print("")
            print("GAME TERMINATED - NO WINNER.")
            print(f"  Player {self.player1.name} (player 1) took " 
                  f"{len(self.player1.shot_history)} shots, and sank "
                  f"{sum(1 - self.player2.board.is_fleet_afloat())} ships.")
            print(f"  Player {self.player2.name} (player 2) took " 
                  f"{len(self.player2.shot_history)} shots, and sank "
                  f"{sum(1 - self.player1.board.is_fleet_afloat())} ships.")
            print(f"Game length: {self.turn_count} turns.")
            print(f"(Game ID: {self.game_id})")
            print("")
            
    # Factory method
    
    # @classmethod
    # def random(cls, seed=None, game_id=None, verbose=True, show=False):
    #     from player import AIPlayer
    #     from offense import HunterOffense
    #     from defense import RandomDefense
    #     np.random.seed(seed)
    #     off_weights = ['isolated', 'flat', 'center', 'maxprob']
    #     def_weights = ['flat', 'clustered', 'isolated']
    #     p1 = AIPlayer(HunterOffense('random', 
    #                                 weight=random_choice(off_weights)),
    #                   RandomDefense(random_choice(def_weights)), 
    #                   name = "Gwen")
    #     p2 = AIPlayer(HunterOffense('random', 
    #                                 weight=random_choice(off_weights)),
    #                   RandomDefense(random_choice(def_weights)), 
    #                   name = "Sunny")
    #     return Game(p1, p2, game_id, verbose, show)