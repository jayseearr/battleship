#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 21:00:32 2023

@author: jason

Class definition for Game object

"""

#%%

### Imports ###

from copy import deepcopy
import numpy as np
import time
# from matplotlib import pyplot as plt
# from collections import Counter

# Import from this package
from battleship.coord import Coord
from battleship.ship import Ship


#%%

### Class Definition ###


class Game:
    
    def __init__(self, 
                 player1, 
                 player2, 
                 salvo = False,
                 game_id = None, 
                 verbose = True,
                 show = False):
        """
        Creates a Battleship game with two players described by the 
        player1 and player2 parameters, and options set by the following
        parameters.

        The game should not care whether a player is human or robot (so don't
        have it access self.player1.type, for example).
        
        A game is played by creating the Game instance, calling game.setup(),
        then game.play().
        
        After a game is completed, another can be played by calling:
            game.reset(), game.setup(), game.play().
            
        Parameters
        ----------
        player1 : Player
            An instance of a subclass of Player.
        player2 : Player
            A second instance of a subclass of Player.
        salvo : bool, optional
            If True, salvo rules are used. If false, standard single-shot rules
            are used for the game. The default is False. 
            *** SALVO RULES ARE NOT YET IMPLEMENTED ***
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
        self.salvo = salvo
        self.turn_count = 0
        self.verbose = verbose
        self.show = show
        self._game_result = None
        self._results_log = []
        
        if not self.player1.name:
            self.player1.name = "Player 1"
        if not self.player2.name:
            self.player2.name = "Player 2"
        if self.player1.name == self.player2.name:
            raise ValueError("Player 1 and Player 2 cannot have the same name.")
            
            
    def __repr__(self):
        return (f"Game({self._player1!r},\n {self._player2!r},\n "
                f"salvo={self.salvo!r}, game_id={self.game_id!r}, "
                f"verbose={self.verbose!r}, show={self.show!r})")
    
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
    
    @property
    def salvo(self):
        return self._salvo
    
    @salvo.setter
    def salvo(self, value):
        if not isinstance(value, bool):
            raise TypeError("salvo must be True or False.")
        if value:
            raise Warning("Salvo rules are not yet implemented. "
                          "Setting salvo property to False.")
            value = False
        self._salvo = value
        
    @property
    def verbose(self):
        return self._verbose
    
    @verbose.setter
    def verbose(self, value):
        if not isinstance(value, bool):
            raise TypeError("verbose must be True or False.")
        self._verbose = value
    
    @property
    def show(self):
        return self._show
    
    @show.setter
    def show(self, value):
        if not isinstance(value, bool):
            raise TypeError("show must be True or False.")
        self._show = value
        
    @property
    def game_result(self):
        return self._game_result
    
    @property
    def results_log(self):
        return self._results_log
    
    # Other methods 
    
    def setup(self):
        """Set up a game by placing both fleets."""
        if ((self.player1.board and self.player1.board.fleet) or 
            (self.player2.board and self.player2.board.fleet)):
            raise ValueError("Boards need to be reset; call reset() method.")
            
        self.player1.opponent = self.player2
        self.player1.prepare_fleet()
        self.player2.prepare_fleet()
        
        self.winner = None
        self.loser = None
        self.turn_count = 0
        self.ready = (self.player1.board.is_ready_to_play() and 
                      self.player2.board.is_ready_to_play())
    
    def reset(self, setup=False):
        """
        Resets the players (which typically would include reseting their
        boards and offense, if applicable), turn count, and the winner/losers
        so that a new game can be played.
        
        Optially, sets up the players boards for a new game, using their 
        appropriate defense methods.

        Parameters
        ----------
        setup : bool, optional
            If True, the setup method will be called after resetting the 
            players. This will let a new game being immediately after
            this method is called. The default is False.

        Returns
        -------
        None.

        """
        self.player1.reset()
        self.player2.reset()
        self.ready = False
        self.winner = None
        self.loser = None
        self.turn_count = 0
        if setup:
            self.setup()
     
    # Gameplay methods 
    
    def play_one_turn(self, first_player, second_player,
                      first_player_target=None,
                      second_player_target=None):
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
        if not self.ready:
            raise ValueError("Game needs to be setup before a turn "
                             "can be played. See setup() method.")
        if not first_player.is_alive():
            raise Warning("First player's ships are all sunk.")
        if not second_player.is_alive():
            raise Warning("Second player's ships are all sunk.")
        first_player.take_turn(first_player_target)
        if self.verbose:
            self.report_turn_outcome(first_player, True)
        if second_player.is_alive():
            second_player.take_turn(second_player_target)
            if self.verbose:
                self.report_turn_outcome(second_player)
        self.turn_count += 1
        if self.show:
            first_player.board.show()
            second_player.board.show()
        return (first_player.is_alive() and 
                second_player.is_alive())
        
    def play(self, first_move=1, max_turns=None, game_id=None):
        """
        Play one game of Battleship. Each player takes a turn until one
        of them has no ships remaining, or until the maximum number of turns
        is reached.

        Parameters
        ----------
        first_move : int or str, optional
            Indicates which player will take the first shot. Can be 1 (player1
            goes first), 2 (player 2 goes first), or '?' (randomly select which
            player will go first). The default is 1.
        max_turns : int, optional
            The maximum number of turns before the game is called a tie. 
            The default is 1 million; by this point, something has likely gone
            wrong with target selection.
        game_id : str or int, optional
            A unique identifier for this particular game. The default is None,
            in which case the game_id defaults to the game_id set during
            initialization of the Game object (which may also be None).

        Returns
        -------
        dict 
            A dictionary containing the results of the game, stored in the
            following key/value pairs. Note that a turn-by-turn history
            can be found in the outcome_history properties of the two players.
            
            'winner':           The instance of the player that won (None in
                                the case of a tie).
            'loser':            The instance of the player that lost (None in
                                the case of a tie).
            'first_move':       Integer indicating whether player1 or player2
                                went first.
            'first_player':     The instance of the player that went first.
            'second_player':    The instance of the player that went second.
            'turn_count':       The number of turns elapsed
            'max_turns':        The maximum number of turns allowed before
                                the game is called a tie.
            'tie':              True if the game ended in a tie, False otherwise.
            'game_id':          The id of the game played (set using the
                                game_id input, or at Game initialization).
            'duration':         The time in seconds that it took to play the
                                game.
            'first_player_stats'    The game statistics for the first player
                                    (shots, hits, ships sank, etc. See
                                     Player.stats()).
            'second_player_stats'   Game stats for second player.      
        
        """
        start_time = time.perf_counter()
        timestamp = time.time()
        
        if max_turns == None:
            max_turns = 1e6
        
        if game_id != None:
            self.game_id = game_id
            
        # Choose which player goes first
        if (isinstance(first_move, str) and 
                first_move == "?" or first_move == "random"):
            first_move = np.random.randint(1,3)
        first_player, second_player = None, None
        if first_move == 1:
            first_player, second_player = self.player1, self.player2
        elif first_move == 2:
            first_player, second_player = self.player2, self.player1
        if first_player == None or second_player == None:
            raise ValueError("Could not determine which player should go first.")
            
        # Check that players/boards/fleets are ready
        if not self.ready:
            print("Game setup not complete.")
            return {'winner': None,
                    'loser': None, 
                    'tie': False,
                    'first_move': first_move,
                    'player1': first_player,
                    'player2': second_player,
                    'player1_history': [],
                    'player2_history': [],
                    'turn_count': 0,
                    'max_turns': max_turns,
                    'game_id': self.game_id,
                    'duration': time.perf_counter() - start_time,
                    'timestamp': timestamp,
                    'player1_stats': {},
                    'player2_stats': {}
                    }
        
        # Play until one player has lost all ships
        game_on = True
        self.turn_count = 0
        while game_on:
            still_playing = self.play_one_turn(first_player, second_player)
            game_on = still_playing and self.turn_count < max_turns
        self.ready = False
        
        # See who won
        tie = False
        if first_player.is_alive() and not second_player.is_alive():
            self.winner, self.loser = first_player, second_player
        elif second_player.is_alive() and not first_player.is_alive():
            self.winner, self.loser = second_player, first_player
        elif self.turn_count >= max_turns:
            tie = True
        else: # players sunk fleet on same turn
            tie = True
        
        if self.verbose:
            self.print_outcome()
            
        self._game_result = {'winner': self.winner,
                             'loser': self.loser, 
                             'tie': tie,
                             'first_move': first_move,
                             'player1': self.player1,
                             'player2': self.player2,
                             'player1_history': 
                                 deepcopy(self.player1.outcome_history),
                             'player2_history': 
                                 deepcopy(self.player2.outcome_history),
                             'turn_count': self.turn_count,
                             'max_turns': max_turns,
                             'game_id': self.game_id,
                             'duration': time.perf_counter() - start_time,
                             'timestamp': timestamp,
                             'player1_stats': self.player1.stats(),
                             'player2_stats': self.player2.stats()
                             }
        self._results_log += [self.game_result]
        
        # add result to players' game histories
        self.player1.add_result_to_history(self._game_result)
        self.player2.add_result_to_history(self._game_result)
        
        return self.game_result
         
    def play_games(self, ngames):
        """
        Plays multiple games, and keeps track of the results.

        Parameters
        ----------
        ngames : int
            The number of games to play.

        Returns
        -------
        results : list
            A list of outcome dictionaries (see the play method description).

        """
        
        
        results = []
        init_verbose = self.verbose
        self.verbose = False
        if not self.ready:
            self.reset()
        self.setup()
        for count in range(ngames):
            result = self.play(first_move="?", 
                               max_turns=100, 
                               game_id=count+1)
            results += [result]
            self.reset(setup=True)
            if result['tie']:
                print(f"Tie game ({results[-1]['turn_count']} turns)")
            else:                
                print(f"Game {count + 1} of {ngames} comleted. "
                      f"{results[-1]['winner'].name} won in "
                      f"{results[-1]['turn_count']} turns.")
            
        self.verbose = init_verbose
        return results
    
    def report_turn_outcome(self, player, include_turn_num=False):
        """Displays text reporting the target and outcome on the most recent
        turn for the input player.
        """
        
        hit_clr = "\x1b[1;31m"
        miss_clr = "\x1b[1;36m"
        sink_clr = "\x1b[1;37;41m"
        name_clr = "\x1b[1;35m"
        #game_over_clr = "\x1b[1;35m"
        reset_clr = "\x1b[0;0m"
        target_clr = "\x1b[1;33m"
        
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
            
        if include_turn_num:
            print(f"--- Turn {len(player.shot_history)} ---")
            
        if not name:
            name = "?"
        
        if hit_or_miss.lower()[0] == "h":
            hit_or_miss = f"{hit_clr}{hit_or_miss}{reset_clr}"
        elif hit_or_miss.lower()[0] == "m":
            hit_or_miss = f"{miss_clr}{hit_or_miss}{reset_clr}"
        else:
            raise ValueError("Invalid value for hit_or_miss.")
            
        outcome_str = (f"{name_clr}{name}{reset_clr} fired a shot at "
                       f"{target_clr}{Coord(target)}{reset_clr}..."
                       f"{hit_or_miss}")
        if sink:
            outcome_str += f"\n  {sink_clr}{sink}{reset_clr}"
        
        print(outcome_str)
        
        # print(f"Turn {len(player.shot_history)}: Player {name} fired a shot at "
        #       f"{target_clr}{Coord(target)}{reset_clr}...{hit_or_miss} {sink}")
        
    def print_outcome(self):
        """Prints the results of the game for the input winner/loser players 
        (which include their respective boards)."""

        if self.winner:
            print("")
            print("GAME OVER!")
            print(f"{self.winner.name} wins.")
            print(f"    {self.winner.name} took "
                  f"{len(self.winner.shot_history)} shots, and sank " 
                  f"{sum(1 - np.array(self.loser.board.is_fleet_afloat()))} ships.")
            print(f"    {self.loser.name} took " 
                  f"{len(self.loser.shot_history)} shots, and sank " 
                  f"{sum(1 - np.array(self.winner.board.is_fleet_afloat()))} ships.")
            print(f"Game length: {self.turn_count} turns.")
            print(f"(Game ID: {self.game_id})")
            print("")
        else:
            player1_ships_sank = sum([1 - afloat for afloat in 
                                      self.player2.board.is_fleet_afloat()])
            player2_ships_sank = sum([1 - afloat for afloat in 
                                      self.player1.board.is_fleet_afloat()])
            print("")
            print("GAME TERMINATED - NO WINNER.")
            print(f"    {self.player1.name} took " 
                  f"{len(self.player1.shot_history)} shots, and sank "
                  f"{player1_ships_sank} ships.")
            print(f"    {self.player2.name} took " 
                  f"{len(self.player2.shot_history)} shots, and sank "
                  f"{player2_ships_sank} ships.")
            print(f"Game length: {self.turn_count} turns.")
            print(f"(Game ID: {self.game_id})")
            print("")
            
    # Factory method
    
    @classmethod
    def quick(cls):
        from battleship.player import AIPlayer
        from battleship.offense import HunterOffense
        from battleship.defense import RandomDefense
        new_game = cls(AIPlayer(HunterOffense("random"), 
                                RandomDefense("random"),
                                name="Frances"), 
                       AIPlayer(HunterOffense("random"), 
                                RandomDefense("random"),
                                name="Albert"))
        return new_game
    
    
    @staticmethod
    def print_results_summary(results):
        """
        Prints a summary of the results parameter, which can be the result
        of a single game, or a list of game results.

        Parameters
        ----------
        results : list or dict
            If a dict, the input should be the game_result property of a Game
            object. It should have the following key/value pairs:
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
                
            If a summary of multiple games is desired, the input should be a 
            list containing dictionaries like the one descried above.

        Returns
        -------
        None

        """
        if isinstance(results, dict):
            if results['tie']:
                winner = "none"
                loser = "none"
                tie_str = "  *** TIE ***\n"
            else:
                winner = results['winner'].name
                loser = results['loser'].name
                tie_str = ""
            p1 = results['player1']
            p2 = results['player2']
            if results['tie']:
                p1_tag, p1_tag = "", ""
            elif winner == p1.name:
                p1_tag = " (WINNER)"
                p2_tag = ""
            else:
                p1_tag = ""
                p2_tag = " (WINNER)"
                
            stats1 = results['player1_stats']
            stats2 = results['player2_stats']
            out = (f"Game {results['game_id']}\n"
                   f"{tie_str}"
                   f"  Winner: {winner}\n"
                   f"  Loser:  {loser}\n\n"
                   f"  {results['turn_count']} turns, "
                   f"{np.round(results['duration'], 2)} seconds.\n"
                   f"  First move: {results['first_move']}\n\n"
                   f"  "
                   f">>> {p1.name} <<<{p1_tag}\n"
                   f"    {stats1['nshots']} shots taken\n"
                   f"    {np.sum(stats1['hits'])} hits / "
                   f"{np.sum(stats1['hits'] == False)} misses\n"
                   f"    Ships sunk: {len(stats1['ships_sank'])} "
                   f"({str(stats1['ships_sank'])})\n\n"
                   f"  "
                   f">>> {p2.name} <<<{p2_tag}\n"
                   f"    {stats2['nshots']} shots taken\n"
                   f"    {np.sum(stats2['hits'])} hits / "
                   f"{np.sum(stats2['hits'] == False)} misses\n"
                   f"    Ships sunk: {len(stats2['ships_sank'])} "
                   f"{str(stats2['ships_sank'])}\n")
            print(out)
        
        elif isinstance(results, list):
            nturns = np.array([r['turn_count'] for r in results])
            duration = np.array([r['duration'] for r in results])
            stats1 = {}
            stats2 = {}
            stats1['hits'] = np.array([np.sum(r['player1_stats']['hits'])
                                       for r in results])
            stats2['hits'] = np.array([np.sum(r['player2_stats']['hits'])
                                       for r in results])
            stats1['ships_sunk'] = np.array(
                [len(r['player1_stats']['ships_sank']) for r in results])
            stats2['ships_sunk'] = np.array(
                [len(r['player2_stats']['ships_sank']) for r in results])
            winner = np.array([1 if r['winner'] == r['player1'] else
                               -1 if r['winner'] == r['player2'] else
                               0 for r in results])
            
            print(f"Games {results[0]['game_id']} to {results[-1]['game_id']}")
            print(f"  {len(results)} games played")
            print("")
            print(f"{results[0]['player1'].name} won "
                  f"{np.sum(winner == 1)} of {len(results)} games "
                  f"({np.sum(winner == 1) / len(winner) * 100:.1f}%).")
            print(f"{results[0]['player2'].name} won "
                  f"{np.sum(winner == -1)} of {len(results)} games "
                  f"({np.sum(winner == -1) / len(winner) * 100:.1f}%).")
            print(f"{np.sum(winner == 0)} games resulted in a tie."
                  f"({np.sum(winner == 0) / len(winner) * 100:.1f}%).")
            print("")
            print(f"# Turns: {np.mean(nturns):.2f} +/- {np.std(nturns):.2f}; "
                  f" Min = {np.min(nturns):.2f}, Median = {np.median(nturns):.1f}, "
                  f"Max = {np.max(nturns):.2f}.")
            print(f"Duration: {np.mean(duration):.2f} +/- {np.std(duration):.2f} s; "
                  f" Min = {np.min(duration):.2f}, Median = {np.median(duration):.2f}, "
                  f"Max = {np.max(duration):.2f}.")
            both_hits = np.array(list(zip(stats1['hits'],stats2['hits'])))
            loser_nhits = np.min(both_hits,axis = 1)
            print("")
            print(f"Loser averaged {np.mean(loser_nhits):.2f} "
                  f"+/- {np.std(loser_nhits):.2f} hits on winner.")
            both_sinks = np.array(list(zip(stats1['ships_sunk'],
                                           stats2['ships_sunk'])))
            loser_sinks = np.min(both_sinks,axis = 1)
            frequency = [np.sum(loser_sinks == n)/len(loser_sinks) 
                         for n in range(1,6)]
            print(f"Loser averaged {np.mean(loser_sinks):.2f} "
                  f"+/- {np.std(loser_sinks):.2f} sinks on winner.")
            print("# Sinks histogram:")
            for n in range(1,6):
                print(f"  {n} ships sunk: {frequency[n-1]*100:.0f}%")
    
        
        
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
