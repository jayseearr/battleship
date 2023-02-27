#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 09:24:47 2023

@author: jason
"""

from battleship.board import Board
from battleship.player import AIPlayer
from battleship.offense import HunterOffense
from battleship.defense import RandomDefense
from battleship import constants

class TestPlayers:
    
    ### Class variables ###
    #
    # Define options dictionaries here
    
    random_options = {'offense': {'hunt_style': 'random',
                                  'hunt_pattern': 'random',
                                  'hunt_options': {},
                                  'kill_options': {}
                                  }, 
                      'defense': {'formation': "random", 
                                  'method': "weighted",
                                  'edge_buffer': 0, 
                                  'ship_buffer': 0, 
                                  'alignment': constants.Align.ANY
                                  }
                      }
    
    ### Initializer ###
    
    def __init__(self, options1=None, options2=None, nturns=0, prepare=True):
        """
        Generates a pair of test players with preset boards for testing.

        Parameters
        ----------
        options1 : dict, optional
            Player 1 settings dictionary. The default is None, in which case
            random_options are used.
        options2 : dict, optional
            Player 2 settings dictionary. The default is None, in which case 
            options are the same as player 1's.
        nturns: int, optional
            Number of turns to simulate. The default is 0.
        prepare : bool, optional
            If True, fleet will be placed based on defense options. The default
            is True.
        

        Returns
        -------
        None.

        """
        if options1 is None:
            options1 = TestPlayers.random_options
        if options2 is None:
            options2 = options1.copy()
        self.p1 = TestPlayers.player_with_options(options1)
        self.p2 = TestPlayers.player_with_options(options2)

        self.nturns = nturns
        self.turn_count = 0
        
        self.setup()
        if prepare:
            self.p1.prepare_fleet()
            self.p2.prepare_fleet()
            
        if nturns > 0:
            self.run(nturns)
        
        
    def __str__(self):
        
        s = (f"Player 1: {self.p1}\n"
             f"Player 2: {self.p2}\n")
        if self.turn_count > 0:
            s += "Player 1 status:\n"
            s += self.p1.board.status_str()
            s += "\nPlayer 2 status:\n"
            s += self.p2.board.status_str()
        return s
            
            
    ### Other Methods ###
    
    def run(self, nturns=0):
        """
        Runs 

        Parameters
        ----------
        nturns : TYPE, optional
            DESCRIPTION. The default is 0.

        Returns
        -------
        None.

        """
        for _ in range(nturns):
            self.p1.fire_at_target(
                self.p1.offense.target(self.p1.board,
                                       self.p1.outcome_history)
                )
            self.p2.fire_at_target(
                self.p1.offense.target(self.p2.board,
                                       self.p2.outcome_history)
                )       
            self.turn_count += 1
            
            
    def setup(self):
        """
        Creates player boards and sets opponents to one another. Required
        before calling 'run'.

        Returns
        -------
        None.

        """
        self.turn_count = 0
        self.p1.board = Board(10)
        self.p2.board = Board(10)           
        self.p1.opponent = self.p2
        
        
    ### Factory Method ###
    
    @classmethod
    def player_with_options(cls, options):
        """
        Returns a player object based on the input options dict.

        Parameters
        ----------
        options : dict
            Must have 'offense' and 'defense' keys, with each value as follows:
                options['offense'] = {
                    'hunt_style': (str),
                    'hunt_pattern': (str),
                    'hunt_options': (dict),
                    'kill_options': (dict)}
                options['defense'] = {
                    'formation': (str), 
                    'method': (str),
                    'edge_buffer': (int), 
                    'ship_buffer': (int), 
                    'alignment': (constants.Align)}

        Returns
        -------
        None.

        """
        return AIPlayer(HunterOffense(options['offense']['hunt_style'],
                                      options['offense']['hunt_pattern'],
                                      options['offense']['hunt_options'],
                                      options['offense']['kill_options']),
                        RandomDefense(options['defense']['formation'],
                                      options['defense']['method'],
                                      options['defense']['edge_buffer'],
                                      options['defense']['ship_buffer'],
                                      options['defense']['alignment'])
                        )
    
    
    @classmethod
    def tester_for_version(cls, version):
        """
        Returns a TestPlayers object with player properties set by the version
        input.

        Parameters
        ----------
        version : int
            0+.

        Returns
        -------
        TestPlayers instance.

        """
        opt1,opt2 = cls.options_for_version(version)
        return TestPlayers(cls.player_with_options(opt1),
                           cls.player_with_options(opt2))
    
    ### Class Method ###
    
    @classmethod
    def options_for_version(cls, version):
        """
        Returns a dictionary of options based on the input version integer.
        The options values are passed to player, offense, and defense 
        objects to set up the properties of the game.

        Parameters
        ----------
        version : int
            An integer that corresponds to a particular setup:
                0:  Two players with completely random offense and defense
                    methods. Ships are not yet placed on the boards.
                1:  Two players with completely random offense and defense
                    methods. Ships have been placed on the boards.
                1:  Two players with completely random offense and defense
                    methods. Ships have been placed on the boards.

        Returns
        -------
        tuple of two dicts.

        """
        
        if version == 0 or version == 1:
            options1 = cls.random_options
            options2 = cls.random_options
                       
        else:
            raise ValueError("Version must be 0 or 1.")
        
        return options1, options2
            
            
def test_defense():
    
    nturns = 20
    props1 = {'offense': {'hunt_style': 'random',
                          'hunt_pattern': 'random',
                          'hunt_options': {},
                          'kill_options': {}
                          }, 
              'defense': {'formation': "random", 
                          'method': "weighted",
                          'edge_buffer': 0, 
                          'ship_buffer': 0, 
                          'alignment': constants.Align.ANY
                          }
              }
    
    # Test 'formation': random, cluster, isolate
    props2 = props1.copy()
    for formation in ['random', 'clustered', 'cluster', 'isolated', 'isolate']:
        props2['defense']['formation'] = formation
        tester = TestPlayers(props1,props2)
        print(tester)
        tester.run(nturns)
        print(tester)
       
    return

    # Test 'method': weighted, optimize, any
    props2 = props1.copy()
    for method in ['weighted', 'optimize', 'any']:
        props2['defense']['method'] = method
        tester = TestPlayers(props1,props2)
        print(tester)
        tester.run(nturns)
        print(tester)
    return

    # Test 'ship_buffer': 0-4
    props2 = props1.copy()
    for ship_buffer in list(range(5)):
        props2['defense']['ship_buffer'] = ship_buffer
        tester = TestPlayers(props1,props2)
        print(tester)
        tester.run(nturns)
        print(tester)
    return

    # Test 'edge_buffer': 0-4
    props2 = props1.copy()
    for edge_buffer in list(range(5)):
        props2['defense']['edge_buffer'] = edge_buffer
        tester = TestPlayers(props1,props2)
        print(tester)
        tester.run(nturns)
        print(tester)
    return

    # Test 'alignment': any, horizontal, vertical
    props2 = props1.copy()
    for alignment in constants.Align:
        props2['defense']['alignment'] = alignment
        tester = TestPlayers(props1,props2)
        print(tester)
        tester.run(nturns)
        print(tester)
    return
        
        
        
                    