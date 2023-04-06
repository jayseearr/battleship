#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 21:07:02 2023

@author: jason
"""

import unittest
import numpy as np

#%% Utils
from battleship import utils
from battleship.coord import Coord

class TestUtilsFunctions(unittest.TestCase):
    
    def test_is_on_board(self):
        self.assertTrue(utils.is_on_board((0,0), 1))
        self.assertFalse(utils.is_on_board((0,1), 1))
        self.assertTrue(utils.is_on_board((1,1), 2))
        self.assertFalse(utils.is_on_board((-1,0), 2))
        
        self.assertTrue(utils.is_on_board((0,0), 10))
        self.assertTrue(utils.is_on_board((9,0), 10))
        self.assertTrue(utils.is_on_board((0,9), 10))
        self.assertTrue(utils.is_on_board((9,9), 10))
        self.assertFalse(utils.is_on_board((-1,-1), 10))
        self.assertFalse(utils.is_on_board((10,0), 10))
        self.assertFalse(utils.is_on_board((0,10), 10))
        self.assertFalse(utils.is_on_board((10,10), 10))
        self.assertFalse(utils.is_on_board((11,4), 10))
        
        self.assertTrue(utils.is_on_board(Coord((0,0)), 10))
        self.assertFalse(utils.is_on_board(Coord((10,10)), 10))
        
        # first parameter should be a tuple or Coord
        self.assertRaises(TypeError, utils.is_on_board, 1, 10)
        self.assertRaises(TypeError, utils.is_on_board, [(0,0), (1,1)], 10)        
        
    def test_random_choice(self):
        x = [0, 1, 2, 3]
        # props must sum to 1:
        self.assertRaises(ValueError, utils.random_choice, x, [1,1,1,1])
        # p parameter must match length of x parameter:
        self.assertRaises(ValueError, utils.random_choice, x, [1./3]*(len(x)-1))
        
        for _ in range(10):
            self.assertTrue(utils.random_choice(x) in x)
            self.assertEqual(utils.random_choice(x,[0,0,1,0]), 2)
        self.assertTrue(utils.random_choice(np.array(x), np.array([1,1,1,1])/4)
                        in x)
        p = np.array([0.001, 0.1, 1, 0.01])
        p = p / np.sum(p)
        y = []
        counts = [0,0,0,0]
        for _ in range(1000):
            r = utils.random_choice(x, p)
            y += [r]
            counts[r] += 1
        # check that order of random selection frequency matches order of
        # probability, p:
        self.assertTrue(np.all(np.argsort(counts) == p.argsort()))
                             
    def test_rotate_coords(self):
        # for 90 degrees, (0,0) --> (0,9); (0,9) --> (9,9); (9,9) --> (9,0); (9,0) --> (0,0)
        # for 180 degrees, (0,0) --> (9,9); (0,9) --> (9,0); (9,9) --> (0,0); (9,0) --> (0,9)    
        # for -90 degrees, (0,0) --> (9,0); (0,9) --> (0,0); (9,9) --> (0,9); (9,0) --> (9,9) 
        self.assertRaises(TypeError, utils.rotate_coords, (0,0), 90, 10)
        
        corners = [(0,0), (0,9), (9,9), (9,0)]
        self.assertEqual(utils.rotate_coords([(0,0)], 90, 10), 
                         [(0,9)])
        self.assertEqual(utils.rotate_coords(corners, 90, 10), 
                         [corners[1], corners[2], corners[3], corners[0]])
        self.assertEqual(utils.rotate_coords(corners, -90, 10), 
                         [corners[3], corners[0], corners[1], corners[2]])
        
        rotated = utils.rotate_coords(corners, 180, 10)
        self.assertAlmostEqual(rotated[0][0], corners[2][0])
        self.assertAlmostEqual(rotated[0][1], corners[2][1])
        self.assertAlmostEqual(rotated[1][0], corners[3][0])
        self.assertAlmostEqual(rotated[1][1], corners[3][1])
        self.assertAlmostEqual(rotated[2][0], corners[0][0])
        self.assertAlmostEqual(rotated[2][1], corners[0][1])
        self.assertAlmostEqual(rotated[3][0], corners[1][0])
        self.assertAlmostEqual(rotated[3][1], corners[1][1])
        
        self.assertEqual(utils.rotate_coords([(2,2)], 45, 5), [(2,2)])
        self.assertEqual(utils.rotate_coords([(1,1)], 180, 5), [(3,3)])
        self.assertEqual(utils.rotate_coords([(2,2)], 90, 4), [(2,1)])
        
        
    def test_mirror_coords(self):
        
        self.assertRaises(TypeError, utils.mirror_coords, (0,0), 0, 10)
        # corners
        self.assertEqual(utils.mirror_coords([(0,0)], 0, 10), [(9,0)])
        self.assertEqual(utils.mirror_coords([(0,9)], 0, 10), [(9,9)])
        self.assertEqual(utils.mirror_coords([(9,9)], 0, 10), [(0,9)])
        self.assertEqual(utils.mirror_coords([(9,0)], 0, 10), [(0,0)])
        self.assertEqual(utils.mirror_coords([(0,0)], 1, 10), [(0,9)])
        self.assertEqual(utils.mirror_coords([(0,9)], 1, 10), [(0,0)])
        self.assertEqual(utils.mirror_coords([(9,9)], 1, 10), [(9,0)])
        self.assertEqual(utils.mirror_coords([(9,0)], 1, 10), [(9,9)])
        
        # middle
        self.assertEqual(utils.mirror_coords([(4,4)], 0, 10), [(5,4)])
        self.assertEqual(utils.mirror_coords([(4,4)], 1, 10), [(4,5)])
        self.assertEqual(utils.mirror_coords([(3,5)], 0, 10), [(6,5)])
        
        # smaller board
        self.assertEqual(utils.mirror_coords([(2,2)], 0, 5), [(2,2)])
        self.assertEqual(utils.mirror_coords([(2,2)], 0, 4), [(1,2)])
        self.assertEqual(utils.mirror_coords([(2,2)], 1, 4), [(2,1)])
        
    def test_parse_alignment(self):
        # "North", "South", "N/S", "N", "S": returns Align.VERTICAL
        # "East", "West", "E/W", "W", "E": returns Align.HORIZONTAL
        # "Any", "All": returns Align.ANY
        for s in ["North", "South", "N/S", "N", "S"]:
            self.assertEqual(utils.parse_alignment(s), constants.Align.VERTICAL)
            self.assertEqual(utils.parse_alignment(s.lower()), constants.Align.VERTICAL)
            self.assertEqual(utils.parse_alignment(s.upper()), constants.Align.VERTICAL)
        for s in ["East", "West", "E/W", "W", "E"]:
            self.assertEqual(utils.parse_alignment(s), constants.Align.HORIZONTAL)
            self.assertEqual(utils.parse_alignment(s.lower()), constants.Align.HORIZONTAL)
            self.assertEqual(utils.parse_alignment(s.upper()), constants.Align.HORIZONTAL)
        for s in ["Any", "All"]:
            self.assertEqual(utils.parse_alignment(s), constants.Align.ANY)
            self.assertEqual(utils.parse_alignment(s.lower()), constants.Align.ANY)
            self.assertEqual(utils.parse_alignment(s.upper()), constants.Align.ANY)
            
        self.assertEqual(utils.parse_alignment(None), constants.Align.ANY)
        self.assertEqual(utils.parse_alignment(constants.Align.VERTICAL), 
                         constants.Align.VERTICAL)
        self.assertEqual(utils.parse_alignment(constants.Align.HORIZONTAL), 
                         constants.Align.HORIZONTAL)
        self.assertEqual(utils.parse_alignment(constants.Align.ANY), 
                         constants.Align.ANY)
        self.assertEqual(utils.parse_alignment(None), constants.Align.ANY)
        
        self.assertRaises(ValueError, utils.parse_alignment, "X")
        self.assertRaises(ValueError, utils.parse_alignment, "abc")
        self.assertRaises(TypeError, utils.parse_alignment, 1)
        
    
#%% Coord
from battleship.coord import Coord

class TestCoordMethods(unittest.TestCase):
    
    def test_init_by_tuple(self):
        self.assertEqual(Coord((1,2)).rowcol, (1,2), 
                         "Expected rowcol property to equal input tuple")
     
    def test_init_by_ints(self):
        self.assertEqual(Coord(1,2).rowcol, (1,2), 
                         "Expected rowcol property to equal input ints")
    
    def test_init_by_str(self):
        self.assertEqual(Coord("k09").rowcol, (10,8), 
                         "Expected rowcol property to equal (10,8).")
        
    def test_init_by_numpy_array(self):
        self.assertEqual(Coord(np.array((3,4))).rowcol, (3,4), 
                         "Expected rowcol property to equal input array.")
        
    def test_lbl(self):
        self.assertEqual(Coord((0,1)).lbl, "A2",
                         "Expected (0,1) label to be 'A2'.")
        
        
    def test_value(self):
        c = Coord((3,4))
        c.value = -1.5
        self.assertEqual(c.value, -1.5,
                         "Expected Coord value to be equal to value at "
                         "init (-1.5)")
        
    def test_str(self):
        self.assertEqual(Coord((1,2)).lbl, str(Coord((1,2))), 
                         "Expected Coord label and str(coord) to be equal.")
        
    def test_eq(self):
        self.assertTrue(Coord("J10") == Coord(9,9),
                        "Expected equal operator to be true to for coords.")
        self.assertFalse(Coord("J10") == Coord(8,9),
                        "Expected equal operator to be False to for different "
                        "coords.")
        
    def test_getitem(self):
        self.assertEqual((Coord((1,2))[0], Coord((1,2))[1]), (1,2),
                         "Expected Coord first element to be row value, "
                         "and Coord second element to be col value.")
        
    def test_setitem(self):
        c = Coord((1,2))
        x = (2,3)
        c[0] = x[0]
        c[1] = x[1]
        self.assertEqual(c.rowcol, x,
                         "Expected Coord first element to be row value, "
                         "and Coord second element to be col value.")
        
    def test_add(self):
        self.assertEqual(Coord((1,3)) + (2,4), (3,7),
                         "Expected Coord row/col to add element-wise "
                         "to a tuple .")
        
        def test_bad_add_fcn():
            return (Coord((1,3)) + (1,2,3))
        self.assertRaises(ValueError, test_bad_add_fcn)
        # "Expected a ValueError when adding a too-long tuple.")
        
        def test_bad_add_fcn_2():
            return (Coord((1,3)) + (0.1, 1))
        self.assertRaises(ValueError, test_bad_add_fcn_2)
        # "Expected a ValueError when adding non-integer.")
        
    def test_sub(self):
        self.assertEqual(Coord((1,3)) - (2,5), (-1,-2),
                         "Expected Coord row/col to subtract element-wise "
                         "with a tuple .")
        
        def test_bad_sub_fcn():
            return (Coord((1,3)) - (1,2,3))
        self.assertRaises(ValueError, test_bad_sub_fcn)
        #Expected a ValueError when subtracting a too-long tuple."
        
        def test_bad_sub_fcn_2():
            return (Coord((1,3)) - (0.1, 1))
        self.assertRaises(ValueError, test_bad_sub_fcn_2)
        # "Expected a ValueError when subtracting non-integer."
        
    def test_mul(self):
        self.assertEqual(Coord((1,3)) * (2,5), (2,15),
                         "Expected Coord row/col to multiply element-wise.")
        self.assertEqual(Coord((1,3)) * -2, (-2,-6),
                         "Expected Coord row/col to multiply with "
                         "negative scalar.")
        self.assertEqual(Coord((1,3)) * 2, (2,6),
                         "Expected Coord row/col to multiply with scalar.")
        
        def test_bad_mul_fcn():
            return (Coord((1,3)) * (1,2,3))
        self.assertRaises(ValueError, test_bad_mul_fcn)
        # "Expected a ValueError when multiplying a too-long tuple."
        
        def test_bad_mul_fcn_2():
            return (Coord((1,3)) * 0.1)
        self.assertRaises(ValueError, test_bad_mul_fcn_2)
        # "Expected a ValueError when multiplying non-integer."
        
    
    
        
#%% Placement

from battleship.placement import Placement

class TestPlacementMethods(unittest.TestCase):
    
    def test_init_by_coord_heading_len(self):
        p1 = Placement((2,3), "W", 5)
        p2 = Placement((8,9), "N", 3)
        self.assertEqual((p1.coord, p1.heading, p1.length), 
                         ((2,3), "W", 5),
                         "Expected coord/heading/length to be consistent "
                         "with input parameters.")
        self.assertEqual((p2.coord, p2.heading, p2.length), 
                         ((8,9), "N", 3),
                         "Expected coord/heading/length to be consistent "
                         "with input parameters.")
        
    def test_init_by_dict(self):
        p1 = Placement({'coord': (2,3), 
                        'heading': "W", 
                        'length': 5})
        self.assertEqual((p1.coord, p1.heading, p1.length), 
                         ((2,3), "W", 5),
                         "Expected coord/heading/length to be consistent "
                         "with input dict.")
        
    def test_coord(self):
        self.assertEqual(Placement((2,3), "W", 5).coord, (2,3))
        
    def test_coord_invalid_input(self):
        self.assertRaises(ValueError, Placement, (1, "W", 5))
        self.assertRaises(ValueError, Placement, ((1,2,3), "W", 5))
        
    def test_heading(self):
        self.assertEqual(Placement((2,3), "w", 5).heading, "W") # checks that 'w' --> 'W'
        self.assertEqual(Placement((2,3), "west", 5).heading, "W") # checks that 'west' --> 'W'
        self.assertEqual(Placement((2,3), "WEST", 5).heading, "W") # checks that 'WEST' --> 'W'
        self.assertEqual(Placement((2,3), "north", 5).heading, "N") 
        self.assertEqual(Placement((2,3), "NORTH", 5).heading, "N") 
        self.assertEqual(Placement((2,3), "n", 5).heading, "N") 
        self.assertEqual(Placement((2,3), "N", 5).heading, "N") 
        self.assertEqual(Placement((2,3), "east", 5).heading, "E") 
        self.assertEqual(Placement((2,3), "EAST", 5).heading, "E") 
        self.assertEqual(Placement((2,3), "e", 5).heading, "E") 
        self.assertEqual(Placement((2,3), "E", 5).heading, "E") 
        self.assertEqual(Placement((2,3), "south", 5).heading, "S") 
        self.assertEqual(Placement((2,3), "SOUTH", 5).heading, "S") 
        self.assertEqual(Placement((2,3), "s", 5).heading, "S") 
        self.assertEqual(Placement((2,3), "S", 5).heading, "S") 
        
    def test_heading_invalid_input(self):
        self.assertRaises(ValueError, Placement, 
                          {'coord': (2,3), 
                           'heading': "G", # This shouldn't be possible! N/W only.
                           'length': 5}
                          )
        
    def test_length(self):
        self.assertEqual(Placement((2,3), "W", 5).length, 5)
        self.assertEqual(len(Placement((2,3), "W", 5)), 5)
        
    def test_length_invalid_input(self):
        self.assertRaises(ValueError, Placement, ((1,2), "W", -1))
        self.assertRaises(ValueError, Placement, ((1,2), "W", (1,2)))
        
    def test_coords(self):
        self.assertEqual(len(Placement((2,3), "W", 3).coords()), 3)
        self.assertEqual(Placement((2,3), "W", 3).coords(), 
                         [(2, 3), (2, 4), (2, 5)])
        self.assertEqual(Placement((1,1), "E", 3).coords(), 
                         [(1, 1), (1, 0), (1, -1)]) # negative coords are ok
        
    def test_rows(self):
        rows = Placement((2,3), "W", 3).rows()
        self.assertTrue(isinstance(rows, np.ndarray),
                        "Expected a numpy ndarray.")
        self.assertTrue((np.all(Placement((1,2), "E", 3).rows() == 
                         np.array((1,1,1)))),
                         "Expected rows to be a 3-element array.") 
        
    def test_cols(self):
        cols = Placement((2,3), "W", 3).cols()
        self.assertTrue(isinstance(cols, np.ndarray),
                        "Expected a numpy ndarray.")
        self.assertTrue((np.all(Placement((1,2), "W", 3).cols() == 
                         np.array((2,3,4)))),
                         "Expected cols to be a 3-element array.")
        
    def test_getitem(self):
        self.assertEqual(Placement((1,2), "N", 2)[0], (1,2),
                         "Expected first element to be (1,2).")
        self.assertEqual(Placement((1,2), "W", 2)[1], (1,3),
                         "Expected second element to be (1,3).")
        def bad_getitem_fcn():
            return Placement((1,2), "W", 2)[3] # length 2 placement shouldn't have a 3rd item
        self.assertRaises(IndexError, bad_getitem_fcn)
        
    def test_eq(self):
        self.assertTrue((Placement((1,2), 'N', 3) 
                         == Placement((3,2), 'S', 3)), 
                        "Expected two opposite-facing placements to equal.")
        
    def test_dist_to_coord(self):
        self.assertTrue(np.all(Placement((1, 2), 'N', 3).dist_to_coord((2, 4),'manhattan')
                               == np.array([3, 2, 3])),
                        "Expected distance a coord to be 3-2-3.")
        self.assertEqual(np.min(Placement((1, 2), 'N', 3).dist_to_coord((2, 2),'dist')), 0,
                        "Expected min distance to a contained coord to be 0.")
        
    def test_total_dist_to_coord(self):
        self.assertEqual(Placement((1, 2), 'N', 3).total_dist_to_coord((2, 4),'dist2'),
                         np.sum((5, 4, 5)),
                        "Expected total distance^2 to coord to be 5+4+5.")
        self.assertEqual(Placement((1, 2), 'N', 3).total_dist_to_coord((2, 4),'dist'),
                         np.sqrt(1**2+2**2) + np.sqrt(0**2+2**2) + np.sqrt((-1)**2+2**2),
                        "Expected total distance to coord to be sqrt(5)+sqrt(4)+sqrt(5).")
        
    def dists_to_placement(self):
        a = Placement((5, 4), 'N', 3)
        b = Placement((5, 6), 'N', 3)
        self.assertEqaul(a.dists_to_placement(b, "manhattan"), 2, 
                         "Distance should be 2 for any method.")
        self.assertEqaul(a.dists_to_placement(b, "dist"), 2, 
                         "Distance should be 2 for any method.")
        
    def test_total_dist_to_placement(self):
        self.assertEqual(Placement((1, 2), 'N', 3).total_dist_to_placement(
                            Placement((2,4), 'W', 2),'dist'),
                         (np.sqrt(1+2**2) + np.sqrt(0+2**2) + np.sqrt(1+2**2) +     
                         np.sqrt(1+3**2) + np.sqrt(0+3**2) + np.sqrt(1+3**2)),
                         "Expected distances to sum to 13.37...")
        self.assertEqual(Placement((1, 2), 'N', 3).total_dist_to_placement(
                            Placement((2,4), 'W', 2),'manhattan'),
                         3 + 2 + 3 + 4 + 3 + 4,
                         "Expected distances to sum to 19")
        
    def test_dist_grid(self):
        self.assertEqual(Placement((1,0),"N",3).total_dist_grid(
                            4,'manhattan').min(),
                         2,
                         "Min of total dist grid should be 2 for length 3 ship.")
        self.assertEqual(Placement((1,0),"N",3).total_dist_grid(
                            4,'manhattan').max(),
                         15,
                         "Max of total dist grid should be 15 on this small test board.")
        
                    

#%% Ship

from battleship.ship import Ship
from battleship import constants

class TestShipMethods(unittest.TestCase):
    
    def test_data(self):
        self.assertEqual(len(Ship.data), len(constants.ShipType),
                         "Expected ShipType and Ship.data to have same len.")
        self.assertEqual(Ship.data[3], Ship.data[constants.ShipType(3)],
                         "Expected ShipType(3) and 3 to give the same data.")
        
    def test_init_(self):
        self.assertFalse(Ship(1) == None, "Expected ship to not be None.")
        self.assertEqual(Ship(1).name, Ship("patrol").name,
                         "Expected str and int to init same type of ship.")
        self.assertEqual(Ship(constants.ShipType(1)).name, Ship("patrol").name,
                         "Expected ShipType and int to init same type of ship.")
        
    def test_len(self):
        self.assertEqual(Ship(5).length, 5,
                         "Ship(5) should have len 5.")
        self.assertEqual(len(Ship(5)), Ship(5).length,
                         "len(Ship(5)) should be the same as Ship(5).length.")
    
    def test_dmg(self):
        s = Ship(1)
        self.assertTrue(np.all(s.damage == np.array((0,0))),
                         "Patrol damage should be two zeros.")
        
    def test_ship_id(self):
        for k in Ship.data:
            self.assertEqual(Ship(k).ship_id, k,
                             "ship_id should equal int used to initialize.")
    
    def test_hit(self):
        s = Ship(1)
        s.hit(0)
        self.assertTrue(np.all(s.damage == np.array((1,0))),
                         "Patrol should be damaged at first slot.")
        s.hit(1)
        self.assertTrue(np.all(s.damage == np.array((1,1))),
                         "Patrol should be damaged at second slot.")
        self.assertRaises(ValueError, s.hit, -1)
        self.assertRaises(ValueError, s.hit, 2)
        
    def test_is_afloat(self):
        s = Ship(1)
        s.hit(0)
        self.assertEqual(s.is_afloat(), True,
                         "Expected True after damage to one slot.")
        s.hit(1)
        self.assertEqual(s.is_afloat(), False,
                         "Expected False after damage to all slots.")
        
    def test_is_sunk(self):
        s = Ship(1)
        self.assertEqual(s.is_sunk(), False,
                         "Expected False with no damage.")
        s.hit(0)
        self.assertEqual(s.is_sunk(), False,
                         "Expected False after damage to one slot.")
        s.hit(1)
        self.assertEqual(s.is_sunk(), True,
                         "Expected True after damage to all slots.")
        
        
    
#%% Board
from battleship.board import Board
#from battleship import constants

class TestBoardMethods(unittest.TestCase):
    
    test_board = Board(11)
    
    def test_init(self):
        board = self.test_board
        self.assertEqual(board.size, 11, "Expected size to be 11.")
        self.assertDictEqual(board.fleet, {}, "Expected empty fleet dict")
        self.assertDictEqual(board.ship_placements, {}, 
                             "Expected empty ship_placements dict")
        self.assertTupleEqual(board.ocean_grid.shape, (11,11), 
                              "Expected ocean_grid to be 11 x 11.")
        self.assertTupleEqual(board.target_grid.shape, (11,11), 
                              "Expected target_grid to be 11 x 11.")
        self.assertRaises(ValueError, Board, -1)
        
    def test_str(self):
        # Makes sure the returned string is long enough to represent the
        # target and ocean grids
        self.assertGreater(len(str(self.test_board)), (11*2)*11)
        
    def test_add_ship(self):
        
        k = 4
        ship = Ship(k)
        p = Placement((0,1), "N", ship.length)
        
        # add ship by coord/heading
        b = Board(10)
        b.add_ship(k, p.coord, p.heading)
        self.assertTrue(k in b.fleet, "Expected ship to be in fleet.")
        self.assertTrue(np.all(b.ocean_grid[[0,1,2,3],[1,1,1,1]] 
                               == k * np.ones(ship.length)))
        
        # add ship by placement
        b = Board(10)
        b.add_ship(k,placement=p)
        self.assertTrue(k in b.fleet, "Expected ship to be in fleet.")
        self.assertTrue(np.all(b.ocean_grid[[0,1,2,3],[1,1,1,1]] 
                               == k * np.ones(ship.length)))
        
        # add ship with wrong length
        self.assertRaises(ValueError, Board(10).add_ship, 
                          (5,), {'placement': p})
        
        # add ship off board
        self.assertRaises(Exception, Board(10).add_ship,
                          (1, (0,0), "S"))
        
        # add ship overlapping
        b = Board(10)
        b.add_ship(k,placement=p)
        self.assertRaises(Exception, b.add_ship,
                          (5, (1,0), "W"))
        # add ship twice
        b = Board(10)
        b.add_ship(k,placement=p)
        self.assertRaises(Exception, b.add_ship, (k, p.coord, p.heading))
        
        # too many arguments
        self.assertRaises(ValueError, b.add_ship, (k, p.coord, p.heading, p))
        
    def test_copy(self):
        b1 = Board(5)
        b1.add_ship(2, (2,4), "N")
        b2 = b1.copy()
        
        self.assertEqual(b1.size, b2.size, 
                         "Expected copy to have same size as original.")
        self.assertEqual(b1.fleet.keys(), b2.fleet.keys(), 
                             "Expected copied fleet to match original fleet.")
        for k in b1.fleet.keys():
            self.assertEqual(b1.fleet[k].ship_id, b2.fleet[k].ship_id)
            self.assertEqual(b1.fleet[k].length, b2.fleet[k].length)
            self.assertEqual(b1.fleet[k].name, b2.fleet[k].name)
            self.assertTrue(np.all(b1.fleet[k].damage == b2.fleet[k].damage))
        self.assertDictEqual(b1.ship_placements, b2.ship_placements, 
                             "Expected copied placements to match originals.")
        
    def test_outcome(self):
        out = {'coord': (2,3), 'hit': True, 'sunk': True, 'sunk_ship_type': 4}
        self.assertDictEqual(Board.outcome(out['coord'], out['hit'],
                                           out['sunk'], out['sunk_ship_type']),
                             out)
        
    def test_size(self):
        self.assertEqual(Board(3).size, 3)
        self.assertEqual(Board(5).ocean_grid.shape[0], Board(5).size)
        
    def test_ocean_grid(self):
        self.assertTrue(np.all(Board(2).ocean_grid.flatten() == 
                        np.zeros(2*2)), "Expected a 2x2 array of zeroes.")
        
    def test_target_grid(self):
        self.assertTrue(np.all(Board(2).target_grid.flatten() == 
                        constants.TargetValue.UNKNOWN * np.ones(2*2)), 
                        "Expected a 2x2 array of -2's.")
        
    def test_fleet(self):
        b = Board(10)
        k = 3
        ship = Ship(k)
        p = Placement((2,3), "N", ship.length)
        b.add_ship(k, p.coord, p.heading)
        #self.assertEqual(b.fleet[k], ship)
        self.assertEqual(len(b.fleet), 1)
        self.assertListEqual(list(b.fleet.keys()), [3])        
        self.assertEqual(b.fleet[k].ship_id, ship.ship_id)
        self.assertIsNone(b.fleet.get(-1))
        self.assertIsNone(b.fleet.get(1))   
        self.assertRaises(KeyError, lambda k: b.fleet[k], 1)
        
    def test_ship_placements(self):
        b = Board(10)
        placements = {1: Placement((0,1),"W",2),
                      2: Placement((2,2),"W",3),
                      3: Placement((3,3),"N",3),
                      4: Placement((5,5),"N",4),
                      5: Placement((1,0),"N",5)}
        for k,p in placements.items():
            b.add_ship(k, p.coord, p.heading)
        for k in placements:
            self.assertEqual(b.ship_placements[k],
                             placements[k])
        
        self.assertIsNone(b.fleet.get(6))
        self.assertRaises(KeyError, lambda k: b.fleet[k], 6)
        
    def test_rel_coords_for_heading(self):
        b = Board(9)
        rows,cols = b.rel_coords_for_heading("N", 3)
        self.assertListEqual(list(rows), [0,1,2])
        self.assertListEqual(list(cols), [0,0,0])
        rows,cols = b.rel_coords_for_heading("S", 3)
        self.assertListEqual(list(rows), [0,-1,-2])
        self.assertListEqual(list(cols), [0,0,0])
        rows,cols = b.rel_coords_for_heading("E", 3)
        self.assertListEqual(list(rows), [0,0,0])
        self.assertListEqual(list(cols), [0,-1,-2])
        rows,cols = b.rel_coords_for_heading("W", 3)
        self.assertListEqual(list(rows), [0,0,0])
        self.assertListEqual(list(cols), [0,1,2])
        rows,cols = b.rel_coords_for_heading("N", 5)
        self.assertListEqual(list(rows), [0,1,2,3,4])
        self.assertListEqual(list(cols), [0,0,0,0,0])
        rows,cols = b.rel_coords_for_heading("N", 1)
        self.assertListEqual(list(rows), [0])
        self.assertListEqual(list(cols), [0])
        # invalid length:
        self.assertRaises(ValueError, b.rel_coords_for_heading, "N", 0)
        # invalid heading:
        self.assertRaises(ValueError, b.rel_coords_for_heading, "Q", 3)
    
    # def test_coords_for_placement(self):
    #     p = Placement((0,0), "N", 3)
    #     self.assertEqual(p.coord(), b.coords_for_placement(p))

    def test_is_valid_coord(self):
        b = Board(10)
        # is_valid_coord returns an array of bools
        self.assertRaises(TypeError, b.is_valid_coord, (0,0))
        
        self.assertTrue(np.all(b.is_valid_coord([(0,0)]))) 
        self.assertTrue(np.all(b.is_valid_coord([Coord(0,0)])))         
        self.assertTrue(np.all(b.is_valid_coord([(9,9)])))
        self.assertFalse(np.all(b.is_valid_coord([(10,9)])))
        self.assertFalse(np.all(b.is_valid_coord([(0,-1)])))
        
    #def test_coords_to_coords_distances(self): # class method
    
    def test_random_coord(self):
        b = Board(2)
        self.assertIn(b.random_coord(), [(0,0),(0,1),(1,0),(1,1)])
        b.add_ship(1, (0,0), "N")
        b.update_target_grid(b.outcome((1,0),False))
        b.update_target_grid(b.outcome((1,1),False))
        for _ in range(10):
            self.assertIn(b.random_coord(unoccuppied=True), [(0,1),(1,1)])
            self.assertIn(b.random_coord(untargeted=True), [(0,0),(0,1)])
        
    def test_random_heading(self):
        b = Board(2)
        headings = ["N", "S", "E", "W"]
        headings_vertical = ["N", "S"]
        headings_horizontal = ["E", "W"]
        for _ in range(10):
            self.assertIn(b.random_heading(), headings)
            self.assertIn(b.random_heading(alignment="any"), headings)
            self.assertIn(b.random_heading(alignment=constants.Align.ANY), 
                          headings)
            
            self.assertIn(b.random_heading(alignment="NS"), 
                          headings_vertical)
            self.assertIn(b.random_heading(alignment="N/S"), 
                          headings_vertical)
            self.assertIn(b.random_heading(alignment="N|S"), 
                          headings_vertical)
            self.assertIn(b.random_heading(alignment=constants.Align.VERTICAL), 
                          headings_vertical)
            
            self.assertIn(b.random_heading(alignment="EW"), 
                          headings_horizontal)
            self.assertIn(b.random_heading(alignment="E/W"), 
                          headings_horizontal)
            self.assertIn(b.random_heading(alignment="E|W"), 
                          headings_horizontal)
            self.assertIn(b.random_heading(alignment=constants.Align.HORIZONTAL), 
                          headings_horizontal)
            
            self.assertIn(b.random_heading(alignment="NW"), 
                          ["N", "W"])
            self.assertIn(b.random_heading(alignment="standard"), 
                          ["N", "W"])
            
            self.assertRaises(ValueError, b.random_heading, "Q")
            
    def test_random_placement(self):
        from battleship.placement import Placement
        b = Board(2)
        self.assertRaises(ValueError, b.random_placement, 3)
        self.assertTrue(isinstance(b.random_placement(2), Placement))
        self.assertTrue(b.random_placement(2) in [Placement((0,0),"N",2),
                                                  Placement((0,1),"N",2),
                                                  Placement((0,0),"W",2),
                                                  Placement((1,0),"W",2)])
        
            
    def test_all_targets(self):
        b = Board(2)
        self.assertListEqual(b.all_targets(), [(0,0),(0,1),(1,0),(1,1)])
        self.assertListEqual(b.all_targets(untargeted=True), 
                             [(0,0),(0,1),(1,0),(1,1)])
        self.assertListEqual(b.all_targets(targeted=True), [])
        
        b.update_target_grid(b.outcome((0,0), hit=False, sunk=False))
        b.update_target_grid(b.outcome((0,1), hit=True, sunk=False))
        self.assertListEqual(b.all_targets(), [(0,0),(0,1),(1,0),(1,1)])
        self.assertListEqual(b.all_targets(untargeted=True), 
                             [(1,0),(1,1)])
        self.assertListEqual(b.all_targets(targeted=True), [(0,0), (0,1)])
        
        self.assertRaises(ValueError, b.all_targets, True,True)
        
    def test_all_targets_where(self):
        b = Board(2)
        self.assertRaises(ValueError, b.all_targets_where) # no arguments is not allowed
        self.assertEqual(b.all_targets_where(ship_type=1), [])
        self.assertEqual(b.all_targets_where(miss=True), [])
        self.assertEqual(b.all_targets_where(hit=True), [])
        
        b.update_target_grid(b.outcome((0,0), hit=True, sunk=False))
        b.update_target_grid(b.outcome((0,1), hit=True, sunk=True, 
                                       sunk_ship_type=1))
        self.assertEqual(b.all_targets_where(miss=True), [])
        self.assertEqual(b.all_targets_where(hit=True), [(0,0), (0,1)])
        self.assertEqual(b.all_targets_where(ship_type=1), [(0, 0), (0, 1)])
        
        self.assertEqual(b.all_targets_where(ship_type=-1), [])
        
    def test_targets_around(self):
                           
        b = Board(10)
        self.assertListEqual(b.targets_around((0,0)), [(0,1),(1,0)])
        self.assertSetEqual(set(b.targets_around((4,4))), 
                            set(((3,4),(5,4),(4,3),(4,5))))
        self.assertSetEqual(set(b.targets_around((0,0), diagonal=True)), 
                             set([(0,1),(1,0),(1,1)]))
        self.assertSetEqual(set(b.targets_around((4,4), diagonal=True)), 
                            set(((3,4),(5,4),(4,3),(4,5),
                                 (3,3),(3,5),(5,5),(5,3))))
        
        hit = constants.TargetValue.HIT
        b.update_target_grid(b.outcome((4,4), hit=False))
        b.update_target_grid(b.outcome((4,5), hit=True, sunk=True, 
                                       sunk_ship_type=1))
        b.update_target_grid(b.outcome((4,6), hit=True, sunk=True, 
                                       sunk_ship_type=1))
        self.assertSetEqual(set(b.targets_around((4,4), targeted=True)),
                            set(((4,5),)))
        self.assertSetEqual(set(b.targets_around((4,4), untargeted=True)),
                            set(((3,4), (5,4), (4,3))))
        self.assertSetEqual(set(b.targets_around((4,4), 
                                                 values = (hit, 1))),
                            set(((4,5),)))
        self.assertSetEqual(set(b.targets_around((4,5), 
                                                 values = (1,))),
                            set(((4,6),)))
        self.assertSetEqual(set(b.targets_around((4,5), 
                                                 values = (2,))),
                            set())
        self.assertSetEqual(set(b.targets_around((4,5), 
                                    values = (constants.TargetValue.MISS,))),
                            set(((4,4),)))
        
    def test_is_valid_target_ship_placement(self):
        b = Board(2)
        self.assertTrue(b.is_valid_target_ship_placement(
            Placement((0,0),"N",2)))
        self.assertTrue(b.is_valid_target_ship_placement(
            Placement((0,0),"W",2)))
        self.assertFalse(b.is_valid_target_ship_placement(
            Placement((0,0),"N",3)))
        self.assertFalse(b.is_valid_target_ship_placement(
            Placement((0,0),"S",2)))
        
        # placement overlap with a miss
        b.update_target_grid(b.outcome((0,0), False))
        self.assertTrue(b.is_valid_target_ship_placement(
            Placement((0,1),"N",2)))
        self.assertFalse(b.is_valid_target_ship_placement(
            Placement((0,0),"N",2)))
        
        # placement overlap with a hit
        b = Board(3)
        b.update_target_grid(b.outcome((0,0), True))
        self.assertTrue(b.is_valid_target_ship_placement(
            Placement((0,0),"N",2), ship=Ship(1)))
        self.assertTrue(b.is_valid_target_ship_placement(
            Placement((0,0),"N",3), ship=Ship(2)))
        
        # placement overlap with a sink
        b = Board(3)
        b.update_target_grid(b.outcome((1,0), True, True, 2))
        b._target_grid[0,0] = 0
        b._target_grid[2,0] = 0
        self.assertFalse(b.is_valid_target_ship_placement(
            Placement((0,0),"N",2), ship=Ship(1)))
        self.assertTrue(b.is_valid_target_ship_placement(
            Placement((0,0),"N",3), ship=Ship(2)))
        # if the target_grid does not have a determined ship type at (0,0),
        # the following should be True:
        self.assertTrue(b.is_valid_target_ship_placement(
            Placement((0,0),"W",2), ship=Ship(1)))
        
        b = Board(3)
        b.update_target_grid(b.outcome((0,0), True))
        b.update_target_grid(b.outcome((2,0), True))
        b.update_target_grid(b.outcome((1,0), True, True, 2))
        self.assertFalse(b.is_valid_target_ship_placement(
            Placement((0,0),"N",2), ship=Ship(1)))
        self.assertTrue(b.is_valid_target_ship_placement(
            Placement((0,0),"N",3), ship=Ship(2)))
        # Since the target_grid now DOES have a determined ship type at (0,0),
        # the following should be False:
        self.assertFalse(b.is_valid_target_ship_placement(
            Placement((0,0),"W",2), ship=Ship(1)))
        
        # placement overlap with a sink and miss (False)
        b = Board(3)
        b.update_target_grid(b.outcome((0,0), False))
        b.update_target_grid(b.outcome((1,0), True, True, 2))
        self.assertFalse(b.is_valid_target_ship_placement(
            Placement((0,0),"N",3), ship=Ship(2)))
        
        # mismatch between placement length and ship length
        self.assertRaises(ValueError, Board(3).is_valid_target_ship_placement,
                          Placement((0,0),"N",3), Ship(5))
        
        
    def test_all_valid_target_placements(self):
        b = Board(3)
        self.assertEqual(len(b.all_valid_target_placements(ship_type=1)), 12)
        self.assertEqual(len(b.all_valid_target_placements(ship_type=2)), 6)
        self.assertEqual(len(b.all_valid_target_placements(ship_type=4)), 0)
        self.assertEqual(len(b.all_valid_target_placements(ship_type=2)), 
                         len(b.all_valid_target_placements(ship_len=3)))
        
        # test with some hits/misses on the target grid.
        b.update_target_grid(b.outcome((0,0),False))
        b.update_target_grid(b.outcome((0,1),False))
        b.update_target_grid(b.outcome((1,1),True))
        self.assertEqual(len(b.all_valid_target_placements(ship_type=1)), 8)
        self.assertEqual(len(b.all_valid_target_placements(ship_type=2)), 3)
        # after a sink, there should be only one placement for the sunk ship type.
        b.update_target_grid(b.outcome((1,0),True,True,1))
        self.assertEqual(len(b.all_valid_target_placements(ship_type=1)), 1)
        self.assertEqual(len(b.all_valid_target_placements(ship_type=2)), 2)
        
        # for different reasons, ship_len < 1 and ship_type < 1 should throw
        # ValueErrors.
        self.assertRaises(ValueError, b.all_valid_target_placements, -1)
        
    def test_target_grid_is(self):
        b = Board(3)
        b.update_target_grid(b.outcome((0,0),False))
        b.update_target_grid(b.outcome((0,1),True))
        b.update_target_grid(b.outcome((0,2),True,True,1))
        
        self.assertEqual(np.sum(b.target_grid_is('miss').flatten()), 1)
        self.assertEqual(np.sum(b.target_grid_is(['miss']).flatten()), 1)
        self.assertEqual(np.sum(b.target_grid_is('hit').flatten()), 2)
        self.assertEqual(np.sum(b.target_grid_is(
            constants.TargetValue.HIT).flatten()), 0) # zero because no spots EXACTLY match 0.
        self.assertEqual(np.sum(b.target_grid_is(['hit','miss']).flatten()), 3)
        self.assertEqual(np.sum(b.target_grid_is([1]).flatten()), 2)        
        self.assertEqual(np.sum(b.target_grid_is(
            [1], exact_ship_type=True).flatten()), 1)   
        self.assertEqual(np.sum(b.target_grid_is(
            [1], exact_ship_type=False).flatten()), 2)  
        
        b.update_target_grid(b.outcome((1,1),True))
        self.assertEqual(np.sum(b.target_grid_is(
            constants.TargetValue.HIT).flatten()), 1) # zero because no spots EXACTLY match 0.
        self.assertEqual(np.sum(b.target_grid_is([1,2]).flatten()), 2)
        self.assertEqual(np.sum(b.target_grid_is([1,2,'hit']).flatten()), 3)
        self.assertEqual(np.sum(b.target_grid_is('hit').flatten()), 3)
        self.assertEqual(np.sum(b.target_grid_is(
            'hit',exact_ship_type=True).flatten()), 1)  # only one space is exaclty 0.
        b.update_target_grid(b.outcome((1,0),True))
        b.update_target_grid(b.outcome((1,2),True,True,2))
        self.assertEqual(np.sum(b.target_grid_is([1,2]).flatten()), 5)
        self.assertEqual(np.sum(b.target_grid_is(['hit']).flatten()), 5)
        self.assertEqual(np.sum(b.target_grid_is([1]).flatten()), 2)
        self.assertEqual(np.sum(b.target_grid_is([2]).flatten()), 3)
        
        
    def test_target_coords_with_type(self):
        # returns coordinates on the target grid associated with a particular 
        # type of ship. Does not accept 'hit' or 'miss' as inputs, just ship_type.
        b = Board(3)
        b._target_grid = -2. * np.ones((3,3))
        b._target_grid[0,1] = 1
        b._target_grid[0,2] = 1 + constants.SUNK_OFFSET
        b._target_grid[0,0] = -1
        b._target_grid[2,0] = 0  # 0 is equal to constants.TargetValue.HIT
        b._target_grid[2,1] = 0
        self.assertEqual(b.target_coords_with_type(1), [(0,1),(0,2)])
        self.assertEqual(b.target_coords_with_type(1,sink=True), [(0,2)])
        self.assertEqual(b.target_coords_with_type(2), [])
        
        b._target_grid[2,2] = 2.
        self.assertEqual(b.target_coords_with_type(2), [(2,2)])
        
        self.assertEqual(set(b.target_coords_with_type(0)), set([(2,0),(2,1)]))
        self.assertEqual(b.target_coords_with_type(6), [])
        
    def test_all_coords(self):
        b = Board(2)
        self.assertListEqual(b.all_coords(), [(0,0),(0,1),(1,0),(1,1)])
        self.assertListEqual(b.all_coords(unoccuppied=True), 
                             [(0,0),(0,1),(1,0),(1,1)])
        self.assertListEqual(b.all_coords(occuppied=True), 
                             [])
        b.add_ship(1, (0,0), "N")
        self.assertListEqual(b.all_coords(), [(0,0),(0,1),(1,0),(1,1)])
        self.assertListEqual(b.all_coords(unoccuppied=True), 
                             [(0,1),(1,1)])
        self.assertListEqual(b.all_coords(occuppied=True), 
                             [(0,0),(1,0)])
        
    def test_coords_around(self):
        # This is for ocean grid, not target grid.
        b = Board(3)
        self.assertSetEqual(set(b.coords_around((1,1))), 
                            set([(0,1),(2,1),(1,0),(1,2)]))
        self.assertSetEqual(set(b.coords_around((1,1), diagonal=True)), 
                            set([(0,0),(0,1),(0,2),
                                 (1,0),(1,2),
                                 (2,0),(2,1),(2,2)]))
        b.add_ship(1, (0,0), "N")
        self.assertSetEqual(set(b.coords_around((1,1),unoccuppied=True)),
                            set([(0,1),(1,2),(2,1)]))
        self.assertSetEqual(set(b.coords_around((1,1),occuppied=True)),
                            set([(1,0)]))
        self.assertSetEqual(set(b.coords_around((1,1),diagonal=True,occuppied=True)),
                            set([(0,0), (1,0)]))
        self.assertSetEqual(set(b.coords_around((0,0),diagonal=True,occuppied=True)),
                            set([(1,0)]))
        
    def test_is_valid_placement(self):
        pgood = Placement((0,5), "W", 4)
        pbad = Placement((1,6), "W", 5)
        poverlap = Placement((0,7), "N", 4)
        b = Board(10)
        self.assertTrue(b.is_valid_placement(pgood, unoccuppied=False))
        self.assertFalse(b.is_valid_placement(pbad, unoccuppied=False))
        self.assertFalse(b.is_valid_placement(Placement((-1,0),"N",2)))
        # check behavior for overlapping ship 
        b.add_ship(4, pgood.coord, pgood.heading)
        self.assertTrue(b.is_valid_placement(poverlap, unoccuppied=False))
        self.assertFalse(b.is_valid_placement(poverlap, unoccuppied=True))
        
    def test_all_valid_ship_placements(self):
        b = Board(3)
        p1 = Placement((0,0), "N", 2)
        p2 = Placement((0,2), "N", 3)
        b.add_ship(1, p1.coord, p1.heading)
        b.add_ship(2, p2.coord, p2.heading)
        self.assertEqual(len(b.all_valid_ship_placements(ship_len=2)), 3)
        self.assertEqual(len(b.all_valid_ship_placements(ship_len=3)), 1)
        # the placements of existing ships should not be in the valid placements
        self.assertFalse(any([p == p1 for p in 
                              b.all_valid_ship_placements(ship_len=2)]))
        self.assertFalse(any([p == p2 for p in 
                              b.all_valid_ship_placements(ship_len=3)]))
        self.assertEqual(len(b.all_valid_ship_placements(ship_type=1)), 3)
        self.assertEqual(len(b.all_valid_ship_placements(ship_type=2)), 1)
        self.assertEqual(len(b.all_valid_ship_placements(ship_type=4)), 0)
        self.assertEqual(len(b.all_valid_ship_placements(ship_type=5)), 0)
        
        # ship buffer test
        b = Board(3)
        b.add_ship(2, (0,1), "N")
        self.assertEqual(len(b.all_valid_ship_placements(
            ship_type=1, ship_buffer=1)), 0)
        b = Board(4)
        b.add_ship(2, (0,1), "N")
        self.assertEqual(len(b.all_valid_ship_placements(
            ship_type=1, ship_buffer=1, diagonal=True)), 4)
        self.assertEqual(len(b.all_valid_ship_placements(
            ship_type=1, ship_buffer=1, diagonal=False)), 3)
        self.assertTrue(all([p.coord[1] == 3 for p in 
                             b.all_valid_ship_placements(
                                 ship_type=1, ship_buffer=1, diagonal=False)
                             ]))
        
        b = Board(4)
        b.add_ship(4, (0,1), "N")
        self.assertEqual(len(b.all_valid_ship_placements(
            ship_type=1, ship_buffer=0, edge_buffer=1)), 1)
        self.assertEqual(b.all_valid_ship_placements(
            ship_type=1, ship_buffer=0, edge_buffer=1)[0].coord, (1,2))
        self.assertEqual(len(b.all_valid_ship_placements(
            ship_type=2, ship_buffer=0, edge_buffer=1)), 0)
        
        b = Board(4)
        self.assertEqual(len(b.all_valid_ship_placements(
            ship_type=1, alignment="any")), 4*(4-1)*2)
        self.assertEqual(len(b.all_valid_ship_placements(
            ship_type=1, alignment="NS")), 4*(4-1))
        self.assertEqual(len(b.all_valid_ship_placements(
            ship_type=1, alignment="EW")), 4*(4-1))
        self.assertTrue(all([(p.heading == "N") for p in 
                            b.all_valid_ship_placements(
                                ship_type=1, alignment="NS")]))
        self.assertRaises(ValueError, b.all_valid_ship_placements, 
            {'ship_type':1, 'alignment':"Q"})        
        
    def test_ship_for_type(self):
        # returned ship should be the actual instance that is part of the 
        # fleet, not just any instance of Ship.
        placements = {1: Placement((0,0),"N",2),
                      2: Placement((1,1),"N",3),
                      3: Placement((2,2),"N",3),
                      4: Placement((3,3),"N",4),
                      5: Placement((4,4),"N",5)
                      }
        b = Board(10)
        b.add_fleet(placements)
        for k in range(1,6):
            self.assertEqual(b.fleet[k], b.ship_for_type(k))
        self.assertNotEqual(b.fleet[1], Ship(1))
        
        b = Board(10)
        b.add_fleet({k:placements[k] for k in range(1,4)})
        self.assertEqual(len(b.fleet), 3)
        self.assertRaises(KeyError, b.ship_for_type, 4)
        
    def test_ship_at_coord(self):
        b = Board(3)
        b.add_ship(3,(0,2),"N")
        b.add_ship(1,(1,1),"N")
        self.assertIsNone(b.ship_at_coord((0,0)))
        self.assertIsNone(b.ship_at_coord((1,0)))
        self.assertIsNone(b.ship_at_coord((2,0)))
        self.assertIsNone(b.ship_at_coord((0,1)))
        self.assertEqual(b.ship_at_coord((1,1)), b.fleet[1])
        self.assertEqual(b.ship_at_coord((2,1)), b.fleet[1])
        self.assertEqual(b.ship_at_coord((0,2)), b.fleet[3])
        self.assertEqual(b.ship_at_coord((1,2)), b.fleet[3])
        self.assertEqual(b.ship_at_coord((2,2)), b.fleet[3])
    
    def test_damage_at_coord(self):
        b = Board(3)
        b.add_ship(3,(0,2),"N")
        b.add_ship(1,(1,1),"N")
        for coord in b.all_coords():
            self.assertEqual(b.damage_at_coord(coord), 0)
        b.fleet[3].damage[1] += 1 # damage 2nd slot (coord (1,2))
        self.assertEqual(b.damage_at_coord((0,2)), 0)
        self.assertEqual(b.damage_at_coord((1,2)), 1)
        self.assertEqual(b.damage_at_coord((2,2)), 0)
        b.fleet[3].damage[1] += 1 # damage 2nd slot (coord (1,2)) again
        self.assertEqual(b.damage_at_coord((1,2)), 2)
        
        self.assertRaises(ValueError, b.damage_at_coord, (-1,0),)
        self.assertRaises(ValueError, b.damage_at_coord, (3,0),)
        
    def test_is_fleet_afloat(self):
        b = Board(3)
        b.add_ship(3,(0,2),"N")
        b.add_ship(1,(1,1),"N")
        self.assertTrue(all(b.is_fleet_afloat() 
                            == np.array([True, True])))
        for i in range(len(b.fleet[3].damage)):
            b.fleet[3].damage[i] = 1.
        self.assertTrue(all(b.is_fleet_afloat() 
                            == np.array([True, False])))
        for i in range(len(b.fleet[1].damage)):
            b.fleet[1].damage[i] = 1.
        self.assertTrue(all(b.is_fleet_afloat() 
                            == np.array([False, False])))
        
    def test_coords_for_ship(self):
        b = Board(10)
        placements = {}
        for k in range(1,6):
            p = b.random_placement(Ship.data[k]["length"], unoccuppied=True)
            placements[k] = p
            b.add_ship(k, p.coord, p.heading)
        self.assertEqual(len(b.fleet), 5)
        for k in b.fleet:
            self.assertEqual(b.coords_for_ship(b.fleet[k]), 
                             placements[k].coords())
        b = Board(10)
        b.add_ship(1,(0,0),"N")
        self.assertEqual(len(b.coords_for_ship(1)), 2)
        self.assertRaises(ValueError, b.coords_for_ship, 2)
        self.assertRaises(ValueError, b.coords_for_ship, 4)
        self.assertRaises(ValueError, b.coords_for_ship, 5)
        
        self.assertEqual(len(b.coords_for_ship(Ship(1))), 2)
        self.assertRaises(ValueError, b.coords_for_ship, Ship(5))
        
    def test_placement_for_ship(self):
        b = Board(10)
        placements = {}
        for k in range(1,6):
            p = b.random_placement(Ship.data[k]["length"], unoccuppied=True)
            placements[k] = p
            b.add_ship(k, p.coord, p.heading)
        self.assertEqual(len(b.fleet), 5)
        for k in b.fleet:
            self.assertEqual(b.placement_for_ship(b.fleet[k]), 
                             placements[k])
        
        b = Board(10)
        b.add_ship(1,(0,0),"N")
        self.assertEqual(b.placement_for_ship(Ship(1)), Placement((0,0),"N",2))
        self.assertRaises(ValueError, b.placement_for_ship, Ship(2))
        self.assertRaises(ValueError, b.placement_for_ship, Ship(3))
        self.assertRaises(ValueError, b.placement_for_ship, Ship(4))
        self.assertRaises(ValueError, b.placement_for_ship, Ship(5))
        
        self.assertEqual(b.placement_for_ship(Ship(1)), Placement((0,0),"N",2))
        self.assertRaises(ValueError, b.placement_for_ship, Ship(5))
        
    def test_placements_containing_coord(self):
        b = Board(4)
        self.assertRaises(TypeError, b.placements_containing_coord, 
                          (1,2), ship_types=1)
        
        self.assertSetEqual(b.placements_containing_coord((1,2), 
                                                          ship_types=[1]), 
                            set([Placement((0,2),"N",2), 
                                 Placement((1,2),"N",2),
                                 Placement((1,1),"W",2),
                                 Placement((1,2),"W",2)]))
        self.assertSetEqual(b.placements_containing_coord((1,2), 
                                                          ship_types=[1,2,3]), 
                            set([Placement((0,2),"N",2), 
                                 Placement((1,2),"N",2),
                                 Placement((1,1),"W",2),
                                 Placement((1,2),"W",2),
                                 Placement((0,2),"N",3),
                                 Placement((1,2),"N",3),
                                 Placement((1,0),"W",3),
                                 Placement((1,1),"W",3)]))
        
        self.assertEqual(len(b.placements_containing_coord((0,0))), 6)
        self.assertEqual(len(Board(10).placements_containing_coord((4,4))), 
                         2*2 + 2*3 + 2*4 + 2*5)
        
    def test_placements_containing_all_coords(self):
        self.assertRaises(TypeError, Board(10).placements_containing_all_coords, 
                          [(1,2),(3,4)], ship_types=1)
        self.assertRaises(TypeError, Board(10).placements_containing_all_coords, 
                          (1,2), ship_types=[1])
        b = Board(10)
        self.assertSetEqual(b.placements_containing_all_coords([(0,1),(0,3)]), 
                            set([Placement((0, 0), 'W', 4),
                                 Placement((0, 0), 'W', 5),
                                 Placement((0, 1), 'W', 3),
                                 Placement((0, 1), 'W', 4),
                                 Placement((0, 1), 'W', 5)]))
        self.assertSetEqual(b.placements_containing_all_coords([(0,1),(0,3)],
                                                               ship_types=[4]), 
                            set([Placement((0, 0), 'W', 4),
                                 Placement((0, 1), 'W', 4)]))
        self.assertSetEqual(b.placements_containing_all_coords([(0,0),(0,3)]), 
                            set([Placement((0, 0), 'W', 4), 
                                 Placement((0, 0), 'W', 5)]))
        b = Board(3)
        self.assertSetEqual(b.placements_containing_all_coords([(1,1),(2,1)]), 
                            set([Placement((1, 1), 'N', 2), 
                                 Placement((0, 1), 'N', 3)]))
        self.assertEqual(len(b.placements_containing_all_coords([(0,0),(1,1)])), 0)
        self.assertSetEqual(b.placements_containing_all_coords([(0,0),(1,1)]),
                            set())
                
    def test_add_fleet(self):
        b = Board(10)
        placements = {}
        for k in range(1,6):
            p = b.random_placement(Ship.data[k]["length"], unoccuppied=True)
            b.add_ship(k, placement=p)  # add ship so that the next placement will truly be unoccuppied
            placements[k] = p
        b = Board(10)   # have to create new board so that all ships can be added at once
        b.add_fleet(placements) # <<<<<<<<< This is what is being tested
        self.assertEqual(len(b.fleet), 5)
        for k in b.fleet:
            self.assertEqual(b.placement_for_ship(b.fleet[k]), 
                             placements[k])
            
    def test_is_ready_to_play(self):
        b = Board(10)
        b.add_ship(1,(0,0),"N")
        b.add_ship(2,(0,1),"N")
        b.add_ship(3,(0,2),"N")
        b.add_ship(4,(0,3),"N")
        self.assertFalse(b.is_ready_to_play())
        b.add_ship(5,(0,4),"N")
        self.assertTrue(b.is_ready_to_play())
        
        b.damage_ship_at_coord(b.fleet[1],(0,0))
        self.assertFalse(b.is_ready_to_play())
        b.fleet[1].damage[0] = 0
        self.assertTrue(b.is_ready_to_play())
        
        b.update_target_grid(b.outcome((0,0),False))
        self.assertFalse(b.is_ready_to_play())
        b._target_grid[0,0] = constants.TargetValue.UNKNOWN
        self.assertTrue(b.is_ready_to_play())
        b.update_target_grid(b.outcome((0,0),True))
        self.assertFalse(b.is_ready_to_play())
        b._target_grid[0,0] = constants.TargetValue.UNKNOWN
        self.assertTrue(b.is_ready_to_play())
        b.update_target_grid(b.outcome((0,0),True,True,1))
        self.assertFalse(b.is_ready_to_play())
        
    def test_incoming_at_coord(self):
        expected_keys = set(["hit", "coord", "sunk", 
                             "sunk_ship_type", "message"])
        b = Board(10)
        b.add_ship(1,(0,0),"N")
        ship = b.fleet[1]
        outcome = b.incoming_at_coord((0,1))
        self.assertSetEqual(set(outcome.keys()), expected_keys)
        self.assertEqual(outcome["hit"], False)
        self.assertEqual(outcome["coord"], (0,1))
        self.assertEqual(outcome["sunk"], False)
        self.assertEqual(outcome["sunk_ship_type"], None)
        # self.assertEqual(outcome["message"], None) 
        # msg may change in future versions so do not force it to be None
        
        outcome = b.incoming_at_coord((0,0))
        self.assertSetEqual(set(outcome.keys()), expected_keys)
        self.assertEqual(outcome["hit"], True)
        self.assertEqual(outcome["coord"], (0,0))
        self.assertEqual(outcome["sunk"], False)
        self.assertEqual(outcome["sunk_ship_type"], None)
        self.assertEqual(b.fleet[1].damage[0], 1)   # ensures incoming_at_coord increases ship's damage
        
        outcome = b.incoming_at_coord((1,0))
        self.assertSetEqual(set(outcome.keys()), expected_keys)
        self.assertEqual(outcome["hit"], True)
        self.assertEqual(outcome["coord"], (1,0))
        self.assertEqual(outcome["sunk"], True)
        self.assertEqual(outcome["sunk_ship_type"], constants.ShipType.PATROL)
        self.assertTrue(any([f"{ship.name} sunk".lower()
                             in line.lower() for line in outcome["message"]]))
        self.assertEqual(b.fleet[1].damage[1], 1)   # ensures incoming_at_coord increases ship's damage
        
        outcome = b.incoming_at_coord((0,1))
        self.assertSetEqual(set(outcome.keys()), expected_keys)
        self.assertEqual(outcome["hit"], False)
        self.assertEqual(outcome["coord"], (0,1))
        self.assertEqual(outcome["sunk"], False)
        self.assertEqual(outcome["sunk_ship_type"], None)
        
        # check that repeat damage triggers a message
        outcome = b.incoming_at_coord((0,0))
        self.assertSetEqual(set(outcome.keys()), expected_keys)
        self.assertEqual(outcome["hit"], True)
        self.assertEqual(outcome["coord"], (0,0))
        self.assertEqual(outcome["sunk"], True)
        self.assertEqual(outcome["sunk_ship_type"], constants.ShipType.PATROL)
        self.assertTrue(any([f"Repeat damage: {outcome['coord']}".lower() 
                             in line.lower() for line in outcome["message"]]))
        
    def test_damage_ship_at_coord(self):
        b = Board(10)
        b.add_ship(1,(0,0),"N")
        b.add_ship(2,(5,0),"N")
        self.assertTrue(np.all(b.fleet[1].damage == np.array([0,0])))
        b.damage_ship_at_coord(b.fleet[1], (0,0))
        self.assertTrue(np.all(b.fleet[1].damage == np.array([1,0])))
        b.damage_ship_at_coord(b.fleet[1], (1,0))
        self.assertTrue(np.all(b.fleet[1].damage == np.array([1,1])))
        self.assertTrue(b.fleet[1].is_sunk())
        
        b.damage_ship_at_coord(b.fleet[1], (1,0))
        self.assertTrue(np.all(b.fleet[1].damage == np.array([1,2])))
        self.assertTrue(b.fleet[1].is_sunk())
        
        # ship not on board
        self.assertRaises(ValueError, b.damage_ship_at_coord, Ship(3), (9,9))
        
        # ship is not part of fleet (despite matching ship type)
        self.assertRaises(ValueError, b.damage_ship_at_coord, Ship(1), (0,0))
        
        # ship not at coord
        self.assertRaises(ValueError, b.damage_ship_at_coord, 
                          b.fleet[2], (0,1))
        
    def test_update_target_grid(self):
        b = Board(10)
        # check that adding a ship to ocean grid does not affect target grid
        b.add_ship(1,(0,0),"N") 
        self.assertTrue(np.all(b.target_grid.flatten() 
                               == constants.TargetValue.UNKNOWN))
        outcome1 = Board.outcome((0,0), False, False, None)
        outcome2 = Board.outcome((0,1), True, False, None)
        outcome3 = Board.outcome((0,2), True, True, constants.ShipType.PATROL)
        b.update_target_grid(outcome1)
        self.assertTrue(b.target_grid[0,0] == constants.TargetValue.MISS)
        self.assertTrue(all([b.target_grid[r,c] 
                             for r in range(10) for c in range(10) 
                             if (r,c) != (0,0) 
                             == constants.TargetValue.UNKNOWN]))
        
        b.update_target_grid(outcome2)
        self.assertTrue(b.target_grid[0,0] == constants.TargetValue.MISS)
        self.assertTrue(b.target_grid[0,1] == constants.TargetValue.HIT)
        self.assertTrue(all([(b.target_grid[r,c] 
                              for r in range(10) 
                              for c in range(10) 
                              if (r,c) not in ((0,0),(0,1))
                             == constants.TargetValue.UNKNOWN)]))
        
        b.update_target_grid(outcome3)
        self.assertTrue(b.target_grid[0,0] == constants.TargetValue.MISS)
        self.assertTrue(b.target_grid[0,1] == constants.ShipType.PATROL)
        self.assertTrue(b.target_grid[0,2] == (constants.ShipType.PATROL 
                                               + constants.SUNK_OFFSET))
        self.assertTrue(all([(b.target_grid[r,c] 
                              for r in range(10) 
                              for c in range(10) 
                              if (r,c) not in ((0,0),(0,1),(0,2))
                             == constants.TargetValue.UNKNOWN)]))
        
    def test_ocean_grid_image(self):
        b = Board(3)
        b.add_ship(1, (0,0), "N")
        b.add_ship(2, (0,2), "N")
        self.assertTrue(np.all(b.ocean_grid_image() == 
                               np.array(((1., 0, 1),(1, 0, 1),(0, 0, 1)))))
        b.damage_ship_at_coord(b.fleet[1], (0,0))
        b.damage_ship_at_coord(b.fleet[2], (0,2))
        b.damage_ship_at_coord(b.fleet[2], (1,2))
        b.damage_ship_at_coord(b.fleet[2], (1,2))
        self.assertTrue(np.all(b.ocean_grid_image() == 
                               np.array(((2., 0, 2),(1, 0, 3),(0, 0, 1)))))
                
    def test_target_grid_image(self):
        b = Board(3)
        b.add_ship(1, (0,0), "N")
        b.add_ship(2, (0,2), "N")

        _Miss = constants.TargetValue.MISS
        _Hit = constants.TargetValue.HIT
        _Unk = constants.TargetValue.UNKNOWN

        self.assertTrue(np.all(b.target_grid_image() == _Unk))
        
        b.update_target_grid(b.outcome((0,0), False))
        self.assertTrue(np.all(b.target_grid_image() == 
                               np.array(((_Miss, _Unk, _Unk),
                                         (_Unk, _Unk, _Unk),
                                         (_Unk, _Unk, _Unk)))))
        
        b.update_target_grid(b.outcome((0,1), True, False))
        self.assertTrue(np.all(b.target_grid_image() == 
                               np.array(((_Miss, _Hit, _Unk),
                                         (_Unk, _Unk, _Unk),
                                         (_Unk, _Unk, _Unk)))))
        
        k = 1
        b.update_target_grid(b.outcome((0,2), True, True, k))
        self.assertTrue(np.all(b.target_grid_image() == 
                               np.array(((_Miss, k, k + constants.SUNK_OFFSET),
                                         (_Unk, _Unk, _Unk),
                                         (_Unk, _Unk, _Unk)))))
        
    def test_ship_rects(self):
        b = Board(4)
        b.add_ship(1,(0,1),"N")
        b.add_ship(4,(3,0),"W")
        self.assertTrue(np.all(b.ship_rects()[1] 
                               == np.array((1.5, 0.5, 1, 2)))) # x, y, w, h
        self.assertTrue(np.all(b.ship_rects()[4] 
                               == np.array((0.5, 3.5, 4, 1)))) 
        for k in (2,3,5):
            self.assertFalse(k in b.ship_rects())
        
    def test_color_str(self):
        b = Board(3)
        b.add_ship(1,(0,0),"N")
        b.damage_ship_at_coord(b.fleet[1], (1,0))
        b.update_target_grid(b.outcome((0,2),False))
        b.update_target_grid(b.outcome((1,2),True))
        s = b.color_str()
        self.assertTrue(len(s) >= (3+1)*(3+1) * 2 * 2)
        self.assertTrue(s.count("\x1b") > (3*3*2))
        
    def test_show(self):
        b = Board(10)
        f = b.show()
        import matplotlib
        self.assertTrue(isinstance(f, matplotlib.figure.Figure))
        self.assertEqual(len(f.axes), 2)
        self.assertEqual(len(b.show("ocean").axes), 1)
        self.assertEqual(len(b.show("target").axes), 1)
        
    def test_show2(self):
        b = Board(10)
        f = b.show()
        import matplotlib
        self.assertTrue(isinstance(f, matplotlib.figure.Figure))
        self.assertEqual(len(f.axes), 2)
        self.assertEqual(len(b.show("ocean").axes), 1)
        self.assertEqual(len(b.show("target").axes), 1)
        
    def test_status_str(self):
        b = Board(3)
        b.add_ship(1,(0,0),"N")
        b.damage_ship_at_coord(b.fleet[1],(1,0))
        b.update_target_grid(b.outcome((0,1),False))
        b.update_target_grid(b.outcome((1,1),True))
        b.update_target_grid(b.outcome((2,1),True,True,2))
        s = b.status_str()
        
        # s should read something like:
        #   Shots fired: 3 (2 hits, 1 misses)\n
        #   Ships:       1 hits on 1 ships, 0 ships sunk
        s = s.lower()
        
        self.assertTrue("ships sunk" in s)
        self.assertTrue("hits" in s)
        self.assertTrue("shots" in s)
        self.assertTrue("misses" in s)
        self.assertTrue(s.count('3') == 1)
        self.assertTrue(s.count('2') == 1)
        self.assertTrue(s.count('1') == 3)
        self.assertTrue(s.count('0') == 1)
        
    def test_possible_ship_types_at_coord(self):
        
        # empty target grid
        b = Board(3)
        self.assertDictEqual(b.possible_ship_types_at_coord((0,0)), 
                             {constants.ShipType.PATROL: 2,
                              constants.ShipType.DESTROYER: 2,
                              constants.ShipType.SUBMARINE: 2,
                              constants.ShipType.BATTLESHIP: 0,
                              constants.ShipType.CARRIER: 0})
        
        b = Board(10)
        self.assertDictEqual(b.possible_ship_types_at_coord((4,4)), 
                             {constants.ShipType.PATROL: 4,
                              constants.ShipType.DESTROYER: 6,
                              constants.ShipType.SUBMARINE: 6,
                              constants.ShipType.BATTLESHIP: 8,
                              constants.ShipType.CARRIER: 10})
        
        # test around a miss
        b = Board(3)
        b.update_target_grid(b.outcome((1,1),False))
        # no possible ships at miss
        self.assertDictEqual(b.possible_ship_types_at_coord((1,1)), 
                             {constants.ShipType.PATROL: 0,
                              constants.ShipType.DESTROYER: 0,
                              constants.ShipType.SUBMARINE: 0,
                              constants.ShipType.BATTLESHIP: 0,
                              constants.ShipType.CARRIER: 0})
        # some possible ships to left of miss
        self.assertDictEqual(b.possible_ship_types_at_coord((1,0)), 
                             {constants.ShipType.PATROL: 2,
                              constants.ShipType.DESTROYER: 1,
                              constants.ShipType.SUBMARINE: 1,
                              constants.ShipType.BATTLESHIP: 0,
                              constants.ShipType.CARRIER: 0})
        
        # test around a hit
        b = Board(4)    # <-- notice bigger board size
        b.update_target_grid(b.outcome((1,1),True))
        # some possible ships at miss
        self.assertDictEqual(b.possible_ship_types_at_coord((1,1)), 
                             {constants.ShipType.PATROL: 4,
                              constants.ShipType.DESTROYER: 4,
                              constants.ShipType.SUBMARINE: 4,
                              constants.ShipType.BATTLESHIP: 2,
                              constants.ShipType.CARRIER: 0})
        # some possible ships to left of miss
        self.assertDictEqual(b.possible_ship_types_at_coord((1,0)), 
                             {constants.ShipType.PATROL: 3,
                              constants.ShipType.DESTROYER: 3,
                              constants.ShipType.SUBMARINE: 3,
                              constants.ShipType.BATTLESHIP: 2,
                              constants.ShipType.CARRIER: 0})
        
        # board with some hits, some misses
        b = Board(5)    
        b.update_target_grid(b.outcome((0,0),False))
        b.update_target_grid(b.outcome((1,1),True))
        b.update_target_grid(b.outcome((0,4),False))
        
        # some possible ships next to miss
        self.assertDictEqual(b.possible_ship_types_at_coord((0,1)), 
                             {constants.ShipType.PATROL: 2,
                              constants.ShipType.DESTROYER: 2,
                              constants.ShipType.SUBMARINE: 2,
                              constants.ShipType.BATTLESHIP: 1,
                              constants.ShipType.CARRIER: 1})
        # adding a miss changes carrier possibilities only
        b.update_target_grid(b.outcome((4,1),False))
        self.assertDictEqual(b.possible_ship_types_at_coord((0,1)), 
                             {constants.ShipType.PATROL: 2,
                              constants.ShipType.DESTROYER: 2,
                              constants.ShipType.SUBMARINE: 2,
                              constants.ShipType.BATTLESHIP: 1,
                              constants.ShipType.CARRIER: 0})
        
        b.update_target_grid(b.outcome((2,2),False))
        self.assertDictEqual(b.possible_ship_types_at_coord((2,1)), 
                             {constants.ShipType.PATROL: 3,
                              constants.ShipType.DESTROYER: 2,
                              constants.ShipType.SUBMARINE: 2,
                              constants.ShipType.BATTLESHIP: 1,
                              constants.ShipType.CARRIER: 0})
        
        # test near edges
        b = Board(10)
        b.update_target_grid(b.outcome((4,3), False))
        self.assertDictEqual(b.possible_ship_types_at_coord((4,0)), 
                             {constants.ShipType.PATROL: 3,
                              constants.ShipType.DESTROYER: 4,
                              constants.ShipType.SUBMARINE: 4,
                              constants.ShipType.BATTLESHIP: 4,
                              constants.ShipType.CARRIER: 5})
        
        # test near known ship
        b = Board(4)
        b.update_target_grid(b.outcome((0,0),True))
        b.update_target_grid(b.outcome((0,1),True))
        b.update_target_grid(b.outcome((0,2),True,True,2))
        ship_coords = b.target_coords_with_type(2)
        other_coords = b.target_coords_with_type(constants.TargetValue.UNKNOWN)
        for coord in ship_coords:
            self.assertEqual(b.possible_ship_types_at_coord(coord)[2], 1)
        for coord in other_coords:
            self.assertEqual(b.possible_ship_types_at_coord(coord)[2], 0)            
        
        b.update_target_grid(b.outcome((1,0),True))
        self.assertDictEqual(b.possible_ship_types_at_coord((1,0)),
                             {constants.ShipType.PATROL: 2,
                              constants.ShipType.DESTROYER: 0,
                              constants.ShipType.SUBMARINE: 2,
                              constants.ShipType.BATTLESHIP: 1,
                              constants.ShipType.CARRIER: 0})
        
        b.update_target_grid(b.outcome((1,1),False))
        self.assertDictEqual(b.possible_ship_types_at_coord((2,0)),
                             {constants.ShipType.PATROL: 3,
                              constants.ShipType.DESTROYER: 0,
                              constants.ShipType.SUBMARINE: 2,
                              constants.ShipType.BATTLESHIP: 1,
                              constants.ShipType.CARRIER: 0})
        
        # check output option functionality
        self.assertSetEqual(set(b.possible_ship_types_at_coord((2,0),'list')),
                            set([1,3,4]))
        b = Board(10)
        self.assertSetEqual(set(b.possible_ship_types_at_coord((2,0),'list')),
                            set([1,2,3,4,5]))
                        


    def test_possible_targets_grid(self):
        b = Board(3)
        ptg = b.possible_targets_grid(False)
        self.assertTupleEqual(ptg.shape, (3,3))
        self.assertTrue(np.all(ptg == np.array(((6,7,6),(7,8,7),(6,7,6)))))
        
        b = Board(4)
        b.update_target_grid(b.outcome((0,0),True))
        b.update_target_grid(b.outcome((0,1),True))
        b.update_target_grid(b.outcome((0,2),True,True,2))
        b.update_target_grid(b.outcome((1,0),True))
        b.update_target_grid(b.outcome((1,1),False))
        ptg = b.possible_targets_grid(False)
        self.assertTrue(np.all(ptg == np.array([[1., 1., 1., 3.],
                                                [2., 0., 3., 6.],
                                                [6., 6., 8., 8.],
                                                [5., 6., 7., 6.]])))
        
        # check by_type option
        ptg = b.possible_targets_grid(True) # ptg is now a dict w/ship type keys
        self.assertEqual(np.sum((ptg[2]==1).flatten()), 3)
        self.assertEqual(np.sum((ptg[2]==0).flatten()), b.size**2 - 3)
        self.assertEqual(np.sum((ptg[5]==0).flatten()), b.size**2)
        
        for k in (1,3,4):
            self.assertEqual(ptg[k][1,1], 0)
            self.assertTrue(np.all(ptg[k][(0,0,0),(0,1,2)] 
                                   == np.array((0,0,0))))
            
        # these numbers are specfic to the above board; they do not represent
        # any particularly special sums, but check that the function behavior
        # is consistent.
        self.assertEqual(np.sum(ptg[1].flatten()), 30)
        self.assertEqual(np.sum(ptg[3].flatten()), 24)
        self.assertEqual(np.sum(ptg[4].flatten()), 12)        
        
    def test_find_hits(self):
        b = Board(4)
        b.update_target_grid(b.outcome((0,0),True))
        b.update_target_grid(b.outcome((0,1),True))
        b.update_target_grid(b.outcome((0,2),True,True,2))
        b.update_target_grid(b.outcome((1,0),True))
        b.update_target_grid(b.outcome((1,1),False))
        
        self.assertSetEqual(b.find_hits(), set([(0,0),(0,1),(0,2),(1,0)]))
        self.assertSetEqual(b.find_hits(unresolved=True), set([(1,0)]))
        b.update_target_grid(b.outcome((2,2),True))
        self.assertSetEqual(b.find_hits(True), set([(1,0),(2,2)]))
        
        self.assertSetEqual(Board(10).find_hits(), set())
    
#%% Defense
from battleship.defense import Defense, RandomDefense

class TestDefenseFunctions(unittest.TestCase):
    
    def test_init(self):
        # should not be able to instantiate abstract class Defense
        self.assertRaises(TypeError, Defense)

        
class TestRandomDefenseFunctions(unittest.TestCase):
    
    def test_init(self):
        
        defense = RandomDefense()
        # default parameters
        self.assertEqual(defense.formation, "random")
        self.assertEqual(defense.method, "weighted")
        self.assertEqual(defense.edge_buffer, 0)
        self.assertEqual(defense.ship_buffer, 0)
        self.assertEqual(defense.alignment, constants.Align.ANY)
        
        # check formation parameters
        self.assertEqual(RandomDefense("RANDOM").formation, "random")
        self.assertEqual(RandomDefense(formation="RANDOM").formation, "random")
        self.assertEqual(RandomDefense("cluster").formation, "clustered")
        self.assertEqual(RandomDefense("clustered").formation, "clustered")
        self.assertEqual(RandomDefense("isolate").formation, "isolated")
        self.assertEqual(RandomDefense("isolated").formation, "isolated")
        self.assertRaises(ValueError, RandomDefense, formation="not allowed")
        
        # check method parameters (depend on formation params)
        self.assertEqual(RandomDefense("RANDOM").method, "weighted") #default
        self.assertEqual(RandomDefense("cluster").method, "weighted")
        self.assertEqual(RandomDefense("isolate").method, "weighted")
        
        self.assertEqual(RandomDefense("random", "any").method, "any")
        self.assertEqual(RandomDefense("random", "weighted").method, 
                         "weighted")
        self.assertEqual(RandomDefense("random", "optimize").method, 
                         "optimize")
        self.assertRaises(ValueError, RandomDefense, "random", "not allowed")
        
        self.assertEqual(RandomDefense("cluster", "any").method, "any")
        self.assertEqual(RandomDefense("cluster", "weighted").method, 
                         "weighted")
        self.assertEqual(RandomDefense("cluster", "optimize").method, 
                         "optimize")
        self.assertRaises(ValueError, RandomDefense, "cluster", "not allowed")
        
        self.assertEqual(RandomDefense("isolate", "any").method, "any")
        self.assertEqual(RandomDefense("isolate", "weighted").method, 
                         "weighted")
        self.assertEqual(RandomDefense("isolate", "optimize").method, 
                         "optimize")
        self.assertRaises(ValueError, RandomDefense, "isolate", "not allowed")
        
        
        # test edge_buffer
        self.assertRaises(TypeError, RandomDefense, 
                          edge_buffer = "NOT AN INT")
        self.assertEqual(RandomDefense(edge_buffer=2).edge_buffer, 2)
        
        # test ship_buffer
        self.assertRaises(TypeError, RandomDefense, 
                          ship_buffer = "NOT AN INT")
        self.assertRaises(TypeError, RandomDefense, 
                          ship_buffer = [1])
        self.assertEqual(RandomDefense(ship_buffer=2).ship_buffer, 2)
        
        # test alignment
        self.assertRaises(ValueError, RandomDefense, 
                          alignment = "NOT VALID")
        self.assertEqual(RandomDefense(alignment="N/S").alignment, 
                         constants.Align.VERTICAL)
        
    def test_formation(self):
        d = RandomDefense("isolate")
        self.assertEqual(d._formation, d.formation)
        d.formation = "cluster"
        self.assertEqual(d.formation, "clustered")
        # test the formation setter using __setattr__
        self.assertRaises(ValueError, d.__setattr__, "formation", "foo")
        
    def test_method(self):
        d = RandomDefense("isolate", method="weighted")
        self.assertEqual(d._formation, d.formation)
        d.method = "any"
        self.assertEqual(d.method, "any")
        # test the method setter using __setattr__
        self.assertRaises(ValueError, d.__setattr__, "method", "foo")
        
    def test_edge_buffer(self):
        d = RandomDefense("isolate", method="weighted", edge_buffer=1)
        self.assertEqual(d._edge_buffer, d.edge_buffer)
        d.edge_buffer = 2
        self.assertEqual(d.edge_buffer, 2)
        # test the edge_buffer setter using __setattr__
        self.assertRaises(TypeError, d.__setattr__, "edge_buffer", "foo")
        self.assertRaises(ValueError, d.__setattr__, "edge_buffer", -10)
        
    def test_ship_buffer(self):
        d = RandomDefense("isolate", method="weighted", ship_buffer=1)
        self.assertEqual(d._ship_buffer, d.ship_buffer)
        d.ship_buffer = 2
        self.assertEqual(d.ship_buffer, 2)
        # test the ship_buffer setter using __setattr__
        self.assertRaises(TypeError, d.__setattr__, "ship_buffer", "foo")
        self.assertRaises(ValueError, d.__setattr__, "ship_buffer", -10)
        
    def test_alignment(self):
        d = RandomDefense("isolate", method="weighted", alignment="NS")
        self.assertEqual(d._alignment, d.alignment)
        d.alignment = "EW"
        self.assertEqual(d.alignment, constants.Align.HORIZONTAL)
        # test the alignment setter using __setattr__
        self.assertRaises(ValueError, d.__setattr__, "alignment", "foo")
        
    def test_placement_for_ship(self):
        b = Board(10)
        ship_type = 5
        p = RandomDefense("Random", "any").placement_for_ship(b, ship_type)
        self.assertEqual(p.length, Ship.data[ship_type]["length"])
        self.assertIn(p.heading, set(["N","S","E","W"]))
        self.assertTrue(b.is_valid_placement(p))
        
        ntests = 50
        
        # test edge buffer
        d = RandomDefense("random", "any", edge_buffer=1)
        for _ in range(ntests):
            p = d.placement_for_ship(b, ship_type)
            coords = np.array(p.coords())
            self.assertTrue(np.all((coords > 0) *
                                   (coords < b.size-1)))
            
        # test ship buffer
        p2 = Placement((0,0), "N", 4)
        forbidden_coords = set([(0,0), (1,0), (2,0), (3,0), (4,0),
                                (0,1), (1,1), (2,1), (3,1)])
        b = Board(10)
        b.add_ship(4, placement = p2)
        d = RandomDefense("random", "any", ship_buffer=1)
        for _ in range(ntests):
            p = d.placement_for_ship(b, ship_type)
            for coord in p.coords():
                self.assertNotIn(coord, forbidden_coords)
                          
        # test alignments
        b = Board(10)
        dv = RandomDefense("random", "any", alignment=constants.Align.VERTICAL)
        dh = RandomDefense("random", "any", alignment=constants.Align.HORIZONTAL)
        for _ in range(ntests):
            pv = dv.placement_for_ship(b, ship_type)
            delta = np.diff(np.array(pv.coords()),axis=0)
            self.assertTrue(np.all(delta[:,1] == 0))
            ph = dh.placement_for_ship(b, ship_type)
            delta = np.diff(np.array(ph.coords()),axis=0)
            self.assertTrue(np.all(delta[:,0] == 0))
            
        
        # statistically test placements
        # average distance between ship placements should be closer than
        # random for clustered, and farther than random for isolated.
        # The distance should increase (decrease) when going from 
        # any --> weighted --> optimize methods for isolated (clustered)
        # formations. This may not be true for clustered, where any means
        # that
        ship_type = 2
        dcluster = RandomDefense(formation="cluster", method="weighted")
        drandom = RandomDefense(formation="random", method="weighted")
        disolate = RandomDefense(formation="isolate", method="weighted")
        
        b = Board(10)
        p0 = Placement((6,8), "N", Ship.data[3]["length"])
        p1 = Placement((5,7), "N", Ship.data[4]["length"])
        b.add_ship(3, placement=p0)
        b.add_ship(4, placement=p1)
        ave_dists = np.zeros((ntests, 3))
        for n in range(ntests):
            pr = drandom.placement_for_ship(b, ship_type)
            pc = dcluster.placement_for_ship(b, ship_type)
            pi = disolate.placement_for_ship(b, ship_type)
            dr = (np.mean(pr.placement_dist_matrix(p0))
                  + np.mean(pr.placement_dist_matrix(p1)))
            dc = (np.mean(pc.placement_dist_matrix(p0))
                  + np.mean(pc.placement_dist_matrix(p1)))
            di = (np.mean(pi.placement_dist_matrix(p0))
                  + np.mean(pi.placement_dist_matrix(p1)))
            ave_dists[n,:] = np.array((dc, dr, di))
            
        self.assertTrue(np.all(np.diff(np.mean(ave_dists, axis=0)) > 0))
            
    
#%% Offense

from collections import Counter
from battleship.offense import Offense, HunterOffense, Mode

class TestOffenseFunctions(unittest.TestCase):
    
    def test_init(self):
        # should not be able to instantiate abstract class Offense
        self.assertRaises(TypeError, Offense)

        
class TestHunterOffenseFunctions(unittest.TestCase):
    
    def test_init(self):
        
        hunter = HunterOffense("random")
        
                                
        # default parameters
        self.assertEqual(hunter._hunt_style, "random")
        self.assertEqual(hunter._hunt_pattern, "random")
        self.assertDictEqual(hunter.hunt_options, {'edge_buffer': 0,
                                                    'ship_buffer': 0,
                                                    'weight': 'any',
                                                    'rotate': 0,
                                                    'mirror': False,
                                                    'spacing': 0,
                                                    'secondary_spacing': 0,
                                                    'no_valid_targets': 'random',
                                                    'dist_method': 'dist2',
                                                    'user_data': None,
                                                    'note': None}
                             )
        self.assertDictEqual(hunter.kill_options, {'method': 'advanced', 
                                                    'user_data': None, 
                                                    'note': None}
                             )
        
        # check hunt_style parameters
        self.assertEqual(HunterOffense("RANDOM")._hunt_style, "random")
        self.assertEqual(HunterOffense(hunt_style="RANDOM")._hunt_style, "random")
        self.assertEqual(HunterOffense("pattern")._hunt_style, "pattern")
        # self.assertRaises(ValueError, HunterOffense, hunt_style="not allowed")
        
        # check hunt_pattern parameters (depend on hunt_style params)
        # hunt_stype = "random"
        self.assertEqual(HunterOffense("RANDOM")._hunt_pattern, "random") #default
        self.assertEqual(HunterOffense("random", "MAXPROB")._hunt_pattern, 
                         "maxprob")
        self.assertEqual(HunterOffense("random", "isolate")._hunt_pattern, 
                         "isolated")
        self.assertEqual(HunterOffense("random", "isolated")._hunt_pattern, 
                         "isolated")
        self.assertEqual(HunterOffense("random", "cluster")._hunt_pattern, 
                         "clustered")
        self.assertEqual(HunterOffense("random", "clustered")._hunt_pattern, 
                         "clustered")
        # self.assertRaises(ValueError, HunterOffense, hunt_style="pattern", 
        #                   hunt_pattern="maxprob")
        # self.assertRaises(ValueError, HunterOffense, hunt_pattern="foo")
        
        # hunt_style = "pattern"
        self.assertEqual(HunterOffense("pattern")._hunt_pattern, "grid")
        self.assertEqual(HunterOffense("pattern", "GRID")._hunt_pattern, "grid")
        self.assertEqual(HunterOffense("pattern", "diagonals")._hunt_pattern, 
                         "diagonals")
        self.assertEqual(HunterOffense("pattern", "spiral")._hunt_pattern, 
                         "spiral")
        # self.assertRaises(ValueError, HunterOffense, "pattern", 
        #                   hunt_pattern="random")
        # self.assertRaises(ValueError, HunterOffense, "pattern", 
        #                   hunt_pattern="foo")
        
        # test hunt_options
        default_options = {'edge_buffer': 0,
                           'ship_buffer': 0,
                           'weight': 'any',
                           'rotate': 0,
                           'mirror': False,
                           'spacing': 0,
                           'secondary_spacing': 0,
                           'no_valid_targets': 'random',
                           'dist_method': 'dist2',
                           'user_data': None,
                           'note': None
                           }
        self.assertDictEqual(HunterOffense(hunt_style="random").hunt_options,
                             default_options)
        self.assertDictEqual(HunterOffense(hunt_style="pattern").hunt_options,
                             default_options)
        self.assertEqual(HunterOffense(hunt_style="random", 
                                           hunt_options={"edge_buffer": 3}
                                           ).hunt_options["edge_buffer"],
                             3)
        self.assertEqual(HunterOffense(hunt_style="random", 
                                           hunt_options={"ship_buffer": 4}
                                           ).hunt_options["ship_buffer"],
                             4)
        self.assertEqual(HunterOffense(hunt_style="random", 
                                           hunt_options={"weight": 1.1}
                                           ).hunt_options["weight"],
                             1.1)
        self.assertEqual(HunterOffense(hunt_style="random", 
                                           hunt_options={"weight": "hits"}
                                           ).hunt_options["weight"],
                             "hits")
        # can technically add any key/value to hunt_options:
        self.assertEqual(HunterOffense(hunt_style="random", 
                                           hunt_options={"foo": "bar"}
                                           ).hunt_options["foo"],
                             "bar")
        
        # test kill_options
        default_options = {'method': 'advanced',
                           'user_data': None, 
                           'note': None}
        
        self.assertDictEqual(HunterOffense(hunt_style="random").kill_options,
                             default_options)
        self.assertDictEqual(HunterOffense(hunt_style="pattern").kill_options,
                             default_options)
        self.assertEqual(HunterOffense(hunt_style="random", 
                                           kill_options={"method": 3}
                                           ).kill_options["method"],
                             3)
        self.assertEqual(HunterOffense(hunt_style="random", 
                                           kill_options={"method": "basic"}
                                           ).kill_options["method"],
                             "basic")
        self.assertEqual(HunterOffense(hunt_style="random", 
                                           kill_options={"method": "dumb"}
                                           ).kill_options["method"],
                             "dumb")
        # value of kill_options is not controlled
        self.assertEqual(HunterOffense(hunt_style="random", 
                                           kill_options={"method": "foo"}
                                           ).kill_options["method"],
                             "foo")
        # any key/value pairs can technically be added:
        self.assertEqual(HunterOffense(hunt_style="random", 
                                           kill_options={"foo": "bar"}
                                           ).kill_options["foo"],
                             "bar")
        
        # need to test other key/value pairs once we enforce what keys and/or
        # values are allowed.
     
    # Basic method tests (should be performed before property tests below)
    def test_update_state(self):
        # In HUNT mode:
        #     If outcome was a HIT:
        #         If outcome was a SINK --> stay in HUNT mode
        #         Otherwise --> go to KILL mode
        #     If outcome was a MISS --> stay in HUNT mode
        
        # In KILL mode:
        #     If outcome was a HIT:
        #         If outcome was a SINK --> go to HUNT mode
        #         Otherwise --> stay in KILL mode
        #     If outcome was a MISS --> stay in KILL mode
        hunter = HunterOffense("random")
        self.assertEqual(hunter.state, Mode.HUNT)
        self.assertRaises(TypeError, hunter.update_state, None)
        self.assertRaises(TypeError, hunter.update_state, 1)
        self.assertRaises(TypeError, hunter.update_state, "foo")
        self.assertRaises(TypeError, hunter.update_state, [])
        self.assertRaises(KeyError, hunter.update_state, {})
        self.assertRaises(KeyError, hunter.update_state, {"foo": "bar"})
        
        hunter.update_state(Board.outcome((0,0), False))
        self.assertEqual(hunter.state, Mode.HUNT)
        
        hunter.update_state(Board.outcome((0,0), True, True, 1))
        self.assertEqual(hunter.state, Mode.HUNT)
        
        hunter.update_state(Board.outcome((0,0), True, False))
        self.assertEqual(hunter.state, Mode.KILL)
        
        hunter.update_state(Board.outcome((0,0), False))
        self.assertEqual(hunter.state, Mode.KILL)
        
        hunter.update_state(Board.outcome((0,0), True, False))
        self.assertEqual(hunter.state, Mode.KILL)
        
        hunter.update_state(Board.outcome((0,0), True, True, 1))
        self.assertEqual(hunter.state, Mode.HUNT)
        
    # Property tests
    
    def test_state(self):
        from battleship.offense import Mode
        hunter = HunterOffense("random")
        self.assertEqual(hunter.state, hunter._state)
        self.assertEqual(hunter.state, Mode.HUNT)
        b = Board(2)
        b.add_ship(1, (0,0), "N")
        hunter.update_state(b.incoming_at_coord((1,1)))
        # ship missed; still in hunt state
        self.assertEqual(hunter.state, Mode.HUNT)
        hunter.update_state(b.incoming_at_coord((0,0)))
        # ship hit; now in kill state
        self.assertEqual(hunter.state, Mode.KILL)
        hunter.update_state(b.incoming_at_coord((1,0)))
        # ship is now sunk; back to hunt state
        self.assertEqual(hunter.state, Mode.HUNT)
        
    def test_first_hit(self):
        # the first hit in the current state.
        hunter = HunterOffense("random")
        self.assertIsNone(hunter.first_hit)
        b = Board(10)
        b.add_ship(2, (0,0), "N")
        hunter.update_state(b.incoming_at_coord((9,9)))
        self.assertEqual(hunter.first_hit, (9,9))
        hunter.update_state(b.incoming_at_coord((0,0)))
        self.assertEqual(hunter.first_hit, (0,0))
        hunter.update_state(b.incoming_at_coord((0,1)))
        self.assertEqual(hunter.first_hit, (0,0))
        hunter.update_state(b.incoming_at_coord((1,0)))
        self.assertEqual(hunter.first_hit, (0,0))
        hunter.update_state(b.incoming_at_coord((2,0)))
        self.assertEqual(hunter.first_hit, (2,0))
    
    def test_last_hit(self):
        hunter = HunterOffense("random")
        history = []
        self.assertIsNone(hunter.last_hit(history))
        history = [Board.outcome((0,0),False)]
        self.assertIsNone(hunter.last_hit(history))
        history += [Board.outcome((1,0),True)]
        self.assertEqual(hunter.last_hit(history), (1,0))
        history += [Board.outcome((2,0),False)]
        self.assertEqual(hunter.last_hit(history), (1,0))
        history += [Board.outcome((3,0),False)]
        self.assertEqual(hunter.last_hit(history), (1,0))
        history += [Board.outcome((0,1),True)]
        self.assertEqual(hunter.last_hit(history), (0,1))
        history += [Board.outcome((1,1),True,True,1)]
        self.assertEqual(hunter.last_hit(history), (1,1))
        history += [Board.outcome((2,1),False)]
        self.assertEqual(hunter.last_hit(history), (1,1))
        history += [Board.outcome((2,1),False)]
        self.assertEqual(hunter.last_hit(history), (1,1))
        
    def test_set_to_hunt(self):
        hunter = HunterOffense("random")
        hunter._state = Mode.HUNT
        hunter.set_to_hunt(Board.outcome((5,6),False)) # since state is not changed, first_hit is not updated.
        self.assertEqual(hunter.state, Mode.HUNT)
        self.assertIsNone(hunter.first_hit)
        hunter._state = Mode.KILL
        hunter.set_to_hunt(Board.outcome((6,7),False))
        self.assertEqual(hunter.state, Mode.HUNT)
        self.assertEqual(hunter.first_hit, (6,7))
        
    def test_set_to_kill(self):
        hunter = HunterOffense("random")
        hunter._state = Mode.HUNT
        hunter.set_to_kill(Board.outcome((5,6),False)) # this sets first_hit
        self.assertEqual(hunter.state, Mode.KILL)
        self.assertEqual(hunter.first_hit, (5,6))
        hunter._state = Mode.KILL
        hunter.set_to_kill(Board.outcome((6,7),False)) # no state change, so first_hit is not updated. Still (5,6).
        self.assertEqual(hunter.state, Mode.KILL)
        self.assertEqual(hunter.first_hit, (5,6))
        hunter._state = Mode.HUNT
        hunter.set_to_kill(Board.outcome((6,7),False)) # now first_hit should update to (6,7) since state went from HUNT to KILL.
        self.assertEqual(hunter.state, Mode.KILL)
        self.assertEqual(hunter.first_hit, (6,7))
        
    # def test_outcomes_in_current_state(self):
    #     hunter = HunterOffense()
    #     self.assertListEqual(hunter.outcomes_in_current_state([]), [])
    #     outcome = Board.outcome((0,0), False)
    #     hunter.update_state(outcome)
    #     history += [outcome]
        
    def test_targets_on_line(self):
        b = Board(10)
        hunter = HunterOffense("random")
        
        # empty grid
        self.assertEqual(hunter.targets_on_line(b,(4,5),(1,0),False), [(5,5)])
        self.assertEqual(hunter.targets_on_line(b,(4,5),(-1,0),False), [(3,5)])
        self.assertEqual(set(hunter.targets_on_line(b,(4,5),(1,0),True)),
                         set([(3,5), (5,5)]))
        self.assertEqual(hunter.targets_on_line(b,(4,5),(0,1),False), [(4,6)])
        self.assertEqual(hunter.targets_on_line(b,(4,5),(0,-1),False), [(4,4)])
        self.assertEqual(set(hunter.targets_on_line(b,(4,5),(0,1),True)),
                         set([(4,6), (4,4)]))
        
        self.assertEqual(hunter.targets_on_line(b,(4,5),(5,0),False), [(5,5)])
        
        # near edge
        self.assertEqual(hunter.targets_on_line(b,(0,0),(1,0),False), [(1,0)])
        self.assertEqual(hunter.targets_on_line(b,(0,0),(-1,0),False), [])
        self.assertEqual(hunter.targets_on_line(b,(0,0),(1,0),True), [(1,0)])
        
        # on a ship
        b.update_target_grid(b.outcome((3,4),True))
        b.update_target_grid(b.outcome((2,4),False))
        b.update_target_grid(b.outcome((3,5),True))
        b.update_target_grid(b.outcome((3,6),True))
        #
        self.assertEqual(hunter.targets_on_line(b,(3,4),(1,0),True), [(4,4)])
        self.assertEqual(set(hunter.targets_on_line(b,(3,4),(0,1),True)), 
                         set([(3,3), (3,7)]))
        
        # if the line encounters a known sunk ship, it does not return a target
        # coord along that direction.
        b.update_target_grid(b.outcome((3,7),True,True,4))
        self.assertEqual(hunter.targets_on_line(b,(3,4),(0,1),True), [(3,3)])
        b.update_target_grid(b.outcome((3,3),True))
        self.assertEqual(set(hunter.targets_on_line(b,(3,4),(0,1),True)), 
                         set([(3,2)]))
        self.assertEqual(set(hunter.targets_on_line(b,(3,3),(0,1),True)), 
                         set([(3,2)]))
        self.assertEqual(set(hunter.targets_on_line(b,(3,3),(1,0),True)), 
                         set([(2,3), (4,3)]))
        
        # near a ship
        self.assertEqual(set(hunter.targets_on_line(b,(2,3),(0,1),True)), 
                         set([(2,2)]))
        self.assertEqual(set(hunter.targets_on_line(b,(2,3),(1,0),True)), 
                         set([(4, 3), (1, 3)]))
        # on a miss
        self.assertEqual(set(hunter.targets_on_line(b,(2,4),(1,0),True)), 
                         set([(1, 4)]))
        self.assertEqual(set(hunter.targets_on_line(b,(2,4),(0,1),True)), 
                         set([(2, 5), (2, 3)]))
        b.update_target_grid(b.outcome((4,3),True))
        self.assertEqual(set(hunter.targets_on_line(b,(3,3),(1,0),True)), 
                         set([(2, 3), (5, 3)]))
        b.update_target_grid(b.outcome((5,3),True))
        self.assertEqual(set(hunter.targets_on_line(b,(3,3),(1,0),True)), 
                         set([(2, 3), (6, 3)]))
        b.update_target_grid(b.outcome((6,3),True,True,3))
        self.assertEqual(set(hunter.targets_on_line(b,(3,3),(1,0),True)), 
                         set([(2, 3)]))
        
    def test_random_hunt_targets(self):
        b = Board(10)
        h = HunterOffense("random")
        history = []
        shots = [(r,c) for r in range(3) for c in range(10)]
        for coord in shots:
            outcome = b.outcome(coord, False)
            history += [outcome]
            b.update_target_grid(outcome)
        targets_and_probs = h.random_hunt_targets(b, history)
        for target, prob in targets_and_probs.items():
            self.assertNotIn(target, shots)
            self.assertEqual(prob, 1/(b.size**2 - 10*3))
            
    def test_hunt_targets(self):
        h = HunterOffense("random")
        b = Board(10)
        X = h.hunt_targets(b,[])
        X = list(X.values())
        self.assertTrue(len(X) == b.size**2)
        self.assertTrue((X[0] == X[-1]) 
                        and X[0] > 0 
                        and X[-1] > 0)
        
        h = HunterOffense("pattern", "grid")
        self.assertRaises(NotImplementedError, 
                          h.pattern_hunt_targets,
                          b, [])
        
    def test_kill_targets_from_placements(self):
        #   1 2 3 4 5 6 7 8 9 10
        # A - - - - - - - - - - 
        # B - - - - - - - - - -  
        # C - - - - 0 X - - - - <-- first hit 
        # D - - - - - X - 0 - - <-- second hit
        # E - - - 4 4 4 4 - - - 
        # F - - - - - - - - - - 
        # G - - - - - - - - - - 
        # H - - - - - - - - - - 
        # I - - - - - - - - - - 
        # J - - - - - - - - - - 
        
        b = Board(3)
        history = []
        h = HunterOffense("random")
        outcome = b.outcome((1,1),False)
        b.update_target_grid(outcome)
        history = [outcome]
        self.assertRaises(ValueError, h.kill_targets_from_placements,
                          b, history) # empty target grid    
        h.update_state(outcome)
        self.assertRaises(ValueError, h.kill_targets_from_placements,
                          b, history) # no hits
        
        b = Board(3)
        history = []
        outcome = b.outcome((1,1),True)
        b.update_target_grid(outcome)
        h.update_state(outcome)
        history = [outcome]
        X = h.kill_targets_from_placements(b,history)
        self.assertSetEqual(set(X), 
                            set([(2, 1), (1, 2), (0, 1), (1, 0)]))
        
        outcome = b.outcome((2,1),True)
        b.update_target_grid(outcome)
        h.update_state(outcome)
        history += [outcome]
        X = h.kill_targets_from_placements(b,history)
        self.assertSetEqual(set(X),
                            set([(0, 1)]))
        
    def test_kill_targets_around_last_hit(self):
        
        b = Board(10)
        h = HunterOffense("random")
        history = []
        self.assertRaises(ValueError, h.kill_targets_around_last_hit,
                          b, history) # empty target grid    
        outcome = b.outcome((1,1),True)
        b.update_target_grid(outcome)
        # history needs to be updated before last hit can be found.
        self.assertRaises(ValueError, h.kill_targets_around_last_hit,
                          b, history) 
        
        b = Board(3)
        h = HunterOffense("random")
        history = []
        outcome = b.outcome((1,1),True)
        b.update_target_grid(outcome)
        h.update_state(outcome)
        history = [outcome]
        X = h.kill_targets_around_last_hit(b,history)
        self.assertSetEqual(set(X), 
                            set([(2, 1), (1, 2), (0, 1), (1, 0)]))
        
        outcome = b.outcome((2,1),True)
        b.update_target_grid(outcome)
        h.update_state(outcome)
        history += [outcome]
        X = h.kill_targets_around_last_hit(b,history)
        self.assertEqual(set(X), set([(2,0), (2,2)]))
        
        b = Board(3)
        history = []
        res = b.outcome((2,1),True)
        b.update_target_grid(res); h.update_state(res); history += [res]
        res = b.outcome((1,1),True)
        b.update_target_grid(res); h.update_state(res); history += [res]
        res = b.outcome((1,0),False)
        b.update_target_grid(res); h.update_state(res); history += [res]
        X = h.kill_targets_around_last_hit(b,history)
        self.assertEqual(set(X), set([(1,2), (0,1)]))
        self.assertEqual(X[(1,2)], X[(0,1)])
        
        
        
    def test_kill_targets_from_shape(self):
        
        b = Board(10)
        h = HunterOffense("random")
        history = []
        self.assertRaises(ValueError, h.kill_targets_from_shape,
                          b, history) # empty target grid & history   
        outcome = b.outcome((1,1),True)
        b.update_target_grid(outcome)
        # history needs to be updated before last hit can be found.
        self.assertRaises(ValueError, h.kill_targets_from_shape,
                          b, history) 
        
        b = Board(3)
        h = HunterOffense("random")
        history = []
        res = b.outcome((1,1),True)
        b.update_target_grid(res); h.update_state(res); history += [res]
        X = h.kill_targets_from_shape(b,history)
        self.assertSetEqual(set(X), {(2, 1), (1, 2), (0, 1), (1, 0)})
        x0 = X[(0, 1)]
        self.assertTrue(all([x == x0 for x in X.values()]))
        res = b.outcome((2,1),True)
        b.update_target_grid(res); h.update_state(res); history += [res]
        X = h.kill_targets_from_shape(b,history)
        self.assertSetEqual(set(X), {(0, 1)})
        
    def test_kill_targets(self):
        
        # Each kill method should handle the below situation slightly 
        # differently. 
        # kill_targets_around_last_hit just returns the untargeted coords
        #   around the last hit.
        # kill_targets_from_shape looks at the last hit and the initial hit
        #   and finds untargeted coords that lie in the same row/col.
        # kill_targets_from_placements looks for the most likely spot nearest
        #   the inital hit (based on total # hits and possible placements),
        #   regardless of where last_hit was.
        b=Board(4); history=[]; h=HunterOffense("random")
        oc = b.outcome((2,2),True); 
        history += [oc]; b.update_target_grid(oc); h.update_state(oc)
        oc = b.outcome((2,3),True); 
        history += [oc]; b.update_target_grid(oc); h.update_state(oc)
        oc = b.outcome((3,2),True); 
        history += [oc]; b.update_target_grid(oc); h.update_state(oc)
        K0 = h.kill_targets_around_last_hit(b, history)
        K1 = h.kill_targets_from_shape(b, history)
        K2 = h.kill_targets_from_placements(b, history)
        
        self.assertSetEqual(set(K0), {(3,3), (3,1)})
        self.assertSetEqual(set(K1), {(1,2)})
        self.assertSetEqual(set(K2), {(1,2), (2,1)})
        
    def test_target(self):
        
        ntest = 100
        
        b = Board(4); 
        history = []; 
        h = HunterOffense("random", hunt_pattern="random", 
                          kill_options={"method":"basic"})
        
        oc = b.outcome((0,0),False); 
        history += [oc]; b.update_target_grid(oc); 
        h.update_state(oc);
        
        # fire a bunch of shots
        targets = Counter([h.target(b, history) for _ in range(ntest)])
        self.assertTrue(len(targets) > 4)   # this 4 is somewhat arbitrary
        
        # upon hitting, the possible targets should be much more restrictive.
        oc = b.outcome((0,1),True); 
        history += [oc]; b.update_target_grid(oc); 
        h.update_state(oc);
        targets = Counter([h.target(b, history) for _ in range(ntest)])
        self.assertEqual(len(targets), 2)
        
        # for an empty board, we should get all possible coords eventually.
        b = Board(3); 
        history = []; 
        h = HunterOffense("random")
        targets = Counter([h.target(b, history) for _ in range(3*3*100)])
        self.assertEqual(len(targets), 3*3)
        
    # The following are tested in the test_init method
    # def test_parse_hunt_options(self):
    # def test_parse_kill_options(self):
        

#%% Player

from battleship.player import HumanPlayer, AIPlayer, DummyPlayer

class TestHumanPlayer(unittest.TestCase):
    """
    All Player instances must implement the following attributes and methods:
    
    Abstract Properties/Methods:
    - player_type (property)
    - take_turn (method)
    - prepare_fleet (method)
    
    This particular subclass has the following properties and methods:
        
    Properties
    - target_select_mode
    - show_targets
    
    Other methods
    - __init__
    """
    
    def test_init(self):
        player = HumanPlayer()
        self.assertEqual(player.player_type, "Human")
        self.assertEqual(player.target_select_mode, "text")
        
    def test_player_type(self):
        player = HumanPlayer()
        self.assertEqual(player.player_type, "Human")
        
    def test_take_turn(self):
        print("unit test on HumanPlayer.take_turn requires "
              "interactive input. Not testing this method currently.")
        pass
    
    def test_prepare_fleet(self):
        print("unit test on HumanPlayer.prepare_fleet requires "
              "interactive input if the defense property is not an instance "
              "of Defense. Not testing this functionality currently.")
        placements = {1: Placement((0,0), "N", Ship(1).length),
                      2: Placement((0,1), "N", Ship(2).length),
                      3: Placement((0,2), "N", Ship(3).length),
                      4: Placement((4,7), "N", Ship(4).length),
                      5: Placement((7,0), "W", Ship(5).length)}
        player = HumanPlayer(defense=placements)
        self.assertDictEqual(player.defense, placements)
        player.board = Board(10)
        player.board.add_fleet(placements)
        self.assertDictEqual(player.board.ship_placements,
                             placements)
        
    def test_target_select_mode(self):
        player = HumanPlayer()
        self.assertEqual(player.target_select_mode, "text")
        self.assertEqual(player.target_select_mode, player._target_select_mode)
        # 'text' is the only valid mode as of now.
        self.assertRaises(ValueError, player.__setattr__, 
                          "target_select_mode", "foo")
        
    def test_input_placements(self):
        # requires interactive input.
        print("unit test on HumanPlayer.input_placements requires "
              "interactive input. Not testing this method currently.")
        pass
    
    def show_targets(self):
        
        player = HumanPlayer()
        self.assertFalse(player.show_targets)
        player.show_targets = True
        self.assertTrue(player.show_targets)
        
class TestAIPlayer(unittest.TestCase):
    
    def test_init(self):
        player = AIPlayer()
        self.assertEqual(player.player_type, "AI")
        player = AIPlayer('a', 'b', 'c')
        self.assertEqual(player.offense, 'a')
        self.assertEqual(player.defense, 'b')
        self.assertEqual(player.name, 'c')
        self.assertFalse(player.verbose)
        self.assertListEqual(player.shot_history, [])
        self.assertListEqual(player.outcome_history, [])
        self.assertListEqual(player.remaining_targets, [])
        self.assertIsNone(player._possible_targets)
        self.assertIsNone(player.board)
        self.assertIsNone(player.opponent)
        
    def test_player_type(self):
        player = AIPlayer()
        self.assertEqual(player.player_type, "AI")
    
    def test_prepare_fleet(self):
        player = AIPlayer()
        self.assertIsNone(player.board)
        # defense cannot be None for AI player
        self.assertRaises(ValueError, player.prepare_fleet) 
        
        # placement-based defense
        placements = {1: Placement((0,0),"N",2),
                      2: Placement((0,1),"N",3)}
        player = AIPlayer(defense = placements)
        self.assertIsNone(player.board)
        player.prepare_fleet()
        self.assertTrue(player.board)
        self.assertEqual(len(player.board.fleet), len(placements))
        self.assertDictEqual(player.board.ship_placements, placements)
        # check that ocean grid has appropriate ship_type numbers at the
        # placement coords
        self.assertEqual(player.board.ocean_grid[placements[1].coords()[-1]], 1)
        self.assertEqual(player.board.ocean_grid[placements[2].coords()[-1]], 2)
        
        # random defense
        player = AIPlayer(defense = RandomDefense())
        self.assertIsNone(player.board)
        player.prepare_fleet()
        self.assertTrue(player.board)
        self.assertEqual(len(player.board.fleet), len(Ship.data))
        placements = player.board.ship_placements
        # check that ocean grid has appropriate ship_type numbers at the
        # placement coords
        for k in Ship.types:
            for coord in placements[k].coords():
                self.assertEqual(player.board.ocean_grid[coord], k)
        
    def test_take_turn(self):
        # "take_turn" should do the same thing as fire_at_target and update
        # the offense.
        ship_type = 2
        ship_len = Ship.data[ship_type]["length"]
        p1 = AIPlayer(offense = HunterOffense("random"), 
                      defense = {ship_type: Placement((0,0), "N", ship_len)})
        p2 = AIPlayer(offense = HunterOffense("random"), 
                      defense = {ship_type: Placement((0,9), "N", ship_len)})
        p1.prepare_fleet()
        p2.prepare_fleet()
        p1.opponent = p2
        
        # test that take_turn does all of the same actions as fire_at_target
        self.assertTrue(np.all(p1.board.target_grid 
                               == constants.TargetValue.UNKNOWN))
        self.assertEqual(len(p1.remaining_targets), 100)
        self.assertListEqual(p1.outcome_history, [])
        self.assertIsNone(p1.last_target)
        self.assertIsNone(p1.last_outcome)
        self.assertTrue(p1.offense.state, Mode.HUNT)    # Hunt mode (offense)
        
        p1.take_turn()  # pick a random target to fire at
        
        self.assertTrue(np.sum(p1.board.target_grid 
                               > constants.TargetValue.UNKNOWN) == 1)
        row,col = np.where(p1.board.target_grid > constants.TargetValue.UNKNOWN)
        coord = (row[0], col[0])
        self.assertEqual(len(p1.remaining_targets), 99)
        self.assertTupleEqual(p1.last_target, coord)
        self.assertFalse(p1.last_target in p1.remaining_targets)
        self.assertFalse(coord in p1.remaining_targets)
        self.assertEqual(len(p1.outcome_history), 1)
        self.assertTupleEqual(p1.outcome_history[0]["coord"], coord)
        
        # check if offense can be put into kill mode (i.e., that offense
        # is updated))
        p1 = AIPlayer(offense = HunterOffense("random"), 
                      defense = {ship_type: Placement((0,0), "N", ship_len)})
        p1.prepare_fleet()
        p2 = AIPlayer(offense = HunterOffense("random"), 
                      defense = RandomDefense())
        p2.prepare_fleet()
        p1.opponent = p2
        
        self.assertEqual(p1.offense.state, Mode.HUNT)
        p1.take_turn()
        count = 0
        while not p1.last_outcome['hit']:
            self.assertEqual(p1.offense.state, Mode.HUNT)
            p1.take_turn()
            if count >= 100:
                raise Exception("No ships found after 100 shots.")
                break
            count += 1
        self.assertEqual(p1.offense.state, Mode.KILL)
    
    # Below methods belong to Player class, but need a concrete instance
    # to test.
    def test_name(self):
        player = AIPlayer('a','b','Eve')
        self.assertEqual(player.name, 'Eve')
        player = AIPlayer(name='Eve')
        self.assertEqual(player.name, 'Eve')   
        
    def test_offense(self):
        player = AIPlayer('x')
        self.assertEqual(player.offense, 'x')
        player = AIPlayer(offense = 'x')
        self.assertEqual(player.offense, 'x')
        player = AIPlayer(defense = 'y', offense = 'x')
        self.assertEqual(player.offense, 'x')
        offense = HunterOffense("random")
        player = AIPlayer(offense)
        self.assertIs(offense, player.offense)
        
    def test_defense(self):
        player = AIPlayer('x')
        self.assertEqual(player.defense, None)
        player = AIPlayer(defense = 'x')
        self.assertEqual(player.defense, 'x')
        player = AIPlayer(defense = 'y', offense = 'x')
        self.assertEqual(player.defense, 'y')
        defense = RandomDefense("cluster")
        player = AIPlayer(defense = defense)
        self.assertIs(defense, player.defense)
        
    def test_outcome_history(self):
        player = AIPlayer()
        self.assertEqual(player._outcome_history, player.outcome_history)
        
    def test_add_outcome_to_history(self):
        
        player = AIPlayer()
        outcome = Board.outcome((3,3), True, True, 2)
        self.assertEqual(len(player.outcome_history), 0)
        player.add_outcome_to_history(outcome)
        self.assertEqual(len(player.outcome_history), 1)
        self.assertDictEqual(player.outcome_history[0], outcome)
        # Input dict does not have to be an outcome dictionary
        bad_outcome = {"foo": "bar"}
        player.add_outcome_to_history(bad_outcome)
        self.assertEqual(len(player.outcome_history), 2)
        self.assertDictEqual(player.outcome_history[-1], bad_outcome)
        
    def test_board(self):
        player = AIPlayer()
        self.assertIsNone(player.board)
        
        b = Board(10)
        b.add_ship(5, (2,2), "N")
        b.update_target_grid(Board.outcome((0,0),False))
        
        player.board = b
        self.assertIs(player.board, b)
        self.assertTrue(5 in player.board.fleet)
        for k in [1,2,3,4]:
            self.assertFalse(k in player.board.fleet)
        
    def test_opponent(self):
        p1 = AIPlayer(name = "Frances")
        p2 = AIPlayer(name = "Gloria")
        self.assertEqual(p1.name, "Frances")
        self.assertEqual(p2.name, "Gloria")
        p1.opponent = p2
        self.assertIs(p1.opponent, p2)
        self.assertIs(p2.opponent, p1)
        self.assertEqual(p1.opponent.name, "Gloria")
        self.assertEqual(p2.opponent.name, "Frances")
        
    def test_remaining_targets(self):
        board_size = 8
        player = AIPlayer()
        self.assertListEqual(player.remaining_targets, [])
        player.board = Board(board_size)
        self.assertEqual(len(player.remaining_targets), board_size**2)
        for r in range(board_size):
            for c in range(board_size):
                self.assertTrue((r,c) in player.remaining_targets)
        self.assertTrue((board_size, board_size) not in player.remaining_targets)
        p2 = AIPlayer()
        p2.board = Board(board_size)
        player.opponent = p2
        target = (1,2)
        player.fire_at_target(target)
        self.assertFalse(target in player.remaining_targets)
        
    def test_verbose(self):
        player = AIPlayer()
        self.assertTrue(player.verbose == player._verbose)
        self.assertFalse(player.verbose)
        player.verbose = True
        self.assertTrue(player.verbose)
        
    # Computed Properties
    def test_shot_history(self):
        p1 = AIPlayer()
        p2 = AIPlayer()
        self.assertListEqual(p1.shot_history, [])
        
        p1.board = Board(10)
        p2.board = Board(10)
        p2.board.add_ship(1,(0,0),"N")
        p1.opponent = p2
        
        self.assertListEqual(p1.shot_history, [])
        p1.fire_at_target((0,1))
        p1.fire_at_target((0,0))
        self.assertListEqual(p1.shot_history, [(0,1), (0,0)])
        
    def test_last_target(self):
        p1 = AIPlayer()
        p2 = AIPlayer()
        self.assertIsNone(p1.last_target)
        
        p1.board = Board(10)
        p2.board = Board(10)
        p2.board.add_ship(1,(0,0),"N")
        p1.opponent = p2
        
        self.assertIsNone(p1.last_target)
        p1.fire_at_target((0,1))
        self.assertTupleEqual(p1.last_target, (0,1))
        p1.fire_at_target((0,0))
        self.assertTupleEqual(p1.last_target, (0,0))
        
    def test_last_outcome(self):
        player = AIPlayer()
        self.assertIsNone(player.last_outcome)
        outcome = Board.outcome((3,3), True, True, 2)
        player.add_outcome_to_history(outcome)
        self.assertEqual(type(player.last_outcome), dict)
        self.assertDictEqual(player.last_outcome, outcome)
        outcome = Board.outcome((4,4), False)
        player.add_outcome_to_history(outcome)
        self.assertDictEqual(player.last_outcome, outcome)
        
        # Input dict does not have to be an outcome dictionary
        bad_outcome = {"foo": "bar"}
        player.add_outcome_to_history(bad_outcome)
        self.assertDictEqual(player.last_outcome, bad_outcome)
        self.assertDictEqual(player.last_outcome, player.outcome_history[-1])
        
    # Other methods
    def test_reset(self):
        p1 = AIPlayer()
        p1.board = Board(10)
        p2 = AIPlayer()
        p2.board = Board(10)
        p2.board.add_ship(1,(0,0),"N")
        p1.opponent = p2
        
        self.assertTrue(len(p2.board.fleet) > 0)
        p2.reset()
        self.assertEqual(len(p2.board.fleet), 0)
        
        p2.board.add_ship(1,(0,0),"N")
        p1.fire_at_target((0,0))
        p2.fire_at_target((1,0))
        self.assertTrue(np.any(p2.board.target_grid 
                               != constants.TargetValue.UNKNOWN))
        self.assertTrue(p2.board.damage_at_coord((0,0)) == 1)
        self.assertTrue(len(p2.remaining_targets) == 99)
        self.assertTrue(len(p2.outcome_history) == 1)
        
        p2.reset()
        self.assertTrue(np.all(p2.board.target_grid 
                               == constants.TargetValue.UNKNOWN))
        for coord in p2.board.all_coords():
            self.assertEqual(p2.board.damage_at_coord(coord), 0)
        self.assertTrue(len(p2.remaining_targets) == 100)
        self.assertTrue(len(p2.outcome_history) == 0)
        
    def test_copy_history_from(self):
        p1 = AIPlayer()
        p1.board = Board(10)
        p2 = AIPlayer()
        p2.board = Board(10)
        p2.board.add_ship(1,(0,0),"N")
        p1.opponent = p2
        p1.fire_at_target((0,0))
        p1.fire_at_target((0,1))
        p1._game_history = "foobar"
        self.assertEqual(len(p1.outcome_history), 2)
        self.assertEqual(len(p1.remaining_targets), 98)
        p3 = AIPlayer()
        p3.board = Board(10)
        self.assertEqual(len(p3.outcome_history), 0)
        self.assertEqual(len(p3.remaining_targets), 100)
        self.assertIsNone(p3._game_history)
        p3.copy_history_from(p1)
        self.assertEqual(len(p3.outcome_history), 2)
        self.assertListEqual(p3.outcome_history, p1.outcome_history)
        self.assertEqual(len(p3.remaining_targets), 98)
        self.assertListEqual(p3.remaining_targets, p1.remaining_targets)
        self.assertEqual(p3._game_history, p1._game_history)
        self.assertEqual(p3._game_history, "foobar")
        
    def test_is_alive(self):
        p1 = AIPlayer()
        p1.board = Board(10)
        p1.board.add_ship(1,(0,0),"N")
        self.assertTrue(p1.is_alive())
        p1.board.incoming_at_coord((0,0))
        p1.board.incoming_at_coord((1,0))
        self.assertFalse(p1.is_alive())
        
    def test_fire_at_target(self):
        p1 = AIPlayer(HunterOffense("random"))
        p1.board = Board(10)
        p2 = AIPlayer()
        p2.board = Board(10)
        p2.board.add_ship(1,(0,0),"N")
        p1.opponent = p2
        
        self.assertEqual(p1.board.target_grid[(0,0)], 
                         constants.TargetValue.UNKNOWN)
        self.assertEqual(len(p1.remaining_targets), 100)
        self.assertTrue((0,0) in p1.remaining_targets)
        self.assertListEqual(p1.outcome_history, [])
        self.assertTrue(p1.offense.state, Mode.HUNT)    # Hunt mode
        
        p1.fire_at_target((0,0))
        self.assertEqual(p1.board.target_grid[(0,0)], 
                         constants.TargetValue.HIT)
        self.assertEqual(len(p1.remaining_targets), 99)
        self.assertFalse((0,0) in p1.remaining_targets)
        self.assertEqual(len(p1.outcome_history), 1)
        self.assertTrue(p1.offense.state, Mode.KILL)    # Kill mode
        
        p1.fire_at_target((1,0))
        self.assertTrue(p1.board.target_grid[(1,0)] >= 1) # Sunk patrol boat
        self.assertEqual(len(p1.remaining_targets), 98)
        self.assertFalse((1,0) in p1.remaining_targets)
        self.assertEqual(len(p1.outcome_history), 2)
        self.assertTrue(p1.offense.state, Mode.HUNT)    # Hunt mode
        
    def test_place_fleet_using_defense(self):
        p1 = AIPlayer(defense = RandomDefense())
        # Must create a board before placing fleet
        self.assertRaises(AttributeError, 
                          p1.place_fleet_using_defense, p1.defense)
        p1.board = Board(10)
        self.assertTrue(len(p1.board.fleet) == 0)
        self.assertTrue(np.sum(p1.board.ocean_grid.flatten()) == 0)
        p1.place_fleet_using_defense(p1.defense)
        self.assertTrue(len(p1.board.fleet) == 5)
        self.assertTrue(len(p1.board.ship_placements) == 5)
        self.assertTrue(np.sum(p1.board.ocean_grid.flatten()) >= 1)
        
    def test_show_possible_targets(self):
        pass
    
    def test_stats(self):
        # stats returns a dict with specific keys
        quantities = {'hits', 'nshots', 'ships_mean', 'ships_sank', 
                      'ships_var', 'shots_mean', 'shots_var'}
        p1 = AIPlayer(defense=RandomDefense(), offense=HunterOffense("random"))
        p1.prepare_fleet()      
        stats = p1.stats()
        self.assertSetEqual(set(stats.keys()), quantities)
        self.assertSetEqual(quantities.difference(set(stats.keys())), 
                            set())
        # check individual quantities
        self.assertEqual(stats['nshots'], 0)
        self.assertEqual(len(stats['hits']), 0)
        self.assertEqual(len(stats['ships_sank']), 0)
        self.assertIsNone(stats['shots_mean'])
        self.assertIsNone(stats['shots_var'])
        self.assertEqual(len(stats['ships_mean']), 2) # row/col center of mass
        self.assertEqual(len(stats['ships_var']), 2) # row/col variances
        self.assertTrue(np.all(stats['ships_mean'] > 0)) # row/col center of mass
        self.assertTrue(np.all(stats['ships_var'] > 0)) # row/col variances
        
        p2 = AIPlayer(defense=RandomDefense(), offense=HunterOffense("random"))
        p1.opponent = p2 
        p2.prepare_fleet()
        p1.fire_at_target((0,0))
        p2.fire_at_target((0,0))
        p1.fire_at_target(p2.board.ship_placements[1].coords()[0])
        p2.fire_at_target(p1.board.ship_placements[1].coords()[0])
        stats = p1.stats()
        
        # Make sure taking turns does not affect stat keys
        self.assertSetEqual(set(stats.keys()), quantities)
        self.assertSetEqual(quantities.difference(set(stats.keys())), 
                            set())
        self.assertSetEqual(set(p2.stats().keys()), quantities)
        self.assertSetEqual(quantities.difference(set(p2.stats().keys())), 
                            set())
        # check individual quantities
        self.assertEqual(stats['nshots'], 2)
        self.assertListEqual(list(stats['hits']), [False, True])
        self.assertEqual(len(stats['ships_sank']), 0)
        self.assertEqual(len(stats['ships_mean']), 2) # row/col center of mass
        self.assertEqual(len(stats['ships_var']), 2) # row/col variances
        self.assertTrue(np.all(stats['ships_mean'] > 0)) # row/col center of mass
        self.assertTrue(np.all(stats['ships_var'] > 0)) # row/col variances
        
#%% Game
from battleship.game import Game

class TestGame(unittest.TestCase):
    
    def test_init(self):
        game = Game('a','b')
        self.assertEqual(game.player1, 'a')
        self.assertEqual(game.player2, 'b')
        self.assertTrue(isinstance(game.salvo, bool))
        self.assertEqual(game.salvo, False)
        self.assertEqual(game.game_id, None)
        self.assertTrue(isinstance(game.verbose, bool))
        self.assertEqual(game.verbose, True)
        self.assertTrue(isinstance(game.show, bool))
        self.assertEqual(game.show, False)
        
        self.assertEqual(game.turn_count, 0)
        self.assertIsNone(game.winner)
        self.assertIsNone(game.loser)
        self.assertEqual(game.ready, False)
        
        self.assertRaises(TypeError, Game, 'a','b','c') # salvo parameter must be a bool
        
    def test_salvo(self):
        game = Game('foo','bar')
        game.salvo = True
        self.assertTrue(game.salvo)
        game.salvo = False
        self.assertTrue(game.salvo)
        self.assertRaises(TypeError, game.__setattr__, "salvo", "NOT ALLOWED")
        self.assertRaises(TypeError, game.__setattr__, "salvo", 0)
        
    def test_game_id(self):
        game = Game('foo','bar')
        game.game_id = '12345678'
        self.assertEqual(game.game_id, '12345678')
        game.game_id = 12345678
        self.assertEqual(game.game_id, 12345678)
        
    def test_verbose(self):
        game = Game('foo','bar')
        game.verbose = True
        self.assertTrue(game.verbose)
        game.verbose = False
        self.assertFalse(game.verbose)
        self.assertRaises(TypeError, game.__setattr__, "verbose", "NOT ALLOWED")
        self.assertRaises(TypeError, game.__setattr__, "verbose", 0)
        
    def test_show(self):
        game = Game('foo','bar')
        game.show = True
        self.assertTrue(game.show)
        game.show = False
        self.assertFalse(game.show)
        self.assertRaises(TypeError, game.__setattr__, "show", "NOT ALLOWED")
        self.assertRaises(TypeError, game.__setattr__, "show", 0)
        
    def test_setup(self):
        p1 = DummyPlayer()
        p2 = DummyPlayer()
        game = Game(p1, p2)
        # p1 and p2 need to be reset since their fleets are placed
        self.assertRaises(ValueError, game.setup) 
        game.player1.reset()
        game.player1.reset()
        game.setup()
        self.assertEqual(len(game.player1.board.fleet), 5)
        self.assertEqual(len(game.player1.board.fleet), 5)
        self.assertEqual(np.sum(game.player1.board.target_grid !=
                                constants.TargetValue.UNKNOWN), 0)
        self.assertEqual(np.sum(game.player2.board.target_grid !=
                                constants.TargetValue.UNKNOWN), 0)
        self.assertTrue(game.ready)
        self.assertIsNone(game.winner)
        self.assertIsNone(game.loser)
        self.assertEqual(game.turn_count, 0)
        
    def test_reset(self):
        p1 = DummyPlayer()
        p2 = DummyPlayer()
        p1.reset()
        p2.reset()
        game = Game(p1, p2)
        game.setup()

        self.assertEqual(len(game.player1.board.fleet), 5)
        self.assertEqual(len(game.player2.board.fleet), 5)
        game.play_one_turn(game.player1, game.player2)
        self.assertEqual(np.sum(game.player1.board.target_grid !=
                                constants.TargetValue.UNKNOWN), 1)
        self.assertEqual(np.sum(game.player2.board.target_grid !=
                                constants.TargetValue.UNKNOWN), 1)
        self.assertEqual(game.turn_count, 1)
        self.assertFalse(game.ready)
        
        game.reset()
        self.assertEqual(len(game.player1.board.fleet), 0)
        self.assertEqual(len(game.player2.board.fleet), 0)
        self.assertEqual(np.sum(game.player1.board.target_grid !=
                                constants.TargetValue.UNKNOWN), 0)
        self.assertEqual(np.sum(game.player2.board.target_grid !=
                                constants.TargetValue.UNKNOWN), 0)
        self.assertEqual(game.turn_count, 0)
        self.assertFalse(game.ready) # need to prepare fleets        
        
    def test_play_one_turn(self): 
        p1 = DummyPlayer()
        p2 = DummyPlayer()
        p1.reset()
        p2.reset()
        game = Game(p1, p2)
        game.setup()
        
        keep_going = game.play_one_turn(game.player1, game.player2)
        self.assertTrue(keep_going)
        self.assertEqual(np.sum(game.player1.board.target_grid !=
                                constants.TargetValue.UNKNOWN), 1)
        self.assertEqual(np.sum(game.player2.board.target_grid !=
                                constants.TargetValue.UNKNOWN), 1)
        self.assertEqual(game.turn_count, 1)
        
        for n in range(2,10):
            keep_going = game.play_one_turn(game.player1, game.player2)
            self.assertTrue(keep_going)
            self.assertEqual(game.turn_count, n)
        
    def test_play(self):    
        p1 = DummyPlayer()
        p2 = DummyPlayer()
        p1.reset()
        p2.reset()
        game = Game(p1, p2)
        
        outcome = game.play(1, 100)     # 100 max turns, in case something is off with parameter inputs
        self.assertTrue('winner' in outcome)
        self.assertTrue('loser' in outcome)
        self.assertTrue('first_move' in outcome)
        self.assertTrue('first_player' in outcome)
        self.assertTrue('second_player' in outcome)
        self.assertTrue('turn_count' in outcome)
        self.assertTrue('max_turns' in outcome)
        self.assertTrue('tie' in outcome)
        self.assertDictEqual(outcome,
                             {'winner': None,
                              'loser': None,
                              'first_move': 1,
                              'first_player': p1,
                              'second_player': p2,
                              'turn_count': 0,
                              'max_turns': 100,
                              'tie': False})
        
        game.setup()
        outcome = game.play(1, 100)
        self.assertDictEqual(outcome,
                             {'winner': None,
                              'loser': None,
                              'first_move': 1,
                              'first_player': p1,
                              'second_player': p2,
                              'turn_count': 100,
                              'max_turns': 100,
                              'tie': True})
        
        game.reset()
        game.setup()
        outcome = game.play(2, 100)
        self.assertDictEqual(outcome,
                             {'winner': None,
                              'loser': None,
                              'first_move': 2,
                              'first_player': p2,
                              'second_player': p1,
                              'turn_count': 100,
                              'max_turns': 100,
                              'tie': True})
        
        self.assertRaises(ValueError, game.play, 3)
        
        game = Game(AIPlayer(HunterOffense("random"), 
                             RandomDefense("random"),
                             name="Frances"), 
                    AIPlayer(HunterOffense("random"), 
                             RandomDefense("random"),
                             name="Albert"))
        game.setup()
        outcome = game.play(1, 1000)
        self.assertTrue(((outcome['winner'] == game.player1) 
                         and (outcome['loser'] == game.player2))
                        or ((outcome['winner'] == game.player2) 
                            and (outcome['loser'] == game.player1)))
        self.assertTrue(outcome['first_player'] == game.player1)
        self.assertTrue(outcome['turn_count'] > 0)
        self.assertTrue(outcome['turn_count'] <= 1000)
        self.assertTrue(outcome['max_turns'] == 1000)
        if outcome['turn_count'] < outcome['max_turns']:
            self.assertFalse(outcome['tie'])
        
                        
        
    def test_report_turn_outcome(self):    
        pass
    
    def test_print_outcome(self):    
        pass
        
#%% Viz

#%% utils

# 