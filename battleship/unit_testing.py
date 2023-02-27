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
        from battleship import constants
        
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
        self.assertTrue(np.all(b.is_fleet_afloat() 
                               == np.array([True, True])))
        for i in range(len(b.fleet[3].damage)):
            b.fleet[3].damage[i] = 1.
        self.assertTrue(np.all(b.is_fleet_afloat() 
                               == np.array([True, False])))
        for i in range(len(b.fleet[1].damage)):
            b.fleet[1].damage[i] = 1.
        self.assertTrue(np.all(b.is_fleet_afloat() 
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
    
#%% Offense

#%% Defense

#%% Player

#%% Game

#%% Viz

#%% utils

# 