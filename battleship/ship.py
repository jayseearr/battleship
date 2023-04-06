#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 21:11:45 2023

@author: jason

Class definition for Ship object

"""

#%%

### Imports ###

import numpy as np

# Imports from this package
from . import constants


#%%

### Class Definition ###
class Ship:
    
    _data = {constants.ShipType.PATROL: {"name": "Patrol".title(), 
                                         "length": 2},
             constants.ShipType.DESTROYER: {"name": "Destroyer".title(), 
                                            "length": 3},
             constants.ShipType.SUBMARINE: {"name": "Submarine".title(), 
                                            "length": 3},
             constants.ShipType.BATTLESHIP: {"name": "Battleship".title(), 
                                             "length": 4},
             constants.ShipType.CARRIER: {"name": "Carrier".title(), 
                                          "length": 5}
             }
    
    def __init__(self, ship_type):
        """
        Initialize a ship object with type ship_type, 

        Parameters
        ----------
        ship_type : int, ShipType (EnumInt), or str
            An integer value (or equivalent EnumInt) between 1 and 5, or
            a string that describes the type of ship desired. Valid strings
            and their corresponding integer values are:
                Patrol (1)
                Destroyer (2)
                Submarine (3)
                Battleship (4)
                Carrier (5)

        Returns
        -------
        An instance of Ship with properties corresponding the the ship_type
        parameter. Each instance has properties:
            ship_id (ShipType enum)
            name (str)
            length (int)
            damage (numpy 1d array with length equal to the length property)
            

        """
        if isinstance(ship_type, str):
            ship_type = Ship.type_for_name(ship_type)
        self._ship_id = constants.ShipType(ship_type)
            
        self.name = Ship.data[self._ship_id]["name"]
        self.length = Ship.data[self._ship_id]["length"]
        self.damage = np.zeros(self.length, dtype=np.int8)
        
    def __len__(self):
        """
        Return the length of the ship, which is equal to both the number
        of adjacent coordinates on the board that the ship occuppies, as well
        as the number of damage slots on the ship.
        
        Returns
        -------
        int
            The number of slots on the ship insance.
        """
        return self.length
    
    def __str__(self):
        """
        Returns a string listing the Ship instance name, length, damage,
        and whether the ship is sunk.

        Returns
        -------
        str
        
        """
        s = (f"{self.name} (id = {self._ship_id}) - {self.length} slots, " 
              f"damage: {str(self.damage)}")
        if self.is_sunk():
            s += " (SUNK)"
        return(s)
    
    def __repr__(self):
        """
        Returns a string representation that can be evaluated to recreate
        the object

        Returns
        -------
        str
        
        """
        return f"Ship({self.ship_id})"        
        
    @property
    def ship_id(self):
        return self._ship_id
    
    @classmethod
    def type_for_name(self, name):
        """
        Returns the ship type integer that corresponds to the input ship name
        string. The output is a value of the ShipType enum, which is equivalent
        to an integer.

        Parameters
        ----------
        name : str
            A string description of the desired ship type. One of 'patrol',
            'destroyer', 'submarine' (or 'sub'), 'battleship', or 'carrier'.
            Case insensitive.

        Returns
        -------
        ShipType (EnumInt)
            An integer value 1-5 that can be used as a key for accessing
            info about the particular ship type from the 'data' class
            method.

        """
        name = name.title()
        try:
            idx = [ship_dict["name"] for ship_dict in 
                   Ship.data.values()].index(name)
        except ValueError:
            if name == "Sub":
                name += "marine"
            elif name == "Patrol Boat":
                name = "Patrol"
            idx = [ship_dict["name"] for ship_dict in 
                   Ship.data.values()].index(name)
        return list(Ship.data.keys())[idx]
    
    @classmethod
    @property
    def data(cls):
        """
        Returns a dictionary containing name and length data for each type
        of ship. The dictionary's keys are the ShipType Enum values.

        Parameters
        ----------
        None

        Returns
        -------
        dict
            A dictionary with the following keys. The corresponding value
            gives the value of property described by the key for the ship type
            given by the ship_type input (1 for Patrol, 2 for Destroyer, etc.)
            'name':     The name of the ship type.
            'length':   The number of coordinates the ship occupied on a board
                        (equal to number of times it can be hit before sinking).

        """
        return cls._data
    
    @classmethod
    @property
    def types(cls):
        """
        Returns a list of the valid ShipType values. For a typical game
        with 5 different ship types, this list will be the equivalent of
        [1,2,3,4,5], with each element actually set to EnumInts 
        ShipType(1), ShipType(2), etc.
        
        Returns
        -------
        ship_types : list
        

        """
        return list(range(1, len(cls.data) + 1))
        
    # Combat Methods
    
    def hit(self, slot):
        """
        Adds a point of damage to the ship at the input slot position.

        Parameters
        ----------
        slot : int
            The slot at which the ship will be damaged. Conforms to standard
            list indexing, so slot should be on the interval [0, ship.length].

        Raises
        ------
        ValueError
            If the slot parameter falls outside of the valid index range for
            the ship's length.

        Returns
        -------
        int     The total amage at the input slot.

        """
        if slot < 0 or slot >= self.length:
            raise ValueError("Hit at slot #" + str(slot) + " is not valid; "
                             "must be 0 to " + str(self.length-1))
        self.damage[slot] += 1
        if self.damage[slot] > 1:
            Warning("Slot " + str(slot) + " is already damaged.")
        return(self.damage[slot])
    
    def is_afloat(self):
        """
        Returns True if the ship has not been sunk (i.e., if it has at least
        one slot that has not taken damage).

        Returns
        -------
        Bool    True if the ship is still afloat, False if it is not.

        """
        
        return(any(dmg < 1 for dmg in self.damage))

    def is_sunk(self):
        """
        Returns True if the ship has been sunk (i.e., if it has at least
        one damage level at each slot).

        Returns
        -------
        Bool    True if the ship is sunk, False if it is afloat.

        """
        return(all(dmg > 0 for dmg in self.damage))