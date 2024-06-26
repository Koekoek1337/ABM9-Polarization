# This is the test file for the main.py file
from mesa.time import RandomActivation
from mesa.space import SingleGrid
import unittest
import os
import sys

from polarization.core import util
from polarization.core.model import CityModel

def clear_model(model: CityModel):
    model.agents = []
    model.n_agents = 0
    model.schedule = RandomActivation(model)
    model.grid = SingleGrid(model.sidelength, model.sidelength, torus=False)

class TestModel(unittest.TestCase):

    def setUp(self) -> None:
        self.model = CityModel()
        return super().setUp()

    def test_model_initialisation(self):
        assert self.model != None
