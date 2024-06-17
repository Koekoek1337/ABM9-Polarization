import mesa
import networkx as nx
import numpy as np

from agent import PolarizationAgent

from typing import Any


OPINION_A = -1
OPINION_B = 1

class PolarizationModel(mesa.Model):
    """

    """
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self.graph = nx.Graph()
        """Graph object for use in network space, for ease of acces with creating and removing nodes/edges"""

        self.space = mesa.space.NetworkGrid(self.graph)
        """Mesa Network space"""
        
        self.scheduler = mesa.time.SimultaneousActivation(self, None)
        """Mesa Scheduler object"""

        self.nAgents: int = 0 
        """Amount of agents present in the model (source of unique ID's)"""

        self.opinionA = OPINION_A
        self.opinionB = OPINION_B
        self.opinionDist = np.abs(OPINION_A) + np.abs(OPINION_B)

    def addAgent(self, opinion: float, conformity: float ,tolerance: float):
        """
        # TODO: Testing
        
        Add an agent to the model and scheduler
        """
        self.graph.add_node(self.nAgents)
        newAgent = PolarizationAgent(self.nAgents, self, opinion, conformity, tolerance)
        self.scheduler.add(newAgent)
        self.nAgents += 1