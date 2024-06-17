import mesa
import networkx as nx
import numpy as np

from agent import PolarizationAgent

from typing import Any, Tuple


OPINION_A = -1
OPINION_B = 1

class PolarizationModel(mesa.Model):
    """
    TODO: Decide on sampling method for random agents

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
        """Extreme value for opinion A"""
        self.opinionB = OPINION_B
        """Extreme value for opinion B"""
        self.opinionDist = np.abs(self.opinionA) + np.abs(self.opinionB)
        """Absolute distance between opinions A and B"""

    def addAgent(self, opinion: float, conformity: float, tolerance: float) -> None:
        """
        # TODO: Testing
        
        Add an agent to the model and scheduler
        
        Args:
            opinion:    (float) The initial opinion value for the agent
            conformity: (float) The fixed conformity value for the agent
            tolerance:  (float) The fixed tolerance value for the agent
        """
        self.graph.add_node(self.nAgents)
        newAgent = PolarizationAgent(self.nAgents, self, opinion, conformity, tolerance)
        self.scheduler.add(newAgent)
        self.nAgents += 1
