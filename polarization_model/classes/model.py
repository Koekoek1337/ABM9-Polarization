import mesa
import networkx as nx
import numpy as np

from agent import PolarizationAgent
from constants import *

from typing import Any, Tuple, List



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

        self.phase = 0
        """Current steptype the agents must follow"""

        return
    
    def step(self):
        # Social step, for making and breaking social bonds
        self.phase = PHASE_SOCIAL
        self.scheduler.step()

        # Agent conformity step
        self.phase = PHASE_CONFORM
        self.scheduler.step()

        # Data collection
        return

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

        return
    
    def resolveInteraction(self, agent: PolarizationAgent) -> None:
        """
        TODO: Testing

        Resolves the pending interactions of an agent.
        """
        source = agent.unique_id
        for action, target in agent.pendingInteraction:
            if action == MAKE and not self.graph.has_edge(source, target):
                self.graph.add_edge(source, target)
        
            if action == BREAK and self.graph.has_edge(source, target):
                self.graph.remove_edge(source, target)
        agent.pendingInteraction.clear()
