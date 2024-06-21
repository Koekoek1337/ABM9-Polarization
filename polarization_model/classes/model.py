import mesa
import networkx as nx
import numpy as np

from random import Random

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
    
    def generateAgents(self, nAgents: int, meanOpinion: float, stDefOpinion: float, minConformity: float=0.0, 
                       maxConformity:float=1.0, minTolerance:float=0.0, maxTolerance: float=1.0) -> None:
        """
        # TODO: Testing

        Generates nAgents agents with randomized properties. Opinion is sampled based on a normal distribution
        with a given mean and standard deviation, truncated to the domain [self.opinionA, self.opinionB]. 
        Conformity and tolerance are sampled from a uniform distribution within the given domain.

        Args:
            nAgents:       (int)   The amount of agents to be generated.
            meanOpinion:   (float) The mean opinion for normal distribution sampling.
            stDevOpinion:  (float) The standard devition in opinion for distribution sampling.
            minConformity: (float) The minimum conformity for uniform sampling.
            maxconformity: (float) The maximum conformity for uniform sampling.
            minTolerance:  (float) The minimum tolerance for uniform sampling.
            maxTolerance:  (float) The maximum tolerance for uniform sampling.
        """
        assert isinstance(self.random, Random)
        for i in range(nAgents):
            opinion    = self.random.gauss(meanOpinion,    stDefOpinion   )
            if opinion < self.opinionA:
                opinion = self.opinionA
            if opinion > self.opinionB:
                opinion = self.opinionB

            conformity = self.random.uniform(minConformity, maxConformity)
            tolerance  = self.random.uniform(minTolerance,  maxTolerance )

            self.addAgent(opinion, conformity, tolerance)
        return
    
    def preConnect_Network(self, nIter: int):
        """
        Initializes a network of the model by running the socialization step nIter times
        """
        self.phase = PHASE_SOCIAL
        for _ in range(nIter):
            self.scheduler.step()

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

        return
    
