import mesa
import networkx as nx
from typing import TYPE_CHECKING, List, Tuple
import numpy as np

from .constants import *

if TYPE_CHECKING:
    from model import PolarizationModel

# TODO: Prevent edge case where two nodes make a pending connection to eachother.

class PolarizationAgent(mesa.Agent):
    def __init__(self, unique_id: int, model: "PolarizationModel", opinion: float, conformity: float, tolerance: float, targetDegree = -1) -> None:
        super().__init__(unique_id, model)
        self.model = model
        
        # Agent attributes
        self.opinion = opinion
        """Relative allegiance of political interest"""
        self.conformity = conformity
        """Pressure on the agent to conform to it's surroundings"""
        self.tolerance = tolerance
        """How tolerant an agent is to different minded individuals"""
        self.targetDegree = targetDegree
        """
        The amount of connections an agent attempts to maintain. -1 means there is no limit in the amount of
        connections an agent has. 

        Currently only used internally in agent. If future external agent behavior relies on targetDegree, 
        it should be compatible with the step-advance structure.
        """


        # Advance attributes
        self.newOpinion = opinion
        """Agent opinion for the next step, for use in the advance method for simultaneous activation."""
        self.pendingInteraction: List[Tuple[int, int]] = []
        """
        List of pending node bond interactions, henceforth described as pending transactions for use in the 
        advance method for simulataneous activation.
        A transaction consists of a tuple of 2 integers. 
        Transaction[0] represents the type of action to occur, which are:
            MAKE  (Literal[+1]): Create a node connection
            BREAK (Literal[-1]): Remove a node connection
        Transaction[1] represents the unique_id of the target node
        """

    def step(self):
        """
        TODO: Testing

        Mesa step function
        """
        degree = nx.degree(self.model.graph, self.unique_id)

        # Steptype conform
        if self.model.phase == PHASE_CONFORM:
            self.conformStep()
            self.fluctuateStep()

        # Steptype Social
        if self.model.phase == PHASE_SOCIAL:
            degree -= self.intoleranceStep()

            if degree < self.targetDegree or self.targetDegree < 0:
                self.socializeStep()

    def advance(self) -> None:
        """
        TODO: Testing
        """
        if self.model.phase == PHASE_CONFORM:
            self.opinion = self.newOpinion
            self.newOpinion = 0

            # Enforce opinion boundary conditions
            if self.opinion < self.model.opinionA:
                self.opinion = self.model.opinionA
            elif self.opinion > self.model.opinionB:
                self.opinion = self.model.opinionB

        if self.model.phase == PHASE_SOCIAL:
            self.model.resolveInteraction(self)

    def conformStep(self) -> float:
        """
        TODO: Testing

        Make agent conform to it's neighbor's opinions according to
        $$op_{t+1} = {\overline{op}_{neigh}conf + op_t(1 - conf)}$$
        and update newOpinion

        Returns (float) the new opinion of the agent
        """
        if self.conformity == 0.0:
            self.newOpinion = self.opinion
            return self.newOpinion
        
        assert isinstance(self.model, PolarizationModel)
        
        neighbors = self.model.space.get_neighbors(self.unique_id)              # Get list of connected neighbors
        meanOpinion = np.mean([n.opinion for n in neighbors])                   # Calculate mean opinion of neighbors
        newOpinion = (meanOpinion * self.conformity + self.opinion * (1-self.conformity)) / 2 # calculate new opinion
        
        self.newOpinion = newOpinion
        return newOpinion

    def fluctuateStep(selt) -> float:
        """
        TODO: Implement random fluctuation of opinion (eg sampling from normal distribution centered around)

        Fluctuates the current opinion of the agent by a random amount and update newOpinion.
        
        Returns (float) the new opinion of the agent
        """
        PLACEHOLDER = 0
        randomVal = PLACEHOLDER
        newOpinion = newOpinion + randomVal

        return newOpinion

    def intoleranceStep(self) -> int:
        """
        TODO: Testing

        Tests all for all connected nodes whether their opinion is tolerated. Adds a pending break
        transaction for all nodes which are not tolerated

        Returns: The amount of nodes from which their connection is to be removed.
        """
        breaks = 0

        neighbors = self.model.space.get_neighbors(self.unique_id)
        for neighbor in neighbors:
            assert isinstance(neighbor, PolarizationAgent)

            pBreak = (1 - self.tolerance) * np.abs(self.opinion - neighbor.opinion) / self.model.opinionDist
            """Probability of an agent relation breaking as result from an intolerance of opinion"""
            if self.random.random() < pBreak:
                self.pendingInteraction.append((BREAK, neighbor.unique_id))
                breaks += 1
        
        return breaks

    def socializeStep(self):
        """
        TODO: add method that creates a pending transaction between a node if they click
        TODO: Decide on method for sampling from existing population
        """
        targetID = self.sampleAcquaintance()

        if targetID is None:
            return
        
        # TODO: Better "click" function in `agent behavior.md` current iteration is heavily flawed
        PLACEHOLDER = 1
        pClick = PLACEHOLDER
        
        if self.random.random() < pClick:
            self.pendingInteraction.append((+1, targetID))

    def sampleAcquaintance(self) -> int:
        """
        TODO: Decide on a more robust way of sampling agents (eg. by preferring friends of friends rather
              than random agents in the system)

        Simple sampling function that allows a node to attempt a connection to a random agent in the
        system

        Returns: (int) The node ID containing an agent to attempt a bond with, which is currently not
        bonded to the agent. Returns None if there are no possible targets.
        """
        possibleTargets = self.nonBondedNodes()
        if not possibleTargets: 
            return None
        targetID = self.random.choice(possibleTargets)
        return targetID

    def nonBondedNodes(self) -> List[int]:
        """
        TODO: Testing
        Returns a list of node_ids/ unique agent IDs of agents that are not yet bonded for simple
        random sampling.
        
        Returns: (List[int]) The node_ids which are not bonded to the agent node.
        """
        neighbor_nodes = self.model.space.get_neighborhood(self.unique_id)
        nonBonded = [i for i in range(self.model.nAgents) if i not in neighbor_nodes if i != self.unique_id]

        return nonBonded