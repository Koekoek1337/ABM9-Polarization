import mesa
from typing import TYPE_CHECKING, List, Tuple
import numpy as np

MAKE = +1
BREAK = -1


if TYPE_CHECKING:
    from model import PolarizationModel

# TODO: Prevent edge case where two nodes make a pending connection to eachother.

class PolarizationAgent(mesa.Agent):
    def __init__(self, unique_id: int, model: PolarizationModel, opinion: float, conformity: float, tolerance: float, targetDegree = -1) -> None:
        super().__init__(unique_id, model)

        self.opinion = opinion
        """Relative allegiance of political interest"""
        self.conformity = conformity
        """Pressure on the agent to conform to it's surroundings"""
        self.tolerance = tolerance
        """How tolerant an agent is to different minded individuals"""
        self.targetDegree = targetDegree
        """The amount of connections an agent attempts to maintain. -1 means there is no limit in the amount of
        connections an agent has"""

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
        TODO: Implement two separate step types for social network interactions (eg. making and breaking bonds) 
              and diffuse interactions (changing opinion).
              This guarantees that an agent may choose to break off a social bond before conforming to
              an opinion they do not tolerate using the simultaneous activation scheduler
        """
        pass
        # Steptype conform
        self.conformStep()
        # Steptype Social

    def advance(self) -> None:
        """
        TODO: Implement advance method that processes self.pendingInteractions

        TODO: Also use step types as described in self.step().
        """
        pass
        # Steptype conform
        self.opinion = self.newOpinion
        # Steptype Social

    def conformStep(self) -> float:
        """
        TODO: Testing

        Make agent conform to it's neighbor's opinions according to
        $$op_{t+1} = {\overline{op}_{neigh}conf + op_t(1 - conf)}$$

        Returns the new opinion of the agent
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

    def intoleranceStep(self) -> int:
        """
        TODO: Testing

        Tests all for all connected nodes whether their opinion is tolerated. Adds a pending break
        transaction for all nodes which are not tolerated

        Returns: The amount of nodes from which their connection is to be removed.
        """
        assert isinstance(self.model, PolarizationModel)

        neighbors = self.model.space.get_neighbors(self.unique_id)
        for neighbor in neighbors:
            assert isinstance(neighbor, PolarizationAgent)

            pBreak = (1 - self.tolerance) * np.abs(self.opinion - neighbor.opinion) / self.model.opinionDist
            """Probability of an agent relation breaking as result from an intolerance of opinion"""
            if self.random.random() < pBreak:
                self.pendingInteraction.append((BREAK, neighbor.unique_id))

    def socializeStep(self):
        """
        TODO: add method that creates a pending transaction between a node if they click
        """
    

    def nonBondedNodes(self) -> List[int]:
        """
        TODO: Testing
        
        Returns (List[int]): The node_ids which are not bonded to the agent node.
        """
        assert isinstance(self.model, PolarizationModel)

        neighbor_nodes = self.model.space.get_neighborhood(self.unique_id)
        nonBonded = [i for i in range(self.model.nAgents) if i not in neighbor_nodes]

        return nonBonded