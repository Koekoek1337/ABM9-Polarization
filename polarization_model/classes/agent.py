import mesa
from typing import TYPE_CHECKING, List, Tuple
import numpy as np

MAKE = +1
BREAK = -1


if TYPE_CHECKING:
    from model import PolarizationModel

# TODO: Prevent edge case where two nodes make a pending connection to each other.

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
        """
        The amount of connections an agent attempts to maintain. -1 means there is no limit in the amount of
        connections an agent has. 

        Currently only used internally in agent. If future external agent behavior relies on targetDegree, 
        it should be compatible with the step-advance structure.
        """

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
    
    def add_pending_interaction(self, action: int, target_id: int) -> None:
        """
        Adds a pending interaction, ensuring that mutual connections are not duplicated.

        Args:
            action: (int) +1 for MAKE connection, -1 for BREAK connection
            target_id: (int) The unique ID of the target agent
        """
        if action == 1:  # +1 represents MAKE connection
            # Check if the target agent already has a pending MAKE connection to this agent
            target_agent = self.model.schedule.get_agent(target_id)
            if (1, self.unique_id) not in target_agent.pendingInteraction:
                self.pendingInteraction.append((action, target_id))
        else:
            # For BREAK connection or other actions, just add it
            self.pendingInteraction.append((action, target_id))

    def step(self):
        """
        TODO: Implement two separate step types for social network interactions (eg. making and breaking bonds) 
              and diffuse interactions (changing opinion).
              This guarantees that an agent may choose to break off a social bond before conforming to
              an opinion they do not tolerate using the simultaneous activation scheduler
        TODO: Find simple way to fetch node degree from networkx graph or equivalent solution (eg. just
              keep track of edges separately)
        """
        PLACEHOLDER = 0

        # Steptype conform (TODO)
        if True:
            self.conformStep()
            self.fluctuateStep()

        # Steptype Social (TODO)
        if True:
            self.intoleranceStep()

            nEdges = PLACEHOLDER
            if nEdges < self.targetDegree or self.targetDegree == -1:
                self.socializeStep()


    def advance(self) -> None:
        """
        TODO: Implement method that processes self.pendingInteractions for steptype social.
        TODO: Also use step types as described in todo of self.step().
        """
        assert isinstance(self.model, PolarizationModel)

        # Steptype conform (TODO)
        self.opinion = self.newOpinion
        self.newOpinion = 0

        # Enforce opinion boundary conditions
        if self.opinion < self.model.opinionA:
            self.opinion = self.model.opinionA
        elif self.opinion > self.model.opinionB:
            self.opinion = self.model.opinionB
        
        # Steptype Social (TODO)
        if self.pendingInteraction:
            # Process pending transactions
            self.pendingInteraction.clear()

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
        TODO: Decide on method for sampling from existing population
        """
        targetID = self.sampleAcquaintance()
        # TODO: Better "click" function in `agent behavior.md` current iteration is heavily flawed
        PLACEHOLDER = 1
        pClick = PLACEHOLDER
        
        if self.random.random() < pClick:
            self.pendingInteraction.append((+1, targetID))

    def sampleAcquaintance(self) -> int:
        """
        TODO: Decide ona  more robust way of sampling agents (eg. by preferring friends of friends rather
              than random agents in the system)

        Simple sampling function that allows a node to attempt a connection to a random agent in the
        system

        Returns: (int) The node ID containing an agent to attempt a bond with, which is currently not
        bonded to the agent.
        """
        assert isinstance(self.model, PolarizationModel)

        targetID = self.random.choice(self.nonBondedNodes())
        return targetID

    def nonBondedNodes(self) -> List[int]:
        """
        TODO: Testing
        Returns a list of node_ids/ unique agent IDs of agents that are not yet bonded for simple
        random sampling.
        
        Returns: (List[int]) The node_ids which are not bonded to the agent node.
        """
        assert isinstance(self.model, PolarizationModel)

        neighbor_nodes = self.model.space.get_neighborhood(self.unique_id)
        nonBonded = [i for i in range(self.model.nAgents) if i not in neighbor_nodes]

        return nonBonded