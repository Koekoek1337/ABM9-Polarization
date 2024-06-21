import mesa
from typing import TYPE_CHECKING, List, Tuple
import numpy as np

MAKE = +1
BREAK = -1


if TYPE_CHECKING:
    from model import PolarizationModel

# TODO: Prevent edge case where two nodes make a pending connection to each other. - Done.

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
        #TODO: Implement two separate step types for social network interactions (eg. making and breaking bonds) 
              and diffuse interactions (changing opinion).
              This guarantees that an agent may choose to break off a social bond before conforming to
              an opinion they do not tolerate using the simultaneous activation scheduler
        # TODO: Find simple way to fetch node degree from networkx graph or equivalent solution (eg. just
              keep track of edges separately)
        """
        #PLACEHOLDER = 0 # Placeholder for the amount of edges of the agent
        
        nEdges = self.model.graph.degree(self.unique_id)

        # Steptype conform (TODO)
        if True:
            self.conformStep()
            self.fluctuateStep()

        # Steptype Social (TODO)
        if True:
            self.intoleranceStep()

            #nEdges = PLACEHOLDER
            if nEdges < self.targetDegree or self.targetDegree == -1:
                self.socializeStep()


    def advance(self) -> None:
        """
        # TODO: Implement method that processes self.pendingInteractions for steptype social.
        # TODO: Also use step types as described in todo of self.step().
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
        
        neighbors = self.model.graph.get_neighbors(self.unique_id)              # Get list of connected neighbors
        meanOpinion = np.mean([n.opinion for n in neighbors])                   # Calculate mean opinion of neighbors
        newOpinion = (meanOpinion * self.conformity + self.opinion * (1-self.conformity)) / 2 # calculate new opinion
        
        self.newOpinion = newOpinion
        return newOpinion

    def fluctuateStep(self) -> float:
        """
        # TODO: Implement random fluctuation of opinion (eg sampling from normal distribution centered around)

        Fluctuates the current opinion of the agent by a random amount and update newOpinion.
        
        Returns (float) the new opinion of the agent
        """
        randomVal = np.random.normal(0, 0.1) # 0.1 can be adjusted based on desired fluctuation magnitude
        newOpinion = newOpinion + randomVal
        
        # Ensure newOpinion stays within the bounds [self.model.opinionA, self.model.opinionB]
        self.newOpinion = max(self.model.opinionA, min(self.newOpinion, self.model.opinionB))

        return newOpinion

    def intoleranceStep(self) -> int:
        """
        TODO: Testing

        Tests all for all connected nodes whether their opinion is tolerated. Adds a pending break
        transaction for all nodes which are not tolerated

        Returns: The amount of nodes from which their connection is to be removed.
        """
        assert isinstance(self.model, PolarizationModel)

        neighbors = self.model.graph.get_neighbors(self.unique_id)
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
        PLACEHOLDER = 1 # Replace with  with a click function that determines the probability of a connection
        pClick = PLACEHOLDER
        
        if self.random.random() < pClick:
            self.pendingInteraction.append((+1, targetID))

    def sampleAcquaintance(self) -> int:
        """
        TODO: Decide on a more robust way of sampling agents (e.g., by preferring friends of friends rather
            than random agents in the system)

        Modified sampling function that allows a node to attempt a connection to a neighbor or neighbor of a neighbor
        in the system, with a preference for closer nodes in terms of network distance, using exponential distribution.

        Returns: (int) The node ID containing an agent to attempt a bond with, which is currently not
        bonded to the agent.
        """
        assert isinstance(self.model, PolarizationModel)

        neighbors = self.model.graph.get_neighbors(self.unique_id)  # Get direct neighbors
        neighbors_of_neighbors = set()  # Set to store neighbors of neighbors

        for neighbor in neighbors:
            neighbors_of_neighbors.update(self.model.graph.get_neighbors(neighbor))  # Add neighbors of neighbors to set

        # Remove nodes already bonded to self
        non_bonded_nodes = [node_id for node_id in neighbors_of_neighbors if node_id not in self.model.graph.neighbors(self.unique_id)]

        if not non_bonded_nodes:
            non_bonded_nodes = self.nonBondedNodes()  # Fall back to all non-bonded nodes if no neighbors of neighbors found

        # Calculate distances of non-bonded nodes from self
        distances = [nx.shortest_path_length(self.model.graph, self.unique_id, node_id) for node_id in non_bonded_nodes]

        # Calculate probabilities based on exponential distribution (higher probability for closer nodes)
        distances = np.array(distances)
        probabilities = np.exp(-distances / distances.max())  # Exponential distribution based on distances

        # Normalize probabilities
        probabilities /= probabilities.sum()

        # Sample node ID based on calculated probabilities
        targetID = self.random.choice(non_bonded_nodes, p=probabilities)
        
        return targetID

    def nonBondedNodes(self) -> List[int]:
        """
        TODO: Testing
        Returns a list of node_ids/ unique agent IDs of agents that are not yet bonded for simple
        random sampling.
        
        Returns: (List[int]) The node_ids which are not bonded to the agent node.
        """
        assert isinstance(self.model, PolarizationModel)

        neighbor_nodes = self.model.graph.get_neighborhood(self.unique_id)
        nonBonded = [i for i in range(self.model.nAgents) if i not in neighbor_nodes]

        return nonBonded
    
class IdeologueAgent(PolarizationAgent):
    def __init__(self, unique_id: int, model: PolarizationModel, opinion: float) -> None:
        super().__init__(unique_id, model, opinion, conformity=0.0, tolerance=0.0, targetDegree=-1)
        """Ideologues have fixed opinions (e.g., -1 or +1), no conformity, and zero tolerance."""

class FollowerAgent(PolarizationAgent):
    def __init__(self, unique_id: int, model: PolarizationModel, opinion: float, conformity: float, tolerance: float) -> None:
        super().__init__(unique_id, model, opinion, conformity, tolerance, targetDegree=-1)
        """Followers have varying opinions, conformity, and tolerance."""