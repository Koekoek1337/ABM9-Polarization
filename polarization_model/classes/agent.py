import mesa
import networkx as nx
from typing import TYPE_CHECKING, List, Tuple
import numpy as np

from .constants import *

if TYPE_CHECKING:
    from .model import PolarizationModel

class PolarizationAgent(mesa.Agent):
    def __init__(self, unique_id: int, model: 'PolarizationModel', opinion: float, conformity: float, tolerance: float, targetDegree=-1) -> None:
        super().__init__(unique_id, model)
        self.opinion = opinion
        self.conformity = conformity
        self.tolerance = tolerance
        self.targetDegree = targetDegree
        self.newOpinion = opinion
        self.pendingInteraction: List[Tuple[int, int]] = []

    def step(self):
        degree = nx.degree(self.model.graph, self.unique_id)

        if self.model.phase == PHASE_CONFORM:
            self.conformStep()
            self.fluctuateStep()

        if self.model.phase == PHASE_SOCIAL:
            degree -= self.intoleranceStep()
            if degree < self.targetDegree or self.targetDegree < 0:
                self.socializeStep()

    def advance(self) -> None:
        if self.model.phase == PHASE_CONFORM:
            self.opinion = self.newOpinion
            self.newOpinion = 0
            if self.opinion < self.model.opinionA:
                self.opinion = self.model.opinionA
            elif self.opinion > self.model.opinionB:
                self.opinion = self.model.opinionB

        if self.model.phase == PHASE_SOCIAL:
            self.model.resolveInteraction(self)

    def conformStep(self) -> float:
        if self.conformity == 0.0:
            self.newOpinion = self.opinion
            return self.newOpinion
        
        neighbors = self.model.space.get_neighbors(self.unique_id)
        if neighbors:
            meanOpinion = np.mean([n.opinion for n in neighbors])
            newOpinion = (meanOpinion * self.conformity + self.opinion * (1 - self.conformity)) / 2
        else:
            newOpinion = self.opinion
        
        self.newOpinion = newOpinion
        return newOpinion

    def fluctuateStep(self) -> float:
        PLACEHOLDER = 0
        randomVal = PLACEHOLDER
        newOpinion = self.opinion + randomVal
        return newOpinion

    def intoleranceStep(self) -> int:
        breaks = 0
        neighbors = self.model.graph.neighbors(self.unique_id)
        for neighbor_id in neighbors:
            neighbor = self.model.scheduler.agents[neighbor_id]
            pBreak = (1 - self.tolerance) * np.abs(self.opinion - neighbor.opinion) / self.model.opinionDist
            if self.random.random() < pBreak:
                self.pendingInteraction.append((BREAK, neighbor.unique_id))
                breaks += 1
        return breaks

    def socializeStep(self):
        targetID = self.sampleAcquaintance()
        if targetID is None:
            return
        targetAgent = self.model.scheduler.agents[targetID]
        dOpinion = abs(self.opinion - targetAgent.opinion) / self.model.opinionDist
        pClick = 1 - (1 - min(self.tolerance, targetAgent.tolerance)) * dOpinion
        if self.random.random() < pClick:
            self.pendingInteraction.append((MAKE, targetID))

    def sampleAcquaintance(self) -> int:
        targetID = self.random.choice(self.nonBondedNodes())
        return targetID

    def weightedSample(self) -> int:
        neighbors = list(self.model.graph.neighbors(self.unique_id))
        neighbors_of_neighbors = set()
        for neighbor in neighbors:
            neighbors_of_neighbors.update(self.model.graph.neighbors(neighbor))
        non_bonded_nodes = [node_id for node_id in neighbors_of_neighbors if node_id not in self.model.graph.neighbors(self.unique_id)]
        if not non_bonded_nodes:
            non_bonded_nodes = self.nonBondedNodes()
        distances = [nx.shortest_path_length(self.model.graph, self.unique_id, node_id) for node_id in non_bonded_nodes]
        distances = np.array(distances)
        probabilities = np.exp(-distances / distances.max())
        probabilities /= probabilities.sum()
        targetID = self.random.choice(non_bonded_nodes, p=probabilities)
        return targetID

    def nonBondedNodes(self) -> List[int]:
        neighbor_nodes = set(self.model.graph.neighbors(self.unique_id))
        nonBonded = [i for i in range(self.model.nAgents) if i not in neighbor_nodes and i != self.unique_id]
        return nonBonded

class IdeologueAgent(PolarizationAgent):
    def __init__(self, unique_id: int, model: 'PolarizationModel', opinion: float) -> None:
        super().__init__(unique_id, model, opinion, conformity=0.0, tolerance=0.0, targetDegree=-1)

class FollowerAgent(PolarizationAgent):
    def __init__(self, unique_id: int, model: 'PolarizationModel', opinion: float, conformity: float, tolerance: float) -> None:
        super().__init__(unique_id, model, opinion, conformity, tolerance, targetDegree=-1)