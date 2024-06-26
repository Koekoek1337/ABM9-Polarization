import mesa
import networkx as nx
import numpy as np
from random import Random
from .agent import PolarizationAgent
from .constants import *
from typing import Any, List
from networkx.algorithms.community import greedy_modularity_communities
from networkx.algorithms.cluster import average_clustering
from networkx.algorithms.community.quality import modularity
# from scipy.stats import entropy
from mesa.datacollection import DataCollector

from spatialentropy import leibovici_entropy
from spatialentropy import altieri_entropy
import sys


class PolarizationModel(mesa.Model):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.graph = nx.Graph()
        self.space = mesa.space.NetworkGrid(self.graph)
        self.scheduler = mesa.time.SimultaneousActivation(self)
        self.nAgents: int = 0
        self.opinionA = OPINION_A
        self.opinionB = OPINION_B
        self.opinionDist = np.abs(self.opinionA) + np.abs(self.opinionB)
        self.phase = 0

        self.datacollector = DataCollector(
            model_reporters={
                "graph_modularity": self.calculate_modularity,
                # "cluster_coefficient": self.calculate_clustercoef,
                "leibovici_entropy_index": self.calculate_l_entropyindex,
                "altieri_entropy_index": self.calculate_a_entropyindex,

            },
            agent_reporters={
                "opinion": lambda x: x.opinion,
                # "position": lambda p: p.pos,
            }
        )
        self.running = True

    def step(self):
        self.phase = PHASE_SOCIAL
        self.scheduler.step()
        self.phase = PHASE_CONFORM
        self.scheduler.step()
        # self.datacollector.collect(self)
        return


    def generateAgents(self, nAgents: int, meanOpinion: float, stDevOpinion: float, minConformity: float=0.0, 
                       maxConformity: float=1.0, minTolerance: float=0.0, maxTolerance: float=1.0) -> None:        
        for i in range(nAgents):
            opinion = self.random.gauss(meanOpinion, stDevOpinion)
            if opinion < self.opinionA:
                opinion = self.opinionA
            if opinion > self.opinionB:
                opinion = self.opinionB
            conformity = self.random.uniform(minConformity, maxConformity)
            tolerance = self.random.uniform(minTolerance, maxTolerance)
            self.addAgent(opinion, conformity, tolerance)
        return

    def preConnect_Network(self, nIter: int):
        self.phase = PHASE_SOCIAL
        for _ in range(nIter):
            self.scheduler.step()

    def addAgent(self, opinion: float, conformity: float, tolerance: float) -> None:
        self.graph.add_node(self.nAgents)
        self.graph.nodes[self.nAgents]["agent"] = self.space.default_val()
        newAgent = PolarizationAgent(self.nAgents, self, opinion, conformity, tolerance)
        self.space.place_agent(newAgent, self.nAgents)
        self.scheduler.add(newAgent)
        self.nAgents += 1
        return
    
    def resolveInteraction(self, agent: PolarizationAgent) -> None:
        source = agent.unique_id
        for action, target in agent.pendingInteraction:
            if action == MAKE and not self.graph.has_edge(source, target):
                self.graph.add_edge(source, target)
            if action == BREAK and self.graph.has_edge(source, target):
                self.graph.remove_edge(source, target)
        agent.pendingInteraction.clear()
        return
    
    def agentOpinions(self) -> List[float]:
        opinions = [0] * self.nAgents
        for agent in self.scheduler.agents:
            assert isinstance(agent, PolarizationAgent)
            opinions[agent.unique_id] = agent.opinion
        return opinions
    
    def lockDegree(self, target: int=None):
        for agent in self.scheduler.agents:
            assert isinstance(agent, PolarizationAgent)
            if target is None:
                agent.targetDegree = nx.degree(self.graph, agent.unique_id)
            else:
                agent.targetDegree = target

    def calculate_modularity(self):
        communities = greedy_modularity_communities(self.graph)
        return modularity(self.graph, communities)

    def calculate_cluster_coefficient(self):
        return average_clustering(self.graph)

    def calculate_entropy(self):
        opinions = [agent.opinion for agent in self.scheduler.agents]
        if opinions:
            opinion_counts = np.histogram(opinions, bins=np.linspace(self.opinionA, self.opinionB, num=10))[0]
            opinion_prob = opinion_counts / np.sum(opinion_counts)
            return entropy(opinion_prob)
        return 0

    def calculate_l_entropyindex(self):
        """Calculation of the Leibovici entropy index, using the spatial entropy packaged as described
            on the following github: https://github.com/Mr-Milk/SpatialEntropy

        Returns:
            [float]: [Leibovici entropy index]
        """
        agent_infolist = [[agent.pos, agent.opinion] for agent in self.schedule.agents]
        points = []
        types = []

        for i in range(len(agent_infolist)):
            points.append([agent_infolist[i][0][0], agent_infolist[i][0][1]])

        for i in agent_infolist:
                if i[1]<3:
                    types.append("left")

                elif 3<i[1]<7:
                    types.append("middle")
                else:
                    types.append("right")

        points = np.array(points)
        types = np.array(types)

        e = leibovici_entropy(points, types, d=2)
        e_entropyind = e.entropy

        return e_entropyind

    def calculate_a_entropyindex(self):
        """Calculation of the Altieri entropy index, using the spatial entropy packaged as described
            on the following github: https://github.com/Mr-Milk/SpatialEntropy

        Returns:
            [float]: [Altieri entropy index]
        """
        agent_infolist = [[agent.pos, agent.opinion] for agent in self.schedule.agents]
        points = []
        types = []

        for i in range(len(agent_infolist)):
            points.append([agent_infolist[i][0][0], agent_infolist[i][0][1]])


        for i in agent_infolist:
            if i[1]<3:
                types.append("left")
            elif 3<i[1]<7:
                types.append("middle")
            else:
                types.append("right")

        points = np.array(points)
        types = np.array(types)

        a = altieri_entropy(points, types, cut=2)
        a_entropyind = a.entropy

        return a_entropyind