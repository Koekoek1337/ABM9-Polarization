import mesa
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from typing import Any, Tuple, List
from mesa import Agent, Model
from mesa.time import RandomActivation


OPINION_A = -1
OPINION_B = 1
MAKE = +1
BREAK = -1

class MergedOpinionAgent(Agent):
    def __init__(self, unique_id, model, opinion, is_fixed, targetDegree) -> None:
        super().__init__(unique_id, model)

        self.opinion = opinion
        self.is_fixed = is_fixed
        self.conformity = 0 if is_fixed else self.random.uniform(0, 1)
        self.tolerance = self.calculate_tolerance()
        self.targetDegree = targetDegree

        self.newOpinion = opinion
        self.pendingInteraction: List[Tuple[int, int]] = []

    def calculate_tolerance(self) -> float:
        if self.is_fixed:
            return 0
        return 1 - abs(self.opinion)

    def step(self):
        self.intoleranceStep()
        self.socializeStep()
        if not self.is_fixed:
            self.conformStep()
            self.fluctuateStep()

    def advance(self) -> None:
        if not self.is_fixed:
            self.opinion = self.newOpinion
            self.opinion = np.clip(self.opinion, self.model.opinionA, self.model.opinionB)
            self.tolerance = self.calculate_tolerance()

        for interaction in self.pendingInteraction:
            action, targetID = interaction
            if action == MAKE:
                if not self.model.graph.has_edge(self.unique_id, targetID):
                    self.model.graph.add_edge(self.unique_id, targetID)
            elif action == BREAK:
                if self.model.graph.has_edge(self.unique_id, targetID):
                    self.model.graph.remove_edge(self.unique_id, targetID)

        self.pendingInteraction.clear()

    def conformStep(self) -> float:
        if self.conformity == 0.0:
            return self.opinion
        
        neighbors = self.model.grid.get_neighbors(self.unique_id, include_center=False)
        neighbor_agents = [self.model.schedule.agents[n] for n in neighbors]
        neighbor_opinions = [agent.opinion for agent in neighbor_agents if isinstance(agent, MergedOpinionAgent)]
        
        if neighbor_opinions:
            mean_neighbor_opinion = np.mean(neighbor_opinions)
            self.newOpinion = mean_neighbor_opinion * self.conformity + self.opinion * (1 - self.conformity)
        
        return self.newOpinion

    def fluctuateStep(self) -> float:
        opinionRange = self.model.opinionB - self.model.opinionA
        stdDev = opinionRange * 0.05
        fluctuation = self.random.normalvariate(mu=0, sigma=stdDev)
        self.newOpinion = np.clip(self.newOpinion + fluctuation, self.model.opinionA, self.model.opinionB)
        return self.newOpinion

    def intoleranceStep(self) -> int:
        neighbors = self.model.grid.get_neighbors(self.unique_id, include_center=False)
        broken_connections = 0
        for neighbor in neighbors:
            neighbor_agent = self.model.schedule.agents[neighbor]
            assert isinstance(neighbor_agent, MergedOpinionAgent)

            opinionDiff = abs(self.opinion - neighbor_agent.opinion)
            pBreak = (opinionDiff / (self.model.opinionB - self.model.opinionA)) * (1 - self.tolerance)
            if self.random.random() < pBreak:
                self.pendingInteraction.append((BREAK, neighbor))
                broken_connections += 1

        return broken_connections

    def socializeStep(self):
        degree = self.model.graph.degree(self.unique_id)
        if degree < self.targetDegree or self.targetDegree == -1:
            targetID = self.sampleAcquaintance()

            targetAgent = self.model.schedule.agents[targetID]
            assert isinstance(targetAgent, MergedOpinionAgent)

            opinionDiff = abs(self.opinion - targetAgent.opinion)
            pClick = (1 - opinionDiff / (self.model.opinionB - self.model.opinionA)) * self.tolerance * targetAgent.tolerance
            if self.random.random() < pClick:
                self.pendingInteraction.append((MAKE, targetID))

    def sampleAcquaintance(self) -> int:
        friends = set(self.model.grid.get_neighbors(self.unique_id, include_center=False))
        friends_of_friends = set()
        for friend in friends:
            friends_of_friends.update(self.model.grid.get_neighbors(friend, include_center=False))
        
        potential_acquaintances = friends_of_friends - friends - {self.unique_id}
        
        if potential_acquaintances:
            return self.random.choice(list(potential_acquaintances))
        else:
            return self.random.choice(self.nonBondedNodes())

    def nonBondedNodes(self) -> List[int]:
        allNodes = list(self.model.graph.nodes)
        neighborNodes = list(self.model.grid.get_neighbors(self.unique_id, include_center=False))
        nonBonded = [nodeID for nodeID in allNodes if nodeID not in neighborNodes and nodeID != self.unique_id]

        return nonBonded

class MergedPolarizationModel(Model):
    def __init__(self, num_agents: int, num_fixed: int, initial_connectivity: int, *args: Any, **kwargs: Any) -> None:
        super().__init__()

        self.num_agents = num_agents
        self.num_fixed = num_fixed
        self.initial_connectivity = initial_connectivity

        self.graph = nx.Graph()
        self.grid = mesa.space.NetworkGrid(self.graph)
        self.scheduler = mesa.time.SimultaneousActivation(self)

        self.opinionA = OPINION_A
        self.opinionB = OPINION_B
        self.opinionDist = np.abs(self.opinionA) + np.abs(self.opinionB)

        self.datacollector = mesa.DataCollector(
            model_reporters={
                "Opinions": self.get_opinion_distribution,
                "Tolerances": self.get_tolerance_distribution,
                "Number of Edges": self.get_num_edges
            }
        )

        self.initialize()

    def initialize(self) -> None:
        for i in range(self.num_agents):
            if i < self.num_fixed // 2:
                self.addAgent(self.opinionA, is_fixed=True)
            elif i < self.num_fixed:
                self.addAgent(self.opinionB, is_fixed=True)
            else:
                self.addAgent(self.random.uniform(self.opinionA, self.opinionB))

        while self.graph.number_of_edges() < self.initial_connectivity:
            a1, a2 = self.random.sample(list(self.graph.nodes()), 2)
            if not self.graph.has_edge(a1, a2):
                self.graph.add_edge(a1, a2)

    def step(self) -> None:
        self.scheduler.step()
        self.datacollector.collect(self)

    def addAgent(self, opinion: float, is_fixed: bool = False, targetDegree: int = -1) -> None:
        agent_id = self.next_id()
        self.graph.add_node(agent_id, agent=[])
        newAgent = MergedOpinionAgent(agent_id, self, opinion, is_fixed, targetDegree)
        self.grid.place_agent(newAgent, agent_id)
        self.scheduler.add(newAgent)

    def payoffs(self) -> Tuple[float, float]:
        return (self.opinionA, self.opinionB)

    def get_opinion_distribution(self) -> List[float]:
        return [agent.opinion for agent in self.scheduler.agents if isinstance(agent, MergedOpinionAgent)]

    def get_tolerance_distribution(self) -> List[float]:
        return [agent.tolerance for agent in self.scheduler.agents if isinstance(agent, MergedOpinionAgent)]

    def get_num_edges(self) -> int:
        return self.graph.number_of_edges()

    def run_model(self, n: int) -> None:
        for _ in range(n):
            self.step()

def run_model():
    num_agents = 100
    num_fixed = 10
    initial_connectivity = 200
    num_steps = 100

    model = MergedPolarizationModel(num_agents=num_agents, num_fixed=num_fixed, initial_connectivity=initial_connectivity)
    model.run_model(num_steps)

    model_data = model.datacollector.get_model_vars_dataframe()

    print("Final opinion distribution:", model_data["Opinions"].iloc[-1])
    print("Final tolerance distribution:", model_data["Tolerances"].iloc[-1])
    print("Final number of edges:", model_data["Number of Edges"].iloc[-1])

    plot_results(model_data)

def plot_results(model_data):
    plt.figure(figsize=(15, 5))

    plt.subplot(131)
    plt.title("Opinion Distribution Over Time")
    plt.imshow(np.array(model_data["Opinions"].tolist()).T, aspect='auto', cmap='viridis')
    plt.colorbar(label='Opinion')
    plt.xlabel('Time Step')
    plt.ylabel('Agent')

    plt.subplot(132)
    plt.title("Tolerance Distribution Over Time")
    plt.imshow(np.array(model_data["Tolerances"].tolist()).T, aspect='auto', cmap='viridis')
    plt.colorbar(label='Tolerance')
    plt.xlabel('Time Step')
    plt.ylabel('Agent')

    plt.subplot(133)
    plt.title("Number of Edges Over Time")
    plt.plot(model_data["Number of Edges"])
    plt.xlabel('Time Step')
    plt.ylabel('Number of Edges')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_model()
