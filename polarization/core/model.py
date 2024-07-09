from mesa import Agent, Model
from mesa.space import SingleGrid
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
from tqdm import trange
import random
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities
from networkx.algorithms.community.quality import modularity
from networkx.algorithms.cluster import average_clustering
import numpy as np
from spatialentropy import leibovici_entropy, altieri_entropy

random.seed(711)


class PolarizationAgent(Agent):
    """
    An agent in the Polarization model

    Attributes:
        unique_id (int): The unique identifier of the agent.
        model (PolarizationModel): The model the agent belongs to.
        pos (tuple): The position of the agent on the grid.
        is_ideologue (bool): Whether the agent is an ideologue with a fixed opinion.
        ideologue_opinion (float): The fixed opinion of the agent if it is an ideologue.
        opinion (float): The current opinion of the agent.
        conformity (float): The conformity level of the agent.
        weight_own (float): The weight given to the agent's own opinion when updating.
        weight_connections (float): The weight given to the opinions of the agent's social connections when updating.
        weight_neighbors (float): The weight given to the opinions of the agent's spatial neighbors when updating.
    """
    def __init__(self, unique_id, model, pos, is_ideologue=False, ideologue_opinion=None):
        super().__init__(unique_id, model)
        self.pos = pos
        self.is_ideologue = is_ideologue  # Flag to indicate if opinion should remain fixed
        if is_ideologue:
            self.opinion = ideologue_opinion #random.choice([0, 1])  # Fixed opinion of 0 or 1
        else:
            self.opinion = self.random.uniform(0, 10)
        #self.opinion = self.random.uniform(0, 1)
        self.conformity = 0.8 #self.random.uniform(0.4, 0.8)
        self.weight_own = 1 - self.conformity
        self.weight_connections = self.model.connection_influence * self.conformity
        self.weight_neighbors = (1 - self.model.connection_influence) * self.conformity

    @property
    def connections_ids(self):
        """Get the unique IDs of the agent's social connections."""
        return [connection_id for connection_id in self.model.graph[self.unique_id]]

    @property
    def connections(self):
        """Get the agent objects of the agent's social connections."""
        return [connection for connection in self.model.schedule.agents if connection.unique_id in self.connections_ids]

    @property
    def unconnected_ids(self):
        """Get the unique IDs of the agents not socially connected to the agent."""
        return [unconnected_id for unconnected_id in self.model.graph.nodes if (unconnected_id not in
                                                                                self.connections_ids + [self.unique_id])]

    @property
    def unconnected(self):
        """Get the agent objects of the agents not socially connected to the agent."""
        return [unconnected for unconnected in self.model.schedule.agents if unconnected.unique_id not in
                self.connections_ids]

    @property
    def neighbours(self):
        """Get the agent objects of the agent's spacial neighbors."""
        return self.model.grid.get_neighbors(self.pos, moore=True, include_center=False, radius=1)

    def calc_influence(self):
        """Calculate the influence from the agent's social connections and spatial neighbors."""
        neighbor_influence = 0
        num_neighbors = 0
        connection_influence = 0
        num_connections = 0

        for connection in self.connections:
            if abs(connection.opinion - self.opinion) < self.model.opinion_threshold:
                connection_influence += connection.opinion
                num_connections += 1
        avg_connection_opinion = connection_influence / num_connections if num_connections != 0 else 0

        for neighbor in self.model.grid.get_neighbors(pos=self.pos, moore=True, include_center=False, radius=1):
            if abs(neighbor.opinion - self.opinion) < self.model.opinion_threshold:
                num_neighbors += 1
                neighbor_influence += neighbor.opinion
        avg_neighbor_opinion = neighbor_influence / num_neighbors if num_neighbors != 0 else 0

        return avg_connection_opinion, avg_neighbor_opinion

    def adapt_opinion(self):
        """Update the agent's opinion based on the influence from social connections and spatial neighbors."""
        if not self.is_ideologue:  # Only update if opinion is not fixed
            connection_infl, neighbor_infl = self.calc_influence()
            updated_opinion = self.opinion

            if connection_infl != 0 and neighbor_infl != 0:
                updated_opinion = ((self.weight_own * self.opinion) + (self.weight_connections * connection_infl) +
                               (self.weight_neighbors * neighbor_infl))
            elif connection_infl == 0 and neighbor_infl != 0:
                updated_opinion = (self.weight_own * self.opinion) + ((1 - self.weight_own) * neighbor_infl)
            elif neighbor_infl == 0 and connection_infl != 0:
                updated_opinion = (self.weight_own * self.opinion) + ((1 - self.weight_own) * connection_infl)

            self.opinion = updated_opinion
        # social_infl, nbr_infl = self.calc_influence()
        # new_opinion = self.opinion

        # if social_infl != 0 and nbr_infl != 0:
        #     new_opinion = (self.weight_own * self.opinion) + (self.weight_connections * social_infl) + (self.weight_neighbors * nbr_infl)
        # elif social_infl == 0 and nbr_infl != 0:
        #     new_opinion = (self.weight_own * self.opinion) + ((1 - self.weight_own) * nbr_infl)
        # elif nbr_infl == 0 and social_infl != 0:
        #     new_opinion = (self.weight_own * self.opinion) + ((1 - self.weight_own) * social_infl)

        # self.opinion = new_opinion

    def form_connection(self):
        """Form new social connections with other agents."""
        if len(self.unconnected_ids) < self.model.target_connections:
            num_potential_connections = len(self.unconnected_ids)
        else:
            num_potential_connections = self.model.target_connections

        potential_ids = np.random.choice(self.unconnected_ids, size=num_potential_connections, replace=False)
        potential_agents = [connection for connection in self.model.schedule.agents if connection.unique_id in potential_ids]

        for potential_agent in potential_agents:
            self.evaluate_connection(potential_agent=potential_agent, action="ADD")

    def break_connection(self):
        """Break existing social connections with other agents."""
        num_current_connections = len(self.connections_ids)
        if num_current_connections < self.model.target_connections:
            num_potential_disconnections = 0    #num_current_connections
        else:
            num_potential_disconnections = (num_current_connections -
                                            self.model.target_connections) #self.model.target_connections

        potential_ids = np.random.choice(self.connections_ids, size=num_potential_disconnections, replace=False)
        potential_agents = [connection for connection in self.model.schedule.agents if connection.unique_id in potential_ids]

        for potential_agent in potential_agents:
            self.evaluate_connection(potential_agent, action="REMOVE")

    def evaluate_connection(self, potential_agent, action):
        """Evaluate whether to form or break a social connection with another agent."""
        # p_ij = 1 / (1 + np.exp(self.model.fermi_alpha * (abs(self.opinion - potential_agent.opinion) - self.model.fermi_beta)))
        # print(p_ij)
        
        probability = 0.50 # For now, hange to vary the probability of connection, higher => higher connections

        if action == "ADD":
            if probability > random.random():
                self.model.graph.add_edge(self.unique_id, potential_agent.unique_id)
        if action == "REMOVE":
            if probability < random.random():
                self.model.graph.remove_edge(self.unique_id, potential_agent.unique_id)

    def relocate(self):
        """Relocate the agent to a new position if the opinion difference with neighbors is above a threshold."""
        _, avg_neighbor_opinion = self.calc_influence()
        if abs(self.opinion - avg_neighbor_opinion) > self.model.opinion_threshold:
            self.model.grid.move_to_empty(self)
            self.model.agents_moved += 1
            
        # happiness = 1 / (1 + np.exp(self.model.fermi_alpha * (abs(self.opinion - av_nbr_op) - self.model.fermi_beta)))

        # if happiness < self.model.happiness_threshold:
        #     self.model.grid.move_to_empty(self)
        #     self.model.agents_moved += 1
            
    def connect_different_opinions(self):
        """To connect agents with different opinions, not used in the current model.
        """
        unconnected_agents = [agent for agent in self.model.schedule.agents if agent.unique_id != self.unique_id and
                              agent.unique_id not in self.connections_ids]
        potential_connections = [agent for agent in unconnected_agents if abs(agent.opinion - self.opinion) >=
                                 self.model.opinion_threshold*0.5] # To change this part to connect agents with differing opinions

        if potential_connections:
            potential = random.choice(potential_connections)
            self.evaluate_connection(potential_agent=potential, action="ADD")

    def step(self):
        self.form_connection()
        # self.connect_different_opinions()  # To connect agents with different opinions
        self.break_connection()
        self.relocate()
        self.adapt_opinion()


class PolarizationModel(Model):
    """
    The Polarization Model.

    Attributes:
        width (int): The width of the grid.
        density (float): The initial density of agents on the grid.
        network_m (int): The number of edges to add for each new node in the Barabasi-Albert graph.
        fermi_alpha (float): The parameter that describes the steepness of the Fermi-Dirac distribution curve.
                             It determines how the probability of forming/breaking a connection changes based on
                             the opinion difference between agents.
        fermi_beta (float): The parameter that defines the opinion difference threshold in the Fermi-Dirac
                            distribution. It determines the opinion difference at which the probability of
                            forming/breaking a connection is 0.5.
        connection_influence (float): The influence of social connections on opinion formation.
        target_connections (int): The target number of social connections for each agent.
        opinion_threshold (float): The threshold for opinion difference when forming/breaking social connections.
        schedule (RandomActivation): The scheduler for activating agents.
        agents_moved (int): The number of agents that moved in the current step.
        num_agents (int): The total number of agents in the model.
        grid (SingleGrid): The grid environment for the agents.
        graph (nx.Graph): The social network graph of the agents.
        datacollector (DataCollector): The data collector for recording model and agent data.
    """
    def __init__(self, width=20, density=0.8, network_m=2, fermi_alpha=5, fermi_beta=3, connection_influence=0.8,
                 target_connections=5, opinion_threshold=0.2, happiness_threshold=0.8):
        self.width = width
        self.density = density
        self.network_m = network_m
        self.fermi_alpha = fermi_alpha
        self.fermi_beta = fermi_beta
        self.connection_influence = connection_influence
        self.target_connections = target_connections
        self.opinion_threshold = opinion_threshold
        # self.happiness_threshold = happiness_threshold

        self.schedule = RandomActivation(self)
        self.agents_moved = 0
        self.num_agents = 0

        self.grid = SingleGrid(self.width, self.width, torus=True)
        self.setup_agents()
        self.graph = nx.barabasi_albert_graph(n=self.num_agents, m=self.network_m)

        self.datacollector = DataCollector(
            model_reporters={
                "graph_modularity": self.calc_modularity,
                "agents_moved": lambda m: m.agents_moved,
                "cluster_coefficient": self.calc_clustering,
                "edges": self.get_graph_data,
                "leibovici_entropy_index": self.calc_l_entropy,
                "altieri_entropy_index": self.calc_a_entropy,
            },
            agent_reporters={
                "opinion": lambda x: x.opinion,
                "position": lambda p: p.pos,
            }
        )
        self.running = True

    def calc_modularity(self):
        """Calculate the modularity of the social network graph."""
        max_mod_communities = greedy_modularity_communities(self.graph)
        mod = modularity(self.graph, max_mod_communities)
        return mod

    def calc_clustering(self):
        """Calculate the clustering coefficient of the social network graph."""
        cluster_coefficient = average_clustering(self.graph)
        return cluster_coefficient

    def get_graph_data(self):
        """Get the social network graph data as a dictionary."""
        graph_dict = nx.convert.to_dict_of_dicts(self.graph)
        return graph_dict

    def calc_l_entropy(self):
        """Calculate the Leibovici entropy index of teh agent opinions."""
        agent_info_list = [[agent.pos, agent.opinion] for agent in self.schedule.agents]
        points = []
        types = []

        for i in range(len(agent_info_list)):
            points.append([agent_info_list[i][0][0], agent_info_list[i][0][1]])

        for i in agent_info_list:
            if i[1] < 5:
                types.append("left")
            else:
                types.append("right")

        points = np.array(points)
        types = np.array(types)

        e = leibovici_entropy(points, types, d=2)
        e_entropyind = e.entropy
        return e_entropyind

    def calc_a_entropy(self):
        """Calculate the Altieri entropy index of the agent opinions."""
        agent_info_list = [[agent.pos, agent.opinion] for agent in self.schedule.agents]
        points = []
        types = []

        for i in range(len(agent_info_list)):
            points.append([agent_info_list[i][0][0], agent_info_list[i][0][1]])

        for i in agent_info_list:
            if i[1] < 5:
                types.append("left")
            else:
                types.append("right")

        points = np.array(points)
        types = np.array(types)

        a = altieri_entropy(points, types, cut=2)
        a_entropyind = a.entropy
        return a_entropyind

    def setup_agents(self):
        """Set up the agents on the grid."""
        num_agents = int(self.width * self.width * self.density)
        fixed_opinion_counter = 0  # Counter to alternate fixed opinions between 0 and 1

        for cell in self.grid.coord_iter():
            x, y = cell[1], cell[2]
            if (x is not None) and (y is not None):
                if self.random.uniform(0, 1) < self.density:
                    if random.random() < 0.01:  # Approximately 1% of agents have is_ideologue=True
                        fixed_opinion_value = fixed_opinion_counter % 2  # Alternate between 0 and 1
                        agent = PolarizationAgent(self.num_agents, self, (x, y), is_ideologue=True,
                                                  ideologue_opinion=fixed_opinion_value)
                        fixed_opinion_counter += 1
                    else:
                        agent = PolarizationAgent(self.num_agents, self, (x, y))

                    self.grid.place_agent(agent, (x, y))
                    self.schedule.add(agent)
                    self.num_agents += 1

    def step(self):
        self.schedule.step()

    def run_model(self, step_count=1, desc="", pos=0, collect_during=True, collect_initial=False):
        if collect_initial:
            self.datacollector.collect(self)
        for i in trange(step_count, desc=desc, position=pos):
            self.step()
            if collect_during:
                self.datacollector.collect(self)
                self.agents_moved = 0
        if not collect_during:
            self.datacollector.collect(self)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    model = PolarizationModel(density=0.9, fermi_alpha=4, fermi_beta=1, width=15, opinion_threshold=0.5,
                              happiness_threshold=0.2)
    stepcount = 50

    model.run_model(step_count=stepcount)
    model_df = model.datacollector.get_model_vars_dataframe()
    agent_df = model.datacollector.get_agent_vars_dataframe()

    fig, axes = plt.subplots(2, 2)
    axes = axes.reshape(-1)

    # Assuming you have plotting functions sim_grid_plot and create_graph
    # sim_grid_plot(agent_df, grid_axis=[axes[2], axes[3]])
    # create_graph(agent_df, model_df, graph_axes=axes[:2], layout=nx.spring_layout)

    fig.show()

    fig, ax = plt.subplots(1, 2)
    ax[0].hist(agent_df.xs(stepcount, level="Step")["opinion"], density=True)
    ax[1].plot(range(stepcount), model_df["agents_moved"], label="Movers per step")
    fig.show()
