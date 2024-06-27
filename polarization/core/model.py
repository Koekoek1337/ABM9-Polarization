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
#from spatialentropy import leibovici_entropy, altieri_entropy

random.seed(711)

class Resident(Agent):
    def __init__(self, unique_id, model, pos, fixed_opinion=False):
        super().__init__(unique_id, model)
        self.pos = pos
        # self.fixed_opinion = fixed_opinion  # Flag to indicate if opinion should remain fixed
        # if fixed_opinion:
        #     self.opinion = random.choice([0, 1])  # Fixed opinion of 0 or 1
        # else:
        #     self.opinion = self.random.uniform(0, 1)
        self.opinion = self.random.uniform(0, 1)
        self.conformity = self.random.uniform(0.4, 0.8)
        self.weight_own = 1 - self.conformity
        self.weight_socials = self.model.social_factor * self.conformity
        self.weight_neighbors = (1 - self.model.social_factor) * self.conformity

    @property
    def socials_ids(self):
        return [social_id for social_id in self.model.graph[self.unique_id]]

    @property
    def socials(self):
        return [social for social in self.model.schedule.agents if social.unique_id in self.socials_ids]

    @property
    def unconnected_ids(self):
        return [id for id in self.model.graph.nodes if (id not in self.socials_ids + [self.unique_id])]

    @property
    def unconnected(self):
        return [unconnected for unconnected in self.model.schedule.agents if unconnected.unique_id not in self.socials_ids]

    @property
    def neighbours(self):
        return self.model.grid.get_neighbors(self.pos, moore=True, include_center=False, radius=1)

    def get_external_influences(self):
        nbr_influence = 0
        n_nbrs = 0
        social_influence = 0
        n_socials = 0

        for social in self.socials:
            if abs(social.opinion - self.opinion) < self.model.opinion_max_diff:
                social_influence += social.opinion
                n_socials += 1
        avg_social = social_influence / n_socials if n_socials != 0 else 0

        for nbr in self.model.grid.get_neighbors(pos=self.pos, moore=True, include_center=False, radius=1):
            if abs(nbr.opinion - self.opinion) < self.model.opinion_max_diff:
                n_nbrs += 1
                nbr_influence += nbr.opinion
        avg_nbr = nbr_influence / n_nbrs if n_nbrs != 0 else 0

        return avg_social, avg_nbr

    def update_opinion(self):
        # if not self.fixed_opinion:  # Only update if opinion is not fixed
        #     social_infl, nbr_infl = self.get_external_influences()
        #     new_opinion = self.opinion

        #     if social_infl != 0 and nbr_infl != 0:
        #         new_opinion = (self.weight_own * self.opinion) + (self.weight_socials * social_infl) + (self.weight_neighbors * nbr_infl)
        #     elif social_infl == 0 and nbr_infl != 0:
        #         new_opinion = (self.weight_own * self.opinion) + ((1 - self.weight_own) * nbr_infl)
        #     elif nbr_infl == 0 and social_infl != 0:
        #         new_opinion = (self.weight_own * self.opinion) + ((1 - self.weight_own) * social_infl)

        #     self.opinion = new_opinion
        social_infl, nbr_infl = self.get_external_influences()
        new_opinion = self.opinion

        if social_infl != 0 and nbr_infl != 0:
            new_opinion = (self.weight_own * self.opinion) + (self.weight_socials * social_infl) + (self.weight_neighbors * nbr_infl)
        elif social_infl == 0 and nbr_infl != 0:
            new_opinion = (self.weight_own * self.opinion) + ((1 - self.weight_own) * nbr_infl)
        elif nbr_infl == 0 and social_infl != 0:
            new_opinion = (self.weight_own * self.opinion) + ((1 - self.weight_own) * social_infl)

        self.opinion = new_opinion

    def new_social(self):
        if len(self.unconnected_ids) < self.model.connections_per_step:
            n_potentials = len(self.unconnected_ids)
        else:
            n_potentials = self.model.connections_per_step

        pot_make_ids = np.random.choice(self.unconnected_ids, size=n_potentials, replace=False)
        pot_makes = [social for social in self.model.schedule.agents if social.unique_id in pot_make_ids]

        for potential in pot_makes:
            self.consider_connection(potential_agent=potential, method="ADD")

    def remove_social(self):
        if len(self.socials_ids) < self.model.connections_per_step:
            n_potentials = len(self.socials_ids)
        else:
            n_potentials = self.model.connections_per_step

        pot_break_ids = np.random.choice(self.socials_ids, size=n_potentials, replace=False)
        pot_breaks = [social for social in self.model.schedule.agents if social.unique_id in pot_break_ids]

        for potential in pot_breaks:
            self.consider_connection(potential, method="REMOVE")

    def consider_connection(self, potential_agent, method):
        # p_ij = 1 / (1 + np.exp(self.model.fermi_alpha * (abs(self.opinion - potential_agent.opinion) - self.model.fermi_b)))
        # print(p_ij)
        
        p_ij = 0.45 # For now, hange to vary the probability of connection, higher => higher connections

        if method == "ADD":
            if p_ij > random.random():
                self.model.graph.add_edge(self.unique_id, potential_agent.unique_id)
        if method == "REMOVE":
            if p_ij < random.random():
                self.model.graph.remove_edge(self.unique_id, potential_agent.unique_id)

    def move_pos(self):
        social_infl, av_nbr_op = self.get_external_influences()
        happiness = 1 / (1 + np.exp(self.model.fermi_alpha * (abs(self.opinion - av_nbr_op) - self.model.fermi_b)))

        if happiness < self.model.happiness_threshold:
            self.model.grid.move_to_empty(self)
            self.model.movers_per_step += 1
            
    def connect_different_opinions(self):
        """To connect agents with different opinions, not used in the current model.
        """
        unconnected_agents = [agent for agent in self.model.schedule.agents if agent.unique_id != self.unique_id and agent.unique_id not in self.socials_ids]
        potential_connections = [agent for agent in unconnected_agents if abs(agent.opinion - self.opinion) >= self.model.opinion_max_diff*2] # To change this part to connect agents with differing opinions

        if potential_connections:
            potential = random.choice(potential_connections)
            self.consider_connection(potential_agent=potential, method="ADD")

    def step(self):
        self.new_social()
        self.connect_different_opinions()  # To connect agents with different opinions
        self.remove_social()
        self.move_pos()
        self.update_opinion()
        

class CityModel(Model):
    def __init__(self, sidelength=20, density=0.8, m_barabasi=2, fermi_alpha=5, fermi_b=3, social_factor=0.8, connections_per_step=5, opinion_max_diff=0.2, happiness_threshold=0.8):
        self.sidelength = sidelength
        self.density = density
        self.m_barabasi = m_barabasi
        self.fermi_alpha = fermi_alpha
        self.fermi_b = fermi_b
        self.social_factor = social_factor
        self.connections_per_step = connections_per_step
        self.opinion_max_diff = opinion_max_diff
        self.happiness_threshold = happiness_threshold

        self.schedule = RandomActivation(self)
        self.movers_per_step = 0
        self.n_agents = 0

        self.grid = SingleGrid(self.sidelength, self.sidelength, torus=True)
        self.initialize_population()
        self.graph = nx.barabasi_albert_graph(n=self.n_agents, m=self.m_barabasi)

        self.datacollector = DataCollector(
            model_reporters={
                "graph_modularity": self.calculate_modularity,
                "movers_per_step": lambda m: m.movers_per_step,
                "cluster_coefficient": self.calculate_clustercoef,
                "edges": self.get_graph_dict,
                # "leibovici_entropy_index": self.calculate_l_entropyindex,
                # "altieri_entropy_index": self.calculate_a_entropyindex,
            },
            agent_reporters={
                "opinion": lambda x: x.opinion,
                "position": lambda p: p.pos,
            }
        )
        self.running = True

    def calculate_modularity(self):
        max_mod_communities = greedy_modularity_communities(self.graph)
        mod = modularity(self.graph, max_mod_communities)
        return mod

    def calculate_clustercoef(self):
        cluster_coefficient = average_clustering(self.graph)
        return cluster_coefficient

    def get_graph_dict(self):
        graph_dict = nx.convert.to_dict_of_dicts(self.graph)
        return graph_dict

    # def calculate_l_entropyindex(self):
    #     agent_infolist = [[agent.pos, agent.opinion] for agent in self.schedule.agents]
    #     points = []
    #     types = []

    #     for i in range(len(agent_infolist)):
    #         points.append([agent_infolist[i][0][0], agent_infolist[i][0][1]])

    #     for i in agent_infolist:
    #         if i[1] < 3:
    #             types.append("left")
    #         elif 3 < i[1] < 7:
    #             types.append("middle")
    #         else:
    #             types.append("right")

    #     points = np.array(points)
    #     types = np.array(types)

    #     e = leibovici_entropy(points, types, d=2)
    #     e_entropyind = e.entropy
    #     return e_entropyind

    # def calculate_a_entropyindex(self):
    #     agent_infolist = [[agent.pos, agent.opinion] for agent in self.schedule.agents]
    #     points = []
    #     types = []

    #     for i in range(len(agent_infolist)):
    #         points.append([agent_infolist[i][0][0], agent_infolist[i][0][1]])

    #     for i in agent_infolist:
    #         if i[1] < 3:
    #             types.append("left")
    #         elif 3 < i[1] < 7:
    #             types.append("middle")
    #         else:
    #             types.append("right")

    #     points = np.array(points)
    #     types = np.array(types)

    #     a = altieri_entropy(points, types, cut=2)
    #     a_entropyind = a.entropy
    #     return a_entropyind

    def initialize_population(self):
        # # Calculate number of agents to have fixed opinion
        # num_agents_fixed_opinion = int(self.n_agents * 0.01)  # 1% of total agents

        # # List of all agent positions to choose from
        # agent_positions = list(self.grid.coord_iter())

        # # Shuffle the agent positions to randomize selection
        # random.shuffle(agent_positions)

        # for idx, (content, (x, y)) in enumerate(agent_positions):
        #     if idx < num_agents_fixed_opinion:
        #         agent = Resident(self.n_agents, self, (x, y), fixed_opinion=True)
        #     else:
        #         agent = Resident(self.n_agents, self, (x, y), fixed_opinion=False)

        #     self.grid.place_agent(agent, (x, y))
        #     self.schedule.add(agent)
        #     self.n_agents += 1
        
        for (content, (x, y)) in self.grid.coord_iter():
        # for cell in self.grid.coord_iter():
        #     x, y = cell[1], cell[2]
            if self.random.uniform(0, 1) < self.density:
                agent = Resident(self.n_agents, self, (x, y))
                self.grid.place_agent(agent, (x, y))
                self.schedule.add(agent)
                self.n_agents += 1

    def step(self):
        self.schedule.step()

    def run_model(self, step_count=1, desc="", pos=0, collect_during=True, collect_initial=False):
        if collect_initial:
            self.datacollector.collect(self)
        for i in trange(step_count, desc=desc, position=pos):
            self.step()
            if collect_during:
                self.datacollector.collect(self)
                self.movers_per_step = 0
        if not collect_during:
            self.datacollector.collect(self)

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    model = CityModel(density=0.9, fermi_alpha=4, fermi_b=1, sidelength=15, opinion_max_diff=0.5, happiness_threshold=0.2)
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
    ax[1].plot(range(stepcount), model_df["movers_per_step"], label="Movers per step")
    fig.show()
