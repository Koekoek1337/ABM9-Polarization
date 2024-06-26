from mesa import Agent, Model
from mesa.space import SingleGrid
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
from tqdm import tqdm, trange
import random
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities
from networkx.algorithms.community.quality import modularity
from networkx.algorithms.cluster import average_clustering
import numpy as np
from spatialentropy import leibovici_entropy
from spatialentropy import altieri_entropy
import sys

class CityModel(Model):
    #these are the default parameters
    def __init__(self,
                 sidelength=20,
                 density=0.8,
                 m_barabasi=2,
                 fermi_alpha=5,
                 fermi_b=3,
                 social_factor=0.8,
                 connections_per_step=5,
                 opinion_max_diff=2,
                 happiness_threshold=0.8):

        # model variables
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

        # setting up the Residents:
        self.grid = SingleGrid(self.sidelength, self.sidelength, torus=True)
        self.initialize_population()

        # build a Barabasi Albert social network
        self.graph = nx.barabasi_albert_graph(n=self.n_agents, m=self.m_barabasi)

        self.datacollector = DataCollector(
            model_reporters={
                "graph_modularity": self.calculate_modularity,
                "movers_per_step": lambda m: m.movers_per_step,
                "cluster_coefficient": self.calculate_clustercoef,
                "edges": self.get_graph_dict,
                "leibovici_entropy_index": self.calculate_l_entropyindex,
                "altieri_entropy_index": self.calculate_a_entropyindex,

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

    def initialize_population(self):
        """Initialisation of the population on the 2D grid, with the density prescribed.
        """
        for cell in self.grid.*():
            x = cell[1]
            y = cell[2]

            if self.random.uniform(0,1) < self.density:
                agent = Resident(self.n_agents, self, (x,y))
                self.grid.position_agent(agent, *(x,y))
                self.schedule.add(agent)
                self.n_agents += 1

    def step(self):
        """Run one step of the model."""
        # the scheduler uses the step() functions of the agents
        self.schedule.step()

    def run_model(self, step_count=1, desc="", pos=0, collect_during=True, collect_initial=False):
        """Method that runs the model for a fixed number of steps"""
        # A better way to do this is with a boolean 'running' that is True when initiated,
        # and becomes False when our end condition is met
        if collect_initial:
            self.datacollector.collect(self)

        for i in trange(step_count, desc=desc, position=pos):
            self.step()

            # collect data
            if collect_during:
                self.datacollector.collect(self)

                #set the counter of movers back to zero
                self.movers_per_step = 0

        if not collect_during:
            self.datacollector.collect(self)

#this has been replaced by batch_run.py
def main(argv):
    from .plot_graph import create_graph
    from .plot_grid import sim_grid_plot
    from matplotlib.pyplot import savefig, subplots, hist
    import networkx as nx

    model = CityModel(density=0.9,fermi_alpha=4, fermi_b=1, sidelength=15, opinion_max_diff=0.5, happiness_threshold=0.2)
    stepcount = 50

    model.run_model(step_count=stepcount)
    model_df = model.datacollector.get_model_vars_dataframe()
    agent_df = model.datacollector.get_agent_vars_dataframe()

    fig, axes = subplots(2,2)
    axes = axes.reshape(-1)

    sim_grid_plot(agent_df, grid_axis=[axes[2], axes[3]])
    create_graph(
        agent_df,
        model_df,
        graph_axes=axes[:2],
        layout=nx.spring_layout
        )
    fig.show()
    fig, ax = fig, ax = subplots(1, 2, )
    ax[0].hist(agent_df.loc[[stepcount], ["opinion"]], density = True)
    ax[1].plot(range(stepcount), model_df.movers_per_step, label = "Movers per step")
    fig.show()


if __name__=="__main__":
    import sys
    main(sys.argv[1:])

