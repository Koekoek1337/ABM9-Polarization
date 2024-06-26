import unittest
from networkx import barabasi_albert_graph, complete_graph, empty_graph, to_edgelist
from unittest.mock import patch

import os, sys
from polarization.core.model import CityModel, Resident
from .test_model import clear_model


global N_POTENTIAL_CONNECTIONS
N_POTENTIAL_CONNECTIONS = 1


class TestResident(unittest.TestCase):
    def setUp(self) -> None:
        self.model = CityModel(sidelength=2,density=1)
        clear_model(self.model)
        self.model.initialize_population()
        self.model.graph = barabasi_albert_graph(n=self.model.n_agents, m=self.model.m_barabasi)

        self.test_agent = self.model.schedule.agents[0]

        self.random_random_patcher = patch('polarization.core.model.random.random')
        self.random_choices_patcher = patch('polarization.core.model.random.choices')

        return super().setUp()

    def test_initialisation(self):
        assert type(self.test_agent) == Resident
        assert type(self.test_agent.model) == CityModel
        assert self.test_agent.pos != 0
        assert self.test_agent.pos == (0, 0)
        assert len(self.model.schedule.agents) == self.model.sidelength * \
            self.model.sidelength

    def test_new_social_accept_all(self):
        self.patch_random()
        self.random_patch.return_value = 0

        self.model.graph = empty_graph(n=self.model.n_agents)
        assert len(to_edgelist(self.model.graph)) == 0

        self.test_agent.new_social()

        assert len(self.test_agent.socials_ids) == 3, "edgelist is empty"

    def test_new_social_accept_none(self):
        self.patch_random()
        self.random_patch.return_value = 1

        self.model.graph = empty_graph(n=self.model.n_agents)

        self.test_agent.new_social()

        assert len(self.test_agent.socials_ids) == 0, "it made friends!"

    def test_new_social_while_all_connected(self):
        self.patch_random()
        self.random_patch.return_value = 1

        self.model.graph = complete_graph(n=self.model.n_agents)
        print(self.test_agent.unconnected_ids)
        assert len(
            self.test_agent.unconnected_ids) == 0,  "unconnected_agents are "

        self.test_agent.new_social()
        assert len(self.test_agent.unconnected_ids) == 0, "It removed stuf?!"

    def test_remove_social(self):
        self.patch_random()

        self.model.graph = complete_graph(n=self.model.n_agents)
        assert len(to_edgelist(self.model.graph)
                   ) > self.model.sidelength * self.model.sidelength
        self.test_agent.remove_social()

        assert ((self.test_agent.unique_id, 1) in to_edgelist(
            self.model.graph)) == False, "Edge has not been removed"

    def test_datacollector(self):
        self.model.graph = empty_graph(n=self.model.n_agents)
        self.model.grid.remove_agent(self.model.schedule.agents[-1])
        self.model.graph.remove_node(self.model.schedule.agents[-1].unique_id)
        self.model.schedule.remove(self.model.schedule.agents[-1])
        self.model.step()
        agent_df = self.model.datacollector.get_agent_vars_dataframe()
        assert agent_df.shape == (0, 2), "shape is different than expected"

    def patch_random(self):
        self.test_resident = self.model.schedule.agents[1]

        self.random_patch = self.random_random_patcher.start()
        self.random_patch.return_value = 1

        self.choices_patch = self.random_choices_patcher.start()
        self.choices_patch.return_value = [self.test_resident]

        self.addCleanup(self.random_random_patcher.stop)
        self.addCleanup(self.random_choices_patcher.stop)
