"""This script contains the visualization support, including the server class"""

from turtle import color
from mesa.visualization.modules import CanvasGrid
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import ChartModule

from polarization.core.model import PolarizationModel, PolarizationAgent


def agent_portrayal(agent):
    if agent == None:
        return

    portrayal = {"Shape": "circle",
                 "Filled": "true",
                 "r": 0.5,
                 "Layer":0}
    if type(agent) is PolarizationAgent:
        if 0 <= agent.opinion < 1:
            portrayal["Color"] = "#1c1cff"
        elif 1 <= agent.opinion < 2:
            portrayal["Color"] = "#2e2eff"
        elif 2 <= agent.opinion < 3:
            portrayal["Color"] = "#4d4dff"
        elif 3 <= agent.opinion < 4:
            portrayal["Color"] = "#7a7aff"
        elif 4 <= agent.opinion < 5:
            portrayal["Color"] = "#adadff"
        elif 5 <= agent.opinion < 6:
            portrayal["Color"] = "#e3e3e3"
        elif 6 <= agent.opinion < 7:
            portrayal["Color"] = "#ff8f8f"
        elif 7 <= agent.opinion < 8:
            portrayal["Color"] = "#ff5252"
        elif 8 <= agent.opinion < 9:
            portrayal["Color"] = "#ff3333"
        elif 9 <= agent.opinion < 10:
            portrayal["Color"] = "#f70000"

    return portrayal


grid = CanvasGrid(agent_portrayal, 20, 20, 500, 500)


# chart1 = ChartModule([{"Label":"graph_modularity",
#                      "Color":"Red"}],
#                      data_collector_name='datacollector')


# chart2 = ChartModule([{"Label":"altieri_entropy_index",
#                      "Color":"Blue"}],
#                      data_collector_name='datacollector')

# chart3 = ChartModule([{"Label":"agents_moved",
#                      "Color":"Black"}],
#                      data_collector_name='datacollector')

model_params = dict(width=20,
                    density=0.9,
                    network_m=2,
                    fermi_alpha=4,
                    fermi_b=1,
                    social_factor =0.8,
                    connections_per_step=5,
                    opinion_max_diff =2,
                    happiness_threshold=0.8)

server = ModularServer(PolarizationModel,
                       [grid],
                       "Polarization Model",
                       model_params)
server.port = 8521
server.launch()
