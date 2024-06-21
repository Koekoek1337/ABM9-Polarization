import matplotlib.pyplot as plt
import networkx as nx

from .agent import PolarizationAgent
from .model import PolarizationModel

class Visualizer():
    def __init__(self, model: PolarizationModel) -> None:
        self.model        = model
        self.graph        = model.graph
        self.pos          = nx.spring_layout(self.graph)
        self.fig, self.ax = plt.subplots()
        
        assert isinstance(self.ax, plt.Axes)

        self.drawNodes()
        plt.show()
    
    def drawNodes(self):
        cmap = []
        for nodeID, opinion in zip(list(range(self.model.nAgents)), self.model.agentOpinions()):
            color = self.colorpicker(opinion)
            cmap.append(color)

        nx.draw(self.graph, node_color=cmap, ax=self.ax)

    def frame(self):
        pass
        

    def colorpicker(self, opinion):
        if opinion < 0: return "Blue"
        if opinion > 0: return "Red"