import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import networkx as nx

from .agent import PolarizationAgent
from .model import PolarizationModel

class Visualizer():
    def __init__(self, model: PolarizationModel) -> None:
        self.model        = model
        self.graph        = model.graph
        self.pos          = nx.spring_layout(self.graph)
        self.fig, ax      = plt.subplots()
        assert isinstance(ax, plt.Axes)
        self.ax = ax

        self.text = self.ax.text(0.8, 0.9, f"i = {0}")
    
        self.drawNetworkx()
    
    def drawNetworkx(self):
        cmap = []
        for nodeID, opinion in zip(list(range(self.model.nAgents)), self.model.agentOpinions()):
            color = self.colorpicker(opinion)
            cmap.append(color)

        nx.draw_circular(self.graph, node_color=cmap, ax=self.ax)

    def colorpicker(self, opinion):
        if opinion < 0.0:  return "Blue"
        if opinion > 0.0:  return "Red"
        else: return "Grey"

    def _frame(self, framenum):
        self.ax.clear()
        self.drawNetworkx()
        self.model.step()

        self.text = self.ax.text(0.8, 0.9, f"i = {framenum}")
        return
    
    def run(self, nFrames = 100, fps = 1):
        self.ani = FuncAnimation(self.fig, self._frame, frames=nFrames, interval=1000 / fps, repeat = False)
        plt.show()
        self.ani.save(f"test.mp4")

