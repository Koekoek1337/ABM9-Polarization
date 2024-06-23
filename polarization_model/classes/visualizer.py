import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import networkx as nx

from .agent import PolarizationAgent
from .model import PolarizationModel

class Visualizer():
    def __init__(self, model: PolarizationModel, bins = 9) -> None:
        self.model        = model
        """PolarizationModel"""
        self.graph        = model.graph
        """Networkx Graph instance of polarization model"""
        self.fig, axes    = plt.subplots(1, 2)
        axNx, axDist      = axes

        assert isinstance(axNx,   plt.Axes)
        assert isinstance(axDist, plt.Axes)

        self.bins         = bins
        """Amount of bins for opinion Histogram"""

        self.axNx   = axNx
        """Axes instance for graph plot"""
        self.axDist = axDist
        """Axes instance for distribution plot"""

        self.text = self.axNx.text(0.8, 0.9, f"i = {0}")
        self._opinions = self.model.agentOpinions()
    
        self.drawNetworkx()
    
    def drawNetworkx(self):
        self.axNx.clear()
        cmap = []
        for nodeID, opinion in zip(list(range(self.model.nAgents)), self._opinions):
            color = self.colorpicker(opinion)
            cmap.append(color)

        nx.draw_circular(self.graph, node_color=cmap, ax=self.axNx, node_size=20)
    
    def drawDist(self):
        self.axDist.clear()
        self.axDist.hist(self._opinions, self.bins, range=(-1,1), density=True)
        self.axDist.set_xbound(-1, 1)
        # self.axDist.set_ybound(0, 1)

    def colorpicker(self, opinion):
        if opinion < 0.0:  return "Blue"
        if opinion > 0.0:  return "Red"
        else: return "Grey"

    def _frame(self, framenum):
        self._opinions = self.model.agentOpinions()
        self.drawNetworkx()
        self.drawDist()
        self.model.step()

        self.text = self.axNx.text(0.8, 0.9, f"i = {framenum}")
        return
    
    def run(self, nFrames = 100, fps = 1):
        self.ani = FuncAnimation(self.fig, self._frame, frames=nFrames, interval=1000 / fps, repeat = False)
        plt.show()
        self.ani.save(f"test.mp4")

