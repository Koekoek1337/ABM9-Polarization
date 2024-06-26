import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import networkx as nx
import numpy as np

from .agent import PolarizationAgent
from .model import PolarizationModel

import matplotlib.colors as mcolors
import imageio

class Visualizer():
    def __init__(self, model: PolarizationModel) -> None:
        self.model = model
        self.graph = model.graph
        self.pos = nx.spring_layout(self.graph)
        self.fig, ax = plt.subplots()
        assert isinstance(ax, plt.Axes)
        self.ax = ax

        self.cmap = mcolors.LinearSegmentedColormap.from_list(
            "opinion_cmap", ["blue", "grey", "red"])

        self.text = self.ax.text(0.8, 0.9, f"i = {0}")
    
        self.drawNetworkx()
    
    def drawNetworkx(self):
        cmap = []
        for nodeID in range(self.model.nAgents):
            opinion = self.model.agentOpinions()[nodeID]
            color = self.colorpicker(opinion)
            cmap.append(color)

        nx.draw_circular(self.graph, node_color=cmap, ax=self.ax)

    def colorpicker(self, opinion):
        
        return self.cmap((opinion + 1) / 2)  

    def _frame(self, framenum):
        self.ax.clear()
        self.drawNetworkx()
        self.model.step()

        self.text = self.ax.text(0.8, 0.9, f"i = {framenum}")
        return
    
    def run(self, nFrames = 100, fps = 10):
        self.ani = FuncAnimation(self.fig, self._frame, frames=nFrames, interval=300 / fps, repeat = False)
        plt.show()
        self.ani.save(f"test.gif")

# import matplotlib.pyplot as plt
# import matplotlib.colors as mcolors
# from matplotlib.animation import FuncAnimation
# import networkx as nx
# import numpy as np
# from .agent import PolarizationAgent  # Adjust import paths as necessary
# from .model import PolarizationModel  # Adjust import paths as necessary
# import imageio

# class Visualizer():
#     def __init__(self, model):
#         self.model = model
#         self.graph = model.graph
#         self.pos = nx.spring_layout(self.graph)
#         self.fig, self.ax = plt.subplots()
#         self.frames = []

#         # 创建一个从蓝到红的颜色映射，经过灰色
#         self.cmap = mcolors.LinearSegmentedColormap.from_list(
#             "opinion_colormap",
#             ["blue", "gray", "red"]
#         )
#         self.norm = plt.Normalize(-1, 1)  # 假设opinion值范围从-1到1

#     def drawNetworkx(self, step_number):
#         """Draw the network graph with a step number overlay."""
#         self.ax.clear()
#         cmap = [self.colorpicker(agent.opinion) for agent in self.model.scheduler.agents]
#         nx.draw_networkx(self.graph, pos=self.pos, node_color=cmap, ax=self.ax)
#         # Display the current step number in the top left corner of the graph
#         self.ax.text(0.05, 0.95, f'Step: {step_number}', transform=self.ax.transAxes,
#                      fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', edgecolor='0.3'))
#         self.fig.canvas.draw()

#     def colorpicker(self, opinion):
#         """Return a color based on the opinion value."""
#         return self.cmap(self.norm(opinion))

#     def _frame(self, framenum):
#         """Capture each frame including the step number."""
#         self.drawNetworkx(framenum)
#         self.model.step()
#         frame_image = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype='uint8')
#         frame_image = frame_image.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
#         self.frames.append(frame_image)

#     def run(self, nFrames=100, fps=30):
#         """Generate frames and finish the animation."""
#         for i in range(nFrames):
#             self._frame(i)
#         self.finish(fps)

#     def finish(self, fps):
#         """Save the captured frames as a GIF with the specified frames per second."""
#         with imageio.get_writer('animation.gif', mode='I', fps=fps) as writer:
#             for frame in self.frames:
#                 writer.append_data(frame)
#         print("Animation saved successfully.")



