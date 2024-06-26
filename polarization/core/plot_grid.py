"""This script contains the functions to plot the agents on 2D grid, coloured by their political opinion"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
plt.style.use("default")
import matplotlib.cm as cm
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

def grid_plot(agent_df, plot_step, sidelength, ax=None):
    """Plotting the agents on a grid, coloured by their political opinion

    Args:
        agent_df ([type]): df containing agents data from datacollector 
        plot_step ([type]): 
        sidelength (int): side length used in model
    """
    agentdf = pd.DataFrame(agent_df.loc[[plot_step], ["position", "opinion"]])\
        .sort_values("position")\
        .droplevel(0)

    op_grid = np.empty((sidelength, sidelength))

    #since density of agents is not 1, there will be some empty positions
    op_grid[:] = np.nan

    for index, row in agentdf.iterrows():
        op_grid[row["position"]] = row["opinion"]

    if ax == None:
        fig, ax = plt.subplots()

    current_cmap = ScalarMappable(norm=Normalize(vmin=-1, vmax=1), cmap="bwr").get_cmap().copy()
    current_cmap.set_bad(color='lightgray')

    ax.imshow(op_grid, cmap=current_cmap)
    ax.set_title(f'step number {plot_step}')
    ax.grid(False)


def sim_grid_plot(agent_df, grid_axis=[]):
    """Plotting the grid at the first and last step of the simulation

    """
    max_step = agent_df.index.max()[0]
    sidelength = agent_df["position"].max()[0] + 1
    print(sidelength)
    if len(grid_axis) == 0:
        fig, grid_axis = plt.subplots(1,2, figsize=(10,4))
        fig.suptitle("Grid plot")
    grid_plot(agent_df, 1, sidelength, grid_axis[0])
    grid_plot(agent_df, max_step, sidelength, grid_axis[1])


