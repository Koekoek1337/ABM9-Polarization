"""This script contains the functions to plots the networkx graphs of the agent's social network, coloured by their opinion"""
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

def create_graph(agent_df, model_df, graph_axes= [], colormap="bwr", layout=nx.spring_layout, remove_loners=False):
    """The visualisation of the network graph at initital and final steps.

    Args:
        agent_df : df containing agents data from datacollector 
        model_df : df containing model data from datacollector
        graph_axes : Defaults to [].
        colormap : the colouring of agent's opinions. Defaults to "bwr".
        layout (optional): specific layout of network from networkx visualisation. Defaults to nx.spring_layout.
        remove_loners (bool, optional): remove agents that do not have any connections. Defaults to False.
    """
    first_run_dict = model_df.loc[model_df.index[0],"edges"]
    G_init = nx.from_dict_of_dicts(first_run_dict)

    last_run_dict = model_df.loc[model_df.index[-1], "edges"]
    G_last = nx.from_dict_of_dicts(last_run_dict)
    if remove_loners:
        G_last.remove_nodes_from(list(nx.isolates(G_last)))

# permute data
    max_step = agent_df.index.max()[0]

    agent_df = agent_df.reset_index([0])
    agents_firststep = agent_df[agent_df['Step'] == 1] # this can stay, first step is always 1
    agents_laststep = agent_df[agent_df['Step'] == max_step]
# define coloring
    color_map_first = []
    color_map_last = []
    cmap = plt.get_cmap(colormap)

    for node in G_last: # same IDs in both graphs
        opinion_first = agents_firststep['opinion'][node]
        rgba = cmap(opinion_first/10) # maps from 0 to 1, so need to normalize
        color_map_first.append(rgba)

        opinion_last = agents_laststep['opinion'][node]
        rgba = cmap(opinion_last/10) # maps from 0 to 1, so need to normalize
        color_map_last.append(rgba)

# plot
    if len(graph_axes) == 0:
        fig, graph_axes = plt.subplots(1,2)
        cbar = fig.colorbar(ScalarMappable(norm=Normalize(0,1), cmap=cmap), orientation='horizontal',label="Opinion", ticks=[0,1])
        cbar.ax.set_xticklabels(['Far left', 'Far right'])

    nx.draw(G_init, ax=graph_axes[0], node_size=20, node_color=color_map_first, width=0.3, edgecolors='k', pos=layout(G_init))
    nx.draw(G_last, ax=graph_axes[1], node_size=20, node_color=color_map_last, width=0.3, edgecolors='k', pos=layout(G_last))

    graph_axes[0].set_title('Initialized')
    graph_axes[1].set_title('Final State')

def plot_single_graph(model_df, agent_df, colormap = "bwr",layout=nx.spring_layout, ax=None, remove_loners=False):
    """ Similar to previous function, except only plotting one network graph. Used for visualising specific experiments.

    Args:
        agent_df : df containing agents data from datacollector 
        model_df : df containing model data from datacollector
        graph_axes : Defaults to [].
        colormap : the colouring of agent's opinions. Defaults to "bwr".
        layout (optional): specific layout of network from networkx visualisation. Defaults to nx.spring_layout.
        remove_loners (bool, optional): remove agents that do not have any connections. Defaults to False.
    """
    last_run_dict = model_df.loc[model_df.index[-1], "edges"]
    G_last = nx.from_dict_of_dicts(last_run_dict)
    if remove_loners:
        G_last.remove_nodes_from(list(nx.isolates(G_last)))

    max_step = agent_df.index.max()[0]
    agent_df = agent_df.reset_index([0])
    agents_laststep = agent_df[agent_df['Step'] == max_step]

    color_map_last = []
    cmap = plt.get_cmap(colormap)
    for node in G_last:
        opinion_last = agents_laststep['opinion'][node]
        rgba = cmap(opinion_last/10) # maps from 0 to 1, so need to normalize!
        color_map_last.append(rgba)

    if ax == None:
        fig, ax = plt.subplots(1, 1, )

    nx.draw(G_last, ax=ax, node_size=20, node_color=color_map_last, width=0.3, edgecolors='k', pos=layout(G_last))
