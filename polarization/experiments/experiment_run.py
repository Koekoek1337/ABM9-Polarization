import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from polarization.core.model import PolarizationModel
from polarization.core.plot_graph import plot_single_graph
from polarization.core.plot_grid import grid_plot

sns.set_theme()
sns.set_color_codes()

def plot_errorHue(mean_list, std_list, label, start=0, sample_data=None, sample_style='-r', ax=None):
    """Plotting the information from all repetitions of a run.

    Args:
        mean_list : mean result from all repetitions
        std_list : std of result from all repetitions
        label : name of output measure
        start : Defaults to 0.
        sample_data : The sample experiment run. Defaults to None.
        sample_style : colouring the sample data differently. Defaults to '-r'.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    x_array = range(start, len(mean_list) + start)
    ax.plot(
        x_array,
        mean_list,
        label=label
    )
    if sample_data is not None and not sample_data.empty:
        ax.plot(x_array, sample_data, sample_style, label="sample")

    ax.fill_between(
        x_array,
        mean_list + std_list,
        mean_list - std_list,
        alpha=0.5
    )
    ax.legend()

PARAMS_NAMES = ["width", "density", "network_m", "fermi_alpha", "fermi_b", "social_factor", "opinion_max_diff", "conformity"]

def run_experiment(iterations, stepcount, experiment):
    """ Running experiment and collecting data

    Args:
        iterations : number of repetitions of run
        stepcount : length of run
        experiment : particular set of parameters

    Returns:
        agents_dfs, model_dfs
    """
    model_dfs = []
    agent_dfs = []
    for i in range(iterations):
        model = PolarizationModel(*(experiment["values"]))
        model.run_model(step_count=stepcount, desc=f'step {i}', collect_initial=True)

        model_dataframe = model.datacollector.get_model_vars_dataframe()
        model_dataframe['step'] = model_dataframe.index
        model_dfs.append(model_dataframe)
        agent_dfs.append(model.datacollector.get_agent_vars_dataframe())
    return agent_dfs, model_dfs


def plot_experiment(agent_dfs, model_dfs, stepcount, experiment):
    """ Plots a 2x3 grid of visual results from an experiment.
    Visuals included are:
    Network graph, Modularity, Movers per step, Spatial grid, Entropy, Sample Opinion Distribution.
    Where applicable, the plots show the data from all repetitions with mean and std in blue and then
    also plots the sample run in red.

    Args:
        agent_dfs : df containing data from agent reporters
        model_dfs : df containing data from model reporters
        stepcount : length of run
        experiment : particular set of parameters
    """
    samples = 20
    if len(model_dfs) < 20: 
        samples = len(model_dfs)
        
    # Aggregated data for mean and std calculations
    x1, x2, x3 = [], [], []
    y1, y2, y3 = [], [], []
    
    for si in range(samples):
        sample = agent_dfs[si], model_dfs[si]
        model_df = pd.concat(model_dfs)
        model_df = model_df.drop(columns=['edges', 'leibovici_entropy_index'])
        agent_df = pd.concat(agent_dfs)
        agg_model_df = model_df.groupby(by='step').aggregate(['std', 'mean'])

        headers = agg_model_df.columns.unique(level=0)
        x1.append(agg_model_df[headers[0]]['mean'].values)
        y1.append(agg_model_df[headers[0]]['std'].values)
        x2.append(agg_model_df[headers[3]]['mean'].values)
        y2.append(agg_model_df[headers[3]]['std'].values)
        x3.append(agg_model_df[headers[1]]['mean'].values)
        y3.append(agg_model_df[headers[1]]['std'].values)

    # Calculate mean and std across samples
    x1_mean = np.mean(x1, axis=0)
    x2_mean = np.mean(x2, axis=0)
    x3_mean = np.mean(x3, axis=0)
    y1_std = np.mean(y1, axis=0)
    y2_std = np.mean(y2, axis=0)
    y3_std = np.mean(y3, axis=0)
    
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(12, 7))
    
    ax[0][0].set(title="Network Graph")
    plot_single_graph(sample[1], sample[0], ax=ax[0][0], layout=nx.nx_pydot.graphviz_layout)
    ax[1][0].set(title="Grid")
    grid_plot(sample[0], stepcount, experiment["values"][0], ax=ax[1][0])

    plot_errorHue(x1_mean, y1_std, ax=ax[0][1], label='Modularity') 
    plot_errorHue(x2_mean, y2_std, ax=ax[1][1], label='Altieri Entropy Index')
    plot_errorHue(x3_mean, y3_std, ax=ax[0][2], label='Movers Per Step', start=1)

    ax[1][2].hist(sample[0].loc[stepcount, ["opinion"]], color='r', density=True)
    
    ax[0][1].set(xlabel="step", title="Modularity ")
    ax[1][1].set(xlabel="step", title="Altieri Entropy")
    ax[0][2].set(xlabel="step", title="Movers Step")
    ax[1][2].set(xlabel="Opinion", title="Opinion Distribution")

    plt.tight_layout()
    
    filename = 'without0.5_50'
    plt.savefig(f"figures/{filename}.png")

