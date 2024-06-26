"""This script sets up and runs the OFAT Sensitivity Analysis"""
import os, sys
from turtle import color
from mesa.batchrunner import BatchRunner, BatchRunnerMP
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from itertools import combinations

from polarization.core.model import CityModel, Resident
plt.style.use('seaborn')

#the set of parameters to vary, and their corresponding ranges of variation
problem = {
    'num_vars':6,
    'names':['fermi_alpha','fermi_b', 'social_factor',
    'connections_per_step','opinion_max_diff', 'happiness_threshold'],
    'bounds':[[2,10],[0,10],[0,1],[1,5],[0,5],[0,1]],
}

#set repitions(compensate for stochasticity), number of steps and amount of distinct values per variable (N. 100 is good for us)
#total sample size = N * (num_vars+2)
replicates = 5
max_steps = 25
distinct_samples = 100

#set output
model_reporters={"graph_modularity": lambda m:m.calculate_modularity(),
                "altieri_entropy_index": lambda m:m.calculate_a_entropyindex()}

data={}

#generate the samples and run the SA
for i,var in enumerate(problem['names']):
    #get bounds for the variable and get <distinct_samples> samples within this space (uniform)
    param_values = np.linspace(*problem['bounds'][i],num=distinct_samples)

    #set this parameter value to be an integer
    if var == 'connections_per_step':
        param_values = np.linspace(*problem['bounds'][i], num=distinct_samples, dtype=int)

    batch = BatchRunnerMP(CityModel,
                        max_steps=max_steps,
                        iterations=replicates,
                        variable_parameters={var:param_values},
                        model_reporters=model_reporters,
                        display_progress=True)
    batch.run_all()

    data[var]=batch.get_model_vars_dataframe()

#print(data)

#plotting both outputs
def plot_param_var_conf(ax,df,var,param,i):
    """
    Helper function for plot_all_vars. Plots the individual parameter vs
    variables passed.

    Args:
        ax: the axis to plot to
        df: dataframe that holds the data to be plotted
        var: variables to be taken from the dataframe
        param: which output variable to plot
    """
    x = df.groupby(var).mean().reset_index()[var]
    y = df.groupby(var).mean()[param]

    replicates = df.groupby(var)[param].count()
    err_series = (1.96 * df.groupby(var)[param].std()) / np.sqrt(replicates)
    err = err_series.tolist()

    ax.plot(x,y,c='k')
    ax.fill_between(x,y-err,y+err,alpha=0.2,color='k')

    if var == 'fermi_alpha': xlabel = "Fermi-Dirac alpha"
    elif var == 'fermi_b': xlabel = "Fermi-Dirac b"
    elif var == "social_factor": xlabel = "Social Network Influence"
    elif var == "connections_per_step":
        xlabel = "Connections per step"
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    elif var == "opinion_max_diff": xlabel = "Max. difference in opinion"
    elif var == "happiness_threshold": xlabel = "Happiness threshold"

    ax.set_xlabel(xlabel)

def plot_all_vars(df,param):
    """
    Plots the parameters passed vs each of the output variables.

    Args:
        df: dataframe that holds all data
        param: the parameter to be plotted
    """
    fig,axs=plt.subplots(6, figsize=(8,15))

    for i, var in enumerate(problem['names']):
        plot_param_var_conf(axs[i], df[var], var, param, i)

    ylabel = "Graph Modularity" if param == "graph_modularity" else "Altieri Entropy Index"
    fig.supylabel(ylabel)
    fig.tight_layout()

# Saving and loading functions
def saver(dictex):
    for key, val in dictex.items():
        val.to_csv("./data/ofat_{}.csv".format(str(key)))

    with open("./data/keys.txt", "w") as f: #saving keys to file
        f.write(str(list(dictex.keys())))

def loader():
    """Reading data from keys"""
    with open("./data/keys.txt", "r") as f:
        keys = eval(f.read())

    dictex = {}
    for key in keys:
        dictex[key] = pd.read_csv("./data/ofat_{}.csv".format(str(key)))

    return dictex


#saver(data)

data_loaded = loader()
#plot OFAT SA for each output
for param in (['graph_modularity','altieri_entropy_index']):
    print(f"Param: {param}")
    plot_all_vars(data_loaded, param)
    #plt.savefig(f"{param}_OFAT.png")

plt.show()
