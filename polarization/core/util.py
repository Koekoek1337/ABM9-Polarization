"""This script contains various utilies used throughout the project. """
from collections import namedtuple
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) # This is your Project Root

def fermi_dirac_graph(d, FERMI_ALPHA, FERMI_B):
    """The Fermi-Dirac probability 

    Args:
        d (float): distance between two agents
        FERMI_ALPHA (int): the parameter that describes the smoothness of the function
        FERMI_B (int): the parameter that defines the homophily 

    Returns:
        [type]: [description]
    """
    pij = 1 / ( 1 + np.exp(FERMI_ALPHA*(abs(d) - FERMI_B)))
    return pij

def plot_fermidirac():
    """A graph to visualise the effect of alpha and b on the probability to connect for a given distance."""
    params = [(10,1),(1,3)]

    distances = np.linspace(0, 10, 100)
    plot_data =[]
    for param in params:
        data = {
            "y": fermi_dirac_graph(distances, param[0], param[1]),
            "alpha": param[0],
            "b": param[1]
            }
        plot_data.append(data)

    for data in plot_data:
        plt.plot(distances, data['y'], label = f"alpha = {data['alpha']}, b = {data['b']}")
    plt.xlabel("Absolute distance in the political belief-space")
    plt.ylabel("Probability to connect")
    plt.title("Fermi-Dirac probability to connect to another node based on political distance")
    plt.legend()
    plt.show()

def read_dataframe(params, dir="./data/"):
    """Reads dataframe from .pkl file that is created by the simulate_parallel function. It uses *params to see get the matching directory

    Args:
        params (ModelParams): Parameters that you want to read the data for

    Returns:
        tuple(Dataframe,Dataframe): agent dataframe[0] and model dataframe[1]
    """
    agent_dir = f"{dir}_agentdf_{params.to_dir()}.pkl"
    agent_df  = pd.read_pickle(agent_dir)
    model_dir = f"{dir}_modeldf_{params.to_dir()}.pkl"
    model_df  = pd.read_pickle(model_dir)
    return (agent_df, model_df)

print(ROOT_DIR)
testagent_df, testmodel_df = (
    pd.read_pickle(f"{ROOT_DIR}/mock_data/test_agentdf.pkl"),
    pd.read_pickle(f"{ROOT_DIR}/mock_data/test_modeldf.pkl")
)
