"""This script sets up the sobol Global Sensitivity Analysis, using the data collected and saved in data/sobol"""

from itertools import combinations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from SALib.analyze import sobol
plt.style.use("default")

problem = {
    'num_vars':5,
    'names':['Fermi-Dirac alpha','Fermi-Dirac b', 'Social Network Influence',
    'Max. difference in opinion','happiness_threshold'],
    'bounds':[[0,4],[0,6],[0,1],[0,4],[0,1]],
}
#set up data, loading from all csv files
DATA_DIR = "./data/sobol/"
datafiles = [dir for dir in os.listdir(DATA_DIR) if dir.startswith("combined_results")]

dataframes = []
for dir in datafiles:
    dataframes.append(pd.read_csv(f"{DATA_DIR}{dir}"))

sobol_df = pd.concat(dataframes)
sobol_df = sobol_df.drop(columns=["Unnamed: 0"])


# Creating a sortable column so the dataframe is ordered
sobol_df['indexing'] = sobol_df[['fermi_alpha', 'fermi_b', 'social_factor', 'opinion_max_diff', 'happiness_threshold']].sum(axis=1)
sobol_id_describe = sobol_df.groupby(by=["indexing"]).describe()
sobol_df = sobol_df.sort_values("indexing")

#Counting duplicates
#sobol_id_describe["fermi_alpha"]["count"].value_counts()

columns = sobol_df.columns
columns = [x.title().replace("_"," ") for x in columns]
columns[0] = "Fermi-Dirac alpha"
columns[1] = "Fermi-Dirac b"
columns[2] = "Social Network Influence"
columns[3] = "Max. difference in opinion"
sobol_df.columns = columns
sobol_df.head()

#SOBOL ANALYZE
si_resident_modularity = sobol.analyze(problem, sobol_df['Network Modularity'].values[:4438], print_to_console=True, calc_second_order=False)
si_resident_entropy = sobol.analyze(problem, sobol_df['Altieri Entropy Index'].values[:4438], print_to_console=True, calc_second_order=False)


def plot_index(s, params, i, ax=None, title=''):
    # Exclude 'happiness_threshold' from the plot
    if 'happiness_threshold' in params:
        index_to_exclude = params.index('happiness_threshold')
        params = [p for j, p in enumerate(params) if j != index_to_exclude]

        if i == '2':
            p = len(params)
            combinations_params = list(combinations(range(p+1), 2))  # include original count
            indices = s['S' + i].reshape((p+1)**2)
            errors = s['S' + i + '_conf'].reshape((p+1)**2)
            
            # Filtering out indices and errors that include the excluded parameter
            indices = [indices[j * (p + 1) + k] for j, k in combinations_params if j != index_to_exclude and k != index_to_exclude]
            errors = [errors[j * (p + 1) + k] for j, k in combinations_params if j != index_to_exclude and k != index_to_exclude]
        else:
            indices = np.delete(s['S'+i], index_to_exclude)
            errors = np.delete(s['S'+i+'_conf'], index_to_exclude)

    else:
        if i == '2':
            p = len(params)
            indices = s['S' + i].reshape((p**2))
            indices = indices[~np.isnan(indices)]
            errors = s['S'+ i + '_conf'].reshape((p**2))
            errors = errors[~np.isnan(errors)]
        else:
            indices = s['S'+i]
            errors = s['S'+ i + '_conf']

    if ax is None:
        fig, ax = plt.subplots()
    l = len(indices)
    ax.set_yticks(range(l), params)
    ax.set(title=title)
    ax.errorbar(indices, range(l), xerr=errors, linestyle='None', marker='o', capsize=5)
    ax.axvline(0, c='k')
    ax.grid(True)
    plt.tight_layout()

# Adjust the plotting calls to remove 'happiness_threshold'
fig, ax = plt.subplots(2, 2, figsize=(10, 8), sharey=True, sharex=True)
plot_index(si_resident_modularity, problem["names"], i="1", ax=ax[0][0], title="1st Order Sensitivity\nModularity")
plot_index(si_resident_entropy, problem["names"], i="1", ax=ax[1][0], title="1st Order Sensitivity\nEntropy")
plot_index(si_resident_modularity, problem["names"], i="T", ax=ax[0][1], title="Total Order Sensitivity\nModularity")
plot_index(si_resident_entropy, problem["names"], i="T", ax=ax[1][1], title="Total Order Sensitivity\nEntropy")
plt.savefig("figures/GSA4.png")
