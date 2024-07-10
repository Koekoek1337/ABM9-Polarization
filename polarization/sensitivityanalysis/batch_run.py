# """This script runs Sobol Sensitvity analysis on a given set of parameters and saves the results to a csv file"""
# import gc
# import os
# from SALib.sample import saltelli
# import numpy as np

# from polarization.mesa_fix.batchrunner import BatchRunnerMP
# from polarization.core.model import PolarizationModel

# # param_Nina      = param_values[0:160]
# # param_Maurits   = param_values[160:320]
# # param_Johanna   = param_values[320:480]
# # param_Sasha     = param_values[480:640]
# # param_Noah      = param_values[640:]

# ###### --- FILL IN THESE VALUES --- #######

# WHO_IS_RUNNING = "maurits"
# MY_PARAM_SET = (0, 19)
# ps = MY_PARAM_SET

# ###### --- UNTIL HERE --- #######

# replicates = 1
# max_steps = 20
# distinct_samples = 128


# # We define our variables and bounds
# problem = {
#     'num_vars':5,
#     'names':['fermi_alpha','fermi_beta', 'connection_influence',
#     'opinion_threshold', 'happiness_threshold'],
#     'bounds':[[0,4],[0,6],[0,1],[0,4],[0,1]],
# }
# model_reporters={"Network Modularity": lambda m:m.calc_modularity(),
#                  "Leibovici Entropy Index": lambda m: m.calc_l_entropy(),
#                  "Altieri Entropy Index": lambda m: m.calc_a_entropy()}

# param_values_all = saltelli.sample(problem, distinct_samples, calc_second_order=False)

# #divided the problem into intervals so that data could be saved throughout and all would not be lost if computer crashed
# divide_into = 20 # actually size of the division
# intervals = []
# for i in np.arange(*ps, divide_into):
#     interval = (i, i+divide_into)
#     intervals.append(interval)
# print(intervals)

# GENERAL_DIR ="./data/sobol/"

# for interval in intervals:
# 	param_values = param_values_all[interval[0]:interval[1]]
# 	WHICH_SAMPLES = f"#{interval[0]}:{interval[1]}"
# 	DIR_TO_SAVE = f"{GENERAL_DIR}sobol3-{WHICH_SAMPLES}-{WHO_IS_RUNNING}-maxstp={max_steps}_distsmpls={distinct_samples}_rpl={replicates}.csv"
# 	if os.path.isfile(DIR_TO_SAVE):
# 		print(f"\nThe interval {interval} has already been simulated. moving to the next...\n")
# 		continue

# 	tuples = set()
# 	for i in range(len(param_values)):
# 		tuples.add(tuple(param_values[i]))

# 	print(f"Running {replicates} replicate(s) for {len(tuples)} unique parameter combinations")

# 	variable_parameters = [
# 		{"fermi_alpha":param_values[i][0],
# 		"fermi_beta":param_values[i][1],
# 		"connection_influence":param_values[i][2],
# 		"opinion_threshold":param_values[i][3],
# 		"happiness_threshold":param_values[i][4]}
# 		for i in range(len(param_values))
# 	]

# 	batch = BatchRunnerMP(PolarizationModel,
# 						iterations=replicates,
# 						variable_parameters=variable_parameters,
# 						max_steps=max_steps,
# 						model_reporters=model_reporters,
# 						display_progress=True)
# 	batch.run_all()

# 	dataframe = batch.get_model_vars_dataframe()
# 	print(dataframe.head())
# 	print(dataframe.describe())
# 	dataframe.to_csv(DIR_TO_SAVE)
# 	print(f"\n Done with interval {interval} \n")

# 	del dataframe, batch, variable_parameters, tuples
# 	gc.collect()

import gc
import os
from SALib.sample import saltelli
import numpy as np

from polarization.mesa_fix.batchrunner import BatchRunnerMP
from polarization.core.model import PolarizationModel
from mesa.batchrunner import FixedBatchRunner
###### --- FILL IN THESE VALUES --- #######

WHO_IS_RUNNING = ""
MY_PARAM_SET = (0, 19)
ps = MY_PARAM_SET

###### --- UNTIL HERE --- #######

replicates = 10
max_steps = 20
distinct_samples = 128


# We define our variables and bounds
problem = {
    'num_vars': 5,
    'names': ['fermi_alpha', 'fermi_beta', 'connection_influence', 'opinion_threshold', 'happiness_threshold'],
    'bounds': [[0, 4], [0, 6], [0, 1], [0, 4], [0, 1]],
}
model_reporters = {
    "Network Modularity": lambda m: m.calc_modularity(),
    "Leibovici Entropy Index": lambda m: m.calc_l_entropy(),
    "Altieri Entropy Index": lambda m: m.calc_a_entropy()
}

param_values_all = saltelli.sample(problem, distinct_samples, calc_second_order=False)

# divided the problem into intervals so that data could be saved throughout and all would not be lost if computer crashed
divide_into = 20  # actually size of the division
intervals = []
for i in np.arange(*ps, divide_into):
    interval = (i, i + divide_into)
    intervals.append(interval)
print(intervals)

GENERAL_DIR = "./data/sobol/"

for interval in intervals:
    param_values = param_values_all[interval[0]:interval[1]]
    WHICH_SAMPLES = f"#{interval[0]}:{interval[1]}"
    DIR_TO_SAVE = f"{GENERAL_DIR}sobol3-{WHICH_SAMPLES}-{WHO_IS_RUNNING}-maxstp={max_steps}_distsmpls={distinct_samples}_rpl={replicates}.csv"
    if os.path.isfile(DIR_TO_SAVE):
        print(f"\nThe interval {interval} has already been simulated. moving to the next...\n")
        continue

    tuples = set()
    for i in range(len(param_values)):
        tuples.add(tuple(param_values[i]))

    print(f"Running {replicates} replicate(s) for {len(tuples)} unique parameter combinations")

    variable_parameters = [
        {"fermi_alpha": param_values[i][0],
         "fermi_beta": param_values[i][1],
         "connection_influence": param_values[i][2],
         "opinion_threshold": param_values[i][3],
         "happiness_threshold": param_values[i][4]}
        for i in range(len(param_values))
    ]

    batch = FixedBatchRunner(PolarizationModel,
                          iterations=replicates,
                          parameters_list=variable_parameters,
                          max_steps=max_steps,
                          model_reporters=model_reporters,
                          display_progress=True)
    batch.run_all()

    dataframe = batch.get_model_vars_dataframe()
    print(dataframe.head())
    print(dataframe.describe())
    dataframe.to_csv(DIR_TO_SAVE)
    print(f"\n Done with interval {interval} \n")

    del dataframe, batch, variable_parameters, tuples
    gc.collect()

# Combine all CSV files into one
import pandas as pd

all_files = [os.path.join(GENERAL_DIR, f) for f in os.listdir(GENERAL_DIR) if f.endswith('.csv')]
combined_df = pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True)
combined_df.to_csv(f"{GENERAL_DIR}combined_results3.csv", index=False)
print(f"\n All results have been combined into {GENERAL_DIR}combined_results.csv \n")
