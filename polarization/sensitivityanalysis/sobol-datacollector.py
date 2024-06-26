# THIS IS LEGACY CODE -> batch_run.py replaces this file
#
#
import os
import gc
from time import time
from SALib.sample import saltelli
import numpy as np
import pandas as pd
from mesa.batchrunner import BatchRunner, BatchRunnerMP
from IPython.display import clear_output

from polarization.core.model import CityModel

# param_Nina      = param_values[0:160]
# param_Maurits   = param_values[160:320]
# param_Johanna   = param_values[320:480]
# param_Sasha     = param_values[480:640]
# param_Noah      = param_values[640:]

###### --- FILL IN THESE VALUES --- #######

WHO_IS_RUNNING = "maurits"
MY_PARAM_SET = (160, 320)
ps = MY_PARAM_SET

###### --- UNTIL HERE --- #######

replicates = 5 # maybe do 5 for time
max_steps = 50
distinct_samples = 128

problem = {
    'num_vars':6,
    'names':['fermi_alpha','fermi_b', 'social_factor',
    'connections_per_step','opinion_max_diff', 'happiness_threshold'],
    'bounds':[[2,10],[0,10],[0,1],[1,5],[1,10],[0,1]],
}

model_reporters={"Network Modularity": lambda m:m.calculate_modularity(),
                 "Leibovici Entropy Index": lambda m: m.calculate_l_entropyindex(),
                 "Altieri Entropy Index": lambda m: m.calculate_a_entropyindex()}

param_values_all = saltelli.sample(problem, distinct_samples, calc_second_order=False)


divide_into = 10
intervals = []
for i in np.arange(*ps, divide_into):
    interval = (i, i+divide_into)
    intervals.append(interval)
print(intervals)

GENERAL_DIR ="./data/sobol/"

#### Some general checks before starting the simulations ####
# - checks if datadir exists
# - checks if you are sure
if not os.path.isdir(GENERAL_DIR):
    # THIS CHECKS IF THE DIRECTORY ACTUALLY EXISTS before simulating
    raise OSError
print(f"ATENTION: Your about to start the simulation for the following intervals: {intervals}\n")
sure = input("!!!WARNING!!! ARE YOU SURE YOU CORRECTLY FILLED IN THE VALUES ON TOP OF THIS FILE? yes(y) or no(n)")
# sure = input(f" i am sure! (y) \tOR\t Im not sure (n) :")
if sure != "y":
    if sure == "n":
        raise ValueError("You are not sure, please check the values on top")
    else:
        raise ValueError("please put in (y) ot (n)")


# ___________ START LOOP _____________
print("___________ START LOOP _____________")
intervals_collected=0
for interval in intervals:

    param_values = param_values_all[interval[0]:interval[1]]
    count = 0

    # this generates the file in the following directory

    WHICH_SAMPLES = f"#{interval[0]}:{interval[1]}"
    DIR_TO_SAVE = f"{GENERAL_DIR}sobol-{WHICH_SAMPLES}-{WHO_IS_RUNNING}-maxstp={max_steps}_distsmpls={distinct_samples}_rpl={replicates}.csv"
    if os.path.isfile(DIR_TO_SAVE):
        print(f"\nThe interval {interval} has already been simulated. moving to the next...\n")
        intervals_collected +=1
        continue


    batch = BatchRunner(CityModel,
                        max_steps=max_steps,
                        variable_parameters={name:[] for name in problem['names']},
                        model_reporters=model_reporters)

    data = pd.DataFrame(index=range(replicates*len(param_values)),
                                    columns=['fermi_alpha','fermi_b', 'social_factor','connections_perstep','opinion_max_diff', 'happiness_threshold'])

    #these are the outputs that we are measureing
    data['Run'],data["Replicates"],data['Network Modularity'],data["Leibovici Entropy Index"], data["Altieri Entropy Index"] = None, None, None, None, None


    start_time = time()
    for i in range(replicates):
        for vals in param_values:
            #change params that should be integers - ??? still don't get this
            vals = list(vals)
            vals[3] = int(vals[3])

            #transform to dict with parameter names and their values
            variable_params = {}
            for name, val in zip(problem['names'], vals):
                variable_params[name] = val

            batch.run_iteration(variable_params, tuple(vals),count)
            iteration_data = batch.get_model_vars_dataframe().iloc[count]
            iteration_data['Run'] = count #this is quite nb i think
            iteration_data['Replicates'] = i #this is quite nb i think

            data.iloc[count,0:6] = vals
            data.iloc[count,6:] = iteration_data
            count += 1
            clear_output()
            print(f'{count/len((param_values)*(replicates))*100:.2f}% done')

    process_time = time() - start_time
    print(f"Total simulation {process_time}")
    data.to_csv(DIR_TO_SAVE)
    print(f"saved data for interval {interval}")
    del data, batch
    gc.collect()

if intervals_collected == len(intervals):
    print("\n CONGRATULATIONS!!! You are officially done!\n\n")