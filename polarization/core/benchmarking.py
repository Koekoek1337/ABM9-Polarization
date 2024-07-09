# this file is meant to enable benchmarking of the models current state
from time import perf_counter
import numpy as np
import matplotlib.pyplot as plt

def time_model_step(model):
    """Times the next step of a model

    Args:
        model (PolarizationModel): model to time

    Returns:
        float: process time
    """
    start_time = perf_counter()
    model.step()
    elapsed_time = perf_counter() - start_time

    return elapsed_time

def benchmark(model, step_count):
    """times a single step of the model and presents an estimation of how long the simulation will take.
    User then has to decide to continue or not.


    Args:
        model (PolarizationModel): model to benchmark
        step_count (int): number of steps

    Returns:
        Bool: True if user inputs 'y', false if user inputs 'n'
    """
    benchmark = time_model_step(model)
    print(f"The first step of this model took {benchmark:.3f} seconds")
    print(f"This sinulation is going to take arount {step_count*benchmark:.2f} seconds. Do you want to proceed? (y of n)")
    proceed = ""
    while proceed != "y":
        proceed = input()
        if proceed == "n":
            return False
        elif proceed == "y":
            return True
        else:
            print(f"WRONG setting chosen. Please type 'y' of 'n'")

# benchmark the model for multiple sizes with one step
def main():
    print(f"start benchmarking")
    # generate models with different number of agents and time single step
    width_list = np.arange(5, 50, 5)
    agents_list = np.square(width_list)
    time_list = []

    for width in width_list:
        time_list.append(time_model_step(width, width, None))

    # create a plot of the time data
    plt.plot(agents_list, time_list, label="agents")
    plt.xlabel("number of agents in model")
    plt.ylabel("time per step (s)")
    plt.yscale('log')
    plt.show()
    # (optional) Extrapolate data


if __name__ == "__main__":
    main()