"""This file generates the results of a set of Experiments with the model
    """
# order of the parameters:
# sidelength, density, m_barabasi, fermi_alpha, fermi_b, social_factor, connections per step, opinion_max_diff, happiness_threshold


experiments = [
    # default                                       sl  rho  m  a  b   sf  c  ox  ht
    # {"name":"default_small",            "values":[10, 0.8, 2, 4, 1, 0.8, 5, 2, 0.8]},
    # {"name":"default_small_dens",       "values":[10, 0.9, 2, 4, 1, 0.8, 5, 2, 0.8]},
    # {"name":"default_average",          "values":[20, 0.8, 2, 4, 1, 0.8, 5, 2, 0.8]},
    # {"name":"default_average_dens",     "values":[20, 0.9, 2, 4 ,1, 0.8, 5, 2, 0.8]},  # MAIN COMPARIS}N
    # {"name":"default_large",            "values":[30, 0.8, 2, 4, 1, 0.8, 5, 2, 0.8]},
    # {"name":"default_large_dens",       "values":[30, 0.95,2, 4, 1, 0.8, 5, 2, 0.8]},
    # {"name":"default_extra_large_dens", "values":[40, 0.95,2, 4, 1, 0.8, 5, 2, 0.8]},
    # {"name":"tolerance_high",           "values":[20, 0.9, 2, 4, 1, 0.8, 5, 4, 0.8]},
    # {"name":"tolerance_low",            "values":[20, 0.9, 2, 4, 1, 0.8, 5, 1, 0.8]},
    {"name":"alpha_low",                "values":[20, 0.8, 2, 1, 1, 0.3, 5, 1, 0.8]},
    # {"name":"alpha_high",               "values":[20, 0.9, 2, 6, 1, 0.8, 5, 2, 0.8]},
    # {"name":"beta_high",                "values":[20, 0.9, 2, 4, 3, 0.8, 5, 2, 0.8]},
]

from polarization.experiments.experiment_run import run_experiment, plot_experiment
stepcount = 20
iterations = 5
for experiment in experiments:
    agent_dfs, model_dfs = run_experiment(iterations, stepcount, experiment)
    _ = plot_experiment(agent_dfs, model_dfs, stepcount, experiment)
