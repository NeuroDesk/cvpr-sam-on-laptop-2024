import numpy as np
import matplotlib.pyplot as plt
import os
from os.path import join
import pandas as pd
import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    '-models',
     nargs='+',
    default="work_dir",
    help='path to the model metrics',
)

# Calculate dominance
def is_dominated(x, y, X, Y):
    return np.any((X > x) & (Y < y))

def find_pareto_front(x, y):
    pareto_front = []
    for i in range(len(x)):
        if not is_dominated(x[i], y[i], x, y):
            pareto_front.append((x[i], y[i]))
    return pareto_front


if __name__ == '__main__':
    args = parser.parse_args()
    dirname = os.path.dirname(__file__)
    models = args.models
    dsc = []
    runtime = []
    for model in models:
        if not os.path.exists(join(dirname, model + '_results')):
            raise FileNotFoundError(f'{model} does not exist')
        
        dsc_csv = pd.read_csv(join(dirname, model + '_results', 'metrics.csv'), sep=',', usecols=['dsc'])
        runtime_csv = pd.read_csv(join(model + '_results', 'running_time.csv'), sep=',', usecols=['Running time (mean)'])
        # print(dsc_csv.values.flatten(), runtime_csv)
        dsc_mean = np.mean(dsc_csv.values.flatten())
        dsc.append(dsc_mean)
        runtime_mean = np.mean(runtime_csv.values.flatten())
        runtime.append(runtime_mean)
        plt.scatter(dsc_mean, runtime_mean, label=model)

    print(dsc, runtime)
    # Plot the Pareto front
    pareto_front = find_pareto_front(np.array(dsc), np.array(runtime))
    pareto_front_x = [point[0] for point in pareto_front]
    pareto_front_y = [point[1] for point in pareto_front]
    # plt.plot(*zip(*pareto_front), color='red', label='Pareto Front')
    plt.plot(pareto_front_x, pareto_front_y, color='red', label='Pareto Front', linewidth=2)

    # Add labels and legend
    plt.xlim(0, 1)
    plt.xlabel('DSC')
    plt.ylabel('Time')
    plt.title(f'Pareto Front for {", ".join(model for model in models)} on test demo')
    plt.legend()

    # Show the plot
    plt.grid(True)

    # Save the plot
    plt.savefig('pareto_front.png')