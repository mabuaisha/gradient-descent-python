# Standard imports
import os

# Third party imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def evaluate_cost_function(data, output, thetas):
    m = len(output)
    return np.sum((data.dot(thetas) - output) ** 2) / (2 * m)


def evaluate_gradient_descent(data, output, initial_thetas, alpha, iterations):
    cost_history = [0] * iterations
    thetas = []

    for index in range(initial_thetas.size):
        thetas.append([])

    for iteration in range(iterations):
        # h represent the hypothesis
        h = data.dot(initial_thetas)
        # Loss is the difference between hypothesis and the actual value
        loss = h - output
        # gradient value which is going to be used later on with alpha
        # to compute the new values for thetas
        gradient = data.T.dot(loss) / len(output)
        # Compute the new thetas
        initial_thetas = initial_thetas - alpha * gradient
        # Aggregate all thetas since we need them later on for plot against
        # costs
        for index, item in enumerate(initial_thetas):
            thetas[index].append(item)
        # Evaluate cost function using the updated thetas
        cost = evaluate_cost_function(data, output, initial_thetas)
        # For each iteration we should register new cost function in order
        # to plot it later on
        cost_history[iteration] = cost

    return initial_thetas, cost_history, thetas


def plot_cost_and_iterations(iters,
                             cost,
                             plot_location,
                             plot_name,
                             plot_title):
    fig, ax = plt.subplots()
    # ax.plot(np.array(theta_0_lst), np.array(cost_history), 'r')
    # ax.plot(np.array(theta_1_lst), np.array(cost_history), 'g')
    # ax.plot(np.array(theta_2_lst), np.array(cost_history), 'b')
    ax.plot(np.arange(iters), cost, 'r')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Cost')
    ax.set_title(plot_title)
    # fig.show()
    plt.savefig('{0}/{1}.png'.format(plot_location, plot_name))


def parse_dataset(dataset_name):
    return pd.read_csv(
        os.path.join(os.path.dirname(__name__), dataset_name)
    )
