# Standard imports
import os

# Third party imports
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

# Local imports
from linear_regession import (
    evaluate_gradient_descent,
    parse_dataset,
    plot_cost_and_iterations
)


def prepare_dataset(dataset, sample_size, features_num):
    input_data = dataset.iloc[:, :features_num]
    output_data = dataset.iloc[:, -1]

    scaler = StandardScaler()
    input_data = scaler.fit_transform(input_data)

    data_train = input_data[:sample_size, :features_num]
    data_train = np.c_[np.ones(len(data_train), dtype='int64'), data_train]
    output_train = output_data[:sample_size]

    data_test = input_data[sample_size:, :features_num]
    data_test = np.c_[np.ones(len(data_test), dtype='int64'), data_test]
    output_test = output_data[sample_size:]

    return data_train, output_train, data_test, output_test


def clean_hardware_dataset(dataset):
    # Used Encoding Categorical Features to do the cleaning
    # This will help to return all columns types so that we can get them
    # for encoding categorical features into numerical values.
    categorical_feature_mask = dataset.dtypes == object
    categorical_cols = dataset.columns[categorical_feature_mask].tolist()
    label_encoder = LabelEncoder()

    dataset[categorical_cols] = \
        dataset[categorical_cols].apply(
            lambda col: label_encoder.fit_transform(col)
        )


def apply_gradient_descent(data_input,
                           output,
                           initial_thetas,
                           alpha,
                           iterations,
                           plot_location,
                           plot_name,
                           plot_title):
    # Apply training data
    new_thetas, cost_history, thetas_matrix = \
        evaluate_gradient_descent(
            data_input,
            output,
            initial_thetas,
            alpha,
            iterations
        )

    # Plot cost function
    plot_cost_and_iterations(iterations,
                             cost_history,
                             plot_location,
                             plot_name,
                             plot_title)


def apply_hardware_dataset(dataset):
    # These used for different runs
    alphas = [
        0.001,
        0.002,
        0.003,
        0.004,
        0.005,
        0.006,
        0.007,
        0.008,
        0.009,
        0.01,
    ]
    # Pre-processing the dataset
    clean_hardware_dataset(dataset)

    # Provide the number of sample size for training
    sample_size = 147
    features = 9

    data_train, output_train, data_test, output_test = prepare_dataset(
        dataset, sample_size, features
    )

    # Initial thetas
    thetas = np.zeros(data_train.shape[1])
    # Set number of iterations
    iterations = 2000

    plot_location = \
        os.path.join(
            os.path.dirname(__name__),
            'datasets/hardware'
        )

    for index, alpha in enumerate(alphas):
        # Apply gradient descent for training data
        plot_title = 'Training Data, Run {}'.format(index + 1)
        plot_name = 'trainig_data_run_{}'.format(index + 1)
        apply_gradient_descent(data_train,
                               output_train,
                               thetas,
                               alpha,
                               iterations,
                               plot_location,
                               plot_name,
                               plot_title)

        # Apply gradient descent for testing data
        plot_title = 'Testing Data, Run {}'.format(index + 1)
        plot_name = 'testing_data_run_{}'.format(index + 1)
        apply_gradient_descent(data_test,
                               output_test,
                               thetas,
                               alpha,
                               iterations,
                               plot_location,
                               plot_name,
                               plot_title)


if __name__ == '__main__':
    data = parse_dataset('datasets/hardware/machine-data.txt')

    # Apply dataset for Hardware
    apply_hardware_dataset(data)

    # Apply the second dataset

    # Apply the third dataset
