# Standard imports
import os
import logging

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

logging.basicConfig(level=logging.INFO,
                    format='%(levelname)s:%(message)s')
logger = logging.getLogger(__name__)


def prepare_dataset(dataset, sample_size, features_num) -> tuple:
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

    logger.info("Thetas For each run {0}".format(new_thetas))

    # Plot cost function
    plot_cost_and_iterations(iterations,
                             cost_history,
                             plot_location,
                             plot_name,
                             plot_title)


def apply_dataset_iter(runs,
                       thetas,
                       data_train,
                       output_train,
                       data_test,
                       output_test,
                       alphas,
                       iterations,
                       plot_location):

    for index in range(runs):
        # Apply gradient descent for training data
        thetas = thetas * (index + 1)
        plot_title = 'Training Data, Run {0}, Alpah {1}' \
                     ''.format(index + 1, alphas[index])
        plot_name = 'trainig_data_run_{}'.format(index + 1)
        apply_gradient_descent(data_train,
                               output_train,
                               thetas,
                               alphas[index],
                               iterations,
                               plot_location,
                               plot_name,
                               plot_title)

        # Apply gradient descent for testing data
        plot_title = 'Testing Data, Run {0}, Alpah {1}' \
                     ''.format(index + 1, alphas[index])
        plot_name = 'testing_data_run_{}'.format(index + 1)
        apply_gradient_descent(data_test,
                               output_test,
                               thetas,
                               alphas[index],
                               iterations,
                               plot_location,
                               plot_name,
                               plot_title)


def apply_hardware_dataset(dataset):
    # Pre-processing the dataset
    clean_hardware_dataset(dataset)

    # Provide the number of sample size for training
    # Whole dataset it 210 entries, we split 147 (70%) for training data
    # and the remaining for testing data
    sample_size = 147
    features = 9
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

    data_train, output_train, data_test, output_test = prepare_dataset(
        dataset, sample_size, features
    )

    # Initial thetas, the reason why it initialized to 10 since, because x0
    thetas = np.array([0.02, 0.04, 0.06, 0.08, 0.10,
                       0.30, 0.50, 0.70, 0.90, 0.98])

    # Set number of iterations
    iterations = 500
    runs = 10

    plot_location = \
        os.path.join(
            os.path.dirname(__name__),
            'datasets/concrete'
        )

    # Apply the whole iterations to generate cost functions and plots
    apply_dataset_iter(runs,
                       thetas,
                       data_train,
                       output_train,
                       data_test,
                       output_test,
                       alphas,
                       iterations,
                       plot_location)


def apply_concrete_dataset(dataset):
    # Provide the number of sample size for training
    # Whole dataset it 1030 entries, we split 721 (70%) for training data
    # and the remaining for testing data
    sample_size = 721
    features = 9
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

    data_train, output_train, data_test, output_test = prepare_dataset(
        dataset, sample_size, features
    )

    # Initial thetas, the reason why it initialized to 10 since, because x0
    thetas = np.array([0.02, 0.04, 0.06, 0.08, 0.10,
                       0.30, 0.50, 0.70, 0.90, 0.98])

    # Set number of iterations
    iterations = 500
    runs = 10

    plot_location = \
        os.path.join(
            os.path.dirname(__name__),
            'datasets/hardware'
        )

    # Apply the whole iterations to generate cost functions and plots
    apply_dataset_iter(runs,
                       thetas,
                       data_train,
                       output_train,
                       data_test,
                       output_test,
                       alphas,
                       iterations,
                       plot_location)


def apply_qsar_aquatic_toxicity(dataset):
    # Provide the number of sample size for training
    # Whole dataset it 546 entries, we split 382 (70%) for training data
    # and the remaining for testing data
    sample_size = 382
    features = 8
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

    data_train, output_train, data_test, output_test = prepare_dataset(
        dataset, sample_size, features
    )

    # Initial thetas, the reason why it initialized to 9 since, because x0
    thetas = np.array([0.02, 0.04, 0.06, 0.08, 0.10,
                       0.30, 0.50, 0.70, 0.90])

    # Set number of iterations
    iterations = 500
    runs = 10

    plot_location = \
        os.path.join(
            os.path.dirname(__name__),
            'datasets/toxicity'
        )

    # Apply the whole iterations to generate cost functions and plots
    apply_dataset_iter(runs,
                       thetas,
                       data_train,
                       output_train,
                       data_test,
                       output_test,
                       alphas,
                       iterations,
                       plot_location)


if __name__ == '__main__':
    # Apply dataset for Hardware
    logger.info('Start processing Hardware dataset ....')
    hardware_data = parse_dataset('datasets/hardware/machine-data.txt')
    apply_hardware_dataset(hardware_data)
    logger.info('finish processing Hardware dataset')

    # Apply the Concrete dataset
    logger.info('Start processing Concrete dataset ....')
    concrete_data = parse_dataset('datasets/concrete/concrete-data.csv')
    apply_concrete_dataset(concrete_data)
    logger.info('finish processing Concrete dataset')

    # Apply the Toxicity dataset
    logger.info('Start processing Toxicity dataset ....')
    concrete_data = parse_dataset(
        'datasets/toxicity/qsar-aquatic-toxicity.csv'
    )
    apply_qsar_aquatic_toxicity(concrete_data)
    logger.info('finish processing Toxicity dataset')
