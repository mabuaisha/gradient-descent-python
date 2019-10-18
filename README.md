# Applying Gradient Descent Using Python

This Assigment uses the gradient decent algorithm for optimizing the linear regression


## Requirements

In order to run this project  you need to install the following tools:

1. Python 3.7

2. [pip3](https://pip.pypa.io/en/stable/installing/) 

3. [virtualenv](https://virtualenv.pypa.io/en/latest/installation/)


## Installation

In order to run the application run the following commands:

1. Create virtualenv `virtualenv YOUR_ENV`. Example: `virtualenv decent-assigment`

2. Activate the created virtualenv `source /path/to/decent-assigment/bin/activate`

3. `pip3 install -r requirements.txt`


## Running Application

In order to run the application, execute `python3 main.py`

The above script pull the 3 datasets from the following directories:

1. `datasets/concrete`
2. `datasets/hardware`
3. `datasets/toxicity`

When run the application, for each dataset there will be 20 images generated as the following: 

`testing_data_run_1.png`
`testing_data_run_2.png`
...
`training_data_run_1.png`
`training_data_run_2.png`

In order to check the generated images, check each dataset folder to get the results of cost function against iterations

