from math import exp
from random import seed
from random import random
import numpy as np
from models.convert_network import *
from datasets.simple_tests import *


# Initialize a network
def random_initialize_network(n_inputs, n_hidden, n_outputs, layers=1):
    network = list()
    hidden_layer = [{'weights':[np.random.randn()/100. for i in range(n_inputs + 1)]} for i in range(n_hidden)]
    network.append(hidden_layer)
    for l in range(layers-1):
        hidden_layer = [{'weights': [np.random.randn() / 100. for i in range(n_hidden + 1)]} for i in range(n_hidden)]
        network.append(hidden_layer)
    output_layer = [{'weights':[np.random.randn()/100. for i in range(n_hidden + 1)]} for i in range(n_outputs)]
    network.append(output_layer)
    return network

# Initialize a network
def create_initial_network(hidden_weights, output_weights, multi_layered=False):
    network = list()
    if multi_layered:
        for h_w in hidden_weights:
            hidden_layer = [{'weights': [w for w in weights]} for weights in h_w]
            network.append(hidden_layer)
    else:
        hidden_layer = [{'weights': [w for w in weights]} for weights in hidden_weights]
        network.append(hidden_layer)
    output_layer = [{'weights': [w for w in weights]} for weights in output_weights]
    network.append(output_layer)
    return network

# Calculate neuron activation for an input
def activate(weights, inputs):
    activation = weights[-1]
    for i in range(len(weights)-1):
        activation += weights[i] * inputs[i]
    return activation

# Transfer neuron activation
def transfer(activation):
    return np.tanh(activation)
    # return 1.0 / (1.0 + exp(-activation))

# Forward propagate input to a network output
def forward_propagate(network, row):
    all_inputs = []
    inputs = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = activate(neuron['weights'], inputs)
            neuron['output'] = transfer(activation)
            new_inputs.append(neuron['output'])
        all_inputs = np.hstack([all_inputs, new_inputs])
        inputs = new_inputs
        # break
    return inputs, all_inputs

# Calculate the derivative of an neuron output
def transfer_derivative(output):
    return 1. - (np.tanh(output)**2)
    # return output * (1.0 - output)

# Backpropagate error and store in neurons
def backward_propagate_error(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        if i != len(network)-1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i + 1]:
                    error += (neuron['weights'][j] * neuron['delta'])
                errors.append(error)
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(expected[j] - neuron['output'])
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])

# Update network weights with error
def update_weights(network, row, l_rate):
    for i in range(len(network)):
        inputs = row[:-1]
        if i != 0:
            inputs = [neuron['output'] for neuron in network[i - 1]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
            neuron['weights'][-1] += l_rate * neuron['delta']

# Train a network for a fixed number of epochs
def train_network(network, train, l_rate, n_epoch, n_outputs):
    for epoch in range(n_epoch):
        sum_error = 0
        for row in train:
            outputs = forward_propagate(network, row)
            expected = [0 for i in range(n_outputs)]
            expected[int(row[-1])] = 1
            sum_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
            backward_propagate_error(network, expected)
            update_weights(network, row, l_rate)
        print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))

if __name__ == '__main__':
    # Test training backprop algorithm
    seed(1)
    # base_file_name = 'kfold-strat save act noshuff allin out0.0 RL0.99999  - ' \
    #                  'wine fixed_h0 - sw0.4n0.0 - at0.0 - et0.0 - 1.0adr1.0 - 0.0noise 0'
    # base_file_name = 'noOut no-lr0.1 out0.0 RL0.99999  - ' \
    #                  'wine fixed_h0 - sw0.4n0.0 - at0.0 - et0.0 - 1.0adr1.0 - 0.0noise 4.png'
    # base_file_name = 'simple-net150 sm0.0 RL0.99999  - ' \
    #                  'simple fixed_h0 - sw0.6n0.0 - at0.0 - et0.0003 - 1.0adr1.0 - 0.0noise 0.png'
    # dataset = np.load('./data/'+base_file_name+'.npy', allow_pickle=True).item()
    #
    # determine_2D_decision_boundary(dataset['net'], [-1, 2], [-1, 2], 100)
    # conn = np.array(convert_neurons_to_network(dataset['net']))
    centres = [[1, 0],
               [0, 0],
               [-1, 0]]
    # [0, 1]]
    spread = 0.3
    examples = 100
    test_set_size = 0.1
    num_outputs = 2
    # simple_data, simple_labels = create_centroid_classes(centres, spread, examples)
    simple_data, simple_labels = create_bimodal_distribution(centres, spread, examples,
                                                             max_classes=num_outputs)
    dataset = []
    for data, label in zip(simple_data, simple_labels):
        dataset.append(np.hstack([data, label]))
    n_inputs = len(simple_data[0])
    n_hidden = 4
    n_outputs = num_outputs
    network = random_initialize_network(n_inputs, n_hidden, n_outputs, layers=2)
    train_network(network, dataset, 0.1, 200, n_outputs)
    test_2D_network(network, [-2, 2], [-2, 2], 100, simple_data, simple_labels)
    for layer in network:
        print(layer)