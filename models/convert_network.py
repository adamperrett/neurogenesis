# from models.neurogenesis import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
from scipy.spatial import distance
from tests.backprop_from_activations import create_initial_network, forward_propagate

def collect_connections_per_output(net, output, weight_norm=True):
    input_values = [0. for i in range(net.number_of_inputs)]
    input_count = [0 for i in range(net.number_of_inputs)]
    conn_values = []
    for pre in net.neurons['out{}'.format(output)].synapses:
        for inp in net.neurons[pre].synapses:
            for syn in net.neurons[pre].synapses[inp]:
                weight = net.neurons['out{}'.format(output)].synapses[pre][0].weight
                if weight > 0:
                    # find a way for this to work with hidden nodes
                    in_index = int(inp.replace('in', ''))
                    if weight_norm:
                        input_values[in_index] += syn.freq * weight
                        input_count[in_index] += weight
                    else:
                        input_values[in_index] += syn.freq
                        input_count[in_index] += 1
                conn_values.append([syn.freq, net.neurons['out{}'.format(output)].synapses[pre][0].weight])
    for idx, count in enumerate(input_count):
        if count != 0:
            input_values[idx] /= count
    return input_values#, conn_values

'''
saved value to weight:
    v=0 -> 0 - w 
        as input increases it gets further from the saved value
        
    v=1 -> 1 - w
        as input decreases it gets further from the saved value
'''

def convert_neurons_to_network(net, weight_norm=True):
    values = []
    for out in range(net.number_of_classes):
        values.append(collect_connections_per_output(net, out, weight_norm))
    return np.array(values)

def determine_2D_decision_boundary(net, x_range, y_range, resolution, data=[], labels=[], weight_norm=True):
    num_outputs = net.number_of_classes
    points = [[] for i in range(num_outputs)]
    for x in np.linspace(x_range[0], x_range[1], resolution):
        print(x, "/", x_range[1])
        for y in np.linspace(y_range[0], y_range[1], resolution):
            activations = net.convert_inputs_to_activations(np.array([x, y]))
            activations = net.response(activations)
            output_activations = np.zeros(num_outputs)
            non_zero = True
            for output in range(num_outputs):
                output_activations[output] = activations['out{}'.format(output)]
                if output_activations[output] != 0:
                    non_zero = False
            if non_zero:
                output = num_outputs
            else:
                output = output_activations.argmax()
                points[output].append([x, y])
    data_grouped = [[] for i in range(num_outputs)]
    for d, l in zip(data, labels):
        data_grouped[l].append(d)
    colours = pl.cm.plasma(np.linspace(0, 1, num_outputs))
    plt.figure()
    for i in range(num_outputs):
        if len(points[i]):
            plt.scatter(np.array(points[i])[:, 0],
                        np.array(points[i])[:, 1],
                        color=colours[i])
    for i in range(num_outputs):
        plt.scatter(np.array(data_grouped[i])[:, 0],
                        np.array(data_grouped[i])[:, 1])
    centroids = convert_neurons_to_network(net, weight_norm)
    for cent in centroids:
        plt.scatter(cent[0], cent[1], s=200)
    plt.show()
    return points


def determine_boundary_vectors(net):
    centroids = convert_neurons_to_network(net)
    print(centroids)
    midpoints = []
    vectors = []
    done = []
    for a, centroid_a in enumerate(centroids):
        for b, centroid_b in enumerate(centroids):
            if [a, b] not in done and a != b:
                done.append([a, b])
                done.append([b, a])
                midpoints.append([a, b, centroid_a, centroid_b, distance.euclidean(centroid_a, centroid_b),
                                  (np.array(centroid_a) + np.array(centroid_b)) / 2.,
                                  (centroid_b - centroid_a)])
                dot_product = np.dot((np.array(centroid_a) + np.array(centroid_b)) / 2.,
                                     (centroid_b - centroid_a))
                bias = -dot_product
                vectors.append([a, b,
                                np.hstack([centroid_b - centroid_a, bias]) / distance.euclidean(centroid_a, centroid_b)])

    # process for iteration finding boundary
    # for midpoint in midpoints:
    #     activations = net.convert_inputs_to_activations(midpoint[4])
    #     activations = net.response(activations)
    #     if activations['out{}'.format(a)] > activations['out{}'.format(b)]:
    return vectors

def create_network(net):
    neurons = determine_boundary_vectors(net)
    hidden_weights = []
    output_weights = []
    for idx, (out1, out2, weights) in enumerate(neurons):
        hidden_weights.append(weights)
        out = [0. for i in range(net.number_of_classes)]
        out[out1] = -1
        out[out2] = 1
        output_weights.append(out)
    output_weights = [[-1, -1, 0, 0],
                      [1, 0, -1, 0],
                      [0, 1, 1, 0]]
    network = create_initial_network(hidden_weights, output_weights)
    return network

def test_2D_network(net, x_range, y_range, resolution, data=[], labels=[]):
    num_outputs = len(net[-1])
    points = [[] for i in range(num_outputs)]
    on_boundary = [[] for i in range(num_outputs)]
    off_boundary = [[] for i in range(num_outputs)]
    for x in np.linspace(x_range[0], x_range[1], resolution):
        print(x, "/", x_range[1])
        for y in np.linspace(y_range[0], y_range[1], resolution):
            output_activations = forward_propagate(net, [x, y])
            non_zero = True
            for output in range(num_outputs):
                if output_activations[output] - np.average(output_activations) != 0:
                    non_zero = False
                if output_activations[output] > 0:
                    on_boundary[output].append([x, y, output_activations[output]])
                else:
                    off_boundary[output].append([x, y, output_activations[output]])
            if non_zero:
                output = num_outputs
            else:
                output = np.array(output_activations).argmax()
                points[output].append([x, y])
    data_grouped = [[] for i in range(num_outputs)]
    for d, l in zip(data, labels):
        data_grouped[l].append(d)
    colours = pl.cm.plasma(np.linspace(0, 1, num_outputs))
    plt.figure()
    for i in range(num_outputs):
        if len(points[i]):
            plt.scatter(np.array(points[i])[:, 0],
                        np.array(points[i])[:, 1],
                        color=colours[i])
    for i in range(num_outputs):
        plt.scatter(np.array(data_grouped[i])[:, 0],
                        np.array(data_grouped[i])[:, 1])
    plt.show()

def memory_to_procedural(net, x_range, y_range, resolution, data=[], labels=[]):
    network = create_network(net)
    test_2D_network(network, x_range, y_range, resolution, data, labels)
