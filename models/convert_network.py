# from models.neurogenesis import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
from scipy.spatial import distance
from math import comb
import random
from copy import deepcopy
from tests.backprop_from_activations import create_initial_network, forward_propagate

def collect_n_centroids_per_output(net, output, n, only_2D=False):
    class_and_values = []
    input_values = [[0. for i in range(net.number_of_inputs + (net.n_procedural_out * (not only_2D)))] for j in range(n)]
    centroid_count = 0
    for pre in random.sample(list(net.neurons['out{}'.format(output)].synapses),
                             len(list(net.neurons['out{}'.format(output)].synapses))):
        weight = net.neurons['out{}'.format(output)].synapses[pre][0].weight
        if weight > 0:
            for inp in net.neurons[pre].synapses:
                if 'in' in inp:
                    in_index = int(inp.replace('in', ''))
                elif 'p' in inp:
                    if only_2D:
                        continue
                    in_index = int(inp.replace('p', '')) + net.number_of_inputs
                for syn in net.neurons[pre].synapses[inp]:
                    input_values[centroid_count][in_index] += syn.freq
            class_and_values.append([output, np.array(input_values[centroid_count])])
            centroid_count += 1
            if centroid_count >= n:
                return class_and_values
    return class_and_values

def collect_centroids_per_output(net, output, weight_norm=True, only_2D=False):
    input_values = [0. for i in range(net.number_of_inputs + (net.n_procedural_out * (not only_2D)))]
    input_count = [0 for i in range(net.number_of_inputs + (net.n_procedural_out * (not only_2D)))]
    for pre in net.neurons['out{}'.format(output)].synapses:
        weight = net.neurons['out{}'.format(output)].synapses[pre][0].weight
        if weight > 0:
        # if net.neurons[pre].output == output:
            for inp in net.neurons[pre].synapses:
                if 'in' in inp or 'p' in inp:
                    for syn in net.neurons[pre].synapses[inp]:
                        # find a way for this to work with hidden nodes
                        if 'in' in inp:
                            in_index = int(inp.replace('in', ''))
                        elif 'p' in inp:
                            if only_2D:
                                continue
                            in_index = int(inp.replace('p', '')) + net.number_of_inputs
                        if weight_norm:
                            input_values[in_index] += syn.freq * weight
                            input_count[in_index] += weight
                        else:
                            input_values[in_index] += syn.freq
                            input_count[in_index] += 1
    # no_count = True
    for idx, count in enumerate(input_count):
        if count != 0:
            input_values[idx] /= count
            # no_count = False
    # if no_count:
    #     return False
    return output, np.array(input_values)



def collect_polar_centroids(net, output, weight_norm=True, only_2D=False, split=True):
    positive_input_values = [0. for i in range(net.number_of_inputs + (net.n_procedural_out * (not only_2D)))]
    positive_input_count = [0 for i in range(net.number_of_inputs + (net.n_procedural_out * (not only_2D)))]
    negative_input_values = [0. for i in range(net.number_of_inputs + (net.n_procedural_out * (not only_2D)))]
    negative_input_count = [0 for i in range(net.number_of_inputs + (net.n_procedural_out * (not only_2D)))]
    for pre in net.neurons['out{}'.format(output)].synapses:
        weight = net.neurons['out{}'.format(output)].synapses[pre][0].weight
        # if net.neurons[pre].output == output:
        for inp in net.neurons[pre].synapses:
            if 'in' in inp or 'p' in inp:
                for syn in net.neurons[pre].synapses[inp]:
                    # find a way for this to work with hidden nodes
                    if 'in' in inp:
                        in_index = int(inp.replace('in', ''))
                    elif 'p' in inp:
                        if only_2D:
                            continue
                        in_index = int(inp.replace('p', '')) + net.number_of_inputs
                    if weight_norm:
                        if weight > 0:
                            positive_input_values[in_index] += syn.freq * weight
                            positive_input_count[in_index] += weight
                        elif weight < 0:
                            negative_input_values[in_index] += syn.freq * weight
                            negative_input_count[in_index] += weight
                    else:
                        if weight > 0:
                            positive_input_values[in_index] += syn.freq * weight
                            positive_input_count[in_index] += weight
                        elif weight < 0:
                            negative_input_values[in_index] += syn.freq * weight
                            negative_input_count[in_index] += weight
    for idx, count in enumerate(positive_input_count):
        if count != 0:
            positive_input_values[idx] /= count
    for idx, count in enumerate(negative_input_count):
        if count != 0:
            negative_input_values[idx] /= count
    if split:
        splits = [net.number_of_inputs]
        for hidden in net.procedural:
            splits.append(len(hidden) + splits[-1])# + net.number_of_classes)
        split_neg = [negative_input_values[:splits[0]]]
        split_pos = [positive_input_values[:splits[0]]]
        layered_splits = [[output, np.array(split_neg[-1]), np.array(split_pos[-1]), 0]]
        for s in range(len(splits)-1):
            split_neg.append(negative_input_values[splits[s]:splits[s+1]])
            split_pos.append(positive_input_values[splits[s]:splits[s+1]])
            layered_splits.append([output, np.array(split_neg[-1]), np.array(split_pos[-1]), s+1])
        return layered_splits
    return output, np.array(negative_input_values), np.array(positive_input_values)

'''
saved value to weight:
    v=0 -> 0 - w 
        as input increases it gets further from the saved value
        
    v=1 -> 1 - w
        as input decreases it gets further from the saved value
'''

def convert_neurons_to_centroids(net, weight_norm=True, n=5, only_2D=False, polar=False):
    # return [[0, np.array([1, 0])],
    #            [0, np.array([-1, 0])],
    #            [1, np.array([0, 0])]]
    values = []
    for out in range(net.number_of_classes):
        if polar:
            # values.append(collect_polar_centroids(net, out))
            if len(values) == 0:
                values = collect_polar_centroids(net, out)
            else:
                new_centroids = collect_polar_centroids(net, out)
                if len(new_centroids):
                    values = np.vstack([values, new_centroids])
                else:
                    print("\n\n\n\n\nNo centroids for class\n\n\n\n\n")
        else:
            if n > 1:
                if len(values) == 0:
                    values = collect_n_centroids_per_output(net, out, n)
                else:
                    new_centroids = collect_n_centroids_per_output(net, out, n)
                    if len(new_centroids):
                        values = np.vstack([values, new_centroids])
                    else:
                        print("\n\n\n\n\nNo centroids for class\n\n\n\n\n")
            else:
                values.append(collect_centroids_per_output(net, out, weight_norm, only_2D))
    return values

def collect_all_stored_values(net):
    all_values = []
    for pre in net.neurons['out0'].synapses:
        collected = False
        input_values = [0. for i in range(net.number_of_inputs)]
        for inp in net.neurons[pre].synapses:
            for syn in net.neurons[pre].synapses[inp]:
                if 'in' in inp:
                    collected = True
                    in_index = int(inp.replace('in', ''))
                    input_values[in_index] += syn.freq
        if collected:
            all_values.append(input_values)
    return all_values

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
    all_stored = collect_all_stored_values(net)
    if len(all_stored):
        plt.scatter(np.array(all_stored)[:, 0],
                        np.array(all_stored)[:, 1], marker='*')
    centroids = convert_neurons_to_centroids(net, weight_norm, n=1)
    # if centroids:
    for cl, cent in centroids:
        if cent[0] != 0 or cent[1] != 0:
            plt.scatter(cent[0], cent[1], s=200)
    plt.xlim(x_range)
    plt.ylim(y_range)
    plt.show()
    return points


def determine_boundary_vectors(net, only_2D=False, polar=False):
    midpoints = []
    vectors = []
    done = []
    if polar:
        centroids = convert_neurons_to_centroids(net, only_2D=only_2D, polar=polar)
        print(centroids)
        pos_cent = []
        neg_cent = []
        for out, c_a, c_b, layer in centroids:
            neg_cent.append([out, c_a])
            pos_cent.append([out, c_b])
            done.append('{}'.format([c_a, c_b]))
            done.append('{}'.format([c_b, c_a]))
            midpoints.append([out, out, c_a, c_b, distance.euclidean(c_a, c_b),
                              (np.array(c_a) + np.array(c_b)) / 2.,
                              (c_b - c_a)])
            dot_product = np.dot((np.array(c_a) + np.array(c_b)) / 2.,
                                 (c_b - c_a))
            bias = -dot_product
            # square the distance to make it as broad as the centroids are apart
            vectors.append([out, out,
                            # np.hstack([c_b - c_a, bias]) / np.power(distance.euclidean(c_a, c_b), 2), layer])
                            np.hstack([c_b - c_a, bias]) / distance.euclidean(c_a, c_b), layer])
        # for a, c_a in pos_cent:
        #     for b, c_b in pos_cent:
        #         if a != b and '{}'.format([c_a, c_b]) not in done and c_a.all() != c_b.all():
        #             done.append('{}'.format([c_a, c_b]))
        #             done.append('{}'.format([c_b, c_a]))
        #             midpoints.append([a, b, c_a, c_b, distance.euclidean(c_a, c_b),
        #                               (np.array(c_a) + np.array(c_b)) / 2.,
        #                               (c_b - c_a)])
        #             dot_product = np.dot((np.array(c_a) + np.array(c_b)) / 2.,
        #                                  (c_b - c_a))
        #             bias = -dot_product
        #             vectors.append([a, b,
        #                             np.hstack([c_b - c_a, bias]) / distance.euclidean(c_a, c_b)])
        # for a, c_a in neg_cent:
        #     for b, c_b in neg_cent:
        #         if a != b and '{}'.format([c_a, c_b]) not in done and c_a.all() != c_b.all():
        #             done.append('{}'.format([c_a, c_b]))
        #             done.append('{}'.format([c_b, c_a]))
        #             midpoints.append([a, b, c_a, c_b, distance.euclidean(c_a, c_b),
        #                               (np.array(c_a) + np.array(c_b)) / 2.,
        #                               (c_b - c_a)])
        #             dot_product = np.dot((np.array(c_a) + np.array(c_b)) / 2.,
        #                                  (c_b - c_a))
        #             bias = -dot_product
        #             vectors.append([b, a,
        #                             np.hstack([c_b - c_a, bias]) / distance.euclidean(c_a, c_b)])
        # for a, c_a in pos_cent:
        #     for b, c_b in neg_cent:
        #         if a != b and '{}'.format([c_a, c_b]) not in done and c_a.all() != c_b.all():
        #             done.append('{}'.format([c_a, c_b]))
        #             done.append('{}'.format([c_b, c_a]))
        #             midpoints.append([a, b, c_a, c_b, distance.euclidean(c_a, c_b),
        #                               (np.array(c_a) + np.array(c_b)) / 2.,
        #                               (c_b - c_a)])
        #             dot_product = np.dot((np.array(c_a) + np.array(c_b)) / 2.,
        #                                  (c_b - c_a))
        #             bias = -dot_product
        #             vectors.append([b, a,
        #                             np.hstack([c_b - c_a, bias]) / distance.euclidean(c_a, c_b)])
    else:
        centroids = convert_neurons_to_centroids(net, only_2D=only_2D)
        print(centroids)
        for a, c_a in centroids:
            for b, c_b in centroids:
                if a != b and '{}'.format([c_a, c_b]) not in done:
                    done.append('{}'.format([c_a, c_b]))
                    done.append('{}'.format([c_b, c_a]))
                    midpoints.append([a, b, c_a, c_b, distance.euclidean(c_a, c_b),
                                      (np.array(c_a) + np.array(c_b)) / 2.,
                                      (c_b - c_a)])
                    dot_product = np.dot((np.array(c_a) + np.array(c_b)) / 2.,
                                         (c_b - c_a))
                    bias = -dot_product
                    vectors.append([a, b,
                                    np.hstack([c_b - c_a, bias]) / distance.euclidean(c_a, c_b)])

    # process for iteration finding boundary
    # for midpoint in midpoints:
    #     activations = net.convert_inputs_to_activations(midpoint[4])
    #     activations = net.response(activations)
    #     if activations['out{}'.format(a)] > activations['out{}'.format(b)]:
    return vectors

def build_network(net, only_2D=False, polar=False):
    neurons = determine_boundary_vectors(net, only_2D=only_2D, polar=polar)
    n_out = net.number_of_classes
    v_out = len(neurons)
    old_net = deepcopy(net.procedural)
    if len(old_net) == 0:
        old_net.append([])
        old_net.append([{'weights': [0.]} for i in range(n_out)])
    else:
        old_net.append([{'weights': [int(i == j) for j in range(n_out+1)]} for i in range(n_out)])
        old_net.append([{'weights': [int(i == j) for j in range(n_out+1)]} for i in range(n_out)])
    layer_widths = []
    for layer in old_net:
        layer_widths.append(len(layer))

    for idx, (out1, out2, weights, layer) in enumerate(neurons):
        # enlarge added neurons length
        extended_weights = [w for w in weights]
        if layer >= 1:
            for i in range(n_out):
                extended_weights[-1:-1] = [0]
        old_net[layer].append({'weights': extended_weights})
        # enlarge the next layers connections
        for output in range(layer_widths[layer+1]):
            if output < n_out and output == out1:
                old_net[layer+1][output]['weights'][-1:-1] = [1.]
                # old_net[layer+1][output]['weights'].append(0.)
            else:
                old_net[layer+1][output]['weights'][-1:-1] = [0.]
                # old_net[layer+1][output]['weights'].append(0.)
    return old_net

def create_network(net, only_2D=False, polar=False):
    neurons = determine_boundary_vectors(net, only_2D=only_2D, polar=polar)
    n_out = net.number_of_classes
    v_out = len(neurons)
    hidden_weights = []
    output_weights = [[0. for i in range(v_out+1)] for j in range(n_out)]
    for idx, (out1, out2, weights) in enumerate(neurons):
        for w in weights:
            if w != w:
                print("NaN found")
                neurons = determine_boundary_vectors(net, only_2D=only_2D, polar=polar)
        hidden_weights.append(weights)
        if out1 == out2:
            output_weights[out1][idx] = 1
        else:
            output_weights[out1][idx] = -1
            output_weights[out2][idx] = 1
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
            output_activations, _ = forward_propagate(net, [x, y])
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
    plt.xlim(x_range)
    plt.ylim(y_range)
    plt.show()

def memory_to_procedural(net, x_range, y_range, resolution, data=[], labels=[]):
    network = create_network(net, only_2D=True)
    test_2D_network(network, x_range, y_range, resolution, data, labels)
