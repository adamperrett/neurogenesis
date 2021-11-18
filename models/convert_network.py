# from models.neurogenesis import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
from scipy.spatial import distance
from math import comb
import random
import operator
from copy import deepcopy
from tests.backprop_from_activations import *
import seaborn  as sns
from scipy.special import softmax as sm

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

def collect_most_accurate_neurons(net, n=5, only_2D=False):
    neuron_accuracy = [{} for i in range(net.number_of_classes)]
    for neuron in net.neurons:
        if 'in' in neuron or 'out' in neuron:
            continue
        input_vals = net.neurons[neuron].connections
        activations = net.response(deepcopy(input_vals))
        accuracy = [0. for i in range(net.number_of_classes)]
        for out in range(net.number_of_classes):
            accuracy[out] += activations['out{}'.format(out)]
        sm_accuracy = sm(accuracy)
        neuron_accuracy[net.neurons[neuron].output][neuron] = sm_accuracy[net.neurons[neuron].output] * 100.
        # neuron_accuracy[net.neurons[neuron].output][neuron] = accuracy[net.neurons[neuron].output] * 100.

    sorted_accuracy = []
    selected_neurons = []
    centroids = []
    best_neurons = [{} for i in range(net.number_of_classes)]
    for output, accuracy in enumerate(neuron_accuracy):
        if not len(accuracy):
            continue
        sorted_accuracy.append(dict(sorted(accuracy.items(),
                                           key=operator.itemgetter(1),
                                           reverse=True)))
        while len(best_neurons[output]) < min(n, len(accuracy)):#len(accuracy) / 2:
            for neuron in sorted_accuracy[-1]:
                if neuron not in best_neurons[output]:
                    break
            input_vals = net.neurons[neuron].connections
            activations = net.response(deepcopy(input_vals))
            for neur in sorted_accuracy[-1]:
                sorted_accuracy[-1][neur] *= 1 - activations[neur]
            sorted_accuracy.append(dict(sorted(sorted_accuracy[-1].items(),
                                               key=operator.itemgetter(1),
                                               reverse=True)))
            vals = [0. for i in range(net.number_of_inputs + (net.n_procedural_out * (not only_2D)))]
            for inp in input_vals:
                if 'in' in inp:
                    in_index = int(inp.replace('in', ''))
                elif 'p' in inp:
                    if only_2D:
                        continue
                    in_index = int(inp.replace('p', '')) + net.number_of_inputs
                vals[in_index] = input_vals[inp]
            best_neurons[output][neuron] = vals
            if len(best_neurons[output]) <= n:
                centroids.append([output, np.array(vals)])
            if len(best_neurons[output]) == n:
                selected_neurons.append(best_neurons[output])
    return centroids

def collect_n_correct_neurons(net, n=5, only_2D=False):
    sorted_dics = []
    selected_neurons = []
    centroids = []
    for output, correctness in enumerate(net.correctness):
        if len(correctness) == 0:
            continue
        sorted_dics.append(dict(sorted(correctness.items(),
                                       key=operator.itemgetter(1),
                                       reverse=True)))
        best_neurons = {}
        for neuron in sorted_dics[-1]:
            input_vals = {}
            for syn in net.neurons[neuron].synapses:
                for s in net.neurons[neuron].synapses[syn]:
                    input_vals[syn] = s.freq
            vals = [0. for i in range(net.number_of_inputs + (net.n_procedural_out * (not only_2D)))]
            for inp in input_vals:
                if 'in' in inp:
                    in_index = int(inp.replace('in', ''))
                elif 'p' in inp:
                    if only_2D:
                        continue
                    in_index = int(inp.replace('p', '')) + net.number_of_inputs
                vals[in_index] = input_vals[inp]
            best_neurons[neuron] = vals
            if len(best_neurons) <= n:
                centroids.append([output, np.array(vals)])
            if len(best_neurons) == n:
                selected_neurons.append(best_neurons)
                # break
    return centroids


def collect_polar_centroids(net, output, weight_norm=True, only_2D=False, split=False):
    n_positive_input_values = [0. for i in range(net.number_of_inputs + (net.n_procedural_out * (not only_2D)))]
    n_positive_input_count = [0 for i in range(net.number_of_inputs + (net.n_procedural_out * (not only_2D)))]
    n_negative_input_values = [0. for i in range(net.number_of_inputs + (net.n_procedural_out * (not only_2D)))]
    n_negative_input_count = [0 for i in range(net.number_of_inputs + (net.n_procedural_out * (not only_2D)))]
    w_positive_input_values = [0. for i in range(net.number_of_inputs + (net.n_procedural_out * (not only_2D)))]
    w_positive_input_count = [0 for i in range(net.number_of_inputs + (net.n_procedural_out * (not only_2D)))]
    w_negative_input_values = [0. for i in range(net.number_of_inputs + (net.n_procedural_out * (not only_2D)))]
    w_negative_input_count = [0 for i in range(net.number_of_inputs + (net.n_procedural_out * (not only_2D)))]
    weight_values = []
    for pre in net.neurons['out{}'.format(output)].synapses:
        weight = net.neurons['out{}'.format(output)].synapses[pre][0].weight
        weight_values.append(weight)
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
                    # if weight_norm:
                    if weight > 0:
                        w_positive_input_values[in_index] += syn.freq * (1 - weight)
                        w_positive_input_count[in_index] += (1 - weight)
                    elif weight < 0:
                        w_negative_input_values[in_index] += syn.freq * (1 + weight)
                        w_negative_input_count[in_index] += (1 + weight)
                    # else:
                    if weight > 0:
                        n_positive_input_values[in_index] += syn.freq #* weight
                        n_positive_input_count[in_index] += 1#weight
                    elif weight < 0:
                        n_negative_input_values[in_index] += syn.freq #* weight
                        n_negative_input_count[in_index] += 1#weight
    for idx, count in enumerate(w_positive_input_count):
        if count != 0:
            w_positive_input_values[idx] /= count
    for idx, count in enumerate(w_negative_input_count):
        if count != 0:
            w_negative_input_values[idx] /= count
    for idx, count in enumerate(n_positive_input_count):
        if count != 0:
            n_positive_input_values[idx] /= count
    for idx, count in enumerate(n_negative_input_count):
        if count != 0:
            n_negative_input_values[idx] /= count
    diff_pos_values = []
    diff_neg_values = []
    for val in range(len(n_positive_input_values)):
        diff_pos_values.append(n_positive_input_values[val] - w_positive_input_values[val])
    for val in range(len(n_negative_input_values)):
        diff_neg_values.append(n_negative_input_values[val] - w_negative_input_values[val])
    if split:
        splits = [net.number_of_inputs]
        for hidden in net.procedural:
            splits.append(len(hidden) + splits[-1])# + net.number_of_classes)
        split_neg = [w_negative_input_values[:splits[0]]]
        split_pos = [w_positive_input_values[:splits[0]]]
        layered_splits = [[output, np.array(split_neg[-1]), np.array(split_pos[-1]), 0]]
        for s in range(len(splits)-1):
            split_neg.append(w_negative_input_values[splits[s]:splits[s+1]])
            split_pos.append(w_positive_input_values[splits[s]:splits[s+1]])
            layered_splits.append([output, np.array(split_neg[-1]), np.array(split_pos[-1]), s+1])
        return layered_splits
    return output, np.array(n_negative_input_values), np.array(n_positive_input_values)
    # return output, np.array(w_negative_input_values), np.array(w_positive_input_values)

'''
saved value to weight:
    v=0 -> 0 - w 
        as input increases it gets further from the saved value
        
    v=1 -> 1 - w
        as input decreases it gets further from the saved value
'''

def convert_neurons_to_centroids(net, weight_norm=True, n=5, only_2D=False, polar=False, correctness=True):
    # return [[0, np.array([1, 0])],
    #            [0, np.array([-1, 0])],
    #            [1, np.array([0, 0])]]
    if correctness:
        # centroids = collect_n_correct_neurons(net, n)
        centroids = collect_most_accurate_neurons(net, n)
        return centroids
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
    heat_map = [[[0. for i in range(resolution)] for j in range(resolution)] for o in range(num_outputs+1)]
    new_coords = [[] for i in range(num_outputs)]
    for i, x in enumerate(np.linspace(x_range[0], x_range[1], resolution)):
        print(x, "/", x_range[1])
        for j, y in enumerate(reversed(np.linspace(y_range[0], y_range[1], resolution))):
            activations = net.convert_inputs_to_activations(np.array([x, y]))
            activations = net.response(activations)
            output_activations = np.zeros(num_outputs)
            non_zero = True
            for output in range(num_outputs):
                output_activations[output] = activations['out{}'.format(output)]
                heat_map[output][j][i] = output_activations[output]
                if output_activations[output] != 0:
                    non_zero = False
            heat_map[-1][j][i] = heat_map[1][j][i] - heat_map[0][j][i]
            if non_zero:
                output = num_outputs
            else:
                output = output_activations.argmax()
                points[output].append([x, y])
    data_grouped = [[] for i in range(num_outputs)]
    for d, l in zip(data, labels):
        data_grouped[l].append(d)
        activations = net.convert_inputs_to_activations(d)
        activations = net.response(activations)
        new_coords[l].append([activations['out0'], activations['out1']])
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
    centroids = convert_neurons_to_centroids(net, weight_norm, n=5, correctness=True)
    # if centroids:
    for cl, cent in centroids:
        if cent[0] != 0 or cent[1] != 0:
            if cl % 2:
                plt.scatter(cent[0], cent[1], marker='+', s=800)
            else:
                plt.scatter(cent[0], cent[1], marker='x', s=400)
    plt.xlim(x_range)
    plt.ylim(y_range)
    plt.show()
    fig, ax = plt.subplots(2, 2)
    im1 = ax[0][0].imshow(heat_map[0], cmap='viridis', extent=x_range+y_range)#[x_range[0], x_range[1], y_range[0], y_range[1]])
    im2 = ax[0][1].imshow(heat_map[1], cmap='viridis', extent=x_range+y_range)#[x_range[0], x_range[1], y_range[0], y_range[1]])
    im3 = ax[1][0].imshow(heat_map[2], cmap='viridis', extent=x_range+y_range)#[x_range[0], x_range[1], y_range[0], y_range[1]])
    for i in range(num_outputs):
        ax[1][1].scatter(np.array(new_coords[i])[:, 0],
                        np.array(new_coords[i])[:, 1])
        cent = [np.average(np.array(new_coords[i])[:, 0]), np.average(np.array(new_coords[i])[:, 1])]
        ax[1][1].scatter(cent[0], cent[1], s=200)
    cbar1 = fig.colorbar(im1, ax=ax[0][0])
    cbar2 = fig.colorbar(im2, ax=ax[0][1])
    cbar3 = fig.colorbar(im3, ax=ax[1][0])
    fig.tight_layout()
    plt.show()
    # ax = sns.heatmap(heat_map[0], linewidth=0.5)
    # plt.show()
    # ax2 = sns.heatmap(heat_map[1], linewidth=0.5)
    # plt.show()
    # plt.imshow(heat_map[0], cmap='hot', interpolation='nearest')
    # plt.show()
    # plt.imshow(heat_map[1], cmap='hot', interpolation='nearest')
    # fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    # a1 = sns.heatmap(heat_map[0], cmap="YlGnBu", ax=ax1, yticklabels=np.linspace(y_range[0], y_range[1], 10))
    # a2 = sns.heatmap(heat_map[1], cmap="YlGnBu", ax=ax2, yticklabels=np.linspace(y_range[0], y_range[1], 10))
    # a3 = sns.heatmap(heat_map[2], cmap="YlGnBu", ax=ax3, yticklabels=np.linspace(y_range[0], y_range[1], 10))
    # plt.show()
    # fig, (ax2) = plt.subplots(1, 1)
    # ax1 = sns.heatmap(heat_map[0], cmap="YlGnBu")
    # ax2 = sns.heatmap(heat_map[1], cmap="YlGnBu")
    # plt.show()
    return points


def determine_boundary_vectors(net, only_2D=False, polar=False, correctness=False):
    midpoints = []
    vectors = []
    done = []
    if polar and not correctness:
        centroids = convert_neurons_to_centroids(net, only_2D=only_2D, polar=polar, correctness=correctness)
        print(centroids)
        pos_cent = []
        neg_cent = []
        # for out, c_a, c_b, layer in centroids:
        for out, c_a, c_b in centroids:
            neg_cent.append([out, c_a])
            pos_cent.append([out, c_b])
            if c_a.tolist() == c_b.tolist():
                print("This shouldn't happen")
                continue
            if '{}'.format([c_a, c_b]) in done:
                continue
            done.append('{}'.format([c_a, c_b]))
            done.append('{}'.format([c_b, c_a]))
            dot_product = np.dot((np.array(c_a) + np.array(c_b)) / 2.,
                                 (c_b - c_a))

            n = net.number_of_classes
            new_vector = c_b - c_a
            zero_ind = np.argpartition(np.abs(new_vector[:-1]), -n)[:-n]
            new_vector[zero_ind] = 0.
            new_dot = np.dot((np.array(c_a) + np.array(c_b)) / 2., new_vector)
            midpoints.append([out, out, c_a, c_b, distance.euclidean(c_a, c_b),
                              (np.array(c_a) + np.array(c_b)) / 2.,
                              (c_b - c_a), dot_product, new_vector, new_dot])
            bias = -dot_product
            # square the distance to make it as broad as the centroids are apart
            vectors.append([out, out,
                            # np.hstack([c_b - c_a, bias]) / np.power(distance.euclidean(c_a, c_b), 2), layer])
                            # np.hstack([c_b - c_a, bias]) / distance.euclidean(c_a, c_b), layer])
                            np.hstack([c_b - c_a, bias]) / np.power(distance.euclidean(c_a, c_b), 1)])
                            # np.hstack([new_vector, -new_dot]) / np.power(distance.euclidean(c_a, c_b), 1)])
        for a, c_a in pos_cent:
            for b, c_b in pos_cent:
                if a != b and '{}'.format([c_a, c_b]) not in done and c_a.tolist() != c_b.tolist():
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
        centroids = convert_neurons_to_centroids(net, only_2D=only_2D, correctness=correctness)
        print(centroids)
        for a, c_a in centroids:
            for b, c_b in centroids:
                if a != b and '{}'.format([c_a, c_b]) not in done:
                    done.append('{}'.format([c_a, c_b]))
                    done.append('{}'.format([c_b, c_a]))
                    dot_product = np.dot((np.array(c_a) + np.array(c_b)) / 2.,
                                         (c_b - c_a))
                    midpoints.append([a, b, c_a, c_b, distance.euclidean(c_a, c_b),
                                      (np.array(c_a) + np.array(c_b)) / 2.,
                                      (c_b - c_a), dot_product])
                    bias = -dot_product
                    vectors.append([a, b,
                                    (np.hstack([c_b - c_a, bias]) / distance.euclidean(c_a, c_b)) / 1])

    # process for iteration finding boundary
    # for midpoint in midpoints:
    #     activations = net.convert_inputs_to_activations(midpoint[4])
    #     activations = net.response(activations)
    #     if activations['out{}'.format(a)] > activations['out{}'.format(b)]:
    print("vectors:\n", vectors)
    return vectors, centroids

def build_network(net, only_2D=False, polar=False, max_layers=6):
    neurons = determine_boundary_vectors(net, only_2D=only_2D, polar=polar)
    n_out = net.number_of_classes
    v_out = len(neurons)
    old_net = deepcopy(net.procedural)
    if len(old_net) <= max_layers:
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
        for w in weights:
            if w != w:
                print("oh no")
        if layer >= max_layers:
            continue
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

def create_network(net, only_2D=False, polar=False, correctness=False):
    neurons, centroids = determine_boundary_vectors(net, only_2D=only_2D, polar=polar, correctness=correctness)
    n_out = net.number_of_classes
    v_out = len(neurons)
    hidden_weights = []
    output_weights = [[0. for i in range(v_out+1)] for j in range(n_out)]
    for idx, (out1, out2, weights) in enumerate(neurons):
        for w in weights:
            if w != w:
                print("NaN found")
                neurons, centroids = determine_boundary_vectors(net, only_2D=only_2D, polar=polar, correctness=correctness)
        hidden_weights.append(weights)
        if out1 == out2:
            output_weights[out1][idx] = 1
        else:
            output_weights[out1][idx] = -1
            output_weights[out2][idx] = 1
    # network = create_initial_network(hidden_weights, output_weights)
    network = create_matrix_network(hidden_weights, output_weights)
    return network, centroids

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
