import numpy as np
import random
import operator


class Synapses():
    def __init__(self, pre, post, freq, f_width=0.3, weight=1., maturation=1., sensitivity=0.):
        self.pre = pre
        self.post = post
        self.freq = freq
        self.f_width = f_width
        self.weight = weight
        self.maturation = maturation
        self.age = 0.
        self.age_multiplier = 1. #/ self.maturation
        self.sensitivity = sensitivity
        # self.input_spread = input_spread

    def contribution(self):
        return self.sensitivity * self.freq

    def age_weight(self):
        if self.age == self.maturation:
            return
        self.age += 1.
        self.age = min(self.age, self.maturation)
        self.age_multiplier = 1. + self.age#(self.age / self.maturation)

    def response(self, input):
        return self.weight * max(1. - abs((input - self.freq) / self.f_width), 0) #* self.age_multiplier



class Neuron():
    def __init__(self, neuron_label, connections, sensitivities, weights=None, f_width=0.3,
                 input_dimensions=None,
                 input_spread=3):
        self.neuron_label = neuron_label
        self.f_width = f_width
        self.synapses = {}
        self.synapse_count = len(connections)
        self.input_dimensions = input_dimensions
        self.input_spread = input_spread
        self.visualisation = None
        if not weights:
            weights = {}
        for pre in connections:
            # if pre not in weights:
            #     weights[pre] = 1.
            freq = connections[pre]
            self.add_connection(pre, freq, sensitivities=sensitivities, weight=1.)
            # self.synapses[pre] = []
            # self.synapses[pre].append(Synapses(pre + '0', neuron_label, freq,
            #                                    f_width=self.f_width,
            #                                    weight=weights[pre]))

    def add_connection(self, pre, freq, sensitivities, weight=1., maturation=1.):
        self.synapse_count += 1
        if pre not in self.synapses:
            self.synapses[pre] = []
        if 'out' in self.neuron_label:
            sensitivities[pre] = 1.
        self.synapses[pre].append(Synapses(pre + '{}'.format(len(self.synapses[pre])),
                                           self.neuron_label, freq,
                                           weight=weight,
                                           f_width=self.f_width,
                                           maturation=maturation,
                                           sensitivity=sensitivities[pre]))

    def response(self, activations):
        if not self.synapse_count:
            return 0.
        response = 0.
        active_synapse_weight = 0
        for pre in self.synapses:  # do as an intersection of sets? inter = activations.keys() & self.synapses.keys()
            if pre in activations:
                freq = activations[pre]
                for synapse in self.synapses[pre]:
                    # if 'in' in pre:
                    #     response += self.spread_input(pre, activations, synapse)
                    # else:
                    response += synapse.response(freq)
                    active_synapse_weight += 1.#synapse.weight
        if active_synapse_weight:
            return response / active_synapse_weight
        else:
            return response

    def spread_input(self, pre, activations, synapse):
        if not self.input_dimensions:
            return synapse.response(activations[pre])
        input_idx = int(pre.replace('in', ''))
        in_x = input_idx % self.input_dimensions[1]
        in_y = (input_idx - in_x) / self.input_dimensions[1]
        combined_response = 0.
        response_count = 0
        for x in range(-self.input_spread, self.input_spread + 1):
            for y in range(-self.input_spread, self.input_spread + 1):
                new_x = in_x + x
                new_y = in_y + y
                if new_x >= self.input_dimensions[0] or new_x < 0 \
                        or new_y >= self.input_dimensions[1] or new_y < 0:
                    continue
                distance = np.sqrt(np.power(x, 2) + np.power(y, 2))
                weight = max(self.input_spread + 1 - distance, 0)
                if weight == 0:
                    continue
                weight /= self.input_spread + 1
                new_idx = int(new_x + (new_y * self.input_dimensions[0]))
                spread_response = activations['in{}'.format(new_idx)]
                combined_response += synapse.response(spread_response) #* weight
                response_count += 1
        combined_response /= response_count
        return combined_response


class Network():
    def __init__(self, number_of_classes, seed_class, seed_features, seeds,
                 error_threshold=0.1,
                 f_width=0.3,
                 activation_threshold=0.01,
                 maximum_net_size=20,
                 max_hidden_synapses=100,
                 activity_decay_rate=0.9,
                 always_inputs=True,
                 old_weight_modifier=1.,
                 input_dimensions=None,
                 input_spread=3,
                 output_synapse_maturity=1.,
                 fixed_hidden_ratio=0.1,
                 activity_init=1.0):
        self.error_threshold = error_threshold
        self.f_width = f_width
        self.activation_threshold = activation_threshold
        self.hidden_neuron_count = 0
        self.deleted_neuron_count = 0
        self.maximum_net_size = maximum_net_size
        self.max_hidden_synapses = max_hidden_synapses
        self.always_inputs = always_inputs
        self.old_weight_modifier = old_weight_modifier
        self.current_importance = 1.
        self.input_dimensions = input_dimensions
        self.input_spread = input_spread
        self.output_synapse_maturity = output_synapse_maturity
        self.fixed_hidden_ratio = fixed_hidden_ratio
        self.activity_init = activity_init

        self.neurons = {}
        self.neuron_activity = {}
        self.neuron_selectivity = {}
        self.neuron_connectedness = {}
        self.activity_decay_rate = activity_decay_rate
        self.number_of_classes = number_of_classes
        # add seed neuron
        # self.neurons['seed{}'.format(seed_class)] = Neuron('seed{}'.format(seed_class),
        # self.neurons['n0'] = Neuron('n0',
        #                             self.convert_inputs_to_activations(seed_features),
        #                             f_width=f_width)
        self.number_of_inputs = len(seed_features[0])
        for i in range(self.number_of_inputs):
            self.neuron_activity['in{}'.format(i)] = 0.
        # add outputs
        for output in range(number_of_classes):
            self.add_neuron({}, 'out{}'.format(output))
        # connect seed neuron to seed class
        self.add_seed(seeds, seed_class, seed_features)
        # self.neurons['out{}'.format(seed_class)].add_connection('seed{}'.format(seed_class),
        self.layers = 2

    def add_seed(self, seeds, seed_classes, seed_features):
        input_response = {}
        seed_count = 0
        for seed in seeds:
            converted_inputs = self.convert_inputs_to_activations(seed_features[seed])
            for feat in converted_inputs:
                if feat not in input_response:
                    input_response[feat] = 0.
                input_response[feat] += converted_inputs[feat]
            features = seed_features[seed]
            seed_class = seed_classes[seed]
            neuron_label = self.add_neuron(converted_inputs, seeding=True)
            self.neurons['out{}'.format(seed_class)].add_connection(neuron_label, freq=1.)
            self.neuron_connectedness[neuron_label] = 1
            seed_count += 1
            print("Added seed", seed_count, "/", len(seeds))
        for inp in input_response:
            input_response[inp] /= seed_count
            self.neuron_activity[inp] = input_response[inp]
        print("Completed adding seeds")


    def add_neuron(self, connections, neuron_label='', seeding=False):
        if self.hidden_neuron_count - self.deleted_neuron_count == self.maximum_net_size:
            self.delete_neuron()
        if self.max_hidden_synapses and not seeding:
            connections = self.limit_connections(connections)
        if neuron_label == '':
            neuron_label = 'n{}'.format(self.hidden_neuron_count)
            self.hidden_neuron_count += 1
        self.neurons[neuron_label] = Neuron(neuron_label, connections, self.neuron_selectivity,
                                            f_width=self.f_width,
                                            input_spread=self.input_spread,
                                            input_dimensions=self.input_dimensions)
        self.neurons[neuron_label].visualisation = self.visualise_neuron(neuron_label)
        spread_connections = self.spread_connections(connections)
        for conn in spread_connections:
            pre = conn[0]
            freq = conn[1]
            weight = conn[2]
            self.neurons[neuron_label].add_connection(pre, freq, weight)
        # self.count_synapses(connections)
        # self.age_synapses()
        self.neuron_activity[neuron_label] = self.activity_init#self.neurons[neuron_label].response(connections)
        self.neuron_selectivity[neuron_label] = 0.
        return neuron_label

    def delete_neuron(self, delete_type='conn'):
        if delete_type == 'old':
            oldest_neuron = 'n{}'.format(self.deleted_neuron_count)
            delete_neuron = oldest_neuron
        elif delete_type == 'quiet':
            quiet_neuron = min(self.return_hidden_neurons(self.neuron_selectivity).items(),
                               key=operator.itemgetter(1))[0]
            delete_neuron = quiet_neuron
        else:
            # synapse_count = [[key, self.neurons[key].synapse_count] for key in self.neurons]
            # unconnected_neuron = min(self.neuron_connectedness, key=operator.itemgetter(1))[0]
            unconnected_neuron = min(self.return_hidden_neurons(self.neuron_connectedness).items(),
                                     key=operator.itemgetter(1))[0]
            delete_neuron = unconnected_neuron
        if 'in' in delete_neuron or 'out' in delete_neuron:
            print("not sure what deleting does here")
        del self.neurons[delete_neuron]
        del self.neuron_activity[delete_neuron]
        del self.neuron_selectivity[delete_neuron]
        del self.neuron_connectedness[delete_neuron]
        for i in range(self.number_of_classes):
            if delete_neuron in self.neurons['out{}'.format(i)].synapses:
                del self.neurons['out{}'.format(i)].synapses[delete_neuron]
        self.deleted_neuron_count += 1

    def spread_connections(self, connections):
        if not self.input_dimensions or not self.input_spread:
            return []
        else:
            new_connections = []
            for pre in connections:
                if 'in' in pre:
                    input_idx = int(pre.replace('in', ''))
                    in_x = input_idx % self.input_dimensions[1]
                    in_y = (input_idx - in_x) / self.input_dimensions[1]
                    for x in range(-self.input_spread, self.input_spread+1):
                        for y in range(-self.input_spread, self.input_spread+1):
                            if x == 0 and y == 0:
                                continue
                            new_x = in_x + x
                            new_y = in_y + y
                            if new_x >= self.input_dimensions[0] or new_x < 0 \
                                    or new_y >= self.input_dimensions[1] or new_y < 0:
                                continue
                            distance = np.sqrt(np.power(x, 2) + np.power(y, 2))
                            weight = max(self.input_spread + 1 - distance, 0)
                            if weight == 0:
                                continue
                            weight /= self.input_spread + 1
                            new_idx = int(new_x + (new_y * self.input_dimensions[0]))
                            new_connections.append(['in{}'.format(new_idx), connections[pre], weight])
                            # connections['in{}'.format(new_idx)] = connections[pre]
                            # weights['in{}'.format(new_idx)] = 1. * weight
            return new_connections


    def age_synapses(self):
        expected_inputs = self.always_inputs * self.number_of_inputs
        for neuron in self.neuron_connectedness:
            self.neuron_connectedness[neuron] -= (self.max_hidden_synapses - expected_inputs) / \
                                                 self.maximum_net_size

    def count_synapses(self, connections):
        for pre in connections:
            if pre in self.neuron_connectedness:
                self.neuron_connectedness[pre] += 1
            else:
                self.neuron_connectedness[pre] = 1

    def process_selectivity(self, only_positive=False):
        input_selectivity = {}
        hidden_selectivity = {}
        for neuron in self.neuron_selectivity:
            if 'in' in neuron:
                input_selectivity[neuron] = abs(self.neuron_selectivity[neuron])
            elif 'out' not in neuron:
                if only_positive:
                    hidden_selectivity[neuron] = self.neuron_selectivity[neuron]
                else:
                    hidden_selectivity[neuron] = abs(self.neuron_selectivity[neuron])
        return input_selectivity, hidden_selectivity

    def get_max_selectivity(self, selectivity, n_select, random_sampling=False):
        selected = 0
        if len(selectivity) == 0:
            return {}
        total_dic = {}
        if random_sampling:
            sample_dic = random.sample(selectivity.items(), min(n_select-selected, len(selectivity)))
            for key, v in sample_dic:
                total_dic[key] = v
            return total_dic
        ordered_dict = dict(sorted(selectivity.items(),
                                     # key=lambda dict_key: abs(self.neuron_selectivity[dict_key[0]]),
                                     key=operator.itemgetter(1),
                                     reverse=True))
        while selected < n_select and selected < len(selectivity):
            current_max = None
            selected_dic = {}
            for k, v in ordered_dict.items():
                if current_max is None or v > current_max:
                    current_max = v
                if v == current_max:
                    selected_dic[k] = v
                else:
                    break
            sample_dic = random.sample(selected_dic.items(), min(n_select-selected, len(selected_dic)))
            for key, v in sample_dic:
                total_dic[key] = v
                del ordered_dict[key]
            selected += len(sample_dic)
        return total_dic

    def limit_connections(self, connections):
        if len(connections) <= self.max_hidden_synapses:
            return connections
        pruned_connections = {}
        input_selectivity, hidden_selectivity = self.process_selectivity()
        number_of_hidden = int(min(self.hidden_neuron_count - self.deleted_neuron_count,
                                    self.fixed_hidden_ratio * self.max_hidden_synapses))
        selected_hidden = self.get_max_selectivity(hidden_selectivity,
                                                   number_of_hidden)
        selected_input = self.get_max_selectivity(input_selectivity,
                                                  self.max_hidden_synapses - len(selected_hidden))

        # pre_list = random.sample(list(connections), self.max_hidden_synapses)
        for pre in selected_hidden:
            pruned_connections[pre] = connections[pre]
        for pre in selected_input:
            pruned_connections[pre] = connections[pre]
        return pruned_connections

    def response(self, activations):
        # for i in range(self.layers):
        response = activations
        for neuron in self.neurons:
            response[neuron] = self.neurons[neuron].response(activations)
            # line below can be compressed?
            activations[self.neurons[neuron].neuron_label] = response[neuron]
        for neuron in self.remove_output_neurons(activations, cap=False):
            if neuron in self.neuron_activity:
                self.neuron_selectivity[neuron] = response[neuron] - self.neuron_activity[neuron]
                self.neuron_activity[neuron] = (self.neuron_activity[neuron] * self.activity_decay_rate) + \
                                               (response[neuron] * (1. - self.activity_decay_rate))
            else:
                self.neuron_activity[neuron] = response[neuron]
                # self.neuron_selectivity[neuron] = response[neuron] - self.neuron_activity[neuron]
        outputs = ['out{}'.format(i) for i in range(self.number_of_classes)]
        for neuron in outputs:
            response = self.neurons[neuron].response(activations)
            activations[self.neurons[neuron].neuron_label] = response
        return activations

    def convert_inputs_to_activations(self, inputs):
        acti = {}
        for idx, ele in enumerate(inputs):
            acti['in{}'.format(idx)] = ele
        return acti

    def return_hidden_neurons(self, activations, cap=True):
        neural_activations = {}
        for neuron in activations:
            if 'out' not in neuron and 'in' not in neuron:
                neural_activations[neuron] = activations[neuron]
        return neural_activations

    def remove_output_neurons(self, activations, cap=True):
        neural_activations = {}
        for neuron in activations:
            if 'out' not in neuron:
                if cap:
                    if activations[neuron] >= self.activation_threshold:
                        neural_activations[neuron] = activations[neuron]
                else:
                    neural_activations[neuron] = activations[neuron]
        return neural_activations

    def age_output_synapses(self, reward=True):
        if reward:
            weight_modifier = self.old_weight_modifier
        else:
            weight_modifier = 1. / self.old_weight_modifier
        for i in range(self.number_of_classes):
            for synapse in self.neurons['out{}'.format(i)].synapses:
                for syn in self.neurons['out{}'.format(i)].synapses[synapse]:
                    syn.age_weight()

    def error_driven_neuro_genesis(self, activations, output_error):
        if np.max(np.abs(output_error)) > self.error_threshold:
            activations = self.remove_output_neurons(activations)
            neuron_label = self.add_neuron(activations)
            for output, error in enumerate(output_error):
                if abs(error) > self.error_threshold:
                    # self.current_importance += self.old_weight_modifier
                    # error *= self.current_importance
                    # self.age_output_synapses(reward=True)
                    self.neurons['out{}'.format(output)].add_connection(neuron_label,
                                                                        freq=1.,
                                                                        weight=-error,
                                                                        sensitivities=self.neuron_selectivity,
                                                                        maturation=self.output_synapse_maturity)
                    self.neuron_connectedness[neuron_label] = 1

    def consolidate(self):
        return "new connections to create neurons"

    def visualise_neuron(self, neuron, sensitive=False, only_pos=False):
        visualisation = np.zeros([28, 28])
        for pre in self.neurons[neuron].synapses:
            if 'in' in pre:
                idx = int(pre.replace('in', ''))
                x = idx % 28
                y = int((idx - x) / 28)
                for syn in self.neurons[neuron].synapses[pre]:
                    if sensitive:
                        visualisation[y][x] += syn.contribution()
                    else:
                        visualisation[y][x] += syn.freq
            elif 'out' not in pre:
                for syn in self.neurons[neuron].synapses[pre]:
                    if only_pos:
                        if syn.weight > 0.:
                            if sensitive:
                                visualisation += self.neurons[pre].visualisation * syn.contribution() * \
                                                 syn.weight / self.max_hidden_synapses
                            else:
                                visualisation += self.neurons[pre].visualisation * syn.freq * \
                                                 syn.weight / self.max_hidden_synapses
                    else:
                        if sensitive:
                            visualisation += self.neurons[pre].visualisation * syn.contribution() * \
                                             syn.weight / self.max_hidden_synapses
                        else:
                            visualisation += self.neurons[pre].visualisation * syn.freq * \
                                             syn.weight / self.max_hidden_synapses
        return visualisation

    def visualise_mnist_output(self, output, contribution=True):
        visualisation = np.zeros([28, 28])
        output_neuron = self.neurons['out{}'.format(output)]
        for pre in self.neurons['out{}'.format(output)].synapses:
            if self.neurons['out{}'.format(output)].synapses[pre][0].weight > 0:
                for inputs in self.neurons[pre].synapses:
                    if 'in' in inputs:
                        idx = int(inputs.replace('in', ''))
                        x = idx % 28
                        y = int((idx - x) / 28)
                        for syn in self.neurons[pre].synapses[inputs]:
                            if contribution:
                                visualisation[y][x] += syn.contribution()
                            else:
                                visualisation[y][x] += syn.freq
        # plt.imshow(visualisation, cmap='hot', interpolation='nearest', aspect='auto')
        # plt.savefig("./plots/{}{}.png".format('mnist_memory', output), bbox_inches='tight', dpi=200)
        return visualisation

    def visualise_mnist_activations(self, activations):
        visualisation = np.zeros([28, 28])
        for inputs in activations:
            if 'in' in inputs:
                idx = int(inputs.replace('in', ''))
                x = idx % 28
                y = int((idx - x) / 28)
                visualisation[y][x] += activations[inputs]
        # plt.imshow(visualisation, cmap='hot', interpolation='nearest', aspect='auto')
        # plt.savefig("./plots/{}.png".format('mnist_activations_vis'), bbox_inches='tight', dpi=200)
        return visualisation







