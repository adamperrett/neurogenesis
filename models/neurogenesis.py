import numpy as np
import random
import operator
import scipy.stats as st
from models.convert_network import create_network, build_network
from tests.backprop_from_activations import forward_propagate, forward_matrix_propagate


class Synapses():
    def __init__(self, pre, post, freq, f_width=0.3, weight=1., reward_decay=1., sensitivity=0.):
        self.pre = pre
        self.post = post
        self.freq = freq
        self.f_width = f_width
        self.weight = weight
        self.reward_decay = reward_decay
        self.age = 0.
        self.age_multiplier = 1. #/ self.maturation
        self.sensitivity = sensitivity
        self.activity = 0.
        self.reward = 0.1

    def contribution(self):
        return self.sensitivity * self.freq

    def age_weight(self):
        if self.age == self.maturation:
            return
        self.age += 1.
        self.age = min(self.age, self.maturation)
        self.age_multiplier = 1. + self.age#(self.age / self.maturation)
        
    def update_reward(self, reward):
        self.reward = ((1. - self.reward_decay) * self.activity * reward) + (self.reward_decay * self.reward)
        if self.reward < 0:
            print("I'm sorry, what?")
        return self.reward

    def reinforce(self, reward):
        delta_w = reward * self.activity
        if delta_w > 0:
            print("updating - old:", self.weight, "- new:", self.weight + (delta_w * self.reward_decay))
        self.weight += delta_w * self.reward_decay

    def response(self, input):
        # self.activity = st.norm.cdf(-abs((input - self.freq) / self.f_width)) * 2
        self.activity = max(1. - abs((input - self.freq) / self.f_width), 0) #* self.age_multiplier
        return self.activity * self.weight



class Neuron():
    def __init__(self, neuron_label, all_inputs, sensitivities, weights=None, f_width=0.3,
                 input_dimensions=None, reward_decay=1., width_noise=0.):
        self.neuron_label = neuron_label
        self.f_width = f_width
        self.width_noise = width_noise
        self.synapses = {}
        self.synapse_groups = {}
        self.synapse_locations = []
        self.synapse_count = 0
        self.input_dimensions = input_dimensions
        self.visualisation = None
        self.reward_decay = reward_decay
        self.activity = 0.
        self.output = -1
        self.correctness = 0.
        self.all_inputs = all_inputs
        # if not weights:
        #     weights = {}
        if 'out' in neuron_label:
            for pre in all_inputs:
                freq = all_inputs[pre]
                self.add_connection(pre, freq, sensitivities=sensitivities, weight=1.)
        else:
            for inputs, x, y in all_inputs:
                self.add_group(inputs, x, y)

    def add_group(self, inputs, x, y, weight=1., width=None):
        if width == None:
            width = self.f_width + np.random.normal(scale=self.width_noise)
        synapses = {}
        for pre in inputs:
            synapses[pre] = Synapses(pre, self.neuron_label, inputs[pre],
                                     weight=weight,
                                     f_width=width)
        self.synapse_groups['{}, {}'.format(x, y)] = synapses
        self.synapse_locations.append([x, y])
        self.synapse_count += len(synapses)

    def add_connection(self, pre, freq, sensitivities, weight=1., reward_decay=1., width=None):
        if width == None:
            width = self.f_width + np.random.normal(scale=self.width_noise)
        self.synapse_count += 1
        # if 'out' in self.neuron_label:
        #     sensitivities[pre] = 1.
        self.synapses[pre] = Synapses(pre,
                                      self.neuron_label,
                                      freq,
                                      weight=weight,
                                      f_width=width,
                                      reward_decay=self.reward_decay)
        
    def remove_connection(self, pre, idx):
        del self.synapses[pre][idx]
        if len(self.synapses[pre]) == 0:
            del self.synapses[pre]
        self.synapse_count -= 1

    def record_correctness(self, correctness):
        # self.correctness = ((1. - self.reward_decay) * self.activity * correctness) + \
        #                    (self.reward_decay * self.correctness)
        self.correctness += self.activity * correctness
        # print(self.neuron_label, self.activity)
        return self.correctness

    def response(self, all_inputs):
        if not self.synapse_count:
            return 0.

        if 'out' in self.neuron_label:
            response = 0.
            active_synapse_weight = 0
            for pre in self.synapses:  # do as an intersection of sets? inter = activations.keys() & self.synapses.keys()
                if pre in all_inputs:
                    freq = all_inputs[pre]
                    response += self.synapses[pre].response(freq)
                    # active_synapse_weight += 1.  # synapse.weight
            self.activity = response
            return self.activity

        response = 0.
        active_synapse_weight = 0
        number_of_input_groups = len(all_inputs)
        for activations, x, y in all_inputs:
            for syn_x, syn_y in self.synapse_locations:
                # x_value = 1.
                # y_value = 1.
                # if isinstance(x, float) and isinstance(y, float) \
                #         and isinstance(syn_x, float) and isinstance(syn_y, float):
                x_value = max(1. - abs((syn_x - x) / self.f_width), 0)
                y_value = max(1. - abs((syn_y - y) / self.f_width), 0)
                if x_value * y_value > 0:
                    group = self.synapse_groups['{}, {}'.format(syn_x, syn_y)]
                    for synapse in group:
                        freq = activations[synapse]
                        response += group[synapse].response(freq) * x_value * y_value
                        active_synapse_weight += 1

        if active_synapse_weight and 'out' not in self.neuron_label:
            temp_activity = response / active_synapse_weight
        else:
            temp_activity = response
        self.activity = temp_activity #/ number_of_input_groups
        return self.activity


class Network():
    def __init__(self, number_of_classes, number_of_inputs, seed_class=None, seed_features=None, seeds=None,
                 error_threshold=0.1,
                 f_width=0.3,
                 width_noise=0.,
                 activation_threshold=0.01,
                 maximum_total_synapses=20,
                 max_hidden_synapses=100,
                 activity_decay_rate=0.9,
                 always_inputs=True,
                 old_weight_modifier=1.,
                 input_dimensions=None,
                 reward_decay=1.,
                 delete_neuron_type='RL',
                 fixed_hidden_ratio=0.1,
                 activity_init=1.0,
                 replaying=True,
                 hidden_threshold=0.9,
                 conv_size=4):
        self.error_threshold = error_threshold
        self.f_width = f_width
        self.width_noise = width_noise
        self.activation_threshold = activation_threshold
        self.hidden_neuron_count = 0
        self.deleted_neuron_count = 0
        self.maximum_total_synapses = maximum_total_synapses
        self.max_hidden_synapses = max_hidden_synapses
        self.synapse_count = 0
        self.synapse_rewards = []
        self.neuron_rewards = {}
        self.always_inputs = always_inputs
        self.old_weight_modifier = old_weight_modifier
        self.current_importance = 1.
        self.input_dimensions = input_dimensions
        self.reward_decay = reward_decay
        self.delete_neuron_type = delete_neuron_type
        self.fixed_hidden_ratio = fixed_hidden_ratio
        self.activity_init = activity_init
        self.replaying = replaying
        self.hidden_threshold = hidden_threshold
        self.conv_size = conv_size
        self.correctness = [{} for i in range(number_of_classes)]

        self.neurons = {}
        self.neuron_activity = {}
        self.neuron_selectivity = {}
        self.neuron_response = {}
        self.neuron_connectedness = {}
        self.deleted_outputs = {}
        self.activity_decay_rate = activity_decay_rate
        self.number_of_classes = number_of_classes

        self.procedural = []
        self.procedural_value = [0. for i in range(number_of_classes)]
        self.n_procedural_out = 0
        # add seed neuron
        # self.neurons['seed{}'.format(seed_class)] = Neuron('seed{}'.format(seed_class),
        # self.neurons['n0'] = Neuron('n0',
        #                             self.convert_inputs_to_activations(seed_features),
        #                             f_width=f_width)
        self.number_of_inputs = number_of_inputs #len(seed_features[0])
        for i in range(self.number_of_inputs):
            self.neuron_activity['in{}'.format(i)] = 0.
        # add outputs
        for output in range(number_of_classes):
            self.add_neuron({}, 'out{}'.format(output))
        # self.neurons['out{}'.format(seed_class)].add_connection('seed{}'.format(seed_class),
        self.layers = 2


    def add_neuron(self, all_inputs, neuron_label='', seeding=False, output_error=[]):
        if self.max_hidden_synapses and all_inputs:
            all_inputs = self.limit_connections(all_inputs)

        if neuron_label == '':
            neuron_label = 'n{}'.format(self.hidden_neuron_count)
            self.hidden_neuron_count += 1
        self.neurons[neuron_label] = Neuron(neuron_label, all_inputs, self.neuron_selectivity,
                                            f_width=self.f_width,
                                            width_noise=self.width_noise,
                                            input_dimensions=self.input_dimensions,
                                            reward_decay=self.reward_decay)
        self.synapse_count += self.neurons[neuron_label].synapse_count

        #
        # repeated_neuron = False
        # for neuron in self.neurons:
        #     if 'out' in neuron or 'in' in neuron:
        #         continue
        #     if connections == self.neurons[neuron].connections \
        #             and x == self.neurons[neuron].x and y == self.neurons[neuron].y:
        #         repeated_neuron = neuron
        #         for output, error in enumerate(output_error):
        #             if abs(error) > self.error_threshold:
        #                 if neuron in self.neurons['out{}'.format(output)].synapses:
        #                     self.neurons['out{}'.format(output)].synapses[neuron][0].weight -= error
        #                 else:
        #                     self.neurons['out{}'.format(output)
        #                     ].add_connection(neuron,
        #                                      freq=1.,
        #                                      weight=-error,
        #                                      sensitivities=self.neuron_selectivity,
        #                                      # width=0.5,
        #                                      reward_decay=self.reward_decay)
        # if repeated_neuron:
        #     return repeated_neuron
        for output, error in enumerate(output_error):
            if abs(error) > self.error_threshold:
                self.neurons['out{}'.format(output)].add_connection(neuron_label,
                                                                    freq=1.,
                                                                    weight=-error,
                                                                    sensitivities=self.neuron_selectivity,
                                                                    # width=0.5,
                                                                    reward_decay=self.reward_decay)
                self.synapse_count += 1
                self.neuron_connectedness[neuron_label] = 1
        return neuron_label

    def delete_neuron(self, delete_type='RL'):
        # if 'n' not in delete_type:
        #     delete_type = self.delete_neuron_type
        if delete_type == 'old':
            oldest_neuron = 'n{}'.format(self.deleted_neuron_count)
            delete_neuron = oldest_neuron
        elif delete_type == 'new':
            # oldest_neuron = 'n{}'.format(self.deleted_neuron_count)
            delete_neuron = 'n{}'.format(self.hidden_neuron_count - 1)
        elif delete_type == 'quiet':
            quiet_neuron = min(self.return_hidden_neurons(self.neuron_selectivity).items(),
                               key=operator.itemgetter(1))[0]
            delete_neuron = quiet_neuron
        elif 'n' in delete_type:
            delete_neuron = delete_type
        elif delete_type == 'RL':
            delete_neuron = min(self.neuron_rewards.items(),
                               key=operator.itemgetter(1))[0]
        else:
            # synapse_count = [[key, self.neurons[key].synapse_count] for key in self.neurons]
            # unconnected_neuron = min(self.neuron_connectedness, key=operator.itemgetter(1))[0]
            unconnected_neuron = min(self.return_hidden_neurons(self.neuron_connectedness).items(),
                                     key=operator.itemgetter(1))[0]
            delete_neuron = unconnected_neuron
        if 'in' in delete_neuron or 'out' in delete_neuron:
            print("not sure what deleting does here")
        self.synapse_count -= self.neurons[delete_neuron].synapse_count
        if delete_neuron in self.correctness[self.neurons[delete_neuron].output]:
            del self.correctness[self.neurons[delete_neuron].output][delete_neuron]
        del self.neurons[delete_neuron]
        del self.neuron_activity[delete_neuron]
        if delete_neuron in self.neuron_selectivity:
            del self.neuron_selectivity[delete_neuron]
        del self.neuron_connectedness[delete_neuron]
        del self.neuron_rewards[delete_neuron]
        if delete_neuron in self.neuron_response:
            del self.neuron_response[delete_neuron]
        for neuron in self.neurons:
            if delete_neuron in self.neurons[neuron].synapses:
                del self.neurons[neuron].synapses[delete_neuron]
        for i in range(self.number_of_classes):
            if delete_neuron in self.neurons['out{}'.format(i)].synapses:
                self.synapse_count -= len(self.neurons['out{}'.format(i)].synapses[delete_neuron])
                del self.neurons['out{}'.format(i)].synapses[delete_neuron]
        self.deleted_neuron_count += 1

    def delete_synapses(self, amount):
        self.synapse_rewards.sort()
        for i in range(amount):
            neuron = self.synapse_rewards[0][1]
            pre = self.synapse_rewards[0][2]
            idx = self.synapse_rewards[0][3]
            self.neurons[neuron].remove_connection(pre, idx)
            del self.synapse_rewards[0]
            self.synapse_count -= 1
            if self.neurons[neuron].synapse_count == 0:
                self.delete_neuron(neuron)

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
            if 'in' in neuron or 'p' in neuron:
                if len(self.procedural) >= 30:
                    if 'p' in neuron:
                        # if 'p2' in neuron or 'p3' in neuron:
                        input_selectivity[neuron] = abs(self.neuron_selectivity[neuron])
                else:
                    # if 'in' in neuron:
                    input_selectivity[neuron] = abs(self.neuron_selectivity[neuron])
                # input_selectivity[neuron] = abs(self.neuron_selectivity[neuron])
            elif 'out' not in neuron:
                if self.neuron_selectivity[neuron] == 1.:
                    print("da fuck")
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

    def limit_connections(self, all_inputs, selectivity=False, thresholded=False, outlier=True, conv=False):
        all_activations = np.hstack([all_inputs[i][0] for i in range(len(all_inputs))])
        if len(all_activations) <= min(self.max_hidden_synapses, self.number_of_inputs):
            return all_inputs
        # print("add thresholded selection here here")
        trimmed = [[{}, x, y] for _, x, y in all_inputs]
        syn = 0
        while syn < self.max_hidden_synapses:
            glimpse = np.random.choice(len(all_inputs))
            inp = np.random.choice(list(all_inputs[glimpse][0].keys()))
            if inp not in trimmed[glimpse][0]:
                trimmed[glimpse][0][inp] = all_inputs[glimpse][0][inp]
                syn += 1

        return trimmed

    def reset_neuron_activity(self):
        for neuron in self.neurons:
            self.neurons[neuron].activity = 0.

    def response(self, all_inputs):
        response = {}
        for neuron in self.neurons:
            response[neuron] = self.neurons[neuron].response(all_inputs)

        outputs = ['out{}'.format(i) for i in range(self.number_of_classes)]
        for out, neuron in enumerate(outputs):
            response[neuron] = self.neurons[neuron].response(response)
        return response

    def record_correctness(self, correct_output):
        for neuron in self.neurons:
            if 'out' not in neuron and 'in' not in neuron:
                if self.neurons[neuron].output == correct_output:
                    val = self.neurons[neuron].record_correctness(1)
                else:
                    val = self.neurons[neuron].record_correctness(-1)
                self.correctness[self.neurons[neuron].output][neuron] = val
        return self.correctness

    def reinforce_synapses(self, reward, only_output=True, correct_output=0):
        self.synapse_rewards = []
        if only_output:
            for i in range(self.number_of_classes):
                neuron = 'out{}'.format(i)
                if i == correct_output:
                    syn_reward = 1.#reward
                else:
                    syn_reward = -1#reward
                for pre in self.neurons[neuron].synapses:
                    for syn in self.neurons[neuron].synapses[pre]:
                        syn_r = syn.reinforce(syn_reward)
                        self.synapse_rewards.append([syn_r, neuron, pre, self.neurons[neuron].synapses[pre].index(syn)])
        else:
            for neuron in self.neurons:
                for pre in self.neurons[neuron].synapses:
                    for syn in self.neurons[neuron].synapses[pre]:
                        syn_r = syn.update_reward(reward)
                        self.synapse_rewards.append([syn_r, neuron, pre, self.neurons[neuron].synapses[pre].index(syn)])
        return self.synapse_rewards

    def reinforce_neurons(self, reward):
        # self.neuron_rewards = []
        for neuron in self.neurons:
            if 'out' not in neuron and 'in' not in neuron:
                if neuron in self.neuron_rewards and neuron in self.neuron_response:
                    self.neuron_rewards[neuron] = ((1. - self.reward_decay) * self.neuron_response[neuron] * reward) \
                                                  + (self.reward_decay * self.neuron_rewards[neuron])
                # else:
                #     if neuron not in self.deleted_outputs: ### check this
                #         self.neuron_rewards[neuron] = sum(self.neuron_rewards.values()) / len(self.neuron_rewards)
        return self.neuron_rewards

    def remove_worst_output(self):
        delete_neuron = min(self.neuron_rewards.items(),
                            key=operator.itemgetter(1))[0]
        self.deleted_outputs[delete_neuron] = True
        del self.neuron_rewards[delete_neuron]
        for i in range(self.number_of_classes):
            if delete_neuron in self.neurons['out{}'.format(i)].synapses:
                self.synapse_count -= 1
                del self.neurons['out{}'.format(i)].synapses[delete_neuron]

    def convert_inputs_to_activations(self, inputs):
        self.procedural_value, all_p = self.procedural_output(inputs)
        acti = {}
        for idx, ele in enumerate(inputs):
            acti['in{}'.format(idx)] = ele
        if len(self.procedural) > 0:
            for idx, ele in enumerate(all_p):
                acti['p{}'.format(idx)] = ele
        return acti

    def return_hidden_neurons(self, activations, cap=True):
        neural_activations = {}
        for neuron in activations:
            if 'out' not in neuron and 'in' not in neuron:
                neural_activations[neuron] = activations[neuron]
        return neural_activations

    def remove_output_neurons(self, activations, cap=False):
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

    def collect_all_vis(self, error=None, correct_class=0, activations=None):
        # if error == None:
        #     error = np.zeros(self.number_of_classes)
        #     error[correct_class] = 1.
        vis = np.zeros([28, 28])
        for output in range(self.number_of_classes):
            # vis += self.neurons['out{}'.format(output)].visualisation * error[output]
            vis += self.neurons['out{}'.format(output)].visualisation * activations['out{}'.format(output)]
        return vis

    def convert_vis_to_activations(self, neuron='', vis=None, norm=False):
        if neuron != '':
            vis = self.neurons[neuron].visualisation
        if norm and np.max(vis) != np.min(vis):
            vis = (vis - np.min(vis)) / (np.max(vis) - np.min(vis))
        activations = {}
        for y in range(28):
            for x in range(28):
                idx = (y * 28) + x
                activations['in{}'.format(idx)] = vis[y][x]
        return activations

    def remove_all_stored_values(self):
        deleted_neurons = []
        for neuron in self.neurons:
            if 'out' not in neuron:
                deleted_neurons.append(neuron)
        for del_n in deleted_neurons:
            self.delete_neuron(del_n)
        # for i in range(self.number_of_classes):
        #     self.neurons['out{}'.format(i)].synapses = {}

    def remove_hidden_neurons(self, connections):
        new_conn = {}
        for conn in connections:
            if 'in' in conn or 'p' in conn:
                new_conn[conn] = connections[conn]
        return new_conn

    def convert_net_and_clean(self):
        # self.procedural.append(create_network(self))
        # self.n_procedural_out += self.number_of_classes
        if len(self.procedural) >= 4:
            new_net, centroids = create_network(self, polar=True, correctness=False)
        else:
            new_net, centroids = create_network(self, polar=False, correctness=True)
        self.procedural.append(new_net)
        for layer in new_net:
            self.n_procedural_out += len(layer)
        # self.procedural = build_network(self, polar=True)
        # self.n_procedural_out = 0
        # for layer in self.procedural:
        #     self.n_procedural_out += len(layer)
        # self.number_of_inputs + len(self.procedural[-1])
        # for i in range(self.number_of_inputs):
        #     self.neuron_activity['in{}'.format(i)] = 0.
        self.remove_all_stored_values()
        return centroids

    def procedural_output(self, inputs):
        output = [0. for i in range(self.number_of_classes)]
        if len(self.procedural) == 0:
            return output, inputs[2:]
        all_out = []
        all_act = []
        for net in self.procedural:
            # pro_out, neuron_out = forward_propagate(self.procedural, inputs)
            # pro_out, neuron_out = forward_propagate(net, inputs)
            pro_out, neuron_out = forward_matrix_propagate(net, inputs)
            inputs = np.hstack([inputs, neuron_out])
            all_act = np.hstack([all_act, neuron_out])
            if len(all_out):
                all_out = np.vstack([all_out, pro_out])
            else:
                all_out = [pro_out]
        # for out, value in enumerate(pro_out):
        #     output[out] += value

        if len(self.procedural) >= 1:
            for out, value in enumerate(all_out[-1]):
                output[out] += value
        else:
            for net in all_out:
                for out, value in enumerate(net):
                    output[out] += value
        return output, all_act

    def error_driven_neuro_genesis(self, all_inputs, output_error, weight_multiplier=1., label=-1):
        if np.max(np.abs(output_error)) > self.error_threshold:
            # activations = self.remove_output_neurons(activations)
            neuron_label = self.add_neuron(all_inputs, output_error=output_error)
            self.neurons[neuron_label].output = label
            return neuron_label
        else:
            return "thresholded"

    def pass_errors_to_outputs(self, errors, lr):
        for out, error in enumerate(errors):
            for synapse in self.neurons['out{}'.format(out)].synapses:
                for syn in self.neurons['out{}'.format(out)].synapses[synapse]:
                    weight_d = syn.activity * error * lr
                    syn.weight -= weight_d

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
                                                 syn.weight / len(self.neurons[neuron].synapses) # not if multapse
                            else:
                                visualisation += self.neurons[pre].visualisation * syn.freq * \
                                                 syn.weight / len(self.neurons[neuron].synapses) # not if multapse
                    else:
                        if sensitive:
                            visualisation += self.neurons[pre].visualisation * syn.contribution() * \
                                             syn.weight / len(self.neurons[neuron].synapses) # not if multapse
                        else:
                            visualisation += self.neurons[pre].visualisation * syn.freq * \
                                             syn.weight / len(self.neurons[neuron].synapses) # not if multapse
        self.neurons[neuron].visualisation = visualisation
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







