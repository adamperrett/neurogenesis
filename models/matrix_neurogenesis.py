import numpy as np
import random

class Network():
    def __init__(self, number_of_classes, number_of_inputs,
                 error_threshold=0.1,
                 f_width=0.3,
                 maximum_total_synapses=20,
                 maximum_net_size=10000,
                 input_dimensions=None,
                 reward_decay=1.,
                 delete_neuron_type='RL',
                 expecting=False,
                 surprise_threshold=0.4,
                 check_repeat=False,
                 output_thresholding=False):

        self.error_threshold = error_threshold
        self.f_width = f_width
        self.maximum_total_synapses = maximum_total_synapses
        self.maximum_net_size = maximum_net_size
        self.input_dimensions = input_dimensions
        self.reward_decay = reward_decay
        self.delete_neuron_type = delete_neuron_type
        self.expecting = expecting
        self.surprise_threshold = surprise_threshold
        self.check_repeat = check_repeat
        self.output_thresholding = output_thresholding

        self.number_of_inputs = number_of_inputs
        self.number_of_classes = number_of_classes
        self.input_v = []
        self.output_weights = []
        self.synapse_count = 0
        self.neuron_count = 0
        self.deleted_neuron_count = 0
        self.repeated_neuron_count = 0

        self.neuron_activation = []
        self.output_activation = []

        self.procedural = []
        self.procedural_value = [0. for i in range(number_of_classes)]
        self.n_procedural_out = 0

        self.expectation = np.vstack([np.zeros(self.number_of_inputs) for i in range(self.number_of_classes)])
        self.inv_expectation = np.vstack([np.zeros(self.number_of_inputs) for i in range(self.number_of_classes)])

    def first_neuron(self, inputs, output_weights, output):
        self.input_v = np.array(inputs).reshape([1, self.number_of_inputs])
        if self.output_thresholding:
            output_weights = (output_weights > self.error_threshold) * output_weights
        self.output_weights = np.array(output_weights).reshape([1, self.number_of_classes])
        self.neuron_count = 1
        if self.expecting:
            # add weighting to the expectation
            self.expectation[output] = np.nansum(np.vstack([self.expectation[output], inputs]), axis=0)
            self.inv_expectation[output] = np.nansum(np.vstack([self.inv_expectation[output], 1 - inputs]), axis=0)

    def response(self, inputs):
        input_synapses = np.maximum(0, self.f_width - np.abs(np.subtract(self.input_v, inputs)))
        self.neuron_activation = np.nanmean(input_synapses, axis=1) / self.f_width
        hidden_synapses = np.maximum(0, self.f_width - np.abs(self.neuron_activation - 1))
        self.output_activation = np.nansum(np.multiply(hidden_synapses, self.output_weights.T), axis=1) / self.f_width
        return self.output_activation

    def add_neuron(self, connections, output_weights, output):
        if self.check_repeat:
            if connections in self.input_v:
                self.repeated_neuron_count += 1
                return
        self.input_v = np.vstack([self.input_v, connections])
        if self.output_thresholding:
            output_weights = (output_weights > self.error_threshold) * output_weights
        self.output_weights = np.vstack([self.output_weights, output_weights])

        if self.expecting:
            # add weighting to the expectation
            self.expectation[output] = np.nansum(np.vstack([self.expectation[output], connections]), axis=0)
            self.inv_expectation[output] = np.nansum(np.vstack([self.inv_expectation[output], 1 - connections]), axis=0)

    def select_connections(self, inputs, error):
        if self.expecting:
            expectation = self.collect_expectation(error)
            connections = [i if s else np.nan for i, s in zip(inputs,
                                                              np.abs(expectation - inputs) > self.surprise_threshold)]
            return np.array(connections)

        # random selection
        if self.number_of_inputs < self.maximum_total_synapses:
            return inputs

        selection = [True if i < self.maximum_total_synapses else False for i in range(self.number_of_inputs)]
        connections = [i if s else np.nan for i, s in zip(inputs, np.random.shuffle(selection))]
        return np.array(connections)

    def error_driven_neuro_genesis(self, inputs, error, output):
        if np.max(np.abs(error)) > self.error_threshold:
            connections = self.select_connections(inputs, error)
            self.add_neuron(connections, (error > self.error_threshold) * error, output)
            self.neuron_counting()
            self.synapse_counting()

    def collect_expectation(self, error):
        if self.expecting == 'err':
            modulation = error
        elif self.expecting == 'act':
            modulation = self.output_activation
        else:
            modulation = 1
        expectation = np.nansum(np.multiply(self.expectation.T, modulation), axis=1)
        inv_expectation = np.nansum(np.multiply(self.inv_expectation.T, modulation), axis=1)
        total = expectation + inv_expectation
        mask = (total == 0)
        total += mask * 0.00000001
        total_expectation = expectation / total
        return total_expectation + (mask * 5)

    def visualise_classes(self, output, weighted=True, only_pos=False):
        if weighted:
            if only_pos:
                weights = (self.output_weights[:, output] > 0) * self.output_weights[:, output]
            else:
                weights = self.output_weights[:, output]
            neuron_exp = self.input_v / (self.input_v + (1 - self.input_v))
            expectation = np.nansum(neuron_exp.T * weights, axis=1)
            return expectation
        else:
            return self.expectation[output] / (self.expectation[output] + self.inv_expectation[output])

    def neuron_counting(self):
        self.neuron_count = len(self.input_v)
        return self.neuron_count

    def synapse_counting(self):
        self.synapse_count = np.count_nonzero(~np.isnan(self.input_v))
        return self.synapse_count



