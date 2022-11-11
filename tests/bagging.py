import numpy as np
from scipy.special import softmax as sm
from copy import deepcopy
from models.matrix_neurogenesis import Network
from datasets.simple_tests import *
from models.convert_network import *
import random
import matplotlib
saving_plots = True
if saving_plots:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit, LeaveOneOut, StratifiedKFold

test = 'pp_mnist'
if test == 'breast':
    from breast_data import *
    num_outputs = 2
    train_labels = training_set_labels
    train_feat = training_set_breasts
    test_labels = test_set_labels
    test_feat = test_set_breasts
    retest_rate = 10#len(train_labels)
    retest_size = len(test_set_labels)
elif test == 'wine':
    from wine_data import *
    num_outputs = 3
    train_labels = training_set_labels
    train_feat = training_set_wines
    test_labels = test_set_labels
    test_feat = test_set_wines
    retest_rate = 1#len(train_labels)
    retest_size = len(test_set_labels)
elif test == 'mnist':
    from datasets.mnist_csv import *
    num_outputs = 10
    train_labels = mnist_training_labels
    train_feat = mnist_training_data
    test_labels = mnist_testing_labels
    test_feat = mnist_testing_data
    retest_rate = 100
    retest_size = 50
elif test == 'pp_mnist':
    from datasets.preprocessed_mnist import *
    num_outputs = 10
    train_labels = mnist_training_labels
    train_feat = mnist_training_data
    test_labels = mnist_testing_labels
    test_feat = mnist_testing_data
    retest_rate = 1000
    retest_size = 10
elif test == 'noise':
    from datasets.high_noise_inputs import *
    num_inputs = 10000
    num_outputs = 3
    examples = 3000
    test += ' i{}o{}e{}'.format(num_inputs, num_outputs, examples)
    noise_data, noise_labels = generate_date(num_inputs=num_inputs,
                                             num_outputs=num_outputs,
                                             examples=examples)
    test_set_size = 0.1
    train_labels = noise_labels[:int(examples*(1. - test_set_size))]
    train_feat = noise_data[:int(examples*(1. - test_set_size))]
    test_labels = noise_labels[int(examples*(1. - test_set_size)):]
    test_feat = noise_data[int(examples*(1. - test_set_size)):]
    retest_rate = 200
    retest_size = len(test_feat)
if 'mnist' in test:
    input_dimensions = [28, 28]
else:
    input_dimensions = None
num_inputs = len(train_feat[0])

def calculate_error(correct_class, output_activations, test_label):
    error = np.zeros(num_outputs)
    one_hot_encoding = np.zeros(num_outputs)
    one_hot_encoding[correct_class] = 1

    softmax = sm(output_activations)
    if min(softmax) != max(softmax):
        choice = softmax.argmax()
    else:
        choice = num_outputs

    for output in range(num_outputs):
        error[output] += one_hot_encoding[output] - softmax[output]

    if 'esting' not in test_label:
        for output in range(num_outputs):
            print("{} - {}:{} - sm:{} - err:{}".format(one_hot_encoding[output],
                                                       output,
                                                       output_activations[output],
                                                       softmax[output],
                                                       error[output]))
    return error, choice, softmax, output_activations

def test_network(net, all_idx, testing=False):
    classifications = []
    output_values = {}
    synapse_counts = []
    neuron_counts = []

    for idx in all_idx:
        features = X[idx]
        label = Y[idx]
        output = net.response(features)
        print("\n", test_label)
        error, choice, softmax, activations = calculate_error(label, output, 'training')
        output_values['{}'.format(idx)] = {'sm': softmax,
                                           'output': activations,
                                           'vote': [1 if i == choice else 0 for i in range(num_outputs)]}

        if label == choice:
            classifications.append(1)
            print("CORRECT CLASS WAS CHOSEN")
        else:
            classifications.append(0)
            print("INCORRECT CLASS WAS CHOSEN")
        if not testing:
            net.error_driven_neuro_genesis(features, error, label)
            synapse_counts.append(net.synapse_count)
            neuron_counts.append(net.neuron_count)

        print("Performance over all current tests")
        print("Testing:", testing, " - ", len(classifications), "/", len(all_idx))
        print("Neuron count: ", net.neuron_count, " - del", net.deleted_neuron_count)
        print("Synapse count: ", net.synapse_count)
        print("Performance over last tests")
        for window in average_windows:
            print(np.average(classifications[-window:]), ":", window)

    print("The final classification accuracy is ", np.sum(classifications) / len(classifications))

    if not testing:
        print("Beginning evaluation on the test set")
        test_classifications, test_output_values, \
            _, _, _, _, _ = test_network(net, test_index, testing=True)
    else:
        test_classifications = classifications
        test_output_values = output_values

    return classifications, output_values, \
               test_classifications, test_output_values, \
               net, synapse_counts, neuron_counts

def combine_models(classifications, model_outputs, label_idx):
    model_stats = [{
        'accuracy': np.sum(classifications[m]) / len(classifications[m])
    } for m in range(len(classifications))]
    print("Individual model stats:")
    for i, m in enumerate(model_stats):
        print(i, m)
    print("Average model stats:",
          np.average([np.sum(classifications[m]) / len(classifications[m]) for m in range(len(classifications))]))

    ensemble_stats = {
        'vote': 0,
        'output': 0,
        'sm': 0
    }

    for i in label_idx:
        label = Y[i]
        for agg_type in ensemble_stats:
            aggregated_vote = np.sum([m['{}'.format(i)][agg_type] for m in model_outputs], axis=0)
            _, agg_choice, _, _ = calculate_error(label, aggregated_vote, 'testing')
            ensemble_stats[agg_type] += int(agg_choice == label)

    print("Ensemble accuracy:")
    for agg_type in ensemble_stats:
        ensemble_stats[agg_type] /= len(label_idx)
        print("Accuracy for", agg_type, "- ", ensemble_stats[agg_type])

    return ensemble_stats

def plot_ensemble_progress(ensemble_data, classifications):
    plt.figure()
    plt.plot([x for x in range(len(ensemble_data))], [m['vote'] for m in ensemble_data], label='vote')
    plt.axhline(y=[all_ensemble_stats[-1]['vote']], color='r')
    plt.plot([x for x in range(len(ensemble_data))], [m['sm'] for m in ensemble_data], label='sm')
    plt.axhline(y=[all_ensemble_stats[-1]['sm']], color='r')
    plt.plot([x for x in range(len(ensemble_data))], [m['output'] for m in ensemble_data], label='output')
    plt.axhline(y=[all_ensemble_stats[-1]['output']], color='r')

    plt.plot([x for x in range(len(classifications))],
             [np.sum(classifications[m]) / len(classifications[m]) for m in
              range(len(classifications))],
             label='model')

    plt.xlabel('model number')
    plt.ylabel('test accuracy')

    plt.legend(loc='lower right')
    figure = plt.gcf()
    figure.set_size_inches(16, 9)
    plt.tight_layout(rect=[0, 0.3, 1, 0.95])
    plt.suptitle(test_label, fontsize=16)
    plt.savefig("./plots/{}.png".format(test_label), bbox_inches='tight', dpi=200, format='png')
    plt.close()

"""
for T agents:
    select random subset of size N
    train network
    test network (save classification and output values)
Combined vote across agents


"""

np.random.seed(27)

# bagging variables
number_of_models = 100
bag_size = 1000#len(train_labels)

# EDN variables
sensitivity_width = 0.8
surprise_threshold = 0.2
error_threshold = 0.1
expecting = 'neu'#'act'

# Old?
activation_threshold = 0.0
maximum_synapses_per_neuron = 12800
fixed_hidden_ratio = 0.0
maximum_total_synapses = 100*3000000

# Auxilary var
maximum_net_size = int(maximum_total_synapses / maximum_synapses_per_neuron)
delete_neuron_type = 'RL'
always_inputs = False
replaying = False
epochs = 0
repeats = 1
always_save = True
check_repeat = False
output_thresholding = False

test_label = 'bagging{} {} - sth{} outh{} exp-{} sw{} eth{}'.format(
    bag_size,
    test,
    surprise_threshold, output_thresholding,
    expecting,
    sensitivity_width,
    error_threshold
)

average_windows = [30, 100, 300, 1000, 3000, 10000, 100000]
fold_average_windows = [3, 10, 30, 60, 100, 1000]

data_setup = []
if 'mnist' not in test:
    X = train_feat + test_feat
    Y = train_labels + test_labels
    sss = StratifiedKFold(n_splits=repeats, random_state=2727, shuffle=True)
    for repeat, (train_index, test_index) in enumerate(sss.split(X, Y)):
        data_setup.append([repeat, train_index, test_index])
else:
    X = np.vstack([train_feat, test_feat])
    Y = np.hstack([train_labels, test_labels])
    train_index = [i for i in range(60000)]
    test_index = [i + 60000 for i in range(10000)]
    data_setup = [[r, train_index, test_index] for r in range(repeats)]


data_dict = {}
data_dict['epoch_error'] = []
data_dict['fold_testing_accuracy'] = []
data_dict['best_testing_accuracy'] = []
data_dict['training_classifications'] = []
data_dict['running_train_confusion'] = []
data_dict['running_test_confusion'] = []
data_dict['running_synapse_counts'] = []
data_dict['running_neuron_counts'] = []
data_dict['running_error_values'] = []
data_dict['net'] = []

all_ensemble_stats = []
final_ensemble_stats = []
for repeat, train_index, test_index in data_setup:
    print("Beginning repeat:", repeat+1, "/", repeats)
    all_model_classifications = []
    all_model_outputs = []
    all_nets = []
    for model in range(number_of_models):
        print("Beginning model:", model+1, "/", number_of_models)
        bagnet = Network(num_outputs, num_inputs,
                         error_threshold=error_threshold,
                         f_width=sensitivity_width,
                         maximum_synapses_per_neuron=maximum_synapses_per_neuron,
                         input_dimensions=input_dimensions,
                         delete_neuron_type=delete_neuron_type,
                         surprise_threshold=surprise_threshold,
                         expecting=expecting,
                         maximum_net_size=maximum_net_size,
                         output_thresholding=output_thresholding)

        bag_select = np.random.choice(train_index, bag_size)
        classifications, output_values, \
            model_test_classifications, model_test_output_values, \
            model_net, synapse_counts, neuron_counts = \
            test_network(bagnet, bag_select)
        all_model_classifications.append(model_test_classifications)
        all_model_outputs.append(model_test_output_values)
        # all_nets.append(model_net)

        ensemble_stats = combine_models(all_model_classifications,
                                        all_model_outputs,
                                        test_index)
        print("Finished model", model+1)
        all_ensemble_stats.append(ensemble_stats)
        plot_ensemble_progress(all_ensemble_stats, all_model_classifications)

    final_ensemble_stats.append(all_ensemble_stats[-1])

    print("Creating benchmark")
    bagnet = Network(num_outputs, num_inputs,
                     error_threshold=error_threshold,
                     f_width=sensitivity_width,
                     maximum_synapses_per_neuron=maximum_synapses_per_neuron,
                     input_dimensions=input_dimensions,
                     delete_neuron_type=delete_neuron_type,
                     surprise_threshold=surprise_threshold,
                     expecting=expecting,
                     maximum_net_size=maximum_net_size,
                     output_thresholding=output_thresholding)

    single_test = []
    for e in range(epochs):
        classifications, output_values, \
        model_test_classifications, model_test_output_values, \
        bagnet, synapse_counts, neuron_counts = \
            test_network(bagnet, train_index)
        single_test.append(np.average(model_test_classifications))

    print("\n", test_label, "\nFinished repeat", repeat+1)
    final_ensemble_stats[-1]['single'] = single_test


print("Ensemble model")
print({k: np.average([m[k] for m in final_ensemble_stats], axis=0) for k in final_ensemble_stats[0]})
print("Finished")






