import numpy as np
from scipy.special import softmax as sm
from copy import deepcopy
from models.neurogenesis import Network
import random


test = 'mnist'
if test == 'breast':
    from breast_data import *
    num_outputs = 2
    train_labels = training_set_labels
    train_feat = training_set_breasts
    test_labels = test_set_labels
    test_feat = test_set_breasts
    retest_rate = 10
    retest_size = len(test_set_labels)
elif test == 'wine':
    from wine_data import *
    num_outputs = 3
    train_labels = training_set_labels
    train_feat = training_set_wines
    test_labels = test_set_labels
    test_feat = test_set_wines
    retest_rate = 100
    retest_size = 10
elif test == 'mnist':
    from datasets.mnist_csv import *
    num_outputs = 10
    train_labels = mnist_training_labels
    train_feat = mnist_training_data
    test_labels = mnist_testing_labels
    test_feat = mnist_testing_data
    retest_rate = 100
    retest_size = 50
elif test == 'rmnist':
    from datasets.mnist_csv import *
    num_outputs = 10
    train_labels = mnist_training_labels
    train_feat = reduced_mnist_training_data
    test_labels = mnist_testing_labels
    test_feat = reduced_mnist_testing_data
    retest_rate = 1000
    retest_size = 50
elif test == 'pima':
    from datasets.pima_indians import *
    num_outputs = 2
    train_labels = training_set_labels
    train_feat = training_set_pimas
    test_labels = test_set_labels
    test_feat = test_set_pimas
    retest_rate = 100
    retest_size = len(test_set_pimas)
if 'mnist' in test:
    input_dimensions = [28, 28]
else:
    input_dimensions = None
num_inputs = len(train_feat[0])

def test_net(net, data, labels, indexes=None, test_net_label='', classifications=None,
             fold_test_accuracy=None, fold_string='', max_fold=[]):
    if not indexes:
        indexes = [i for i in range(len(labels))]
    activations = {}
    train_count = 0
    correct_classifications = 0
    # incorrect_classes = []
    for test in indexes:
        train_count += 1
        features = data[test]
        label = labels[test]
        activations = net.convert_inputs_to_activations(features)
        activations = net.response(activations)
        print(test_net_label, "\nEpoch ", epoch, "/", epochs)
        print(fold_string)
        print('test ', train_count, '/', len(indexes))
        error, choice = calculate_error(label, activations, train_count, num_outputs)
        print("neuron count", len(activations) - len(features) - num_outputs)
        print(test_label)
        for ep in epoch_error:
            print(ep)
        if label == choice:
            correct_classifications += 1
            if 'esting' not in test_net_label:
                classifications.append(1)
            print("CORRECT CLASS WAS CHOSEN")
        else:
            print("INCORRECT CLASS WAS CHOSEN")
            if 'esting' not in test_net_label:
                classifications.append(0)
                net.error_driven_neuro_genesis(activations, error)
            # incorrect_classes.append('({}) {}: {}'.format(train_count, label, choice))
        # classifications.append([choice, label])
        print("Performance over all current tests")
        print(correct_classifications / train_count)
        print("Performance over last tests")
        for window in average_windows:
            print(np.average(classifications[-window:]), ":", window)
        if fold_testing_accuracy:
            print("Fold testing accuracy", fold_testing_accuracy)
            print("Maximum fold = ", maximum_fold_accuracy)
            print("Performance over last folds")
            for window in average_windows:
                print(np.average(fold_test_accuracy[-window:]), ":", window)
        print("\n")
    # print(incorrect_classes)
    # all_incorrect_classes.append(incorrect_classes)
    # for ep in all_incorrect_classes:
    #     print(len(ep), "-", ep)
    correct_classifications /= train_count
    print('Epoch', epoch, '/', epochs, '\nClassification accuracy: ',
          correct_classifications)
    return correct_classifications, classifications

def normalise_outputs(out_activations):
    min_out = min(out_activations)
    max_out = max(out_activations)
    out_range = max_out - min_out
    norm_out = []
    for out in out_activations:
        norm_out.append((out - min_out) / out_range)
    return np.array(norm_out)

def calculate_error(correct_class, activations, test_label, num_outputs=2):
    output_activations = np.zeros(num_outputs)
    error = np.zeros(num_outputs)
    one_hot_encoding = np.ones(num_outputs)
    one_hot_encoding *= -0
    one_hot_encoding[correct_class] = 1
    for output in range(num_outputs):
        output_activations[output] = activations['out{}'.format(output)]
    softmax = sm(output_activations)
    # softmax = normalise_outputs(output_activations)
    # softmax = output_activations
    if sum(softmax) > 0.:
        choice = softmax.argmax()
    else:
        choice = -1
    for output in range(num_outputs):
        error[output] += softmax[output] - one_hot_encoding[output]
        # error[output] = - one_hot_encoding[output]

    print("Error for test ", test_label, " is ", error)
    print("output")
    for output in range(num_outputs):
        print("{} - {}:{} - sm:{} - err:{}".format(one_hot_encoding[output],
                                                   output,
                                                   output_activations[output],
                                                   softmax[output],
                                                   error[output]))
    return error, choice

read_args = False
if read_args:
    import sys
    sensitivity_width = float(sys.argv[1])
    activation_threshold = float(sys.argv[2])
    error_threshold = float(sys.argv[3])
    maximum_total_synapses = int(sys.argv[4])
    maximum_synapses_per_neuron = int(sys.argv[5])
    input_spread = int(sys.argv[6])
    activity_decay_rate = float(sys.argv[7])
    number_of_seeds = int(sys.argv[8])
    print("Variables collected")
    for i in range(5):
        print(sys.argv[i+1])
else:
    sensitivity_width = 0.99
    activation_threshold = 0.0
    error_threshold = 0.01
    maximum_synapses_per_neuron = 500
    maximum_total_synapses = 100*1000000
    input_spread = 3
    activity_decay_rate = 0.99
    number_of_seeds = 300

maximum_net_size = int(maximum_total_synapses / maximum_synapses_per_neuron)
old_weight_modifier = 1.
always_inputs = False
epochs = 200
np.random.seed(27)
number_of_seeds = min(number_of_seeds, len(train_labels))
seed_classes = random.sample([i for i in range(len(train_labels))], number_of_seeds)
test_label = 'max_net:{}_{}  - {}{} aging{} - sw{} - ' \
             'at{} - et{} - adr{} - inp_{}'.format(maximum_net_size, maximum_synapses_per_neuron,
                                                   number_of_seeds, test, old_weight_modifier,
                                                   sensitivity_width,
                                                   activation_threshold,
                                                   error_threshold,
                                                   activity_decay_rate,
                                                   always_inputs
                                                   )
if 'mnist' in test:
    test_label += ' spread{}'.format(input_spread)


average_windows = [10, 30, 50, 100, 200, 300, 500, 1000]

CLASSnet = Network(num_outputs, train_labels, train_feat, seed_classes,
                   error_threshold=error_threshold,
                   f_width=sensitivity_width,
                   activation_threshold=activation_threshold,
                   maximum_net_size=maximum_net_size,
                   max_hidden_synapses=maximum_synapses_per_neuron,
                   activity_decay_rate=activity_decay_rate,
                   always_inputs=always_inputs,
                   old_weight_modifier=old_weight_modifier,
                   input_dimensions=input_dimensions,
                   input_spread=input_spread)
all_incorrect_classes = []
epoch_error = []

fold_testing_accuracy = []
maximum_fold_accuracy = [[0, 0]]
for epoch in range(epochs):
    if epoch == 3:
        for ep, error in enumerate(epoch_error):
            print(ep, error)
        print("it reached 10")
    max_folds = int(len(train_labels) / retest_rate) + 1
    training_count = 0
    training_classifications = []
    while training_count < len(train_labels):
        training_indexes = [i for i in range(training_count, min(training_count + retest_rate, len(train_labels)))]
        training_count += retest_rate
        current_fold = training_count / retest_rate
        fold_string = 'fold {} / {}'.format(int(current_fold), max_folds)
        training_accuracy, training_classifications = test_net(CLASSnet, train_feat, train_labels,
                                                               indexes=training_indexes,
                                                               test_net_label='Training',
                                                               fold_test_accuracy=fold_testing_accuracy,
                                                               classifications=training_classifications,
                                                               fold_string=fold_string,
                                                               max_fold=maximum_fold_accuracy)
        testing_indexes = random.sample([i for i in range(len(test_labels))], retest_size)
        testing_accuracy, training_classifications = test_net(CLASSnet, test_feat, test_labels,
                                                              test_net_label='Testing',
                                                              indexes=testing_indexes,
                                                              classifications=training_classifications,
                                                              fold_test_accuracy=fold_testing_accuracy,
                                                              fold_string=fold_string,
                                                              max_fold=maximum_fold_accuracy)
        fold_testing_accuracy.append(round(testing_accuracy, 3))
        if testing_accuracy > maximum_fold_accuracy[-1][0]:
            total_test_accuracy, _ = test_net(CLASSnet, train_feat+test_feat, train_labels+test_labels,
                                              test_net_label='Testing',
                                              classifications=training_classifications,
                                              fold_test_accuracy=fold_testing_accuracy,
                                              fold_string=fold_string,
                                              max_fold=maximum_fold_accuracy
                                              )
            maximum_fold_accuracy.append([testing_accuracy, total_test_accuracy, epoch, current_fold,
                                          CLASSnet.hidden_neuron_count])

    # testing_accuracy, training_classifications = test_net(CLASSnet, test_feat, test_labels,
    #                                                       test_net_label='Testing',
    #                                                       classifications=training_classifications,
    #                                                       fold_test_accuracy=fold_testing_accuracy,
    #                                                       fold_string=fold_string,
    #                                                       max_fold=maximum_fold_accuracy)

    epoch_error.append([np.mean(training_classifications[-len(train_labels):]), testing_accuracy,
                        CLASSnet.hidden_neuron_count])
    print(test_label)
    for ep, error in enumerate(epoch_error):
        print(ep, error)
    print(test_label)

print("done")




