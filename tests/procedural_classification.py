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
import seaborn as sn
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit, LeaveOneOut, StratifiedKFold


test = 'mnist'
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
    retest_rate = 10000
    retest_size = 1000
elif test == 'pima':
    from datasets.pima_indians import *
    num_outputs = 2
    train_labels = training_set_labels
    train_feat = training_set_pimas
    test_labels = test_set_labels
    test_feat = test_set_pimas
    retest_rate = 50
    retest_size = len(test_set_pimas)
elif test == "simple":
    # centres = [[.5, -.5],
    #            [.5, .5],
    #            [-.5, .5],
    #            [-.5, -.5]]
    centres = [[1, 0],
               [0, 0],
               # [0, 1]]#,
               [-1, 0]]
    x_range = [-2, 2]
    y_range = [-2, 2]
    spread = 0.3
    examples = 200
    test_set_size = 0.1
    # simple_data, simple_labels = create_centroid_classes(centres, spread, examples)
    num_outputs = 2
    # num_outputs = len(centres)
    simple_data, simple_labels = create_bimodal_distribution(centres, spread, examples, max_classes=num_outputs)
    train_labels = simple_labels[:int(examples*len(centres)*(1. - test_set_size))]
    train_feat = simple_data[:int(examples*len(centres)*(1. - test_set_size))]
    test_labels = simple_labels[int(examples*len(centres)*(1. - test_set_size)):]
    test_feat = simple_data[int(examples*len(centres)*(1. - test_set_size)):]
    retest_rate = 1
    retest_size = len(test_feat)
elif test == 'yinyang':
    examples = 500
    num_outputs = 3
    test_set_size = 0.1
    x_range = [-0.1, 1.1]
    y_range = [-0.1, 1.1]
    yy = YinYangDataset(size=examples, seed=np.random.randint(0, 1000))
    simple_data = yy._YinYangDataset__vals
    simple_labels = yy._YinYangDataset__cs
    train_labels = simple_labels[:int(examples*(1. - test_set_size))]
    train_feat = simple_data[:int(examples*(1. - test_set_size))]
    test_labels = simple_labels[int(examples*(1. - test_set_size)):]
    test_feat = simple_data[int(examples*(1. - test_set_size)):]
    retest_rate = 1
    retest_size = len(test_feat)
elif test == 'spiral':
    examples = 250
    num_outputs = 2
    simple_data, simple_labels = twospirals(examples)
    examples *= 2
    test_set_size = 0.1
    train_labels = simple_labels[:int(examples*(1. - test_set_size))].tolist()
    train_feat = simple_data[:int(examples*(1. - test_set_size))].tolist()
    test_labels = simple_labels[int(examples*(1. - test_set_size)):].tolist()
    test_feat = simple_data[int(examples*(1. - test_set_size)):].tolist()
    x_range = [-0.1, 1.1]
    y_range = [-0.1, 1.1]
    retest_rate = 1
    retest_size = len(test_feat)
if 'mnist' in test:
    input_dimensions = [28, 28]
else:
    input_dimensions = None
num_inputs = len(train_feat[0])

def test_net(net, data, labels, indexes=None, test_net_label='', classifications=None,
             fold_string='', max_fold=None, noise_stdev=0., save_activations=False):
    global previous_full_accuracy, fold_testing_accuracy, best_testing_accuracy
    if not isinstance(indexes, np.ndarray):
        indexes = [i for i in range(len(labels))]
    activations = {}
    train_count = 0
    correct_classifications = 0
    confusion_matrix = np.zeros([num_outputs+1, num_outputs+1])
    synapse_counts = []
    neuron_counts = []
    error_values = []
    delete_list = []
    all_activations = []
    for test in indexes:
        train_count += 1
        features = data[test]
        label = labels[test]
        features, procedural_out = preprocess_inputs(features, hypernet)
        output = net.response(features)
        output += procedural_out
        error, choice, softmax = calculate_error(label, output, test_net_label)

        if label == choice:
            correct_classifications += 1
            if 'esting' not in test_net_label:
                # net.reinforce_neurons(1.)
                classifications.append(1)
                net.error_driven_neuro_genesis(features, error, label)
                print("CORRECT CLASS WAS CHOSEN")
        else:
            if 'esting' not in test_net_label:
                print("INCORRECT CLASS WAS CHOSEN")
                # net.reinforce_neurons(-1.)
                classifications.append(0)
                net.error_driven_neuro_genesis(features, error, label)

        if 'esting' not in test_net_label:
            print(test_net_label, "\nEpoch ", epoch, "/", epochs)
            print(fold_string)
            print('test ', train_count, '/', len(indexes))
            print("Neuron count: ", CLASSnet.neuron_count, " - del", CLASSnet.deleted_neuron_count)
            print("Synapse count: ", CLASSnet.synapse_count)
            print("Repeated neuron count: ", CLASSnet.repeated_neuron_count)
            print(test_label)
            for ep in epoch_error:
                print(ep)
            confusion_matrix[choice][label] += 1
            synapse_counts.append(CLASSnet.synapse_count)
            neuron_counts.append(CLASSnet.neuron_count)
        if 'esting' not in test_net_label and train_count % retest_rate == retest_rate - 1:
            print("retesting")
            if len(test_index) > 1000:
                test_index_sample = np.random.choice(test_index, retest_size)
            else:
                test_index_sample = test_index
            testing_accuracy, training_classifications, \
            testing_confusion, _, _, _ = test_net(CLASSnet, X, y,
                                                  indexes=test_index_sample,
                                                  test_net_label='Testing',
                                                  classifications=classifications,
                                                  # fold_test_accuracy=fold_testing_accuracy,
                                                  fold_string=fold_string,
                                                  max_fold=maximum_fold_accuracy)
            fold_testing_accuracy.append(round(testing_accuracy, 3))
            best_testing_accuracy.append(round(testing_accuracy, 3))
            print("finished")
        if 'esting' not in test_net_label:
            # print(label, features)
            # correctness = net.record_correctness(label)
            print("Performance over all current tests")
            print(correct_classifications / train_count)
            print("Performance over last tests")
            for window in average_windows:
                print(np.average(classifications[-window:]), ":", window)
            if fold_testing_accuracy:
                print("Fold testing accuracy", fold_testing_accuracy)
                print("Maximum fold = ", maximum_fold_accuracy)
                print("Performance over last", len(fold_testing_accuracy), "folds")
                for window in fold_average_windows:
                    print(np.average(fold_testing_accuracy[-window:]), ":", window)
            print("\n")
        else:
            if train_count % 1000 == 0:
                print(train_count, "/", len(indexes))
    correct_classifications /= train_count
    if 'esting' not in test_net_label:
        print('Epoch', epoch, '/', epochs, '\nClassification accuracy: ',
              correct_classifications)
    if save_activations:
        return correct_classifications, classifications, confusion_matrix, \
               synapse_counts, neuron_counts, all_activations
    else:
        return correct_classifications, classifications, confusion_matrix, \
               synapse_counts, neuron_counts, error_values

def convert_net_and_reset(centroid_type):
    collected_centroids = []
    for i in range(num_outputs):
        if centroid_type == 'exp':
            collected_centroids.append(CLASSnet.visualise_classes(i, weighted=False))
        elif centroid_type == 'pos':
            collected_centroids.append(CLASSnet.visualise_classes(i, only_pos=True))
        elif centroid_type == 'neg':
            collected_centroids.append(CLASSnet.visualise_classes(i, only_pos=False))
        else:
            print("Empty")

    hidden_connections = np.empty((0, CLASSnet.number_of_inputs + 1))
    output_connections = np.empty((0, CLASSnet.number_of_classes))
    for out_a, c_a in enumerate(collected_centroids):
        for out_b, c_b in enumerate(collected_centroids):
            if out_a < out_b:
                dot_product = np.dot((np.array(c_a) + np.array(c_b)) / 2.,
                                     (c_b - c_a))
                bias = -dot_product
                hidden_connections = np.vstack([hidden_connections,
                                                np.hstack([c_b - c_a, bias]) / distance.euclidean(c_a, c_b)])
                output_weights = np.zeros(CLASSnet.number_of_classes)
                output_weights[out_a] = -1
                output_weights[out_b] = 1
                output_connections = np.vstack([output_connections,
                                                output_weights])
    hidden_count = len(hidden_connections)
    output_count = num_outputs
    for h, o in hypernet:
        hidden_count += len(h)
        output_count += num_outputs
    planeNet = Network(num_outputs, num_inputs+hidden_count+output_count,
                       error_threshold=error_threshold,
                       f_width=sensitivity_width,
                       maximum_synapses_per_neuron=maximum_synapses_per_neuron,
                       input_dimensions=input_dimensions,
                       reward_decay=reward_decay,
                       delete_neuron_type=delete_neuron_type,
                       surprise_threshold=surprise_threshold,
                       expecting=expecting,
                       maximum_net_size=maximum_net_size,
                       output_thresholding=output_thresholding)
    return [hidden_connections, output_connections], planeNet

def preprocess_inputs(inputs, hyper_net):
    o_act = np.zeros(num_outputs)
    for hidden, output in hyper_net:
        h_sum = np.matmul(np.hstack([inputs, 1.]), hidden.T)
        h_act = np.tanh(h_sum)
        o_act = np.matmul(h_act, output)
        inputs = np.hstack([inputs, h_act, o_act])
    return inputs, o_act

def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def plot_learning_curve(correct_or_not, fold_test_accuracy, training_confusion, testing_confusion,
                        synapse_counts, neuron_counts,
                        test_label, save_flag=False):
    if not saving_plots:
        return 0
    fig, axs = plt.subplots(2, 2)
    ave_err10 = moving_average(correct_or_not, 10)
    ave_err100 = moving_average(correct_or_not, 100)
    ave_err1000 = moving_average(correct_or_not, 1000)
    axs[0][0].scatter([i for i in range(len(correct_or_not))], correct_or_not, s=5)
    axs[0][0].plot([i + 5 for i in range(len(ave_err10))], ave_err10, 'r')
    axs[0][0].plot([i + 50 for i in range(len(ave_err100))], ave_err100, 'b')
    axs[0][0].plot([i + 500 for i in range(len(ave_err1000))], ave_err1000, 'g')
    if len(ave_err1000):
        axs[0][0].plot([0, len(correct_or_not)], [ave_err1000[-1], ave_err1000[-1]], 'g')
    axs[0][0].set_xlim([0, len(correct_or_not)])
    axs[0][0].set_ylim([0, 1])
    axs[0][0].set_title("running average of training classification")
    ave_err10 = moving_average(fold_test_accuracy, 4)
    ave_err100 = moving_average(fold_test_accuracy, 10)
    ave_err1000 = moving_average(fold_test_accuracy, 20)
    axs[0][1].scatter([i for i in range(len(fold_test_accuracy))], fold_test_accuracy, s=5)
    axs[0][1].plot([i + 2 for i in range(len(ave_err10))], ave_err10, 'r')
    axs[0][1].plot([i + 5 for i in range(len(ave_err100))], ave_err100, 'b')
    axs[0][1].plot([i + 10 for i in range(len(ave_err1000))], ave_err1000, 'g')
    axs[0][1].plot([i for i in range(len(best_testing_accuracy))], best_testing_accuracy, 'k')
    if len(ave_err1000):
        axs[0][1].plot([0, len(correct_or_not)], [ave_err1000[-1], ave_err1000[-1]], 'g')
    axs[0][1].set_xlim([0, len(fold_test_accuracy)])
    axs[0][1].set_ylim([0, 1])
    axs[0][1].set_title("test classification")
    axs[1][0].plot([i for i in range(len(neuron_counts))], neuron_counts)
    axs[1][0].set_title("Neuron and synapse count")
    ax_s = axs[1][0].twinx()
    ax_s.plot([i for i in range(len(synapse_counts))], synapse_counts, 'r')
    if len(epoch_error):
        if len(epoch_error) <= 10:
            data = np.hstack([np.array(epoch_error)[:, 0].reshape([len(epoch_error), 1]),
                    np.array(epoch_error)[:, 1].reshape([len(epoch_error), 1]),
                    np.array(epoch_error)[:, 2].reshape([len(epoch_error), 1]),
                    np.array(epoch_error)[:, 3].reshape([len(epoch_error), 1])])
            axs[1][1].table(cellText=data, colLabels=['training accuracy', 'full final training',
                                                      'testing accuracy', 'final testing accuracy'],
                            rowLabels=['{}'.format(i) for i in range(len(epoch_error))],
                            loc="center")
            axs[1][1].axis('off')
        else:
            axs[1][1].set_xlim([0, len(epoch_error)])
            axs[1][1].set_ylim([0, 1])
            axs[1][1].plot([i for i in range(len(epoch_error))], np.array(epoch_error)[:, 1])
            ave_err10 = moving_average(np.array(epoch_error)[:, 1], min(2, len(epoch_error)))
            axs[1][1].plot([i + 1 for i in range(len(ave_err10))], ave_err10, 'r')
            ave_err10 = moving_average(np.array(epoch_error)[:, 1], min(10, len(epoch_error)))
            axs[1][1].plot([i + 5 for i in range(len(ave_err10))], ave_err10, 'g')
            axs[1][1].plot([0, len(epoch_error)], [ave_err10[-1], ave_err10[-1]], 'g')
            axs[1][1].set_title("Epoch test classification")
    figure = plt.gcf()
    figure.set_size_inches(16, 9)
    plt.tight_layout(rect=[0, 0.3, 1, 0.95])
    plt.suptitle(test_label, fontsize=16)
    if save_flag:
        plt.savefig("./plots/{} {}.png".format(test_label, repeat), bbox_inches='tight', dpi=200, format='png')
    plt.close()
    np.save("./data/{}".format(test_label), data_dict) #data = np.load('./tests/data/file_name.npy', allow_pickle=True).item()

def extend_data(epoch_length):
    global running_neuron_counts, running_synapse_counts
    running_neuron_counts = running_neuron_counts.tolist()
    running_synapse_counts = running_synapse_counts.tolist()
    for i in range(epoch_length):
        best_testing_accuracy.append(best_testing_accuracy[-1])
        fold_testing_accuracy.append(fold_testing_accuracy[-1])
        training_classifications.append(training_classifications[-1])
        running_neuron_counts.append(running_neuron_counts[-1])
        running_synapse_counts.append(running_synapse_counts[-1])
    running_neuron_counts = np.array(running_neuron_counts)
    running_synapse_counts = np.array(running_synapse_counts)

def normalise_outputs(out_activations):
    min_out = min(out_activations)
    max_out = max(out_activations)
    out_range = max_out - min_out
    norm_out = []
    for out in out_activations:
        norm_out.append((out - min_out) / out_range)
    return np.array(norm_out)

def calculate_error(correct_class, output_activations, test_label):
    error = np.zeros(num_outputs)
    one_hot_encoding = np.zeros(num_outputs)
    one_hot_encoding[correct_class] = 1
    if error_type == 'sm':
        softmax = sm(output_activations)
    elif error_type == 'norm':
        softmax = normalise_outputs(output_activations)
    else:
        softmax = output_activations
    if min(softmax) != max(softmax):
        choice = softmax.argmax()
    else:
        choice = num_outputs
    if error_type == 'zero':
        softmax = np.zeros(num_outputs)
    for output in range(num_outputs):
        error[output] += one_hot_encoding[output] - softmax[output]

    if 'esting' not in test_label:
        for output in range(num_outputs):
            print("{} - {}:{} - sm:{} - err:{}".format(one_hot_encoding[output],
                                                       output,
                                                       output_activations[output],
                                                       softmax[output],
                                                       error[output]))
    return error, choice, softmax

read_args = False
if read_args:
    import sys
    sensitivity_width = float(sys.argv[1])
    activation_threshold = float(sys.argv[2])
    error_threshold = float(sys.argv[3])
    maximum_total_synapses = int(sys.argv[4])
    maximum_synapses_per_neuron = int(sys.argv[5])
    input_spread = int(sys.argv[6])
    activity_init = 1.
    activity_decay_rate = float(sys.argv[7])
    number_of_seeds = int(sys.argv[8])
    fixed_hidden_amount = float(sys.argv[9])
    fixed_hidden_ratio = fixed_hidden_amount / maximum_synapses_per_neuron
    print("Variables collected")
    for i in range(9):
        print(sys.argv[i+1])
else:
    sensitivity_width = 0.4
    activation_threshold = 0.0
    error_threshold = 0.2
    maximum_synapses_per_neuron = 1
    # fixed_hidden_amount = 0
    fixed_hidden_ratio = 0.0
    # fixed_hidden_ratio = fixed_hidden_amount / maximum_synapses_per_neuron
    maximum_total_synapses = 100*3000000
    input_spread = 0
    activity_decay_rate = 1.#0.9999
    activity_init = 1.
    number_of_seeds = 0

maximum_net_size = int(maximum_total_synapses / maximum_synapses_per_neuron)
old_weight_modifier = 1.01
maturity = 100.
hidden_threshold = 0.95
delete_neuron_type = 'RL'
reward_decay = 0.9999
conv_size = 9
max_out_synapses = 50000
# activity_init = 1.0
always_inputs = False
replaying = False
error_type = 'sm'
epochs = 10
repeats = 10
width_noise = 0.#5
noise_level = 0.#5
out_weight_scale = 0.0#0075
learning_rate = 1.0
visualise_rate = 1
np.random.seed(27)
confusion_decay = 0.8
always_save = True
remove_class = 2
check_repeat = False
expecting = 'err'
convert_type = 'exp'
hypernet = []
surprise_threshold = 0.1

output_thresholding = True

# number_of_seeds = min(number_of_seeds, len(train_labels))
# seed_classes = random.sample([i for i in range(len(train_labels))], number_of_seeds)
test_label = 'matrix {} - sth{} outh{} retest{} exp-{} mns{}x{} er-{} sw{}eth{}'.format(
    test,
    surprise_threshold, output_thresholding,
    retest_rate, expecting,
    maximum_net_size, maximum_synapses_per_neuron,
    error_type,
    sensitivity_width,
    error_threshold
)

average_windows = [30, 100, 300, 1000, 3000, 10000, 100000]
fold_average_windows = [3, 10, 30, 60, 100, 1000]

X = train_feat + test_feat
y = train_labels + test_labels
if 'mnist' not in test:
    # sss = StratifiedShuffleSplit(n_splits=repeats, test_size=0.1, random_state=27)
    sss = StratifiedKFold(n_splits=repeats, random_state=2727, shuffle=True)
    # sss = LeaveOneOut()
else:
    # class mnist_data():
    #     def __init__(self):
    #         self.train_labels = mnist_training_labels
    #         self.train_feat = mnist_training_data
    #         self.test_labels = mnist_testing_labels
    #         self.test_feat = mnist_testing_data
    #
    # sss = StratifiedShuffleSplit(n_splits=repeats, test_size=0.3, random_state=0)
    train_index = [i for i in range(60000)]
    test_index = [i + 60000 for i in range(10000)]
    combined_index = [[np.array(train_index), np.array(test_index)]]
    # print("Not currently setup for MNIST")
    # Exception

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

# for repeat, (train_index, test_index) in enumerate(sss.split(X, y)):
for repeat, (train_index, test_index) in enumerate(combined_index):
    np.random.seed(int(100*learning_rate))

    CLASSnet = Network(num_outputs, num_inputs,
                       error_threshold=error_threshold,
                       f_width=sensitivity_width,
                       maximum_synapses_per_neuron=maximum_synapses_per_neuron,
                       input_dimensions=input_dimensions,
                       reward_decay=reward_decay,
                       delete_neuron_type=delete_neuron_type,
                       surprise_threshold=surprise_threshold,
                       expecting=expecting,
                       maximum_net_size=maximum_net_size,
                       output_thresholding=output_thresholding)
    all_incorrect_classes = []
    epoch_error = []
    # noise_results = []
    previous_accuracy = 0
    previous_full_accuracy = 0

    fold_testing_accuracy = [0]
    best_testing_accuracy = [0]
    maximum_fold_accuracy = [[0, 0]]
    training_classifications = []
    running_train_confusion = np.zeros([num_outputs+1, num_outputs+1])
    running_test_confusion = np.zeros([num_outputs+1, num_outputs+1])
    running_synapse_counts = np.zeros([1])
    running_neuron_counts = np.zeros([1])
    running_error_values = np.array([])
    only_lr = True
    for epoch in range(epochs):
        if epoch % 4 == 3:
            only_lr = not only_lr
        if epoch % 10 == 0 and epoch:
            for ep, error in enumerate(epoch_error):
                print(ep, error)
            print("it reached 10")
        max_folds = int(len(train_labels) / retest_rate)
        training_count = 0
        while training_count < len(train_index):
            training_count += len(train_index)
            current_fold = training_count / len(train_index)
            fold_string = 'fold {} / {}'.format(int(current_fold), max_folds)
            np.random.shuffle(train_index)
            training_accuracy, training_classifications, \
            training_confusion, synapse_counts, \
            neuron_counts, error_values = test_net(CLASSnet, X, y,
                                                   indexes=train_index,
                                                   test_net_label='Training',
                                                   # fold_test_accuracy=fold_testing_accuracy,
                                                   classifications=training_classifications,
                                                   fold_string=fold_string,
                                                   max_fold=maximum_fold_accuracy, noise_stdev=noise_level
                                                   )
        if 'mnist' in test:
            final_accuracy = np.mean(training_classifications[-len(train_index):])
        else:
            final_accuracy, _, _, _, _, _ = test_net(CLASSnet, X, y,
                                                     indexes=train_index,
                                                     test_net_label='new neuron testing',
                                                     classifications=training_classifications,
                                                     # fold_test_accuracy=fold_testing_accuracy,
                                                     fold_string=fold_string,
                                                     max_fold=maximum_fold_accuracy
                                                     )

            running_synapse_counts = np.hstack([running_synapse_counts, synapse_counts])
            running_neuron_counts = np.hstack([running_neuron_counts, neuron_counts])
            running_error_values = np.hstack([running_error_values, error_values])
            # training_classifications += new_classifications
            testing_indexes = random.sample([i for i in range(len(test_labels))], retest_size)
            testing_accuracy, training_classifications, \
            testing_confusion, _, _, _ = test_net(CLASSnet, X, y,
                                                  test_net_label='Testing',
                                                  indexes=test_index,
                                                  classifications=training_classifications,
                                                  # fold_test_accuracy=fold_testing_accuracy,
                                                  fold_string=fold_string,
                                                  max_fold=maximum_fold_accuracy
                                                  )

            previous_accuracy = testing_accuracy
            # fold_testing_accuracy.append(round(testing_accuracy, 3))
            running_train_confusion *= confusion_decay
            running_train_confusion += training_confusion
            running_test_confusion *= confusion_decay
            running_test_confusion += testing_confusion
            plot_learning_curve(training_classifications, fold_testing_accuracy,
                                running_train_confusion, running_test_confusion,
                                running_synapse_counts, running_neuron_counts, test_label, save_flag=True)

            if current_fold % 10 == 0 and current_fold:
                print("it reached 10 folds")

        if retest_size < len(test_labels):  #depricated with retest above
            full_testing_accuracy, training_classifications, \
            full_test_confusion, _, _, _ = test_net(CLASSnet, X, y,
                                                    indexes=test_index,
                                                    test_net_label='Testing',
                                                    classifications=training_classifications,
                                                    # fold_test_accuracy=fold_testing_accuracy,
                                                    fold_string=fold_string,
                                                    max_fold=maximum_fold_accuracy
                                                    )

            epoch_error.append([round(np.mean(training_classifications[-len(train_index):]), 4),
                                round(final_accuracy, 4),
                                # round(final_procedural_accuracy, 4),
                                round(fold_testing_accuracy[-1]),
                                round(full_testing_accuracy, 4),
                                CLASSnet.neuron_counting(), CLASSnet.synapse_counting()])
            running_test_confusion *= confusion_decay
            running_test_confusion += full_test_confusion
        else:
            epoch_error.append([round(np.mean(training_classifications[-len(train_index):]), 4),
                                round(final_accuracy, 4),
                                # round(final_procedural_accuracy, 4),
                                round(fold_testing_accuracy[-1], 4),
                                round(testing_accuracy, 4),
                                CLASSnet.neuron_counting(), CLASSnet.synapse_counting()]
                               )
            running_test_confusion *= confusion_decay
            running_test_confusion += testing_confusion
        epoch_error[-1] = np.array(epoch_error[-1])

        new_layer, new_neurogenet = convert_net_and_reset(convert_type)
        CLASSnet = new_neurogenet
        hypernet.append(new_layer)

        plot_learning_curve(training_classifications, fold_testing_accuracy,
                            running_train_confusion, running_test_confusion,
                            running_synapse_counts, running_neuron_counts, test_label, save_flag=True)

        print(test_label)
        for ep, error in enumerate(epoch_error):
            print(ep, error)
        print(test_label)

    data_dict['epoch_error'].append(epoch_error)
    data_dict['fold_testing_accuracy'].append(fold_testing_accuracy)
    data_dict['best_testing_accuracy'].append(best_testing_accuracy)
    data_dict['training_classifications'].append(training_classifications)
    data_dict['running_train_confusion'].append(running_train_confusion)
    data_dict['running_test_confusion'].append(running_test_confusion)
    data_dict['running_synapse_counts'].append(running_synapse_counts)
    data_dict['running_neuron_counts'].append(running_neuron_counts)
    data_dict['running_error_values'].append(running_error_values)
    data_dict['net'].append(CLASSnet)
    print("Finished repeat", repeat)

ave_data = {}
for key in data_dict:
    if 'net' not in key:
        ave_test = []
        for j in range(len(data_dict[key][0])):
            total = 0.
            for i in range(len(data_dict[key])):
                total += data_dict[key][i][j]
            ave_test.append(total / len(data_dict[key]))
        ave_data[key] = ave_test
data_dict['ave_data'] = ave_data
np.save("./data/{}".format(test_label), data_dict)

# import matplotlib.pyplot as plt
plt.plot(ave_data['fold_testing_accuracy'])
plt.suptitle(test_label)
plt.savefig("./plots/ave_test {}.png".format(test_label), bbox_inches='tight', dpi=200, format='png')

print(ave_data['fold_testing_accuracy'])
print(test_label)
print(ave_data['fold_testing_accuracy'][:int(len(train_index) / retest_rate)])
print(ave_data['epoch_error'])

print("done")




