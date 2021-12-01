
import numpy as np
from scipy.special import softmax as sm
from copy import deepcopy
from models.neurogenesis import Network
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
from sklearn.model_selection import StratifiedShuffleSplit, LeaveOneOut, StratifiedKFold, ShuffleSplit


test = 'mpg'
if test == 'mpg':
    from datasets.mpg_regression import norm_features, norm_mpg, min_mpg, max_mpg
    num_inputs = len(norm_features[0])
    num_outputs = 2
    retest_rate = 1
    retest_size = int(0.1 * len(norm_mpg))

def test_net(net, data, values, indexes=None, test_net_label='', all_errors=None,
             fold_string='', max_fold=None, noise_stdev=0.):
    global previous_full_accuracy, fold_testing_accuracy, best_testing_accuracy
    if not isinstance(indexes, np.ndarray):
        indexes = [i for i in range(len(values))]
    activations = {}
    train_count = 0
    regression_error = 0
    # all_errors = []
    synapse_counts = []
    neuron_counts = []
    all_activations = []
    for test in indexes:
        train_count += 1
        features = data[test]
        value = values[test]

        noise = np.random.normal(scale=noise_stdev, size=np.array(features).shape)
        activations = net.convert_inputs_to_activations(np.array(features) + noise)
        activations = net.response(activations)
        error, error_value = calculate_error(value, activations, test_net_label, num_outputs)
        regression_error += error_value
        neuron_count = CLASSnet.hidden_neuron_count - CLASSnet.deleted_neuron_count

        if 'esting' not in test_net_label:
            all_errors.append(error_value)
            # if only_lr:
            # CLASSnet.pass_errors_to_outputs(error, learning_rate)
            print(test_net_label, "\nEpoch ", epoch, "/", epochs)
            print(fold_string)
            print('test ', train_count, '/', len(indexes))
            print("Neuron count: ", CLASSnet.hidden_neuron_count, " - ", CLASSnet.deleted_neuron_count, " = ",
                  CLASSnet.hidden_neuron_count - CLASSnet.deleted_neuron_count)
            print("Synapse count: ", CLASSnet.synapse_count)
            print(test_label)
            for ep, err in enumerate(epoch_error):
                print(ep, err)
            print(test_label)
            synapse_counts.append(CLASSnet.synapse_count)
            neuron_counts.append(neuron_count)

            neuron_label = net.error_driven_neuro_genesis(
                activations, -error,
                weight_multiplier=1.,
                label=value)

        if 'esting' not in test_net_label and train_count % retest_rate == retest_rate - 1:
            print("retesting")
            if len(test_index) > 1000:
                test_index_sample = np.random.choice(test_index, retest_size)
            else:
                test_index_sample = test_index
            testing_accuracy, training_classifications, \
            _, _ = test_net(CLASSnet, X, y,
                                               indexes=test_index_sample,
                                               test_net_label='Testing',
                                               all_errors=all_errors,
                                               # fold_test_accuracy=fold_testing_accuracy,
                                               fold_string=fold_string,
                                               max_fold=maximum_fold_accuracy)
            fold_testing_accuracy.append(round(testing_accuracy, 4))
            best_testing_accuracy.append(round(testing_accuracy, 4))
            print("finished")
            # if train_count % 10 == 0:
                # CLASSnet.convert_net_and_clean()
                # determine_2D_decision_boundary(CLASSnet, x_range, y_range, 100, X, y)
            # determine_2D_decision_boundary(CLASSnet, [-1, 2], [-1, 2], 100, X, y)
        # neuron_counts.append(CLASSnet.hidden_neuron_count - CLASSnet.deleted_neuron_count)
        if 'esting' not in test_net_label:
            # print(label, features)
            # correctness = net.record_correctness(label)
            print("Performance over all current tests")
            print(regression_error)
            print("Performance over last tests")
            for window in average_windows:
                print(np.average(all_errors[-window:]), ":", window)
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
    regression_error /= train_count
    if 'esting' not in test_net_label:
        # del neuron_counts[-1]
        # neuron_counts.append(CLASSnet.hidden_neuron_count - CLASSnet.deleted_neuron_count)
        # del synapse_counts[-1]
        # synapse_counts.append(CLASSnet.synapse_count)
        print('Epoch', epoch, '/', epochs, '\nRegression accuracy: ',
              regression_error)
    return regression_error, all_errors, synapse_counts, neuron_counts

def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def plot_learning_curve(correct_or_not, fold_test_accuracy,
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
    # axs[0][0].set_ylim([0, 1])
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
    # axs[0][1].set_ylim([0, 1])
    axs[0][1].set_title("test classification")
    axs[1][0].plot([i for i in range(len(neuron_counts))], neuron_counts)
    axs[1][0].set_title("Neuron count")
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
        plt.savefig("./plots/{}.png".format(test_label), bbox_inches='tight', dpi=200, format='png')
    plt.close()

    data_dict = {}
    data_dict['training classifications'] = correct_or_not
    data_dict['fold_test_accuracy'] = fold_test_accuracy
    data_dict['best_testing_accuracy'] = best_testing_accuracy
    data_dict['synapse_counts'] = synapse_counts
    data_dict['neuron_counts'] = neuron_counts
    data_dict['epoch error'] = epoch_error
    data_dict['noise_results'] = noise_results
    # data_dict['all_activations'] = all_activations
    data_dict['net'] = CLASSnet
    np.save("./data/{}.png".format(test_label), data_dict) #data = np.load('./tests/data/file_name.npy', allow_pickle=True).item()

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

def calculate_error(correct_value, activations, test_label, num_outputs=2):
    output_activations = np.zeros(num_outputs)
    for output in range(num_outputs):
        output_activations[output] = activations['out{}'.format(output)]
    if output_activations[0] == 0 and output_activations[1] == 0:
        output_value = -10
        error = 1.
    else:
        output_value = output_activations[0] / (sum(output_activations))
        if error_type == 'linear':
            error = correct_value - output_value
            error = np.abs(error)
        else:
            error = np.square(correct_value - output_value)
    converted_correct_value = np.array([correct_value, 1. - correct_value]) * error

    # print("output")
    if 'esting' not in test_label:
        print("Error for test ", test_label, " is ", error)
        print(error, " = ", correct_value, " - ", output_value)
        print(error, " = ", output_activations[0], " \/ ", output_activations[1])
    #     for output in range(num_outputs):
    #         print("{} - {}:{} - sm:{} - err:{}".format(one_hot_encoding[output],
    #                                                    output,
    #                                                    output_activations[output],
    #                                                    softmax[output],
    #                                                    error[output]))
    return converted_correct_value, error

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
    sensitivity_width = 0.8
    activation_threshold = 0.0
    error_threshold = 0.0
    maximum_synapses_per_neuron = 8
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
error_type = 'square'
epochs = 20
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

noise_tests = np.linspace(0, 2., 21)

# number_of_seeds = min(number_of_seeds, len(train_labels))
# seed_classes = random.sample([i for i in range(len(train_labels))], number_of_seeds)
base_label = 'regression{}ms{} {}{} {}{}  - {} fixed_h{} - sw{}n{} - ' \
             'at{} - et{} - {}adr{} - {}noise'.format(retest_rate, maximum_synapses_per_neuron, error_type, out_weight_scale,
                                            delete_neuron_type, reward_decay,
                                                     # maximum_net_size, maximum_synapses_per_neuron,
                                                   test,
                                                   fixed_hidden_ratio,
                                                    # fixed_hidden_amount,
                                                   sensitivity_width, width_noise,
                                                   activation_threshold,
                                                   error_threshold,
                                                   activity_init, activity_decay_rate,
                                                      noise_level
                                                   )

average_windows = [30, 100, 300, 1000, 3000, 10000, 100000]
fold_average_windows = [3, 10, 30, 60, 100, 1000]

X = norm_features
y = norm_mpg

# sss = StratifiedShuffleSplit(n_splits=repeats, test_size=0.1, random_state=27)
# sss = StratifiedKFold(n_splits=repeats, random_state=2727, shuffle=True)
sss = ShuffleSplit(n_splits=repeats, test_size=0.1, random_state=27272)
# sss = LeaveOneOut()

for repeat, (train_index, test_index) in enumerate(sss.split(X, y)):
# for repeat, (train_index, test_index) in enumerate(combined_index):
    if repeats == 1:
        test_label = base_label
    else:
        test_label = base_label + ' {}'.format(repeat)
    np.random.seed(int(100*learning_rate))

    CLASSnet = Network(num_outputs, num_inputs,
                       error_threshold=error_threshold,
                       f_width=sensitivity_width,
                       width_noise=width_noise,
                       activation_threshold=activation_threshold,
                       maximum_total_synapses=maximum_total_synapses,
                       max_hidden_synapses=maximum_synapses_per_neuron,
                       activity_decay_rate=activity_decay_rate,
                       always_inputs=always_inputs,
                       old_weight_modifier=old_weight_modifier,
                       reward_decay=reward_decay,
                       delete_neuron_type=delete_neuron_type,
                       fixed_hidden_ratio=fixed_hidden_ratio,
                       activity_init=activity_init,
                       replaying=replaying,
                       hidden_threshold=hidden_threshold,
                       conv_size=conv_size)
    all_incorrect_classes = []
    epoch_error = []
    noise_results = []
    previous_accuracy = 0
    previous_full_accuracy = 0

    fold_testing_accuracy = []
    best_testing_accuracy = []
    maximum_fold_accuracy = [[0, 0]]
    all_errors = []
    running_train_confusion = np.zeros([num_outputs+1, num_outputs+1])
    running_test_confusion = np.zeros([num_outputs+1, num_outputs+1])
    running_synapse_counts = np.zeros([1])
    running_neuron_counts = np.zeros([1])
    only_lr = True
    for epoch in range(epochs):
        if epoch % 4 == 3:
            only_lr = not only_lr
        if epoch % 10 == 0 and epoch:
        # if (epoch == 10 or epoch == 30) and epoch:
            for ep, error in enumerate(epoch_error):
                print(ep, error)
            print("it reached 10")
        max_folds = int(len(train_index) / retest_rate)
        training_count = 0
        while training_count < len(train_index):
            training_count += len(train_index)
            current_fold = training_count / len(train_index)
            fold_string = 'fold {} / {}'.format(int(current_fold), max_folds)
            np.random.shuffle(train_index)
            training_accuracy, training_classifications, \
            synapse_counts, \
            neuron_counts = test_net(CLASSnet, X, y,
                                     indexes=train_index,
                                     test_net_label='Training',
                                     # fold_test_accuracy=fold_testing_accuracy,
                                     all_errors=all_errors,
                                     fold_string=fold_string,
                                     max_fold=maximum_fold_accuracy, noise_stdev=noise_level)

            final_accuracy, _, _, _ = test_net(CLASSnet, X, y,
                                                indexes=train_index,
                                                test_net_label='new neuron testing',
                                                all_errors=all_errors,
                                                # fold_test_accuracy=fold_testing_accuracy,
                                                fold_string=fold_string,
                                                max_fold=maximum_fold_accuracy)
            running_synapse_counts = np.hstack([running_synapse_counts, synapse_counts])
            running_neuron_counts = np.hstack([running_neuron_counts, neuron_counts])

            testing_accuracy, training_classifications, \
            _, _ = test_net(CLASSnet, X, y,
                                               test_net_label='Testing',
                                               indexes=test_index,
                                               all_errors=all_errors,
                                               # fold_test_accuracy=fold_testing_accuracy,
                                               fold_string=fold_string,
                                               max_fold=maximum_fold_accuracy)

            plot_learning_curve(all_errors, fold_testing_accuracy,
                                running_synapse_counts, running_neuron_counts, test_label, save_flag=True)

        epoch_error.append([round(np.mean(all_errors[-len(train_index):]), 4),
                            round(final_accuracy, 4),
                            # round(final_procedural_accuracy, 4),
                            fold_testing_accuracy[-1],
                            round(testing_accuracy, 4),
                            CLASSnet.hidden_neuron_count, CLASSnet.deleted_neuron_count]
                           )
        if len(epoch_error) > 1:
            epoch_error[-1][-1] -= epoch_error[-2][-2]
        if 'mnist' not in test_label:
            epoch_noise_results = []
            for noise_std in noise_tests:
                noise_accuracy, _, _, _ = test_net(CLASSnet, X, y,
                                                      indexes=test_index,
                                                      test_net_label='Testing',
                                                      all_errors=all_errors,
                                                      # fold_test_accuracy=fold_testing_accuracy,
                                                      fold_string=fold_string,
                                                      max_fold=maximum_fold_accuracy,
                                                      noise_stdev=noise_std)
                epoch_noise_results.append(noise_accuracy)
            noise_results.append(epoch_noise_results)

        plot_learning_curve(all_errors, fold_testing_accuracy,
                            running_synapse_counts, running_neuron_counts, test_label, save_flag=True)

        print(test_label)
        for ep, error in enumerate(epoch_error):
            print(ep, error)
        print(test_label)
    print("Finished repeat", repeat)

print("done")




