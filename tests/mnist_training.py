import numpy as np
from scipy.special import softmax as sm
from copy import deepcopy
from models.neurogenesis import Network
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd


test = 'mnist'
if test == 'breast':
    from breast_data import *
    num_outputs = 2
    train_labels = training_set_labels
    train_feat = training_set_breasts
    test_labels = test_set_labels
    test_feat = test_set_breasts
    retest_rate = 1
    retest_size = len(test_set_labels)
elif test == 'wine':
    from wine_data import *
    num_outputs = 3
    train_labels = training_set_labels
    train_feat = training_set_wines
    test_labels = test_set_labels
    test_feat = test_set_wines
    retest_rate = 10
    retest_size = len(test_set_labels)
elif test == 'mnist':
    from datasets.mnist_csv import *
    num_outputs = 10
    train_labels = mnist_training_labels
    train_feat = mnist_training_data
    test_labels = mnist_testing_labels
    test_feat = mnist_testing_data
    retest_rate = 1000
    retest_size = 100
elif test == 'pima':
    from datasets.pima_indians import *
    num_outputs = 2
    train_labels = training_set_labels
    train_feat = training_set_pimas
    test_labels = test_set_labels
    test_feat = test_set_pimas
    retest_rate = 50
    retest_size = len(test_set_pimas)
if 'mnist' in test:
    input_dimensions = [28, 28]
else:
    input_dimensions = None
num_inputs = len(train_feat[0])

def test_net(net, data, labels, indexes=None, test_net_label='', classifications=None,
             fold_test_accuracy=None, fold_string='', max_fold=None, noise_stdev=0):
    if not indexes:
        indexes = [i for i in range(len(labels))]
    activations = {}
    train_count = 0
    correct_classifications = 0
    confusion_matrix = np.zeros([num_outputs+1, num_outputs+1])
    synapse_counts = []
    neuron_counts = []
    repeated_count = []
    for test in indexes:
        train_count += 1
        features = data[test]
        label = labels[test]
        noise = np.random.normal(scale=noise_stdev, size=np.array(features).shape)
        activations = net.convert_inputs_to_activations(np.array(features) + noise)
        activations = net.response(activations)
        # h_act = net.return_hidden_neurons(activations=activations)
        print(test_net_label, "\nEpoch ", epoch, "/", epochs)
        print(fold_string)
        print('test ', train_count, '/', len(indexes))
        error, choice, softmax = calculate_error(label, activations, test_label, num_outputs)
        neuron_count = CLASSnet.hidden_neuron_count - CLASSnet.deleted_neuron_count
        print("Neuron count: ", CLASSnet.hidden_neuron_count, " - ", CLASSnet.deleted_neuron_count, " = ",
              CLASSnet.hidden_neuron_count - CLASSnet.deleted_neuron_count)
        print("Synapse count: ", CLASSnet.synapse_count)
        print("Repeated neuron count: ", CLASSnet.repeated_neuron_count)
        print(test_label)
        for ep in epoch_error:
            print(ep)
        confusion_matrix[choice][label] += 1
        synapse_counts.append(CLASSnet.synapse_count)
        neuron_counts.append(neuron_count)
        repeated_count.append(CLASSnet.repeated_neuron_count)

        # if train_count % 100 == 0 and 'esti' not in test_net_label:
        #     input = CLASSnet.convert_inputs_to_activations(np.array(features))
        #     act_expectation = CLASSnet.collect_expectation(activations, softmax)
        #     err_expectation = CLASSnet.collect_expectation(activations, softmax,
        #                                                    activity_dependant=False)
        #     general_expectation = CLASSnet.collect_expectation(activations, softmax,
        #                                                        activity_dependant=False,
        #                                                        error_driven=False)
        #     input = np.reshape(CLASSnet.convert_inp_dict_to_list(input), [28, 28])
        #     act_expectation = np.reshape(CLASSnet.convert_inp_dict_to_list(act_expectation), [28, 28])
        #     err_expectation = np.reshape(CLASSnet.convert_inp_dict_to_list(err_expectation), [28, 28])
        #     general_expectation = np.reshape(CLASSnet.convert_inp_dict_to_list(general_expectation), [28, 28])
        #     fig, ax = plt.subplots(2, 2)
        #     im1 = ax[0][0].imshow(input, cmap='viridis')#,
        #                           # extent=x_range + y_range)
        #     im2 = ax[0][1].imshow(act_expectation, cmap='viridis')#,
        #                           # extent=x_range + y_range)
        #     im3 = ax[1][0].imshow(err_expectation, cmap='viridis')#,
        #                           # extent=x_range + y_range)
        #     im4 = ax[1][1].imshow(general_expectation, cmap='viridis')#,
        #                           # extent=x_range + y_range)
        #     cbar1 = fig.colorbar(im1, ax=ax[0][0])
        #     cbar2 = fig.colorbar(im2, ax=ax[0][1])
        #     cbar3 = fig.colorbar(im3, ax=ax[1][0])
        #     cbar4 = fig.colorbar(im4, ax=ax[1][1])
        #     fig.tight_layout()
        #     # plt.show()
        #     plt.savefig("./plots/expectation {} {}.png".format(test_label, len(classifications)),
        #                 bbox_inches='tight', dpi=200, format='png')
        #     plt.close()
        #     print("plotted")

        if expect_type == 'act':
            expectation = CLASSnet.collect_expectation(activations, softmax)
        elif expect_type == 'err':
            expectation = CLASSnet.collect_expectation(activations, softmax,
                                                       activity_dependant=False)
        else:
            expectation = CLASSnet.collect_expectation(activations, softmax,
                                                       activity_dependant=False,
                                                       error_driven=False)
        if label == choice:
            correct_classifications += 1
            if 'esting' not in test_net_label:
                # net.reinforce_neurons(1.)
                classifications.append(1)
                neuron_label = net.error_driven_neuro_genesis(activations, error, expectation,
                                                              label=label)
            print("CORRECT CLASS WAS CHOSEN")
            # if error[choice] < -0.5:
            #     net.error_driven_neuro_genesis(activations, error, label)
        else:
            print("INCORRECT CLASS WAS CHOSEN")
            if 'esting' not in test_net_label:
                # net.reinforce_neurons(-1.)
                classifications.append(0)
                neuron_label = net.error_driven_neuro_genesis(activations, error, expectation,
                                                              label=label)
                # print("total visualising neuron", CLASSnet.hidden_neuron_count)
                # vis = CLASSnet.visualise_neuron(neuron_label, only_pos=False)
                # print("plotting neuron", CLASSnet.hidden_neuron_count)
                # plt.imshow(vis, cmap='hot', interpolation='nearest', aspect='auto')
                # plt.savefig("./plots/{}neuron cl{} {}.png".format(CLASSnet.hidden_neuron_count,
                #                                                   label, test_label),
                #             bbox_inches='tight', dpi=20)
                # if net.hidden_neuron_count > max_out_synapses:
                #     net.remove_worst_output()
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
            print("Performance over last", len(fold_test_accuracy), "folds")
            for window in fold_average_windows:
                print(np.average(fold_test_accuracy[-window:]), ":", window)
        # if procedural_testing_accuracy:
        #     print("Procedural testing accuracy", procedural_testing_accuracy)
        #     print("Performance over last", len(procedural_testing_accuracy), "folds")
        #     for window in fold_average_windows:
        #         print(np.average(procedural_testing_accuracy[-window:]), ":", window)
        print("\n")
    # print(incorrect_classes)
    # all_incorrect_classes.append(incorrect_classes)
    # for ep in all_incorrect_classes:
    #     print(len(ep), "-", ep)
    correct_classifications /= train_count
    print('Epoch', epoch, '/', epochs, '\nClassification accuracy: ',
          correct_classifications)
    return correct_classifications, classifications, confusion_matrix, \
           synapse_counts, neuron_counts, repeated_count

def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def plot_learning_curve(correct_or_not, fold_test_accuracy, training_confusion, testing_confusion,
                        synapse_counts, neuron_counts, repeat_counts,
                        test_label, save_flag=False):
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
    if len(ave_err1000):
        axs[0][1].plot([0, len(correct_or_not)], [ave_err1000[-1], ave_err1000[-1]], 'g')
    axs[0][1].set_xlim([0, len(fold_test_accuracy)])
    axs[0][1].set_ylim([0, 1])
    axs[0][1].set_title("test classification")
    axs[1][0].plot([i for i in range(len(neuron_counts))], neuron_counts)
    axs[1][0].set_title("Neuron count")
    if len(epoch_error):
        if len(epoch_error) <= 10:
            data = np.hstack([np.array(epoch_error)[:, 0].reshape([len(epoch_error), 1]),
                    np.array(epoch_error)[:, 1].reshape([len(epoch_error), 1]),
                    np.array(epoch_error)[:, 2].reshape([len(epoch_error), 1])])#,
                    # np.array(epoch_error)[:, 3].reshape([len(epoch_error), 1])])
            axs[1][1].table(cellText=data, colLabels=['training accuracy',# 'procedural training',
                                                      'testing accuracy', 'neuron count'],#'procedural accuracy'],
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

    # fig, axs = plt.subplots(2, 1)
    # train_df = pd.DataFrame(training_confusion, range(num_outputs+1), range(num_outputs+1))
    # axs[0] = sn.heatmap(train_df, annot=True, annot_kws={'size': 8}, ax=axs[0])
    # axs[0].set_title("Training confusion")
    # test_df = pd.DataFrame(training_confusion, range(num_outputs+1), range(num_outputs+1))
    # axs[1] = sn.heatmap(test_df, annot=True, annot_kws={'size': 8}, ax=axs[1])
    # axs[1].set_title("Testing confusion")
    # figure = plt.gcf()
    # figure.set_size_inches(16, 9)
    # plt.tight_layout(rect=[0, 0.3, 1, 0.95])
    # plt.suptitle(test_label, fontsize=16)
    # if save_flag:
    #     plt.savefig("./plots/confusion {}.png".format(test_label), bbox_inches='tight', dpi=200)
    # plt.close()
    data_dict = {}
    data_dict['training classifications'] = correct_or_not
    data_dict['fold_test_accuracy'] = fold_test_accuracy
    data_dict['training_confusion'] = training_confusion
    data_dict['testing_confusion'] = testing_confusion
    data_dict['synapse_counts'] = synapse_counts
    data_dict['neuron_counts'] = neuron_counts
    data_dict['repeat_counts'] = repeated_counts
    data_dict['epoch error'] = epoch_error
    data_dict['noise_results'] = noise_results
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
        error[output] += softmax[output] - one_hot_encoding[output]
        # error[output] = - one_hot_encoding[output]

    # print("Error for test ", test_label, " is ", error)
    # print("output")
    # if 'esting' not in test_label:
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
    error_threshold = 0.1
    maximum_synapses_per_neuron = 16
    # fixed_hidden_amount = 0
    fixed_hidden_ratio = 0.
    # fixed_hidden_ratio = fixed_hidden_amount / maximum_synapses_per_neuron
    maximum_total_synapses = 100*3000000
    input_spread = 0
    activity_decay_rate = 1.#0.999999
    activity_init = 1.
    number_of_seeds = 0

maximum_net_size = int(maximum_total_synapses / maximum_synapses_per_neuron)
old_weight_modifier = 1.01
maturity = 100.
hidden_threshold = 0.95
delete_neuron_type = 'RL'
reward_decay = 0.99999
conv_size = 9
max_out_synapses = 50000
# activity_init = 1.0
always_inputs = False
replaying = False
expecting = False
expect_type = 'act'
error_type = 'sm'
epochs = 7
repeats = 1
visualise_rate = 10
np.random.seed(27)
confusion_decay = 0.8

noise_tests = np.linspace(0, .3, 21)

# number_of_seeds = min(number_of_seeds, len(train_labels))
# seed_classes = random.sample([i for i in range(len(train_labels))], number_of_seeds)
base_label = 'expecting-{}-{} retest{} {} {}{} net{}x{}  - {} fixed_h{} - sw{} - ' \
             'at{} - et{} - {}adr{}'.format(expecting, expect_type, retest_rate, error_type,
                                            delete_neuron_type, reward_decay,
                                                     maximum_net_size, maximum_synapses_per_neuron,
                                                   test,
                                                   fixed_hidden_ratio,
                                                    # fixed_hidden_amount,
                                                   sensitivity_width,
                                                   activation_threshold,
                                                   error_threshold,
                                                   activity_init, activity_decay_rate
                                                   )

average_windows = [30, 100, 300, 1000, 3000, 10000, 100000]
fold_average_windows = [3, 10, 30, 60, 100, 1000]

for repeat in range(repeats):
    if 'mnist' not in test:
        combined_features = train_feat + test_feat
        combined_labels = train_labels + test_labels

        test_set_indexes = random.sample([i for i in range(len(combined_labels))],
                                         int(0.3 * len(combined_labels)))
        test_feat = [combined_features[i] for i in test_set_indexes]
        test_labels = [combined_labels[i] for i in test_set_indexes]
        test_set_indexes.sort(reverse=True)
        for i in test_set_indexes:
            del combined_features[i]
            del combined_labels[i]
        train_feat = combined_features
        train_labels = combined_labels

    if repeats == 1:
        test_label = base_label
    else:
        test_label = base_label + ' {}'.format(repeat)

    CLASSnet = Network(num_outputs, num_inputs,
                       error_threshold=error_threshold,
                       f_width=sensitivity_width,
                       activation_threshold=activation_threshold,
                       maximum_total_synapses=maximum_total_synapses,
                       max_hidden_synapses=maximum_synapses_per_neuron,
                       activity_decay_rate=activity_decay_rate,
                       always_inputs=always_inputs,
                       old_weight_modifier=old_weight_modifier,
                       input_dimensions=input_dimensions,
                       reward_decay=reward_decay,
                       delete_neuron_type=delete_neuron_type,
                       fixed_hidden_ratio=fixed_hidden_ratio,
                       activity_init=activity_init,
                       replaying=replaying,
                       hidden_threshold=hidden_threshold,
                       conv_size=conv_size,
                       expecting=expecting)
    all_incorrect_classes = []
    epoch_error = []
    noise_results = []
    previous_accuracy = 0
    previous_full_accuracy = 0

    fold_testing_accuracy = [0]
    procedural_testing_accuracy = [0]
    best_testing_accuracy = [0]
    maximum_fold_accuracy = [[0, 0]]
    training_classifications = []
    running_train_confusion = np.zeros([num_outputs+1, num_outputs+1])
    running_test_confusion = np.zeros([num_outputs+1, num_outputs+1])
    running_synapse_counts = np.zeros([1])
    running_neuron_counts = np.zeros([1])
    running_repeated_counts = np.zeros([1])
    for epoch in range(epochs):
        if epoch % 3 == 0 and epoch:
            for ep, error in enumerate(epoch_error):
                print(ep, error)
            print("it reached 10")
        max_folds = int(len(train_labels) / retest_rate)
        training_count = 0
        while training_count < len(train_labels):
            if len(epoch_error) > 0:
                if epoch_error[-1][0] == 1.0:
                    extend_data(len(train_labels))
                    break
            training_indexes = [i for i in range(training_count, min(training_count + retest_rate, len(train_labels)))]
            training_count += retest_rate
            current_fold = training_count / retest_rate
            fold_string = 'fold {} / {}'.format(int(current_fold), max_folds)
            training_accuracy, training_classifications, \
            training_confusion, synapse_counts, neuron_counts, \
            repeated_counts = test_net(CLASSnet, train_feat, train_labels,
                                       indexes=training_indexes,
                                       test_net_label='Training',
                                       fold_test_accuracy=fold_testing_accuracy,
                                       classifications=training_classifications,
                                       fold_string=fold_string,
                                       max_fold=maximum_fold_accuracy)
            running_synapse_counts = np.hstack([running_synapse_counts, synapse_counts])
            running_neuron_counts = np.hstack([running_neuron_counts, neuron_counts])
            running_repeated_counts = np.hstack([running_repeated_counts, repeated_counts])
            # training_classifications += new_classifications
            testing_indexes = random.sample([i for i in range(len(test_labels))], retest_size)
            testing_accuracy, training_classifications, \
            testing_confusion, _, _, _ = test_net(CLASSnet, test_feat, test_labels,
                                                  test_net_label='Testing',
                                                  indexes=testing_indexes,
                                                  classifications=training_classifications,
                                                  fold_test_accuracy=fold_testing_accuracy,
                                                  fold_string=fold_string,
                                                  max_fold=maximum_fold_accuracy)
            fold_testing_accuracy.append(round(testing_accuracy, 3))
            # if training_count < len(train_labels):
            #     CLASSnet.convert_net_and_clean()
            #     procedural_accuracy, training_classifications, \
            #     testing_confusion, _, _ = test_net(CLASSnet, test_feat, test_labels,
            #                                        test_net_label='Testing',
            #                                        indexes=testing_indexes,
            #                                        classifications=training_classifications,
            #                                        fold_test_accuracy=fold_testing_accuracy,
            #                                        fold_string=fold_string,
            #                                        max_fold=maximum_fold_accuracy)
            #     procedural_testing_accuracy.append(round(procedural_accuracy, 3))
            # if previous_accuracy > testing_accuracy + 0.1 and len(fold_testing_accuracy) > 500 and 'mnist' not in test:
            #     test_training_accuracy, training_classifications, \
            #     test_training_confusion, synapse_counts, \
            #     neuron_counts = test_net(CLASSnet, train_feat, train_labels,
            #                              # indexes=training_indexes,
            #                              test_net_label='Train testing',
            #                              fold_test_accuracy=fold_testing_accuracy,
            #                              classifications=training_classifications,
            #                              fold_string=fold_string,
            #                              max_fold=maximum_fold_accuracy)
            #     print("this si where it fucked")
            previous_accuracy = testing_accuracy
            running_train_confusion *= confusion_decay
            running_train_confusion += training_confusion
            running_test_confusion *= confusion_decay
            running_test_confusion += testing_confusion
            plot_learning_curve(training_classifications, fold_testing_accuracy,
                                running_train_confusion, running_test_confusion,
                                running_synapse_counts, running_neuron_counts,
                                running_repeated_counts, test_label, save_flag=True)
            print("visualising features")
            if current_fold % visualise_rate == 0 and 'mnist' in test:
                for i in range(10):
                    print("positive visualising class", i)
                    vis = CLASSnet.visualise_neuron('out{}'.format(i), only_pos=True)
                    print("plotting class", i)
                    plt.imshow(vis, cmap='hot', interpolation='nearest', aspect='auto')
                    print("saving class", i)
                    plt.savefig("./plots/{}pos {}.png".format(i, test_label), bbox_inches='tight', dpi=20)
                    print("negative visualising class", i)
                    vis = CLASSnet.visualise_neuron('out{}'.format(i), only_pos=False)
                    print("plotting class", i)
                    plt.imshow(vis, cmap='hot', interpolation='nearest', aspect='auto')
                    print("saving class", i)
                    plt.savefig("./plots/{}both {}.png".format(i, test_label), bbox_inches='tight', dpi=20)
            if current_fold % 10 == 0 and current_fold:
                print("it reached 10 folds")

        if retest_size < len(test_labels):
            full_testing_accuracy, training_classifications, \
            full_test_confusion, _, _, _ = test_net(CLASSnet, test_feat, test_labels,
                                                    test_net_label='Testing',
                                                    classifications=training_classifications,
                                                    fold_test_accuracy=fold_testing_accuracy,
                                                    fold_string=fold_string,
                                                    max_fold=maximum_fold_accuracy)
            running_test_confusion *= confusion_decay
            running_test_confusion += full_test_confusion
            # CLASSnet.convert_net_and_clean()
            # full_procedural_accuracy, training_classifications, \
            # full_test_confusion, _, _ = test_net(CLASSnet, test_feat, test_labels,
            #                                      test_net_label='Testing',
            #                                      classifications=training_classifications,
            #                                      fold_test_accuracy=fold_testing_accuracy,
            #                                      fold_string=fold_string,
            #                                      max_fold=maximum_fold_accuracy)
            epoch_error.append([np.mean(training_classifications[-len(train_labels):]), full_testing_accuracy,
                                running_neuron_counts[-1]])
                                # CLASSnet.hidden_neuron_count - CLASSnet.deleted_neuron_count])
        else:
            epoch_error.append([np.mean(training_classifications[-len(train_labels):]), testing_accuracy,
                                CLASSnet.hidden_neuron_count - CLASSnet.deleted_neuron_count])
            running_test_confusion *= confusion_decay
            running_test_confusion += testing_confusion

        plot_learning_curve(training_classifications, fold_testing_accuracy,
                            running_train_confusion, running_test_confusion,
                            running_synapse_counts, running_neuron_counts,
                            running_repeated_counts, test_label, save_flag=True)

        print(test_label)
        for ep, error in enumerate(epoch_error):
            print(ep, error)
        print(test_label)
    print("Finished repeat", repeat)

print("done")




