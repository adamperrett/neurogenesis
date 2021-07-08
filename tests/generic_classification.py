import numpy as np
from scipy.special import softmax as sm
from copy import deepcopy
from models.neurogenesis import Network
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt



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
    retest_rate = 10
    retest_size = 10
elif test == 'mnist':
    from datasets.mnist_csv import *
    num_outputs = 10
    train_labels = mnist_training_labels
    train_feat = mnist_training_data
    test_labels = mnist_testing_labels
    test_feat = mnist_testing_data
    retest_rate = 1000
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
    retest_rate = 10
    retest_size = len(test_set_pimas)
if 'mnist' in test:
    input_dimensions = [28, 28]
else:
    input_dimensions = None
num_inputs = len(train_feat[0])

def test_net(net, data, labels, indexes=None, test_net_label='', classifications=None,
             fold_test_accuracy=None, fold_string='', max_fold=None):
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
        neuron_count = len(activations) - len(features) - num_outputs
        print("Neuron count: ", CLASSnet.hidden_neuron_count, " - ", CLASSnet.deleted_neuron_count, " = ",
              CLASSnet.hidden_neuron_count - CLASSnet.deleted_neuron_count)
        print("Synapse count: ", CLASSnet.synapse_count)
        print(test_label)
        for ep in epoch_error:
            print(ep)
        if label == choice:
            correct_classifications += 1
            if 'esting' not in test_net_label:
                CLASSnet.reinforce_neurons(1.)
                classifications.append(1)
                # net.age_output_synapses(reward=True)
            print("CORRECT CLASS WAS CHOSEN")
            # if error[choice] < -0.5:
            #     net.error_driven_neuro_genesis(activations, error, label)
        else:
            print("INCORRECT CLASS WAS CHOSEN")
            if 'esting' not in test_net_label:
                CLASSnet.reinforce_neurons(-1.)
                classifications.append(0)
                net.error_driven_neuro_genesis(activations, error)#, label)
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
        print("\n")
    # print(incorrect_classes)
    # all_incorrect_classes.append(incorrect_classes)
    # for ep in all_incorrect_classes:
    #     print(len(ep), "-", ep)
    correct_classifications /= train_count
    print('Epoch', epoch, '/', epochs, '\nClassification accuracy: ',
          correct_classifications)
    return correct_classifications, classifications

def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def plot_learning_curve(correct_or_not, fold_test_accuracy, test_label, save_flag=False):
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
    axs[0][1].set_title("fold test classification")
    neuron_count_over_time = []
    neuron_count = number_of_seeds
    for classification in correct_or_not:
        neuron_count_over_time.append(neuron_count)
        if classification == 0:
            neuron_count += 1
    axs[1][0].plot([i for i in range(len(neuron_count_over_time))], neuron_count_over_time)
    axs[1][0].set_title("neuron count")
    if len(epoch_error):
        axs[1][1].set_xlim([0, len(epoch_error)])
        axs[1][1].set_ylim([0, 1])
        axs[1][1].plot([i for i in range(len(epoch_error))], np.array(epoch_error)[:, 1])
        ave_err10 = moving_average(np.array(epoch_error)[:, 1], min(2, len(epoch_error)))
        axs[1][1].plot([i + 1 for i in range(len(ave_err10))], ave_err10, 'r')
        axs[1][1].plot([0, len(epoch_error)], [epoch_error[-1][1], epoch_error[-1][1]], 'g')
        axs[1][1].set_title("Epoch test classification")
    figure = plt.gcf()
    figure.set_size_inches(16, 9)
    plt.tight_layout(rect=[0, 0.3, 1, 0.95])
    plt.suptitle(test_label, fontsize=16)
    if save_flag:
        plt.savefig("./plots/{}.png".format(test_label), bbox_inches='tight', dpi=200)
    plt.close()

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
        choice = -1
    if error_type == 'zero':
        error = np.zeros(num_outputs)
    for output in range(num_outputs):
        error[output] += softmax[output] - one_hot_encoding[output]
        # error[output] = - one_hot_encoding[output]

    # print("Error for test ", test_label, " is ", error)
    # print("output")
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
    activity_init = float(sys.argv[7])
    activity_decay_rate = 1.
    number_of_seeds = int(sys.argv[8])
    fixed_hidden_ratio = float(sys.argv[9])
    print("Variables collected")
    for i in range(9):
        print(sys.argv[i+1])
else:
    sensitivity_width = 0.4
    activation_threshold = 0.0
    error_threshold = 0.0
    maximum_synapses_per_neuron = 100
    fixed_hidden_ratio = 0.5
    maximum_total_synapses = 100*3000
    input_spread = 0
    activity_decay_rate = 1.
    activity_init = 0.
    number_of_seeds = 0

maximum_net_size = int(maximum_total_synapses / maximum_synapses_per_neuron)
old_weight_modifier = 1.01
maturity = 100.
delete_neuron_type = 'old'
reward_decay = 0.5
# activity_init = 1.0
always_inputs = False
replaying = False
error_type = 'out'
epochs = 20
visualise_rate = 5
np.random.seed(27)
# number_of_seeds = min(number_of_seeds, len(train_labels))
# seed_classes = random.sample([i for i in range(len(train_labels))], number_of_seeds)
test_label = 'del avinit {} {}{} net{}x{}  - {} fixed_h{} - sw{} - ' \
             'at{} - et{} - {}adr{}'.format(error_type,
                                            delete_neuron_type, reward_decay,
                                                     maximum_net_size, maximum_synapses_per_neuron,
                                                   test,
                                                   fixed_hidden_ratio,
                                                   sensitivity_width,
                                                   activation_threshold,
                                                   error_threshold,
                                                   activity_init, activity_decay_rate
                                                   )


average_windows = [30, 100, 300, 1000, 3000, 10000, 100000]
fold_average_windows = [3, 10, 30, 60, 100, 1000]


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
                   replaying=replaying)
all_incorrect_classes = []
epoch_error = []

fold_testing_accuracy = []
maximum_fold_accuracy = [[0, 0]]
training_classifications = []
for epoch in range(epochs):
    if epoch % 3 == 0 and epoch:
        for ep, error in enumerate(epoch_error):
            print(ep, error)
        print("it reached 10")
    max_folds = int(len(train_labels) / retest_rate)
    training_count = 0
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
        # training_classifications += new_classifications
        testing_indexes = random.sample([i for i in range(len(test_labels))], retest_size)
        testing_accuracy, training_classifications = test_net(CLASSnet, test_feat, test_labels,
                                                              test_net_label='Testing',
                                                              indexes=testing_indexes,
                                                              classifications=training_classifications,
                                                              fold_test_accuracy=fold_testing_accuracy,
                                                              fold_string=fold_string,
                                                              max_fold=maximum_fold_accuracy)
        fold_testing_accuracy.append(round(testing_accuracy, 3))
        plot_learning_curve(training_classifications, fold_testing_accuracy, test_label, save_flag=True)
        # print("visualising features")
        # if current_fold % visualise_rate == 0 and 'mnist' in test:
        #     for i in range(10):
        #         print("positive visualising class", i)
        #         vis = CLASSnet.visualise_neuron('out{}'.format(i), only_pos=True)
        #         print("plotting class", i)
        #         plt.imshow(vis, cmap='hot', interpolation='nearest', aspect='auto')
        #         print("saving class", i)
        #         plt.savefig("./plots/{}pos {}.png".format(i, test_label), bbox_inches='tight', dpi=20)
        #         print("negative visualising class", i)
        #         vis = CLASSnet.visualise_neuron('out{}'.format(i), only_pos=False)
        #         print("plotting class", i)
        #         plt.imshow(vis, cmap='hot', interpolation='nearest', aspect='auto')
        #         print("saving class", i)
        #         plt.savefig("./plots/{}both {}.png".format(i, test_label), bbox_inches='tight', dpi=20)
        if current_fold % 10 == 0 and current_fold:
            print("it reached 10 folds")
        if testing_accuracy > maximum_fold_accuracy[-1][0] and 'mnist' not in test:
            total_test_accuracy, _ = test_net(CLASSnet, train_feat+test_feat, train_labels+test_labels,
                                              test_net_label='Testing',
                                              classifications=training_classifications,
                                              fold_test_accuracy=fold_testing_accuracy,
                                              fold_string=fold_string,
                                              max_fold=maximum_fold_accuracy
                                              )
            maximum_fold_accuracy.append([testing_accuracy, total_test_accuracy, epoch, current_fold,
                                          CLASSnet.hidden_neuron_count])

    if retest_size < len(test_labels):
        full_testing_accuracy, training_classifications = test_net(CLASSnet, test_feat, test_labels,
                                                                   test_net_label='Testing',
                                                                   classifications=training_classifications,
                                                                   fold_test_accuracy=fold_testing_accuracy,
                                                                   fold_string=fold_string,
                                                                   max_fold=maximum_fold_accuracy)

        epoch_error.append([np.mean(training_classifications[-len(train_labels):]), full_testing_accuracy,
                            CLASSnet.hidden_neuron_count - CLASSnet.deleted_neuron_count])
    else:
        epoch_error.append([np.mean(training_classifications[-len(train_labels):]), testing_accuracy,
                            CLASSnet.hidden_neuron_count - CLASSnet.deleted_neuron_count])

    plot_learning_curve(training_classifications, fold_testing_accuracy, test_label, save_flag=True)

    print(test_label)
    for ep, error in enumerate(epoch_error):
        print(ep, error)
    print(test_label)

print("done")




