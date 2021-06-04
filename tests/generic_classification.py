import numpy as np
from scipy.special import softmax as sm
from copy import deepcopy
from models.neurogenesis import Network



def calculate_error(correct_class, activations, test_label, num_outputs=2):
    output_activations = np.zeros(num_outputs)
    error = np.zeros(num_outputs)
    one_hot_encoding = np.zeros(num_outputs)
    one_hot_encoding[correct_class] = 1
    for output in range(num_outputs):
        output_activations[output] = activations['out{}'.format(output)]
    # softmax = sm(output_activations)
    softmax = output_activations
    if sum(softmax) > 0.:
        choice = softmax.argmax()
    else:
        choice = -1
    for output in range(num_outputs):
        error[output] += softmax[output] - one_hot_encoding[output]

    print("Error for test ", test_label, " is ", error)
    print("output")
    for output in range(num_outputs):
        print("{} - {}:{} - sm:{} - err:{}".format(one_hot_encoding[output],
                                                   output,
                                                   output_activations[output],
                                                   softmax[output],
                                                   error[output]))
    return error, choice


epochs = 20
sensitivity_width = 0.6
activation_threshold = 0.0
error_threshold = 0.01
maximum_net_size = 200
seed_class = 0
test = 'rmnist'
test_label = '{}{} - sw{} - at{} - et{}'.format(seed_class, test,
                                                sensitivity_width,
                                                activation_threshold,
                                                error_threshold)
average_windows = [10, 30, 50, 100, 200, 300, 500, 1000]

if test == 'breast':
    from breast_data import *
    num_outputs = 2
    train_labels = training_set_labels
    train_feat = training_set_breasts
    test_labels = test_set_labels
    test_feat = test_set_breasts
elif test == 'wine':
    from wine_data import *
    num_outputs = 3
    train_labels = training_set_labels
    train_feat = training_set_wines
    test_labels = test_set_labels
    test_feat = test_set_wines
elif test == 'mnist':
    from datasets.mnist_csv import *
    num_outputs = 10
    train_labels = mnist_training_labels
    train_feat = mnist_training_data
    test_labels = mnist_testing_labels
    test_feat = mnist_testing_data
elif test == 'rmnist':
    from datasets.mnist_csv import *
    num_outputs = 10
    train_labels = mnist_training_labels
    train_feat = reduced_mnist_training_data
    test_labels = mnist_testing_labels
    test_feat = reduced_mnist_testing_data
num_inputs = len(train_feat[0])

CLASSnet = Network(num_outputs, train_labels[seed_class], train_feat[seed_class],
                   error_threshold=error_threshold,
                   f_width=sensitivity_width,
                   activation_threshold=activation_threshold,
                   maximum_net_size=maximum_net_size)
all_incorrect_classes = []
epoch_error = []

for epoch in range(epochs):
    if epoch == 10:
        for ep, error in enumerate(epoch_error):
            print(ep, error)
        print("it reached 10")
    activations = {}
    train_count = 0
    correct_classifications = 0
    classifications = []
    incorrect_classes = []
    # for breast, label in zip(norm_breast, breast_labels):
    for features, label in zip(train_feat, train_labels):
        activations = CLASSnet.convert_inputs_to_activations(features)
        activations = CLASSnet.response(activations)
        print("Epoch ", epoch, "/", epochs)
        error, choice = calculate_error(label, activations, train_count, num_outputs)
        print("neuron count", len(activations) - len(features) - num_outputs)
        print(test_label)
        for ep in epoch_error:
            print(ep)
        if label == choice:
            correct_classifications += 1
            classifications.append(1)
            print("CORRECT CLASS WAS CHOSEN\n")
        else:
            print("INCORRECT CLASS WAS CHOSEN\n")
            classifications.append(0)
            incorrect_classes.append('({}) {}: {}'.format(train_count, label, choice))
            CLASSnet.error_driven_neuro_genesis(activations, error)
        print("Performance over last tests")
        for window in average_windows:
            print(np.average(classifications[-window:]), ":", window)
        train_count += 1
    # print(incorrect_classes)
    all_incorrect_classes.append(incorrect_classes)
    for ep in all_incorrect_classes:
        print(len(ep), "-", ep)
    correct_classifications /= train_count
    print('Epoch', epoch, '/', epochs, '\nClassification accuracy: ',
          correct_classifications)
    test_count = 0
    test_classifications = 0
    for features, label in zip(test_feat, test_labels):
        test_count += 1
        activations = CLASSnet.convert_inputs_to_activations(features)
        activations = CLASSnet.response(activations)
        print("Test ", test_count, "/", test_count)
        print(test_label)
        for ep in epoch_error:
            print(ep)
        error, choice = calculate_error(label, activations, test_count, num_outputs)
        if label == choice:
            test_classifications += 1
            print("CORRECT CLASS WAS CHOSEN\n")
        else:
            print("INCORRECT CLASS WAS CHOSEN\n")
            print('({}) {}: {}'.format(test_count, label, choice))
        print("Performance over last tests")
        for window in average_windows:
            print(np.average(classifications[-window:]), ":", window)
        print("Current testing accuracy: ", test_classifications / test_count)

    print("neuron count", len(activations) - len(features) - num_outputs)
    print('Epoch', epoch, '/', epochs, '\nClassification accuracy: ',
          correct_classifications)
    print("Test accuracy is ", test_classifications / test_count,
          "(", test_classifications, "/", test_count, ")")
    epoch_error.append([correct_classifications, test_classifications / test_count])
    for ep in epoch_error:
        print(ep)






