import numpy as np
from scipy.special import softmax as sm
from breast_data import *
from copy import deepcopy
from models.neurogenesis import Network



def calculate_error(correct_class, activations, breast_count):
    output_activations = np.zeros(2)
    error = np.zeros(2)
    one_hot_encoding = np.zeros(2)
    one_hot_encoding[correct_class] = 1
    for output in range(2):
        output_activations[output] = activations['out{}'.format(output)]
    # softmax = sm(output_activations)
    softmax = output_activations
    if sum(softmax) > 0.:
        choice = softmax.argmax()
    else:
        choice = -1
    for output in range(2):
        error[output] += softmax[output] - one_hot_encoding[output]

    print("Error for test ", breast_count, " is ", error)
    print("output \n"
          "{} - 1:{} - sm:{}\n"
          "{} - 2:{} - sm:{}".format(one_hot_encoding[0], output_activations[0], softmax[0],
                                     one_hot_encoding[1], output_activations[1], softmax[1]))
    # "{} - 2:{}\n".format(int(label == 0), activations['out0'],
    #                      int(label == 1), activations['out1'],
    #                      int(label == 2), activations['out2']))
    return error, choice


epochs = 200
sensitivity_width = 0.9
error_threshold = 0.01
seed_class = 0
BREASneT = Network(2, breast_labels[seed_class], norm_breast[seed_class],
                   error_threshold=error_threshold,
                   f_width=sensitivity_width)
all_incorrect_classes = []
epoch_error = []

for epoch in range(epochs):
    if epoch == 10:
        for ep, error in enumerate(epoch_error):
            print(ep, error)
        print("it reached 10")
    activations = {}
    breast_count = 0
    correct_classifications = 0
    incorrect_classes = []
    # for breast, label in zip(norm_breast, breast_labels):
    for breast, label in zip(training_set_breasts, training_set_labels):
        activations = BREASneT.convert_inputs_to_activations(breast)
        activations = BREASneT.response(activations)
        print("Epoch ", epoch, "/", epochs)
        error, choice = calculate_error(label, activations, breast_count)
        print("neuron count", len(activations) - len(breast) - 2)
        if label == choice:
            correct_classifications += 1
            print("CORRECT CLASS WAS CHOSEN\n")
        else:
            print("INCORRECT CLASS WAS CHOSEN\n")
            incorrect_classes.append('({}) {}: {}'.format(breast_count, label, choice))
            BREASneT.error_driven_neuro_genesis(activations, error)
        breast_count += 1
    # print(incorrect_classes)
    all_incorrect_classes.append(incorrect_classes)
    for ep in all_incorrect_classes:
        print(len(ep), "-", ep)
    correct_classifications /= breast_count
    print('Epoch', epoch, '/', epochs, '\nClassification accuracy: ',
          correct_classifications)
    breast_count = 0
    test_classifications = 0
    for breast, label in zip(test_set_breasts, test_set_labels):
        activations = BREASneT.convert_inputs_to_activations(breast)
        activations = BREASneT.response(activations)
        print("Test ", breast_count + 1, "/", test_set_size)
        error, choice = calculate_error(label, activations, breast_count)
        if label == choice:
            test_classifications += 1
            print("CORRECT CLASS WAS CHOSEN\n")
        else:
            print("INCORRECT CLASS WAS CHOSEN\n")
            print('({}) {}: {}'.format(breast_count, label, choice))
        breast_count += 1

    print("neuron count", len(activations) - len(breast) - 2)
    print('Epoch', epoch, '/', epochs, '\nClassification accuracy: ',
          correct_classifications)
    print("Test accuracy is ", test_classifications / test_set_size,
          "(", test_classifications, "/", test_set_size, ")")
    epoch_error.append([correct_classifications, test_classifications / test_set_size])






