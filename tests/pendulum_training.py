import numpy as np
from scipy.special import softmax as sm
import gym
from copy import deepcopy
from models.neurogenesis import Network


def calculate_error(correct_class, activations, wine_count):
    output_activations = np.zeros(3)
    error = np.zeros(3)
    one_hot_encoding = np.zeros(3)
    one_hot_encoding[correct_class] = 1
    for output in range(3):
        output_activations[output] = activations['out{}'.format(output)]
    # softmax = sm(output_activations)
    softmax = output_activations
    if sum(softmax) > 0.:
        choice = softmax.argmax()
    else:
        choice = -1
    for output in range(3):
        error[output] += softmax[output] - one_hot_encoding[output]

    print("Error for test ", wine_count, " is ", error)
    print("output \n"
          "{} - 1:{} - sm:{}\n"
          "{} - 2:{} - sm:{}\n"
          "{} - 3:{} - sm:{}".format(one_hot_encoding[0], output_activations[0], softmax[0],
                                       one_hot_encoding[1], output_activations[1], softmax[1],
                                       one_hot_encoding[2], output_activations[2], softmax[2]))
          # "{} - 3:{}\n".format(int(label == 0), activations['out0'],
          #                      int(label == 1), activations['out1'],
          #                      int(label == 2), activations['out2']))
    return error, choice



env = gym.make('CartPole-v1')
input_size = env.observation_space.shape[0]
output_size = env.action_space.n
MAX_TIMESTEPS = 500
test_run = []
observation = env.reset()
for t in range(MAX_TIMESTEPS):
    action = np.random.randint(output_size)
    prev_observation = observation
    observation, reward, done, info = env.step()
    # Keep a store of the agent's experiences
    test_run.append([done, action, prev_observation, observation])
    if done:
        # If the pole has tipped over, end this episode
        final_reward = reward
        break

epochs = 200
sensitivity_width = 0.9
error_threshold = 0.01
seed_class = 0
WINEt = Network(2, wine_labels[seed_class], norm_wine[seed_class],
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
    wine_count = 0
    correct_classifications = 0
    incorrect_classes = []
    # for wine, label in zip(norm_wine, wine_labels):
    for wine, label in zip(training_set_wines, training_set_labels):
        activations = WINEt.convert_inputs_to_activations(wine)
        activations = WINEt.response(activations)
        print("Epoch ", epoch, "/", epochs)
        error, choice = calculate_error(label, activations, wine_count)
        print("neuron count", len(activations) - len(wine) - 2)
        if label == choice:
            correct_classifications += 1
            print("CORRECT CLASS WAS CHOSEN\n")
        else:
            print("INCORRECT CLASS WAS CHOSEN\n")
            incorrect_classes.append('({}) {}: {}'.format(wine_count, label, choice))
            WINEt.error_driven_neuro_genesis(activations, error)
        wine_count += 1
    # print(incorrect_classes)
    all_incorrect_classes.append(incorrect_classes)
    for ep in all_incorrect_classes:
        print(len(ep), "-", ep)
    correct_classifications /= wine_count
    print('Epoch', epoch, '/', epochs, '\nClassification accuracy: ',
          correct_classifications)
    wine_count = 0
    test_classifications = 0
    for wine, label in zip(test_set_wines, test_set_labels):
        activations = WINEt.convert_inputs_to_activations(wine)
        activations = WINEt.response(activations)
        print("Test ", wine_count + 1, "/", test_set_size)
        error, choice = calculate_error(label, activations, wine_count)
        if label == choice:
            test_classifications += 1
            print("CORRECT CLASS WAS CHOSEN\n")
        else:
            print("INCORRECT CLASS WAS CHOSEN\n")
            print('({}) {}: {}'.format(wine_count, label, choice))
        wine_count += 1

    print("neuron count", len(activations) - len(wine) - 2)
    print('Epoch', epoch, '/', epochs, '\nClassification accuracy: ',
          correct_classifications)
    print("Test accuracy is ", test_classifications / test_set_size,
          "(", test_classifications, "/", test_set_size, ")")
    epoch_error.append([correct_classifications, test_classifications / test_set_size])






