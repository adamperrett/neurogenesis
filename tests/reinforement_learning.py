import numpy as np
from scipy.special import softmax as sm
from copy import deepcopy
from models.neurogenesis import Network
import random
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt



test = 'pen'
if test == 'pen':
    import gym
    num_outputs = 2
    num_inputs = 4
if 'mnist' in test:
    input_dimensions = [28, 28]
else:
    input_dimensions = None

def test_net(net, max_timesteps, episodes, memory_length=10, test_net_label=''):
    env = gym.make('CartPole-v1')

    # The main program loop
    all_times = []
    for i_episode in range(episodes):
        states = []
        observation = env.reset()
        # Iterating through time steps within an episode
        for t in range(max_timesteps):
            env.render()
            activations = net.convert_inputs_to_activations(observation)
            activations = net.response(activations)
            action = select_binary_action(activations)
            prev_obs = observation
            observation, reward, done, info = env.step(action)
            # Keep a store of the agent's experiences
            states.append([done, action, observation, prev_obs, deepcopy(activations)])
            # epsilon decay
            if done:
                # If the pole has tipped over, end this episode
                # scores_last_timesteps.append(t + 1)
                all_times.append(t+1)
                print('\nEpisode {} ended after {} timesteps'.format(i_episode, t + 1))
                # unless balanced
                if t >= 499:
                    print("WOW")
                else:
                    for r, state in enumerate(states[-memory_length:]):
                        activ = state[4]
                        action = state[1]
                        # error = generate_error(r/len(states), action, activ)
                        error = generate_error(r/memory_length, action, activ)
                        print(r, action, "error = ", error, " for ", activ['out0'], " & ", activ['out1'])
                        net.error_driven_neuro_genesis(activ, error)
                # all_states.append(states[-memory_length:])
                break
    plt.figure()
    plt.title("+ve")
    plt.ylim([0, 500])
    plt.plot([i for i in range(len(all_times))], all_times, 'r')
    ave_err10 = moving_average(all_times, 4)
    plt.plot([i for i in range(len(ave_err10))], ave_err10, 'b')
    plt.show()
    return 0

def generate_error(reward, action, activations):
    # add in neuron activations to error
    error = np.zeros(num_outputs)
    error[action] += reward

    error[0] += activations['out0']
    error[1] += activations['out1']
    # error[0] -= activations['out0']
    # error[1] -= activations['out1']
    return error

def select_binary_action(activations):
    if activations['out0'] > activations['out1']:
        action = 0
        print("0", end='')
    elif activations['out0'] < activations['out1']:
        action = 1
        print("1", end='')
    else:
        # action = np.random.randint(2)
        action = 1
        print("r", end='')
    return action

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
    # if len(epoch_error):
    #     axs[1][1].set_xlim([0, len(epoch_error)])
    #     axs[1][1].set_ylim([0, 1])
    #     axs[1][1].plot([i for i in range(len(epoch_error))], np.array(epoch_error)[:, 1])
    #     ave_err10 = moving_average(np.array(epoch_error)[:, 1], min(2, len(epoch_error)))
    #     axs[1][1].plot([i + 1 for i in range(len(ave_err10))], ave_err10, 'r')
    #     axs[1][1].plot([0, len(epoch_error)], [epoch_error[-1][1], epoch_error[-1][1]], 'g')
    #     axs[1][1].set_title("Epoch test classification")
    figure = plt.gcf()
    figure.set_size_inches(16, 9)
    plt.tight_layout(rect=[0, 0.3, 1, 0.95])
    plt.suptitle(test_label, fontsize=16)
    if save_flag:
        plt.savefig("./plots/{}.png".format(test_label), bbox_inches='tight', dpi=200)
    plt.close()

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
    fixed_hidden_ratio = 0.0
    maximum_total_synapses = 100*10000000
    input_spread = 0
    activity_decay_rate = 1.
    activity_init = 1.
    number_of_seeds = 0

maximum_net_size = int(maximum_total_synapses / maximum_synapses_per_neuron)
old_weight_modifier = 1.01
maturity = 100.
activity_init = 1.0
always_inputs = False
replaying = False
error_type = 'out'
epochs = 20
visualise_rate = 5
np.random.seed(27)
# number_of_seeds = min(number_of_seeds, len(train_labels))
# seed_classes = random.sample([i for i in range(len(train_labels))], number_of_seeds)
test_label = '{} net{}x{}  - {} fixed_h{} - sw{} - ' \
             'at{} - et{} - {}adr{} - inp_{}'.format(error_type,
                                                     maximum_net_size, maximum_synapses_per_neuron,
                                                   test,
                                                   fixed_hidden_ratio,
                                                   sensitivity_width,
                                                   activation_threshold,
                                                   error_threshold,
                                                   activity_init, activity_decay_rate,
                                                   always_inputs
                                                   )


average_windows = [30, 100, 300, 1000, 3000, 10000, 100000]
fold_average_windows = [3, 10, 30, 60, 100, 1000]


CLASSnet = Network(num_outputs, num_inputs,
                   error_threshold=error_threshold,
                   f_width=sensitivity_width,
                   activation_threshold=activation_threshold,
                   maximum_net_size=maximum_net_size,
                   max_hidden_synapses=maximum_synapses_per_neuron,
                   activity_decay_rate=activity_decay_rate,
                   always_inputs=always_inputs,
                   old_weight_modifier=old_weight_modifier,
                   input_dimensions=input_dimensions,
                   input_spread=input_spread,
                   output_synapse_maturity=maturity,
                   fixed_hidden_ratio=fixed_hidden_ratio,
                   activity_init=activity_init,
                   replaying=replaying)

test_net(CLASSnet, 1000, 50)

print("done")




