import numpy as np
from scipy.special import softmax as sm
from copy import deepcopy
from models.neurogenesis import Network
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt



test = 'pen'
if test == 'pen':
    import gym
    num_outputs = 2
    num_inputs = 4
if test == 'nov':
    import gym
    num_outputs = 2
    num_inputs = 2
elif test == 'rf':
    import gym
    num_outputs = 2
    fields_per_inp = 20
    field_width = 0.6
    num_inputs = 4 * fields_per_inp
if 'mnist' in test:
    input_dimensions = [28, 28]
else:
    input_dimensions = None

def test_net(net, max_timesteps, episodes, memory_length=10, test_net_label='', repeat=None, pole_length=0.5):
    env = gym.make('CartPole-v1', length=pole_length)

    # The main program loop
    all_times = []
    for i_episode in range(episodes):
        states = []
        observation = env.reset()
        # Iterating through time steps within an episode
        for t in range(max_timesteps):
            # env.render()
            prev_obs = observation
            if test == 'rf':
                observation = convert_observation_to_fields(observation)
            elif test == 'nov':
                observation = [observation[0], observation[2]]
            activations = net.convert_inputs_to_activations(observation)
            activations = net.response(activations)
            action = select_binary_action(activations)
            observation, reward, done, info = env.step(action)
            # Keep a store of the agent's experiences
            states.append([done, action, observation, prev_obs, deepcopy(activations)])
            # CLASSnet.reinforce_synapses(t+1)
            CLASSnet.reinforce_neurons(1.)
            # epsilon decay
            if done:
                # If the pole has tipped over, end this episode
                # scores_last_timesteps.append(t + 1)
                all_times.append(t+1)
                print('\nEpisode {} of repeat {} ended after {} timesteps - pl{} {}'.format(i_episode, repeat,
                                                                                       t + 1,
                                                                                            pole_length,
                                                                                            test_label))
                print("Neuron count: ", CLASSnet.hidden_neuron_count, " - ", CLASSnet.deleted_neuron_count, " = ",
                      CLASSnet.hidden_neuron_count - CLASSnet.deleted_neuron_count)
                print("Synapse count: ", CLASSnet.synapse_count)
                if len(all_times) > 100 and np.average(all_times[-100:]) > 475:
                    while len(all_times) < episodes:
                        all_times.append(500)
                    print("SUPER WOW")
                    return all_times
                # unless balanced
                if t >= 499:
                    print("WOW")
                else:
                    if 'exp' in error_type:
                        for r, state in reversed(list(enumerate(states))):
                            activ = state[4]
                            action = state[1]
                            error = generate_error(r+1, action, activ, memory_length, len(states))
                            if np.max(np.abs(error)) > error_threshold:
                                # error = generate_error(r/memory_length, action, activ)
                                print(r, action, "error = ", error, " for ", activ['out0'], " & ", activ['out1'])
                                net.error_driven_neuro_genesis(activ, error)
                    else:
                        for r, state in enumerate(states[-memory_length:]):
                            activ = state[4]
                            action = state[1]
                            error = generate_error(r+1, action, activ, memory_length, len(states))
                            # error = generate_error(r/memory_length, action, activ)
                            print(r, action, "error = ", error, " for ", activ['out0'], " & ", activ['out1'])
                            net.neuron_response = activ
                            net.error_driven_neuro_genesis(activ, error)
                            # plot_activations(activ, r)
                # all_states.append(states[-memory_length:])
                break
    return all_times

def plot_activations(activations, reward):
    extractivations = []
    for neuron in activations:
        if 'in' not in neuron and 'out' not in neuron:
            extractivations.append(activations[neuron])
    act = extractivations
    print("mean", np.mean(act))
    print("stdev", np.std(act))
    plt.figure()
    plt.scatter([i for i in range(len(act))], act)
    plt.plot([0, len(act)], [np.mean(act), np.mean(act)], 'g')
    plt.plot([0, len(act)], [np.mean(act) + np.std(act), np.mean(act) + np.std(act)], 'r')
    plt.plot([0, len(act)], [np.mean(act) - np.std(act), np.mean(act) - np.std(act)], 'r')
    plt.plot([0, len(act)], [np.mean(act) + (2 * np.std(act)), np.mean(act) + (2 * np.std(act))], 'r')
    plt.plot([0, len(act)], [np.mean(act) - (2 * np.std(act)), np.mean(act) - (2 * np.std(act))], 'r')
    plt.plot([0, len(act)], [np.mean(act) + (3 * np.std(act)), np.mean(act) + (3 * np.std(act))], 'r')
    plt.plot([0, len(act)], [np.mean(act) - (3 * np.std(act)), np.mean(act) - (3 * np.std(act))], 'r')
    plt.ylim([-0.5, 1.5])
    plt.savefig("./plots/activations {} {}.png".format(test_label, reward), bbox_inches='tight', dpi=200)
    plt.close()
    return extractivations

def convert_observation_to_fields(observation):
    new_observations = []
    for obv in observation:
        for val in np.linspace(-1, 1, fields_per_inp):
            new_obv = max(1. - abs((val - obv) / field_width), 0)
            new_observations.append(new_obv)
    return new_observations

def generate_error(reward, action, activations, memory_length, test_duration):
    # add in neuron activations to error
    error = np.zeros(num_outputs)
    if error_type == 'mem':
        error[action] += reward / memory_length
    elif error_type == 'len':
        error[action] += reward
    elif 'exp' in error_type:
        error[action] = np.power(error_decay_rate, test_duration - reward)
    if 'len' in error_type:
        error /= test_duration

    error[0] += activations['out0']
    error[1] += activations['out1']
    # error[0] -= activations['out0']
    # error[1] -= activations['out1']
    return error

def select_binary_action(activations):
    if activations['out0'] > activations['out1']:
        action = 0
        # print("0", end='')
    elif activations['out0'] < activations['out1']:
        action = 1
        # print("1", end='')
    else:
        action = np.random.randint(2)
        # action = 1
        # print("r", end='')
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
    sensitivity_width = 0.6
    activation_threshold = 0.0
    error_threshold = 0.0
    maximum_synapses_per_neuron = 10
    fixed_hidden_ratio = 0.0
    maximum_total_synapses = 250*10
    input_spread = 0
    activity_decay_rate = 1.
    activity_init = 1.
    number_of_seeds = 0

maximum_net_size = int(maximum_total_synapses / maximum_synapses_per_neuron)
old_weight_modifier = 1.01
maturity = 100.
delete_neuron_type = 'RL'
reward_decay = 0.9999

long_length = 0.5
short_length = 0.25

# activity_init = 1.0
always_inputs = False
replaying = False
error_type = 'mem'
error_decay_rate = 0.
window_size = 10
number_of_episodes = 400
repeat_test = 20
epochs = 20
visualise_rate = 5
np.random.seed(27)
# number_of_seeds = min(number_of_seeds, len(train_labels))
# seed_classes = random.sample([i for i in range(len(train_labels))], number_of_seeds)
test_label = 'trans random long{} w{} {}{} {}{} net{}x{}  - {} fixed_h{} - sw{} - ' \
             'at{} - et{} - {}adr{}'.format(number_of_episodes,
                                            window_size, error_type, error_decay_rate,
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


all_times = []
short_times = []
for repeat in range(repeat_test):
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


    times = test_net(CLASSnet, 1000, number_of_episodes,
                     test_net_label=test_label,
                     memory_length=window_size,
                     repeat=repeat,
                     pole_length=long_length)
    all_times.append(times)
    np.save("./data/pl{} - {}.png".format(long_length, test_label), all_times)

    # all_times = np.array(all_times)
    max_time = []
    min_time = []
    ave_time = []
    std_err = []
    for i in range(len(all_times[0])):
        test_scores = []
        for j in range(len(all_times)):
            test_scores.append(all_times[j][i])
        max_time.append(max(test_scores))
        min_time.append(min(test_scores))
        ave_time.append(np.average(test_scores))
        std_err.append(np.std(test_scores))
    plt.figure()
    for j in range(len(all_times)):
        plt.scatter([i for i in range(len(all_times[j]))], all_times[j], s=4)
    # plt.title(test_label)
    plt.ylim([-10, 510])
    plt.plot([i for i in range(len(max_time))], max_time, 'r')
    plt.plot([i for i in range(len(min_time))], min_time, 'r')
    plt.plot([i for i in range(len(ave_time))], ave_time, 'b')
    plt.plot([i for i in range(len(std_err))], (np.array(std_err)*(1/len(all_times))) + np.array(ave_time), 'g')
    plt.plot([i for i in range(len(std_err))], (-1*np.array(std_err)*(1/len(all_times))) + np.array(ave_time), 'g')
    plt.plot([0, len(ave_time)], [475, 475], 'g')
    for j in range(len(all_times)):
        ave_err100 = moving_average(all_times[j], 100)
        plt.plot([i + 50 for i in range(len(ave_err100))], ave_err100, 'r')
    figure = plt.gcf()
    figure.set_size_inches(16, 9)
    plt.tight_layout(rect=[0, 0.3, 1, 0.95])
    plt.suptitle(test_label, fontsize=16)
    plt.savefig("./plots/pl{} - {}.png".format(long_length, test_label), bbox_inches='tight', dpi=200)
    # plt.show()
    plt.close()

    times = test_net(CLASSnet, 1000, number_of_episodes,
                     test_net_label=test_label,
                     memory_length=window_size,
                     repeat=repeat,
                     pole_length=short_length)
    short_times.append(times)
    np.save("./data/pl{}after{} - {}.png".format(short_length, long_length, test_label), short_times)

    # all_times = np.array(all_times)
    max_time = []
    min_time = []
    ave_time = []
    std_err = []
    for i in range(len(short_times[0])):
        test_scores = []
        for j in range(len(short_times)):
            test_scores.append(short_times[j][i])
        max_time.append(max(test_scores))
        min_time.append(min(test_scores))
        ave_time.append(np.average(test_scores))
        std_err.append(np.std(test_scores))
    plt.figure()
    for j in range(len(short_times)):
        plt.scatter([i for i in range(len(short_times[j]))], short_times[j], s=4)
    # plt.title(test_label)
    plt.ylim([-10, 510])
    plt.plot([i for i in range(len(max_time))], max_time, 'r')
    plt.plot([i for i in range(len(min_time))], min_time, 'r')
    plt.plot([i for i in range(len(ave_time))], ave_time, 'b')
    plt.plot([i for i in range(len(std_err))], (np.array(std_err)*(1/len(short_times))) + np.array(ave_time), 'g')
    plt.plot([i for i in range(len(std_err))], (-1*np.array(std_err)*(1/len(short_times))) + np.array(ave_time), 'g')
    plt.plot([0, len(ave_time)], [475, 475], 'g')
    for j in range(len(short_times)):
        ave_err100 = moving_average(short_times[j], 100)
        plt.plot([i + 50 for i in range(len(ave_err100))], ave_err100, 'r')
    figure = plt.gcf()
    figure.set_size_inches(16, 9)
    plt.tight_layout(rect=[0, 0.3, 1, 0.95])
    plt.suptitle(test_label, fontsize=16)
    plt.savefig("./plots/pl{}after{} - {}.png".format(short_length, long_length, test_label), bbox_inches='tight', dpi=200)
    # plt.show()
    plt.close()
print("done")




