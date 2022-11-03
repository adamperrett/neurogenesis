# multi-class classification with Keras
import pandas
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GaussianNoise
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras import utils as np_utils
from tensorflow.keras import backend as K
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedShuffleSplit, LeaveOneOut, StratifiedKFold, ShuffleSplit
from tensorflow.python.keras.callbacks import LambdaCallback
import numpy as np

# load dataset
test = 'noise'
if test == 'breast':
    from breast_data import *
    num_outputs = 2
    train_labels = training_set_labels
    train_feat = training_set_breasts
    num_inputs = len(train_feat[0])
    test_labels = test_set_labels
    test_feat = test_set_breasts
    retest_rate = 1
    retest_size = len(test_set_labels)
    X = np.array(train_feat + test_feat)
    Y = np.array(train_labels + test_labels)
    # X = pandas.DataFrame(train_feat + test_feat)
    # Y = pandas.DataFrame(train_labels + test_labels)
elif test == 'wine':
    from wine_data import *
    num_inputs = 13
    num_outputs = 3
    train_labels = training_set_labels
    train_feat = training_set_wines
    test_labels = test_set_labels
    test_feat = test_set_wines
    X = np.array(train_feat + test_feat)
    Y = np.array(train_labels + test_labels)
    # X = pandas.DataFrame(train_feat + test_feat)
    # Y = pandas.DataFrame(train_labels + test_labels)
elif test == 'pp_mnist':
    from datasets.preprocessed_mnist import *
    num_outputs = 10
    train_labels = mnist_training_labels
    train_feat = mnist_training_data
    test_labels = mnist_testing_labels
    test_feat = mnist_testing_data
    num_inputs = len(train_feat[0])
    X = np.array(train_feat.tolist() + test_feat.tolist())
    Y = np.array(train_labels + test_labels)
elif test == 'mnist':
    from datasets.mnist_csv import *
    num_inputs = 28 * 28
    num_outputs = 10
    train_labels = mnist_training_labels
    train_feat = mnist_training_data
    test_labels = mnist_testing_labels
    test_feat = mnist_testing_data
    X = np.array(train_feat + test_feat)
    Y = np.array(train_labels + test_labels)
elif test == 'mpg':
    from datasets.mpg_regression import norm_features, norm_mpg, min_mpg, max_mpg
    num_inputs = len(norm_features[0])
    num_outputs = 1
    retest_rate = 1
    retest_size = int(0.1 * len(norm_mpg))
    X = np.array(norm_features)
    Y = np.array(norm_mpg)
elif test == 'noise':
    from datasets.high_noise_inputs import *
    num_inputs = 10000
    num_outputs = 3
    examples = 3000
    test += ' i{}o{}e{}'.format(num_inputs, num_outputs, examples)
    train_feat, train_labels = generate_date(num_inputs=num_inputs,
                                             num_outputs=num_outputs,
                                             examples=examples)
    X = np.array(train_feat)
    Y = np.array(train_labels)
    retest_rate = 1
    retest_size = examples / 10
else:
    dataframe = pandas.read_csv("../datasets/iris.data", header=None)
    dataset = dataframe.values
    X = dataset[:,0:4].astype(float)
    Y = dataset[:,4]
    num_inputs = 4
    num_outputs = 3
# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)
if test == 'mpg':
    dummy_y = Y

# define baseline model
def baseline_model(n_neurons, lr):
    # create model
    model = Sequential()
    # model.add(GaussianNoise(stddev=k_stdev))
    model.add(Dense(n_neurons, input_dim=num_inputs, activation='relu'))
    # model.add(Dense(n_neurons, input_dim=num_inputs, activation='relu'))
    if test == 'mpg':
        model.add(Dense(num_outputs))
        # model.add(Dense(num_outputs, activation='sigmoid'))
        # model.add(Dense(num_outputs, activation='linear'))
    else:
        model.add(Dense(num_outputs, activation='softmax'))
    # Compile model
    optimizer = tf.keras.optimizers.Adam(lr=lr)
    if test == 'mpg':
        loss = 'mean_squared_error'
    else:
        loss = 'categorical_crossentropy'
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    return model

num_neurons = 128*1
learning_rate = 0.03
batch_size = 16
epochs = 20
# noise_tests = np.linspace(0, 2., 21)
# k_stdev = K.variable(value=0.0)

splits = 10
if 'mnist' in test:
    splits = 1
    epochs = 20
testing_data = [[] for i in range(splits)]
training_data = [{} for i in range(splits)]
# all_noise_results = []
if 'mnist' in test:
    splits += 1

test_label = 'bp testing {} n{} lr{} b{}'.format(test, num_neurons, learning_rate, batch_size)

if test == 'mpg':
    sss = ShuffleSplit(n_splits=splits, test_size=0.1, random_state=2727)
else:
    sss = StratifiedKFold(n_splits=splits, random_state=2727, shuffle=True)
for repeat, (train_index, test_index) in enumerate(sss.split(X, Y)):
    net = baseline_model(num_neurons, learning_rate)
    if test == 'mpg':
        retest_callback = LambdaCallback(
            on_batch_end=lambda batch, logs:
            testing_data[repeat].append(
                np.average([np.square(a - b) for a, b in zip(net.predict(X[test_index]), Y[test_index])])))
    elif 'mnist' in test:
        train_index = [i for i in range(60000)]
        test_index = [i+60000 for i in range(10000)]
        # retest_callback = LambdaCallback(
        #     on_batch_end=lambda batch, logs:
        #     testing_data[repeat].append([batch, logs]))
        retest_callback = LambdaCallback(
            on_batch_end=lambda batch, logs:
            testing_data[repeat].append(
                np.average([np.argmax(a) == b for a, b in zip(net.predict(X[test_index]), Y[test_index])])))
    else:
        retest_callback = LambdaCallback(
            on_batch_end=lambda batch, logs:
            testing_data[repeat].append(
                np.average([np.argmax(a) == b for a, b in zip(net.predict(X[test_index]), Y[test_index])])))

    print("training model for repeat", repeat, test_label)
    history = net.fit(X[train_index], dummy_y[train_index],
                      batch_size=batch_size, validation_data=(X[test_index], dummy_y[test_index]),
                      callbacks=retest_callback,
                      epochs=epochs, verbose=True)

    print(test_label)
    for k in history.history.keys():
        training_data[repeat][k] = history.history[k]
        if 'acc' in k:
            print(k, history.history[k])

    if 'mnist' in test:
        break
    # scce2 = tf.keras.losses.sparse_categorical_crossentropy(dummy_y[test_index], net.predict(X))

data_dict = {}

ave_test = []
if test != 'not mnist':
    for j in range(len(testing_data[0])):
        total = 0.
        for i in range(len(testing_data)):
            total += testing_data[i][j]
        ave_test.append(total / len(testing_data))
else:
    correct_or_not = []
    for i in range(len(testing_data[0])):
        batch = testing_data[0][i][0]
        accuracy = testing_data[0][i][1]['accuracy']
        n_correct = int(np.round(batch * batch_size * accuracy))
        if len(ave_test):
            batch_correct = n_correct - sum(correct_or_not)
            correct_or_not.append(batch_correct)
            batch_accuracy = batch_correct / batch_size
            ave_test.append(batch_accuracy)
        else:
            correct_or_not.append(n_correct)
            ave_test.append(accuracy)
    data_dict['correct_or_not'] = correct_or_not

print(ave_test[:20])
print(ave_test[-20:])

data_dict['testing_data'] = testing_data
data_dict['ave_test'] = ave_test
data_dict['training_data'] = training_data

print('Averaged:')
ave_train = {}
for k in training_data[0]:
    total = np.zeros(epochs)
    for i in range(epochs):
        for r in range(len(training_data)):
            total[i] += training_data[r][k][i]
    ave_train[k] = total / len(training_data)
    print(k, ave_train[k])

print(test_label)
np.save("./data/{}".format(test_label), data_dict)


print('Done')

'''
import matplotlib.pyplot as plt
plt.figure()
plt.plot([i*batch_size for i in range(len(ave_test))], ave_test, 'r')
edn = [0.0, 0.32649999999999996, 0.3264, 0.32639999999999997, 0.3297, 0.3264, 0.3297, 0.3332, 0.32639999999999997, 0.32300000000000006, 0.32639999999999997, 0.33640000000000003, 0.3297, 0.3365, 0.3398, 0.3365, 0.3298, 0.3465, 0.3365, 0.3332, 0.3598, 0.3497, 0.3632, 0.3331, 0.3398, 0.3498, 0.3365, 0.36329999999999996, 0.33659999999999995, 0.3532, 0.35650000000000004, 0.3364000000000001, 0.35650000000000004, 0.32970000000000005, 0.34650000000000003, 0.35329999999999995, 0.3264000000000001, 0.3364000000000001, 0.35309999999999997, 0.34320000000000006, 0.38649999999999995, 0.3666, 0.3765, 0.3631, 0.3865, 0.3632, 0.3732, 0.34320000000000006, 0.3633, 0.34650000000000003, 0.34320000000000006, 0.3799, 0.3798, 0.3465, 0.38980000000000004, 0.3498, 0.37639999999999996, 0.35660000000000003, 0.37, 0.3866, 0.3665, 0.38330000000000003, 0.3599, 0.35660000000000003, 0.37, 0.3866, 0.3398, 0.3666, 0.33649999999999997, 0.3433, 0.3699, 0.3833, 0.3866, 0.39990000000000003, 0.36650000000000005, 0.3498, 0.3831000000000001, 0.3899, 0.33980000000000005, 0.42000000000000004, 0.4333, 0.3899, 0.39990000000000003, 0.3799, 0.3699, 0.42989999999999995, 0.3831, 0.39, 0.4, 0.3834, 0.3999, 0.4365, 0.38669999999999993, 0.3798, 0.3598, 0.3866, 0.4231999999999999, 0.3465999999999999, 0.44659999999999994, 0.3898999999999999, 0.38989999999999997, 0.34659999999999996, 0.3901, 0.42329999999999995, 0.3832, 0.4401, 0.4434, 0.3966, 0.4133, 0.3665, 0.3799, 0.3833, 0.4133, 0.41319999999999996, 0.40669999999999995, 0.43, 0.4866999999999999, 0.4267999999999999, 0.39659999999999995, 0.4601, 0.4, 0.4266, 0.4467, 0.38999999999999996, 0.4699, 0.4332999999999999, 0.44659999999999994, 0.3765, 0.37, 0.4366, 0.4633, 0.4399, 0.4032, 0.4332999999999999, 0.4766, 0.4966000000000001, 0.4833, 0.45669999999999994, 0.49000000000000005, 0.45, 0.4133, 0.4767, 0.5465, 0.4867, 0.48660000000000003, 0.4567, 0.47990000000000005, 0.4932, 0.4667, 0.47350000000000003, 0.44329999999999997, 0.4567, 0.4633999999999999, 0.49319999999999997, 0.5133, 0.5001, 0.4199, 0.49000000000000005, 0.5134, 0.4833, 0.5334000000000001, 0.4601, 0.4868, 0.5065999999999999, 0.4734, 0.4699999999999999, 0.4332, 0.49330000000000007, 0.5001, 0.47990000000000005, 0.48660000000000003, 0.4434, 0.44329999999999997, 0.4434, 0.5101, 0.4932, 0.4833, 0.5066999999999999, 0.51, 0.4967, 0.48690000000000005, 0.47329999999999994, 0.49659999999999993, 0.4801, 0.49670000000000003, 0.51, 0.45330000000000004, 0.53, 0.4566, 0.4331999999999999, 0.49000000000000005, 0.4934, 0.4366, 0.4633, 0.5367, 0.5631999999999999, 0.48, 0.5367, 0.4898999999999999, 0.4967, 0.5301000000000001, 0.5067000000000002, 0.5699, 0.5034, 0.5601, 0.43339999999999995, 0.4598000000000001, 0.47990000000000005, 0.46319999999999995, 0.5367999999999999, 0.5732, 0.5334, 0.55, 0.5033999999999998, 0.53, 0.5032, 0.4231999999999999, 0.43660000000000004, 0.5033, 0.5067, 0.5333000000000001, 0.42990000000000006, 0.4965, 0.4633, 0.5233, 0.51, 0.5531, 0.5332999999999999, 0.5999, 0.5599999999999999, 0.55, 0.49990000000000007, 0.5301, 0.5567, 0.5167, 0.5132, 0.5168, 0.5465000000000001, 0.55, 0.5366, 0.5534000000000001, 0.5432, 0.49660000000000004, 0.5166000000000001, 0.5065999999999999, 0.5666, 0.5034, 0.5034000000000001, 0.49989999999999996, 0.4566, 0.45339999999999997, 0.4801, 0.49349999999999994, 0.5, 0.5567, 0.5334, 0.47330000000000005, 0.4898999999999999, 0.4600000000000001, 0.4899, 0.5267000000000001, 0.5565, 0.5234, 0.5700999999999999, 0.5700999999999999, 0.5433, 0.5334, 0.5367000000000001, 0.4766, 0.5133, 0.5134000000000001, 0.5233000000000001, 0.5033, 0.5399999999999999, 0.5599, 0.5833999999999999, 0.6002000000000001, 0.6068, 0.52, 0.5599999999999999, 0.6164999999999999, 0.6465, 0.6201000000000001, 0.6633000000000001, 0.5868, 0.6198, 0.6368, 0.6335, 0.6367, 0.6199, 0.6098999999999999, 0.6333999999999999, 0.6199000000000001, 0.5900999999999998, 0.5934000000000001, 0.5433, 0.5768, 0.5932999999999999, 0.6434000000000001, 0.6401000000000001, 0.6000000000000001, 0.5666000000000001, 0.5700999999999999, 0.5766000000000001, 0.6099, 0.5466, 0.5667000000000001, 0.5968000000000001, 0.5767999999999999, 0.6, 0.6, 0.5867, 0.59, 0.5901, 0.5866, 0.5868, 0.63, 0.6233, 0.6167, 0.6433, 0.6600999999999999, 0.6435, 0.6467, 0.6466999999999999, 0.6432, 0.6267, 0.5733, 0.5765, 0.5633, 0.5932999999999999, 0.5867, 0.5899, 0.5967, 0.5833, 0.5900999999999998, 0.6232, 0.6001, 0.6068, 0.5632, 0.5533999999999999, 0.5934999999999999, 0.5932999999999999, 0.5567, 0.6068, 0.6333000000000001, 0.6301, 0.6033999999999999, 0.5767, 0.5901000000000001, 0.6300999999999999, 0.6266, 0.5999000000000001, 0.5899000000000001, 0.5699000000000001, 0.62, 0.5867, 0.6267, 0.6333, 0.6500000000000001, 0.6733, 0.6365999999999999, 0.6567, 0.6699999999999999, 0.6698000000000001, 0.6466999999999999, 0.6865, 0.6601000000000001, 0.6632999999999999, 0.6767000000000001, 0.6634, 0.6799999999999999, 0.6699999999999999, 0.6598999999999999, 0.6333000000000001, 0.6199, 0.6466999999999999, 0.5967, 0.6, 0.6268, 0.6667, 0.6769000000000001, 0.6399, 0.6466000000000001, 0.6567000000000001, 0.6501, 0.6333, 0.6500000000000001, 0.6266, 0.6299, 0.5901000000000001, 0.5700000000000001, 0.6065999999999999, 0.64, 0.6732000000000001, 0.6033999999999999, 0.6301, 0.6167999999999999, 0.6266999999999999, 0.6201, 0.66, 0.6666, 0.6698999999999999, 0.6668, 0.6599, 0.6734, 0.7034, 0.68, 0.6632999999999999, 0.6965999999999999, 0.6534, 0.6534, 0.5998999999999999, 0.6365999999999999, 0.6398999999999999, 0.6765000000000001, 0.64, 0.6133, 0.6199999999999999, 0.5867, 0.6167, 0.6433, 0.6266999999999999, 0.6267, 0.6632, 0.6266999999999999, 0.6334000000000001, 0.6333, 0.6532, 0.6298999999999999, 0.6432, 0.6199999999999999, 0.6234000000000001, 0.6300999999999999, 0.63, 0.6165999999999999, 0.5966999999999999, 0.6334000000000001, 0.6233, 0.6467999999999999, 0.6598999999999999, 0.6432, 0.6397999999999999, 0.6532999999999999, 0.5966, 0.6333, 0.6399, 0.6235, 0.6098999999999999, 0.6467, 0.6298, 0.6567000000000001, 0.6500000000000001, 0.6733, 0.6599999999999999, 0.6533, 0.7001000000000001, 0.6833000000000001, 0.6834, 0.6399, 0.6134, 0.6201000000000001, 0.6567999999999999, 0.6266, 0.62, 0.6233, 0.6598999999999999, 0.6531, 0.63, 0.66, 0.6734, 0.6601, 0.6534, 0.6366999999999999, 0.6666000000000001, 0.6466, 0.6601, 0.6634, 0.6202, 0.6400999999999999, 0.6533, 0.6499999999999998, 0.6367, 0.6365999999999999, 0.6365999999999999, 0.6435, 0.6233, 0.6633, 0.6633, 0.6632, 0.6434, 0.6367, 0.6634, 0.6833, 0.7133999999999999, 0.6866, 0.6467, 0.6533, 0.6433, 0.6433, 0.7199, 0.7032999999999998, 0.6832999999999999, 0.6399, 0.67, 0.6801999999999999, 0.6732, 0.6599999999999999, 0.6600999999999999, 0.6733, 0.6567000000000001, 0.6668000000000001, 0.6499999999999999, 0.6701000000000001, 0.6765, 0.6868, 0.6634, 0.6767, 0.6498999999999999, 0.6299, 0.6333, 0.6367, 0.6165999999999999, 0.65, 0.6266, 0.6632, 0.7034, 0.6733, 0.6467, 0.6633, 0.6233, 0.6199999999999999, 0.5766000000000001, 0.6066999999999999, 0.6635, 0.6399999999999999, 0.65, 0.6433, 0.6498999999999999, 0.6598999999999999, 0.6598999999999999, 0.6466, 0.65, 0.6499999999999999, 0.6398999999999999, 0.6566, 0.6498999999999999, 0.6433000000000001, 0.6601999999999999, 0.6998999999999999, 0.6601000000000001, 0.6400000000000001, 0.6467, 0.6466999999999999, 0.6434, 0.6133, 0.6432999999999999, 0.65, 0.6866999999999999, 0.6734, 0.6634999999999999, 0.6666999999999998, 0.6967000000000001, 0.6967000000000001, 0.7, 0.6834, 0.6768, 0.6833000000000001, 0.6933999999999999, 0.6799999999999999, 0.6935, 0.6865, 0.6866000000000001, 0.6698999999999999, 0.6532, 0.6466000000000001, 0.6734, 0.6931999999999999, 0.6800999999999999, 0.6933999999999999, 0.6900000000000001, 0.6799999999999999, 0.6766, 0.6767000000000001, 0.6798999999999998, 0.6934, 0.6434, 0.6534, 0.6832, 0.6698999999999999, 0.6632, 0.6934, 0.6934999999999999, 0.6666000000000001, 0.6432, 0.66, 0.6734, 0.6598999999999999, 0.6765, 0.6900000000000001, 0.6567999999999999, 0.66, 0.6333000000000001, 0.6434, 0.6597999999999999, 0.6398, 0.6300000000000001, 0.6433000000000001, 0.6399999999999999, 0.6666999999999998, 0.6699999999999999, 0.6634, 0.6799999999999999, 0.6534999999999999, 0.6566000000000001, 0.6701, 0.6566000000000001, 0.6434, 0.6232, 0.5966, 0.6166, 0.6598999999999999, 0.6434, 0.6533, 0.6334000000000001, 0.6500999999999999, 0.6433, 0.6534, 0.6234, 0.6233000000000001, 0.6234, 0.6265999999999999, 0.6365999999999999, 0.6399999999999999, 0.6466999999999998, 0.6233, 0.6099, 0.6033, 0.6166, 0.6199000000000001, 0.6266, 0.6302000000000001, 0.6500000000000001, 0.65, 0.6367, 0.6399000000000001, 0.6398999999999999, 0.6498999999999999, 0.6498999999999999, 0.6566, 0.6467, 0.6565999999999999, 0.6101, 0.6167, 0.6434, 0.6401, 0.6365999999999999, 0.6432000000000001, 0.6367, 0.6368, 0.6433000000000001, 0.6367, 0.6234, 0.6367, 0.6499999999999999, 0.6401, 0.6601, 0.6333000000000001, 0.6598999999999999, 0.6500999999999999, 0.6566, 0.6567, 0.6466000000000001, 0.6566, 0.6432999999999999, 0.6466, 0.6499999999999999, 0.6433, 0.6664999999999999, 0.6399999999999999, 0.6633, 0.6399999999999999, 0.6534, 0.6666999999999998, 0.6767, 0.6734, 0.65, 0.67, 0.67, 0.6699999999999999, 0.6498999999999999, 0.6399999999999999, 0.6634, 0.6767000000000001, 0.6601, 0.6634, 0.6700999999999999, 0.6600999999999999, 0.6733, 0.6731999999999999, 0.6634, 0.6466000000000001, 0.6435, 0.6301, 0.6666, 0.6533, 0.6399, 0.6300000000000001, 0.6399, 0.6367, 0.6233000000000001, 0.64, 0.6467999999999999, 0.6433, 0.6400999999999999, 0.6399999999999999, 0.6234, 0.6234, 0.62, 0.62, 0.6266, 0.6332, 0.6233, 0.65, 0.6233, 0.6698999999999999, 0.6666, 0.6599999999999999, 0.6698, 0.6767, 0.6767, 0.6698999999999999, 0.6567000000000001, 0.6467, 0.6467, 0.6533, 0.6666999999999998, 0.6666999999999998, 0.6765999999999999, 0.67, 0.6767, 0.6634, 0.6701, 0.6634, 0.6933999999999999, 0.7, 0.6832999999999999, 0.6533, 0.6299999999999999, 0.6366999999999999, 0.6534, 0.6399999999999999, 0.6367999999999999, 0.65, 0.6500999999999999, 0.6565999999999999, 0.65, 0.6498, 0.6466000000000001, 0.6267, 0.6432, 0.6466999999999999, 0.6332, 0.6498999999999999, 0.6365999999999999, 0.6468, 0.6234000000000001, 0.6201000000000001, 0.6301, 0.6100999999999999, 0.63, 0.6534000000000001, 0.6399, 0.6466000000000001, 0.6433, 0.6433000000000001, 0.6301, 0.6433, 0.6466000000000001, 0.6266, 0.6298999999999999, 0.6299, 0.6499, 0.6366, 0.6266, 0.6267, 0.6234, 0.63, 0.6333, 0.6199999999999999, 0.6068, 0.6100999999999999, 0.6333, 0.6367, 0.62, 0.6365999999999999, 0.6266999999999999, 0.6333, 0.6466999999999999, 0.6399999999999999, 0.6500000000000001, 0.6499999999999999, 0.63, 0.6199999999999999, 0.6232999999999999, 0.6302000000000001, 0.6367, 0.6201, 0.6333, 0.6335, 0.6367, 0.6268]
# first = [i*1000 for i in range(6)]
# second = [6000 + (i*5000) for i in range(len(edn)-6)]
# all_iterations = first + second
all_iterations = [i for i in range(len(edn))]
all_iterations.append(len(ave_test)*batch_size)
edn.append(edn[-1])
plt.plot(all_iterations, edn, 'b')
plt.title(test_label)
plt.show()
'''