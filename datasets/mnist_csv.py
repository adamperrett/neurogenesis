# import csv
# from ast import literal_eval
#
# mnist_training_data = []
# mnist_training_labels = []
# with open('../datasets/mnist_train.csv') as csv_file:
#     csv_reader = csv.reader(csv_file, delimiter=',')
#     for row in csv_reader:
#         data_row = []
#         if row[0] == 'label':
#             continue
#         for ele in row[1:]:
#             data_row.append(float(literal_eval(ele)) / float(255))
#         mnist_training_data.append(data_row)
#         mnist_training_labels.append(literal_eval(row[0]))
#
# print("collected train data")
#
# mnist_testing_data = []
# mnist_testing_labels = []
# with open('../datasets/mnist_test.csv') as csv_file:
#     csv_reader = csv.reader(csv_file, delimiter=',')
#     for row in csv_reader:
#         data_row = []
#         if row[0] == 'label':
#             continue
#         for ele in row[1:]:
#             data_row.append(float(literal_eval(ele)) / float(255))
#         mnist_testing_data.append(data_row)
#         mnist_testing_labels.append(literal_eval(row[0]))
#
# print("collected test data")
#
# import pickle
# filename = "mnist_training.pickle"
# outfile = open(filename, 'wb')
# pickle.dump([mnist_training_data, mnist_training_labels], outfile)
# outfile.close()
# print("pickled training")
# filename = "mnist_testing.pickle"
# outfile = open(filename, 'wb')
# pickle.dump([mnist_testing_data, mnist_testing_labels], outfile)
# outfile.close()
# print("pickled testing")

import pickle
print("Loading MNIST training data")
infile = open("../datasets/mnist_training.pickle", 'rb')
mnist_training_data, mnist_training_labels = pickle.load(infile)
infile.close()
print("Loading MNIST testing data")
infile = open("../datasets/mnist_testing.pickle", 'rb')
mnist_testing_data, mnist_testing_labels = pickle.load(infile)
infile.close()
# print("Loading reduced MNIST training data")
# infile = open("../datasets/reduced_mnist_training.pickle", 'rb')
# reduced_mnist_training_data, mnist_training_labels = pickle.load(infile)
# infile.close()
# print("Loading MNIST testing data")
# infile = open("../datasets/reduced_mnist_testing.pickle", 'rb')
# reduced_mnist_testing_data, mnist_testing_labels = pickle.load(infile)
# infile.close()
print("Completed loading")

if __name__ == '__main__':
    combined_data = mnist_training_data + mnist_testing_data
    non_changing_indexes = [(28*28) - i - 1 for i in range(28*28)]
    for test in combined_data:
        to_be_deleted = []
        for index in non_changing_indexes:
            if test[index] != combined_data[0][index]:
                to_be_deleted.append(index)
        for deleted_index in to_be_deleted:
            print("deleted", deleted_index)
            del non_changing_indexes[non_changing_indexes.index(deleted_index)]
    print("indexes which remain unchanged through testing:\n", non_changing_indexes)
    print("finished")

    from copy import deepcopy

    print("reducing training data")
    reduced_mnist_training_data = []
    for test in mnist_training_data:
        reduced_test = deepcopy(test)
        for deleted_index in non_changing_indexes:
            del reduced_test[deleted_index]
        reduced_mnist_training_data.append(reduced_test)

    print("reducing testing data")
    reduced_mnist_testing_data = []
    for test in mnist_testing_data:
        reduced_test = deepcopy(test)
        for deleted_index in non_changing_indexes:
            del reduced_test[deleted_index]
        reduced_mnist_testing_data.append(reduced_test)

    import pickle
    print("Beginning pickling")
    filename = "reduced_mnist_training.pickle"
    outfile = open(filename, 'wb')
    pickle.dump([reduced_mnist_training_data, mnist_training_labels], outfile)
    outfile.close()
    print("pickled training")
    filename = "reduced_mnist_testing.pickle"
    outfile = open(filename, 'wb')
    pickle.dump([reduced_mnist_testing_data, mnist_testing_labels], outfile)
    outfile.close()
    print("pickled reduced testing")




