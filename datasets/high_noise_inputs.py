import numpy as np


def generate_date(num_inputs=100,
                  num_outputs=3,
                  examples=300,
                  seed=2727):
    np.random.seed(seed)
    training_data = []
    training_labels = []

    for i in range(examples):
        training_data.append(np.random.random(num_inputs))
        class_label = np.random.randint(num_outputs)
        for j in range(num_outputs):
            training_data[-1][-1-j] = 0
        training_data[-1][-1-class_label] = 1
        training_labels.append(class_label)

    return training_data, training_labels


if __name__ == '__main__':
    d, l = generate_date(num_inputs=10,
                         num_outputs=3)

    print("done")
