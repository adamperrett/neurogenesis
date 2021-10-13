import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pl


def create_centroid_classes(class_centres, spread, examples):
    data = []
    labels = []
    for out, centre in enumerate(class_centres):
        for i in range(examples):
            data.append(np.random.normal(centre, spread))
            labels.append(out)
    return data, labels


if __name__ == "__main__":
    # centres = [[1, 0],
    #            [0, 0],
    #            [0, 1]]
    centres = [[1, 1],
               [0, 0]]
    spread = 0.3
    examples = 100
    data, labels = create_centroid_classes(centres, spread, examples)

    colours = pl.cm.plasma(np.linspace(0, 1, len(centres)))
    plt.figure()
    for i in range(len(centres)):
            plt.scatter(np.array(data)[i*examples:(i+1)*examples, 0],
                        np.array(data)[i*examples:(i+1)*examples, 1],
                        color=colours[i])
    plt.show()
    print("done")