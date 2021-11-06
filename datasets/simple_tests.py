import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import math


def create_centroid_classes(class_centres, spread, examples):
    data = []
    labels = []
    for out, centre in enumerate(class_centres):
        for i in range(examples):
            data.append(np.random.normal(centre, spread))
            labels.append(out)
    return data, labels

def create_bimodal_distribution(class_centres, spread, examples, max_classes=2):
    data = []
    labels = []
    for out, centre in enumerate(class_centres):
        for i in range(examples):
            data.append(np.random.normal(centre, spread))
            labels.append(out % max_classes)
    return data, labels

class YinYangDataset():
    def __init__(self, r_small=0.1, r_big=0.5, size=1000, seed=42, transform=None):
        super(YinYangDataset, self).__init__()
        # using a numpy RNG to allow compatibility to other deep learning frameworks
        self.rng = np.random.RandomState(seed)
        self.transform = transform
        self.r_small = r_small
        self.r_big = r_big
        self.__vals = []
        self.__cs = []
        self.class_names = ['yin', 'yang', 'dot']
        for i in range(size):
            # keep num of class instances balanced by using rejection sampling
            # choose class for this sample
            goal_class = self.rng.randint(3)
            x, y, c = self.get_sample(goal=goal_class)
            # add mirrod axis values
            # x_flipped = 1. - x
            # y_flipped = 1. - y
            val = np.array([x, y])#, x_flipped, y_flipped])
            self.__vals.append(val)
            self.__cs.append(c)

    def get_sample(self, goal=None):
        # sample until goal is satisfied
        found_sample_yet = False
        while not found_sample_yet:
            # sample x,y coordinates
            x, y = self.rng.rand(2) * 2. * self.r_big
            # check if within yin-yang circle
            if np.sqrt((x - self.r_big)**2 + (y - self.r_big)**2) > self.r_big:
                continue
            # check if they have the same class as the goal for this sample
            c = self.which_class(x, y)
            if goal is None or c == goal:
                found_sample_yet = True
                break
        return x, y, c

    def which_class(self, x, y):
        # equations inspired by
        # https://link.springer.com/content/pdf/10.1007/11564126_19.pdf
        d_right = self.dist_to_right_dot(x, y)
        d_left = self.dist_to_left_dot(x, y)
        criterion1 = d_right <= self.r_small
        criterion2 = d_left > self.r_small and d_left <= 0.5 * self.r_big
        criterion3 = y > self.r_big and d_right > 0.5 * self.r_big
        is_yin = criterion1 or criterion2 or criterion3
        is_circles = d_right < self.r_small or d_left < self.r_small
        if is_circles:
            return 2
        return int(is_yin)

    def dist_to_right_dot(self, x, y):
        return np.sqrt((x - 1.5 * self.r_big)**2 + (y - self.r_big)**2)

    def dist_to_left_dot(self, x, y):
        return np.sqrt((x - 0.5 * self.r_big)**2 + (y - self.r_big)**2)

    def __getitem__(self, index):
        sample = (self.__vals[index].copy(), self.__cs[index])
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.__cs)

def twospirals(n_points, noise=.5):
    """
     Returns the two spirals dataset.
    """
    n = np.sqrt(np.random.rand(n_points,1)) * 780 * (2*np.pi)/360
    d1x = -np.cos(n)*n + np.random.rand(n_points,1) * noise
    d1y = np.sin(n)*n + np.random.rand(n_points,1) * noise
    return ((np.vstack((np.hstack((d1x,d1y)),np.hstack((-d1x,-d1y)))) / 25.) + 0.5,
            np.hstack((np.zeros(n_points, dtype=int), np.ones(n_points, dtype=int))))

if __name__ == "__main__":
    centres = [[1, 0],
               [0, 0],
               [0, 1]]
    # centres = [[1, 1],
    #            [0, 0]]
    spread = 0.3
    examples = 1000
    # data, labels = create_centroid_classes(centres, spread, examples)
    # data, labels = create_bimodal_distribution(centres, spread, examples)

    X, y = twospirals(1000)

    plt.title('training set')
    plt.plot(X[y == 0, 0], X[y == 0, 1], '.', label='class 1')
    plt.plot(X[y == 1, 0], X[y == 1, 1], '.', label='class 2')
    plt.legend()
    plt.show()

    # yy = YinYangDataset(size=examples, seed=42)
    # data = [[],
    #         [],
    #         []]
    # for d, l in zip(yy._YinYangDataset__vals, yy._YinYangDataset__cs):
    #     data[l].append(d)
    colours = pl.cm.plasma(np.linspace(0, 1, len(centres)))
    plt.figure()
    for i in range(len(data)):
            # plt.scatter(np.array(data)[i*examples:(i+1)*examples, 0],
            #             np.array(data)[i*examples:(i+1)*examples, 1],
            #             color=colours[i])
            # plt.scatter(np.array(data[i])[:, 0],
            #             np.array(data[i])[:, 1],
            #             color=colours[i])
            plt.scatter(np.array(data[i])[:, 0],
                        np.array(data[i])[:, 1],
                        color=colours[i])
    plt.show()
    print("done")
