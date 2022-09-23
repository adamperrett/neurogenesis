import pickle
import numpy as np
from scipy import signal
# print("Loading MNIST training data")
# infile = open("../datasets/processed_mnist_training_k9_2max.pickle", 'rb')
# mnist_training_data, mnist_training_labels = pickle.load(infile)
# infile.close()
# print("Loading MNIST testing data")
# infile = open("../datasets/processed_mnist_testing_k9_2max.pickle", 'rb')
# mnist_testing_data, mnist_testing_labels = pickle.load(infile)
# infile.close()
print("Loading MNIST training data")
infile = open("../datasets/processed_mnist_training_k9_2no.pickle", 'rb')
mnist_training_data, mnist_training_labels = pickle.load(infile)
infile.close()
print("Loading MNIST testing data")
infile = open("../datasets/processed_mnist_testing_k9_2no.pickle", 'rb')
mnist_testing_data, mnist_testing_labels = pickle.load(infile)
infile.close()
# print("Loading MNIST training data")
# infile = open("../datasets/mnist_training.pickle", 'rb')
# mnist_training_data, mnist_training_labels = pickle.load(infile)
# infile.close()
# print("Loading MNIST testing data")
# infile = open("../datasets/mnist_testing.pickle", 'rb')
# mnist_testing_data, mnist_testing_labels = pickle.load(infile)
# infile.close()

def process(image, kernel, min=0):
    if len(np.array(image).shape) == 1:
        image = np.array(image).reshape((28, 28))
    processed_image = signal.convolve2d(image, kernel)
    processed_image = processed_image * (processed_image > min)
    return processed_image

def make_kernels(size, min, max, width=3):
    kernels = []

    half_width = int(width/2)
    # diag1
    k = np.ones((size, size)) * min
    for i in range(size):
        for j in range(width):
            shift = j-half_width
            if i + shift < 0 or i + shift >= size:
                shift = 0
            k[i+shift][i] = max
    kernels.append(k)
    # diag2
    k = np.ones((size, size)) * min
    for i in range(size):
        for j in range(width):
            shift1 = j-half_width
            shift2 = j-half_width
            if size-1-i + shift1 < 0 or size-1-i + shift1 >= size:
                shift1 = 0
            if i + shift2 < 0 or i + shift2 >= size:
                shift2 = 0
            k[size-1-i+shift1][i] = max
    kernels.append(k)

    midpoint = int(size / 2)
    # verticle
    k = np.ones((size, size)) * min
    for i in range(size):
        for j in range(width):
            shift = j-half_width
            k[i][midpoint+shift] = max
    kernels.append(k)
    # horizontal
    k = np.ones((size, size)) * min
    for i in range(size):
        for j in range(width):
            shift = j-half_width
            k[midpoint+shift][i] = max
    kernels.append(k)
    return kernels

def gaussian_kernel(size, width = 0.4):
    x, y = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))
    d = np.sqrt(x * x + y * y)
    sigma, mu = width, 0.0
    g = np.exp(-((d - mu) ** 2 / (2.0 * sigma ** 2)))
    return g

if __name__ == '__main__':

    from copy import deepcopy
    import matplotlib.pyplot as plt
    import skimage.measure as ds

    kernel_size = 9
    pooling = 'no'
    downsample_size = 2
    plot = False
    kernels = make_kernels(size=kernel_size,
                           min=0,
                           max=1)

    print("processing training data")
    processed_mnist_training_data = []
    for train in mnist_training_data:
        processing_steps = []
        all_processed = []
        for k in kernels:
            processed_train = deepcopy(train)
            # processed_train = process(train, gaussian_kernel(kernel_size))
            # processing_steps.append(process(processed_train, k).flatten())
            # all_processed.append(process(process(processed_train, k), gaussian_kernel(kernel_size)))
            if pooling == 'max':
                all_processed.append(process(process(processed_train, k), gaussian_kernel(kernel_size)))
                all_processed[-1] = ds.block_reduce(all_processed[-1], (downsample_size, downsample_size), np.max)
            elif pooling == 'ave':
                all_processed.append(process(process(processed_train, k), gaussian_kernel(kernel_size)))
                all_processed[-1] = ds.block_reduce(all_processed[-1], (downsample_size, downsample_size), np.average)
            else:
                all_processed.append(process(processed_train, k))
            processing_steps.append(all_processed[-1].flatten())
        processed_mnist_training_data.append(np.hstack(processing_steps))
        if plot:
            max_val = np.max(processed_mnist_training_data)
            fig, axs = plt.subplots(3, 2)
            axs[0][0].imshow(np.array(processed_train).reshape((28, 28)),
                             cmap='hot', interpolation='nearest', aspect='auto', vmin=0)
            # pcm = axs[0][0].pcolormesh(np.array(processed_train).reshape((28, 28)), cmap='hot')
            # fig.colorbar(pcm, ax=axs[0][0])
            blurred_image = process(processed_train, gaussian_kernel(kernel_size))
            axs[0][1].imshow(blurred_image,
                             cmap='hot', interpolation='nearest', aspect='auto', vmin=0)

            axs[1][0].imshow(all_processed[0] / max_val, cmap='hot', interpolation='nearest', aspect='auto', vmin=0, vmax=1)
            # pcm = axs[1][0].pcolormesh(np.array(all_processed[0]) / max_val, cmap='hot')
            # fig.colorbar(pcm, ax=axs[1, 0])
            axs[1][1].imshow(all_processed[1] / max_val, cmap='hot', interpolation='nearest', aspect='auto', vmin=0, vmax=1)
            # pcm = axs[1][1].pcolormesh(np.array(all_processed[1]) / max_val, cmap='hot')
            # fig.colorbar(pcm, ax=axs[1, 1])
            axs[2][0].imshow(all_processed[2] / max_val, cmap='hot', interpolation='nearest', aspect='auto', vmin=0, vmax=1)
            # pcm = axs[2][0].pcolormesh(np.array(all_processed[2]) / max_val, cmap='hot')
            # fig.colorbar(pcm, ax=axs[2, 0])
            axs[2][1].imshow(all_processed[3] / max_val, cmap='hot', interpolation='nearest', aspect='auto', vmin=0, vmax=1)
            # pcm = axs[2][1].pcolormesh(np.array(all_processed[3]) / max_val, cmap='hot')
            # fig.colorbar(pcm, ax=axs[2, 1])
            plt.show()

    max_val = np.max(processed_mnist_training_data)
    processed_mnist_training_data = np.array(processed_mnist_training_data) / max_val

    print("processing testing data")
    processed_mnist_testing_data = []
    for test in mnist_testing_data:
        processing_steps = []
        all_processed = []
        for k in kernels:
            processed_test = deepcopy(test)
            # all_processed.append(process(process(processed_train, k), gaussian_kernel(kernel_size)))
            if pooling == 'max':
                all_processed.append(process(process(processed_test, k), gaussian_kernel(kernel_size)))
                all_processed[-1] = ds.block_reduce(all_processed[-1], (downsample_size, downsample_size), np.max)
            elif pooling == 'ave':
                all_processed.append(process(process(processed_test, k), gaussian_kernel(kernel_size)))
                all_processed[-1] = ds.block_reduce(all_processed[-1], (downsample_size, downsample_size), np.average)
            else:
                all_processed.append(process(processed_test, k))
            processing_steps.append(all_processed[-1].flatten())
        processed_mnist_testing_data.append(np.hstack(processing_steps))
    processed_mnist_testing_data = np.array(processed_mnist_testing_data) / max_val

    import pickle
    print("Beginning pickling")
    filename = "processed_mnist_training_k{}_{}{}.pickle".format(kernel_size, downsample_size, pooling)
    outfile = open(filename, 'wb')
    pickle.dump([processed_mnist_training_data, mnist_training_labels], outfile)
    outfile.close()
    print("pickled training")
    filename = "processed_mnist_testing_k{}_{}{}.pickle".format(kernel_size, downsample_size, pooling)
    outfile = open(filename, 'wb')
    pickle.dump([processed_mnist_testing_data, mnist_testing_labels], outfile)
    outfile.close()
    print("pickled testing")

