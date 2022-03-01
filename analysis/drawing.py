import numpy as np
import matplotlib.pyplot as plt
import scipy.stats

def gaussian_kernel(distance):
    return scipy.stats.norm(0, 1).pdf(distance)

def triangle_kernel(distance):
    return max(0, 1 - (np.abs(distance) / 2.5)) * 0.4

def kernel_density_estimation(samples):
    resolution = 100
    input_range = 10
    kernel_locations = np.random.random(samples) * input_range
    x = 0
    y = 1
    total_output = [[], []]
    total_triangle = [[], []]
    kernel_outputs = [[[], []] for i in range(samples)]
    triangle_outputs = [[[], []] for i in range(samples)]
    for loc in np.linspace(-5, input_range+5, resolution):
        total_output[x].append(loc)
        total_triangle[x].append(loc)
        g_total = 0
        t_total = 0
        for i in range(samples):
            distance = kernel_locations[i] - loc
            k_value = gaussian_kernel(distance)
            t_value = triangle_kernel(distance)
            if k_value > 0.001:
                kernel_outputs[i][x].append(loc)
                kernel_outputs[i][y].append(k_value)
            if t_value != 0:
                triangle_outputs[i][x].append(loc)
                triangle_outputs[i][y].append(t_value)
            g_total += k_value
            t_total += t_value
        total_output[y].append(g_total)
        total_triangle[y].append(t_total)
    plt.figure()
    for i in range(samples):
        plt.plot(kernel_outputs[i][x], kernel_outputs[i][y], 'r--')
        plt.plot(triangle_outputs[i][x], triangle_outputs[i][y], 'g--')
    plt.plot(total_output[x], total_output[y], 'b')
    plt.plot(total_triangle[x], total_triangle[y], 'r')
    for i in range(samples):
        plt.plot([kernel_locations[i], kernel_locations[i]],
                 [0, 0.1], 'k')
    plt.axis('off')
    plt.show()

def error_density_estimation(samples):
    resolution = 100
    input_range = 10
    kernel_locations = np.random.random(samples) * input_range
    triangle_errors = np.ones(samples)#np.random.normal(size=samples)
    x = 0
    y = 1
    total_output = [[], []]
    total_triangle = [[], []]
    kernel_outputs = [[[], []] for i in range(samples)]
    triangle_outputs = [[[], []] for i in range(samples)]
    for loc in np.linspace(-5, input_range+5, resolution):
        total_output[x].append(loc)
        total_triangle[x].append(loc)
        g_total = 0
        t_total = 0
        for i in range(samples):
            distance = kernel_locations[i] - loc
            k_value = gaussian_kernel(distance)
            t_value = triangle_kernel(distance) * triangle_errors[i]
            if k_value > 0.001:
                kernel_outputs[i][x].append(loc)
                kernel_outputs[i][y].append(k_value)
            if t_value != 0:
                triangle_outputs[i][x].append(loc)
                triangle_outputs[i][y].append(t_value)
            g_total += k_value
            t_total += t_value
        total_output[y].append(g_total)
        total_triangle[y].append(t_total)
    fig, ax = plt.subplots(1, 2)
    for i in range(samples):
        ax[0].plot(kernel_outputs[i][x], kernel_outputs[i][y], 'r--')
        ax[1].plot(triangle_outputs[i][x], triangle_outputs[i][y], 'g--')
    ax[0].plot(total_output[x], total_output[y], 'b')
    ax[1].plot(total_triangle[x], total_triangle[y], 'r')
    for i in range(samples):
        ax[0].plot([kernel_locations[i], kernel_locations[i]],
                 [0, 0.1], 'k')
        ax[1].plot([kernel_locations[i], kernel_locations[i]],
                 [0, 0.1], 'k')
    ax[0].set_ylim(min(np.min([np.min(triangle_outputs[i][y]) for i in range(len(triangle_outputs))]),
                       np.min(total_triangle[y])),
                   np.max(total_output[y]) + 0.1)
    ax[1].set_ylim(min(np.min([np.min(triangle_outputs[i][y]) for i in range(len(triangle_outputs))]),
                       np.min(total_triangle[y])),
                   np.max(total_output[y]) + 0.1)
    ax[0].axis('off')
    ax[1].axis('off')
    plt.show()

num_samples = 9
# kernel_density_estimation(num_samples)
# kernel_density_estimation(num_samples)
# kernel_density_estimation(num_samples)
# kernel_density_estimation(num_samples)
# kernel_density_estimation(num_samples)
error_density_estimation(num_samples)
error_density_estimation(num_samples)
error_density_estimation(num_samples)
error_density_estimation(num_samples)
error_density_estimation(num_samples)
print("Kernel esitmation done")

'''
Show the activation of EDSAN neuron
    difference of less/more inputs
    difference of output threshold

Show the activation of sigmoid neuron
'''

x_range = [0, 1]
y_range = [0, 1]
resolution = 100

c = {
    'class0': [1, 1, 1],
    'class1': [0.9, 0, 0.9],
    'point0': [1, 0.5, 0],
    'point1': [0.5, 0, 0.5]
}
c['syn0'] = [c['class1'][0], 0, 0]
c['syn1'] = [0, 0, c['class1'][2]]

opacity_rescale = 0.1
dot_size = 400

data_points = [
    # [0.25, 0.25],
    # [0.25, 0.75],
    # [0.75, 0.25],
    # [0.75, 0.75]
    [
        # [0.25, 0.25],
        # [0.75, 0.75]
    ],
    [
        # [0.25, 0.75],
        [0.5, 0.5]
    ]
]

ng_width = 0.3
ng_points = data_points

def hat_f(x, px):
    return np.max([0, 1 - (np.abs(x - px) / ng_width)])

def total_hat_f(x, y, px, py):
    return hat_f((hat_f(x, px) + hat_f(y, py)) / 2, 1)

tf_vector = [
    [1, -0.25, -0.1]
]

# for cl, cent in centroids:
#     if cent[0] != 0 or cent[1] != 0:
#         if cl % 2:
#             plt.scatter(cent[0], cent[1], marker='+', s=800)
#         else:
#             plt.scatter(cent[0], cent[1], marker='x', s=400)
# plt.xlim(x_range)
# plt.ylim(y_range)
# plt.show()
# fig, ax = plt.subplots(2, 2)
# im1 = ax[0][0].imshow(heat_map[0], cmap='viridis',
#                       extent=x_range + y_range)  # [x_range[0], x_range[1], y_range[0], y_range[1]])
# im2 = ax[0][1].imshow(heat_map[1], cmap='viridis',
#                       extent=x_range + y_range)  # [x_range[0], x_range[1], y_range[0], y_range[1]])
# im3 = ax[1][0].imshow(heat_map[2], cmap='viridis',
#                       extent=x_range + y_range)  # [x_range[0], x_range[1], y_range[0], y_range[1]])
# for i in range(num_outputs):
#     ax[1][1].scatter(np.array(new_coords[i])[:, 0],
#                      np.array(new_coords[i])[:, 1])
#     cent = [np.average(np.array(new_coords[i])[:, 0]), np.average(np.array(new_coords[i])[:, 1])]
#     ax[1][1].scatter(cent[0], cent[1], s=200)
# cbar1 = fig.colorbar(im1, ax=ax[0][0])
# cbar2 = fig.colorbar(im2, ax=ax[0][1])
# cbar3 = fig.colorbar(im3, ax=ax[1][0])
# fig.tight_layout()
# plt.show()

num_classes = len(data_points)
tf_boundary = [[] for i in range(num_classes)]
tf_alphas = [[] for i in range(num_classes)]
ng_boundary = [[] for i in range(num_classes)]
ng_alphas = [[] for i in range(num_classes)]
ng_s_boundary = [[] for i in range(2)]
ng_s_alphas = [[] for i in range(2)]
# heat_map = [[[0. for i in range(resolution)] for j in range(resolution)] for o in range(num_outputs + 1)]

for i, x in enumerate(np.linspace(x_range[0], x_range[1], resolution)):
    print(x, "/", x_range[1])
    for j, y in enumerate(reversed(np.linspace(y_range[0], y_range[1], resolution))):
        tf_output = [a*x + b*y + c for a, b, c in tf_vector]
        if tf_output[0] <= 0.:
            tf_boundary[0].append(np.array([x, y]))
            tf_alphas[0].append(np.hstack([c['class0'], 0]))
        else:
            tf_boundary[1].append(np.array([x, y]))
            tf_alphas[1].append(np.hstack([c['class1'], tf_output[0] * opacity_rescale]))
        ng_output = []
        for points in ng_points:
            output = 0.
            for px, py in points:
                output += total_hat_f(x, y, px, py)
                x_act = hat_f(x, px)
                y_act = hat_f(y, py)
                if x_act > 0:
                    ng_s_boundary[0].append(np.array([x, y]))
                    ng_s_alphas[0].append(np.hstack([c['syn0'], x_act * opacity_rescale]))
                if y_act > 0:
                    ng_s_boundary[1].append(np.array([x, y]))
                    ng_s_alphas[1].append(np.hstack([c['syn1'], y_act * opacity_rescale]))
            ng_output.append(output)
        if np.max(ng_output) > 0:
            ng_boundary[int(np.argmax(ng_output))].append(np.array([x, y]))
            np.hstack([c['class{}'.format(int(np.argmax(ng_output)))], np.max(ng_output)])
            ng_alphas[int(np.argmax(ng_output))].append(
                np.hstack([c['class{}'.format(int(np.argmax(ng_output)))], np.max(ng_output) * opacity_rescale]))


fig, ax = plt.subplots(1, 2)
plt.setp(ax, xlim=x_range, ylim=y_range)
ax[0].set_title('ReLU neuron activation')
ax[1].set_title('EDSAN neuron activation')
for i in range(2):
    ax[i].set_xlabel('x1', fontsize=14)
    ax[i].set_ylabel('x2', fontsize=14)
    if len(ng_s_boundary[i]):
        ax[1].scatter(np.array(ng_s_boundary[i])[:, 0],
                      np.array(ng_s_boundary[i])[:, 1],
                      s=dot_size, color=ng_s_alphas[i], label='Synapse {} activation'.format(i))
for i in range(num_classes):
    if len(tf_boundary[i]):
        ax[0].scatter(np.array(tf_boundary[i])[:, 0],
                      np.array(tf_boundary[i])[:, 1],
                      s=dot_size, color=tf_alphas[i])
    if len(ng_boundary[i]):
        ax[1].scatter(np.array(ng_boundary[i])[:, 0],
                      np.array(ng_boundary[i])[:, 1],
                      s=dot_size, color=ng_alphas[i], label='Neuron activation')
ax[1].legend(loc='upper right')
if len(data_points[0]):
    ax[0].scatter(np.array(data_points[0])[:, 0],
                  np.array(data_points[0])[:, 1],
                  s=400, color=c['point0'])
    ax[1].scatter(np.array(data_points[0])[:, 0],
                  np.array(data_points[0])[:, 1],
                  s=400, color=c['point0'])
if len(data_points[1]):
    ax[0].scatter(np.array(data_points[1])[:, 0],
                  np.array(data_points[1])[:, 1],
                  s=400, color=c['point1'])
    ax[1].scatter(np.array(data_points[1])[:, 0],
                  np.array(data_points[1])[:, 1],
                  s=400, color=c['point1'])
plt.show()

print('done')