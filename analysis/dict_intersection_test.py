import time
import numpy as np


long_dict = {}
long_length = 6000
for i in range(long_length):
    long_dict['{}'.format(i)] = i
short_dict = {}
short_length = 600
for i in range(short_length):
    short_dict['{}'.format(i+(long_length/2))] = 100 - i

repeat = 100

time_short = []
time_long = []
time_and = []

for i in range(repeat):
    t0 = time.time()

    for ele in short_dict:
        if ele in long_dict:
            print(ele)

    t1 = time.time()
    time_short.append(t1 - t0)

for i in range(repeat):
    t0 = time.time()
    for ele in long_dict:
        if ele in short_dict:
            print(ele)

    t1 = time.time()
    time_long.append(t1 - t0)

for i in range(repeat):
    t0 = time.time()
    for ele in short_dict.keys() & long_dict.keys():
        print(ele)

    t1 = time.time()
    time_and.append(t1 - t0)

print("time for short loop = ", np.average(time_short))
print("time for long loop = ", np.average(time_long))
print("time for and = ", np.average(time_and))
print("done")