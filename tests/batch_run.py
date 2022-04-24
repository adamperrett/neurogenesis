import os
import subprocess
import time

sensitivity_width = [0.4]#[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
output_thresholding = [0, 1]
error_threshold = [0.1]
surprise_threshold = [0.0, 0.1, 0.2, 0.3, 0.35, 0.4, 0.45]
maximum_synapses = [50]#, 100, 150, 200, 250, 300, 600]
input_spread = [0, 'act']
activity_decay_rate = [0.99]
number_of_seeds = [0]
fixed_hidden_amount = [0]

processes = []
logs = []
for sw in sensitivity_width:
    for at in output_thresholding:
        for er in error_threshold:
            for mn in surprise_threshold:
                for ms in maximum_synapses:
                    for ins in input_spread:
                        for adr in activity_decay_rate:
                            for ns in number_of_seeds:
                                for fha in fixed_hidden_amount:
                                    screen_name = "sw{}_ot{}_er{}_st{}_ms{}_exp_{}_adr{}_ns{}_fha{}".format(
                                        sw, at, er, mn, ms, ins, adr, ns, fha)
                                    open_screen = "screen -dmS " + screen_name + " bash -c "
                                    move_and_source = "neurogenesis_source && "
                                    # command = "\"" + move_and_source + " python3 incremental_shd.py {} {} {} {} {}\"".format(h, r, v1, v2, fb)
                                    command = "\"python3 mnist_training.py {} {} {} {} {} {} {} {} {}; " \
                                              "exec bash\"".\
                                        format(sw, at, er, mn, ms, ins, adr, ns, fha)

                                    # logs.append(open("log_output_{}.txt".format(screen_name), 'a'))
                                    # process = subprocess.Popen(open_screen+command, stdout=subprocess.PIPE)
                                    processes.append(subprocess.Popen(
                                        'screen -d -m -S {} bash -c {}'.format(screen_name, command),
                                        shell=True,
                                        stdout=subprocess.PIPE,
                                        stdin=subprocess.PIPE,
                                        stderr=subprocess.PIPE))
                                    print("Set up config", screen_name)
                                    time.sleep(0.1)

days = 0
print("Done - beginning wait of", days, "days")
# time.sleep(60*60*24*days)
# print("Finished waiting")
