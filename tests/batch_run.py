import os
import subprocess
import time

sensitivity_width = [0.6, 0.9]
activation_threshold = [0.0]
error_threshold = [0.01]
maximum_net_size = [1000, 10000]
maximum_synapses = [50, 500]

processes = []
logs = []
for sw in sensitivity_width:
    for at in activation_threshold:
        for er in error_threshold:
            for mn in maximum_net_size:
                for ms in maximum_synapses:
                    screen_name = "sw{}_at{}_er{}_mn{}_ms{}".format(sw, at, er, mn, ms)
                    open_screen = "screen -dmS " + screen_name + " bash -c "
                    move_and_source = "neurogenesis_source && "
                    # command = "\"" + move_and_source + " python3 incremental_shd.py {} {} {} {} {}\"".format(h, r, v1, v2, fb)
                    command = "\"python3 generic_classification.py {} {} {} {} {}; exec bash\"".\
                        format(sw, at, er, mn, ms)

                    # logs.append(open("log_output_{}.txt".format(screen_name), 'a'))
                    # process = subprocess.Popen(open_screen+command, stdout=subprocess.PIPE)
                    processes.append(subprocess.Popen(
                        'screen -d -m -S {} bash -c {}'.format(screen_name, command),
                        shell=True,
                        stdout=subprocess.PIPE,
                        stdin=subprocess.PIPE,
                        stderr=subprocess.PIPE))
                    print("Set up config", screen_name)

days = 4
print("Done - beginning wait of", days, "days")
time.sleep(60*60*24*days)
print("Finished waiting")
