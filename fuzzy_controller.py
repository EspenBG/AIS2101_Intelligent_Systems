#%%
import os
import sys
import simpful as fs  # changed line 149 to show(block=False) (in simpful.py)
from matplotlib import pylab as plt
import numpy as np
from scipy import signal as sig

enable_sugeno = False

# Make the fuzzy controller
FC = fs.FuzzySystem(show_banner=False)

#%%
# Linguistic variables
temp_t1 = fs.FuzzySet(points=[[-15., 1.],  [0., 0.]], term="cold")
temp_t2 = fs.FuzzySet(points=[[-10., 0.],  [0., 1.], [10., 0.]], term="good")
temp_t3 = fs.FuzzySet(points=[[0., 0.],  [15., 1.]], term="hot")
temp_lv = fs.LinguisticVariable([temp_t1, temp_t2, temp_t3], universe_of_discourse=[-20, 20],
                                concept="Temperature")
# Add the linguistic variable to the system
FC.add_linguistic_variable("temp", temp_lv)

# Plot the variable
FC.plot_variable("temp")

# Linguistic variables
flow_f1 = fs.FuzzySet(points=[[-.8, 1.],  [0., 0.]], term="soft")
flow_f2 = fs.FuzzySet(points=[[-.4, 0.],  [0., 1.], [.4, 0.]], term="good")
flow_f3 = fs.FuzzySet(points=[[0., 0.],  [0.8, 1.]], term="hard")
flow_lv = fs.LinguisticVariable([flow_f1, flow_f2, flow_f3], universe_of_discourse=[-1, 1],
                                concept="Flow")
# Add the linguistic variable to the system
FC.add_linguistic_variable("flow", flow_lv)

# Plot the variable
FC.plot_variable("flow")

# Linguistic variables
water_o1 = fs.FuzzySet(points=[[-1., 0.], [-.6, 1.], [-.3, 0.]], term="close_fast")
water_o2 = fs.FuzzySet(points=[[-.6, 0.], [-.3, 1.], [0., 0.]], term="close_slow")
water_o3 = fs.FuzzySet(points=[[-.3, 0.], [0., 1.], [.3, 0.]], term="steady")
water_o4 = fs.FuzzySet(points=[[0., 0.], [.3, 1.], [.6, 0.]], term="open_slow")
water_o5 = fs.FuzzySet(points=[[.3, 0.], [.6, 1.], [1., 0.]], term="open_fast")
water_hot = fs.LinguisticVariable([water_o1, water_o2, water_o3, water_o4, water_o5], universe_of_discourse=[-1, 1],
                                  concept="Cold Water")
water_cold = fs.LinguisticVariable([water_o1, water_o2, water_o3, water_o4, water_o5], universe_of_discourse=[-1, 1],
                                   concept="Hot Water")
# Add the linguistic variable to the system
FC.add_linguistic_variable("hot", water_hot)
FC.add_linguistic_variable("cold", water_cold)

# Plot the variable
FC.plot_variable("hot")
FC.plot_variable("cold")

FC.produce_figure("controller_ling_var.png", max_figures_per_row=2)

#%%
# Rule base for the system
R1 = "IF (temp IS cold) AND (flow IS soft) THEN (cold IS open_slow)"
R2 = "IF (temp IS cold) AND (flow IS soft) THEN (hot IS open_fast)"

R3 = "IF (temp IS cold) AND (flow IS good) THEN (cold IS close_slow)"
R4 = "IF (temp IS cold) AND (flow IS good) THEN (hot IS open_slow)"

R5 = "IF (temp IS cold) AND (flow IS hard) THEN (cold IS close_fast)"
R6 = "IF (temp IS cold) AND (flow IS hard) THEN (hot IS close_slow)"

R7 = "IF (temp IS good) AND (flow IS soft) THEN (cold IS open_slow)"
R8 = "IF (temp IS good) AND (flow IS soft) THEN (hot IS open_slow)"

R9 = "IF (temp IS good) AND (flow IS good) THEN (cold IS steady)"
R10 = "IF (temp IS good) AND (flow IS good) THEN (hot IS steady)"

R11 = "IF (temp IS good) AND (flow IS hard) THEN (cold IS close_slow)"
R12 = "IF (temp IS good) AND (flow IS hard) THEN (hot IS close_slow)"

R13 = "IF (temp IS hot) AND (flow IS soft) THEN (cold IS open_fast)"
R14 = "IF (temp IS hot) AND (flow IS soft) THEN (hot IS open_slow)"

R15 = "IF (temp IS hot) AND (flow IS good) THEN (cold IS open_slow)"
R16 = "IF (temp IS hot) AND (flow IS good) THEN (hot IS close_slow)"

R17 = "IF (temp IS hot) AND (flow IS hard) THEN (cold IS close_slow)"
R18 = "IF (temp IS hot) AND (flow IS hard) THEN (hot IS close_fast)"

rule_base = [R1, R2, R3, R4, R5, R6, R7, R8, R9, R10, R11, R12, R13, R14, R15, R16, R17, R18]
FC.add_rules(rule_base)

if enable_sugeno:
    FC.set_crisp_output_value("close_fast", -0.6)
    FC.set_crisp_output_value("close_slow", -0.3)
    FC.set_crisp_output_value("steady", 0.)
    FC.set_crisp_output_value("open_slow", 0.3)
    FC.set_crisp_output_value("open_fast", 0.6)


#%%
# Make the water valves
class WaterValve:
    """Class to simulate a water valve"""
    max_valve_pos = 2

    def __init__(self, water_temp, max_flow, sample_time):
        self.time_step = sample_time
        self.max_flow = max_flow
        self.temp = water_temp
        self.valve_pos = 0.1
        self.valve_input = 0
        self.flow = 0

    def set_new_pos(self, max_valve_pos=max_valve_pos):
        self.valve_pos += self.valve_input*self.time_step
        # Override the valve position if its over max
        if self.valve_pos > WaterValve.max_valve_pos:
            self.valve_pos = WaterValve.max_valve_pos

    def update_outputs(self):
        self.set_new_pos()
        self.set_flow()

    def set_input(self, value):
        self.valve_input = value

    def set_flow(self):
        # The flow is calculated based on the valve position, but can not be over the max flow
        self.flow = self.valve_pos*(self.valve_pos <= self.max_flow)
        self.flow += self.max_flow*(self.valve_pos > self.max_flow)

    def get_flow(self):
        return self.flow

    def get_temp(self):
        return self.temp


#%%
sample_time = 0.01
start_time = 0
stop_time = 50
samples = int((stop_time-start_time)/sample_time + 1)

time = np.linspace(start_time, stop_time, samples)

# Make the setpoints as square waves
flow_setpoint = 0.7 - .2*sig.square(0.3*time)
temp_setpoint = 23 - 4*sig.square(0.214320*time)
temp = np.zeros(time.size)
flow = np.zeros(time.size)

# Make the water valves
hot_valve = WaterValve(30, 2, sample_time)
cold_valve = WaterValve(10, 2, sample_time)

#%%
# Loop thru the computation
for t in np.arange(time.size):
    # Compute the flow rate
    cold_valve.update_outputs()
    hot_valve.update_outputs()
    flow[t] = hot_valve.get_flow() + cold_valve.get_flow()
    # Compute the temp
    temp[t] = (hot_valve.get_flow()*hot_valve.get_temp() + cold_valve.get_flow()*cold_valve.get_temp())/(flow[t])

    flow_error = flow[t] - flow_setpoint[t]
    temp_error = temp[t] - temp_setpoint[t]

    # Compute the outputs
    FC.set_variable("temp", temp_error)
    FC.set_variable("flow", flow_error)
    if enable_sugeno:
        output = FC.Sugeno_inference(["cold", "hot"])
    else:
        output = FC.Mamdani_inference(["cold", "hot"])

    cold_valve.set_input(output["cold"])
    hot_valve.set_input(output["hot"])

    # Print progress bar
    percent = ("{0:." + str(1) + "f}").format(100 * (time[t] / float(time[-1])))
    filledLength = int(100 * time[t] // time[-1])
    bar = '█' * filledLength + '-' * (100 - filledLength)
    print(f'\r{"Simulating"} |{bar}| {percent}% {"generated"}', end="\r")


#%%
plt.figure()
plt.plot(time, flow)
plt.plot(time, flow_setpoint)
plt.ylim([0, 1])
plt.legend(["flow", "setpoint"])
plt.grid(True)

plt.figure()
plt.plot(time, temp, label="temp")
plt.plot(time, temp_setpoint, label="setpoint")
plt.ylim([0, 30])
plt.legend(["temp", "setpoint"])
plt.grid(True)

print("The max temperature is: ", temp.max())
print()
