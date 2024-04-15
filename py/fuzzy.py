import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

"""
In this example, we're modeling a brake control system for a car based on the distance to the car in front and the car's speed. 
We define three input variables: distance, speed, and an output variable brake.

We then define the membership functions for each variable, 
which specify how strongly the input values belong to each category (e.g. "near", "medium", "far" for distance).

Next, we define the rules that map the input values to the output value. 
For example, rule1 states that if the car is "near" and "fast", 
then the brake should be "high".

We create a control system using these rules and simulate it by passing in inputs for distance and speed. 
The output is the degree to which the brake should be applied, with a value between 0 and 10.

Finally, we print the output value and display a graph of the output membership functions using brake.view().
"""
# Define the input and output variables
distance = ctrl.Antecedent(np.arange(0, 11, 1), 'distance') # Antecedent 前提变量
speed = ctrl.Antecedent(np.arange(0, 101, 1), 'speed')      # Antecedent 前提变量
brake = ctrl.Consequent(np.arange(0, 11, 1), 'brake')       # Consequent 结论变量


"""
The most common shape of membership functions is triangular, 
although trapezoidal and bell curves are also used, 
but the shape is generally less important than the number of curves and their placement. 
From three to seven curves are generally appropriate to cover the required range of an input value, 
or the "universe of discourse" in fuzzy jargon.
"""
# Define the membership functions for the input variables
# distance = ctrl.Antecedent(np.arange(0, 11, 1), 'distance') # Antecedent 前提变量
# trimf = triangular membership function
distance['near'] = fuzz.trimf(distance.universe, [1, 1, 5])
distance['medium'] = fuzz.trimf(distance.universe, [0, 5, 10])
distance['far'] = fuzz.trimf(distance.universe, [5, 10, 10])

# Plot the membership functions
plt.plot(distance.universe, distance['near'].mf, 'b', linewidth=1.5, label='near')
plt.plot(distance.universe, distance['medium'].mf, 'g', linewidth=1.5, label='medium')
plt.plot(distance.universe, distance['far'].mf, 'r', linewidth=1.5, label='far')
plt.title('Membership functions')
plt.legend()
plt.show()

# speed = ctrl.Antecedent(np.arange(0, 101, 1), 'speed')      # Antecedent 前提变量
speed['slow'] = fuzz.trimf(speed.universe, [0, 0, 50])
speed['fast'] = fuzz.trimf(speed.universe, [50, 100, 100])

# Define the membership functions for the output variable
# brake = ctrl.Consequent(np.arange(0, 11, 1), 'brake')       # Consequent 结论变量
brake['low'] = fuzz.trimf(brake.universe, [0, 0, 5])
brake['medium'] = fuzz.trimf(brake.universe, [0, 5, 10])
brake['high'] = fuzz.trimf(brake.universe, [5, 10, 10])

# Define the rules
rule1 = ctrl.Rule(distance['near'] | speed['fast'], brake['high'])
rule2 = ctrl.Rule(distance['medium'] & speed['slow'], brake['medium'])
rule3 = ctrl.Rule(distance['far'] & speed['slow'], brake['low'])

# Create a control system with the rules
brake_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
brake_simulation = ctrl.ControlSystemSimulation(brake_ctrl)

# Pass in inputs and simulate
brake_simulation.input['distance'] = 7
brake_simulation.input['speed'] = 60
brake_simulation.compute()

# Print the output
print(brake_simulation.output['brake'])