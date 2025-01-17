import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
def projection(x, from_range, to_range):
    return ((x - from_range[0]) * (to_range[1] - to_range[0]) / (from_range[1] - from_range[0])) + to_range[0]

def gen_antecedent(key, min, max):
    d = (max - min) / 1000
    antecedent = ctrl.Antecedent(np.arange(min, max+1e-10, d), key) # Antecedent 前提变量 [0 ~ 1]
    names = ['small', 'medium', 'large']
    antecedent.automf(names=names)
    # for mfn in  ['very small', 'small', 'medium', 'large', 'very large']:
    #     plt.plot(antecedent.universe, antecedent[mfn].mf, linewidth=1.5, label=mfn)
    # plt.title(f'Membership functions of {key}')
    # plt.legend()
    # plt.show()
    return antecedent

def gen_consequent(key, min, max, defuzzify_method='centroid', consequent_num_mf=5):
    d = (max - min) / 1000
    consequent = ctrl.Consequent(np.arange(min, max+1e-10, d), key, defuzzify_method=defuzzify_method) # consequent 前提变量 [0 ~ 1]
    if consequent_num_mf==5:
        names = ['very small', 'small', 'medium', 'large', 'very large']
        consequent.automf(names=names)
    elif consequent_num_mf==7:
        names = ['extreme small', 'very small', 'small', 'medium', 'large', 'very large', 'extreme large']
        consequent.automf(names=names)
    else:
        assert False

    # for mfn in  ['very small', 'small', 'medium', 'large', 'very large']:
    #     plt.plot(consequent.universe, consequent[mfn].mf, linewidth=1.5, label=mfn)
    # plt.title(f'Membership functions of {key}')
    # plt.legend()
    # plt.show()
    return consequent
    
# input_arr = [consequent_1_select, consequent_2_select, consequent_3_select]
def gen_rule_list(input_arr, antecedent, consequent_arr):
    antecedent_membership = [t for t in antecedent.terms]
    consequent_membership = [t for t in consequent_arr[0].terms]    # assume all consequent has same memberships
    # try:
    assert len(antecedent_membership) * len(consequent_arr) == len(input_arr)
    # except:
        # pass
    rule_list = []
    p = 0
    for consequent in consequent_arr:
        for k in antecedent_membership:
            rule_list.append(ctrl.Rule(antecedent[k],  consequent[consequent_membership[input_arr[p]]])); p += 1
    assert p == len(input_arr)
    return rule_list


def gen_feedback_sys_generic(
        antecedent_key,
        antecedent_min,
        antecedent_max,
        consequent_key,
        consequent_min,
        consequent_max,
        consequent_num_mf,
        fuzzy_controller_param, 
        defuzzify_method='centroid',
        compute_fn=None
    ):

    # input [-1.0, +1.0] ---> fuzzy
    fuzzy_antecedent = gen_antecedent(key=antecedent_key, min=antecedent_min, max=antecedent_max)
    
    # defuzzy --> [-1.5, +1.5] --> [-31.6, 0.0316]
    fuzzy_consequent = gen_consequent(key=consequent_key, min=consequent_min, max=consequent_max, defuzzify_method=defuzzify_method, consequent_num_mf=consequent_num_mf)

    rule_list = gen_rule_list(
        input_arr=fuzzy_controller_param, 
        antecedent=fuzzy_antecedent, 
        consequent_arr=[fuzzy_consequent], 
    )

    controller = ctrl.ControlSystem(rule_list)
    feedback_sys = ctrl.ControlSystemSimulation(controller)
    feedback_sys.compute_fn = lambda x: compute_fn(feedback_sys, x)

    feedback_sys.register_input = fuzzy_antecedent
    feedback_sys.register_output = fuzzy_consequent

    return feedback_sys

def int2consq(a, fuzzy_consequent):
    q = ['extreme small', 'very small', 'small', 'medium', 'large', 'very large', 'extreme large']
    return fuzzy_consequent[q[a]]

def gen_feedback_sys_generic_multi_input(
        antecedent_list,
        consequent_key,
        consequent_min,
        consequent_max,
        consequent_num_mf,
        fuzzy_controller_param, 
        defuzzify_method='centroid',
        compute_fn=None
    ):
    fuzzy_antecedents = {}
    for antecedent_key, antecedent_min, antecedent_max in antecedent_list:
        fuzzy_antecedent = gen_antecedent(key=antecedent_key, min=antecedent_min, max=antecedent_max)
        fuzzy_antecedents[antecedent_key] = fuzzy_antecedent

 
    # defuzzy --> [-1.5, +1.5] --> [-31.6, 0.0316]
    fuzzy_consequent = gen_consequent(key=consequent_key, min=consequent_min, max=consequent_max, defuzzify_method=defuzzify_method, consequent_num_mf=consequent_num_mf)
    
    # consequent_key='intrinsic_reward',
    rule_list = []
    # names = ['extreme small', 'very small', 'small', 'medium', 'large', 'very large', 'extreme large']
    # Check to make sure this set of input values will activate at least one connected Term in each Antecedent via the current set of Rules.
    rule_list.append(ctrl.Rule(
        fuzzy_antecedents['lifelen_norm']['small'] & fuzzy_antecedents['recent_winrate']['small'] ,  int2consq(fuzzy_controller_param[0], fuzzy_consequent)
    ))
    rule_list.append(ctrl.Rule(
        fuzzy_antecedents['lifelen_norm']['medium'] & fuzzy_antecedents['recent_winrate']['small'],  int2consq(fuzzy_controller_param[1], fuzzy_consequent)
    ))
    rule_list.append(ctrl.Rule(
        fuzzy_antecedents['lifelen_norm']['large'] & fuzzy_antecedents['recent_winrate']['small'] ,  int2consq(fuzzy_controller_param[2], fuzzy_consequent)
    ))

    rule_list.append(ctrl.Rule(
        fuzzy_antecedents['lifelen_norm']['small'] & fuzzy_antecedents['recent_winrate']['medium'] ,  int2consq(fuzzy_controller_param[3], fuzzy_consequent)
    ))
    rule_list.append(ctrl.Rule(
        fuzzy_antecedents['lifelen_norm']['medium'] & fuzzy_antecedents['recent_winrate']['medium'],  int2consq(fuzzy_controller_param[4], fuzzy_consequent)
    ))
    rule_list.append(ctrl.Rule(
        fuzzy_antecedents['lifelen_norm']['large'] & fuzzy_antecedents['recent_winrate']['medium'] ,  int2consq(fuzzy_controller_param[5], fuzzy_consequent)
    ))

    rule_list.append(ctrl.Rule(
        fuzzy_antecedents['lifelen_norm']['small'] & fuzzy_antecedents['recent_winrate']['large'] ,  int2consq(fuzzy_controller_param[6], fuzzy_consequent)
    ))
    rule_list.append(ctrl.Rule(
        fuzzy_antecedents['lifelen_norm']['medium'] & fuzzy_antecedents['recent_winrate']['large'],  int2consq(fuzzy_controller_param[7], fuzzy_consequent)
    ))
    rule_list.append(ctrl.Rule(
        fuzzy_antecedents['lifelen_norm']['large'] & fuzzy_antecedents['recent_winrate']['large'] ,  int2consq(fuzzy_controller_param[8], fuzzy_consequent)
    ))

    controller = ctrl.ControlSystem(rule_list)
    feedback_sys = ctrl.ControlSystemSimulation(controller)
    feedback_sys.compute_fn = lambda x: compute_fn(feedback_sys, x)

    feedback_sys.register_input = fuzzy_antecedent
    feedback_sys.register_output = fuzzy_consequent

    return feedback_sys



fuzzy_controller_param = [1, 2, 5, 4, 2, 1, 6, 2, 3]
# generate fuzzy control
def fuzzy_compute(feedback_sys, antecedents ):
    lifelen_norm = antecedents[0]
    recent_winrate = antecedents[1]
    feedback_sys.input['lifelen_norm'] = lifelen_norm
    feedback_sys.input['recent_winrate'] = recent_winrate
    feedback_sys.compute()
    intrinsic_reward = feedback_sys.output['intrinsic_reward']
    return intrinsic_reward

# expected input [-1, +1], 
# expected fuzzy output [-1.5*_scale, +1.5*_scale]
# expected final output [10^(-1.5*_scale),10^(+1.5*_scale)]
feedback_sys_agent_wise = gen_feedback_sys_generic_multi_input(
    antecedent_list = [
        ('lifelen_norm', -1.5, 1.5),
        ('recent_winrate', 0.2, 0.8),
    ],
    consequent_key='intrinsic_reward',
    consequent_min=-5,
    consequent_max=+5,
    fuzzy_controller_param=fuzzy_controller_param,
    consequent_num_mf=7,
    compute_fn=fuzzy_compute
)



test_input_1 = np.arange(-1.5, 1.5, 0.05)
test_input_2 = np.arange(0, 1, 0.05)
X, Y = np.meshgrid(test_input_1, test_input_2)

Z = np.array(list(map(lambda xi, yi: 
    feedback_sys_agent_wise.compute_fn((xi, yi)), 
    X.flatten(), Y.flatten()))).reshape(X.shape)

feedback_sys_agent_wise.register_input.view(sim=feedback_sys_agent_wise)
feedback_sys_agent_wise.register_output.view(sim=feedback_sys_agent_wise)

# Z = np.array([feedback_sys_agent_wise.compute_fn((x, y)) for x,y in zip(X,Y)])
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z)
# Set the axis labels
ax.set_xlabel('lifelen norm')
ax.set_ylabel('recent winrate')
ax.set_zlabel('intrinsic reward')

# Set the plot title
# plt.title('3D Surface Plot')

# Show the plot
plt.show()
# plt.plot(test_input, out[:, 0]


