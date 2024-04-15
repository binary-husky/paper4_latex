import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

def projection(x, from_range, to_range):
    return ((x - from_range[0]) * (to_range[1] - to_range[0]) / (from_range[1] - from_range[0])) + to_range[0]


def gen_antecedent(key, min, max):
    d = (max - min) / 1000
    antecedent = ctrl.Antecedent(np.arange(min-0.5, max+0.5, d), key) # Antecedent 前提变量 [0 ~ 1]
    default_mf1 = np.array([-100, -100, 0,    0.25]); default_mf1 = projection(x=default_mf1, from_range=[0,1], to_range=[min, max])
    default_mf2 = np.array([0,     0.25,  0.5]); default_mf2 = projection(x=default_mf2, from_range=[0,1], to_range=[min, max])
    default_mf3 = np.array([0.25, 0.5,   0.75]); default_mf3 = projection(x=default_mf3, from_range=[0,1], to_range=[min, max])
    default_mf4 = np.array([0.5,  0.75,     1]); default_mf4 = projection(x=default_mf4, from_range=[0,1], to_range=[min, max])
    default_mf5 = np.array([0.75, 1,     100, 100]); default_mf5 = projection(x=default_mf5, from_range=[0,1], to_range=[min, max])
    antecedent['very small']    = fuzz.trapmf(antecedent.universe, default_mf1)
    antecedent['small']         = fuzz.trimf(antecedent.universe, default_mf2)
    antecedent['medium']        = fuzz.trimf(antecedent.universe, default_mf3)
    antecedent['large']         = fuzz.trimf(antecedent.universe, default_mf4)
    antecedent['very large']    = fuzz.trapmf(antecedent.universe, default_mf5)
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

    assert len(antecedent_membership) * len(consequent_arr) == len(input_arr)
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


fuzzy_controller_param = [6,4,3,2,0, 6,4,3,2,0]
scale_param = [1]
wr_input_expand = projection(x=scale_param[0], from_range=[0,  1.0], to_range=[0.1,  0.5])
def compute_output_ppo_epoch_floating(feedback_sys, winrate):
    feedback_sys.input['winrate'] = winrate
    feedback_sys.compute()
    ppo_epoch_floating = feedback_sys.output['ppo_epoch_floating']
    return int(ppo_epoch_floating), ppo_epoch_floating
fc_1 = gen_feedback_sys_generic(
    antecedent_key='winrate',
    antecedent_min=0.5-wr_input_expand,
    antecedent_max=0.5+wr_input_expand,
    consequent_key='ppo_epoch_floating',
    consequent_min=16 - 12,
    consequent_max=16 + 12,
    consequent_num_mf=7,
    fuzzy_controller_param=fuzzy_controller_param[0:5],
    compute_fn=compute_output_ppo_epoch_floating
)


## --- <2>  --- ##
# expected input [0, +1], 
# expected fuzzy output [1, 15] floating
# expected final output [4, 60] int
def compute_output_trajnum_multiplier_floating(feedback_sys, winrate):
    feedback_sys.input['winrate'] = winrate
    feedback_sys.compute()
    trajnum_multiplier_floating = feedback_sys.output['trajnum_multiplier_floating']
    return int(trajnum_multiplier_floating * 4), trajnum_multiplier_floating
fc_2 = gen_feedback_sys_generic(
    antecedent_key='winrate',
    antecedent_min=0.5-wr_input_expand,
    antecedent_max=0.5+wr_input_expand,
    consequent_key='trajnum_multiplier_floating',
    consequent_min=8 - 7,
    consequent_max=8 + 7,
    consequent_num_mf=7,
    fuzzy_controller_param=fuzzy_controller_param[5:10],
    compute_fn=compute_output_trajnum_multiplier_floating
)

test_input = np.arange(0-0.5, 1+0.5, 0.01)

out = np.array([fc_1.compute_fn(inp) for inp in test_input])
plt.plot(test_input, out[:, 0])
plt.plot(test_input, out[:, 1])

out = np.array([fc_2.compute_fn(inp) for inp in test_input])

plt.plot(test_input, out[:, 0])
plt.plot(test_input, out[:, 1])

fc_1.register_input.view(sim=fc_1)
fc_1.register_output.view(sim=fc_1)

from matplotlib.ticker import ScalarFormatter
y_formatter = ScalarFormatter(useOffset=False)
plt.gcf().axes[0].yaxis.set_major_formatter(y_formatter)
plt.tight_layout()
plt.legend()
plt.show()














# # avoid 0 div
# _scale = projection(x=scale_param[0], from_range=[0,1.0], to_range=[1e-8, 1.0])
# # generate fuzzy control
# def fuzzy_compute(feedback_sys, agent_life_std):
#     feedback_sys.input['agent_life_std'] = agent_life_std
#     feedback_sys.compute()
#     adv_log_multiplier = feedback_sys.output['adv_log_multiplier']
#     adv_multiplier = 10 ** adv_log_multiplier
#     return adv_multiplier

# feedback_sys_agent_wise = gen_feedback_sys_generic(
#     antecedent_key='agent_life_std',
#     antecedent_min=-1.0,
#     antecedent_max=+1.0,
#     consequent_key='adv_log_multiplier',
#     consequent_min=-1.5*_scale,
#     consequent_max=+1.5*_scale,
#     consequent_num_mf=5,
#     fuzzy_controller_param=fuzzy_controller_param,
#     compute_fn=fuzzy_compute
# )

# import numpy as np
# agent_life_std = np.arange(-1, 1+1e-9, 0.01)
# res = [feedback_sys_agent_wise.compute_fn(a) for a in agent_life_std]
# plt.plot(agent_life_std, res)
# feedback_sys_agent_wise.input['win_rate'] = 0.1
# feedback_sys_agent_wise.compute()
# adv_log_multiplier = feedback_sys_agent_wise.output['adv_log_multiplier']
# feedback_sys_agent_wise.register_input.view(sim=feedback_sys_agent_wise)
# feedback_sys_agent_wise.register_output.view(sim=feedback_sys_agent_wise)
# from matplotlib.ticker import ScalarFormatter
# y_formatter = ScalarFormatter(useOffset=False)
# plt.gcf().axes[0].yaxis.set_major_formatter(y_formatter)
# plt.tight_layout()
# plt.legend()
# plt.show()








# def fuzzy_compute(feedback_sys, win_rate_actual):
#     feedback_sys.input['win_rate'] = win_rate_actual
#     feedback_sys.compute()
#     lr_log_multiplier = feedback_sys.output['lr_log_multiplier']
#     lr_multiplier = 10 ** lr_log_multiplier
#     return lr_multiplier

# def fuzzy_compute2(feedback_sys, agent_life_std, is_array=False):
#     def compute_single_fuzzy(feedback_sys, agent_life_std):
#         feedback_sys.input['agent_life_std'] = agent_life_std
#         feedback_sys.compute()
#         adv_log_multiplier = feedback_sys.output['adv_log_multiplier']
#         adv_multiplier = 10 ** adv_log_multiplier
#         return adv_multiplier
    
#     if is_array:
#         return np.array([compute_single_fuzzy(feedback_sys, a) for a in agent_life_std])
#     else:
#         return compute_single_fuzzy(feedback_sys, agent_life_std)
    
# fuzzy_controller_param = [0,1,2,3,4]
# scale_param = [1]
# feedback_sys_agent_wise = gen_feedback_sys_agent_wise(fuzzy_controller_param, scale_param, defuzzify_method='centroid')
"""
mode : string
    Controls which defuzzification method will be used.
    * 'centroid': Centroid of area
    * 'bisector': bisector of area
    * 'mom'     : mean of maximum
    * 'som'     : min of maximum
    * 'lom'     : max of maximum
"""
####################################################################################