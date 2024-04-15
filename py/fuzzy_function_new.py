import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

def projection(x, from_range, to_range):
    return ((x - from_range[0]) * (to_range[1] - to_range[0]) / (from_range[1] - from_range[0])) + to_range[0]


def gen_antecedent(key, min, max):
    d = (max - min) / 100
    antecedent = ctrl.Antecedent(np.arange(min, max+1e-10, d), key) # Antecedent 前提变量 [0 ~ 1]
    default_mf1 = np.array([-0.25, 0,    0.25]); default_mf1 = projection(x=default_mf1, from_range=[0,1], to_range=[min, max])
    default_mf2 = np.array([0,     0.25,  0.5]); default_mf2 = projection(x=default_mf2, from_range=[0,1], to_range=[min, max])
    default_mf3 = np.array([0.25, 0.5,   0.75]); default_mf3 = projection(x=default_mf3, from_range=[0,1], to_range=[min, max])
    default_mf4 = np.array([0.5,  0.75,     1]); default_mf4 = projection(x=default_mf4, from_range=[0,1], to_range=[min, max])
    default_mf5 = np.array([0.75, 1,     1.25]); default_mf5 = projection(x=default_mf5, from_range=[0,1], to_range=[min, max])
    antecedent['very small']    = fuzz.trimf(antecedent.universe, default_mf1)
    antecedent['small']         = fuzz.trimf(antecedent.universe, default_mf2)
    antecedent['medium']        = fuzz.trimf(antecedent.universe, default_mf3)
    antecedent['large']         = fuzz.trimf(antecedent.universe, default_mf4)
    antecedent['very large']    = fuzz.trimf(antecedent.universe, default_mf5)

    # for mfn in  ['very small', 'small', 'medium', 'large', 'very large']:
    #     plt.plot(antecedent.universe, antecedent[mfn].mf, linewidth=1.5, label=mfn)
    # plt.title(f'Membership functions of {key}')
    # plt.legend()
    # plt.show()
    # antecedent['medium'].view()

    return antecedent

def gen_consequent(key, min, max):
    d = (max - min) / 100
    consequent = ctrl.Consequent(np.arange(min, max+1e-10, d), key) # consequent 前提变量 [0 ~ 1]
    default_mf1 = np.array([-0.25, 0,    0.25]); default_mf1 = projection(x=default_mf1, from_range=[0,1], to_range=[min, max])
    default_mf2 = np.array([0,     0.25,  0.5]); default_mf2 = projection(x=default_mf2, from_range=[0,1], to_range=[min, max])
    default_mf3 = np.array([0.25, 0.5,   0.75]); default_mf3 = projection(x=default_mf3, from_range=[0,1], to_range=[min, max])
    default_mf4 = np.array([0.5,  0.75,     1]); default_mf4 = projection(x=default_mf4, from_range=[0,1], to_range=[min, max])
    default_mf5 = np.array([0.75, 1,     1.25]); default_mf5 = projection(x=default_mf5, from_range=[0,1], to_range=[min, max])
    consequent['very small']    = fuzz.trimf(consequent.universe, default_mf1)
    consequent['small']         = fuzz.trimf(consequent.universe, default_mf2)
    consequent['medium']        = fuzz.trimf(consequent.universe, default_mf3)
    consequent['large']         = fuzz.trimf(consequent.universe, default_mf4)
    consequent['very large']    = fuzz.trimf(consequent.universe, default_mf5)

    # for mfn in  ['very small', 'small', 'medium', 'large', 'very large']:
    #     plt.plot(consequent.universe, consequent[mfn].mf, linewidth=1.5, label=mfn)
    # plt.title(f'Membership functions of {key}')
    # plt.legend()
    # plt.show()

    return consequent
    
# input_arr = [consequent_1_select, consequent_2_select, consequent_3_select]
def gen_rule_list(input_arr, antecedent, consequent_arr, member_ship = ['small', 'medium', 'large']):
    assert len(consequent_arr) * len(member_ship) == len(input_arr)
    rule_list = []
    p = 0
    for consequent in consequent_arr:
        for k in member_ship:
            # print(f'antecedent {k}, consequent {member_ship[input_arr[p]]}')
            rule_list.append(ctrl.Rule(antecedent[k],  consequent[member_ship[input_arr[p]]])); p += 1
    assert p == len(input_arr)
    return rule_list

def gen_feedback_sys(fuzzy_controller_param, scale_param):

    win_rate = gen_antecedent(key='win_rate', min=0.0, max=1.0)
    
    # scale[0]: std 0~1
    s1 = projection(x=scale_param[0], from_range=[0,1.0], to_range=[1e-8, 1.0])   # avoid 0 div
    lr_log_multiplier = gen_consequent(key='lr_log_multiplier', min=-2.5*s1, max=+2.5*s1)

    rule_list = gen_rule_list(
        input_arr=fuzzy_controller_param, 
        antecedent=win_rate, 
        consequent_arr=[lr_log_multiplier], 
        member_ship = ['very small', 'small', 'medium', 'large', 'very large']
    )
    controller = ctrl.ControlSystem(rule_list)
    # controller.view()
    feedback_sys = ctrl.ControlSystemSimulation(controller)
    feedback_sys.register_input = win_rate
    feedback_sys.register_output = lr_log_multiplier
    return feedback_sys

def gen_feedback_sys_agent_wise(fuzzy_controller_param, scale_param):

    # input [-1.0, +1.0] ---> fuzzy
    agent_life = gen_antecedent(key='agent_life_std', min=-1.0, max=1.0)
    
    # scale[0]: 0 ~ 1
    s2 = projection(x=scale_param[0], from_range=[0,1.0], to_range=[1e-8, 1.0])   # avoid 0 div

    # defuzzy --> [-1.5, +1.5] --> [-31.6, 0.0316]
    adv_log_multiplier = gen_consequent(key='adv_log_multiplier', min=-1.5*s2, max=+1.5*s2)

    rule_list = gen_rule_list(
        input_arr=fuzzy_controller_param, 
        antecedent=agent_life, 
        consequent_arr=[adv_log_multiplier], 
        member_ship = ['very small', 'small', 'medium', 'large', 'very large']
    )

    controller = ctrl.ControlSystem(rule_list)
    
    feedback_sys = ctrl.ControlSystemSimulation(controller)
    return feedback_sys

def fuzzy_compute(feedback_sys, win_rate_actual):
    feedback_sys.input['win_rate'] = win_rate_actual
    feedback_sys.compute()
    lr_log_multiplier = feedback_sys.output['lr_log_multiplier']
    lr_multiplier = 10 ** lr_log_multiplier
    return lr_multiplier

def fuzzy_compute2(feedback_sys, agent_life_std, is_array=False):
    def compute_single_fuzzy(feedback_sys, agent_life_std):
        feedback_sys.input['agent_life_std'] = agent_life_std
        feedback_sys.compute()
        adv_log_multiplier = feedback_sys.output['adv_log_multiplier']
        adv_multiplier = 10 ** adv_log_multiplier
        return adv_multiplier
    
    if is_array:
        return np.array([compute_single_fuzzy(feedback_sys, a) for a in agent_life_std])
    else:
        return compute_single_fuzzy(feedback_sys, agent_life_std)
    

fuzzy_controller_param = [0,0,0,0,0]
scale_param = [1]
feedback_sys = gen_feedback_sys(fuzzy_controller_param, scale_param)
feedback_sys.input['win_rate'] = 0.1
feedback_sys.compute()
lr_log_multiplier = feedback_sys.output['lr_log_multiplier']
feedback_sys.register_input.view(sim=feedback_sys)
feedback_sys.register_output.view(sim=feedback_sys)
plt.show()