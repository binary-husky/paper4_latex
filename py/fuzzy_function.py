
def gen_feedback_sys(fuzzy_controller_param):
    import numpy as np
    import skfuzzy as fuzz
    from skfuzzy import control as ctrl
    import matplotlib.pyplot as plt
    def gen_antecedent(key, min, max):
        d = (max - min) / 10
        antecedent = ctrl.Antecedent(np.arange(min, max+1e-10, d), key) # Antecedent 前提变量 [0 ~ 1]

        antecedent['small']   = fuzz.trimf(antecedent.universe, [min,             min,              min/2 + max/2   ])
        antecedent['medium']  = fuzz.trimf(antecedent.universe, [min,             min/2 + max/2,    max             ])
        antecedent['large']   = fuzz.trimf(antecedent.universe, [min/2 + max/2,   max,              max             ])
        # plt.plot(antecedent.universe, antecedent['small'].mf, 'b', linewidth=1.5, label='small')
        # plt.plot(antecedent.universe, antecedent['medium'].mf, 'g', linewidth=1.5, label='medium')
        # plt.plot(antecedent.universe, antecedent['large'].mf, 'r', linewidth=1.5, label='large')
        # plt.title(f'Membership functions of {key}')
        # plt.legend()
        # plt.show()
        return antecedent

    def gen_consequent(key, min, max):
        d = (max - min) / 10
        consequent = ctrl.Consequent(np.arange(min, max+1e-10, d), key) # consequent 前提变量 [0 ~ 1]

        consequent['small']   = fuzz.trimf(consequent.universe, [min,             min,              min/2 + max/2   ])
        consequent['medium']  = fuzz.trimf(consequent.universe, [min,             min/2 + max/2,    max             ])
        consequent['large']   = fuzz.trimf(consequent.universe, [min/2 + max/2,   max,              max             ])
        return consequent


    win_rate      = gen_antecedent(key='win_rate',      min=0, max=1)
    lr_log_multiplier = gen_consequent(key='lr_log_multiplier', min=-1.0, max=+1.0)
    ppo_epoch_floating = gen_consequent(key='ppo_epoch_floating', min=4, max=32)


    # input_arr = [consequent_1_select, consequent_2_select, consequent_3_select]
    def gen_rule_list(input_arr, antecedent, consequent_arr, member_ship = ['small', 'medium', 'large']):
        assert len(consequent_arr) * len(['small', 'medium', 'large']) == len(input_arr)
        rule_list = []
        p = 0
        for consequent in consequent_arr:
            rule_list.append(ctrl.Rule(antecedent['small'],  consequent[member_ship[input_arr[p]]])) ; p += 1
            rule_list.append(ctrl.Rule(antecedent['medium'],  consequent[member_ship[input_arr[p]]])) ; p += 1
            rule_list.append(ctrl.Rule(antecedent['large'],  consequent[member_ship[input_arr[p]]])) ; p += 1
        assert p == len(input_arr)
        return rule_list

    rule_list = gen_rule_list(
        input_arr=fuzzy_controller_param, 
        antecedent=win_rate, 
        consequent_arr=[lr_log_multiplier, ppo_epoch_floating], 
        member_ship = ['small', 'medium', 'large']
    )
    controller = ctrl.ControlSystem(rule_list)
    feedback_sys = ctrl.ControlSystemSimulation(controller)
    return feedback_sys

def fuzzy_compute(feedback_sys, win_rate_actual):
    feedback_sys.input['win_rate'] = win_rate_actual
    feedback_sys.compute()
        
    lr_log_multiplier = feedback_sys.output['lr_log_multiplier']
    ppo_epoch_floating = feedback_sys.output['ppo_epoch_floating']

    lr_multiplier = 10 ** lr_log_multiplier
    ppo_epoch = int(ppo_epoch_floating)
    return lr_multiplier, ppo_epoch

fuzzy_controller_param = [0,1,2,2,1,0]
feedback_sys = gen_feedback_sys(fuzzy_controller_param)
import numpy as np
wr_list = np.arange(0,1,0.1)
c1_list = np.arange(0,1,0.1)
c2_list = np.arange(0,1,0.1)


for i, wr in enumerate(wr_list):
    c1_list[i], c2_list[i] = fuzzy_compute(feedback_sys, wr)


import matplotlib.pyplot as plt
plt.plot(wr_list, c1_list)
plt.plot(wr_list, c2_list)
plt.legend()
plt.show()