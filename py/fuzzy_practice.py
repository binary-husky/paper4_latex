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
    input_arr=[ 0,1,2, 2,1,0 ], 
    antecedent=win_rate, 
    consequent_arr=[lr_log_multiplier, ppo_epoch_floating], 
    member_ship = ['small', 'medium', 'large']
)

controller = ctrl.ControlSystem(rule_list)
feedback_sys = ctrl.ControlSystemSimulation(controller)

feedback_sys.input['win_rate'] = 0.5
feedback_sys.compute()

print(feedback_sys.output['lr_log_multiplier'], feedback_sys.output['ppo_epoch_floating'])

lr_log_multiplier = feedback_sys.output['lr_log_multiplier']
ppo_epoch_floating = feedback_sys.output['ppo_epoch_floating']

lr_multiplier = 10 ** lr_log_multiplier
ppo_epoch = int(ppo_epoch_floating)

# rule1 = ctrl.Rule(distance['near'] | speed['fast'], brake['high'])
# rule2 = ctrl.Rule(distance['medium'] & speed['slow'], brake['medium'])
# rule3 = ctrl.Rule(distance['far'] & speed['slow'], brake['low'])



lr = 10**lr_log_multiplier

# lr_log_multiplier['small']    = fuzz.trimf(lr_log_multiplier.universe, [0.1,     0,    10 ])
# lr_log_multiplier['medium']   = fuzz.trimf(lr_log_multiplier.universe, [0,     0.1,    1  ])

