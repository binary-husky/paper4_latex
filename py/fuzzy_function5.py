
def projection(x, from_range, to_range):
    return ((x - from_range[0]) * (to_range[1] - to_range[0]) / (from_range[1] - from_range[0])) + to_range[0]


def gen_feedback_sys(fuzzy_controller_param, scale_param):
    import numpy as np
    import skfuzzy as fuzz
    from skfuzzy import control as ctrl
    import matplotlib.pyplot as plt
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
    
    win_rate = gen_antecedent(key='win_rate', min=0.0, max=1.0)
    
    # scale[0]: std 0~1
    s1 = projection(x=scale_param[0], from_range=[0,1.0], to_range=[1e-8, 1.0])   # avoid 0 div
    lr_log_multiplier = gen_consequent(key='lr_log_multiplier', min=-2.5*s1, max=+2.5*s1)

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

    rule_list = gen_rule_list(
        input_arr=fuzzy_controller_param, 
        antecedent=win_rate, 
        consequent_arr=[lr_log_multiplier], 
        member_ship = ['very small', 'small', 'medium', 'large', 'very large']
    )
    controller = ctrl.ControlSystem(rule_list)
    # controller.view()
    feedback_sys = ctrl.ControlSystemSimulation(controller)
    return feedback_sys

def fuzzy_compute(feedback_sys, win_rate_actual):
    feedback_sys.input['win_rate'] = win_rate_actual
    feedback_sys.compute()
        
    lr_log_multiplier = feedback_sys.output['lr_log_multiplier']
    # ppo_epoch_floating = feedback_sys.output['ppo_epoch_floating']

    lr_multiplier = 10 ** lr_log_multiplier
    # lr_multiplier = lr_log_multiplier
    # ppo_epoch = int(ppo_epoch_floating)
    return lr_multiplier #, ppo_epoch


fuzzy_controller_param = [0,0,0,0,0]
scale_param = [1]
feedback_sys = gen_feedback_sys(fuzzy_controller_param, scale_param)



import numpy as np
wr_list = np.arange(0,1+1e-9,0.01)
c1_list = np.arange(0,1+1e-9,0.01)
c2_list = np.arange(0,1+1e-9,0.01)


for i, wr in enumerate(wr_list):
    c1_list[i] = fuzzy_compute(feedback_sys, wr)


import matplotlib.pyplot as plt
print(c1_list)
axes = plt.plot(wr_list, c1_list)
from matplotlib.ticker import ScalarFormatter
y_formatter = ScalarFormatter(useOffset=False)
plt.gcf().axes[0].yaxis.set_major_formatter(y_formatter)
plt.tight_layout()
plt.legend()
plt.show()


##