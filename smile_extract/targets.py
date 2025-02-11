import pandas as pd
import numpy as np

def parse_smile_meta(td, trial, meta={}, num_random_targs=8):
    trial_name = trial['Overview']['task']
    ct_reach_state = next(state for state in trial['Parameters']['StateTable'] if state['stateName'] == 'Reach to Center')
    center_target_idx = [i for i, name in enumerate(ct_reach_state['StateTargets']['names']) if name in ['starttarget', 'start']]
    td['ct_location'] = ct_reach_state['StateTargets']['location'][center_target_idx[0]]
    td['ct_location'][1] = -td['ct_location'][1]

    if trial_name.startswith('RandomTargetTask_20220630'):
        td['rt_locations'] = np.zeros((num_random_targs, 3))
        for targetnum in range(num_random_targs):
            targ_reach_state = next(state for state in trial['Parameters']['StateTable'] if state['stateName'] == f'Reach to Target {targetnum}')
            targ_idx = next(i for i, name in enumerate(targ_reach_state['StateTargets']['names']) if name == f'randomtarg{targetnum}')
            td['rt_locations'][targetnum] = targ_reach_state['StateTargets']['location'][targ_idx]
        td['rt_locations'][:, 1] = -td['rt_locations'][:, 1]

    if 'CST' in trial_name:
        td['lambda'] = trial['Parameters']['ForceParameters']['initialLambda']

    for key, value in meta.items():
        td[key] = value

    return td

def get_trial_targets(smile_trial: dict) -> pd.DataFrame:
    pass
