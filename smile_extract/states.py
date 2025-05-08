import pandas as pd

def get_trial_state_table(smile_trial: dict) -> pd.DataFrame:
    # return pd.from_dict(
    #     smile_trial['Parameters']['StateTable'],
    # )
    pass

def get_trial_state_transition_table(smile_trial: dict) -> pd.DataFrame:
    state_transition_table = (
        pd.DataFrame(smile_trial['Parameters']['StateTable'])
        .rename(columns={
            'stateName':'state',
        })
        .assign(**{
            'pass state': lambda x: x['windowPassState'].map(lambda y: x['state'].iloc[y-1]),
            'fail state': lambda x: x['windowFailState'].map(lambda y: x['state'].iloc[y-1]),
            'interval': lambda x: x['Interval'].map(lambda y: y['name']),
            'target': lambda x: x['Hand'].map(lambda y: y['targetName']),
            'target window check type': lambda x: x['Hand'].map(lambda y: y['targetWindowCheckType']),
            'pass condition': lambda x: x['target window check type'].map({1:'enter',2:'hold'}),
        })
        .set_index('state')
        [['target','pass condition','interval','pass state','fail state']]
    )

    return state_transition_table

def get_trial_events(smile_trial: dict) -> pd.DataFrame:
    def get_state_name(state_id):
        if state_id == -1:
            return 'end'
        else:
            state_name = smile_trial['Parameters']['StateTable'][state_id-1]['stateName']
            if isinstance(state_name, list) and len(state_name) == 1:
                state_name = state_name[0]
            return state_name

    return (
        pd.DataFrame(
            [
                {
                    'event': get_state_name(state_id),
                    'time': pd.to_timedelta(float(event_frame-1), unit='ms'),
                }
                for [state_id,event_frame] in smile_trial['TrialData']['stateTransitions'].T
            ],
        )
        .set_index('event')
    )

def get_trial_states(smile_trial: dict, bin_size: str='10ms') -> pd.Series:
    return (
        get_trial_events(smile_trial)
        .reset_index(level='event')
        .set_index('time')
        .groupby('time')
        .last()
        .resample(bin_size)
        .ffill()
        .squeeze()
        .rename('')
    )