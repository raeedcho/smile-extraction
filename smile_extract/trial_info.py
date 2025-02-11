from datetime import datetime
import re
import numpy as np
import pandas as pd

def get_trial_id(smile_trial: dict) -> int:
    return int(smile_trial['Overview']['trialNumber'].replace('Trial',''))

def get_trial_datetime_str(smile_trial: dict) -> str:
    return (
        datetime.strptime(
            str(int(smile_trial['Overview']['date'])),
            '%Y%m%d%H%M%S',
        ).strftime('%Y-%m-%d %H:%M:%S')
    )

def get_trial_result(smile_trial: dict) -> str:
    result_code = smile_trial['Overview']['trialStatus']
    if type(result_code) is np.ndarray:
        result_code = result_code.item()

    return (
        {
            0: 'failure',
            1: 'success',
            2: 'abort',
        }[result_code]
    )

def get_trial_task(smile_trial: dict) -> str:
    trial_name = smile_trial['Overview']['trialName']
    if trial_name.startswith('RandomTargetTask'):
        task = 'RTT'
    elif trial_name.startswith('CST'):
        task = 'CST'
    elif trial_name.startswith('CenterOut'):
        task = 'CO'
    elif re.match(r'^R.T.$', trial_name):
        task = 'DCO'
    elif re.match(r'^R.T.C$', trial_name):
        task = 'DCO-catch'
    else:
        task = trial_name

    return task

def get_smile_meta(smile_data: list, block: str='', **kwargs) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                'monkey': smile_trial['Overview']['subjectName'],
                'date_time': get_trial_datetime_str(smile_trial),
                'task': get_trial_task(smile_trial),
                'result': get_trial_result(smile_trial),
                'block': block,
            }
            for smile_trial in smile_data
        ],
        index=pd.Index(
            [get_trial_id(smile_trial) for smile_trial in smile_data],
            name='trial_id',
        )
    )