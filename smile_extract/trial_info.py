from datetime import datetime
import numpy as np
import pandas as pd

def get_trial_id(smile_trial: dict) -> int:
    return int(smile_trial['Overview']['trialNumber'].replace('Trial',''))

def get_trial_datetime(smile_trial: dict) -> pd.Timestamp:
    return pd.to_datetime(
        datetime.strptime(
            str(int(smile_trial['Overview']['date'])),
            '%Y%m%d%H%M%S',
        ).strftime('%Y-%m-%d %H:%M:%S')
    )

def get_trial_session_date(smile_trial: dict) -> pd.Timestamp:
    return pd.to_datetime(
        datetime.strptime(
            str(int(smile_trial['Overview']['date'])),
            '%Y%m%d%H%M%S',
        ).strftime('%Y-%m-%d')
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

def get_smile_meta(smile_data: list, **kwargs) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                'monkey': smile_trial['Overview']['subjectName'],
                'session date': get_trial_session_date(smile_trial),
                'trial datetime': get_trial_datetime(smile_trial),
                'trial name': smile_trial['Overview']['trialName'],
                'result': get_trial_result(smile_trial),
            } | kwargs
            for smile_trial in smile_data
        ],
        index=pd.Index(
            [get_trial_id(smile_trial) for smile_trial in smile_data],
            name='trial_id',
        )
    )