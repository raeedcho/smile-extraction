import pandas as pd
import numpy as np
from scipy.signal import resample_poly
import fractions
from typing import Optional

# Phasespace data
def get_trial_hand_data(
        smile_trial: dict,
        resample_window: tuple=('kaiser',20.0),
        final_sampling_rate: float=1000,
        reference_loc: Optional[np.ndarray] = None,
        **kwargs,
    ) -> pd.DataFrame:

    if reference_loc is None:
        reference_loc = np.array([0,0,0])

    phasespace_data = smile_trial['TrialData']['Marker']['rawPositions']
    if phasespace_data.shape[0] == 0 or phasespace_data.ndim != 2:
        return pd.DataFrame(index=pd.Index([],name='time'),columns=pd.Index(['x','y','z'],name='signal'))

    phasespace_freq = smile_trial['TrialData']['Marker']['frequency']
    marker_position = (phasespace_data[:,1:4] - reference_loc)# * [1,-1,1] # flip y-axis for data collected in rig 1 before 2023-10-01
    framevec = (phasespace_data[:,4]).astype(int)
    full_framevec = np.arange(framevec[0], framevec[-1]+1)
    final_timevec = pd.timedelta_range(
        start=0,
        end=convert_phasespace_frame_to_time(full_framevec[-1], smile_trial),
        freq=pd.to_timedelta(1/final_sampling_rate, unit='s'),
        name='time',
    )

    marker_pos_interp = (
        pd.DataFrame(
            marker_position,
            columns=pd.Index(['x','y','z'],name='channel'),
            index=pd.Index(framevec,name='phasespace_frame')
        )
        .reindex(full_framevec)
        .interpolate(method='linear')
        .reset_index()
        .assign(
            time=lambda x: convert_phasespace_frame_to_time(x['phasespace_frame'], smile_trial),
        )
        .set_index('time')
        .drop(columns='phasespace_frame')
        .pipe(sig_resample, final_sampling_rate, old_sampling_rate=phasespace_freq, window=resample_window)
        .pipe(interpolating_reindex, final_timevec)
    )

    return marker_pos_interp

def convert_phasespace_frame_to_time(framevec, smile_trial):
    phasespace_sync_frame = smile_trial['TrialData']['Marker']['SyncParameters']['phasespaceFrame']
    phasespace_sync_time = pd.to_timedelta(float(smile_trial['TrialData']['Marker']['SyncParameters']['startTime']),unit='ms')
    phasespace_freq = smile_trial['TrialData']['Marker']['frequency']
    return pd.to_timedelta((framevec-phasespace_sync_frame)/phasespace_freq,unit='s') + phasespace_sync_time

def get_trial_eye_data(smile_trial: dict) -> pd.DataFrame:
    pass

def sig_resample(df: pd.DataFrame, new_sampling_rate: float, old_sampling_rate: Optional[float]=None, **kwargs)->pd.DataFrame:
    assert type(df.index) is pd.TimedeltaIndex, "Index must be a TimedeltaIndex."

    if old_sampling_rate is None:
        old_timevec_period = (
            df.index
            .diff()
            .value_counts()
            .idxmax()
            .total_seconds()
        )
        old_sampling_rate = 1/old_timevec_period

    resample_factor = fractions.Fraction.from_float(new_sampling_rate / old_sampling_rate).limit_denominator()
    new_signal = resample_poly(
        df.values,
        resample_factor.numerator,
        resample_factor.denominator,
        axis=0,
        padtype='line',
        **kwargs,
    )
    new_timevec = pd.timedelta_range(
        start=df.index[0],
        periods=new_signal.shape[0],
        freq=pd.to_timedelta(1/new_sampling_rate, unit='s'),
        name='time',
    )

    return pd.DataFrame(
        index=new_timevec,
        data=new_signal,
        columns=df.columns,
    )

def multicol_interp(x, xp, fp, **kwargs):
    assert xp.shape[0] == fp.shape[0], "xp and fp must have the same number of rows."
    assert x.ndim == 1, "x must be 1D."
    assert xp.ndim == 1, "xp must be 1D."
    assert fp.ndim == 2, "fp must be 2D."

    return np.column_stack([np.interp(x, xp, fp[:,i], **kwargs) for i in range(fp.shape[1])])

def interpolating_reindex(df, new_index):
    assert type(new_index) is pd.TimedeltaIndex, "new_index must be a pandas Index."

    return pd.DataFrame(
        index=new_index,
        data=multicol_interp(new_index, df.index, df.values),
        columns=df.columns,
    )

# CST error cursor
def get_trial_cst_cursor(
        smile_trial: dict,
        resample_window: tuple=('kaiser',20.0),
        final_sampling_rate: float=1000,
        reference_loc: Optional[np.ndarray] = None,
        **kwargs,
    ) -> pd.DataFrame:

    if reference_loc is None:
        reference_loc = np.array([0,0,0])

    cursor_data = smile_trial['TrialData']['Marker']['errorCursor']
    if cursor_data is None or cursor_data.shape[0] == 0 or cursor_data.ndim != 2:
        return pd.DataFrame(
            index=pd.Index([], name='time'),
            columns=pd.Index(['x', 'y', 'z'], name='signal')
        )

    cursor_timevec = pd.to_timedelta(
        cursor_data[:,3],
        unit='ms',
    )
    sample_spacing = (
        cursor_timevec
        .diff()
        .value_counts()
        .idxmax()
        .total_seconds()
    )
    final_timevec = pd.timedelta_range(
        start=0,
        end=cursor_timevec[-1],
        freq=pd.to_timedelta(1/final_sampling_rate, unit='s'),
        name='time',
    )

    cursor_position = (
        pd.DataFrame(
            cursor_data[:,:3]-reference_loc,
            columns=pd.Index(['x','y','z'],name='channel'),
            index=pd.Index(cursor_timevec,name='time'),
        )
        .pipe(sig_resample, final_sampling_rate, old_sampling_rate=1/sample_spacing, window=resample_window)
        .pipe(interpolating_reindex, final_timevec)
        .loc[slice(cursor_timevec[0],None),:]
    )

    return cursor_position