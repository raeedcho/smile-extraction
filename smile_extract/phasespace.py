import pandas as pd
import numpy as np
from scipy.signal import resample_poly
import fractions
from typing import Optional
from .states import get_trial_state_transition_table, get_trial_events, get_state_target_table
from .targets import get_trial_targets
import logging
logger = logging.getLogger(__name__)

# Phasespace data
def get_trial_hand_data(
        smile_trial: dict,
        resample_window: tuple=('kaiser',20.0),
        final_sampling_rate: float=1000,
        reference_target: Optional[str] = None,
        **kwargs,
    ) -> pd.DataFrame:

    if reference_target is not None:
        target = (
            get_trial_targets(smile_trial)
            .loc[reference_target]
        )
        reference_loc = np.array([target['x'], target['y'], 0])
    else:
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
    sync_params = smile_trial['TrialData']['Marker']['SyncParameters']
    if np.all(sync_params['phasespaceSyncData']):
        phasespace_sync_frame, phasespace_sync_time = infer_phasespace_sync_from_data(smile_trial)
    else:
        phasespace_sync_frame = sync_params['phasespaceFrame']
        phasespace_sync_time = pd.to_timedelta(float(sync_params['startTime']),unit='ms')

    phasespace_freq = smile_trial['TrialData']['Marker']['frequency']
    return pd.to_timedelta((framevec-phasespace_sync_frame)/phasespace_freq,unit='s') + phasespace_sync_time

def infer_phasespace_sync_from_data(smile_trial) -> tuple[int, pd.Timedelta]:
    '''
    In the case where phasespace sync data is not available (e.g. rig 2 for Sulley),
    we need to infer a synchronization point on the phasespace side from the marker
    data and target location data, and the state transition times on the trial control side.
    '''
    state_table = get_trial_state_transition_table(smile_trial)
    state_target_table = get_state_target_table(smile_trial)

    assert state_table['pass condition'].values[0] == 'enter', "First state must be an 'enter' state."
    ref_state = state_table.index[0]
    reference_target = state_target_table.loc[ref_state]

    # get frame where hand enters reference target
    phasespace_data = smile_trial['TrialData']['Marker']['rawPositions']
    marker_position = phasespace_data[:,1:4]
    framevec = phasespace_data[:,4].astype(int)

    relative_marker_pos = marker_position[:,:2].astype(np.float32) - reference_target[['x','y']].values.astype(np.float32)
    marker_dist = np.linalg.norm(relative_marker_pos,axis=1)
    try:
        enter_idx = np.nonzero(marker_dist < reference_target['radius'].values[0])[0][0]
    except IndexError:
        logger.warning(f"Trial {smile_trial['Overview']['trialNumber']} Could not find hand entering reference target; using first frame as sync point.")
        return framevec[0], pd.to_timedelta(0,unit='ms')
    enter_frame = framevec[enter_idx]

    trial_events = get_trial_events(smile_trial)
    pass_state = state_table.loc[ref_state,'pass state']
    try:
        enter_time = trial_events.loc[pass_state,'time']
    except KeyError:
        logger.warning(f"Trial {smile_trial['Overview']['trialNumber']} Could not find pass event for reference target; using time of first frame as sync point.")
        return framevec[0], pd.to_timedelta(0,unit='ms')
    
    if isinstance(enter_time, pd.Series):
        enter_time = enter_time.iloc[0]

    return enter_frame, enter_time

def get_analog_channel_names(smile_trial: dict) -> list[str]:
    """Return the list of analog channel names from a SMILE trial.

    Parameters:
        smile_trial: Single SMILE trial dictionary.

    Returns:
        List of channel name strings from Definitions.analogChannelNames.
    """
    names = smile_trial.get('Definitions', {}).get('analogChannelNames', [])
    if hasattr(names, 'tolist'):
        return names.tolist()
    return list(names)


# Default mapping from SMILE analog channel names to output column names.
# The first available eye (left, then right) will be used.
EYE_CHANNEL_SETS = [
    {
        'eye_x': 'Left Eye X',
        'eye_y': 'Left Eye Y',
        'pupil': 'Left Pupil',
    },
    {
        'eye_x': 'Right Eye X',
        'eye_y': 'Right Eye Y',
        'pupil': 'Right Pupil',
    },
]


def get_trial_eye_data(
        smile_trial: dict,
        final_sampling_rate: float = 1000,
        blink_threshold: float = -9,
        resample_window: tuple = ('kaiser', 20.0),
        **kwargs,
) -> pd.DataFrame:
    """Extract eye position and pupil diameter from SMILE analog channels.

    Reads analog channel data at its native 1 kHz rate, detects blinks,
    flips the Y axis to match the phasespace coordinate convention,
    and resamples to ``final_sampling_rate``.

    Parameters:
        smile_trial: Single SMILE trial dictionary.
        final_sampling_rate: Target sampling rate in Hz (default 1000).
        blink_threshold: Voltage threshold for blink detection. Samples
            where **all** eye signals are ≤ this value are set to NaN.
        resample_window: Window specification passed to ``sig_resample``.

    Returns:
        DataFrame with ``TimedeltaIndex(name='time')`` and columns
        ``['eye_x', 'eye_y', 'pupil']`` (column index named ``'channel'``).
        Returns an empty DataFrame if no eye channels are found.
    """
    empty = pd.DataFrame(
        index=pd.TimedeltaIndex([], name='time'),
        columns=pd.Index(['x', 'y', 'pupil'], name='channel'),
        dtype=float,
    )

    # --- locate analog data ---
    try:
        analog_data = np.asarray(smile_trial['TrialData']['analogData'], dtype=float)
    except (KeyError, TypeError):
        logger.debug('No analogData found in trial; skipping eye extraction.')
        return empty

    if analog_data.ndim != 2 or analog_data.shape[0] == 0:
        logger.debug('analogData is empty or malformed; skipping eye extraction.')
        return empty

    channel_names = get_analog_channel_names(smile_trial)
    if not channel_names:
        logger.debug('No analogChannelNames found; skipping eye extraction.')
        return empty

    # --- resolve which eye channels are present ---
    channel_map: dict[str, str] | None = None
    for candidate in EYE_CHANNEL_SETS:
        if all(name in channel_names for name in candidate.values()):
            channel_map = candidate
            break

    if channel_map is None:
        logger.debug(
            'No complete set of eye channels found in analogChannelNames: %s',
            channel_names,
        )
        return empty

    # --- extract columns ---
    native_rate = 1000  # analog channels are always 1 kHz
    n_samples = analog_data.shape[0]

    col_indices = {
        out_name: channel_names.index(smile_name)
        for out_name, smile_name in channel_map.items()
    }

    eye_df = pd.DataFrame(
        {out_name: analog_data[:, idx] for out_name, idx in col_indices.items()},
        index=pd.timedelta_range(
            start=pd.to_timedelta(0, unit='ms'),
            periods=n_samples,
            freq=pd.to_timedelta(1 / native_rate, unit='s'),
            name='time',
        ),
    )
    eye_df.columns.name = 'channel'

    # --- blink detection: mask where ALL eye signals ≤ threshold ---
    blink_mask = (eye_df <= blink_threshold).all(axis=1)
    eye_df.loc[blink_mask] = np.nan

    # --- flip Y axis (phasespace convention) ---
    eye_df['eye_y'] = -eye_df['eye_y']

    # --- resample if needed ---
    if final_sampling_rate != native_rate:
        # Drop NaN rows before resampling to avoid spreading NaN, then reinsert
        eye_df = (
            eye_df
            .pipe(sig_resample, final_sampling_rate, old_sampling_rate=native_rate, window=resample_window)
        )

    # --- ensure evenly-spaced index ---
    final_timevec = pd.timedelta_range(
        start=pd.to_timedelta(0, unit='ms'),
        end=eye_df.index[-1],
        freq=pd.to_timedelta(1 / final_sampling_rate, unit='s'),
        name='time',
    )
    eye_df = interpolating_reindex(eye_df, final_timevec)

    return eye_df

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