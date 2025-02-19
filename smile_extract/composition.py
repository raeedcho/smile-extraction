import pandas as pd
from . import trial_info, targets, states, phasespace, neural

def compose_session_frame(
        smile_data,
        bin_size: str='10ms',
        min_firing_rate: float=0.1,
        max_spike_coincidence: float=0.2,
        rate_artifact_threshold: float=350,
        **kwargs,
) -> pd.DataFrame:
    # meta
    meta = trial_info.get_smile_meta(smile_data, **kwargs)
    state_list = concat_trial_func_results(states.get_trial_states, smile_data, bin_size=bin_size)
    hand_pos = concat_trial_func_results(
        phasespace.get_trial_hand_data,
        smile_data,
        final_sampling_rate=1/pd.to_timedelta(bin_size).total_seconds(),
        **kwargs,
    )
    binned_spikes = (
        get_smile_spike_times(smile_data)
        .pipe(neural.remove_abnormal_firing_units, min_firing_rate=min_firing_rate, rate_artifact_threshold=rate_artifact_threshold)
        .pipe(neural.remove_artifact_trials, rate_artifact_threshold=rate_artifact_threshold)
        .pipe(neural.remove_correlated_units, max_spike_coincidence=max_spike_coincidence)
        .pipe(neural.bin_spikes, bin_size=bin_size)
        .pipe(neural.collapse_channel_unit_index)
    )
    
    # targets
    return (
        pd.concat(
            [state_list,hand_pos,binned_spikes],
            axis=1,
            join='inner',
            keys=['state','hand position','motor cortex'],
            names=['channel','signal'],
        )
        .reset_index(level='time')
        .assign(**meta)
        .set_index('time',append=True)
        [['monkey','session date','block','trial datetime','task','result','state','hand position','motor cortex']]
    )

def concat_trial_func_results(trial_func, smile_data: list, **func_kwargs) -> pd.DataFrame:
    return pd.concat(
        [trial_func(trial,**func_kwargs) for trial in smile_data],
        axis=0,
        keys=[trial_info.get_trial_id(trial) for trial in smile_data],
        names=['trial_id'],
    )

def get_smile_spike_times(smile_data: list, keep_sorted_only=True) -> pd.DataFrame:
    return concat_trial_func_results(neural.get_trial_spike_times, smile_data, keep_sorted_only=keep_sorted_only)

def get_spike_waveforms(smile_data: list) -> pd.DataFrame:
    return concat_trial_func_results(neural.get_trial_waveforms, smile_data)
