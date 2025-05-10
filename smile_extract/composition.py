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
    
    return compose_from_frames(
        meta,
        {
            'state': state_list,
            'hand position': hand_pos,
            'motor cortex': binned_spikes,
        },
    )

def compose_from_frames(meta: pd.DataFrame, trialframe_dict: dict[str,pd.DataFrame]) -> pd.DataFrame:
    trialframe = (
        pd.concat(
            {
                key: frame
                for key, frame in trialframe_dict.items()
            },
            axis=1,
            join='inner',
            names=['channel','signal'],
        )
        .reset_index(level='time')
        .assign(**meta)
        .set_index('time',append=True)
        [['monkey','session date','trial datetime','task','result','state','hand position','motor cortex']]
    )
    return trialframe

def concat_trial_func_results(trial_func, smile_data: list, **func_kwargs) -> pd.DataFrame:
    return pd.concat(
        {trial_info.get_trial_id(trial): trial_func(trial,**func_kwargs) for trial in smile_data},
        axis=0,
        names=['trial_id'],
    )

def concat_block_func_results(block_func, smile_data_blocks: dict[str,list], **func_kwargs) -> pd.DataFrame:
    return pd.concat(
        {block: block_func(block_data,**func_kwargs) for block,block_data in smile_data_blocks.items()},
        axis=0,
        names=['block'],
    )

def concat_block_trial_func_results(trial_func, smile_data_blocks: dict[str,list], **func_kwargs) -> pd.DataFrame:
    return pd.concat(
        {
            block: concat_trial_func_results(trial_func, block_data, **func_kwargs)
            for block,block_data in smile_data_blocks.items()
        },
        axis=0,
        names=['block'],
    )

def get_smile_spike_times(smile_data: list, keep_sorted_only=True) -> pd.DataFrame:
    return concat_trial_func_results(neural.get_trial_spike_times, smile_data, keep_sorted_only=keep_sorted_only)

def get_spike_waveforms(smile_data: list) -> pd.DataFrame:
    return concat_trial_func_results(neural.get_trial_waveforms, smile_data)
