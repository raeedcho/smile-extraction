import pandas as pd
import numpy as np
import logging
logger = logging.getLogger(__name__)

def get_trial_spike_times(trial: dict, keep_sorted_only=True) -> pd.DataFrame:
    trial_spikes = (
        pd.DataFrame(
            trial['TrialData']['TDT']['snippetInfo'].T,
            columns=['channel', 'unit', 'frame'],
        )
        .assign(**{
            'timestamp': lambda x: pd.to_timedelta(x['frame']-1, unit='ms'),
        })
        .drop(columns='frame')
        .rename_axis('snippet_id', axis=0)
    )

    if keep_sorted_only:
        trial_spikes = trial_spikes.loc[lambda x: (x['unit']>0) & (x['unit']<31)]
    
    return trial_spikes

def get_trial_waveforms(trial: dict) -> pd.DataFrame:
    waveforms = trial['TrialData']['TDT']['snippetWaveforms'].T
    return (
        pd.DataFrame(
            waveforms,
            columns=pd.RangeIndex(start=0, stop=waveforms.shape[1], name='snippet frame'),
        )
        .rename_axis('snippet_id', axis=0)
        .assign(**get_trial_spike_times(trial,keep_sorted_only=False)[['channel','unit']])
        .set_index(['channel','unit'],append=True)
    )

def bin_spikes(spike_times: pd.DataFrame, bin_size: str='10ms') -> pd.DataFrame:
    # may need to adjust this to handle cases where there are no spikes at time 0
    # possibly using .asfreq() before slicing to ensure that all timepoints are present
    spike_counts = (
        spike_times
        .assign(spikes=1)
        .set_index(['channel','unit','timestamp'],append=True)
        .reset_index(level='snippet_id',drop=True)
        .squeeze()
        .unstack(level=['channel','unit'],fill_value=0)
        .sort_index(level=['trial_id','timestamp'],axis=0)
        .sort_index(level=['channel','unit'],axis=1)
        .loc[(slice(None),slice('0s',None)),:]
        .groupby('trial_id')
        .resample(bin_size,level='timestamp')
        .sum()
        .rename_axis(index={'timestamp':'time'})
    )

    return spike_counts

def remove_low_firing_units(spike_times: pd.DataFrame, min_firing_rate: float=0.1) -> pd.DataFrame:
    recording_duration = (
        spike_times
        .groupby('trial_id')
        ['timestamp']
        .agg(np.ptp)
        .sum()
    )
    average_firing_rate = (
        spike_times
        .groupby(['channel','unit'])
        ['timestamp']
        .agg('count')
        .apply(lambda x: x / recording_duration.total_seconds())
        .rename('average firing rate')
    )

    num_units = len(average_firing_rate)
    num_units_removed = (average_firing_rate <= min_firing_rate).sum()
    logger.info(f"Removing {num_units_removed} of {num_units} units with average firing rate less than {min_firing_rate} Hz.")
    logger.debug(f"Units removed:\n {average_firing_rate[average_firing_rate <= min_firing_rate]}")

    return (
        spike_times
        .reset_index()
        .set_index(['channel','unit'])
        .loc[average_firing_rate > min_firing_rate]
        .reset_index()
        .set_index(['trial_id','snippet_id'])
    )

def remove_correlated_units(spike_times: pd.DataFrame, max_spike_coincidence: float=0.2) -> pd.DataFrame:
    binned_spikes = bin_spikes(spike_times, bin_size='1ms')

    corr_mat = np.corrcoef(binned_spikes.T)
    coincident_unit_idx = np.any(np.tril(corr_mat,k=-1) > max_spike_coincidence, axis=1)
    coincident_units = binned_spikes.columns[coincident_unit_idx]

    num_total_units = binned_spikes.shape[1]
    num_units_removed = len(coincident_units)
    logger.info(f"Removing {num_units_removed} of {num_total_units} units with spike coincidence greater than {max_spike_coincidence}.")
    logger.debug(f"Units removed:\n {coincident_units}")

    return (
        spike_times
        .reset_index()
        .set_index(['channel','unit'])
        .drop(index=coincident_units)
        .reset_index()
        .set_index(['trial_id','snippet_id'])
    )

def collapse_channel_unit_index(binned_spikes: pd.DataFrame) -> pd.DataFrame:
    assert binned_spikes.columns.names == ['channel','unit'], "Columns must be MultiIndex with channel and unit levels."

    signal_ids = binned_spikes.columns.to_series().apply(lambda x: f'ch{x[0]}u{x[1]}')
    return pd.DataFrame(
        binned_spikes.values,
        index=binned_spikes.index,
        columns=pd.Index(signal_ids,name='signal'),
    )

def get_array_channels(monkey_name: str) -> dict:
    if monkey_name == 'Prez':
        array_channels = {
            'M1': np.arange(33, 96),
            'PMd': np.concatenate([np.arange(1,33), np.arange(96, 129)]),
        }
    elif monkey_name == 'Dwight':
        array_channels = {
            'M1': np.concatenate([np.arange(1,33), np.arange(96, 129)]),
            'PMd': np.arange(33, 96),
        }

    return array_channels