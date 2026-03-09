"""Tests for eye-tracking extraction from SMILE analog channels."""

import numpy as np
import pandas as pd
import pytest

from smile_extract.phasespace import (
    get_analog_channel_names,
    get_trial_eye_data,
    EYE_CHANNEL_SETS,
)


def _make_smile_trial(
    n_samples: int = 500,
    channel_names: list[str] | None = None,
    analog_data: np.ndarray | None = None,
    include_definitions: bool = True,
    include_analog: bool = True,
) -> dict:
    """Build a minimal mock SMILE trial dict for testing."""
    if channel_names is None:
        channel_names = [
            'Left Eye X', 'Left Eye Y', 'Right Eye X', 'Right Eye Y',
            'Left Pupil', 'Right Pupil', 'Sync_AI',
        ]

    if analog_data is None:
        rng = np.random.default_rng(42)
        analog_data = rng.standard_normal((n_samples, len(channel_names)))

    trial: dict = {
        'TrialData': {},
        'Definitions': {},
        'Overview': {'trialNumber': 'Trial1'},
        'Parameters': {'StateTable': []},
    }
    if include_analog:
        trial['TrialData']['analogData'] = analog_data
    if include_definitions:
        trial['Definitions']['analogChannelNames'] = np.array(channel_names)

    return trial


class TestGetAnalogChannelNames:
    def test_returns_list(self):
        trial = _make_smile_trial()
        names = get_analog_channel_names(trial)
        assert isinstance(names, list)
        assert 'Left Eye X' in names

    def test_missing_definitions(self):
        trial = _make_smile_trial(include_definitions=False)
        trial.pop('Definitions', None)
        names = get_analog_channel_names(trial)
        assert names == []

    def test_missing_analog_channel_names_key(self):
        trial = _make_smile_trial()
        del trial['Definitions']['analogChannelNames']
        names = get_analog_channel_names(trial)
        assert names == []


class TestGetTrialEyeData:
    def test_basic_extraction(self):
        trial = _make_smile_trial(n_samples=100)
        result = get_trial_eye_data(trial, final_sampling_rate=1000)
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ['eye_x', 'eye_y', 'pupil']
        assert result.index.name == 'time'
        assert result.columns.name == 'channel'
        assert len(result) == 100

    def test_resampling(self):
        trial = _make_smile_trial(n_samples=1000)
        result = get_trial_eye_data(trial, final_sampling_rate=100)
        # Resampled from 1kHz to 100Hz: ~100 samples
        assert len(result) == pytest.approx(100, abs=5)

    def test_y_axis_flip(self):
        """Left Eye Y should be negated."""
        n = 100
        channel_names = ['Left Eye X', 'Left Eye Y', 'Left Pupil']
        analog_data = np.ones((n, 3)) * 5.0
        trial = _make_smile_trial(
            n_samples=n, channel_names=channel_names, analog_data=analog_data,
        )
        result = get_trial_eye_data(trial, final_sampling_rate=1000)
        # eye_y should be -5.0 (flipped)
        assert result['eye_y'].iloc[50] == pytest.approx(-5.0)
        # eye_x and pupil should stay positive
        assert result['eye_x'].iloc[50] == pytest.approx(5.0)
        assert result['pupil'].iloc[50] == pytest.approx(5.0)

    def test_blink_detection(self):
        """Samples where all eye signals ≤ threshold should become NaN."""
        n = 200
        channel_names = ['Left Eye X', 'Left Eye Y', 'Left Pupil']
        analog_data = np.ones((n, 3)) * 2.0  # normal values
        # Insert blink at samples 50-60 (all channels below threshold)
        analog_data[50:60, :] = -10.0
        trial = _make_smile_trial(
            n_samples=n, channel_names=channel_names, analog_data=analog_data,
        )
        result = get_trial_eye_data(trial, final_sampling_rate=1000, blink_threshold=-9)
        # Blink samples should be NaN
        assert result.iloc[50:60].isna().all().all()
        # Non-blink samples should not be NaN
        assert not result.iloc[100:110].isna().any().any()

    def test_partial_blink_not_masked(self):
        """Only mask when ALL channels are below threshold."""
        n = 100
        channel_names = ['Left Eye X', 'Left Eye Y', 'Left Pupil']
        analog_data = np.ones((n, 3)) * 2.0
        # Only one channel below threshold
        analog_data[30:35, 0] = -10.0
        trial = _make_smile_trial(
            n_samples=n, channel_names=channel_names, analog_data=analog_data,
        )
        result = get_trial_eye_data(trial, final_sampling_rate=1000, blink_threshold=-9)
        # Should NOT be masked (not all channels below threshold)
        assert not result.iloc[30:35].isna().all().all()

    def test_no_eye_channels_returns_empty(self):
        """If no recognized eye channels, return empty DataFrame."""
        trial = _make_smile_trial(
            channel_names=['Sync_AI', 'Motor_Force', 'Push Button'],
            analog_data=np.ones((50, 3)),
        )
        result = get_trial_eye_data(trial)
        assert result.empty
        assert list(result.columns) == ['x', 'y', 'pupil']

    def test_no_analog_data_returns_empty(self):
        """If no analogData, return empty DataFrame."""
        trial = _make_smile_trial(include_analog=False)
        result = get_trial_eye_data(trial)
        assert result.empty

    def test_right_eye_fallback(self):
        """If only right eye channels exist, use those."""
        n = 50
        channel_names = ['Right Eye X', 'Right Eye Y', 'Right Pupil']
        analog_data = np.ones((n, 3)) * 3.0
        trial = _make_smile_trial(
            n_samples=n, channel_names=channel_names, analog_data=analog_data,
        )
        result = get_trial_eye_data(trial, final_sampling_rate=1000)
        assert len(result) == n
        assert list(result.columns) == ['eye_x', 'eye_y', 'pupil']

    def test_timedelta_index_starts_at_0(self):
        """Time index should start at 0 (matching pipeline convention)."""
        trial = _make_smile_trial(n_samples=50)
        result = get_trial_eye_data(trial, final_sampling_rate=1000)
        assert result.index[0] == pd.to_timedelta(0)
