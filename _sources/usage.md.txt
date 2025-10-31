# Usage

Below are some common workflows. Replace paths and parameters as appropriate.

## Loading data

```python
from smile_extract import io

dataset = io.load_dataset("/path/to/data")
print(dataset)
```

## Working with neural signals

```python
from smile_extract import neural

spikes = neural.compute_spike_counts(dataset, bin_ms=5)
```

## Extracting targets and trial info

```python
from smile_extract import targets, trial_info

T = targets.extract_targets(dataset)
trials = trial_info.extract_trials(dataset)
```

See the API reference for full module documentation.
