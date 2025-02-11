import scipy.io
import pandas as pd

path = '/Users/raeed/Library/CloudStorage/OneDrive-UniversityofPittsburgh/0-projects/cst-rtt/cst-data/preTD/Dwight/2025-01-07/Dwight_2025-01-07_reward-type_medium_cst-rtt-dco_sorted.mat'

def direct_load_smile_data(filename: str) -> list:
    try:
        mat = scipy.io.loadmat(filename, simplify_cells=True)
    except NotImplementedError:
        try:
            import mat73
        except ImportError:
            raise ImportError("Must have mat73 installed to load mat73 files.")
        else:
            mat = mat73.loadmat(filename)

    real_keys = [k for k in mat.keys() if not (k.startswith("__") and k.endswith("__"))]
    assert len(real_keys) == 1, "Multiple top-level keys found in mat file."
    data_name = real_keys[0]
    smile_data = mat[data_name]

    # convert dict of list to list of dict (in case of mat73 file load)
    if type(smile_data) is dict:
        smile_data = [dict(zip(smile_data.keys(), values)) for values in zip(*smile_data.values())]

    return smile_data