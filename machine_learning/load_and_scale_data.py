from typing import Tuple
import numpy as np
from sklearn.preprocessing import RobustScaler


def load_single_subject_data(subject_num: int, is_training: bool) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads preprocessed data of a given subject. Preprocessed data is created by running 'main_preprocess_data.py'.
    If a trial is split into multiple windows, the resulting images will be treated as independent data points.

    Args:
        subject_num: subject number that is used
        is_training: True for training data (from A0XT.mat), false for testing data (from A0XE.mat)

    Returns:
        (x, y): images and labels for subject
                x is of shape (Num_valid_trials * num_windows, 64, 64, 3)
                y is of shape (Num_valid_trials * num_windows,)

    Raises:
        FileNotFoundError: if the data can't be located in the "./data/" subdirectory. Run 'main_preprocess_data.py' to
        generate it.
    """
    try:
        if is_training:
            filename_suffix = f"A0{subject_num}T"
        else:
            filename_suffix = f"A0{subject_num}E"
        x = np.load(f"./data/preprocessed_{filename_suffix}.npy")
        y = np.load(f"./data/labels_{filename_suffix}.npy")

        num_trials = x.shape[0]
        num_windows = x.shape[1]
        x = np.reshape(x, (num_trials * num_windows, 64, 64, 3))
        y = np.repeat(y, num_windows)

        return x, y
    except FileNotFoundError:
        raise FileNotFoundError("Preprocessed data not found. Run main_preprocess_data.py and try again.")


def scale_data(x: np.ndarray, scaler=None) -> Tuple[np.ndarray, object]:
    """
    Scales data x using a robust scaler. If a fitted scaler is provided, it will be used. If none is provided, a new one
    will be fitted. In either case the scaler is returned with the data.

    Returns
    """
    original_shape = x.shape

    # reshape data to 2d array
    x = x.reshape((x.shape[0], -1))

    if scaler is None:
        scaler = RobustScaler().fit(x)

    x = scaler.transform(x)

    # reshape back
    x = x.reshape(original_shape)
    return x, scaler
