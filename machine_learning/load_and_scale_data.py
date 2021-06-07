import json
from typing import Tuple
import numpy as np
import globals as g
from sklearn.preprocessing import StandardScaler


def load_preprocessed_bci2000_data(num_classes: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads all preprocessed data and labels
    Returns:
        X: preprocessed of all subject
        y: labels of all subject
    """
    try:
        img_size = json.load(open("./data/args_bci2000_preprocess.txt", 'r'))['grid_points']
        num_windows = json.load(open("./data/args_bci2000_preprocess.txt", 'r'))['num_windows']

        # if num_windows is 1, 'squeeze' will remove it as a dimension
        X = np.empty((0, num_windows, img_size, img_size, 3), dtype='float32').squeeze()
        y = np.empty((0,), dtype='uint8')
        for subj_num in g.subject_nums_bci2000:
            Xi = np.load(f"./data/bci2000_preprocessed_subj_{subj_num}_class_{num_classes}.npy").squeeze()
            yi = np.load(f"./data/bci2000_labels_subj_{subj_num}_class_{num_classes}.npy")

            X = np.concatenate((X, Xi))
            y = np.concatenate((y, yi))
        return X, y
    except FileNotFoundError:
        raise FileNotFoundError("Preprocessed data not found. Run main_preprocess_data_bci2000.py and try again.")


def load_single_subject_bci2aiv_data(subject_num: int, is_training: bool) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads preprocessed data of a given subject. Preprocessed data is created by running 'main_preprocess_data_bci2aiv.py'.

    Args:
        subject_num: subject number that is used
        is_training: True for training data (from A0XT.mat), false for testing data (from A0XE.mat)

    Returns:
        (x, y): images and labels for subject
                x is of shape (Num_valid_trials, num_windows, 64, 64, 3).squeeze
                y is of shape (Num_valid_trials)

    Raises:
        FileNotFoundError: if the data can't be located in the "./data/" subdirectory.
    """
    try:
        if is_training:
            filename_suffix = f"A0{subject_num}T"
        else:
            filename_suffix = f"A0{subject_num}E"

        # if num_windows is 1, 'squeeze' will remove it as a dimension
        X = np.load(f"./data/bci2aiv_preprocessed_{filename_suffix}.npy").squeeze()
        y = np.load(f"./data/bci2aiv_labels_{filename_suffix}.npy")

        return X, y
    except FileNotFoundError:
        raise FileNotFoundError("Preprocessed data not found. Run main_preprocess_data_bci2aiv.py and try again.")


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
        scaler = StandardScaler().fit(x)

    x = scaler.transform(x)

    # reshape back
    x = x.reshape(original_shape)
    return x, scaler
