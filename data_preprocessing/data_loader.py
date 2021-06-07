from typing import Tuple
import numpy as np
import globals as g
from data_preprocessing.physionet_data_loader.load_physionet_dataset import get_data
from scipy.io import loadmat


class BCI2aIVDataLoader:
    """
    A BCI2aIVDataLoader object does the data loading and returns it as a python object.
    """

    def __init__(self, path: str):
        self.path = path

    def get_motor_imagery_data(self, subject_number: int, is_training: bool) -> Tuple[np.ndarray, np.ndarray]:
        """
        Loads one session of the data set 2a of the BCI Competition IV
        :param subject_number: must be in [1, 9]
        :param is_training: True for training data, False for evaluation data
        :return: session_signals, numpy ndarray of shape (num_valid_trial, 22, 750), motor imagery signal part
                 session_labels, numpy ndarray of shape (num_valid_trial)
        """
        if is_training:
            file_path = f"{self.path}A0{subject_number}T.mat"
        else:
            file_path = f"{self.path}A0{subject_number}E.mat"

        try:
            data = loadmat(file_path)['data']
        except FileNotFoundError:
            raise FileNotFoundError("Raw data not found. Modify '-p' input argument."
                                    "Run 'python main_preprocess_data.py -h for help'")

        session_labels = np.empty(g.num_trials)
        session_signals = np.empty((g.num_trials,
                                    g.num_channels_bci2aiv,
                                    g.num_samples_per_run))

        valid_trial_idx = 0
        for run_idx in range(0, data.size):
            run_data = [data[0, run_idx][0, 0]][0]  # some silly index magic
            run_signal = run_data[0]
            run_trials_start_idx = run_data[1]
            run_labels = run_data[2]
            run_artifacts = run_data[5]

            for trial_idx in range(0, run_trials_start_idx.size):
                num_artifacts_in_this_trial = run_artifacts[trial_idx]
                if num_artifacts_in_this_trial == 0:
                    trial_signals_start = run_trials_start_idx[trial_idx][0]
                    trial_signals_end = trial_signals_start + g.num_samples_per_run
                    trial_signals_all_channels = run_signal[trial_signals_start:trial_signals_end, 0:g.num_channels_bci2aiv]

                    session_signals[valid_trial_idx, :, :] = np.transpose(trial_signals_all_channels)
                    session_labels[valid_trial_idx] = int(run_labels[trial_idx])

                    valid_trial_idx += 1

        session_signals = session_signals[0:valid_trial_idx, :, :]
        session_labels = session_labels[0:valid_trial_idx]

        # Only use motor imagery part of signal
        session_signals = session_signals[:, :, g.motor_imagery_samples_start:g.motor_imagery_samples_stop]

        return session_signals, session_labels


class Bci2000DataLoader:
    """
    A Bci2000DataLoader object does the data loading and returns it as a python object.

    """

    def __init__(self, path: str, num_classes: int):
        file_path = f"{path}{num_classes}class.npz"
        try:
            npz_file = np.load(file_path)
        except FileNotFoundError:
            print(f'no file "{num_classes}class.npz found. Searching {path} for physionet bci2000 dataset"')
            try:
                X, y = get_data(path=path, n_classes=num_classes)
                np.savez(f"./data/{num_classes}class", X=X, y=y)
                file_path = "./data/"
            except FileNotFoundError:
                raise FileNotFoundError(f"physionet dataset not found. Point path to folder containing either "
                                        f"{num_classes}class.npz or original bci2000 dataset downloaded from "
                                        f"https://physionet.org/content/eegmmidb/1.0.0/")

            npz_file = np.load(file_path)
        self.num_classes = num_classes
        self.X, self.y = npz_file['X'], npz_file['y'].astype(dtype='uint8')

    def get_data_and_labels(self, subject_number: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get part bci2000 channel data and labels associated with a specific subject_number.

        Returns: X, y
            X: signals of shape [Num runs, num channels, num signals]
            y: labels of shape [Num runs]
        """
        subj_index = g.subject_nums_bci2000.index(subject_number)
        subj_trial_range = range(subj_index * g.num_trials_per_class_bci2000 * self.num_classes,
                                 (subj_index + 1) * g.num_trials_per_class_bci2000 * self.num_classes)
        X_subj = self.X[subj_trial_range, :, :]
        y_subj = self.y[subj_trial_range]

        return X_subj, y_subj
