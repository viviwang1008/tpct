from typing import Tuple
import numpy as np
import scipy.io as sio
import globals as g


class DataLoader:
    """
    A DataLoader object does the data loading and returns it as a python object.
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
            data = sio.loadmat(file_path)['data']
        except FileNotFoundError:
            raise FileNotFoundError("Raw data not found. Modify '-p' input argument."
                                    "Run 'python main_preprocess_data.py -h for help'")

        session_labels = np.empty(g.num_trials)
        session_signals = np.empty((g.num_trials,
                                    g.num_channels,
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
                    trial_signals_all_channels = run_signal[trial_signals_start:trial_signals_end, 0:g.num_channels]

                    session_signals[valid_trial_idx, :, :] = np.transpose(trial_signals_all_channels)
                    session_labels[valid_trial_idx] = int(run_labels[trial_idx])

                    valid_trial_idx += 1

        session_signals = session_signals[0:valid_trial_idx, :, :]
        session_labels = session_labels[0:valid_trial_idx]

        # Only use motor imagery part of signal
        session_signals = session_signals[:, :, g.motor_imagery_samples_start:g.motor_imagery_samples_stop]

        return session_signals, session_labels
