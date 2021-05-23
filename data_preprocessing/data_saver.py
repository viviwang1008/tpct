import numpy as np


class DataSaver:
    """
    Stores data and labels as .npy files
    """
    @staticmethod
    def store_data(preprocessed_session_data: np.ndarray,
                   session_labels: np.ndarray,
                   subject_number: int,
                   is_training: bool) -> None:
        """
        Stores preprocessed data and labels for one subject's session as a .npy file
        :param preprocessed_session_data: numpy ndarray of shape (num_valid_trials, 64, 64)
        :param session_labels: numpy ndarray of shape (num_valid_trials,)
        :param subject_number: must be in [1, 9]
        :param is_training: True for training data, False for evaluation data
        """
        if is_training:
            filename_suffix = f"A0{subject_number}T"
        else:
            filename_suffix = f"A0{subject_number}E"

        np.save(f"./data/labels_{filename_suffix}", session_labels.astype('uint8'))
        np.save(f"./data/preprocessed_{filename_suffix}", preprocessed_session_data.astype('float32'))
