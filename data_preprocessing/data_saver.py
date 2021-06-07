import numpy as np


class BCI2aIVDataSaver:
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
        :param preprocessed_session_data: numpy ndarray of shape (num_valid_trials, grid_points, grid_points)
        :param session_labels: numpy ndarray of shape (num_valid_trials,)
        :param subject_number: must be in [1, 9]
        :param is_training: True for training data, False for evaluation data
        """
        if is_training:
            filename_suffix = f"A0{subject_number}T"
        else:
            filename_suffix = f"A0{subject_number}E"

        np.save(f"./data/bci2aiv_labels_{filename_suffix}", session_labels.astype('uint8'))
        np.save(f"./data/bci2aiv_preprocessed_{filename_suffix}", preprocessed_session_data.astype('float32'))


class BCI2000DataSaver:
    """
    Stores BCI 2000 data and labels as .npy files
    """
    @staticmethod
    def store_data(preprocessed_session_data: np.ndarray,
                   session_labels: np.ndarray,
                   subject_number: int,
                   num_classes: int) -> None:
        """
        Store BCI2000 preprocessed data and labels
        """
        filename_suffix = f"subj_{subject_number}"

        np.save(f"./data/bci2000_labels_{filename_suffix}_class_{num_classes}", session_labels.astype('uint8'))
        np.save(f"./data/bci2000_preprocessed_{filename_suffix}_class_{num_classes}", preprocessed_session_data.astype('float32'))
