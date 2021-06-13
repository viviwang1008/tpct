import argparse
import json

import globals as g
from data_preprocessing.data_loader import BCI2aIVDataLoader
from data_preprocessing.data_saver import BCI2aIVDataSaver
from data_preprocessing.data_preprocessor import DataPreprocessor


def write_used_input_arguments(args) -> None:
    """
    Write arguments from argparse to files
    """
    with open("data/args_bci2aiv_preprocess.txt", 'w') as file:
        input_args = vars(args)
        file.write(json.dumps(input_args, indent=4))


def main():
    """
    Goes through all session data files, preprocesses them and stores image data for each trial
    """

    parser = argparse.ArgumentParser(description="Calculate preprocessed data from data set 2a of BCI Competition IV. "
                                                 "It can be downloaded from http://bnci-horizon-2020.eu/database/data-sets")
    parser.add_argument("-p", "--path",
                        type=str,
                        help="Path to location where training data such as 'A01T.mat': 'path/to/data/'",
                        default="/usr/scratch/sassauna1/sem21f18/tpct/")
    parser.add_argument("-f", "--frequency_band_set",
                        type=int,
                        help="which set of frequency bands is used",
                        choices=[0, 1, 2],
                        default=1)
    parser.add_argument("-w", "--num_windows",
                        type=int,
                        help="number of windows to split trial into. Default = 1",
                        default=1)
    parser.add_argument("-g", "--grid_points",
                        type=int,
                        help="number of grid points per dimension for output image. Default = 64",
                        default=64)

    args = parser.parse_args()
    write_used_input_arguments(args)

    data_loader = BCI2aIVDataLoader(path=args.path)
    data_preprocessor = DataPreprocessor(dataset="BCI2aIV",
                                         num_windows=args.num_windows,
                                         num_grid_points=args.grid_points,
                                         frequency_band_set=args.frequency_band_set,
                                         pseudo_channels=True)
    data_saver = BCI2aIVDataSaver()

    print("Starting preprocessing")
    for is_training in [True, False]:
        for subject_number in g.subject_num_range_bci2aiv:
            print(f"Processing subject_number={subject_number}, is_training={is_training}")

            session_signals, session_labels = data_loader.get_motor_imagery_data(subject_number, is_training)

            preprocessed_session_data = data_preprocessor.process_session_data(session_signals)
            session_labels = (session_labels - 1).astype(dtype='uint8')

            data_saver.store_data(preprocessed_session_data, session_labels, subject_number, is_training)


if __name__ == '__main__':
    main()
