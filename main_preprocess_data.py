import argparse
import json
import subprocess

import globals as g
from data_preprocessing.data_loader import DataLoader
from data_preprocessing.data_saver import DataSaver
from data_preprocessing.data_preprocessor import DataPreprocessor


def write_used_input_arguments(args) -> None:
    """
    Write arguments from argparse to files and adds a git hash.
    """
    git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip().decode('ascii')
    with open("./data/args.txt", 'w') as file:
        input_args_and_hash = vars(args)
        input_args_and_hash["git_hash"] = git_hash
        file.write(json.dumps(input_args_and_hash, indent=4))


def main():
    """
    Goes through all session data files, preprocesses them and stores image data for each trial
    """

    parser = argparse.ArgumentParser(description="Calculate preprocessed data from data set 2a of BCI Competition IV. "
                                                 "It can be downloaded from http://bnci-horizon-2020.eu/database/data-sets")
    parser.add_argument("-p", "--path",
                        type=str,
                        help="Path to location where training data is stored, e.g. 'A01T.mat'",
                        default="/usr/scratch/sassauna1/sem21f18/tpct/")
    parser.add_argument("-w", "--num_windows",
                        type=int,
                        help="number of windows to split trial into. Default = 6",
                        default=6)
    parser.add_argument("-g", "--grid_points",
                        type=int,
                        help="number of grid points per dimension for output image. Default = 64",
                        default=64)
    parser.add_argument("-ps", "--do_use_pseudo_channels",
                        type=bool,
                        help="whether to use pseudo channels at corner of image with signal value 0. Default = True",
                        default=True)

    args = parser.parse_args()
    write_used_input_arguments(args)

    data_loader = DataLoader(path=args.path)
    data_preprocessor = DataPreprocessor(num_windows=args.num_windows,
                                         num_grid_points=args.grid_points,
                                         pseudo_channels=args.do_use_pseudo_channels)
    data_saver = DataSaver()

    print("Starting preprocessing")
    for is_training in [True, False]:
        for subject_number in g.subject_num_range:
            print(f"Processing subject_number={subject_number}, is_training={is_training}")

            session_signals, session_labels = data_loader.get_motor_imagery_data(subject_number, is_training)

            preprocessed_session_data = data_preprocessor.process_session_data(session_signals)
            session_labels = (session_labels - 1).astype(dtype='uint8')

            data_saver.store_data(preprocessed_session_data, session_labels, subject_number, is_training)


if __name__ == '__main__':
    main()
