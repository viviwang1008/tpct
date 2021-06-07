import argparse
import json
import subprocess

import globals as g
from data_preprocessing.data_loader import Bci2000DataLoader
from data_preprocessing.data_saver import BCI2000DataSaver
from data_preprocessing.data_preprocessor import DataPreprocessor


def write_used_input_arguments(args) -> None:
    """
    Write arguments from argparse to files and adds a git hash.
    """
    git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip().decode('ascii')
    with open("data/args_bci2000_preprocess.txt", 'w') as file:
        input_args_and_hash = vars(args)
        input_args_and_hash["git_hash"] = git_hash
        file.write(json.dumps(input_args_and_hash, indent=4))


def main():
    """
    Goes through all session data files, preprocesses them and stores image data for each trial
    """

    parser = argparse.ArgumentParser(description="Calculate preprocessed data from BCI 2000 dataset. "
                                                 "It can be downloaded from https://physionet.org/content/eegmmidb/1.0.0/")
    parser.add_argument("-p", "--path",
                        type=str,
                        help="Path to npz file of BCI 2000 dataset or unzipped edf data",
                        default="./data/")
    parser.add_argument("-f", "--frequency_band_set",
                        type=int,
                        help="which set of frequency bands is used",
                        choices=[0, 1, 2],
                        default=1)
    parser.add_argument("-w", "--num_windows",
                        type=int,
                        help="number of windows to split trial into. Default = 1",
                        default=1)
    parser.add_argument("-c", "--num_classes",
                        type=int,
                        help="number of classes used for dataset",
                        choices=[2, 3, 4],
                        default=4)
    parser.add_argument("-g", "--grid_points",
                        type=int,
                        help="number of grid points per dimension for output image. Default = 32",
                        default=32)
    parser.add_argument("-ps", "--do_use_pseudo_channels",
                        type=bool,
                        help="whether to use pseudo channels at corner of image with signal value 0. Default = True",
                        default=True)

    args = parser.parse_args()
    write_used_input_arguments(args)

    data_loader = Bci2000DataLoader(path=args.path,
                                    num_classes=args.num_classes)
    data_preprocessor = DataPreprocessor(dataset="BCI2000",
                                         num_windows=args.num_windows,
                                         num_grid_points=args.grid_points,
                                         frequency_band_set=args.frequency_band_set,
                                         pseudo_channels=args.do_use_pseudo_channels)
    data_saver = BCI2000DataSaver()

    print("Starting preprocessing")
    for subject_number in g.subject_nums_bci2000:
        print(f"Processing subject_number={subject_number}")
        X_subj, y_subj = data_loader.get_data_and_labels(subject_number=subject_number)
        preprocessed_X_subj = data_preprocessor.process_session_data(session_signals=X_subj)
        data_saver.store_data(preprocessed_X_subj, y_subj, subject_number=subject_number, num_classes=args.num_classes)


if __name__ == '__main__':
    main()
