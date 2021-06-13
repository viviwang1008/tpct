import numpy as np
import json


def print_bci2000_model_comparison_data():
    base_dir = '../../logs/fit/bci2000_model_comparison/'
    sub_dirs = ['eegA', 'eegB', 'eegC', 'eegD']
    num_filters = ['16', '32']
    suffixes = ['S', '']
    subject_numbers = range(1, 6)
    epochs = 80

    print("BCI2000")
    for sub_dir in sub_dirs:
        for suffix in suffixes:
            for num_filter in num_filters:
                average_val_accuracy = np.zeros(epochs)
                for subject_number in subject_numbers:
                    filename = base_dir + sub_dir + num_filter + suffix + '/' + str(subject_number) + '/history.txt'
                    data_dict = json.load(open(filename))
                    val_acc = data_dict['val_accuracy']
                    average_val_accuracy += val_acc
                average_val_accuracy /= len(subject_numbers)
                average_val_accuracy_of_last_five_values = np.max(average_val_accuracy[-10:])
                print(f"{sub_dir + num_filter + suffix}: \t{average_val_accuracy_of_last_five_values:.3}")
    print("\n")


def print_bci2aiv_table_data():
    base_dir = '../../logs/fit/bci2aiv_model_comparison/'
    sub_dirs = ['eegA', 'eegB', 'eegC', 'eegD']
    num_filters = ['16', '32']
    suffixes = ['S', '']
    subject_numbers = range(1, 10)
    epochs = 80

    print("BCI2aiv")
    for sub_dir in sub_dirs:
        for suffix in suffixes:
            for num_filter in num_filters:
                average_val_accuracy = np.zeros(epochs)
                for subject_number in subject_numbers:
                    filename = base_dir + sub_dir + num_filter + suffix + '/' + str(
                        subject_number) + f'/history_{subject_number}.txt'
                    data_dict = json.load(open(filename))
                    val_acc = data_dict['val_accuracy']
                    average_val_accuracy += val_acc
                average_val_accuracy /= len(subject_numbers)
                average_val_accuracy_of_last_five_values = np.max(average_val_accuracy[-10:])
                print(f"{sub_dir + num_filter + suffix}: \t{average_val_accuracy_of_last_five_values:.3}")
    print("\n")


def print_model_summaries():
    base_dir = '../../logs/fit/bci2000_model_comparison/'
    sub_dirs = ['eegA', 'eegB', 'eegC', 'eegD']
    num_filters = ['16', '32']
    suffixes = ['S', '']

    print("NUMBER OF PARAMS")
    for sub_dir in sub_dirs:
        for suffix in suffixes:
            for num_filter in num_filters:
                filename = base_dir + sub_dir + num_filter + suffix + "/1" + f'/model_summary.txt'
                with open(filename) as file:
                    line = file.readlines()[-4]
                    num_params = line.split()[2]
                    print(f"{sub_dir + num_filter + suffix}: \t{num_params}")


if __name__ == "__main__":
    print_bci2000_model_comparison_data()
    print_bci2aiv_table_data()
    print_model_summaries()
