import json
from typing import List

import cv2
import matplotlib.pyplot as plt
import numpy as np


def create_bci2000_averaged_cv_comparison_plot(dirs: List[str], names: List[str], img_name: str) -> None:
    assert (len(dirs) == len(names))

    fig, ax = plt.subplots(nrows=1, ncols=2)
    plot_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive',
                   'tab:cyan']
    plot_line_styles = ['solid', 'dashed']
    plot_quantities = ["accuracy", "val_accuracy", "loss", "val_loss"]

    cross_validation_run_indices = range(1, 6)
    plot_epochs = [0]

    for dir_idx, dir in enumerate(dirs):
        for quantity_idx, quantity in enumerate(plot_quantities):
            # determine length of line to be plotted
            file_name = dir + '1/history.txt'
            data_dict = json.load(open(file_name))
            epochs = len(data_dict[quantity])
            data = np.zeros(epochs)
            plot_epochs.append(epochs)

            # average
            for cv_run in cross_validation_run_indices:
                file_name = dir + str(cv_run) + '/history.txt'
                data_dict = json.load(open(file_name))
                data += data_dict[quantity]
            data /= len(cross_validation_run_indices)

            # only add label on right plot
            if quantity_idx >= 2:
                if quantity_idx % 2:
                    label = f"{names[dir_idx]} val"
                else:
                    label = f"{names[dir_idx]} train"
                ax[quantity_idx // 2].plot(range(1, len(data)+1), data, color=plot_colors[dir_idx],
                                           linestyle=plot_line_styles[quantity_idx % 2],
                                           linewidth=1, label=label)
            else:
                ax[quantity_idx // 2].plot(range(1, len(data)+1), data, color=plot_colors[dir_idx],
                                           linestyle=plot_line_styles[quantity_idx % 2],
                                           linewidth=1)

    fig.set_size_inches(10, 4)
    ax[0].set_xlabel("epoch")
    ax[1].set_xlabel("epoch")
    ax[0].set_ylabel("accuracy")
    ax[1].set_ylabel("loss")
    ax[0].set_ylim([0, 1])
    ax[0].set_yticks(np.linspace(0, 1, 11))
    ax[0].set_xlim([0, np.max(plot_epochs)])
    ax[0].set_xticks(list(set(plot_epochs)))
    ax[1].set_xlim([0, np.max(plot_epochs)])
    ax[1].set_xticks(list(set(plot_epochs)))
    ax[0].grid(True)
    ax[1].grid(True)

    box = ax[0].get_position()
    ax[0].set_position([box.x0 + 0.2 * box.width, box.y0, box.width * 0.8, box.height])
    box = ax[1].get_position()
    ax[1].set_position([box.x0, box.y0, box.width * 0.8, box.height])

    ax[1].legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)

    plt.savefig(img_name)
    plt.close()
    crop_margins(img_name)


def create_bci2aiv_averaged_plot(dirs: List[str], names: List[str], img_name: str):
    assert (len(dirs) == len(names))

    fig, ax = plt.subplots(nrows=1, ncols=2)
    plot_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive',
                   'tab:cyan']
    plot_line_styles = ['solid', 'dashed']
    plot_quantities = ["accuracy", "val_accuracy", "loss", "val_loss"]
    plot_epochs = [0]

    subject_nums = range(1, 10)

    for dir_idx, dir in enumerate(dirs):
        for quantity_idx, quantity in enumerate(plot_quantities):
            # determine length of line to be plotted
            file_name = dir + '1/history_1.txt'
            data_dict = json.load(open(file_name))
            epochs = len(data_dict[quantity])
            data = np.zeros(epochs)
            plot_epochs.append(epochs)

            # average
            for subj_num in subject_nums:
                file_name = dir + str(subj_num) + f'/history_{subj_num}.txt'
                data_dict = json.load(open(file_name))
                data += data_dict[quantity]
            data /= len(subject_nums)

            # only add label on right plot
            if quantity_idx >= 2:
                if quantity_idx % 2:
                    label = f"{names[dir_idx]} val"
                else:
                    label = f"{names[dir_idx]} train"
                ax[quantity_idx // 2].plot(range(1, len(data)+1), data, color=plot_colors[dir_idx],
                                           linestyle=plot_line_styles[quantity_idx % 2],
                                           linewidth=1, label=label)
            else:
                ax[quantity_idx // 2].plot(range(1, len(data)+1), data, color=plot_colors[dir_idx],
                                           linestyle=plot_line_styles[quantity_idx % 2],
                                           linewidth=1)

    fig.set_size_inches(10, 4)
    ax[0].set_xlabel("epoch")
    ax[1].set_xlabel("epoch")
    ax[0].set_ylabel("accuracy")
    ax[1].set_ylabel("loss")
    ax[0].set_ylim([0, 1])
    ax[0].set_yticks(np.linspace(0, 1, 11))
    ax[0].set_xlim([0, np.max(plot_epochs)])
    ax[0].set_xticks(list(set(plot_epochs)))
    ax[1].set_xlim([0, np.max(plot_epochs)])
    ax[1].set_xticks(list(set(plot_epochs)))
    ax[0].grid(True)
    ax[1].grid(True)

    box = ax[0].get_position()
    ax[0].set_position([box.x0 + 0.2 * box.width, box.y0, box.width * 0.8, box.height])
    box = ax[1].get_position()
    ax[1].set_position([box.x0, box.y0, box.width * 0.8, box.height])

    ax[1].legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)

    plt.savefig(img_name)
    plt.close()
    crop_margins(img_name)


def crop_margins(img_name: str) -> None:
    img = cv2.imread(img_name)  # Read in the image and convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = 255 * (gray < 128).astype(np.uint8)  # To invert the text to white
    coords = cv2.findNonZero(gray)  # Find all non-zero points (text)
    x, y, w, h = cv2.boundingRect(coords)  # Find minimum spanning bounding box
    rect = img[y:y + h, x:x + w]  # Crop the image - note we do this on the original image
    cv2.imwrite(img_name, rect)  # Save the image


if __name__ == "__main__":
    base_dir = '../../logs/fit/bci2000_model_comparison/'
    sub_dirs = ['eegA32/', 'eegB32/', 'eegC32/', 'eegD32/']
    create_bci2000_averaged_cv_comparison_plot(dirs=[f"{base_dir}{sub_dir}" for sub_dir in sub_dirs],
                                               names=['eegA', 'eegB', 'eegC', 'eegD'],
                                               img_name="bci2000_model_32filter_stride_comparison.png")

    base_dir = '../../logs/fit/bci2000_model_comparison/'
    sub_dirs = ['eegA32S/', 'eegA_LSTM/', 'eegD32S/', 'eegD_LSTM/', ]
    create_bci2000_averaged_cv_comparison_plot(dirs=[f"{base_dir}{sub_dir}" for sub_dir in sub_dirs],
                                               names=['eegA', 'eegA_LSTM', 'eegD', 'eegD_LSTM'],
                                               img_name="bci2000_LSTM_comparison.png")

    base_dir = '../../logs/fit/bci2000_dropout_comparison/'
    sub_dirs = ['eegA0.1/', 'eegA0.5/', 'eegA0.9/']
    create_bci2000_averaged_cv_comparison_plot(dirs=[f"{base_dir}{sub_dir}" for sub_dir in sub_dirs],
                                               names=['dropout 0.1', 'dropout 0.5', 'dropout 0.9'],
                                               img_name="bci2000_dropout_comparison.png")

    dirs = ['../../logs/fit/bci2000_class_number_comparison/2class/',
            '../../logs/fit/bci2000_class_number_comparison/3class/',
            '../../logs/fit/bci2000_model_comparison/eegA32S/']
    create_bci2000_averaged_cv_comparison_plot(dirs=dirs,
                                               names=['2 class', '3 class', '4 class'],
                                               img_name="bci2000_class_number_comparison.png")

    dirs = ['../../logs/fit/bci2000_frequency_band_comparison/0band/',
            '../../logs/fit/bci2000_frequency_band_comparison/1band/',
            '../../logs/fit/bci2000_model_comparison/eegA32S/']
    create_bci2000_averaged_cv_comparison_plot(dirs=dirs,
                                               names=["8-13, 13-21, 21-30", "4-8, 8-13, 13-30", "0.5-4, 8-13, 13-30"],
                                               img_name="bci2000_frequency_band_comparison.png")

    dirs = ['../../logs/fit/bci2000_model_comparison/eegA32S/',
            '../../logs/fit/bci2000_image_size_comparison/64pixel/']
    create_bci2000_averaged_cv_comparison_plot(dirs=dirs,
                                               names=["32 pixel", "64 pixel"],
                                               img_name="bci2000_image_size_comparison.png")

    base_dir = '../../logs/fit/bci2aiv_model_comparison/'
    sub_dirs = ['eegA32/', 'eegB32/', 'eegC32/', 'eegD32/']
    create_bci2aiv_averaged_plot(dirs=[f"{base_dir}{sub_dir}" for sub_dir in sub_dirs],
                                 names=['eegA', 'eegB', 'eegC', 'eegD'],
                                 img_name="bci2aiv_model_32filter_stride_comparison.png")

    base_dir = '../../logs/fit/bci2aiv_model_comparison/'
    sub_dirs = ['eegA32S/', 'eegA_LSTM/', 'eegD32S/', 'eegD_LSTM/', ]
    create_bci2aiv_averaged_plot(dirs=[f"{base_dir}{sub_dir}" for sub_dir in sub_dirs],
                                 names=['eegA', 'eegA_LSTM', 'eegD', 'eegD_LSTM'],
                                 img_name="bci2aiv_LSTM_comparison.png")

    dirs = ['../../logs/fit/bci2000_model_comparison/eegA32S/',
            '../../logs/fit/bci2000_image_scrambled/']
    names = ['regular', 'shuffled']
    create_bci2000_averaged_cv_comparison_plot(dirs=dirs,
                                               names=names,
                                               img_name="bci2000_channel_location_comparison.png")