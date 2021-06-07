import json

import cv2
import matplotlib.pyplot as plt
import numpy as np


def create_bci2000_model_comparison_plot():
    fig, ax = plt.subplots(nrows=1, ncols=2)

    base_dir = '../../logs/fit/bci2000_model_comparison/'
    sub_dirs = ['tuned', 'eegA', 'eegB', 'eegC', 'eegD']
    quantities = ["accuracy", "val_accuracy", "loss", "val_loss"]

    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']
    Linestyles = ['solid', 'dashed']

    for subdir_idx, sub_dir in enumerate(sub_dirs):
        for quantity_idx in range(4):
            data = np.zeros(150)
            for cv_run in range(1, 6):
                file_name = base_dir + sub_dir + str(cv_run) + '/history.txt'
                data_dict = json.load(open(file_name))
                data += data_dict[quantities[quantity_idx]]
            data /= 5
            if quantity_idx >= 2:
                if quantity_idx % 2:
                    label = f"{sub_dir} test"
                else:
                    label = f"{sub_dir} validation"
                ax[quantity_idx // 2].plot(data, color=colors[subdir_idx], linestyle=Linestyles[quantity_idx % 2],
                                           linewidth=1, label=label)
            else:
                ax[quantity_idx // 2].plot(data, color=colors[subdir_idx], linestyle=Linestyles[quantity_idx % 2],
                                           linewidth=1)

    fig.set_size_inches(14, 6)
    ax[0].set_xlabel("epoch")
    ax[1].set_xlabel("epoch")
    ax[0].set_ylabel("accuracy")
    ax[1].set_ylabel("loss")
    ax[0].grid(True)
    ax[1].grid(True)

    box = ax[0].get_position()
    ax[0].set_position([box.x0 + 0.2 * box.width, box.y0, box.width * 0.8, box.height])
    box = ax[1].get_position()
    ax[1].set_position([box.x0, box.y0, box.width * 0.8, box.height])

    ax[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))

    img_name = "bci2000_model_comparison.png"
    plt.savefig(img_name)
    crop_margins(img_name)


def create_bci2000_LSTM_model_comparison_plot():
    fig, ax = plt.subplots(nrows=1, ncols=2)

    base_dir = '../../logs/fit/bci2000_model_comparison/'
    sub_dirs = ['tuned_LSTM', 'eegD_LSTM']
    quantities = ["accuracy", "val_accuracy", "loss", "val_loss"]

    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    Linestyles = ['solid', 'dashed']

    for subdir_idx, sub_dir in enumerate(sub_dirs):
        for quantity_idx in range(4):
            data = np.zeros(80)
            for cv_run in range(1, 6):
                file_name = base_dir + sub_dir + str(cv_run) + '/history.txt'
                data_dict = json.load(open(file_name))
                data += data_dict[quantities[quantity_idx]]
            data /= 5
            if quantity_idx >= 2:
                if quantity_idx % 2:
                    label = f"{sub_dir} test"
                else:
                    label = f"{sub_dir} validation"
                ax[quantity_idx // 2].plot(data, color=colors[subdir_idx], linestyle=Linestyles[quantity_idx % 2],
                                           linewidth=1, label=label)
            else:
                ax[quantity_idx // 2].plot(data, color=colors[subdir_idx], linestyle=Linestyles[quantity_idx % 2],
                                           linewidth=1)

    fig.set_size_inches(14, 6)
    ax[0].set_xlabel("epoch")
    ax[1].set_xlabel("epoch")
    ax[0].set_ylabel("accuracy")
    ax[1].set_ylabel("loss")
    ax[0].grid(True)
    ax[1].grid(True)

    box = ax[0].get_position()
    ax[0].set_position([box.x0 + 0.2 * box.width, box.y0, box.width * 0.8, box.height])
    box = ax[1].get_position()
    ax[1].set_position([box.x0, box.y0, box.width * 0.8, box.height])

    ax[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))

    img_name = "bci2000_LSTM_model_comparison.png"
    plt.savefig(img_name)
    crop_margins(img_name)


def create_bci2000_image_size_comparison_plot():
    fig, ax = plt.subplots(nrows=1, ncols=2)

    base_dir = '../../logs/fit/bci2000_image_size_comparison/'

    sub_dirs = ['32pixel', '64pixel']
    quantities = ["accuracy", "val_accuracy", "loss", "val_loss"]

    colors = ['tab:blue', 'tab:orange']
    Linestyles = ['solid', 'dashed']

    for subdir_idx, sub_dir in enumerate(sub_dirs):
        for quantity_idx in range(4):
            data = np.zeros(150)
            for cv_run in range(1, 6):
                file_name = base_dir + sub_dir + str(cv_run) + '/history.txt'
                data_dict = json.load(open(file_name))
                data += data_dict[quantities[quantity_idx]]
            data /= 5
            if quantity_idx >= 2:
                if quantity_idx % 2:
                    label = f"{sub_dir} test"
                else:
                    label = f"{sub_dir} validation"
                ax[quantity_idx // 2].plot(data, color=colors[subdir_idx], linestyle=Linestyles[quantity_idx % 2],
                                           linewidth=1, label=label)
            else:
                ax[quantity_idx // 2].plot(data, color=colors[subdir_idx], linestyle=Linestyles[quantity_idx % 2],
                                           linewidth=1)

    fig.set_size_inches(14, 6)
    ax[0].set_xlabel("epoch")
    ax[1].set_xlabel("epoch")
    ax[0].set_ylabel("accuracy")
    ax[1].set_ylabel("loss")
    ax[0].grid(True)
    ax[1].grid(True)

    box = ax[0].get_position()
    ax[0].set_position([box.x0 + 0.2 * box.width, box.y0, box.width * 0.8, box.height])
    box = ax[1].get_position()
    ax[1].set_position([box.x0, box.y0, box.width * 0.8, box.height])

    ax[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))

    img_name = "bci2000_image_size_comparison.png"
    plt.savefig(img_name)
    crop_margins(img_name)


def create_bci2000_frequency_band_comparison_plot():
    fig, ax = plt.subplots(nrows=1, ncols=2)

    base_dir = '../../logs/fit/bci2000_frequency_band_comparison/'

    sub_dirs = ['0bands', '1bands', '2bands']
    quantities = ["accuracy", "val_accuracy", "loss", "val_loss"]

    colors = ['tab:blue', 'tab:orange', 'tab:green']
    Linestyles = ['solid', 'dashed']

    label_names = ["8-13, 13-21, 21-30", "4-8, 8-13, 13-30", "0.5-4, 8-13, 13-30"]

    for subdir_idx, sub_dir in enumerate(sub_dirs):
        for quantity_idx in range(4):
            data = np.zeros(150)
            for cv_run in range(1, 6):
                file_name = base_dir + sub_dir + str(cv_run) + '/history.txt'
                data_dict = json.load(open(file_name))
                data += data_dict[quantities[quantity_idx]]
            data /= 5
            if quantity_idx >= 2:
                if quantity_idx % 2:
                    label = f"{label_names[subdir_idx]} test"
                else:
                    label = f"{label_names[subdir_idx]} validation"
                ax[quantity_idx // 2].plot(data, color=colors[subdir_idx], linestyle=Linestyles[quantity_idx % 2],
                                           linewidth=1, label=label)
            else:
                ax[quantity_idx // 2].plot(data, color=colors[subdir_idx], linestyle=Linestyles[quantity_idx % 2],
                                           linewidth=1)

    fig.set_size_inches(14, 6)
    ax[0].set_xlabel("epoch")
    ax[1].set_xlabel("epoch")
    ax[0].set_ylabel("accuracy")
    ax[1].set_ylabel("loss")
    ax[0].grid(True)
    ax[1].grid(True)

    box = ax[0].get_position()
    ax[0].set_position([box.x0 + 0.25 * box.width, box.y0, box.width * 0.75, box.height])
    box = ax[1].get_position()
    ax[1].set_position([box.x0, box.y0, box.width * 0.75, box.height])

    ax[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    img_name = "bci2000_frequency_band_comparison.png"
    plt.savefig(img_name)
    crop_margins(img_name)


def create_bci2000_class_number_comparison_plot():
    fig, ax = plt.subplots(nrows=1, ncols=2)

    base_dir = '../../logs/fit/bci2000_class_number_comparison/'
    sub_dirs = ['2class', '3class', '4class']
    quantities = ["accuracy", "val_accuracy", "loss", "val_loss"]

    colors = ['tab:blue', 'tab:orange', 'tab:green']
    Linestyles = ['solid', 'dashed']

    for subdir_idx, sub_dir in enumerate(sub_dirs):
        for quantity_idx in range(4):
            data = np.zeros(150)
            for cv_run in range(1, 6):
                file_name = base_dir + sub_dir + str(cv_run) + '/history.txt'
                data_dict = json.load(open(file_name))
                data += data_dict[quantities[quantity_idx]]
            data /= 5
            if quantity_idx >= 2:
                if quantity_idx % 2:
                    label = f"{sub_dir} test"
                else:
                    label = f"{sub_dir} validation"
                ax[quantity_idx // 2].plot(data, color=colors[subdir_idx], linestyle=Linestyles[quantity_idx % 2],
                                           linewidth=1, label=label)
            else:
                ax[quantity_idx // 2].plot(data, color=colors[subdir_idx], linestyle=Linestyles[quantity_idx % 2],
                                           linewidth=1)

    fig.set_size_inches(14, 6)
    ax[0].set_xlabel("epoch")
    ax[1].set_xlabel("epoch")
    ax[0].set_ylabel("accuracy")
    ax[1].set_ylabel("loss")
    ax[0].grid(True)
    ax[1].grid(True)

    box = ax[0].get_position()
    ax[0].set_position([box.x0 + 0.2 * box.width, box.y0, box.width * 0.8, box.height])
    box = ax[1].get_position()
    ax[1].set_position([box.x0, box.y0, box.width * 0.8, box.height])

    ax[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    img_name = "bci2000_class_number_comparison.png"
    plt.savefig(img_name)
    crop_margins(img_name)


def create_bci2aiv_model_comparison_plot():
    fig, ax = plt.subplots(nrows=1, ncols=2)

    base_dir = '../../logs/fit/bci2aiv_model_comparison/'
    sub_dirs = ['tuned', 'eegA', 'eegB', 'eegC', 'eegD']
    quantities = ["accuracy", "val_accuracy", "loss", "val_loss"]

    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']
    Linestyles = ['solid', 'dashed']

    for subdir_idx, sub_dir in enumerate(sub_dirs):
        for quantity_idx in range(4):
            data = np.zeros(150)
            for subj_num in range(1, 10):
                file_name = base_dir + sub_dir + str(subj_num) + f'/history_{subj_num}.txt'
                data_dict = json.load(open(file_name))
                data += data_dict[quantities[quantity_idx]]
            data /= 9
            if quantity_idx >= 2:
                if quantity_idx % 2:
                    label = f"{sub_dir} test"
                else:
                    label = f"{sub_dir} validation"
                ax[quantity_idx // 2].plot(data, color=colors[subdir_idx], linestyle=Linestyles[quantity_idx % 2],
                                           linewidth=1, label=label)
            else:
                ax[quantity_idx // 2].plot(data, color=colors[subdir_idx], linestyle=Linestyles[quantity_idx % 2],
                                           linewidth=1)

    fig.set_size_inches(14, 6)
    ax[0].set_xlabel("epoch")
    ax[1].set_xlabel("epoch")
    ax[0].set_ylabel("accuracy")
    ax[1].set_ylabel("loss")
    ax[0].grid(True)
    ax[1].grid(True)

    box = ax[0].get_position()
    ax[0].set_position([box.x0 + 0.2 * box.width, box.y0, box.width * 0.8, box.height])
    box = ax[1].get_position()
    ax[1].set_position([box.x0, box.y0, box.width * 0.8, box.height])

    ax[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))

    img_name = "bci2aiv_model_comparison.png"
    plt.savefig(img_name)
    crop_margins(img_name)


def create_bci2aiv_LSTM_model_comparison_plot():
    fig, ax = plt.subplots(nrows=1, ncols=2)

    base_dir = '../../logs/fit/bci2aiv_model_comparison/'
    sub_dirs = ['tuned_LSTM', 'eegD_LSTM']
    quantities = ["accuracy", "val_accuracy", "loss", "val_loss"]

    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    Linestyles = ['solid', 'dashed']

    for subdir_idx, sub_dir in enumerate(sub_dirs):
        for quantity_idx in range(4):
            data = np.zeros(150)
            for subj_num in range(1, 10):
                file_name = base_dir + sub_dir + str(subj_num) + f'/history_{subj_num}.txt'
                data_dict = json.load(open(file_name))
                data += data_dict[quantities[quantity_idx]]
            data /= 9
            if quantity_idx >= 2:
                if quantity_idx % 2:
                    label = f"{sub_dir} test"
                else:
                    label = f"{sub_dir} validation"
                ax[quantity_idx // 2].plot(data[0:80], color=colors[subdir_idx], linestyle=Linestyles[quantity_idx % 2],
                                           linewidth=1, label=label)
            else:
                ax[quantity_idx // 2].plot(data[0:80], color=colors[subdir_idx], linestyle=Linestyles[quantity_idx % 2],
                                           linewidth=1)

    fig.set_size_inches(14, 6)
    ax[0].set_xlabel("epoch")
    ax[1].set_xlabel("epoch")
    ax[0].set_ylabel("accuracy")
    ax[1].set_ylabel("loss")
    ax[0].grid(True)
    ax[1].grid(True)

    box = ax[0].get_position()
    ax[0].set_position([box.x0 + 0.2 * box.width, box.y0, box.width * 0.8, box.height])
    box = ax[1].get_position()
    ax[1].set_position([box.x0, box.y0, box.width * 0.8, box.height])

    ax[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))

    img_name = "bci2aiv_LSTM_model_comparison.png"
    plt.savefig(img_name)
    crop_margins(img_name)


def create_bci2aiv_image_size_comparison_plot():
    fig, ax = plt.subplots(nrows=1, ncols=2)

    base_dir = '../../logs/fit/bci2aiv_image_size_comparison/'

    sub_dirs = ['32pixel', '64pixel']
    quantities = ["accuracy", "val_accuracy", "loss", "val_loss"]

    colors = ['tab:blue', 'tab:orange']
    Linestyles = ['solid', 'dashed']

    for subdir_idx, sub_dir in enumerate(sub_dirs):
        for quantity_idx in range(4):
            data = np.zeros(150)
            for subj_num in range(1, 10):
                file_name = base_dir + sub_dir + str(subj_num) + f'/history_{subj_num}.txt'
                data_dict = json.load(open(file_name))
                data += data_dict[quantities[quantity_idx]]
            data /= 9
            if quantity_idx >= 2:
                if quantity_idx % 2:
                    label = f"{sub_dir} test"
                else:
                    label = f"{sub_dir} validation"
                ax[quantity_idx // 2].plot(data, color=colors[subdir_idx], linestyle=Linestyles[quantity_idx % 2],
                                           linewidth=1, label=label)
            else:
                ax[quantity_idx // 2].plot(data, color=colors[subdir_idx], linestyle=Linestyles[quantity_idx % 2],
                                           linewidth=1)

    fig.set_size_inches(14, 6)
    ax[0].set_xlabel("epoch")
    ax[1].set_xlabel("epoch")
    ax[0].set_ylabel("accuracy")
    ax[1].set_ylabel("loss")
    ax[0].grid(True)
    ax[1].grid(True)

    box = ax[0].get_position()
    ax[0].set_position([box.x0 + 0.2 * box.width, box.y0, box.width * 0.8, box.height])
    box = ax[1].get_position()
    ax[1].set_position([box.x0, box.y0, box.width * 0.8, box.height])

    ax[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))

    img_name = "bci2aiv_image_size_comparison.png"
    plt.savefig(img_name)
    crop_margins(img_name)


def create_bci2aiv_frequency_band_comparison_plot():
    fig, ax = plt.subplots(nrows=1, ncols=2)

    base_dir = '../../logs/fit/bci2aiv_frequency_band_comparison/'

    sub_dirs = ['0bands', '1bands', '2bands']
    quantities = ["accuracy", "val_accuracy", "loss", "val_loss"]

    colors = ['tab:blue', 'tab:orange', 'tab:green']
    Linestyles = ['solid', 'dashed']

    label_names = ["8-13, 13-21, 21-30", "4-8, 8-13, 13-30", "0.5-4, 8-13, 13-30"]

    for subdir_idx, sub_dir in enumerate(sub_dirs):
        for quantity_idx in range(4):
            data = np.zeros(150)
            for subj_num in range(1, 10):
                file_name = base_dir + sub_dir + str(subj_num) + f'/history_{subj_num}.txt'
                data_dict = json.load(open(file_name))
                data += data_dict[quantities[quantity_idx]]
            data /= 9
            if quantity_idx >= 2:
                if quantity_idx % 2:
                    label = f"{label_names[subdir_idx]} test"
                else:
                    label = f"{label_names[subdir_idx]} validation"
                ax[quantity_idx // 2].plot(data, color=colors[subdir_idx], linestyle=Linestyles[quantity_idx % 2],
                                           linewidth=1, label=label)
            else:
                ax[quantity_idx // 2].plot(data, color=colors[subdir_idx], linestyle=Linestyles[quantity_idx % 2],
                                           linewidth=1)

    fig.set_size_inches(14, 6)
    ax[0].set_xlabel("epoch")
    ax[1].set_xlabel("epoch")
    ax[0].set_ylabel("accuracy")
    ax[1].set_ylabel("loss")
    ax[0].grid(True)
    ax[1].grid(True)

    box = ax[0].get_position()
    ax[0].set_position([box.x0 + 0.25 * box.width, box.y0, box.width * 0.75, box.height])
    box = ax[1].get_position()
    ax[1].set_position([box.x0, box.y0, box.width * 0.75, box.height])

    ax[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    img_name = "bci2aiv_frequency_band_comparison.png"
    plt.savefig(img_name)
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
    create_bci2000_model_comparison_plot()
    create_bci2000_LSTM_model_comparison_plot()
    create_bci2000_image_size_comparison_plot()
    create_bci2000_frequency_band_comparison_plot()
    create_bci2000_class_number_comparison_plot()
    create_bci2aiv_model_comparison_plot()
    create_bci2aiv_LSTM_model_comparison_plot()
    create_bci2aiv_image_size_comparison_plot()
    create_bci2aiv_frequency_band_comparison_plot()
