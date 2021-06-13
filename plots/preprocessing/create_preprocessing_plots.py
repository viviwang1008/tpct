"""
Collection of code used to create plots
"""
from data_preprocessing.data_preprocessor import DataPreprocessor
import matplotlib.pyplot as plt
import numpy as np
import json


def create_coordinates_plot() -> None:
    img_size = json.load(open("../../data/args_bci2000_preprocess.txt", 'r'))['grid_points']

    data_preprocessor = DataPreprocessor(num_windows=1, num_grid_points=img_size, pseudo_channels=True, frequency_band_set=1, dataset="BCI2000")
    n_coordinates = data_preprocessor.channel_coordinates
    gx, gy = data_preprocessor.x_image, data_preprocessor.y_image
    plt.figure(figsize=(10, 10))
    plt.scatter(x=gx.flatten(), y=gy.flatten(), marker='.', color='grey')
    plt.scatter(x=n_coordinates[:, 0], y=n_coordinates[:, 1], marker='x', color='m', s=125)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"BCI2000 2D Channel Coordinates")
    plt.legend(("Image Samples", "Channel Coordinates'"))
    plt.savefig("BCI2000_Coordinates.png")


def create_rgb_images(subject_num: int) -> None:
    data = np.load(f"../../data/bci2000_preprocessed_subj_{subject_num}_class_4.npy").squeeze()
    labels = np.load(f"../../data/bci2000_labels_subj_{subject_num}_class_4.npy")

    num_img_x = 5
    num_img_y = 5

    # pick first few images
    data = data[0:num_img_x * num_img_y, :, :, :]

    # normalize each channel to [0..1]
    for color_idx in range(3):
        mean = np.mean(data[:, :, :, color_idx])
        std = np.std(data[:, :, :, color_idx])
        data[:, :, :, color_idx] = np.abs((data[:, :, :, color_idx] - mean)/std)
        data[:, :, :, color_idx] = data[:, :, :, color_idx] / np.max(data[:, :, :, color_idx])

    # plot all images in one figure
    fig, ax = plt.subplots(5, 5)
    image_counter = 0
    for img_y in range(num_img_y):
        for img_x in range(num_img_x):
            image_data = data[image_counter, :, :, :]
            image_counter += 1
            plt.subplot(num_img_x, num_img_y, image_counter)
            plt.imshow(image_data, origin='lower', interpolation='None', resample=False)
            #label = int(labels[image_counter])
            if img_x == 2 and img_y == 0:
                plt.title(f"subject {subject_num}")
            #plt.title(f"{image_counter}:label={label}")
            plt.axis("off")
    fig.tight_layout()
    #plt.savefig(f"subj{subject_num}_Images.png")
    plt.show()


def create_bci2000_separated_images(subject_num: int) -> None:
    data = np.load(f"../../data/bci2000_preprocessed_subj_{subject_num}_class_4.npy").squeeze()
    labels = np.load(f"../../data/bci2000_labels_subj_{subject_num}_class_4.npy")

    # normalize each channel to [0..1]
    for color_idx in range(3):
        data[:, :, :, color_idx] = data[:, :, :, color_idx] - np.min(data[:, :, :, color_idx])
        data[:, :, :, color_idx] = data[:, :, :, color_idx] / np.max(data[:, :, :, color_idx])

    num_trials_per_class = 6

    label_idx_map = []
    for label_class in range(0, 4):
        label_idx_map.append(np.where(labels == label_class)[0][0:num_trials_per_class])

    color_maps = ['Reds', 'Greens', 'Blues']

    fig, ax = plt.subplots(num_trials_per_class, 3 * 4)
    fig.set_size_inches(18, 9)
    for y_idx in range(0, num_trials_per_class):
        for x_idx in range(0, 4):
            for color_idx in range(0, 3):
                trial_idx = label_idx_map[x_idx][y_idx]
                image_data = data[trial_idx, :, :, color_idx]
                ax[y_idx, 3 * x_idx + color_idx].imshow(image_data, cmap=color_maps[color_idx], origin='lower',
                                                        interpolation='None')
                ax[y_idx, 3 * x_idx + color_idx].set_xticks([])
                ax[y_idx, 3 * x_idx + color_idx].set_yticks([])
                if color_idx == 0:
                    ax[y_idx, 3 * x_idx + color_idx].set_ylabel(f"trial_idx: {trial_idx}")
                if y_idx == 0 and color_idx == 1:
                    ax[y_idx, 3 * x_idx + color_idx].set_title(f"label: {x_idx}")
                if y_idx == num_trials_per_class - 1:
                    ax[y_idx, 3 * x_idx + color_idx].set_xlabel(color_maps[color_idx][0])

    fig.tight_layout()
    plt.savefig(f"subj{subject_num}_bci2000_RGB_split_images.png")


if __name__ == "__main__":
    create_coordinates_plot()
    create_bci2000_separated_images(subject_num=2)
    create_rgb_images(subject_num=21)
