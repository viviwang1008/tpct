"""
Collection of code used to create plots
"""
from data_preprocessing.data_preprocessor import DataPreprocessor
import globals as g
import matplotlib.pyplot as plt
import numpy as np


def create_coordinates_plot() -> None:
    data_preprocessor = DataPreprocessor(num_windows=6, num_grid_points=64, pseudo_channels=True)
    n_coordinates = data_preprocessor.channel_coordinates
    gx, gy = data_preprocessor.x_image, data_preprocessor.y_image
    plt.figure(figsize=(10, 10))
    plt.scatter(x=gx.flatten(), y=gy.flatten(), marker='.', color='grey')
    plt.scatter(x=n_coordinates[:, 0], y=n_coordinates[:, 1], marker='x', color='m', s=125)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"2dCoordinates")
    plt.legend(("M", "N'"))
    plt.savefig("Coordinates.png")


def create_rgb_images(subject_num: int) -> None:
    data = np.load(f"../data/preprocessed_A0{subject_num}T.npy")
    labels = np.load(f"../data/labels_A0{subject_num}T.npy")

    num_img_x = 6
    num_img_y = 6

    # pick first few images
    data = data[0:num_img_x * num_img_y, :, :, :]

    # normalize each channel to [0..1]
    for color_idx in range(3):
        data[:, :, :, color_idx] = data[:, :, :, color_idx] - np.min(data[:, :, :, color_idx])
        data[:, :, :, color_idx] = data[:, :, :, color_idx] / np.max(data[:, :, :, color_idx])

    # plot all images in one figure
    image_counter = 0
    for img_y in range(num_img_y):
        for img_x in range(num_img_x):
            image_data = data[image_counter, :, :, :]
            label = int(labels[image_counter])
            image_counter += 1
            plt.subplot(num_img_x, num_img_y, image_counter)
            plt.imshow(image_data, origin='lower', interpolation='None', resample=False)
            plt.title(f"{image_counter}:label={label}")
            plt.axis("off")
    plt.savefig(f"subj{subject_num}_Images.png")


def create_rgb_power_histogram(subject_num: int) -> None:
    data = np.load(f"../data/preprocessed_A0{subject_num}T.npy")
    labels = np.load(f"../data/labels_A0{subject_num}T.npy")
    fig, ax = plt.subplots(1, 3)
    fig.set_size_inches(18, 7)

    for color_idx in range(0, 3):
        color_data = data[:, :, :, color_idx]
        class_1_data = color_data[labels == 1, :, :].flatten()
        class_2_data = color_data[labels == 2, :, :].flatten()
        class_3_data = color_data[labels == 3, :, :].flatten()
        class_4_data = color_data[labels == 4, :, :].flatten()
        rgb_data = [class_1_data, class_2_data, class_3_data, class_4_data]
        ax[color_idx].hist(rgb_data, bins=30, label=['1', '2', '3', '4'], density=True)
        ax[color_idx].set_xlabel("Power")
        ax[color_idx].legend(['1', '2', '3', '4'])
        ax[color_idx].set_title(f"{g.frequency_bands_hz[color_idx]}hz")
        if color_idx == 0:
            ax[color_idx].set_ylabel("density")

    fig.tight_layout()
    plt.savefig(f"subj{subject_num}_RGB_histogram.png")


def create_separated_images(subject_num: int) -> None:
    data = np.load(f"../data/preprocessed_A0{subject_num}T.npy")
    labels = np.load(f"../data/labels_A0{subject_num}T.npy")

    # normalize each channel to [0..1]
    for color_idx in range(3):
        data[:, :, :, color_idx] = data[:, :, :, color_idx] - np.min(data[:, :, :, color_idx])
        data[:, :, :, color_idx] = data[:, :, :, color_idx] / np.max(data[:, :, :, color_idx])

    num_trials_per_class = 6

    label_idx_map = []
    for label_class in range(1, 5):
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
                    ax[y_idx, 3 * x_idx + color_idx].set_title(f"label: {x_idx + 1}")
                if y_idx == num_trials_per_class - 1:
                    ax[y_idx, 3 * x_idx + color_idx].set_xlabel(color_maps[color_idx][0])

    fig.tight_layout()
    plt.savefig(f"subj{subject_num}_RGB_split_images.png")


if __name__ == "__main__":
    # create_coordinates_plot()
    # create_rgb_images(subject_num=1)
    # create_separated_images(subject_num=1)
    # create_rgb_power_histogram(subject_num=1)
    raise NotImplementedError("These plot making functions were made for the old format where the windows were averaged."
                              "They have to be slightly adapted before working again.")
