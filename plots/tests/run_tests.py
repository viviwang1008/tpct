from typing import Tuple

import globals as g
import numpy as np
import math as m
import matplotlib.pyplot as plt
from data_preprocessing.data_preprocessor import DataPreprocessor


def create_test_session_signals() -> Tuple[np.ndarray, np.ndarray]:
    label_1_electrodes = np.asarray([1, 4, 7]) - 1  # "-1" for 0-based indexing
    label_2_electrodes = np.asarray([57, 58, 59]) - 1
    label_3_electrodes = np.asarray([43, 11, 44]) - 1
    label_4_electrodes = np.asarray([23, 18, 64]) - 1

    first_frequency = np.average(g.frequency_bands_hz_sets[0][0])*0.8
    second_frequency = np.average(g.frequency_bands_hz_sets[0][1])
    third_frequency = np.average(g.frequency_bands_hz_sets[0][2])

    first_signal = np.asarray([m.sin(first_frequency * 2 * m.pi * i) for i in np.linspace(0, 3, g.num_motor_imagery_samples)])
    second_signal = np.asarray([m.sin(second_frequency * 2 * m.pi * i) for i in np.linspace(0, 3, g.num_motor_imagery_samples)])
    third_signal = np.asarray([m.sin(third_frequency * 2 * m.pi * i) for i in np.linspace(0, 3, g.num_motor_imagery_samples)])

    color_signals = np.empty((3, g.num_motor_imagery_samples))
    color_signals[0, :] = first_signal
    color_signals[1, :] = second_signal
    color_signals[2, :] = third_signal

    data = np.zeros((4, g.num_channels_bci2000, g.num_motor_imagery_samples))
    for color_idx in range(0, 3):
        data[0, label_1_electrodes[color_idx], :] = color_signals[color_idx]
        data[1, label_2_electrodes[color_idx], :] = color_signals[color_idx]
        data[2, label_3_electrodes[color_idx], :] = color_signals[color_idx]
        data[3, label_4_electrodes[color_idx], :] = color_signals[color_idx]

    labels = np.asarray([1, 2, 3, 4])
    return data, labels


def plot_test_images(data: np.ndarray) -> None:
    color_maps = ['Reds', 'Greens', 'Blues']

    fig, ax = plt.subplots(4, 3)
    for trial_idx in range(0, 4):
        for color_idx in range(0, 3):
            image_data = data[trial_idx, :, :, color_idx]
            ax[trial_idx, color_idx].imshow(image_data, cmap=color_maps[color_idx], origin='lower', interpolation='None')
            ax[trial_idx, color_idx].set_xticks([])
            ax[trial_idx, color_idx].set_yticks([])
            if trial_idx == 0:
                ax[trial_idx, color_idx].set_title(f"{g.frequency_bands_hz_sets[0][color_idx]} hz")
            if color_idx == 0:
                ax[trial_idx, color_idx].set_ylabel(f"trial: {trial_idx}")

    fig.tight_layout()
    plt.savefig("RGB_test_images.png")


if __name__ == "__main__":
    test_data, test_labels = create_test_session_signals()

    preprocessed_test_data = DataPreprocessor(num_windows=1,
                                              num_grid_points=64,
                                              pseudo_channels=True,
                                              dataset="BCI2000",
                                              frequency_band_set=1).process_session_data(test_data).squeeze()

    plot_test_images(preprocessed_test_data)
