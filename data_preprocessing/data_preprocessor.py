from typing import Tuple
import numpy as np
from mne.channels import make_standard_montage
from scipy.interpolate import griddata
import globals as g
import math as m


def cart2sph(x: float, y: float, z: float) -> Tuple[float, float, float]:
    """
    Transform Cartesian coordinates to spherical
    :param x: X coordinate
    :param y: Y coordinate
    :param z: Z coordinate
    :return: radius, elevation, azimuth
    """
    x2_y2 = x ** 2 + y ** 2
    r = m.sqrt(x2_y2 + z ** 2)  # r
    elev = m.atan2(z, m.sqrt(x2_y2))  # Elevation
    az = m.atan2(y, x)  # Azimuth
    return r, elev, az


def pol2cart(theta: float, rho: float) -> Tuple[float, float]:
    """
    Transform polar coordinates to Cartesian
    :param theta: angle value
    :param rho: radius value
    :return: X, Y
    """
    return rho * m.cos(theta), rho * m.sin(theta)


class DataPreprocessor:
    """
    A DataPreprocessor object does the power extraction from signals, calculates an interpolator and applies it
    """

    def __init__(self, num_windows: int, num_grid_points: int, pseudo_channels: bool,
                 frequency_band_set: int, dataset: str):
        self.dataset = dataset
        self.num_windows = num_windows
        self.num_grid_points = num_grid_points
        self.frequency_band_set = frequency_band_set
        self.pseudo_channels = pseudo_channels
        self.channel_coordinates = self._calculate_2d_channel_coordinates()
        self.x_image, self.y_image = self._calculate_image_coordinates()

    def _calculate_2d_channel_coordinates(self) -> np.ndarray:
        """
        Calculates 2d channel coordinates. Locations are taken from mne.channels.make_standard_montage and projected
        using azimuthal projection.
        :return: coordinates. numpy ndarray of shape (num_channels, 2)
        """
        montage = make_standard_montage(kind='standard_1020')
        if self.dataset == "BCI2000":
            positions_3d = np.asarray([montage.get_positions()['ch_pos'][ch_name] for ch_name in g.ch_names_bci2000])
        elif self.dataset == "BCI2aIV":
            positions_3d = np.asarray([montage.get_positions()['ch_pos'][ch_name] for ch_name in g.ch_names_bci2aiv])
        else:
            raise ValueError()

        azimuth_projected_positions = []
        for pos_3d in positions_3d:
            [_, elev, az] = cart2sph(pos_3d[0], pos_3d[1], pos_3d[2])
            azimuth_projected_positions.append(pol2cart(az, np.pi / 2 - elev))

        azimuth_projected_positions = np.asarray(azimuth_projected_positions)
        if self.pseudo_channels:
            x_min = np.min(azimuth_projected_positions[:, 0])
            x_max = np.max(azimuth_projected_positions[:, 0])
            y_min = np.min(azimuth_projected_positions[:, 1])
            y_max = np.max(azimuth_projected_positions[:, 1])
            corner_positions = np.asarray([[x_min, y_min],
                                           [x_min, y_max],
                                           [x_max, y_min],
                                           [x_max, y_max]])
            azimuth_projected_positions = np.concatenate([azimuth_projected_positions, corner_positions])

        return azimuth_projected_positions

    def _calculate_image_coordinates(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculates image coordinates for the (grid_points x grid_points) image.
        :return: Tuple of numpy ndarray mesh grids
        """
        x_min = min(self.channel_coordinates[:, 0])
        x_max = max(self.channel_coordinates[:, 0])
        y_min = min(self.channel_coordinates[:, 1])
        y_max = max(self.channel_coordinates[:, 1])
        x = np.linspace(x_min, x_max, self.num_grid_points)
        y = np.linspace(y_min, y_max, self.num_grid_points)
        x_image, y_image = np.meshgrid(x, y)
        return x_image, y_image

    def process_session_data(self, session_signals: np.ndarray) -> np.ndarray:
        """
        Calculates frequency range features for each electrode and interpolates that data to one RGB image for each
        window of a trial.
        :param session_signals: numpy array containing all signals of a session
        :return: numpy array containing an image for each trial
        """
        session_signals = self._normalize_signals(session_signals)
        signal_power = self._calculate_signal_power(session_signals)
        interpolated_images = self._interpolate_all_data(signal_power)
        return interpolated_images

    def _normalize_signals(self, session_signals) -> np.ndarray:
        """
        normalize the data for each subject per each channel
        :param session_signals: numpy array containing all signals of a session
        :return: session signals with each channel normalized to [0..1]
        """
        if self.dataset == "BCI2000":
            channel_range = range(g.num_channels_bci2000)
        elif self.dataset == "BCI2aIV":
            channel_range = range(g.num_channels_bci2aiv)
        else:
            raise ValueError()

        for channel_idx in channel_range:
            channel_data = session_signals[:, channel_idx, :]
            if np.min(channel_data) == np.max(channel_data) == 0.0:  # this should only be true in tests, not real data
                print("all zero signal. continue")
                continue
            channel_data -= np.min(channel_data)
            channel_data /= np.max(channel_data)
            session_signals[:, channel_idx, :] = channel_data
        return session_signals

    def _calculate_signal_power(self, session_signals: np.ndarray) -> np.ndarray:
        """
        For each trial and channel, calculates one set of frequency band features for each window.
        :param session_signals: Signal data from a session, shape (num_valid_trial, num_channels, num_motor_imagery_samples)
        :return: numpy array, shape (num_valid_trial, num_windows, num_channels, num_frequency_bands)
        """

        if self.dataset == "BCI2000":
            num_channels = g.num_channels_bci2000
        elif self.dataset == "BCI2aIV":
            num_channels = g.num_channels_bci2aiv
        else:
            raise ValueError()

        num_valid_trials = session_signals.shape[0]
        session_frequency_features = np.empty((num_valid_trials,
                                               self.num_windows,
                                               num_channels,
                                               g.num_frequency_bands))

        for trial_idx in range(num_valid_trials):
            for channel_idx in range(num_channels):
                trial_channel_signal = session_signals[trial_idx, channel_idx, :]
                signal_windows = np.array_split(trial_channel_signal, self.num_windows)
                for window_idx, window in enumerate(signal_windows):
                    frequency_features = self._calculate_frequency_feature(window)
                    session_frequency_features[trial_idx, window_idx, channel_idx, :] = frequency_features

        return session_frequency_features

    def _calculate_frequency_feature(self, window_data: np.ndarray) -> np.ndarray:
        """
        Calculates the power across the three frequency bands for one channel as described in the Fadel paper
        :param window_data: data of one window, numpy array of shape (num_motor_imagery_samples/num_time_windows, )
        :return: three frequency features values, numpy ndarray of shape (3,)
        """
        if self.dataset == "BCI2000":
            sampling_rate = g.sampling_frequency_hz_bci2000
        elif self.dataset == "BCI2aIV":
            sampling_rate = g.sampling_frequency_hz_bci2aiv
        else:
            raise ValueError()

        squared_absolute_fft_bands = np.empty(3)
        fft = np.fft.fft(window_data)
        freq_hz = np.fft.fftfreq(n=window_data.shape[-1], d=1 / sampling_rate)
        for idx, frequency_band_hz in enumerate(g.frequency_bands_hz_sets[self.frequency_band_set]):
            low = frequency_band_hz[0]
            high = frequency_band_hz[1]
            selected_fft_values = fft[(abs(freq_hz) >= low) & (abs(freq_hz) <= high)]
            squared_absolute_fft_bands[idx] = np.sum(np.abs(selected_fft_values) ** 2)
        return squared_absolute_fft_bands

    def _interpolate_all_data(self, frequency_features: np.ndarray) -> np.ndarray:
        """
        For each trial, assigns frequency features to coordinates and interpolates those values to a square image
        :param: frequency_features: for each valid trial, numpy ndarray of shape (num_valid_trial, num_windows, num_channels, 3)
        :return: interpolated image for each trial, numpy ndarray of shape (num_valid_trial, num_windows, grid_points, grid_points, 3)
        """
        num_valid_trials = frequency_features.shape[0]
        interpolated_data = np.empty((num_valid_trials,
                                      self.num_windows,
                                      self.num_grid_points,
                                      self.num_grid_points,
                                      3))

        for trial_idx in range(num_valid_trials):
            for window_idx in range(self.num_windows):
                channel_band_powers = frequency_features[trial_idx, window_idx, :, :]
                interpolated_data[trial_idx, window_idx, :, :, :] = self._interpolate_image(channel_band_powers)

        return interpolated_data

    def _interpolate_image(self, channel_band_powers: np.ndarray) -> np.ndarray:
        """
        Interpolates band power data using scipy.interpolate.griddata
        :param channel_band_powers: band power data for channels used for interpolation of shape (num_channels, 3)
        :return: interpolated data
        """

        interpolated_data_rgb = np.empty((self.num_grid_points,
                                          self.num_grid_points,
                                          g.num_frequency_bands))

        if self.pseudo_channels:
            # add four (0, 0, 0) values for the pseudo-channels in the corners
            channel_band_powers = np.concatenate([channel_band_powers, np.zeros([4, 3])])

        for color_channel_idx in range(g.num_frequency_bands):
            interpolated_data_rgb[:, :, color_channel_idx] = griddata(points=self.channel_coordinates,
                                                                      values=channel_band_powers[:, color_channel_idx],
                                                                      xi=(self.x_image, self.y_image),
                                                                      method='cubic',
                                                                      fill_value=0)

        return interpolated_data_rgb
