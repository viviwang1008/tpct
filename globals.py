"""
globals.py: collection of constants and parameters used in different places of the code
"""
# From data description
subject_num_range = range(1, 10)

num_channels = 22

num_runs = 6
num_trials_per_run = 48
num_trials = num_runs * num_trials_per_run  # 288

run_time_s = 7
sampling_frequency_hz = 250
num_samples_per_run = run_time_s * sampling_frequency_hz  # 1750

motor_imagery_start_s = 3
motor_imagery_stop_s = 6
motor_imagery_time_s = motor_imagery_stop_s - motor_imagery_start_s

motor_imagery_samples_start = motor_imagery_start_s * sampling_frequency_hz  # 750
motor_imagery_samples_stop = motor_imagery_stop_s * sampling_frequency_hz  # 1500
num_motor_imagery_samples_per_run = motor_imagery_samples_stop - motor_imagery_samples_start  # 750

# For processing
ch_names = [
                            'Fz',
            'FC3',  'FC1',  'FCz',  'FC2',  'FC4',
    'C5',   'C3',   'C1',   'Cz',   'C2',   'C4',   'C6',
            'CP3',  'CP1',  'CPz',  'CP2',  'CP4',
                    'P1',   'Pz',   'P2',
                            'POz',
]

frequency_bands_hz = [[0.5, 4], [8, 13], [13, 30]]
num_frequency_bands = len(frequency_bands_hz)
