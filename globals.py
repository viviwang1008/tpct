"""
globals.py: collection of constants and parameters used in different places of the code
"""
# For BCI 2a IV dataset
subject_num_range_bci2aiv = range(1, 10)

num_channels_bci2aiv = 22

num_runs = 6
num_trials_per_run = 48
num_trials = num_runs * num_trials_per_run  # 288

run_time_s = 7
sampling_frequency_hz_bci2aiv = 250
num_samples_per_run = run_time_s * sampling_frequency_hz_bci2aiv  # 1750

motor_imagery_start_s = 3
motor_imagery_stop_s = 6
motor_imagery_time_s = motor_imagery_stop_s - motor_imagery_start_s

motor_imagery_samples_start = motor_imagery_start_s * sampling_frequency_hz_bci2aiv  # 750
motor_imagery_samples_stop = motor_imagery_stop_s * sampling_frequency_hz_bci2aiv  # 1500
num_motor_imagery_samples_per_run = motor_imagery_samples_stop - motor_imagery_samples_start  # 750

# For processing
ch_names_bci2aiv = [
                            'Fz',
            'FC3',  'FC1',  'FCz',  'FC2',  'FC4',
    'C5',   'C3',   'C1',   'Cz',   'C2',   'C4',   'C6',
            'CP3',  'CP1',  'CPz',  'CP2',  'CP4',
                    'P1',   'Pz',   'P2',
                            'POz',
]

# For BCI 2000 dataset
excluded_subjects_bci2000 = [88, 92, 100, 104]
subject_nums_bci2000 = [x for x in range(1, 110) if (x not in excluded_subjects_bci2000)]

num_channels_bci2000 = 64
num_trials_per_class_bci2000 = 21
sampling_frequency_hz_bci2000 = 160
motor_imagery_time_s_bci2000 = 4
num_motor_imagery_samples = sampling_frequency_hz_bci2000 * motor_imagery_time_s_bci2000  # 640

# For processing
ch_names_bci2000 = [
                                    'Fp1',  'Fpz',  'Fp2',
            'AF7',          'AF3',          'AFz',          'AF4',          'AF8',
            'F7',   'F5',   'F3',   'F1',   'Fz',   'F2',   'F4',   'F6',   'F8',
            'FT7',  'FC5',  'FC3',  'FC1',  'FCz',  'FC2',  'FC4',  'FC6',  'FT8',
    'T9',   'T7',   'C5',   'C3',   'C1',   'Cz',   'C2',   'C4',   'C6',   'T8',   'T10',
            'TP7',  'CP5',  'CP3',  'CP1',  'CPz',  'CP2',  'CP4',  'CP6',  'TP8',
            'P7',   'P5',   'P3',   'P1',   'Pz',   'P2',   'P4',   'P6',   'P8',
            'PO7',          'PO3',          'POz',          'PO4',          'PO8',
                                    'O1',   'Oz',   'O2',
                                            'Iz'
]

frequency_bands_hz_sets = [[[8, 13], [13, 21], [21, 30]],
                           [[4, 8], [8, 13], [13, 30]],
                           [[0.5, 4], [8, 13], [13, 30]]]
num_frequency_bands = 3
