# Time Domain Power Clough-Tocher Interpolation Imaging for Motor Imagery Classification

## Semester Project at the Department of Information Technology and Electrical Engineering

### How to run
Make sure the packages listed in "requirements.txt" are installed.

To run training for [dataset 2a of BCI Competition IV](http://bnci-horizon-2020.eu/database/data-sets):

- ```python main_preprocess_data_bci2aiv.py -p "path/to/2a of BCI Competition IV"/data/```

- ```python main_train_bci2aiv.py```

To run training for main_train for [BCI 2000 dataset](https://physionet.org/content/eegmmidb/1.0.0/):

- ```python main_preprocess_data_bci2aiv.py -p "path/to/BCI 2000/data/```

- ```python main_train_bci2000.py```

All ```main_*.py``` scripts use argparse. Get more information by running ```python main_*.py -h```.


### References
[1] Fadel, Ward & Kollod, Csaba & Wahdow, Moutz & Ibrahim, Yahya & Ulbert, István. (2020). Multi-Class Classification of Motor Imagery EEG Signals Using Image-Based Deep Recurrent Convolutional Neural Network. 1-4. 10.1109/BCI48061.2020.9061622.

[2] Bashivan, Pouya & Rish, Irina & Yeasin, M. & Codella, Noel. (2015). Learning Representations from EEG with Deep Recurrent-Convolutional Neural Networks.
