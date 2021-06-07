# different models
python main_preprocess_data_bci2000.py -w 1 -p /scratch/sem21f18/
python main_train_bci2000.py -m tuned -lr 0.0005 -b1 0.96 -b2 0.99965 -e 150 -p 150 -cf 45 -o bci2000_model_comparison/tuned
python main_train_bci2000.py -m eegA -lr 0.0005 -b1 0.96 -b2 0.99965 -e 150 -p 150 -o bci2000_model_comparison/eegA
python main_train_bci2000.py -m eegB -lr 0.0005 -b1 0.96 -b2 0.99965 -e 150 -p 150 -o bci2000_model_comparison/eegB
python main_train_bci2000.py -m eegC -lr 0.0005 -b1 0.96 -b2 0.99965 -e 150 -p 150 -o bci2000_model_comparison/eegC
python main_train_bci2000.py -m eegD -lr 0.0005 -b1 0.96 -b2 0.99965 -e 150 -p 150 -o bci2000_model_comparison/eegD

python main_preprocess_data_bci2000.py -w 6 -p /scratch/sem21f18/
python main_train_bci2000.py -m tuned_LSTM -lr 0.0005 -b1 0.96 -b2 0.99965 -e 80 -p 80 -cf 45 -o bci2000_model_comparison/tuned_LSTM
python main_train_bci2000.py -m eegD_LSTM -lr 0.0005 -b1 0.96 -b2 0.99965 -e 80 -p 80 -o bci2000_model_comparison/eegD_LSTM

# different frequency bands
python main_preprocess_data_bci2000.py -f 0 -p /scratch/sem21f18/
python main_train_bci2000.py -m tuned -lr 0.0005 -b1 0.96 -b2 0.99965 -e 150 -p 150 -cf 45 -o bci2000_frequency_band_comparison/0bands
python main_preprocess_data_bci2000.py -f 2 -p /scratch/sem21f18/
python main_train_bci2000.py -m tuned -lr 0.0005 -b1 0.96 -b2 0.99965 -e 150 -p 150 -cf 45 -o bci2000_frequency_band_comparison/2bands

# different number of classes
python main_preprocess_data_bci2000.py -f 1 -p /scratch/sem21f18/
python main_train_bci2000.py -m tuned -lr 0.0005 -b1 0.96 -b2 0.99965 -e 150 -p 150 -c 2 -cf 45 -o bci2000_class_number_comparison/2class
python main_train_bci2000.py -m tuned -lr 0.0005 -b1 0.96 -b2 0.99965 -e 150 -p 150 -c 3 -cf 45 -o bci2000_class_number_comparison/3class

# different image size
python main_preprocess_data_bci2000.py -g 64 -p /scratch/sem21f18/
python main_train_bci2000.py -m tuned -lr 0.0005 -b1 0.96 -b2 0.99965 -e 150 -p 150 -cf 45 -o bci2000_image_size_comparison/64pixel
