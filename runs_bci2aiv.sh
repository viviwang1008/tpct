# different models
python main_preprocess_data_bci2aiv.py -g 32 -w 1 -p /usr/scratch/sassauna1/sem21f18/tpct/
python main_train_bci2aiv.py -m tuned -lr 0.0005 -b1 0.96 -b2 0.99965 -e 150 -p 150 -cf 45 -o bci2aiv_model_comparison/tuned
python main_train_bci2aiv.py -m eegA -lr 0.0005 -b1 0.96 -b2 0.99965 -e 150 -p 150 -o bci2aiv_model_comparison/eegA
python main_train_bci2aiv.py -m eegB -lr 0.0005 -b1 0.96 -b2 0.99965 -e 150 -p 150 -o bci2aiv_model_comparison/eegB
python main_train_bci2aiv.py -m eegC -lr 0.0005 -b1 0.96 -b2 0.99965 -e 150 -p 150 -o bci2aiv_model_comparison/eegC
python main_train_bci2aiv.py -m eegD -lr 0.0005 -b1 0.96 -b2 0.99965 -e 150 -p 150 -o bci2aiv_model_comparison/eegD

python main_preprocess_data_bci2aiv.py -g 32 -w 6 -p /usr/scratch/sassauna1/sem21f18/tpct/
python main_train_bci2aiv.py -m tuned_LSTM -lr 0.0005 -b1 0.96 -b2 0.99965 -e 80 -p 80 -o bci2aiv_model_comparison/tuned_LSTM
python main_train_bci2aiv.py -m eegD_LSTM -lr 0.0005 -b1 0.96 -b2 0.99965 -e 80 -p 80 -o bci2aiv_model_comparison/eegD_LSTM

# different frequency bands
python main_preprocess_data_bci2aiv.py -g 32 -f 0 -p /usr/scratch/sassauna1/sem21f18/tpct/
python main_train_bci2aiv.py -m tuned -lr 0.0005 -b1 0.96 -b2 0.99965 -e 150 -p 150 -cf 45 -o bci2aiv_frequency_band_comparison/0bands
python main_preprocess_data_bci2aiv.py -g 32 -f 2 -p /usr/scratch/sassauna1/sem21f18/tpct/
python main_train_bci2aiv.py -m tuned -lr 0.0005 -b1 0.96 -b2 0.99965 -e 150 -p 150 -cf 45 -o bci2aiv_frequency_band_comparison/2bands

# different image size
python main_preprocess_data_bci2aiv.py -g 64 -p /usr/scratch/sassauna1/sem21f18/tpct/
python main_train_bci2aiv.py -m tuned -lr 0.0005 -b1 0.96 -b2 0.99965 -e 150 -p 150 -cf 45 -o bci2aiv_img_size_comparison/64pixel
