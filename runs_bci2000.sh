python main_preprocess_data_bci2000.py -w 1 -p /scratch/sem21f18/ --randomize_channel_locations
python main_train_bci2000.py -o bci2000_image_scrambled/ -m eegA -cf 32 --stride -e 80 -p 80

python main_preprocess_data_bci2000.py -w 1 -p /scratch/sem21f18/
python main_train_bci2000.py -o bci2000_model_comparison/eegA32S/ -m eegA -cf 32 --stride -e 80 -p 80
python main_train_bci2000.py -o bci2000_model_comparison/eegA32/ -m eegA -cf 32 --no-stride -e 80 -p 80
python main_train_bci2000.py -o bci2000_model_comparison/eegB32S/ -m eegB -cf 32 --stride -e 80 -p 80
python main_train_bci2000.py -o bci2000_model_comparison/eegB32/ -m eegB -cf 32 --no-stride -e 80 -p 80
python main_train_bci2000.py -o bci2000_model_comparison/eegC32S/ -m eegC -cf 32 --stride -e 80 -p 80
python main_train_bci2000.py -o bci2000_model_comparison/eegC32/ -m eegC -cf 32 --no-stride -e 80 -p 80
python main_train_bci2000.py -o bci2000_model_comparison/eegD32S/ -m eegD -cf 32 --stride -e 80 -p 80
python main_train_bci2000.py -o bci2000_model_comparison/eegD32/ -m eegD -cf 32 --no-stride -e 80 -p 80
python main_train_bci2000.py -o bci2000_model_comparison/eegA16S/ -m eegA -cf 16 --stride -e 80 -p 80
python main_train_bci2000.py -o bci2000_model_comparison/eegA16/ -m eegA -cf 16 --no-stride -e 80 -p 80
python main_train_bci2000.py -o bci2000_model_comparison/eegB16S/ -m eegB -cf 16 --stride -e 80 -p 80
python main_train_bci2000.py -o bci2000_model_comparison/eegB16/ -m eegB -cf 16 --no-stride -e 80 -p 80
python main_train_bci2000.py -o bci2000_model_comparison/eegC16S/ -m eegC -cf 16 --stride -e 80 -p 80
python main_train_bci2000.py -o bci2000_model_comparison/eegC16/ -m eegC -cf 16 --no-stride -e 80 -p 80
python main_train_bci2000.py -o bci2000_model_comparison/eegD16S/ -m eegD -cf 16 --stride -e 80 -p 80
python main_train_bci2000.py -o bci2000_model_comparison/eegD16/ -m eegD -cf 16 --no-stride -e 80 -p 80

python main_train_bci2000.py -o bci2000_dropout_comparison/eegA0.1/ -dr 0.1 -e 80 -p 80
python main_train_bci2000.py -o bci2000_dropout_comparison/eegA0.5/ -dr 0.5 -e 80 -p 80

python main_preprocess_data_bci2000.py -c 2 -p /scratch/sem21f18/
python main_train_bci2000.py -o bci2000_class_number_comparison/2class/ -c 2 -e 80 -p 80
python main_preprocess_data_bci2000.py -c 3 -p /scratch/sem21f18/
python main_train_bci2000.py -o bci2000_class_number_comparison/3class/ -c 3 -e 80 -p 80

python main_preprocess_data_bci2000.py -w 6 -p /scratch/sem21f18/
python main_train_bci2000.py -o bci2000_model_comparison/eegA_LSTM/ -m eegA_LSTM -e 50 -p 50
python main_train_bci2000.py -o bci2000_model_comparison/eegD_LSTM/ -m eegD_LSTM -e 50 -p 50

python main_preprocess_data_bci2000.py -w 1 -f 0 -p /scratch/sem21f18/
python main_train_bci2000.py -o bci2000_frequency_band_comparison/0band/ -m eegA -cf 32 --stride -e 80 -p 80
python main_preprocess_data_bci2000.py -w 1 -f 1 -p /scratch/sem21f18/
python main_train_bci2000.py -o bci2000_frequency_band_comparison/1band/ -m eegA -cf 32 --stride -e 80 -p 80

python main_preprocess_data_bci2000.py -w 1 -g 64 -p /scratch/sem21f18/
python main_train_bci2000.py -o bci2000_image_size_comparison/64pixel/ -m eegA -cf 32 --stride -e 80 -p 80
