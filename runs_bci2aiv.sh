python main_preprocess_data_bci2aiv.py -w 1 -g 32 -p /scratch/sem21f18/
python main_train_bci2aiv.py -o bci2aiv_model_comparison/eegA32S/ -m eegA -cf 32 -e 80 -p 80
python main_train_bci2aiv.py -o bci2aiv_model_comparison/eegB32S/ -m eegB -cf 32 -e 80 -p 80
python main_train_bci2aiv.py -o bci2aiv_model_comparison/eegC32S/ -m eegC -cf 32 -e 80 -p 80
python main_train_bci2aiv.py -o bci2aiv_model_comparison/eegD32S/ -m eegD -cf 32 -e 80 -p 80
python main_train_bci2aiv.py -o bci2aiv_model_comparison/eegA32/ -m eegA -cf 32 --no-stride -e 80 -p 80
python main_train_bci2aiv.py -o bci2aiv_model_comparison/eegB32/ -m eegB -cf 32 --no-stride -e 80 -p 80
python main_train_bci2aiv.py -o bci2aiv_model_comparison/eegC32/ -m eegC -cf 32 --no-stride -e 80 -p 80
python main_train_bci2aiv.py -o bci2aiv_model_comparison/eegD32/ -m eegD -cf 32 --no-stride -e 80 -p 80
python main_train_bci2aiv.py -o bci2aiv_model_comparison/eegA16S/ -m eegA -cf 16 -e 80 -p 80
python main_train_bci2aiv.py -o bci2aiv_model_comparison/eegB16S/ -m eegB -cf 16 -e 80 -p 80
python main_train_bci2aiv.py -o bci2aiv_model_comparison/eegC16S/ -m eegC -cf 16 -e 80 -p 80
python main_train_bci2aiv.py -o bci2aiv_model_comparison/eegD16S/ -m eegD -cf 16 -e 80 -p 80
python main_train_bci2aiv.py -o bci2aiv_model_comparison/eegA16/ -m eegA -cf 16 --no-stride -e 80 -p 80
python main_train_bci2aiv.py -o bci2aiv_model_comparison/eegB16/ -m eegB -cf 16 --no-stride -e 80 -p 80
python main_train_bci2aiv.py -o bci2aiv_model_comparison/eegC16/ -m eegC -cf 16 --no-stride -e 80 -p 80
python main_train_bci2aiv.py -o bci2aiv_model_comparison/eegD16/ -m eegD -cf 16 --no-stride -e 80 -p 80

python main_preprocess_data_bci2aiv.py -w 1 -g 32 -p /scratch/sem21f18/
python main_train_bci2aiv.py -o bci2aiv_dropout_comparison/eegA0.1/ -dr 0.1 -e 80 -p 80
python main_train_bci2aiv.py -o bci2aiv_dropout_comparison/eegA0.5/ -dr 0.5 -e 80 -p 80

python main_preprocess_data_bci2aiv.py -w 6 -g 32 -p /scratch/sem21f18/
python main_train_bci2aiv.py -o bci2aiv_model_comparison/eegA_LSTM/ -m eegA_LSTM -e 50 -p 50
python main_train_bci2aiv.py -o bci2aiv_model_comparison/eegD_LSTM/ -m eegD_LSTM -e 50 -p 50
