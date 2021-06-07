import os
from datetime import datetime

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # 0, 1, 2, or 3. check nvidia-smi
import globals as g
import argparse
import json
from machine_learning.model_factory import ModelFactory
from machine_learning.load_and_scale_data import scale_data, load_single_subject_bci2aiv_data


def write_history(history: dict, subject_num: int, log_dir: str) -> None:
    with open(f"{log_dir}/history_{subject_num}.txt", 'w') as file:
        file.write(json.dumps(history, indent=4))


def main():
    """
    Does the following:
    - For each subject:
        - Load preprocessed data from subject (preprocessed from 'A0XT.mat')
        - Train model on ALL data from 'A0XT.mat'
        - Evaluate model on test data originating from 'A0XE.mat'
    """

    parser = argparse.ArgumentParser(description="Train and run model for data set 2a of BCI Competition IV.")
    parser.add_argument("-m", "--model_name", type=str, default="tuned", choices=["tuned", "eegA", "eegB", "eegC",
                                                                                  "eegD", "tuned_LSTM", "eegD_LSTM"])
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.001)
    parser.add_argument("-cf", "--num_conv_filters", type=int, default=32)
    parser.add_argument("-b1", "--beta_1", type=float, default=0.9)
    parser.add_argument("-b2", "--beta_2", type=float, default=0.999)
    parser.add_argument("-dr", "--dropout_rate", type=float, default=0.5)
    parser.add_argument("-bs", "--batch_size", type=int, default=16)
    parser.add_argument("-e", "--epochs", type=int, default=100)
    parser.add_argument("-p", "--patience", type=int, default=10)
    parser.add_argument("-kf", "--k_fold", type=int, default=5)
    parser.add_argument("-o", "--output_name", type=str, default=None)

    args = parser.parse_args()

    # input validation
    try:
        num_windows = json.load(open("./data/args_bci2aiv_preprocess.txt", 'r'))['num_windows']
    except FileNotFoundError:
        raise FileNotFoundError("Preprocessed data arguments not found. Run main_preprocess_data_bci2aiv.py and try again.")
    if num_windows == 1 and 'LSTM' in args.model_name:
        raise ValueError("LSTM can only be chosen for data preprocessed with -w > 1")
    if num_windows > 1 and 'LSTM' not in args.model_name:
        raise ValueError("Only LSTM models can be chosen for data preprocessed with -w > 1")

    if args.output_name is None:
        args.output_name = datetime.now().strftime('%Y%m%d-%H%M%S')

    model_factory = ModelFactory(dataset="BCI2aIV",
                                 output_name=args.output_name,
                                 model_name=args.model_name,
                                 num_conv_filters=args.num_conv_filters,
                                 dropout_rate=args.dropout_rate,
                                 learning_rate=args.learning_rate,
                                 beta_1=args.beta_1,
                                 beta_2=args.beta_2)  # num_classes is always 4 for this dataset

    for subject_num in g.subject_num_range_bci2aiv:
        X_train, y_train = load_single_subject_bci2aiv_data(subject_num=subject_num, is_training=True)
        X_test, y_test = load_single_subject_bci2aiv_data(subject_num=subject_num, is_training=False)

        X_train, scaler = scale_data(X_train)
        X_test, _ = scale_data(X_test, scaler)

        model = model_factory.get_model()

        history = model.fit(x=X_train,
                            y=y_train,
                            batch_size=args.batch_size,
                            epochs=args.epochs,
                            validation_data=(X_test, y_test),
                            callbacks=model_factory.get_callbacks(patience=args.patience,
                                                                  log_dir_suffix=f"{subject_num}"),
                            shuffle=True)

        write_history(history.history, subject_num=subject_num, log_dir=model_factory.get_log_dir())

        with open(f"{model_factory.get_log_dir()}/model_summary.txt", 'w') as file:
            model.summary(print_fn=lambda x: file.write(x + '\n'))

        # write parameters used for training
        with open(f"{model_factory.get_log_dir()}/input_args.txt", 'w') as file:
            file.write(json.dumps(args.__dict__, indent=4))


if __name__ == '__main__':
    main()
