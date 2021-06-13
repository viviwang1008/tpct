import os
from datetime import datetime

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # 0, 1, 2, or 3. check nvidia-smi
import argparse
import json
from models.model_factory import ModelFactory
from models.load_and_scale_data import scale_data, load_preprocessed_bci2000_data
from sklearn.model_selection import KFold


def write_history(history: dict, log_dir: str) -> None:
    history.pop('lr')  # don't store learning rate
    with open(f"{log_dir}/history.txt", 'w') as file:
        file.write(json.dumps(history, indent=4))


def main():
    parser = argparse.ArgumentParser(description="Train and run k-fold cross-validation on physionet BCI 2000 dataset")
    parser.add_argument("-c", "--num_classes", type=int, default=4, choices=[2, 3, 4])
    parser.add_argument("-m", "--model_name", type=str, help="Name of the model used", default="eegA",
                        choices=["eegA", "eegB", "eegC", "eegD", "eegA_LSTM", "eegD_LSTM"])
    parser.add_argument("-cf", "--num_conv_filters", type=int, default=32)
    parser.add_argument('--stride', dest='stride', help="Whether stride is used in the last Conv2D of first block",
                        action='store_true')
    parser.add_argument('--no-stride', dest='stride', action='store_false')
    parser.set_defaults(stride=True)
    parser.add_argument("-dr", "--dropout_rate", type=float, default=0.5)
    parser.add_argument("-bs", "--batch_size", type=int, default=16)
    parser.add_argument("-e", "--epochs", type=int, default=10)
    parser.add_argument("-p", "--patience", help="Parameter for EarlyStopping callback", type=int, default=5)
    parser.add_argument("-kf", "--k_fold", type=int, default=5)
    parser.add_argument("-o", "--output_name", type=str, help="logs will be put in ./logs/fit/output_name. If none is"
                                                              "provided, time at run start is chosen", default=None)

    args = parser.parse_args()

    # input validation
    try:
        num_windows = json.load(open("./data/args_bci2000_preprocess.txt", 'r'))['num_windows']
    except FileNotFoundError:
        raise FileNotFoundError(
            "Preprocessed data arguments not found. Run main_preprocess_data_bci2000.py and try again.")
    if num_windows == 1 and 'LSTM' in args.model_name:
        raise ValueError("LSTM can only be chosen for data preprocessed with -w > 1")
    if num_windows > 1 and 'LSTM' not in args.model_name:
        raise ValueError("Only LSTM models can be chosen for data preprocessed with -w > 1")

    if args.output_name is None:
        args.output_name = datetime.now().strftime('%Y%m%d-%H%M%S')

    model_factory = ModelFactory(dataset="BCI2000",
                                 output_name=args.output_name,
                                 model_name=args.model_name,
                                 num_classes=args.num_classes,
                                 num_conv_filters=args.num_conv_filters,
                                 dropout_rate=args.dropout_rate,
                                 use_stride=args.stride)

    X, y = load_preprocessed_bci2000_data(num_classes=args.num_classes)

    kf = KFold(n_splits=args.k_fold, shuffle=True, random_state=42)
    for idx, [train, test] in enumerate(kf.split(X, y)):
        X_train = X[train]
        X_test = X[test]
        y_train = y[train]
        y_test = y[test]

        X_train, scaler = scale_data(X_train)
        X_test, _ = scale_data(X_test, scaler)

        model = model_factory.get_model()

        history = model.fit(x=X_train,
                            y=y_train,
                            batch_size=args.batch_size,
                            epochs=args.epochs,
                            validation_data=(X_test, y_test),
                            callbacks=model_factory.get_callbacks(patience=args.patience,
                                                                  log_dir_suffix=f"{idx + 1}"),
                            shuffle=True)

        write_history(history.history, log_dir=model_factory.get_log_dir())

        with open(f"{model_factory.get_log_dir()}/model_summary.txt", 'w') as file:
            model.summary(print_fn=lambda x: file.write(x + '\n'))

        # write parameters used for training
        with open(f"{model_factory.get_log_dir()}/input_args.txt", 'w') as file:
            file.write(json.dumps(args.__dict__, indent=4))


if __name__ == '__main__':
    main()
