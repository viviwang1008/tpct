import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # 0, 1, 2, or 3. check nvidia-smi
import globals as g
import argparse
import json
from sklearn.model_selection import train_test_split
from itertools import product
from machine_learning.model_factory import ModelFactory
from machine_learning.load_and_scale_data import scale_data, load_single_subject_data


def write_history(history: dict, subject_num: int, log_dir: str) -> None:
    with open(f"{log_dir}/history_{subject_num}.txt", 'w') as file:
        file.write(json.dumps(history, indent=4))


def main():
    """
    Does the following:
    - For each subject:
        - Load preprocessed data from subject (preprocessed from 'A0XT.mat')
        - Split part of training data into validation data
        - Use train/validation data to find best hyper-parameters
        - Train model on ALL data from 'A0XT.mat'
        - Evaluate model on test data originating from 'A0XE.mat'
    """

    parser = argparse.ArgumentParser(description="Train and run VGG")
    # model choice
    parser.add_argument("-m", "--model_name", type=str, default="eegD", choices=["eegD"])
    # hyperparameters
    parser.add_argument("-bs", "--batch_size", type=int, nargs='+', default=[16])
    parser.add_argument("-lr", "--learning_rate", type=float, nargs='+', default=[0.001, 0.0001])
    parser.add_argument("-dr", "--dropout_rate", type=float, nargs='+', default=[0.5])
    # other parameters
    parser.add_argument("-ts", "--train_size", type=float, default=0.8)
    parser.add_argument("-e", "--epochs", type=int, default=1000)
    parser.add_argument("-et", "--epochs_for_tuning", type=int, default=50)

    args = parser.parse_args()

    model_factory = ModelFactory(model_name=args.model_name)

    for subject_num in g.subject_num_range:
        # if multiple hyperparameters are supplied, find the best one ...
        if len(args.batch_size) > 1 or len(args.learning_rate) > 1 or len(args.dropout_rate) > 1:
            print(f"Finding best hyperparameter values for subject {subject_num}")
            x, y = load_single_subject_data(subject_num=subject_num, is_training=True)

            x_train, x_val, y_train, y_val = train_test_split(x, y, train_size=args.train_size, random_state=42)
            x_train, scaler = scale_data(x_train)
            x_val, _ = scale_data(x_val, scaler)

            best_val_accuracy = 0
            best_params = {'learning_rate': 0,
                           'batch_size': 0,
                           'dropout_rate': 0}
            for lr, bs, dr in product(args.learning_rate, args.batch_size, args.dropout_rate):
                validation_model = model_factory.get_model(learning_rate=lr,
                                                           dropout_rate=dr)
                validation_history = validation_model.fit(x_train,
                                                          y_train,
                                                          batch_size=bs,
                                                          epochs=args.epochs_for_tuning,
                                                          validation_data=(x_val, y_val),
                                                          callbacks=model_factory.get_callbacks(patience=10,
                                                                                                log_dir_suffix=f"tuning_{subject_num}"))

                write_history(validation_history.history, subject_num=subject_num, log_dir=model_factory.get_log_dir())
                val_accuracy = validation_history.history['val_accuracy'][-1]
                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    best_params['learning_rate'] = lr
                    best_params['batch_size'] = bs
                    best_params['dropout_rate'] = dr
        # ... otherwise just set the best values as the ones provided
        else:
            best_params = {'learning_rate': args.learning_rate,
                           'batch_size': args.batch_size,
                           'dropout_rate': args.dropout_rate}

        print(f"Running model with train and test data for subject {subject_num}")
        x_train, y_train = load_single_subject_data(subject_num=subject_num, is_training=True)
        x_test, y_test = load_single_subject_data(subject_num=subject_num, is_training=False)

        x_train, scaler = scale_data(x_train)
        x_test, _ = scale_data(x_test, scaler)

        model = model_factory.get_model(learning_rate=best_params['learning_rate'],
                                        dropout_rate=best_params['dropout_rate'])
        history = model.fit(x=x_train,
                            y=y_train,
                            batch_size=best_params['batch_size'],
                            epochs=args.epochs,
                            callbacks=model_factory.get_callbacks(patience=100, log_dir_suffix=f"train_{subject_num}"),
                            shuffle=True)

        write_history(history.history, subject_num=subject_num, log_dir=model_factory.get_log_dir())

        results = model.evaluate(x=x_test, y=y_test)
        with open(f"{model_factory.get_log_dir()}/result.txt", 'w') as file:
            file.write(f"[loss, accuracy]={results}")

        # write parameters used for training
        with open(f"{model_factory.get_log_dir()}/hyper_params.txt", 'w') as file:
            file.write(json.dumps(best_params, indent=4))


if __name__ == '__main__':
    main()
