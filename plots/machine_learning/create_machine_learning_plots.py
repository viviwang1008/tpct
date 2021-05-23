import json
import matplotlib.pyplot as plt
import numpy as np


def create_0419_plots():
    files = ['../logs/fit/20210419-215934_eegA/history.txt',
             '../logs/fit/20210419-220045_eegB/history.txt',
             '../logs/fit/20210419-220156_eegC/history.txt',
             '../logs/fit/20210419-220308_eegD/history.txt']

    fig, ax = plt.subplots(2, 2)
    fig.set_size_inches(12, 12)

    for file in files:
        data = json.load(open(file))
        ax[0, 0].plot(data["accuracy"])
        ax[1, 0].plot(data["val_accuracy"])
        ax[0, 1].plot(data["loss"])
        ax[1, 1].plot(data["val_loss"])
    ax[0, 0].set_title("Accuracy")
    ax[1, 0].set_title("Validation Accuracy")
    ax[0, 1].set_title("Loss")
    ax[1, 1].set_title("Validation Loss")

    ax[0, 0].set_ylim([0, 1.05])
    ax[1, 0].set_ylim([0, 1.05])

    ax[1, 0].set_xlabel("Epoch")
    ax[1, 1].set_xlabel("Epoch")
    ax[1, 1].legend(['A', 'B', 'C', 'D'])

    ax[0, 0].grid(True)
    ax[1, 0].grid(True)
    ax[0, 1].grid(True)
    ax[1, 1].grid(True)

    plt.savefig("04_19_accuracy_loss.png")


def create_0419_0425_plots():
    files0419 = ['../logs/fit/20210419-215934_eegA/history.txt',
                 '../logs/fit/20210419-220045_eegB/history.txt',
                 '../logs/fit/20210419-220156_eegC/history.txt',
                 '../logs/fit/20210419-220308_eegD/history.txt']

    files0425 = ['../logs/fit/20210425-105709_eegA/history.txt',
                 '../logs/fit/20210425-105748_eegB/history.txt',
                 '../logs/fit/20210425-105945_eegC/history.txt',
                 '../logs/fit/20210425-110128_eegD/history.txt']

    fig, ax = plt.subplots(2, 2)
    fig.set_size_inches(12, 12)

    for idx, f in enumerate(files0419):
        data = json.load(open(f))
        ax[idx % 2, idx // 2].plot(data["accuracy"], '-r')
        ax[idx % 2, idx // 2].plot(data["val_accuracy"], '-g')
        ax[idx % 2, idx // 2].plot(data["loss"], '-b')
        ax[idx % 2, idx // 2].plot(data["val_loss"], '-y')

    for idx, f in enumerate(files0425):
        data = json.load(open(f))
        ax[idx % 2, idx // 2].plot(data["accuracy"], '--r')
        ax[idx % 2, idx // 2].plot(data["val_accuracy"], '--g')
        ax[idx % 2, idx // 2].plot(data["loss"], '--b')
        ax[idx % 2, idx // 2].plot(data["val_loss"], '--y')

    titles = ['A', 'B', 'C', 'D']
    for idx in range(0, 4):
        ax[idx % 2, idx // 2].set_title(titles[idx])
        ax[idx % 2, idx // 2].legend(["acc", "val_acc", "loss", "val_loss",
                                      "Dropout acc", "Dropout val_acc", "Dropout loss", "Dropout val_loss"])
        ax[idx % 2, idx // 2].set_ylim([0, 5])
        ax[idx % 2, idx // 2].grid(True)
        ax[idx % 2, idx // 2].set_xlabel("epoch")

    plt.savefig("04_19_25_comparison.png")


def create_0427_0425_plots():
    files0425 = ['../logs/fit/20210425-105709_eegA/history.txt',
                 '../logs/fit/20210425-105748_eegB/history.txt',
                 '../logs/fit/20210425-105945_eegC/history.txt',
                 '../logs/fit/20210425-110128_eegD/history.txt']

    files0427_bs16 = ['../logs/fit/20210427-221730_eegA/history.txt',
                      '../logs/fit/20210427-222012_eegB/history.txt',
                      '../logs/fit/20210427-222226_eegC/history.txt',
                      '../logs/fit/20210427-222329_eegD/history.txt']

    files0427_bs4 = ['../logs/fit/20210427-221826_eegA/history.txt',
                     '../logs/fit/20210427-222114_eegB/history.txt',
                     '../logs/fit/20210427-222323_eegC/history.txt',
                     '../logs/fit/20210427-222525_eegD/history.txt']

    fig, ax = plt.subplots(2, 2)
    fig.set_size_inches(12, 12)

    for idx, f in enumerate(files0425):
        data = json.load(open(f))
        ax[idx % 2, idx // 2].plot(data["val_accuracy"], '--')

    for idx, f in enumerate(files0427_bs16):
        data = json.load(open(f))
        ax[idx % 2, idx // 2].plot(data["val_accuracy"], '-')

    for idx, f in enumerate(files0427_bs4):
        data = json.load(open(f))
        ax[idx % 2, idx // 2].plot(data["val_accuracy"])

    titles = ['A val_acc', 'B val_acc', 'C val_acc', 'D val_acc']
    for idx in range(0, 4):
        ax[idx % 2, idx // 2].set_title(titles[idx])
        ax[idx % 2, idx // 2].legend(["bs64, lr0.001", "bs16, lr0.0001", "bs4, lr0.0001"])
        ax[idx % 2, idx // 2].set_ylim([0, 1])
        ax[idx % 2, idx // 2].grid(True)
        ax[idx % 2, idx // 2].set_xlabel("epoch")
    plt.savefig("04_25_27_learning_rate.png")


def create_loocv_plots():
    files = ['../logs/fit/20210503-211806_eegA/history_1.txt',
             '../logs/fit/20210503-211806_eegA/history_2.txt',
             '../logs/fit/20210503-211806_eegA/history_3.txt',
             '../logs/fit/20210503-211806_eegA/history_4.txt',
             '../logs/fit/20210503-211806_eegA/history_5.txt',
             '../logs/fit/20210503-211806_eegA/history_6.txt',
             '../logs/fit/20210503-211806_eegA/history_7.txt',
             '../logs/fit/20210503-211806_eegA/history_8.txt',
             '../logs/fit/20210503-211806_eegA/history_9.txt']
    fig, ax = plt.subplots(2, 2)
    fig.set_size_inches(12, 12)

    av_accuracy = np.zeros(500)
    av_val_accuracy = np.zeros(500)
    av_loss = np.zeros(500)
    av_val_loss = np.zeros(500)
    min_len = 500
    for f in files:
        data = json.load(open(f))
        av_accuracy[0:len(data["accuracy"])] += data["accuracy"]
        av_val_accuracy[0:len(data["val_accuracy"])] += data["val_accuracy"]
        av_loss[0:len(data["loss"])] += data["loss"]
        av_val_loss[0:len(data["val_loss"])] += data["val_loss"]
        min_len = np.min((min_len, len(data["accuracy"])))
    av_accuracy /= 9
    av_val_accuracy /= 9
    av_loss /= 9
    av_val_loss /= 9

    quantities = ["accuracy", "val_accuracy", "loss", "val_loss"]
    for plot_idx in range(4):
        for f in files:
            data = json.load(open(f))
            ax[plot_idx % 2, plot_idx // 2].plot(data[quantities[plot_idx]])
            ax[plot_idx % 2, plot_idx // 2].set_title(quantities[plot_idx])
            ax[plot_idx % 2, plot_idx // 2].legend([str(x) for x in range(1, 10)])
            ax[plot_idx % 2, plot_idx // 2].grid(True)
    ax[0, 0].plot(av_accuracy[0:min_len], 'k', linewidth=5)
    ax[1, 0].plot(av_val_accuracy[0:min_len], 'k', linewidth=5)
    ax[0, 1].plot(av_loss[0:min_len], 'k', linewidth=5)
    ax[1, 1].plot(av_val_loss[0:min_len], 'k', linewidth=5)
    ax[1, 0].set_xlabel("epoch")
    ax[1, 1].set_xlabel("epoch")
    ax[0, 0].set_ylim([0, 1])
    ax[1, 0].set_ylim([0, 1])
    ax[0, 1].set_ylim([0, 2])
    ax[1, 1].set_ylim([0, 12])
    plt.savefig("05_06_loocv.png")


def create_single_subject_plots():
    files = ['../logs/fit/20210505-134915_eegA/history_1.txt',
             '../logs/fit/20210505-134915_eegA/history_2.txt',
             '../logs/fit/20210505-134915_eegA/history_3.txt',
             '../logs/fit/20210505-134915_eegA/history_4.txt',
             '../logs/fit/20210505-134915_eegA/history_5.txt',
             '../logs/fit/20210505-134915_eegA/history_6.txt',
             '../logs/fit/20210505-134915_eegA/history_7.txt',
             '../logs/fit/20210505-134915_eegA/history_8.txt',
             '../logs/fit/20210505-134915_eegA/history_9.txt']
    fig, ax = plt.subplots(2, 2)
    fig.set_size_inches(12, 12)

    av_accuracy = np.zeros(500)
    av_val_accuracy = np.zeros(500)
    av_loss = np.zeros(500)
    av_val_loss = np.zeros(500)
    min_len = 500
    for f in files:
        data = json.load(open(f))
        av_accuracy[0:len(data["accuracy"])] += data["accuracy"]
        av_val_accuracy[0:len(data["val_accuracy"])] += data["val_accuracy"]
        av_loss[0:len(data["loss"])] += data["loss"]
        av_val_loss[0:len(data["val_loss"])] += data["val_loss"]
        min_len = np.min((min_len, len(data["accuracy"])))
    av_accuracy /= 9
    av_val_accuracy /= 9
    av_loss /= 9
    av_val_loss /= 9

    quantities = ["accuracy", "val_accuracy", "loss", "val_loss"]
    for plot_idx in range(4):
        for f in files:
            data = json.load(open(f))
            ax[plot_idx % 2, plot_idx // 2].plot(data[quantities[plot_idx]])
            ax[plot_idx % 2, plot_idx // 2].set_title(quantities[plot_idx])
            ax[plot_idx % 2, plot_idx // 2].legend([str(x) for x in range(1, 10)])
            ax[plot_idx % 2, plot_idx // 2].grid(True)
    ax[0, 0].plot(av_accuracy[0:min_len], 'k', linewidth=5)
    ax[1, 0].plot(av_val_accuracy[0:min_len], 'k', linewidth=5)
    ax[0, 1].plot(av_loss[0:min_len], 'k', linewidth=5)
    ax[1, 1].plot(av_val_loss[0:min_len], 'k', linewidth=5)
    ax[1, 0].set_xlabel("epoch")
    ax[1, 1].set_xlabel("epoch")
    ax[0, 0].set_ylim([0, 1])
    ax[1, 0].set_ylim([0, 1])
    ax[0, 1].set_ylim([0, 2])
    ax[1, 1].set_ylim([0, 12])
    plt.savefig("05_06_single_subject.png")


def create_reference_plot():
    file = '../logs/fit/20210505-180315_eegA/history.txt'
    fig, ax = plt.subplots(2, 2)
    fig.set_size_inches(12, 12)

    data = json.load(open(file))
    av_accuracy = data["accuracy"]
    av_val_accuracy = data["val_accuracy"]
    av_loss = data["loss"]
    av_val_loss = data["val_loss"]

    quantities = ["accuracy", "val_accuracy", "loss", "val_loss"]
    for plot_idx in range(4):
        ax[plot_idx % 2, plot_idx // 2].set_title(quantities[plot_idx])
        ax[plot_idx % 2, plot_idx // 2].grid(True)
    ax[0, 0].plot(av_accuracy, 'k', linewidth=5)
    ax[1, 0].plot(av_val_accuracy, 'k', linewidth=5)
    ax[0, 1].plot(av_loss, 'k', linewidth=5)
    ax[1, 1].plot(av_val_loss, 'k', linewidth=5)
    ax[1, 0].set_xlabel("epoch")
    ax[1, 1].set_xlabel("epoch")
    ax[0, 0].set_ylim([0, 1])
    ax[1, 0].set_ylim([0, 1])
    ax[0, 1].set_ylim([0, 2])
    ax[1, 1].set_ylim([0, 12])
    plt.savefig("05_06_reference.png")


def create_comparison_plot_old_new_images():
    files = ['../logs/fit/20210505-180315_eegA/history.txt',
             '../logs/fit/20210506-094714_eegA/history.txt']
    fig, ax = plt.subplots(2, 2)
    fig.set_size_inches(12, 12)

    quantities = ["accuracy", "val_accuracy", "loss", "val_loss"]
    for plot_idx in range(4):
        for f in files:
            data = json.load(open(f))
            ax[plot_idx % 2, plot_idx // 2].plot(data[quantities[plot_idx]])
            ax[plot_idx % 2, plot_idx // 2].set_title(quantities[plot_idx])
            ax[plot_idx % 2, plot_idx // 2].legend(["old images", "new images (Fadel)"])
            ax[plot_idx % 2, plot_idx // 2].grid(True)
    ax[1, 0].set_xlabel("epoch")
    ax[1, 1].set_xlabel("epoch")
    ax[0, 0].set_ylim([0, 1])
    ax[1, 0].set_ylim([0, 1])
    ax[0, 1].set_ylim([0, 2])
    ax[1, 1].set_ylim([0, 12])
    plt.savefig("05_06_old_new_image_comparison.png")


if __name__ == "__main__":
    # create_0419_plots()
    # create_0419_0425_plots()
    # create_0427_0425_plots()
    # create_loocv_plots()
    # create_single_subject_plots()
    # create_reference_plot()
    # create_comparison_plot_old_new_images()
    pass
