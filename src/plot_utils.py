import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os


def plot_figure1(test_errors_by_class, dataset_name, model_name, method):
    sns.set()
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.hist(test_errors_by_class, range=(0, 1), bins=20)
    ax.set_xlabel("Per-class validation error")
    ax.set_ylabel("Frequency of class")
    ax.set_title(f"{model_name} on {dataset_name}")
    plt.savefig(f"{model_name}_{dataset_name}_figure1_{method}.pdf")


def plot_figure3_figure4(log1, log2, dataset_name, model_name, n_classes):
    log1_iters = log1["log_iters"]
    log1_epochs = log1["log_epochs"]
    log1_train_loss = log1["log_train_loss"]
    log1_train_error = log1["log_train_error"]
    log1_test_error = log1["log_test_error"]

    log2_iters = log2["log_iters"]
    log2_epochs = log2["log_epochs"]
    log2_train_loss = log2["log_train_loss"]
    log2_train_error = log2["log_train_error"]
    log2_test_error = log2["log_test_error"]
    log2_pi = log2["log_pi"]

    sns.set()
    fig, axes = plt.subplots(nrows=2, ncols=3)
    axes[0, 0].plot(log2_iters, log2_train_loss)
    axes[0, 1].plot(log2_epochs, log2_train_error)
    axes[0, 2].plot(log2_epochs, log2_test_error)

    rng_tup = (0.5/n_classes, 2/n_classes)
    axes[1, 0].hist(log2_pi[0], range=rng_tup, bins=20)
    axes[1, 1].hist(log2_pi[len(log2_pi) // 2], range=rng_tup, bins=20)
    axes[1, 2].hist(log2_pi[len(log2_pi) - 1], range=rng_tup, bins=20)

    axes[0, 0].plot(log1_iters, log1_train_loss)
    axes[0, 1].plot(log1_epochs, log1_train_error)
    axes[0, 2].plot(log1_epochs, log1_test_error)

    rng_lst = [0.5/n_classes, 1/n_classes, 1.5/n_classes, 2/n_classes]
    axes[1, 0].set_xticks(rng_lst, labels=rng_lst)
    axes[1, 1].set_xticks(rng_lst, labels=rng_lst)
    axes[1, 2].set_xticks(rng_lst, labels=rng_lst)

    axes[0, 0].set_xlabel("Training iterations")
    axes[0, 1].set_xlabel("Training epochs")
    axes[0, 2].set_xlabel("Training epochs")
    axes[1, 0].set_xlabel("Adversarial weights")
    axes[1, 1].set_xlabel("Adversarial weights")
    axes[1, 2].set_xlabel("Adversarial weights")

    axes[0, 0].set_ylabel("Train loss")
    axes[0, 1].set_ylabel("Train error")
    axes[0, 2].set_ylabel("Test error")
    axes[1, 0].set_ylabel("Frequency")
    axes[1, 1].set_ylabel("Frequency")
    axes[1, 2].set_ylabel("Frequency")

    axes[0, 0].set_title(f"Loss of {model_name} on {dataset_name}")
    axes[0, 1].set_title("Train")
    axes[0, 2].set_title("Test")
    axes[1, 0].set_title("Epoch 0")
    axes[1, 1].set_title(f"Epoch {len(log2_pi) // 2}")
    axes[1, 2].set_title(f"Epoch {len(log2_pi)}")

    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.legend(labels=["AdvShift", "Baseline"], loc='lower center', bbox_to_anchor=(0.5, 0.9), ncol=2)
    fig.savefig(f"{model_name}_{dataset_name}_figure3_figure4.pdf")


def save_log(log, filename):
    log_per_iter, log_per_epoch = {}, {}
    log_per_iter["log_iters"] = log["log_iters"]
    log_per_iter["log_train_loss"] = log["log_train_loss"]
    log_per_epoch["log_epochs"] = log["log_epochs"]
    log_per_epoch["log_train_error"] = log["log_train_error"]
    log_per_epoch["log_test_error"] = log["log_test_error"]

    if "log_pi" in log:
        log_per_epoch["log_pi"] = log["log_pi"]

    dirpath = os.path.join(os.getcwd(), "results")
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

    filepath = os.path.join(dirpath, filename)
    with pd.ExcelWriter(filepath) as writer:
        df_per_iter, df_per_epoch = pd.DataFrame(log_per_iter), pd.DataFrame(log_per_epoch)
        df_per_iter.to_excel(writer, sheet_name="per_iter")
        df_per_epoch.to_excel(writer, sheet_name="per_epoch")


def extract_log(filename):
    filepath = os.path.join(os.getcwd(), "results", filename)
    df = pd.read_excel(filepath, index_col=[0], sheet_name=None)

    df_per_iter = df["per_iter"]
    df_per_epoch = df["per_epoch"]
    log = {"log_iters": df_per_iter["log_iters"],
           "log_train_loss": df_per_iter["log_train_loss"],
           "log_epochs": df_per_epoch["log_epochs"],
           "log_train_error": df_per_epoch["log_train_error"],
           "log_test_error": df_per_epoch["log_test_error"]}

    if "log_pi" in df_per_epoch:
        log["log_pi"] = [[float(x) for x in pi[1:-1].split()] for pi in df_per_epoch["log_pi"]]
    return log
