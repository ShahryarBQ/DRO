import torch

import AdvShift
import ml_utils
import plot_utils


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on {device}...")

    train, test = ml_utils.get_dataset(dataset_name, input_size)
    test = ml_utils.get_classes_with_prob(test, [(i+1)/n_classes for i in range(n_classes)])
    train = ml_utils.get_classes_with_prob(train, [(n_classes-i)/n_classes for i in range(n_classes)])
    initial_model = ml_utils.get_model(model_name, n_classes, n_channels, pretrained=True)
    print(f"train size = {len(train)}, test size = {len(test)}, model dim = {ml_utils.model_dim(initial_model)}")

    # Hyperparameters
    criterion = torch.nn.CrossEntropyLoss()
    learning_rate = 0.01
    batch_size = 256
    max_epochs = 10  # Section 4.1
    adv_radius = 0.1  # Section 4.2
    gamma_c = 1 / (2 * learning_rate)  # Appendix B
    ma_param = 0.999  # Appendix B
    clip_max = 2  # Section 3.5
    grad_stabilizer = 0.001  # Section 3.5
    freeze_ratio = 0

    # Standard training as a baseline
    model = ml_utils.reset_model(initial_model, device)
    log1 = ml_utils.train_model(model, train, test, criterion, learning_rate,
                                batch_size, max_epochs, device, freeze_ratio)
    per_clss_tst_errs = ml_utils.per_class_test_errors(test, model, n_classes, device)
    plot_utils.save_log(log1, filename1)
    plot_utils.plot_figure1(per_clss_tst_errs, dataset_name, model_name, "Baseline")

    # Training via AdvShift
    model = ml_utils.reset_model(initial_model, device)
    log2 = AdvShift.adv_shift(train,
                              test,
                              model,
                              criterion,
                              n_classes,
                              learning_rate,
                              batch_size,
                              max_epochs,
                              adv_radius,
                              gamma_c,
                              ma_param,
                              clip_max,
                              grad_stabilizer,
                              device)
    plot_utils.save_log(log2, filename2)
    plot_utils.plot_figure3_figure4(log1, log2, dataset_name, model_name, n_classes)

    per_clss_tst_errs = ml_utils.per_class_test_errors(test, model, n_classes, device)
    plot_utils.plot_figure1(per_clss_tst_errs, dataset_name, model_name, "AdvShift")


def plot():
    log1 = plot_utils.extract_log(filename1)
    log2 = plot_utils.extract_log(filename2)
    plot_utils.plot_figure3_figure4(log1, log2, dataset_name, model_name, n_classes)


if __name__ == "__main__":
    dataset_name = "fmnist"
    model_name = "lenet5_v2"
    n_classes, n_channels, input_size = ml_utils.get_aux_info(dataset_name, model_name)
    filename1 = f"log_{dataset_name}_{model_name}_baseline.xlsx"
    filename2 = f"log_{dataset_name}_{model_name}_advshift.xlsx"

    main()
    # plot()
