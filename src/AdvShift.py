from copy import deepcopy

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import ml_utils


def adv_shift(train,
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
              device):
    """
    :param train:               (x, y) from p_tr
    :param test:                (x, y) from p_te
    :param model:               NNOpt
    :param criterion:           l(x, y, theta)
    :param n_classes:           L
    :param learning_rate:       lambda
    :param batch_size:          b
    :param max_epochs:          T / (n / batch_size)
    :param adv_radius:          r (KL divergence threshold)
    :param gamma_c:             gamma_c
    :param ma_param:            beta
    :param clip_max:            max clipping value
    :param grad_stabilizer:     epsilon
    :param device:              'cpu' or 'cuda'
    :return:
    """

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=2)
    train_by_class = [[data for data in train if data[1] == j] for j in range(n_classes)]
    p_emp = torch.tensor([len(c) / len(train) for c in train_by_class])

    # Step 1
    pi = torch.tensor([1 / n_classes for _ in range(n_classes)], requires_grad=True)

    # logs
    log_iters = []
    log_epochs = [0]
    log_train_loss = []
    log_train_error = [ml_utils.calculate_error(train, model, device)]
    log_test_error = [ml_utils.calculate_error(test, model, device)]
    log_pi = [pi.detach().numpy()]

    # Step 2
    n_layers = len(list(model.parameters()))
    for epoch in range(1, max_epochs + 1):

        # Per epoch log
        print(f"epoch {epoch}")

        # Step 3
        for X_batch, y_batch in iter(train_loader):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            X_loader = DataLoader(X_batch, batch_size=1, shuffle=False)
            y_loader = DataLoader(y_batch, batch_size=1, shuffle=False)

            # Per iteration log
            iteration = 0 if not log_iters else log_iters[-1] + 1
            if iteration % 20 == 0:
                print(f"    iter {iteration}")

            # Appendix B
            batch_by_class = [[data for data in zip(X_batch, y_batch) if data[1] == j]
                              for j in range(n_classes)]
            p_batch = torch.tensor([len(c) / len(train) for c in batch_by_class])
            p_emp = ma_param * p_emp + (1 - ma_param) * p_batch

            # Step 4
            g_theta, loss = [0 for _ in range(n_layers)], 0
            for X, y in zip(iter(X_loader), iter(y_loader)):
                sample_grad_theta, sample_loss = extract_gradient_theta(X, y, model, criterion)
                for i in range(n_layers):
                    g_theta[i] += (1 / batch_size) * (pi[y] / p_emp[y]) * sample_grad_theta[i]
                loss += sample_loss

            # Step 5
            # TODO: add ADAM option too
            with torch.no_grad():
                idx = 0
                for param in model.parameters():
                    param -= learning_rate * g_theta[idx]
                    idx += 1

            # Step 6
            # 1. KL(P || Q) = F.kl_div(Q, P) = F.kl_div(input, target)
            # 2. reduction='batchmean' is the mathematically correct method
            # 3. F.kl_div expects "input" to be in log-space
            KL = F.kl_div(torch.log(p_emp), pi, reduction='batchmean')
            alpha = 0 if adv_radius > KL else 2 * gamma_c * learning_rate

            # Step 7
            g_pi = torch.tensor([0.0 for _ in pi])
            for X, y in zip(iter(X_loader), iter(y_loader)):
                sample_grad_pi = extract_gradient_pi(X, y, model, criterion, pi)
                g_pi += (1 / batch_size) * (1 / p_emp[y]) * sample_grad_pi
            g_pi = torch.clip(g_pi, max=clip_max)  # Section 3.5

            # Step 8
            eta_pi = 1 / ((gamma_c + 1 / (2 * learning_rate)) * (1 + alpha))
            with torch.no_grad():
                pi = ((pi * (p_emp ** alpha)) ** (1 / (1 + alpha))) * torch.exp(eta_pi * g_pi)
                pi /= torch.linalg.vector_norm(pi, ord=1)
                pi += grad_stabilizer  # Section 3.5
                pi.requires_grad_(True)

            # Per iteration logs
            log_iters.append(iteration)
            log_train_loss.append(loss / batch_size)

        # Per epoch logs
        train_error = ml_utils.calculate_error(train, model, device)
        test_error = ml_utils.calculate_error(test, model, device)
        log_epochs.append(epoch)
        log_train_error.append(train_error)
        log_test_error.append(test_error)
        log_pi.append(pi.detach().numpy())

    log = {"log_iters": log_iters,
           "log_epochs": log_epochs,
           "log_train_loss": log_train_loss,
           "log_train_error": log_train_error,
           "log_test_error": log_test_error,
           "log_pi": log_pi}
    return log


def extract_gradient_theta(X, y, model, criterion):
    model.train()
    model.zero_grad()

    # forward pass
    pred = model(X)
    loss = criterion(pred, y)

    # backpropagation
    loss.backward(retain_graph=True)

    gradient = [param.grad for param in model.parameters()]
    return gradient, loss.item()


def extract_gradient_pi(X, y, model, criterion, pi):
    model_tmp = deepcopy(model)
    if pi.grad is not None:
        pi.grad.zero_()
    pred = model_tmp(X)
    loss = criterion(- pi[y] * pred, y)
    loss.backward()
    return pi.grad
