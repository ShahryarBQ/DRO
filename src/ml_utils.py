from copy import deepcopy
from random import random

import torch
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

from LeNet5_v2 import LeNet5_v2


def get_aux_info(dataset_name, model_name):
    n_classes, n_channels, input_size = -1, -1, -1

    if dataset_name in ["mnist", "fmnist", "cifar10"]:
        n_classes = 10
    elif dataset_name == "cifar100":
        n_classes = 100
    elif dataset_name == "imagenet":
        n_classes = 1000

    if dataset_name in ["mnist", "fmnist"]:
        n_channels = 1
    elif dataset_name in ["cifar10", "cifar100", "imagenet"]:
        n_channels = 3

    if model_name == "lenet5_v2":
        input_size = 32
    elif model_name in ["alexnet", "resnet50"]:
        input_size = 224

    return n_classes, n_channels, input_size


def get_dataset(name, input_size):
    normalize = transforms.ToTensor()
    if name == "mnist":
        normalize = transforms.Normalize((0.1307,), (0.3081,))
    elif name == "fmnist":
        normalize = transforms.Normalize((0.5,), (0.5,))
    elif name == "cifar10":
        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    elif name == "cifar100":
        normalize = transforms.Normalize((0.5074, 0.4867, 0.4411), (0.2011, 0.1987, 0.2025))

    transform_train = transforms.Compose([
        transforms.Resize(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])
    transform_test = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        normalize
    ])

    if name == "mnist":
        return mnist(transform_train, transform_test)
    elif name == "fmnist":
        return fmnist(transform_train, transform_test)
    elif name == "cifar10":
        return cifar10(transform_train, transform_test)
    elif name == "cifar100":
        return cifar100(transform_train, transform_test)
    elif name == "imagenet":
        return imagenet(input_size)


def get_model(name, n_classes, n_channels, pretrained=False):
    if name == "lenet5_v2":
        return LeNet5_v2(n_classes, n_channels)
    elif name == "alexnet":
        return models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1) if pretrained else models.alexnet()
    elif name == "resnet50":
        return models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1) if pretrained else models.resnet50()


def mnist(transform_train, transform_test):
    train = datasets.MNIST('../data', train=True, download=True, transform=transform_train)
    test = datasets.MNIST('../data', train=False, download=True, transform=transform_test)
    return train, test


def fmnist(transform_train, transform_test):
    train = datasets.FashionMNIST(root='../data', train=True, download=True, transform=transform_train)
    test = datasets.FashionMNIST(root='../data', train=False, download=True, transform=transform_test)
    return train, test


def cifar10(transform_train, transform_test):
    train = datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_train)
    test = datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)
    return train, test


def cifar100(transform_train, transform_test):
    train = datasets.CIFAR100(root='../data', train=True, download=True, transform=transform_train)
    test = datasets.CIFAR100(root='../data', train=False, download=True, transform=transform_test)
    return train, test


def imagenet(input_size):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    transform_val = transforms.Compose([
        transforms.Resize(input_size * 8 / 7),  # 256 = 224 * 8/8
        transforms.CenterCrop(input_size),  # 224
        transforms.ToTensor(),
        normalize,
    ])
    train = datasets.ImageNet('../data/imagenet_root', split='train', transform=transform_train)
    test = datasets.ImageNet('../data/imagenet_root', split='val', transform=transform_val)
    return train, test


def calculate_error(dataset, model, device):
    model.eval()
    loader = DataLoader(dataset, batch_size=1000, shuffle=False, num_workers=2)

    error = torch.tensor(0.0)
    for X_batch, y_batch in iter(loader):
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        pred = model(X_batch)
        yhat = pred.argmax(dim=1)
        error += torch.sum(torch.eq(yhat, y_batch))

    return 1 - error.item()/len(dataset)


def per_class_test_errors(test, model, n_classes, device):
    test_by_class = [[data for data in test if data[1] == j] for j in range(n_classes)]
    test_loader = DataLoader(test, batch_size=1, shuffle=False, num_workers=1)
    model.eval()

    test_errors = torch.tensor([0.0 for _ in range(n_classes)])
    for X, y in iter(test_loader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        yhat = pred.argmax()  # dim=1
        test_errors[y] += yhat != y

    test_errors = [test_errors[i] / len(test_by_class[i]) if len(test_by_class[i]) != 0 else 1
                   for i in range(n_classes)]
    print(test_errors)
    return test_errors


def train_model(model, train, test, criterion, learning_rate, batch_size, max_epochs, device, freeze_ratio=0):
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=2)
    model = freeze_model_parameters(model, freeze_ratio)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # logs
    log_iters = []
    log_epochs = [0]
    log_train_loss = []
    log_train_error = [calculate_error(train, model, device)]
    log_test_error = [calculate_error(test, model, device)]

    for epoch in range(1, max_epochs + 1):
        print(f"epoch {epoch}")
        for X_batch, y_batch in iter(train_loader):
            iteration = 0 if not log_iters else log_iters[-1] + 1

            # Optimize
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()

            # Per iteration logs
            log_iters.append(iteration)
            log_train_loss.append(loss.item())

        # Per epoch logs
        train_error = calculate_error(train, model, device)
        test_error = calculate_error(test, model, device)
        log_epochs.append(epoch)
        log_train_error.append(train_error)
        log_test_error.append(test_error)

    log = {"log_iters": log_iters,
           "log_epochs": log_epochs,
           "log_train_loss": log_train_loss,
           "log_train_error": log_train_error,
           "log_test_error": log_test_error}
    return log


def freeze_model_parameters(model, freeze_ratio):
    n_layers = len(list(model.parameters()))
    freeze = int(freeze_ratio * n_layers)
    layer = 1
    for child in model.children():
        for param in child.parameters():
            if layer <= freeze:
                param.requires_grad = False
                layer += 1
    return model


def reset_model(initial_model, device):
    return deepcopy(initial_model).to(device)


def model_dim(model):
    dim = 0
    for param in model.parameters():
        dim += len(param.flatten())
    return dim


def get_classes(dataset, labels):
    dataset_by_classes = [data for data in dataset if data[1] in labels]
    return dataset_by_classes


def get_classes_with_prob(dataset, probs):
    dataset_by_probs = [data for data in dataset if random() <= probs[data[1]]]
    return dataset_by_probs
