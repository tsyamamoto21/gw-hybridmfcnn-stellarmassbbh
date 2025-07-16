#!/usr/bin/env python
import os
import time
import shutil
import argparse
from datetime import datetime
from pytz import timezone
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchinfo import summary
from torchvision.transforms import RandomCrop
# from torcheval.metrics import MulticlassAccuracy
# import torchmetrics
from torchmetrics.classification import Accuracy
from omegaconf import OmegaConf
from hydra.utils import instantiate
# from dl4longcbc.dataset import MyDataset, load_dataset
from dl4longcbc.dataset import make_pathlist_and_labellist, LabelDataset
from dl4longcbc.net import instantiate_neuralnetwork
from dl4longcbc.utils import if_not_exist_makedir, plot_training_curve


# ----------------------------------------------------------------
# Training loop
# ----------------------------------------------------------------
def main(args):
    # Load parameters
    config = OmegaConf.load(args.config)

    # Make model directory
    if args.dirname is None:
        now_datetime = datetime.now(timezone('Asia/Tokyo')).strftime('%Y%m%d_%H%M%S')
        modeldirectory = f'./data/model/{config.train.experiment_name}/{now_datetime}'
    else:
        modeldirectory = f'./data/model/{config.train.experiment_name}/{args.dirname}'
    if_not_exist_makedir(modeldirectory)
    shutil.copy(args.config, modeldirectory)

    config = OmegaConf.load('./config/config_train.yaml')

    # Set device
    print(f'Is gpu available? {torch.cuda.is_available()}')
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Make dataloader
    img_height = config.dataset.img_height
    img_width = config.dataset.img_width
    img_channel = config.dataset.img_channel
    input_height = config.net.input_height
    input_width = config.net.input_width
    input_channel = config.net.input_channel
    transforms = nn.Sequential(
        RandomCrop((input_height, input_width))
    )

    inputpaths, labels = make_pathlist_and_labellist(f'{config.dataset.datadir}/train/', ['noise', 'cbc'], [0, 1])
    dataset_tr = LabelDataset(inputpaths, labels, transform=transforms)
    dataloader_tr = DataLoader(dataset_tr, batch_size=config.train.batchsize, shuffle=True, drop_last=True, num_workers=4)
    inputpaths, labels = make_pathlist_and_labellist(f'{config.dataset.datadir}/validate/', ['noise', 'cbc'], [0, 1])
    dataset_val = LabelDataset(inputpaths, labels, transform=transforms)
    dataloader_val = DataLoader(dataset_val, batch_size=config.train.batchsize, shuffle=True, drop_last=True, num_workers=4)

    # inputs, labels = load_dataset(f'{config.dataset.datadir}/train/', ['noise', 'cbc'], (img_channel, img_height, img_width))
    # tensor_dataset_tr = MyDataset(inputs, labels, transforms)
    # dataloader_tr = DataLoader(tensor_dataset_tr, batch_size=config.train.batchsize, shuffle=True, drop_last=True, num_workers=4)
    # inputs, labels = load_dataset(f'{config.dataset.datadir}/validate/', ['noise', 'cbc'], (img_channel, img_height, img_width))
    # tensor_dataset_val = MyDataset(inputs, labels, transforms)
    # dataloader_val = DataLoader(tensor_dataset_val, batch_size=config.train.batchsize, shuffle=True, drop_last=True, num_workers=4)

    # Create model
    model = instantiate_neuralnetwork(config)
    model = model.to(device)
    print("Network model created")
    # Save the model structure
    with open(f'{modeldirectory}/model_structure.txt', 'w') as f:
        f.write(repr(summary(model, input_size=(1, input_channel, input_height, input_width))))

    # Define loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = instantiate(config.train.optimizer, model.parameters())
    optimizer.zero_grad()
    accuracy = Accuracy(task="multiclass", num_classes=2).to(device)
    # Train model
    trainloss_list = []  # [gpstime, train loss, epoch]
    validateloss_list = []  # [gpstime, validate loss, epoch]
    trainaccuracy_list = []
    validateaccuracy_list = []
    train_starttime = time.time()
    for epoch in range(config.train.num_epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(dataloader_tr):
            inputs, labels = inputs.to(device), labels.to(device)
            # Zero the parameter gradients
            optimizer.zero_grad()
            # Forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            accuracy.update(outputs, labels)
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
        running_loss /= (i + 1)
        acc = accuracy.compute()
        trainloss_list.append([time.time(), running_loss, epoch + 1])
        trainaccuracy_list.append([time.time(), acc.cpu(), epoch + 1])
        accuracy.reset()

        # Evaluate model
        valloss = 0
        model.eval()
        with torch.no_grad():
            for j, data in enumerate(dataloader_val):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                valloss += criterion(outputs, labels).item()
                accuracy.update(outputs, labels)
        valloss /= (j + 1)
        acc = accuracy.compute()
        validateloss_list.append([time.time(), valloss, epoch + 1])
        validateaccuracy_list.append([time.time(), acc.cpu(), epoch + 1])
        print(f"[Epoch {epoch + 1}] Validate loss: {valloss:.5f}")
        # Reset metric
        accuracy.reset()

    train_endtime = time.time()
    print(f'Run time: {train_endtime - train_starttime} [sec]')

    # Plot the training and validation curve
    filename = f'{modeldirectory}/learning_curve.pdf'
    plot_training_curve(trainloss_list, validateloss_list, filename)
    filename = f'{modeldirectory}/accuracy_curve.pdf'
    plot_training_curve(trainaccuracy_list, validateaccuracy_list, filename)

    # Save train loss
    with open(f'{modeldirectory}/train_crossentropy_loss.txt', 'w') as f:
        for row in trainloss_list:
            print(*row, file=f)
    with open(f'{modeldirectory}/train_accuracy_loss.txt', 'w') as f:
        for row in trainaccuracy_list:
            print(*row, file=f)

    # Save validation loss
    with open(f'{modeldirectory}/validate_crossentropy_loss.txt', 'w') as f:
        for row in validateloss_list:
            print(*row, file=f)
    with open(f'{modeldirectory}/validate_accuracy_loss.txt', 'w') as f:
        for row in validateaccuracy_list:
            print(*row, file=f)

    # Save the trained model
    torch.save(model.to('cpu').state_dict(), f'{modeldirectory}/model.pth')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a neural network")
    parser.add_argument('--config', type=str, default='./config/config_train.yaml', help='Configure file.')
    parser.add_argument('--dirname', type=str, default=None, help='Output directory')
    args = parser.parse_args()

    assert os.path.exists(args.config), f"File `{args.config}` does not exit."
    main(args)
