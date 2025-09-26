#!/usr/bin/env python
import os
import time
import shutil
import argparse
from datetime import datetime
from pytz import timezone
import torch
from torch.utils.data import DataLoader
from torchinfo import summary
from torchmetrics.classification import Accuracy
from omegaconf import OmegaConf
from hydra.utils import instantiate
import dl4longcbc.dataset_train as ds
from dl4longcbc.net import instantiate_neuralnetwork
import dl4longcbc.utils as utils


# ----------------------------------------------------------------
# Training loop
# ----------------------------------------------------------------
def main(args):

    # Load parameters
    config = OmegaConf.load(args.config)
    snr_range_schduler = utils.SNRRangeScheduler(config)

    # Make model directory
    if args.dirname is None:
        now_datetime = datetime.now(timezone('Asia/Tokyo')).strftime('%Y%m%d_%H%M%S')
        modeldirectory = os.path.join('./data/model', config.train.experiment_name, now_datetime)
    else:
        modeldirectory = os.path.join('./data/model', config.train.experiment_name, args.dirname)
    utils.if_not_exist_makedir(modeldirectory)
    shutil.copy(args.config, modeldirectory)

    # Set device
    print(f'Is gpu available? {torch.cuda.is_available()}')
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Make dataloader
    input_height = config.net.input_height
    input_width = config.net.input_width
    input_channel = config.net.input_channel

    # Get dataset
    num_workers = config.train.num_workers
    # Training dataset and data loader
    traindatadir = os.path.join(config.dataset.directory, 'train')
    inputpaths_tr, labellists_tr = ds.make_pathlist_and_labellist(traindatadir, config.dataset.signals_tr)
    noisepaths_tr = ds.get_noise_filepaths(traindatadir, config.dataset.noises_tr)
    # Validation dataset and data loader
    valdatadir = os.path.join(config.dataset.directory, 'validate')
    inputpaths_val, labellists_val = ds.make_pathlist_and_labellist(valdatadir, config.dataset.signals_val)
    noisepaths_val = ds.get_noise_filepaths(valdatadir, config.dataset.noises_val)

    # Create model
    model = instantiate_neuralnetwork(config)
    model = model.to(device)
    print("Network model created")
    # Save the model structure
    with open(os.path.join(modeldirectory, 'model_structure.txt'), 'w') as f:
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
        # Prepare dataset if schedule flg is True
        flg, snrrange = snr_range_schduler.step(epoch)
        if flg:
            dataset_tr = ds.LabelDataset(inputpaths_tr, labellists_tr, noisepaths_tr, snrrange, smearing_kernel=config.train.smearing_kernel)
            dataloader_tr = DataLoader(dataset_tr, batch_size=config.train.batchsize, shuffle=True, drop_last=True, num_workers=num_workers)
            dataset_val = ds.LabelDataset(inputpaths_val, labellists_val, noisepaths_val, snrrange, smearing_kernel=config.train.smearing_kernel)
            dataloader_val = DataLoader(dataset_val, batch_size=config.train.batchsize, shuffle=True, drop_last=True, num_workers=num_workers)

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
            if config.train.gradient_max_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.train.gradient_max_norm)
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
    filename = os.path.join(modeldirectory, 'learning_curve.pdf')
    utils.plot_training_curve(trainloss_list, validateloss_list, filename, ylabel='Cross entropy loss')
    filename = os.path.join(modeldirectory, 'accuracy_curve.pdf')
    utils.plot_training_curve(trainaccuracy_list, validateaccuracy_list, filename, ylabel='Accuracy')

    # Save train loss
    with open(os.path.join(modeldirectory, 'train_crossentropy_loss.txt'), 'w') as f:
        for row in trainloss_list:
            print(*row, file=f)
    with open(os.path.join(modeldirectory, 'train_accuracy_loss.txt'), 'w') as f:
        for row in trainaccuracy_list:
            print(*row, file=f)

    # Save validation loss
    with open(os.path.join(modeldirectory, 'validate_crossentropy_loss.txt'), 'w') as f:
        for row in validateloss_list:
            print(*row, file=f)
    with open(os.path.join(modeldirectory, 'validate_accuracy_loss.txt'), 'w') as f:
        for row in validateaccuracy_list:
            print(*row, file=f)

    # Save the trained model
    torch.save(model.to('cpu').state_dict(), os.path.join(modeldirectory, 'model.pth'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a neural network")
    parser.add_argument('--config', type=str, default='./config/config_train.yaml', help='Configure file.')
    parser.add_argument('--dirname', type=str, default=None, help='Output directory')
    args = parser.parse_args()

    assert os.path.exists(args.config), f"File `{args.config}` does not exit."
    main(args)
