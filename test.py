#!/usr/bin/env python
import os
import argparse
import pickle
import torch
import torch.nn as nn
from torch.nn.functional import softmax
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
import gw_hybridmfcnn.dataset_test as ds
from gw_hybridmfcnn.net import instantiate_neuralnetwork
from gw_hybridmfcnn.utils import if_not_exist_makedir


# >>> test loop >>>
def main(args):

    modeldir = args.modeldir
    datadir = args.datadir
    outdir = args.outdir
    if_not_exist_makedir(outdir)
    ndata = args.ndata
    config_tr = OmegaConf.load(os.path.join(modeldir, 'config_train.yaml'))

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # >>> Load model >>>
    model = instantiate_neuralnetwork(config_tr)
    model.load_state_dict(torch.load(os.path.join(modeldir, 'model.pth'), weights_only=True))
    model = model.to(device)
    model.eval()
    # <<< Load model <<<

    # Sequential transformation
    transforms = nn.Sequential()
    if config_tr.train.smearing_kernel is not None:
        transforms.append(ds.SmearMFImage(config_tr.train.smearing_kernel))
    transforms.append(ds.NormalizeTensor())

    # Make data loader
    nb = args.batchsize
    if args.noise:
        inputpathlist, labellist = ds.make_pathlist_and_labellist(datadir, 10, ['noise'], [0], snr_threshold=None)
    elif args.cbc:
        inputpathlist, labellist = ds.make_pathlist_and_labellist(datadir, 10, ['cbc'], [1], snr_threshold=None)
    dataset = ds.LabelDataset(inputpathlist, labellist, transform=transforms)
    dataloader = DataLoader(dataset, batch_size=nb, shuffle=False, drop_last=False, num_workers=8)
    ndata = len(inputpathlist)

    presoftmax_tensor = torch.empty((ndata, 2), dtype=torch.float32)
    outputtensor = torch.empty((ndata, 2), dtype=torch.float32)
    labeltensor = torch.empty((ndata,), dtype=torch.long)
    # Test model
    with torch.no_grad():
        idx_offset = 0
        for i, data in enumerate(dataloader, 0):
            inputs, labels = data
            inputs = inputs.to(device)
            outputs_presoftmax = model(inputs).cpu()
            outputs = softmax(outputs_presoftmax, dim=1)
            kini = idx_offset
            kend = idx_offset + len(inputs)
            presoftmax_tensor[kini: kend] = outputs_presoftmax
            outputtensor[kini: kend] = outputs
            labeltensor[kini: kend] = labels
            idx_offset = kend
    labeltensor = nn.functional.one_hot(labeltensor, num_classes=2)
    print("Test: Test data processed.")
    result_dict = {
        "label": labeltensor,
        "output": outputtensor,
        "presoftmax": presoftmax_tensor
    }
    # result_model = ds.TestResult(result_dict)
    # torch.save(result_model, os.path.join(outdir, 'result.pth'))
    with open(os.path.join(outdir, 'result.pkl'), 'wb') as fo:
        pickle.dump(result_dict, fo)
    print("Test: Result saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('--outdir', type=str, help='output directory name')
    parser.add_argument('--modeldir', type=str, help='Directory where the trained model is saved.')
    parser.add_argument('--datadir', type=str, help='dataset directory (including ***/test/)')
    parser.add_argument('--ndata', type=int, help='The number of test data')
    parser.add_argument('--batchsize', type=int, default=200, help='Batch size')
    parser.add_argument('--noise', action='store_true')
    parser.add_argument('--cbc', action='store_true')
    args = parser.parse_args()

    # Check arguments consistency
    if args.noise * args.cbc:
        raise ValueError("noise and signal cannot be true at the same time.")
    elif not (args.noise or args.cbc):
        raise ValueError("Specify --noise or --signal")

    main(args)
