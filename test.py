#!/usr/bin/env python
import argparse
import torch
import torch.nn as nn
from torch.nn.functional import softmax
from torch.utils.data import DataLoader
from torchvision.transforms import RandomCrop
from omegaconf import OmegaConf
from dl4longcbc.dataset import make_pathlist_and_labellist, LabelDataset
from dl4longcbc.net import instantiate_neuralnetwork
from dl4longcbc.dataset import TestResult


# >>> test loop >>>
def main(args):

    modeldir = args.modeldir
    datadir = args.datadir
    outdir = args.outdir
    ndata = args.ndata
    config_tr = OmegaConf.load(f'{modeldir}/config_train.yaml')

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # >>> Load model >>>
    model = instantiate_neuralnetwork(config_tr)
    model.load_state_dict(torch.load(f'{modeldir}/model.pth', weights_only=True))
    model = model.to(device)
    model.eval()
    # <<< Load model <<<

    # Make dataloader
    # img_height = config_tr.dataset.img_height
    # img_width = config_tr.dataset.img_width
    # img_channel = config_tr.dataset.img_channel
    input_height = config_tr.net.input_height
    input_width = config_tr.net.input_width
    # input_channel = config_tr.net.input_channel
    transforms = nn.Sequential(
        RandomCrop((input_height, input_width))
    )
    nb = args.batchsize
    inputpaths, labels = make_pathlist_and_labellist(f'{datadir}/test/', ['noise'], [0])
    dataset = LabelDataset(inputpaths, labels, transform=transforms)
    dataloader = DataLoader(dataset, batch_size=nb, shuffle=False, drop_last=False, num_workers=8)
    ndata = len(inputpaths)

    outputtensor = torch.empty((ndata, 2), dtype=torch.float32)
    labeltensor = torch.empty((ndata,), dtype=torch.long)
    # Test model
    with torch.no_grad():
        idx_offset = 0
        for i, data in enumerate(dataloader, 0):
            inputs, labels = data
            inputs = inputs.to(device)
            outputs = softmax(model(inputs).cpu(), dim=1)
            kini = idx_offset
            kend = idx_offset + len(inputs)
            outputtensor[kini: kend] = outputs
            labeltensor[kini: kend] = labels
            idx_offset = kend
    labeltensor = nn.functional.one_hot(labeltensor, num_classes=2)
    print("Test: Test data processed.")
    result_dict = {
        "label": labeltensor,
        "output": outputtensor
    }
    result_model = TestResult(result_dict)
    torch.save(result_model, f"{outdir}/result.pth")
    print("Test: Result saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('--outdir', type=str, help='output directory name')
    parser.add_argument('--modeldir', type=str, help='Directory where the trained model is saved.')
    parser.add_argument('--datadir', type=str, help='dataset directory (including ***/test/ or noise)')
    parser.add_argument('--ndata', type=int, help='The number of test data')
    parser.add_argument('--batchsize', type=int, default=200, help='Batch size')
    args = parser.parse_args()

    main(args)
