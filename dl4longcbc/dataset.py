import os
import re
import torch
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, data, target, transform=None):
        self.data = data
        self.target = target
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        if self.transform:
            x = self.transform(x)
        y = self.target[index]
        return x, y


def normalize_tensor(x):
    # x.size = (1, H, W) is assumed.
    xmin = torch.min(x)
    xmax = torch.max(x)
    return (x - xmin) / (xmax - xmin)


def load_dataset(datadir, labelnamelist, ndata, imgsize, labellist=None, ninit=0):
    '''
    Assuming the file path '{datadir}/{label}/inputs_{GPSTime}_{foreground index}_{data index}.pth'.
    '''

    C, H, W = imgsize
    # The number of classes
    nclass = len(labelnamelist)
    if labellist is None:
        labellist = [i for i in range(nclass)]
    # List up all pth files in the direcotry
    filelist = {}
    pattern = re.compile(r"inputs_\d{10}_\d{2}_\d+\.pth")

    ndatalist = {}
    for labelname in labelnamelist:
        target_dir = f'{datadir}/{labelname}/'
        assert os.path.exists(target_dir), f"Directory `{target_dir}` does not exist."
        all_files = os.listdir(target_dir)
        filelist[labelname] = [f for f in all_files if pattern.fullmatch(f)]
        ndatalist[labelname] = len(filelist[labelname])

    # Prepare input tensors and label tensors
    input_tensors = torch.zeros((ndata, C, H, W), dtype=torch.float32)
    label_tensors = torch.zeros((ndata,), dtype=torch.long)
    idx = 0
    for (label, labelname) in zip(labellist, labelnamelist):
        target_dir = f'{datadir}/{labelname}/'
        for j in range(ndatalist[labelname]):
            input_tensors[idx] = normalize_tensor(torch.load(f"{target_dir}/{filelist[labelname][j]}"))
            label_tensors[idx] = label
            idx += 1
    return input_tensors, label_tensors
