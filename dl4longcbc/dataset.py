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


class LabelDataset(torch.utils.data.Dataset):
    '''
    data: List of paths
    label: List of labels
    '''
    def __init__(self, data, label, transform=None):
        self.transform = transform
        self.data = data
        self.data_num = len(data)
        self.label = label

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        out_data = self.transform(torch.load(self.data[idx], weights_only=True))
        out_label = torch.tensor(self.label[idx], dtype=torch.long)
        return out_data, out_label


def normalize_tensor(x):
    # x.size = (N, C, H, W) is assumed.
    xmin = torch.amin(x, dim=(1, 2, 3), keepdims=True)
    xmax = torch.amax(x, dim=(1, 2, 3), keepdims=True)
    return (x - xmin) / (xmax - xmin)


def load_dataset(datadir, labelnamelist, imgsize, labellist=None):
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
    ndata = 0
    for labelname in labelnamelist:
        target_dir = f'{datadir}/{labelname}/'
        assert os.path.exists(target_dir), f"Directory `{target_dir}` does not exist."
        all_files = os.listdir(target_dir)
        filelist[labelname] = [f for f in all_files if pattern.fullmatch(f)]
        ndatalist[labelname] = len(filelist[labelname])
        ndata += ndatalist[labelname]

    # Prepare input tensors and label tensors
    input_tensors = torch.zeros((ndata, C, H, W), dtype=torch.float32)
    label_tensors = torch.zeros((ndata,), dtype=torch.long)
    idx = 0
    for (label, labelname) in zip(labellist, labelnamelist):
        target_dir = f'{datadir}/{labelname}/'
        for j in range(ndatalist[labelname]):
            input_tensors[idx] = torch.load(f"{target_dir}/{filelist[labelname][j]}")
            label_tensors[idx] = label
            print(f'{labelname}: {idx} th data loaded.')
            idx += 1
    return normalize_tensor(input_tensors), label_tensors


def make_pathlist_and_labellist(datadir, labelnames, labels=None):
    '''
    e.g.
    datadir is the direcotry.
    labelnames = ['noise', 'cbc'] or ['noise', 'cbc', 'glitch']
    labels = [0, 1] or [0, 1, 2]
    '''

    # The number of classes
    nclass = len(labelnames)
    if labels is None:
        labels = [i for i in range(nclass)]

    # List up all pth files in the direcotry
    filelist = []
    labellist = []
    pattern = re.compile(r"inputs_\d{10}_\d{2}_\d+\.pth")
    for label, labelname in zip(labels, labelnames):
        target_dir = f'{datadir}/{labelname}/'
        assert os.path.exists(target_dir), f"Directory `{target_dir}` does not exist."
        all_files_in_target_dir = os.listdir(target_dir)
        all_file_paths = [f'{target_dir}/{f}' for f in all_files_in_target_dir if pattern.fullmatch(f)]
        filelist.extend(all_file_paths)
        labellist.extend([label] * len(all_file_paths))
    return filelist, labellist
