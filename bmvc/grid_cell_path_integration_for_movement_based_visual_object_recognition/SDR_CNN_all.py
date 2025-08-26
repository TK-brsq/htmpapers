import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from nupic.torch.modules import KWinners2d, rezero_weights, update_boost_strength
from torch.utils.data import random_split
from torch.utils.data import ConcatDataset
from tqdm import tqdm
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from torch.utils.data import SubsetRandomSampler

seed_val = 1
torch.manual_seed(seed_val)
np.random.seed(seed_val)
LEARNING_RATE = 0.01  # Recommend 0.01
MOMENTUM = 0.5
EPOCHS = 10  # Recommend 10
FIRST_EPOCH_BATCH_SIZE = 4  # Used for optimizing k-WTA
TRAIN_BATCH_SIZE = 128  # Recommend 128
TEST_BATCH_SIZE = 512
PERCENT_ON = 0.15  # Recommend 0.15
BOOST_STRENGTH = 20.0  # Recommend 20
DATASET = "mnist"  # Options are "mnist" or "fashion_mnist"; note in some cases
#==================================================================================
GRID_SIZE = [5] # <-----------
NUM_CLASSES = 10 # < --------------


class SDRCNNBase_(nn.Module):
    def __init__(self, percent_on, boost_strength, grid=5):
        self.grid = grid
        super(SDRCNNBase_, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=0)# k=5, p=2
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=0)
        self.pool2 = nn.AdaptiveMaxPool2d((grid, grid))
        self.k_winner = KWinners2d(channels=128, percent_on=percent_on, boost_strength=boost_strength, local=True)
        self.dense1 = nn.Linear(in_features=128 * grid**2, out_features=256)
        self.dense2 = nn.Linear(in_features=256, out_features=128)
        self.output = nn.Linear(in_features=128, out_features=NUM_CLASSES) # CHANGED HERE
        self.softmax = nn.LogSoftmax(dim=1)

    def until_kwta(self, inputs):
        x = F.relu(self.conv1(inputs))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.k_winner(x)
        x = x.view(-1, 128 * self.grid**2)
        return x

    def forward(self, inputs):
        x = self.until_kwta(inputs)
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        x = self.softmax(self.output(x))
        return x

    def output_sdr(self, inputs):
        x = self.until_kwta(inputs)
        x = (x > 0).float()
        return x

class SequentialSubSampler(torch.utils.data.Sampler):
    r"""Custom sampler to take elements sequentially from a given list of indices.
    Performing sampling sequentially is helpful for keeping track of the generated SDRs
    and their correspondence to examples in the MNIST data-set.
    Using indices to sub-sample enables creating a splits of the training data.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)

class RelabelImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None, target_transform=None):
        super().__init__(root, transform=transform, target_transform=target_transform)
        
        # 自動生成されたクラス名→0,1,...のマップ
        # 例えば {'10': 0, '11': 1}
        self.class_to_true_label = {0: 10, 1: 11}
        #self.class_to_true_label = {i: int(cls_name) for cls_name, i in self.class_to_idx.items()}

    def __getitem__(self, index):
        image, label = super().__getitem__(index)
        true_label = self.class_to_true_label[label]
        return image, true_label
    
def post_batch(model):
    model.apply(rezero_weights)

def data_setup10():
    if DATASET == "mnist":
        print("Using MNIST data-set")
        normalize = transforms.Compose([transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))])
        train_dataset = datasets.MNIST("data", train=True, download=True,
                                       transform=normalize)
        test_dataset = datasets.MNIST("data", train=False,
                                      download=True,
                                      transform=normalize)

    total_training_len = len(train_dataset)
    indices = range(total_training_len)
    val_split = int(np.floor(0.1 * total_training_len))
    train_idx, test_cnn_idx = indices[val_split:], indices[:val_split]

    train_sampler = SequentialSubSampler(train_idx)
    test_cnn_sampler = SequentialSubSampler(test_cnn_idx)
    test_sdr_class_len = len(test_dataset)
    test_sdr_classifier_sample = SequentialSubSampler(range(test_sdr_class_len))

    # Configure data loaders
    first_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=FIRST_EPOCH_BATCH_SIZE,
                                               sampler=train_sampler)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=TRAIN_BATCH_SIZE,
                                               sampler=train_sampler)
    test_cnn_loader = torch.utils.data.DataLoader(train_dataset,
                                                  batch_size=TEST_BATCH_SIZE,
                                                  sampler=test_cnn_sampler)
    test_sdrc_loader = torch.utils.data.DataLoader(test_dataset,
                                                   batch_size=TEST_BATCH_SIZE,
                                                   sampler=test_sdr_classifier_sample)

    return first_loader, train_loader, test_cnn_loader, test_sdrc_loader

def data_setup12():
    ##### Ordinal data #####
    normalize = transforms.Compose([transforms.Grayscale(), transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset1 = datasets.MNIST("data", train=True, download=True, transform=normalize)
    test_dataset1 = datasets.MNIST("data", train=False, download=True, transform=normalize)
    labels = [train_dataset1[i][1] for i in range(len(train_dataset1))]
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=1)
    for train_idx, val_idx in splitter.split(np.zeros(len(labels)), labels):
        train_dataset0 = Subset(train_dataset1, train_idx)
        val_dataset1 = Subset(train_dataset1, val_idx)
    ##### New Data #####
    dataset = RelabelImageFolder(root='mydata', transform=normalize)
    labels2 = [dataset[i][1] for i in range(len(dataset))]
    splitter2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=1)
    for train_idx, val_idx in splitter2.split(np.zeros(len(labels2)), labels2):
        val_dataset2 = Subset(dataset, train_idx)
        test_dataset2 = Subset(dataset, val_idx)
    # Concat
    train_dataset = ConcatDataset([train_dataset0, dataset])
    gcn_tr_dataset = ConcatDataset([val_dataset1, val_dataset2])
    gcn_ts_dataset = ConcatDataset([test_dataset1, test_dataset2])
    first_loader = torch.utils.data.DataLoader(train_dataset, batch_size=FIRST_EPOCH_BATCH_SIZE, drop_last=True, shuffle=False)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, drop_last=True, shuffle=False)
    gcn_training_loader = torch.utils.data.DataLoader(gcn_tr_dataset, batch_size=TEST_BATCH_SIZE, drop_last=False, shuffle=False)
    gcn_testing_loader = torch.utils.data.DataLoader(gcn_ts_dataset, batch_size=TEST_BATCH_SIZE, drop_last=False, shuffle=False)

    return first_loader, train_loader, gcn_training_loader, gcn_testing_loader

def train(model, loader, optimizer, criterion, post_batch_callback=None):

    model.train()
    for data, target in tqdm(loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if post_batch_callback is not None:
            post_batch_callback(model)

def test(model, loader, criterion, epoch, sdr_output_subset=None, grid_size=5):
    model.eval()
    loss = 0
    total_correct = 0
    all_sdrs = []
    all_labels = []

    with torch.no_grad():
        for data, target in loader:
            output = model(data)
            all_sdrs.append(np.array(model.output_sdr(data)))
            all_labels.append(target)

            loss += criterion(output, target, reduction="sum").item()  # sum up batch
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max
            total_correct += pred.eq(target.view_as(pred)).sum().item()

    all_sdrs = np.concatenate(all_sdrs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    if sdr_output_subset is not None:
        print("Saving  SDR and label")
        name = str(NUM_CLASSES) + '-' + str(grid_size) + '/' if NUM_CLASSES > 10 else str(grid_size) + '/'
        np.save("python2_htm_docker/docker_dir/dataset" + name + DATASET + "_SDRs_" + sdr_output_subset, all_sdrs)
        np.save("python2_htm_docker/docker_dir/dataset" + name + DATASET + "_labels_" + sdr_output_subset, all_labels)

    return {"accuracy": total_correct / len(loader.dataset),
            "loss": loss / len(loader.dataset),
            "total_correct": total_correct}

if __name__ == "__main__":

    for grid_size in GRID_SIZE:

        print('Loading Data')
        if NUM_CLASSES == 12:
            first_loader, train_loader, test_cnn_loader, test_sdrc_loader = data_setup12()
        elif NUM_CLASSES == 10:
            first_loader, train_loader, test_cnn_loader, test_sdrc_loader = data_setup10()
        model = SDRCNNBase_(PERCENT_ON, BOOST_STRENGTH, grid_size)
        sgd = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

        if os.path.exists("saved_networks/") is False:
            try:
                os.mkdir("saved_networks/")
            except OSError:
                pass
        name = str(NUM_CLASSES) + '-' + str(grid_size) + '/'
        if (os.path.exists("python2_htm_docker/docker_dir/dataset" + name) is False):
            try:
                os.mkdir("python2_htm_docker/docker_dir/dataset" + name)
            except OSError:
                pass

        print("Performing first epoch")
        train(model=model, loader=first_loader, optimizer=sgd, criterion=F.nll_loss, post_batch_callback=post_batch)
        model.apply(update_boost_strength)
        test(model=model, loader=test_cnn_loader, epoch="pre_epoch", \
             criterion=F.nll_loss, grid_size=grid_size)

        print("Performing full training")
        for epoch in range(1, EPOCHS):
            train(model=model, loader=train_loader, optimizer=sgd, criterion=F.nll_loss, post_batch_callback=post_batch)
            model.apply(update_boost_strength)
            results = test(model=model, loader=test_cnn_loader, epoch=epoch, \
                           criterion=F.nll_loss, grid_size=grid_size)
            print(results)

        print("Saving network state")
        name = str(NUM_CLASSES) + '-' + str(grid_size) + '.pt'
        torch.save(model.state_dict(), "saved_networks/" + name)

        print("\nResults from training data-set.")
        results = test(model=model, loader=train_loader,
                    epoch="final_train", criterion=F.nll_loss,
                    sdr_output_subset="base_net_training", grid_size=grid_size)
        print(results)

        print("\nResults from data-set for training GCN.")
        results = test(model=model, loader=test_cnn_loader,
                    epoch="test_CNN", criterion=F.nll_loss,
                    sdr_output_subset="SDR_classifiers_training", grid_size=grid_size)
        print(results)

        print("\nResults from data-set for evaluating GCN")
        results = test(model=model, loader=test_sdrc_loader,
                    epoch="test_SDRs", criterion=F.nll_loss,
                    sdr_output_subset="SDR_classifiers_testing", grid_size=grid_size)
        print(results)
