"""Dataset setting and data loader for MNIST."""


import torch
from torchvision import datasets, transforms
from datasets.memcached_dataset import McDataset
import params


def get_mnist(train):
    """Get MNIST dataset loader."""
    # image pre-processing
    pre_process = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize(
                                          mean=params.dataset_mean,
                                          std=params.dataset_std)])

    # dataset and data loader
    if train:
      mnist_dataset = McDataset(root_dir=params.src_dataset_root,meta_file = params.src_dataset_list,
                                   transform=pre_process)
    else:
      mnist_dataset = McDataset(root_dir=params.src_dataset_eval_root,meta_file = params.src_dataset_eval_list, transform=pre_process)

    mnist_data_loader = torch.utils.data.DataLoader(
        dataset=mnist_dataset,
        batch_size=params.batch_size,
        shuffle=True)

    return mnist_data_loader
