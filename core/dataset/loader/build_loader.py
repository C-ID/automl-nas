# Copy from mmcv
import platform
from functools import partial
import torch, random, numpy as np

from tools.parallel import collate
from tools.runner.dist_utils import get_dist_info
# from torch.utils.data import DataLoader
from tools.utils import digit_version, DataLoader

#from .sampler import DistributedGroupSampler, DistributedSampler, GroupSampler
from .sampler import DistributedSampler, AysncDistributedSampler

# if platform.system() != 'Windows':
#     # https://github.com/pytorch/pytorch/issues/973
#     import resource
#     rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
#     resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))
#
#
# def build_dataloader(dataset,
#                      imgs_per_gpu,
#                      workers_per_gpu,
#                      num_gpus=1,
#                      dist=True,
#                      shuffle=True,
#                      **kwargs):
#     """Build PyTorch DataLoader.
#
#     In distributed training, each GPU/process has a dataloader.
#     In non-distributed training, there is only one dataloader for all GPUs.
#
#     Args:
#         dataset (Dataset): A PyTorch dataset.
#         imgs_per_gpu (int): Number of images on each GPU, i.e., batch size of
#             each GPU.
#         workers_per_gpu (int): How many subprocesses to use for data loading
#             for each GPU.
#         num_gpus (int): Number of GPUs. Only used in non-distributed training.
#         dist (bool): Distributed training/test or not. Default: True.
#         shuffle (bool): Whether to shuffle the data at every epoch.
#             Default: True.
#         kwargs: any keyword argument to be used to initialize DataLoader
#
#     Returns:
#         DataLoader: A PyTorch dataloader.
#     """
#     if dist:
#         rank, world_size = get_dist_info()
#         # DistributedGroupSampler will definitely shuffle the data to satisfy
#         # that images on each GPU are in the same group
#         if shuffle:
#             sampler = DistributedGroupSampler(dataset, imgs_per_gpu,
#                                               world_size, rank)
#         else:
#             sampler = DistributedSampler(
#                 dataset, world_size, rank, shuffle=False)
#         batch_size = imgs_per_gpu
#         num_workers = workers_per_gpu
#     else:
#         sampler = GroupSampler(dataset, imgs_per_gpu) if shuffle else None
#         batch_size = num_gpus * imgs_per_gpu
#         num_workers = num_gpus * workers_per_gpu
#
#     data_loader = DataLoader(
#         dataset,
#         batch_size=batch_size,
#         sampler=sampler,
#         num_workers=num_workers,
#         collate_fn=partial(collate, samples_per_gpu=imgs_per_gpu),
#         pin_memory=False,
#         **kwargs)
#
#     return data_loader


def build_dataloader(dataset,
                     samples_per_gpu,
                     workers_per_gpu,
                     num_gpus=1,
                     dist=True,
                     shuffle=True,
                     round_up=True,
                     seed=None,
                     pin_memory=True,
                     persistent_workers=True,
                     dist_type="one2many",
                     **kwargs):
    """Build PyTorch DataLoader.

    In distributed training, each GPU/process has a dataloader.
    In non-distributed training, there is only one dataloader for all GPUs.

    Args:
        dataset (Dataset): A PyTorch dataset.
        samples_per_gpu (int): Number of training samples on each GPU, i.e.,
            batch size of each GPU.
        workers_per_gpu (int): How many subprocesses to use for data loading
            for each GPU.
        num_gpus (int): Number of GPUs. Only used in non-distributed training.
        dist (bool): Distributed training/test or not. Default: True.
        shuffle (bool): Whether to shuffle the data at every epoch.
            Default: True.
        round_up (bool): Whether to round up the length of dataset by adding
            extra samples to make it evenly divisible. Default: True.
        pin_memory (bool): Whether to use pin_memory in DataLoader.
            Default: True
        persistent_workers (bool): If True, the data loader will not shutdown
            the worker processes after a dataset has been consumed once.
            This allows to maintain the workers Dataset instances alive.
            The argument also has effect in PyTorch>=1.7.0.
            Default: True
        kwargs: any keyword argument to be used to initialize DataLoader

    Returns:
        DataLoader: A PyTorch dataloader.
    """

    rank, world_size = get_dist_info()

    if dist:
        if dist_type == "one2many":
            sampler = DistributedSampler(
                dataset, world_size, rank, shuffle=shuffle, round_up=round_up)
        elif dist_type == "many2many":
            sampler = AysncDistributedSampler(
                dataset, world_size, rank, shuffle=shuffle, round_up=round_up)
        else:
            raise Error("Sampler Type Error")
        shuffle = False
        batch_size = samples_per_gpu
        num_workers = workers_per_gpu
    else:
        sampler = None
        batch_size = num_gpus * samples_per_gpu
        num_workers = num_gpus * workers_per_gpu

    init_fn = partial(
        worker_init_fn, num_workers=num_workers, rank=rank,
        seed=seed) if seed is not None else None

    if digit_version(torch.__version__) >= digit_version('1.8.0'):
        kwargs['persistent_workers'] = persistent_workers

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=partial(collate, samples_per_gpu=samples_per_gpu),
        pin_memory=pin_memory,
        shuffle=shuffle,
        worker_init_fn=init_fn,
        **kwargs)
    return data_loader


def worker_init_fn(worker_id, num_workers, rank, seed):
    # The seed of each worker equals to
    # num_worker * rank + worker_id + user_seed
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)
