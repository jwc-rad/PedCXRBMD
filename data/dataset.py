import copy
import glob
import importlib
import itertools
import json
import numpy as np
import os
import pandas as pd
import pickle
from PIL import Image
import random
from sklearn.model_selection import KFold
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler

from monai.transforms import Compose, LoadImage, BorderPad

from utils.hydra import instantiate_list

from .utils import TwoStreamSampler

class BMDXRDataset(Dataset):
    def __init__(
        self,
        transform,
        image_dir: Dict,
        label_name: str = None,
        phase: str = "train",
        iterations_per_epoch: int = None,
        transform_seed=None,
        **prepare_data_kwargs,
    ):
        super().__init__()
        self.prepare_transforms(transform, transform_seed)
        self.image_dir = image_dir
        self.label_name = label_name
        self.phase = phase
        self.iterations_per_epoch = iterations_per_epoch

        self.prepare_data(**prepare_data_kwargs)

    def __len__(self):
        if self.phase == "train":
            return (
                self.label_size
                if self.iterations_per_epoch is None
                else self.iterations_per_epoch
            )
        else:
            return self.image_size

    def __getitem__(self, index):
        read_items = {}
        metadata = {}
        for k, p in self.image_paths.items():
            imageX_path = p[index % self.image_size]
            read_items[k] = imageX_path
            metadata[f"{k}_path"] = imageX_path

        if hasattr(self, "table_paths"):
            table_path = self.table_paths[index % self.image_size]
            read_items["table"] = table_path

        if hasattr(self, "label_paths"):
            label_path = self.label_paths[index % self.image_size]
            read_items["label"] = np.array([label_path])
            # read_items['label_raw'] = label_path
            # metadata['label_path'] = label_path

        read_items["metadata"] = metadata

        return_items = self.run_transform(read_items)
        return return_items

    ## override this to define transforms
    def prepare_transforms(self, transform, transform_seed=None):
        tfm = instantiate_list(transform)
        self.run_transform = Compose(tfm)
        if isinstance(transform_seed, int):
            self.run_transform.set_random_state(seed=transform_seed)

    ## override this to define self.keys, paths, and etc.
    def prepare_data(
        self,
        dataset_file=None,
        cv_split=5,
        cv_fold=0,
        split_seed=12345,
        table_cols=[],
        image_extension="npy",
        override_split_phase=None,
        override_cv_split=False,
        **kwargs,
    ):
        if override_split_phase is None:
            ppp = self.phase
        else:
            ppp = override_split_phase
        
        if dataset_file is not None:
            with open(dataset_file, "rb") as f:
                dsf = pickle.load(f)
        
        all_keys = []
        for d in self.image_dir.values():
            _paths = sorted(glob.glob(os.path.join(d, f"*.{image_extension}")))
            _keys = [
                os.path.basename(x).split(f".{image_extension}")[0] for x in _paths
            ]
            all_keys.append(set(_keys))

        _c_keys = sorted(set.intersection(*all_keys))

        this_split = dsf["split"][cv_fold][ppp]
        _filtered_keys = [x for x in _c_keys if x in this_split]

        images_paths = {}
        image_size = -1
        for k, d in self.image_dir.items():
            _paths = sorted(glob.glob(os.path.join(d, f"*.{image_extension}")))
            _paths = [
                x
                for x in _paths
                if os.path.basename(x).split(f".{image_extension}")[0] in _filtered_keys
            ]
            images_paths[k] = _paths
            image_size = len(_paths)
        self.image_paths = images_paths
        self.image_size = image_size
        print_str = f"image num: {self.image_size}"

        if len(table_cols) > 0:
            _tables = []
            for c in table_cols:
                _paths = [dsf["data"][k][c] for k in _filtered_keys]
                _tables.append(_paths)
            self.table_paths = np.array(_tables).T
            self.table_size = len(_tables[0])
            print_str += f", table num: {self.table_size}"

        if getattr(self, "label_name", None) is not None:
            _paths = [dsf["data"][k][self.label_name] for k in _filtered_keys]
            self.label_paths = np.array(_paths)
            self.label_size = len(_paths)
            print_str += f", label num: {self.label_size}"

        print(print_str)


class BoneAgeDataset(Dataset):
    def __init__(
        self,
        transform,
        image_dir: Dict,
        table_file: str,
        label_name: str = None,
        phase: str = "train",
        iterations_per_epoch: int = None,
        transform_seed=None,
        **prepare_data_kwargs,
    ):
        super().__init__()
        self.prepare_transforms(transform, transform_seed)
        self.image_dir = image_dir
        self.table_file = table_file
        self.label_name = label_name
        self.phase = phase
        self.iterations_per_epoch = iterations_per_epoch

        self.prepare_data(**prepare_data_kwargs)

    def __len__(self):
        if self.phase == "train":
            return (
                self.label_size
                if self.iterations_per_epoch is None
                else self.iterations_per_epoch
            )
        else:
            return self.image_size

    def __getitem__(self, index):
        read_items = {}
        metadata = {}
        for k, p in self.image_paths.items():
            imageX_path = p[index % self.image_size]
            read_items[k] = imageX_path
            metadata[f"{k}_path"] = imageX_path

        if hasattr(self, "table_paths"):
            table_path = self.table_paths[index % self.image_size]
            read_items["table"] = table_path

        if hasattr(self, "label_paths"):
            label_path = self.label_paths[index % self.image_size]
            read_items["label"] = label_path
            # read_items['label_raw'] = label_path
            # metadata['label_path'] = label_path

        read_items["metadata"] = metadata

        return_items = self.run_transform(read_items)
        return return_items

    ## override this to define transforms
    def prepare_transforms(self, transform, transform_seed=None):
        tfm = instantiate_list(transform)
        self.run_transform = Compose(tfm)
        if isinstance(transform_seed, int):
            self.run_transform.set_random_state(seed=transform_seed)

    ## override this to define self.keys, paths, and etc.
    def prepare_data(
        self,
        dataset_file=None,
        cv_split=5,
        cv_fold=0,
        split_seed=12345,
        table_cols=[],
        id_name="ID",
        image_extension="npy",
        **kwargs,
    ):
        all_keys = []
        for d in self.image_dir.values():
            _paths = sorted(glob.glob(os.path.join(d, f"*.{image_extension}")))
            _keys = [
                os.path.basename(x).split(f".{image_extension}")[0] for x in _paths
            ]
            all_keys.append(set(_keys))

        dfgt = pd.read_csv(self.table_file)
        assert id_name in dfgt
        _keys = list(dfgt[id_name])
        all_keys.append(set(_keys))
        _c_keys = sorted(set.intersection(*all_keys))

        if dataset_file is None:
            if self.phase in ["train", "valid"]:
                kf = KFold(n_splits=cv_split, shuffle=True, random_state=split_seed)
                _filtered_idx = (
                    [x for x, _ in kf.split(_c_keys)][cv_fold]
                    if self.phase == "train"
                    else [x for _, x in kf.split(_c_keys)][cv_fold]
                )
                _filtered_keys = [_c_keys[i] for i in _filtered_idx]
        else:
            with open(dataset_file, "rb") as f:
                dsf = pickle.load(f)
            this_split = dsf["split"][cv_fold][self.phase]
            _filtered_keys = [x for x in _c_keys if x in this_split]

        images_paths = {}
        image_size = -1
        for k, d in self.image_dir.items():
            _paths = sorted(glob.glob(os.path.join(d, f"*.{image_extension}")))
            _paths = [
                x
                for x in _paths
                if os.path.basename(x).split(f".{image_extension}")[0] in _filtered_keys
            ]
            images_paths[k] = _paths
            image_size = len(_paths)
        self.image_paths = images_paths
        self.image_size = image_size
        print_str = f"image num: {self.image_size}"

        if len(table_cols) > 0:
            _tables = []
            for c in table_cols:
                _dict = {k: v for k, v in zip(dfgt[id_name], dfgt[c])}
                _paths = [_dict[k] for k in _filtered_keys]
                _tables.append(_paths)
            self.table_paths = np.array(_tables).T
            self.table_size = len(_tables[0])
            print_str += f", table num: {self.table_size}"

        if getattr(self, "label_name", None) is not None:
            _dict = {k: v for k, v in zip(dfgt[id_name], dfgt[self.label_name])}
            _paths = [_dict[k] for k in _filtered_keys]
            self.label_paths = np.array(_paths)
            self.label_size = len(_paths)
            print_str += f", label num: {self.label_size}"

        print(print_str)


class SegmentationImageDataset(Dataset):
    def __init__(
        self,
        transform,
        image_dir: Dict,
        label_dir=None,
        phase: str = "train",
        iterations_per_epoch: int = None,
        transform_seed=None,
        **prepare_data_kwargs,
    ):
        super().__init__()
        self.prepare_transforms(transform, transform_seed)
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.phase = phase
        self.iterations_per_epoch = iterations_per_epoch

        self.prepare_data(**prepare_data_kwargs)

    def __len__(self):
        if self.phase == "train":
            return (
                self.label_size
                if self.iterations_per_epoch is None
                else self.iterations_per_epoch
            )
        else:
            return self.image_size

    def __getitem__(self, index):
        read_items = {}
        metadata = {}
        for k, p in self.image_paths.items():
            imageX_path = p[index % self.image_size]
            read_items[k] = imageX_path
            metadata[f"{k}_path"] = imageX_path

        if hasattr(self, "label_paths"):
            label_path = self.label_paths[index % self.image_size]
            read_items["label"] = label_path
            # read_items['label_raw'] = label_path
            metadata["label_path"] = label_path

        read_items["metadata"] = metadata

        return_items = self.run_transform(read_items)
        return return_items

    ## override this to define transforms
    def prepare_transforms(self, transform, transform_seed=None):
        tfm = instantiate_list(transform)
        self.run_transform = Compose(tfm)
        if isinstance(transform_seed, int):
            self.run_transform.set_random_state(seed=transform_seed)

    ## override this to define self.keys, paths, and etc.
    def prepare_data(
        self,
        dataset_file=None,
        cv_split=5,
        cv_fold=0,
        split_seed=12345,
        image_extension="nii.gz",
        label_extension="nii.gz",
        **kwargs,
    ):
        all_keys = []
        for d in self.image_dir.values():
            _paths = sorted(glob.glob(os.path.join(d, f"*.{image_extension}")))
            _keys = [
                os.path.basename(x).split(f".{image_extension}")[0] for x in _paths
            ]
            all_keys.append(set(_keys))
        if getattr(self, "label_dir", None) is not None:
            _paths = sorted(
                glob.glob(os.path.join(self.label_dir, f"*.{label_extension}"))
            )
            _keys = [
                os.path.basename(x).split(f".{label_extension}")[0] for x in _paths
            ]
            all_keys.append(set(_keys))
        _c_keys = sorted(set.intersection(*all_keys))

        if dataset_file is None:
            if self.phase in ["train", "valid"]:
                kf = KFold(n_splits=cv_split, shuffle=True, random_state=split_seed)
                _filtered_idx = (
                    [x for x, _ in kf.split(_c_keys)][cv_fold]
                    if self.phase == "train"
                    else [x for _, x in kf.split(_c_keys)][cv_fold]
                )
                _filtered_keys = [_c_keys[i] for i in _filtered_idx]
        else:
            with open(dataset_file, "rb") as f:
                dsf = pickle.load(f)
            this_split = dsf["split"][cv_fold][self.phase]
            _filtered_keys = [x for x in _c_keys if x in this_split]

        images_paths = {}
        image_size = -1
        for k, d in self.image_dir.items():
            _paths = sorted(glob.glob(os.path.join(d, f"*.{image_extension}")))
            _paths = [
                x
                for x in _paths
                if os.path.basename(x).split(f".{image_extension}")[0] in _filtered_keys
            ]
            images_paths[k] = _paths
            image_size = len(_paths)
        self.image_paths = images_paths
        self.image_size = image_size
        print_str = f"image num: {self.image_size}"

        if getattr(self, "label_dir", None) is not None:
            _paths = sorted(
                glob.glob(os.path.join(self.label_dir, f"*.{label_extension}"))
            )
            _paths = [
                x
                for x in _paths
                if os.path.basename(x).split(f".{label_extension}")[0] in _filtered_keys
            ]
            self.label_paths = _paths
            self.label_size = len(_paths)
            print_str += f", label num: {self.label_size}"

        print(print_str)


class MILImageDataset_FSL(Dataset):
    """
    bag -> k images
    bag-level label
    image_dir = "bag" base dir (single dir)
    image_paths = list of list of image paths for each bag
    image_size = number of bags (e.g. len(image_paths))
    """

    def __init__(
        self,
        transform,
        image_dir: str,
        phase: str = "train",
        iterations_per_epoch: int = None,
        transform_seed=None,
        **prepare_data_kwargs,
    ):
        super().__init__()
        self.prepare_transforms(transform, transform_seed)
        self.image_dir = image_dir
        self.phase = phase
        self.iterations_per_epoch = iterations_per_epoch

        self.prepare_data(**prepare_data_kwargs)

    def __len__(self):
        if self.phase == "train":
            return (
                self.label_size
                if self.iterations_per_epoch is None
                else self.iterations_per_epoch
            )
        else:
            return self.image_size

    def __getitem__(self, index):
        read_items = {}
        metadata = {}

        imageX_bag = self.image_paths[index % self.image_size]
        N = len(imageX_bag)
        _idx = np.arange(N).tolist()
        if self.images_per_bag > 0:
            image_pick = []
            for _ in range(self.images_per_bag // N):
                image_pick += random.sample(_idx, k=N)
            image_pick += random.sample(_idx, k=self.images_per_bag % N)
        else:
            image_pick = _idx
        imageX_paths = [imageX_bag[i] for i in image_pick]
        imageX = [self.run_transform(x) for x in imageX_paths]
        imageX = torch.stack(imageX, axis=0)
        read_items["image"] = imageX
        metadata["image_path"] = imageX_paths

        if hasattr(self, "labels"):
            labelX = self.labels[index % self.image_size]
            read_items["label"] = labelX

        read_items["metadata"] = metadata
        return read_items

    ## override this to define transforms
    def prepare_transforms(self, transform, transform_seed=None):
        tfm = instantiate_list(transform)
        self.run_transform = Compose(tfm)
        if isinstance(transform_seed, int):
            self.run_transform.set_random_state(seed=transform_seed)

    ## override this to define self.keys, paths, and etc.
    def prepare_data(
        self,
        dataset_file,
        cv_fold=0,
        image_extension="png",
        images_per_bag=1,
        sample_size_label=1,
        sample_size_nolabel=1,
        **kwargs,
    ):
        with open(dataset_file, "rb") as f:
            dsf = pickle.load(f)
        # with open(dataset_file, 'r') as f:
        #    dsf = json.load(f)
        this_split = dsf["split"][cv_fold][self.phase]
        label_dict = dsf["label"][cv_fold]
        NO_LABEL = dsf["NO_LABEL"] if "NO_LABEL" in dsf else -1

        all_keys = []
        _dirs = [os.path.join(self.image_dir, x) for x in this_split]
        _keys = [this_split[i] for i, x in enumerate(_dirs) if os.path.exists(x)]
        all_keys.append(set(_keys))
        _filtered_keys = sorted(set.intersection(*all_keys))

        if self.phase == "train":
            _filtered_keys_label = [x for x in _filtered_keys if x in label_dict.keys()]
            _filtered_keys_nolabel = [
                x for x in _filtered_keys if not x in label_dict.keys()
            ]
            _filtered_keys = _filtered_keys_label + _filtered_keys_nolabel
        elif self.phase == "valid":
            _filtered_keys = [x for x in _filtered_keys if x in label_dict.keys()]

        bag_ids = _filtered_keys
        bag_dirs = [os.path.join(self.image_dir, x) for x in bag_ids]
        images_paths = [
            sorted(glob.glob(os.path.join(x, f"*.{image_extension}"))) for x in bag_dirs
        ]
        image_size = len(images_paths)

        self.image_paths = images_paths
        self.image_size = image_size
        self.images_per_bag = images_per_bag

        if self.phase == "train":
            self.labels = [
                label_dict[x] if x in _filtered_keys_label else NO_LABEL
                for x in bag_ids
            ]
            self.label_size = len(_filtered_keys_label)
            print(f"image num: {self.image_size}, label num: {self.label_size}")
        # elif self.phase == 'valid':
        else:
            count_labels = np.array([x in label_dict.keys() for x in bag_ids])
            if count_labels.all():
                self.labels = [label_dict[x] for x in bag_ids]
                print(f"image num: {self.image_size}, label num: {count_labels.sum()}")
            else:
                print(
                    f"image num: {self.image_size}, preparing without labels.. only {count_labels.sum()} labels"
                )

        # not used in FSL
        self.sample_size_label = sample_size_label
        self.sample_size_nolabel = sample_size_nolabel


class MILImageDataset_SSLv0(MILImageDataset_FSL):
    def _sampler(self, shuffle=True):
        if hasattr(self, "label_size"):
            labeled_idxs = list(range(0, self.label_size))
            unlabeled_idxs = list(range(self.label_size, self.image_size))
            return TwoStreamSampler(
                labeled_idxs,
                unlabeled_idxs,
                self.sample_size_label,
                self.sample_size_nolabel,
                shuffle,
                num_samples=self.iterations_per_epoch,
            )
        else:
            return None


class MILImageDataset_SSLv1(MILImageDataset_SSLv0):
    """
    return two images for two transforms
    """

    def __init__(
        self,
        transform,
        image_dir: str,
        phase: str = "train",
        iterations_per_epoch: int = None,
        transform2=None,
        transform_seed=None,
        transform2_seed=None,
        **prepare_data_kwargs,
    ):
        super().__init__(
            transform, image_dir, phase, iterations_per_epoch, **prepare_data_kwargs
        )
        self.prepare_transforms(transform, transform2, transform_seed, transform2_seed)

    def __getitem__(self, index):
        read_items = {}
        metadata = {}

        imageX_bag = self.image_paths[index % self.image_size]
        N = len(imageX_bag)
        _idx = np.arange(N).tolist()
        if self.images_per_bag > 0:
            image_pick = []
            for _ in range(self.images_per_bag // N):
                image_pick += random.sample(_idx, k=N)
            image_pick += random.sample(_idx, k=self.images_per_bag % N)
        else:
            image_pick = _idx
        imageX_paths = [imageX_bag[i] for i in image_pick]
        imageX = [self.run_transform(x) for x in imageX_paths]
        imageX = torch.stack(imageX, axis=0)
        read_items["image"] = imageX
        imageX2 = [self.run_transform2(x) for x in imageX_paths]
        imageX2 = torch.stack(imageX2, axis=0)
        read_items["image2"] = imageX2
        metadata["image_path"] = imageX_paths

        if hasattr(self, "labels"):
            labelX = self.labels[index % self.image_size]
            read_items["label"] = labelX

        read_items["metadata"] = metadata
        return read_items

    ## override this to define transforms
    def prepare_transforms(
        self, transform, transform2=None, transform_seed=None, transform2_seed=None
    ):
        tfm = instantiate_list(transform)
        self.run_transform = Compose(tfm)
        if isinstance(transform_seed, int):
            self.run_transform.set_random_state(seed=transform_seed)

        tfm2 = (
            instantiate_list(transform)
            if transform2 is None
            else instantiate_list(transform2)
        )
        self.run_transform2 = Compose(tfm2)
        if isinstance(transform2_seed, int):
            self.run_transform2.set_random_state(seed=transform2_seed)


### Single Image Dataset


class ImageDataset_FSL(Dataset):
    def __init__(
        self,
        transform,
        image_dir: str,
        phase: str = "train",
        iterations_per_epoch: int = None,
        **prepare_data_kwargs,
    ):
        super().__init__()
        self.prepare_transforms(transform)
        self.image_dir = image_dir
        self.phase = phase
        self.iterations_per_epoch = iterations_per_epoch

        self.prepare_data(**prepare_data_kwargs)

    def __len__(self):
        if self.phase == "train":
            return (
                self.label_size
                if self.iterations_per_epoch is None
                else self.iterations_per_epoch
            )
        else:
            return self.image_size

    def __getitem__(self, index):
        read_items = {}
        metadata = {}

        imageX_path = self.image_paths[index % self.image_size]
        read_items["image"] = imageX_path
        metadata["image_path"] = imageX_path

        if hasattr(self, "labels"):
            labelX = self.labels[index % self.image_size]
            read_items["label"] = labelX

        read_items["metadata"] = metadata
        return self.run_transform(read_items)

    ## override this to define transforms
    def prepare_transforms(self, transform):
        tfm = instantiate_list(transform)
        self.run_transform = Compose(tfm)

    ## override this to define self.keys, paths, and etc.
    def prepare_data(
        self,
        dataset_file,
        cv_fold=0,
        image_extension="npy",
        sample_size_label=1,
        sample_size_nolabel=1,
        **kwargs,
    ):
        with open(dataset_file, "rb") as f:
            dsf = pickle.load(f)
        # with open(dataset_file, 'r') as f:
        #    dsf = json.load(f)
        this_split = dsf["split"][cv_fold][self.phase]
        label_dict = dsf["label"][cv_fold]
        NO_LABEL = dsf["NO_LABEL"] if "NO_LABEL" in dsf else -1

        all_keys = this_split
        all_paths = [
            os.path.join(self.image_dir, f"{x}.{image_extension}") for x in all_keys
        ]

        _filtered_paths = [x for x in all_paths if os.path.exists(x)]
        _filtered_keys = [
            all_keys[i] for i, x in enumerate(all_paths) if x in _filtered_paths
        ]

        if self.phase == "train":
            _filtered_keys_label = [x for x in _filtered_keys if x in label_dict.keys()]
            _filtered_keys_nolabel = [
                x for x in _filtered_keys if not x in label_dict.keys()
            ]
            _filtered_keys = _filtered_keys_label + _filtered_keys_nolabel
        elif self.phase == "valid":
            _filtered_keys = [x for x in _filtered_keys if x in label_dict.keys()]

        image_paths = [
            os.path.join(self.image_dir, f"{x}.{image_extension}")
            for x in _filtered_keys
        ]
        image_size = len(image_paths)

        self.image_paths = image_paths
        self.image_size = image_size

        if self.phase == "train":
            self.labels = [
                label_dict[x] if x in _filtered_keys_label else NO_LABEL
                for x in _filtered_keys
            ]
            self.label_size = len(_filtered_keys_label)
        elif self.phase == "valid":
            self.labels = [label_dict[x] for x in _filtered_keys]

        # not used in FSL
        self.sample_size_label = sample_size_label
        self.sample_size_nolabel = sample_size_nolabel


class ImageDataset_SSLv0(ImageDataset_FSL):
    def _sampler(self, shuffle=True):
        if hasattr(self, "label_size"):
            labeled_idxs = list(range(0, self.label_size))
            unlabeled_idxs = list(range(self.label_size, self.image_size))
            return TwoStreamSampler(
                labeled_idxs,
                unlabeled_idxs,
                self.sample_size_label,
                self.sample_size_nolabel,
                shuffle,
                num_samples=self.iterations_per_epoch,
            )
        else:
            return None


class ImageDataset_SSLv1(ImageDataset_SSLv0):
    """
    return two images for two transforms
    """

    def __getitem__(self, index):
        read_items = {}
        metadata = {}

        imageX_path = self.image_paths[index % self.image_size]
        read_items["image"] = imageX_path
        read_items["image2"] = imageX_path
        metadata["image_path"] = imageX_path

        if hasattr(self, "labels"):
            labelX = self.labels[index % self.image_size]
            read_items["label"] = labelX

        read_items["metadata"] = metadata
        return self.run_transform(read_items)
