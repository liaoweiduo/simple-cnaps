from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Sampler
import torch
from PIL import Image
from torchvision import transforms
from skimage import io
from tqdm import tqdm
import pandas as pd
import numpy as np
import os
from typing import List, Dict, Iterable

import sys
sys.path.append("..")
# from config import DATA_PATH
import platform
if platform.system() == 'Windows':
    DATA_PATH = "//172.18.36.77/datasets/datasets"
else:
    DATA_PATH = "/liaoweiduo/datasets"


class Meta(Dataset):
    def __init__(self, subset, target, preload=False):
        """Dataset class for regular train/val/test,
        background -> train
        evaluation -> test

        # Arguments:
            subset: Whether the dataset represents the background or evaluation set
            target: which dataset to represent
            preload: whether to load whole dataset into memory
        """
        if subset not in ('background', 'evaluation', 'testing'):
            raise(ValueError, 'subset must be one of (background, evaluation, testing)')
        # if target not in ('CUB_Bird', 'DTD_Texture', 'FGVC_Aircraft', 'FGVCx_Fungi'):
        #     raise(ValueError, 'target must be one of (CUB_Bird, DTD_Texture, FGVC_Aircraft, FGVCx_Fungi)')
        self.subset = subset
        self.target = target
        self.preload = preload

        # Setup transforms
        self.transform = transforms.Compose([
            transforms.Resize(84),
            transforms.ToTensor(),      # ToTensor() will normalize to [0, 1]
        ])

        info_dict, self.memory = self.index_subset(self.subset, self.target, self.preload, self.transform)
        self.df = pd.DataFrame(info_dict)

        # Index of dataframe has direct correspondence to item in dataset
        self.df = self.df.assign(id=self.df.index.values)

        # Convert arbitrary class names of dataset to ordered 0-(num_speakers - 1) integers
        self.unique_characters = sorted(self.df['class_name'].unique())     # [16]
        # ['014.Indigo_Bunting', '042.Vermilion_Flycatcher', '051.Horned_Grebe', ...]

        self.class_name_to_id = {self.unique_characters[i]: i for i in range(self.num_classes())}       # {dict: 16}
        # {'014.Indigo_Bunting': 0, '042.Vermilion_Flycatcher': 1, '051.Horned_Grebe': 2, ...}

        self.df = self.df.assign(class_id=self.df['class_name'].apply(lambda c: self.class_name_to_id[c]))
        #   class_name            filepath    subset    id   class_id         {Bird: 960}
        # 0  014.Indigo_Bunting              ...         0          0
        # 1  014.Indigo_Bunting              ...         1          0
        # 2  014.Indigo_Bunting              ...         2          0
        # 3  014.Indigo_Bunting              ...         3          0
        # 4  014.Indigo_Bunting              ...         4          0

        # Create dicts
        self.datasetid_to_filepath = self.df.to_dict()['filepath']          # {dict: 960}
        # {0: '//10.20.2.245/datasets/datasets/meta-dataset/CUB_Bird/val\\014.Indigo_Bunting\\Indigo_Bunting_0001_12469.jpg', ...}

        self.datasetid_to_class_id = self.df.to_dict()['class_id']          # {dict: 960}
        # {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, ...}

    def __getitem__(self, item):
        if self.preload:
            instance = self.memory[item]
        else:
            instance = Image.open(self.datasetid_to_filepath[item])     # JpegImageFile, 84x84
            instance = self.transform(instance)                         # [3, 84, 84]
        label = self.datasetid_to_class_id[item]                        # from 0 -> 16
        return instance, label

    def __len__(self):
        return len(self.df)

    def num_classes(self):
        return len(self.df['class_name'].unique())

    @staticmethod
    def index_subset(subset, target, preload=False, transform=None):
        """Index a subset by looping through all of its files and recording relevant information.
        if preload, store memory {Tensor: {num, 3, 84, 84}} and
                          images {list: num) -> dict{'subset', 'class_name', 'filepath'} into npy file

        # Arguments
            subset: Name of the subset

        # Returns
            A list of dicts containing information about all the image files in a particular subset of the
            miniImageNet dataset
        """
        images = []
        memory = []
        print('Indexing {}...{}...'.format(target, subset))

        # Quick first pass to find total for tqdm bar
        subset_len = 0

        # determine target's path
        if subset == 'background':
            folder_name = 'train'
        elif subset == 'evaluation':
            folder_name = 'val'
        else:
            folder_name = 'test'
        if target in ('CUB_Bird', 'DTD_Texture', 'FGVC_Aircraft', 'FGVCx_Fungi'):
            target_path_root = DATA_PATH + '/meta-dataset/{}'.format(target)
        elif target in ('clipart_84', 'infograph_84', 'painting_84', 'quickdraw_84', 'real_84', 'sketch_84'):
            target_path_root = DATA_PATH + '/DomainNet/{}'.format(target)
        else:
            target_path_root = DATA_PATH + '/{}'.format(target)

        if preload:
            if os.path.exists(target_path_root + '/{}_images_memory.npy'.format(folder_name)):
                print('{}: load {}/{}_images_memory.npy'.format(Meta, target_path_root, folder_name))
                data = torch.load(target_path_root + '/{}_images_memory.npy'.format(folder_name))
                images = data['images']
                memory = data['memory']
                return images, memory
            else:
                print('{}: load images into memory.'.format(Meta))

        target_path = target_path_root + '/{}'.format(folder_name)
        print('{}: construct images and memory from target_path: {}'.format(Meta, target_path))

        for root, folders, files in os.walk(target_path):
            subset_len += len([f for f in files if f.endswith('.jpg') or f.endswith('.JPG')])
        if subset_len == 0:
            raise Exception('image file not ended with jpg.')
        print('find {} images.'.format(subset_len))

        progress_bar = tqdm(total=subset_len)
        for root, folders, files in os.walk(target_path):
            if len(files) == 0:
                continue

            class_name = root.split(os.sep)[-1]     # linux / ; windows \\
            # 014.Indigo_Bunting

            for f in [f for f in files if f.endswith('.jpg') or f.endswith('.JPG')]:
                progress_bar.update(1)
                images.append({
                    'subset': subset,
                    'class_name': class_name,
                    'filepath': os.path.join(root, f)
                })
                # filepath: //10.20.2.245/datasets/datasets/meta-dataset/CUB_Bird/val
                #               \\014.Indigo_Bunting\\Indigo_Bunting_0001_12469.jpg

                # load memory
                if preload:
                    instance = Image.open(os.path.join(root, f))     # JpegImageFile, 84x84
                    instance = transform(instance)                   # [3, 84, 84]
                    memory.append(instance)

        progress_bar.close()

        # store npy
        if preload:
            memory = torch.stack(memory)
            print('{}: store {}/{}_images_memory.npy'.format(Meta, target_path_root, folder_name))
            state = {'images': images, 'memory': memory}
            torch.save(state, target_path_root + '/{}_images_memory.npy'.format(folder_name))

        return images, memory


class MultiDataset(Dataset):
    def __init__(self, dataset_list: List[Dataset]):
        """Dataset class representing a list of datasets

        # Arguments:
            :param dataset_list: need to first prepare each sub-dataset
        """
        self.dataset_list = dataset_list
        self.datasetid_to_class_id = self.label_mapping()

        # cat all df in dataset_list
        # e.g., CUB Bird
        #   class_name            filepath    subset    id   class_id         {Bird: 960}
        # 0  014.Indigo_Bunting              ...         0          0
        # 1  014.Indigo_Bunting              ...         1          0
        # 2  014.Indigo_Bunting              ...         2          0
        # 3  014.Indigo_Bunting              ...         3          0
        # 4  014.Indigo_Bunting              ...         4          0
        # store origin dataset_id into column[origin_dataset_id] for each df
        for idx, dataset in enumerate(dataset_list):
            dataset.df['origin_dataset_id'] = idx
        self.df = pd.concat([dataset.df for dataset in dataset_list], keys=[dataset.target for dataset in dataset_list])
        # store origin id into column[origin_id]
        self.df.rename(columns={'id': 'origin_id'}, inplace=True)
        # store origin class_id into column[origin_class_id]
        self.df.rename(columns={'class_id': 'origin_class_id'}, inplace=True)
        # update id with offset
        self.df['id'] = range(len(self.df))
        # update class_id with datasetid_to_class_id
        self.df = self.df.assign(class_id=self.df['id'].apply(lambda c: self.datasetid_to_class_id[c]))
        #               class_name   ...    origin_id   origin_class_id origin_dataset_id   id  class_id
        # CIFAR100_84/0       crab   ...            0                 0                 0    0         0
        #            /1       crab   ...            1                 0                 0    1         0
        #            /2       crab   ...            2                 0                 0    2         0
        #            /3       crab   ...            3                 0                 0    3         0
        #            /4       crab   ...            4                 0                 0    4         0
        # ...
        # CIFAR10_84 /0       bird   ...            0                 0                 1 1000       100
        #            /1       bird   ...            1                 0                 2 1001       100

        self.datasetid_to_origin_id = list(self.df['origin_id'])
        self.datasetid_to_origin_dataset_id = list(self.df['origin_dataset_id'])

    def label_mapping(self) -> Dict:
        """
        generate mapping dict from datasetid to global class id.
        :return: datasetid_to_class_id
        """
        datasetid_to_class_id = dict()
        index_offset = 0
        class_id_offset = 0
        for dataset in self.dataset_list:
            datasetid_to_class_id.update(
                dict(zip(map(lambda id:             id + index_offset,    dataset.datasetid_to_class_id.keys()),
                         map(lambda class_id: class_id + class_id_offset, dataset.datasetid_to_class_id.values())))
            )

            index_offset = index_offset + len(dataset)
            class_id_offset = class_id_offset + dataset.num_classes()

        return datasetid_to_class_id

    def __getitem__(self, item):
        # dataset_id, index = self.index_mapping(item)
        dataset_id, index = self.datasetid_to_origin_dataset_id[item], self.datasetid_to_origin_id[item]
        instance, true_label = self.dataset_list[dataset_id][index]     # true_label is the label(int) in sub-dataset
        label = self.datasetid_to_class_id[item]                        # label is the label(int) with offset
        return instance, label

    def __len__(self):
        return sum([len(dataset) for dataset in self.dataset_list])

    def num_classes(self):
        sum([dataset.num_classes() for dataset in self.dataset_list])

    def index_mapping(self, index) -> (int, int):
        """
        A mapping method to map index (in __getitem__ method) to the index in the corresponding dataset.

        :param index:
        :return: dataset_id, item
        """
        index_origin = index
        for dataset_id, dataset in enumerate(self.dataset_list):
            if index < len(dataset):
                return dataset_id, index
            else:
                index = index - len(dataset)

        raise(ValueError, f'index exceeds total number of instances, index {index_origin}')


class NShotTaskSampler(Sampler):
    def __init__(self,
                 dataset: torch.utils.data.Dataset,
                 episodes_per_epoch: int = None,
                 n: int = None,
                 k: int = None,
                 q: int = None,
                 num_tasks: int = 1,
                 fixed_tasks: List[Iterable[int]] = None):
        """PyTorch Sampler subclass that generates batches of n-shot, k-way, q-query tasks.

        Each n-shot task contains a "support set" of `k` sets of `n` samples and a "query set" of `k` sets
        of `q` samples. The support set and the query set are all grouped into one Tensor such that the first n * k
        samples are from the support set while the remaining q * k samples are from the query set.

        The support and query sets are sampled such that they are disjoint i.e. do not contain overlapping samples.

        # Arguments
            dataset: Instance of torch.utils.data.Dataset from which to draw samples
            episodes_per_epoch: Arbitrary number of batches of n-shot tasks to generate in one epoch
            n_shot: int. Number of samples for each class in the n-shot classification tasks.
            k_way: int. Number of classes in the n-shot classification tasks.
            q_queries: int. Number query samples for each class in the n-shot classification tasks.
            num_tasks: Number of n-shot tasks to group into a single batch
            fixed_tasks: If this argument is specified this Sampler will always generate tasks from
                the specified classes
        """
        super(NShotTaskSampler, self).__init__(dataset)
        self.episodes_per_epoch = episodes_per_epoch
        self.dataset = dataset
        if num_tasks < 1:
            raise ValueError('num_tasks must be > 1.')

        self.num_tasks = num_tasks
        # TODO: Raise errors if initialise badly
        self.k = k
        self.n = n
        self.q = q
        self.fixed_tasks = fixed_tasks
        self.classes = self.dataset.df['class_id'].unique()

        self.i_task = 0     # count yielded tasks

    def __len__(self):
        return self.episodes_per_epoch

    def __iter__(self):
        # print('np', np.random.randint(1, 100, size=1), 'torch', torch.randint(1, 100, size=(1,)))
        # 这里np和torch的seed都是fix的

        # print(self.dataset.df['class_id'].unique())
        # print(np.random.choice(self.dataset.df['class_id'].unique(), size=self.k, replace=False))

        for _ in range(self.episodes_per_epoch):
            batch = []

            for task in range(self.num_tasks):
                if self.fixed_tasks is None:
                    # Get random classes
                    episode_classes = np.random.choice(self.classes, size=self.k, replace=False)
                else:
                    # Loop through classes in fixed_tasks
                    episode_classes = self.fixed_tasks[self.i_task % len(self.fixed_tasks)]
                    self.i_task += 1

                df = self.dataset.df[self.dataset.df['class_id'].isin(episode_classes)]

                support_k = {k: None for k in episode_classes}
                for k in episode_classes:
                    # Select support examples
                    support = df[df['class_id'] == k].sample(self.n)
                    support_k[k] = support

                    for i, s in support.iterrows():
                        batch.append(s['id'])

                for k in episode_classes:
                    query = df[(df['class_id'] == k) & (~df['id'].isin(support_k[k]['id']))].sample(self.q)
                    for i, q in query.iterrows():
                        batch.append(q['id'])

            # yield the indexs of (img,label) samples for a batch
            # batch contain id in df, total size (n*k + q*k)
            # then using this id as the items in Dataset.__getitem__
            yield np.stack(batch)


class NShotTaskSamplerMultiDomain(Sampler):
    def __init__(self,
                 dataset: MultiDataset,
                 episodes_per_epoch: int = None,
                 n: int = None,
                 k: int = None,
                 q: int = None,
                 num_tasks: int = 1,
                 fixed_tasks: List[Iterable[int]] = None):
        """PyTorch Sampler subclass that generates batches of n-shot, k-way, q-query tasks.

        Each n-shot task contains a "support set" of `k` sets of `n` samples and a "query set" of `k` sets
        of `q` samples. The support set and the query set are all grouped into one Tensor such that the first n * k
        samples are from the support set while the remaining q * k samples are from the query set.

        The support and query sets are sampled such that they are disjoint i.e. do not contain overlapping samples.

        For multi domain, you must use MultiDataset whose df contains origin_dataset_id key

        # Arguments
            dataset: Instance of torch.utils.data.Dataset from which to draw samples
            episodes_per_epoch: Arbitrary number of batches of n-shot tasks to generate in one epoch
            n_shot: int. Number of samples for each class in the n-shot classification tasks.
            k_way: int. Number of classes in the n-shot classification tasks.
            q_queries: int. Number query samples for each class in the n-shot classification tasks.
            num_tasks: Number of n-shot tasks to group into a single batch
            fixed_tasks: If this argument is specified this Sampler will always generate tasks from
                the specified classes
        """
        super(NShotTaskSamplerMultiDomain, self).__init__(dataset)
        self.episodes_per_epoch = episodes_per_epoch
        self.dataset = dataset
        if num_tasks < 1:
            raise ValueError('num_tasks must be > 1.')

        self.num_tasks = num_tasks
        # TODO: Raise errors if initialise badly
        self.k = k
        self.n = n
        self.q = q
        self.fixed_tasks = fixed_tasks

        self.domains = list(range(len(self.dataset.dataset_list)))      # self.dataset.df['origin_dataset_id'].unique()

        self.i_task = 0     # count yielded tasks

    def __len__(self):
        return self.episodes_per_epoch

    def __iter__(self):
        for _ in range(self.episodes_per_epoch):
            batch = []

            for task in range(self.num_tasks):
                if self.fixed_tasks is None:
                    # Get a random domain
                    domain = np.random.choice(self.domains)
                    # Get random classes in the specific domain, note that class_id is origin_class_id
                    episode_classes = np.random.choice(
                        self.dataset.dataset_list[domain].df['class_id'].unique(), size=self.k, replace=False)
                    # filter df for chosen episode_classes
                    df = self.dataset.dataset_list[domain].df[
                        self.dataset.dataset_list[domain].df['class_id'].isin(episode_classes)]
                else:
                    # Loop through classes in fixed_tasks
                    episode_classes = self.fixed_tasks[self.i_task % len(self.fixed_tasks)]
                    self.i_task += 1
                    df = self.dataset.df[self.dataset.df['class_id'].isin(episode_classes)]

                support_k = {k: None for k in episode_classes}
                for k in episode_classes:
                    # Select support examples
                    support = df[df['class_id'] == k].sample(self.n)
                    support_k[k] = support

                    for i, s in support.iterrows():
                        batch.append(s['id'])

                for k in episode_classes:
                    query = df[(df['class_id'] == k) & (~df['id'].isin(support_k[k]['id']))].sample(self.q)
                    for i, q in query.iterrows():
                        batch.append(q['id'])

            # yield the indexs of (img,label) samples for a batch
            # batch contain id in df, total size (n*k + q*k)
            # then using this id as the items in Dataset.__getitem__
            yield np.stack(batch)


def create_nshot_task_label(k: int, q: int) -> torch.Tensor:
    """Creates an n-shot task label.

    Label has the structure:
        [0]*q + [1]*q + ... + [k-1]*q

    # TODO: Test this

    # Arguments
        k: Number of classes in the n-shot classification task
        q: Number of query samples for each class in the n-shot classification task

    # Returns
        y: Label vector for n-shot task of shape [q * k, ]
    """
    y = torch.arange(0, k, 1 / q).long()
    return y


class DatasetReader:
    """
    Class that wraps the dataloader with NShotTaskSamplerMultiDomain
    """
    def __init__(self, data_path, mode, train_set, validation_set, test_set, max_way_train, max_way_test, max_support_train, max_support_test, shuffle=True):

        self.shuffle = shuffle                  # False
        self.data_path = data_path              # "../datasets"
        self.validation_set_dict = {}
        self.test_set_dict = {}

        preload = True
        if platform.system() == 'Windows':
            num_workers = 0
        else:
            num_workers = 1
        if mode == 'train' or mode == 'train_test':
            self.n_shot_train = int(max_support_train / max_way_train)      # 1 = 5/5
            self.k_way_train  = max_way_train      # 5
            self.q_query_train= 15

            # train_set = ['CUB_Bird', 'DTD_Texture', 'FGVC_Aircraft', 'FGVCx_Fungi']
            background = MultiDataset([
                Meta('background', train_dataset, preload=preload)
                for train_dataset in train_set
            ])

            background_taskloader = DataLoader(
                background,
                batch_sampler=NShotTaskSamplerMultiDomain(background, episodes_per_epoch=1,
                                                          n=self.n_shot_train, k=self.k_way_train, q=self.q_query_train,
                                                          num_tasks=1),
                num_workers=num_workers)
            self.train_dataset_next_task = background_taskloader

            self.n_shot_val = int(max_support_test / max_way_test)      # 1 = 5/5
            self.k_way_val  = max_way_test       # 5
            self.q_query_val= 15

            for item in validation_set:
                evaluation = Meta('evaluation', item, preload=preload)
                evaluation_taskloader = DataLoader(
                    evaluation,
                    batch_sampler=NShotTaskSampler(evaluation, episodes_per_epoch=1,
                                                   n=self.n_shot_val, k=self.k_way_val, q=self.q_query_val,
                                                   num_tasks=1),
                    num_workers=num_workers)
                self.validation_set_dict[item] = evaluation_taskloader

        if mode == 'test' or mode == 'train_test':
            self.n_shot_test = int(max_support_test / max_way_test)      # 1 = 5/5
            self.k_way_test  = max_way_test       # 5
            self.q_query_test= 15

            for item in test_set:
                evaluation = Meta('testing', item, preload=preload)
                testing_taskloader = DataLoader(
                    evaluation,
                    batch_sampler=NShotTaskSampler(evaluation, episodes_per_epoch=1,
                                                   n=self.n_shot_test, k=self.k_way_test, q=self.q_query_test,
                                                   num_tasks=1),
                    num_workers=num_workers)
                self.test_set_dict[item] = testing_taskloader

    def _get_task(self, next_task, n_shot, k_way, q_query):     # need to be np
        for batch_index, batch in enumerate(next_task):
            x, y = batch
            # Reshape to `meta_batch_size` number of tasks. Each task contains
            # n*k support samples to train the fast model on and q*k query samples to
            # evaluate the fast model on and generate meta-gradients
            context_images = x[:n_shot * k_way]
            target_images  = x[n_shot * k_way:]

            # Create label
            context_labels = create_nshot_task_label(k_way, n_shot)
            target_labels  = create_nshot_task_label(k_way, q_query)

            task_dict = {
                'context_images': context_images.numpy(),
                'context_labels': context_labels.numpy(),
                'target_images': target_images.numpy(),
                'target_labels': target_labels.numpy()
                }

            return task_dict

    def get_train_task(self):
        return self._get_task(self.train_dataset_next_task, self.n_shot_train, self.k_way_train, self.q_query_train)

    def get_validation_task(self, item):
        return self._get_task(self.validation_set_dict[item], self.n_shot_val, self.k_way_val, self.q_query_val)

    def get_test_task(self, item):
        return self._get_task(self.test_set_dict[item], self.n_shot_test, self.k_way_test, self.q_query_test)


if __name__ == "__main__":
    train_set = ['CUB_Bird', 'DTD_Texture', 'FGVC_Aircraft', 'FGVCx_Fungi']
    validation_set = ['CUB_Bird', 'DTD_Texture', 'FGVC_Aircraft', 'FGVCx_Fungi']
    test_set = ["FGVC_Aircraft", "CUB_Bird", "DTD_Texture", "FGVCx_Fungi",
                "Omniglot_84", "VGG_Flower_84", "traffic_sign_84", "mscoco_84", "mini_84",
                "clipart_84", "infograph_84", "painting_84", "quickdraw_84", "real_84", "sketch_84",
                "CIFAR10_84", "CIFAR100_84", "cars_84", "pets_head_84", "dogs_84"]
    metadataset = DatasetReader(data_path="../../datasets", mode="train_test",
                                train_set=train_set, validation_set=validation_set, test_set=test_set,
                                max_way_train=5, max_way_test=5, max_support_train=5, max_support_test=5, shuffle=False)

    batch = metadataset.get_train_task()

    from matplotlib import pyplot as plt
    plt.imshow(np.transpose(batch['context_images'][0], (1, 2, 0)))
    plt.show()
    plt.imshow(np.transpose(batch['context_images'][1], (1, 2, 0)))
    plt.show()
    plt.imshow(np.transpose(batch['context_images'][2], (1, 2, 0)))
    plt.show()
    plt.imshow(np.transpose(batch['context_images'][3], (1, 2, 0)))
    plt.show()
    plt.imshow(np.transpose(batch['context_images'][4], (1, 2, 0)))
    plt.show()
    plt.imshow(np.transpose(batch['target_images'][0], (1, 2, 0)))
    plt.show()
    plt.imshow(np.transpose(batch['target_images'][15], (1, 2, 0)))
    plt.show()
    plt.imshow(np.transpose(batch['target_images'][30], (1, 2, 0)))
    plt.show()
    plt.imshow(np.transpose(batch['target_images'][45], (1, 2, 0)))
    plt.show()
    plt.imshow(np.transpose(batch['target_images'][60], (1, 2, 0)))
    plt.show()