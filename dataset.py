import torch
import torchvision.transforms as transforms

from PIL import Image
import json
import numpy as np
import os
import random
identity = lambda x: x


def get_transform(image_size, normalize_param, aug):
  if aug:
    transform = transforms.Compose([transforms.RandomResizedCrop(image_size),
                                    transforms.ImageJitter(Brightness=0.4, Contrast=0.4, Color=0.4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize(**normalize_param)])
  else:
    transform = transforms.Compose([transforms.Resize([int(image_size*1.15), int(image_size*1.15)]),
                                    transforms.CenterCrop(image_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize(**normalize_param)])

  return transform


class SubDataset:
  def __init__(self, sub_meta, cl, transform=transforms.ToTensor(), target_transform=identity, min_size=50):
    self.sub_meta = sub_meta
    self.cl = cl
    self.transform = transform
    self.target_transform = target_transform
    if len(self.sub_meta) < min_size:
      idxs = [i % len(self.sub_meta) for i in range(min_size)]
      self.sub_meta = np.array(self.sub_meta)[idxs].tolist()

  def __getitem__(self, i):
    image_path = os.path.join(self.sub_meta[i])
    img = Image.open(image_path).convert('RGB')
    img = self.transform(img)
    target = self.target_transform(self.cl)
    return img, target

  def __len__(self):
    return len(self.sub_meta)


class SetDataset:
  def __init__(self, data_file, batch_size, transform):
    with open(data_file, 'r') as f:
      self.meta = json.load(f)
    self.cl_list = np.unique(self.meta['image_labels']).tolist()

    self.sub_meta = {}
    for cl in self.cl_list:
      self.sub_meta[cl] = []

    for x, y in zip(self.meta['image_names'], self.meta['image_labels']):
      self.sub_meta[y].append(x)

    self.sub_dataloader = []
    for cl in self.cl_list:
      sub_dataset = SubDataset(self.sub_meta[cl], cl, transform=transform)
      self.sub_dataloader.append(torch.utils.data.DataLoader(sub_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False))

  def __getitem__(self, i):
    return next(iter(self.sub_dataloader[i]))

  def __len__(self):
    return len(self.cl_list)


class EpisodicBatchSampler(object):
  def __init__(self, n_classes, n_way, n_episodes):
    self.n_classes = n_classes
    self.n_way = n_way
    self.n_episodes = n_episodes

  def __len__(self):
    return self.n_episodes

  def __iter__(self):
    for i in range(self.n_episodes):
      yield torch.randperm(self.n_classes)[:self.n_way]


class SetDataManager:
  def __init__(self, image_size, n_way, n_support, n_query, n_eposide=100):
    super(SetDataManager, self).__init__()
    self.image_size = image_size
    self.n_way = n_way
    self.batch_size = n_support + n_query
    self.n_eposide = n_eposide

  def get_data_loader(self, data_file, normalize_param=dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), aug=False):
    transform = get_transform(self.image_size, normalize_param, aug)
    dataset = SetDataset(data_file, self.batch_size, transform)
    sampler = EpisodicBatchSampler(len(dataset), self.n_way, self.n_eposide)
    data_loader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler, num_workers=12)
    return data_loader