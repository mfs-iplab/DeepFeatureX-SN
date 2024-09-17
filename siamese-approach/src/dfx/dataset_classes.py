import torch
import torchvision.transforms as T
import pandas as pd
import numpy as np
import os, random
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, Subset


class mydataset(Dataset):
  def __init__(self, dset_dir, guidance, main_class=None, for_basemodel=True, for_testing=False, transforms=T.Compose([])):
    self.dset_dir = Path(dset_dir)
    self.transforms = transforms
    self.files = []
    labels = pd.read_csv(guidance)
    models = sorted(os.listdir(self.dset_dir))
    n=0 if for_basemodel else 1
    if for_testing: n=2
    for model_name in models:
      if for_basemodel:
        assert main_class in models
        class_idx=1 if model_name==main_class else 0
      else:
        class_idx = models.index(model_name) # model class
      model_path = os.path.join(self.dset_dir, model_name)
      architectures = sorted(os.listdir(model_path))
      for architecture_name in architectures:
        class_idx2 = architectures.index(architecture_name) # architecture class
        architecture_path = os.path.join(model_path, architecture_name)         
        for image in os.listdir(architecture_path):
          image_path = os.path.join(architecture_path, image)
          image_dir = image_path.split('datasets')[1]
          if np.array(labels[labels['image_path']==image_dir])[0,0]==n:
            self.files += [{"file": image_path, "class_mod": class_idx, "class_arch": class_idx2}]
          else:
            continue
  def __len__(self):
    return len(self.files)
  def __getitem__(self, i):
    item = self.files[i]
    file = item['file']
    class_mod = torch.tensor(item['class_mod'])
    class_arch = torch.tensor(item['class_arch'])
    img = Image.open(file).convert("RGB")
    img = self.transforms(img)
    return img, class_mod, class_arch


class myaugdataset(Dataset):
    def __init__(self, dset_dir, augdset_dir, guidance_path, percentage_augmented=0.5, for_basemodel=False, main_class=None, transforms=None):
        self.dset_dir = Path(dset_dir)
        self.augdset_dir = Path(augdset_dir)
        self.guidance = pd.read_csv(guidance_path)
        self.transforms = transforms or T.Compose([])
        self.perc = percentage_augmented
        self.n = 0 if for_basemodel else 1
        self.files = []
        
        self._take_from_dset()
        self._take_from_augdset()

        random.shuffle(self.files)

    def _take_from_dset(self):
        engines = list(self.dset_dir.iterdir())

        total_images = int(len(self.guidance[self.guidance['label']==self.n]) * self.perc) + 1
        pbar = tqdm(total=total_images, desc="raw images")
        
        for class_idx, engine in enumerate(engines):
            if self.n == 0: class_idx = int(engine.name == main_class)
            models = sorted(engine.iterdir())
            for class_idx2, model in enumerate(models):
                images = list(model.iterdir())
                random.shuffle(images)
                sample_size = int(len(images) * self.perc)
                
                for image_path in random.sample(images, sample_size):
                    image_dir = '/' + '/'.join(image_path.parts[-3:])
                    if self.guidance[self.guidance['image_path'] == image_dir].iloc[0, 0] == self.n:
                        self.files.append({
                            "file": image_path,
                            "class_mod": class_idx,
                            "class_arch": class_idx2
                        })
                        pbar.update(1)
        
    def _take_from_augdset(self):        
        attacks = list(self.augdset_dir.iterdir())
        total_iterations = int(len(self.guidance[self.guidance['label']==self.n]) * (1 - self.perc))
        
        pbar = tqdm(total=total_iterations, desc="augmented images")
        
        new_files = []
        for _ in range(total_iterations):
            try:
                attack = random.choice(attacks)
                engines = sorted(list(attack.iterdir()))
                engine = random.choice(engines)
                if self.n == 0:
                    class_idx = int(engine.name == main_class)
                else:
                    class_idx = engines.index(engine)
                models = sorted(list(engine.iterdir()))
                model = random.choice(models)
                class_idx2 = models.index(model)
                images = list(model.iterdir())
                image_path = random.choice(images)
                image_dir = '/' + '/'.join(image_path.parts[-3:])
                if self.guidance[self.guidance['image_path'] == image_dir].iloc[0, 0] == self.n:
                    self.files.append({
                        "file": image_path,
                        "class_mod": class_idx,
                        "class_arch": class_idx2
                    })
                    pbar.update(1)
            except:
                continue
    def __len__(self):
        return len(self.files)
    def __getitem__(self, i):
        item = self.files[i]
        file = item['file']
        class_mod = torch.tensor(item['class_mod'])
        class_arch = torch.tensor(item['class_arch'])
        img = Image.open(file).convert("RGB")
        img = self.transforms(img)
        return img, class_mod, class_arch


class pair_dset(Dataset):
  def __init__(self, dset_dir, guidance, main_class=None, for_basemodel=True, for_testing=False, transforms=T.Compose([])):
    self.dset_dir = Path(dset_dir)
    self.transforms = transforms
    self.files = []
    labels = pd.read_csv(guidance)
    models = sorted(os.listdir(self.dset_dir))
    n=0 if for_basemodel else 1
    if for_testing: n=2
    for model_name in models:
      if for_basemodel:
        assert main_class in models
        class_idx=1 if model_name==main_class else 0
      else:
        class_idx = models.index(model_name) # model class
      model_path = os.path.join(self.dset_dir, model_name)
      architectures = sorted(os.listdir(model_path))
      for architecture_name in architectures:
        class_idx2 = architectures.index(architecture_name) # architecture class
        architecture_path = os.path.join(model_path, architecture_name)         
        for image in os.listdir(architecture_path):
          image_path = os.path.join(architecture_path, image)
          image_dir = image_path.split('datasets')[1]
          if np.array(labels[labels['image_path']==image_dir])[0,0]==n:
            self.files += [{"file": image_path, "class_mod": class_idx, "class_arch": class_idx2}]
          else:
            continue
    self.idx_list = [[], []] if for_basemodel else [[], [], []]
    for idx, item in enumerate(self.files):
      class_mod = item['class_mod']
      self.idx_list[class_mod].append(idx)   
    self.generate_pairs()
  def generate_pairs(self):
     self.pair_labels = (torch.rand(len(self.files)) > 0.5).long()
     self.paired_idx = []
     for idx, label in enumerate(self.pair_labels):
        c1 = self.files[idx]['class_mod']
        if label==0:
           j = np.random.choice(self.idx_list[c1])
        else:
           diff_class = np.random.choice(list(set(range(len(self.idx_list))) - {c1}))
           j = np.random.choice(self.idx_list[diff_class])
        self.paired_idx.append(j)
  def __len__(self):
     return len(self.files)
  def __getitem__(self, i):
    item = self.files[i]
    paired_item = self.files[self.paired_idx[i]]
    img, klass = self.transforms(Image.open(item['file']).convert("RGB")), torch.tensor(item['class_mod'])
    paired_img, paired_klass = self.transforms(Image.open(paired_item['file']).convert("RGB")), torch.tensor(paired_item['class_mod'])
    pair_label = self.pair_labels[i]
    return img, paired_img, klass, paired_klass, pair_label 


class triplet_dset(Dataset):
  def __init__(self, dset_dir, guidance, main_class=None, for_basemodel=True, for_testing=False, transforms=T.Compose([])):
    self.dset_dir = Path(dset_dir)
    self.transforms = transforms
    self.files = []
    labels = pd.read_csv(guidance)
    models = sorted(os.listdir(self.dset_dir))
    n=0 if for_basemodel else 1
    if for_testing: n=2
    for model_name in models:
      if for_basemodel:
        assert main_class in models
        class_idx=1 if model_name==main_class else 0
      else:
        class_idx = models.index(model_name) # model class
      model_path = os.path.join(self.dset_dir, model_name)
      architectures = sorted(os.listdir(model_path))
      for architecture_name in architectures:
        class_idx2 = architectures.index(architecture_name) # architecture class
        architecture_path = os.path.join(model_path, architecture_name)         
        for image in os.listdir(architecture_path):
          image_path = os.path.join(architecture_path, image)
          image_dir = image_path.split('datasets')[1]
          if np.array(labels[labels['image_path']==image_dir])[0,0]==n:
            self.files += [{"file": image_path, "class_mod": class_idx, "class_arch": class_idx2}]
          else:
            continue
    self.idx_list = [[], []] if for_basemodel else [[], [], []]
    for idx, item in enumerate(self.files):
      class_mod = item['class_mod']
      self.idx_list[class_mod].append(idx)
    self.generate_triplets()
  def generate_triplets(self):
    self.similar_idx = []
    self.dissimilar_idx = []
    for i in range(len(self.files)):
      c1 = self.files[i]['class_mod']
      j = np.random.choice(self.idx_list[c1])
      diff_class = np.random.choice(list(set(range(len(self.idx_list))) - {c1}))
      k = np.random.choice(self.idx_list[diff_class])
      self.similar_idx.append(j)
      self.dissimilar_idx.append(k)
  def __len__(self):
    return len(self.files)
  def __getitem__(self, idx):
    item1 = self.files[idx]
    item2 = self.files[self.similar_idx[idx]]
    item3 = self.files[self.dissimilar_idx[idx]]
    img1, l1 = self.transforms(Image.open(item1['file']).convert("RGB")), torch.tensor(item1['class_mod'])
    img2, l2 = self.transforms(Image.open(item2['file']).convert("RGB")), torch.tensor(item2['class_mod'])
    img3, l3 = self.transforms(Image.open(item3['file']).convert("RGB")), torch.tensor(item3['class_mod'])
    return img1, img2, img3, l1, l2, l3


class dataset_for_robustness(Dataset):
  def __init__(self, dset_dir, transforms=T.Compose([])):
    self.dset_dir = Path(dset_dir)
    self.transforms = transforms
    self.files = []
    models = sorted(os.listdir(self.dset_dir))
    for model_name in models:
      class_idx = models.index(model_name) # model class
      model_path = os.path.join(self.dset_dir, model_name)
      architectures = sorted(os.listdir(model_path))
      for architecture_name in architectures:
        class_idx2 = architectures.index(architecture_name) # architecture class
        architecture_path = os.path.join(model_path, architecture_name)         
        for image in os.listdir(architecture_path):
          image_path = os.path.join(architecture_path, image)
          self.files += [{"file": image_path, "class_mod": class_idx, "class_arch": class_idx2}]
  def __len__(self):
    return len(self.files)
  def __getitem__(self, i):
    item = self.files[i]
    file = item['file']
    class_mod = torch.tensor(item['class_mod'])
    class_arch = torch.tensor(item['class_arch'])
    img = Image.open(file).convert("RGB")
    img = self.transforms(img)
    return img, class_mod, class_arch


class dataset_for_generaization(Dataset):
  def __init__(self, dset_dir, transforms=T.Compose([])):
    self.dset_dir = Path(dset_dir)
    self.transforms = transforms
    self.files = []
    classes = sorted(os.listdir(self.dset_dir))
    for fold in classes:
      if fold=='0_real': klass=2
      if fold=='1_fake': klass=1
      images_path = os.path.join(self.dset_dir, fold)
      for image in os.listdir(images_path):
        image_path = os.path.join(images_path, image)
        self.files += [{"file": image_path, "class_mod": klass}]
  def __len__(self):
    return len(self.files)
  def __getitem__(self, i):
    item = self.files[i]
    file = item['file']
    class_mod = torch.tensor(item['class_mod'])
    img = Image.open(file).convert("RGB")
    img = self.transforms(img)
    return img, class_mod, 1


def check_len(dset, binary:bool=False, return_perc=False):
  print(f'\nlength dataset: {len(dset)}')
  if not binary:
    dm, gan,real = 0,0,0
    for item in dset.files:
      if item['class_mod']==0: dm += 1 
      if item['class_mod']==1: gan +=1 
      if item['class_mod']==2: real+=1
    perc_dm, perc_gan, perc_real = dm/len(dset), gan/len(dset), real/len(dset)
    print(f'perc_dms: {perc_dm} \nperc_gans: {perc_gan} \nperc_real: {perc_real}')
    if return_perc:
      return perc_dm, perc_gan, perc_real
  else:
    main, others = 0,0
    for item in dset.files:
      if item['class_mod']==1: main += 1 
      if item['class_mod']==0: others +=1 
    perc_main, perc_others = main/len(dset), others/len(dset)
    print(f'perc_main: {perc_main} \nperc_others: {perc_others}')
    if return_perc:
        return perc_main, perc_others


def make_train_valid(dset, validation_ratio=0.2):
  class_counts = [0, 0, 0]
  index_lists = [[], [], []]
  for idx, item in enumerate(dset.files):
    class_mod = item['class_mod']
    class_counts[class_mod] += 1
    index_lists[class_mod].append(idx)
  min_count = min(class_counts)
  num_per_class_valid = int(validation_ratio * min_count)
  valid_indices = []
  for indices in index_lists:
    valid_indices.extend(random.sample(indices, num_per_class_valid))
  all_indices = set(range(len(dset.files)))
  train_indices = list(all_indices - set(valid_indices))
  train_dset = Subset(dset, train_indices)
  valid_dset = Subset(dset, valid_indices)
  return train_dset, valid_dset


def get_counts_idx(dset, binary=False):
  class_counts = [0, 0] if binary else [0, 0, 0]
  index_lists = [[], []] if binary else [[], [], []]
  for idx, item in enumerate(dset.files):
    class_mod = item['class_mod']
    class_counts[class_mod] += 1
    index_lists[class_mod].append(idx)
  return class_counts, index_lists


def make_balanced(dset, binary=False):
    class_counts, index_lists = get_counts_idx(dset, binary=binary)
    min_count = min(class_counts)
    _indices = []
    for indices in index_lists:
        _indices.extend(random.sample(indices, min_count))
    _dset = Subset(dset, _indices)
    return _dset


def balance_binary_test(testing_dset):
    class_counts = [0, 0]
    index_lists = [[], []]
    for idx, item in enumerate(testing_dset.files):
      if item['class_mod']==0: item['class_mod'] +=1
      item['class_mod'] -= 1 # 0:deepfake and 1:real
      class_mod = item['class_mod']
      class_counts[class_mod] += 1
      index_lists[class_mod].append(idx)
    real_count=class_counts[1]
    binary_test_index = []
    for indices in index_lists:
       binary_test_index.extend(random.sample(indices, real_count))
    test_dset = Subset(testing_dset, binary_test_index)
    return test_dset


def make_binary(testing_dset):
  index_list=[]
  for idx, item in enumerate(testing_dset.files):
    if item['class_mod']==0: item['class_mod'] +=1
    item['class_mod'] -= 1 # 0:deepfake and 1:real
    class_mod = item['class_mod']
    index_list.append(idx)
  test_dset = Subset(testing_dset, index_list)
  return test_dset


def get_trans(model_name:str, transformations:list | None = []):
    if transformations:
      trans = transformations
    else:
      trans = []
    if model_name.startswith('vit'):
      base_trans = [
        T.Resize((256, 256)),
        T.CenterCrop((224,224))            
      ]
    else:
      base_trans = [
        T.Resize((256, 256))
      ]
    trans.extend(base_trans)
    last_trans = [
      T.ToTensor(),
      T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
    trans.extend(last_trans)
    trans = T.Compose(trans)
    return trans