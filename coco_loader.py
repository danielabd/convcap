import glob
import math
import numpy as np
import os
import os.path as osp
import string 
import pickle
import json

from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.datasets as datasets
import torchvision.transforms as transforms

class Scale(object):
  """Scale transform with list as size params"""

  def __init__(self, size, interpolation=Image.BILINEAR):
    self.size = size
    self.interpolation = interpolation

  def __call__(self, img):
    return img.resize((self.size[1], self.size[0]), self.interpolation)

class coco_loader(Dataset):
  """Loads train/val/test splits of coco dataset"""

  def __init__(self, coco_root, split='train', max_tokens=15, ncap_per_img=5):
    self.max_tokens = max_tokens
    self.ncap_per_img = ncap_per_img
    self.coco_root = coco_root
    self.split = split
    #Splits from http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip
    self.get_split_info('data/dataset_coco.json')

    worddict_tmp = pickle.load(open('data/wordlist.p', 'rb'))
    wordlist = [l for l in iter(worddict_tmp.keys()) if l != '</S>']
    self.wordlist = ['EOS'] + sorted(wordlist)
    self.numwords = len(self.wordlist)
    print('[DEBUG] #words in wordlist: %d' % (self.numwords))

    self.img_transforms = transforms.Compose([
      Scale([224, 224]),
      transforms.ToTensor(),
      transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ], 
        std = [ 0.229, 0.224, 0.225 ])
      ])

  def get_split_info(self, split_file):
    '''
    this function creates self.annos(dict: key=img id, val=item object form the json file which for each image there is describing), self.ids = list of img ids.
    :param split_file: full path file for describing the images
    :return:
    '''
    print('Loading annotation file...')
    with open(split_file) as fin:
      split_info = json.load(fin)
    annos = {}
    for item in split_info['images']:
      # todo: daniela changes for succeed to run in little images. remain only relevant images in train
      #if not(str(item['filename'])=='COCO_train2014_000000000009.jpg' or \
      #        str(item['filename']) == 'COCO_train2014_000000000025.jpg' or \
      #        str(item['filename']) == 'COCO_train2014_000000000030.jpg' or \
      #        str(item['filename']) == 'COCO_val2014_000000000074.jpg' or \
      #          str(item['filename']) == 'COCO_val2014_000000000073.jpg' or \
      #        str(item['filename']) == 'COCO_val2014_000000000042.jpg' \
      #        ):
      #  continue
      #print('str(item[filename])='+str(item['filename']))
      if self.split == 'train':
        if item['split'] == 'train' or item['split'] == 'restval':
          annos[item['cocoid']] = item
      elif item['split'] == self.split:
        annos[item['cocoid']] = item
    self.annos = annos
    self.ids = list(self.annos.keys())
    print('Found %d images in split: %s'%(len(self.ids), self.split))

  def __getitem__(self, idx):
    img_id = self.ids[idx]
    anno = self.annos[img_id]

    # list of all sentences describing this image
    captions = [caption['raw'] for caption in anno['sentences']]

    # imgpath = '%s/%s/%s'%(self.coco_root, anno['filepath'], anno['filename']) #todo from daniela:source code
    imgpath = os.path.join(self.coco_root, anno['filepath'], anno['filename'])

    img = Image.open(os.path.join(imgpath)).convert('RGB')
    img = self.img_transforms(img)

    if(self.split != 'train'):
      r = np.random.randint(0, len(captions))
      captions = [captions[r]]

    if(self.split == 'train'):
      if(len(captions) > self.ncap_per_img):
        ids = np.random.permutation(len(captions))[:self.ncap_per_img]
        captions_sel = [captions[l] for l in ids]
        captions = captions_sel
      assert(len(captions) == self.ncap_per_img)

    wordclass = torch.LongTensor(len(captions), self.max_tokens).zero_()
    sentence_mask = torch.ByteTensor(len(captions), self.max_tokens).zero_()

    for i, caption in enumerate(captions):
      #conver sentence to list of words without punctuation
      words = str(caption).lower().translate(None, string.punctuation).strip().split()
      words = ['<S>'] + words
      num_words = min(len(words), self.max_tokens-1)
      sentence_mask[i, :(num_words+1)] = 1
      for word_i, word in enumerate(words):
        if(word_i >= num_words):
          break
        if(word not in self.wordlist):
          word = 'UNK'
        #wordclass is a matrix: each row represent caption num  of this image, column represents place in the sentenc, value contain the index of this word
        wordclass[i, word_i] = self.wordlist.index(word)
    # img - the img
    # captions - all sentences of this image. if not train - single class
    # wordclass - matrix: each row represent caption num  of this image,
    #                     column represents word idx in the sentence,
    #                     value contain the index of this word from the dict
    # sentence_mask - vector which contain 1 until the max words we need to relate to it
    # img_id - img id uniqe for image
    return img, captions, wordclass, sentence_mask, img_id

  def __len__(self):
    return len(self.ids)
