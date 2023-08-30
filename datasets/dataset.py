import os
import os.path as osp
import sys
import random
import math
import numpy as np
import torch
import pickle
import PIL
from PIL import Image
import io
import json
import string

from torch.utils.data import Dataset

from .utils import convert_examples_to_features, read_examples
from util.box_ops import box_xyxy_to_cxcywh, box_cxcywh_to_xyxy
from pytorch_pretrained_bert.tokenization import BertTokenizer
from .transforms import PIL_TRANSFORMS


# Meta Information
SUPPORTED_DATASETS = {
    'kbref': {
            'splits': ('train', 'val', 'test'),
            'params': {'dataset': 'kbref', 'split_by': ''}
    }
}


class VGDataset(Dataset):
    def __init__(self, data_root, split_root='data', dataset='kbref', transforms=[],
                 debug=False, test=False, split='train', max_query_len=128,
                 bert_mode='bert-base-uncased', cache_images=False, know_cate_num=3, know_cate_len=4, know_retrieval_results=''):
        super(VGDataset, self).__init__()

        self.data_root = data_root
        self.split_root = split_root
        self.dataset = dataset
        self.test = test
        self.transforms = []

        self.getitem = self.getitem__PIL
        self.read_image = self.read_image_from_path_PIL
        for t in transforms:
            _args = t.copy()
            self.transforms.append(PIL_TRANSFORMS[_args.pop('type')](**_args))

        self.debug = debug

        self.query_len = max_query_len
        self.tokenizer = BertTokenizer.from_pretrained(bert_mode, do_lower_case=True)

        # setting datasource
        self.dataset_root = self.data_root
        self.im_dir = osp.join(self.dataset_root, 'images', 'kbref')

        dataset_split_root = osp.join(self.split_root, self.dataset)
        valid_splits = SUPPORTED_DATASETS[self.dataset]['splits']

        if split not in valid_splits:
            raise ValueError(
                'Dataset {0} does not have split {1}'.format(
                    self.dataset, split))

        # read the image set info
        self.imgset_info = []
        splits = [split]
        for split in splits:
            imgset_file = '{0}_{1}.pth'.format(self.dataset, split)
            imgset_path = osp.join(dataset_split_root, imgset_file)
            self.imgset_info += torch.load(imgset_path, map_location="cpu")

        # process the image set info
        if self.dataset == 'flickr':
            self.img_names, self.bboxs, self.phrases = zip(*self.imgset_info)
        else:
            # self.img_names, _, self.bboxs, self.phrases, _ = zip(*self.imgset_info)
            self.img_names, self.obj_ids, self.bboxs, self.phrases, _ = zip(*self.imgset_info)

        self.cache_images = cache_images
        if cache_images:
            self.images_cached = [None] * len(self) #list()
            self.read_image_orig_func = self.read_image
            self.read_image = self.read_image_from_cache

        self.covert_bbox = []
        # xywh to xyxy
        for bbox in self.bboxs:
            bbox = np.array(bbox, dtype=np.float32)
            bbox[2:] += bbox[:2]
            self.covert_bbox.append(bbox)

        self.know_cate_num = know_cate_num
        self.know_cate_len = know_cate_len
        self.know_retrieval_results = know_retrieval_results
        # For model training, pseudo annotations used in Segment Detection module
        pseudo_annotations_path = osp.join(self.dataset_root, 'annotations', 'pseudo_annotations.json')
        with open(pseudo_annotations_path, 'r') as f:
            self.pseudo_annotation = json.load(f)
        # For model training, knowledge categories obtained from the Prompt-Based Retrieval module
        knowledge_categories_path = osp.join(self.dataset_root, self.know_retrieval_results)
        with open(knowledge_categories_path, 'r') as f:
            self.knowledge_category = json.load(f)

    def __len__(self):
        return len(self.img_names)

    def image_path(self, idx):  # notice: db index is the actual index of data.
        return osp.join(self.im_dir, self.img_names[idx])

    def annotation_box(self, idx):
        return self.covert_bbox[idx].copy()

    def phrase(self, idx):
        return self.phrases[idx]

    def obj_id(self, idx):
        return self.obj_ids[idx]

    def image_name(self, idx):
        return self.img_names[idx]

    def cache(self, idx):
        self.images_cached[idx] = self.read_image_orig_func(idx)

    def read_image_from_path_PIL(self, idx):
        image_path = self.image_path(idx)
        pil_image = Image.open(image_path).convert('RGB')
        return pil_image

    def read_image_from_cache(self, idx):
        image = self.images_cached[idx]
        return image

    def __getitem__(self, idx):
        return self.getitem(idx)


    def getitem__PIL(self, idx):
        # reading images
        image = self.read_image(idx)

        # read bbox annotation
        bbox = self.annotation_box(idx)
        bbox = torch.tensor(bbox)

        # read phrase
        phrase = self.phrase(idx)
        phrase = phrase.lower()

        phrase = phrase.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))
        phrase = " ".join(phrase.split())
        sample_id = self.image_name(idx)[:-4] + '_' + self.obj_id(idx)
        empty_knowledge_category = [''] * self.know_cate_num
        pre_knowledge_category = self.knowledge_category.get(sample_id, empty_knowledge_category)

        target = {}
        target['phrase'] = phrase
        target['bbox'] = bbox
        if self.test or self.debug:
            target['orig_bbox'] = bbox.clone()

        for transform in self.transforms:
            image, target = transform(image, target)

        # For BERT
        examples = read_examples(target['phrase'], idx)
        features = convert_examples_to_features(
            examples=examples, seq_length=self.query_len, tokenizer=self.tokenizer)
        word_id = features[0].input_ids
        word_mask = features[0].input_mask

        word_id_knowledge = []
        word_mask_knowledge = []
        for ii in range(self.know_cate_num):
            examples = read_examples(pre_knowledge_category[ii], idx)
            features = convert_examples_to_features(examples=examples, seq_length=self.know_cate_len, tokenizer=self.tokenizer)
            word_id_knowledge.append(features[0].input_ids)
            word_mask_knowledge.append(features[0].input_mask)

        target['word_id'] = torch.tensor(word_id, dtype=torch.long)
        target['word_mask'] = torch.tensor(word_mask, dtype=torch.bool)
        target['knowledge_category_word_id'] = torch.tensor(word_id_knowledge, dtype=torch.long)
        target['knowledge_category_word_mask'] = torch.tensor(word_mask_knowledge, dtype=torch.bool)
        target['pseudo_annotation'] = torch.tensor(self.pseudo_annotation[sample_id], dtype=torch.float32)
        sample_id_list = [int(self.image_name(idx)[:-4]), int(self.obj_id(idx))]
        target['sample_id'] = torch.tensor(sample_id_list, dtype=torch.long)

        if 'mask' in target:
            mask = target.pop('mask')
            return image, mask, target

        return image, target
