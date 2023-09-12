# SLCO
Resource for the ACL paper, [Segment-Level and Category-Oriented Network for Knowledge-Based Referring Expression Comprehension](https://aclanthology.org/2023.findings-acl.557.pdf).

## Introduction

We propose a **segment-level and category-oriented network (SLCO)** for the knowledge-based referring expression comprehension task (KB-REC). This network utilizes knowledge segments to retrieve knowledge categories and delegate them to visual segments for grounding target objects. Experimental results on the KB-Ref dataset demonstrate its effectiveness.

## Environment

1. Clone the repository.
2. Create a virtual environment and install other dependencies.

```
conda create -n slco python=3.6
source activate slco

conda install cudatoolkit=10.2
pip3 install torch==1.6.0
pip3 install torchvision==0.7.0
pip install pytorch-pretrained-bert
pip install nltk
```

## Data Preparation

1. Download the KB-Ref images from [KB-Ref GitHub](https://github.com/wangpengnorman/KB-Ref_dataset) and place them in `data/images`.
2. Download our cleaned KB-Ref annotation files (removing unicode in expressions) from [Google Drive](https://drive.google.com/drive/folders/11Cv7ypIQwjRaoXGNya1sWC_84mzhDWsf?usp=sharing) and place them in `data/annotations`.
3. Download the data for knowledge retrieval from [Google Drive](https://drive.google.com/drive/folders/1HUthFWP8mDX9mVgyQKW047XkJwA2nuh2?usp=sharing) and place it in `knowledge_retrieval/data`.
4. Download the pretrained model of DETR from [Google Drive](https://drive.google.com/drive/folders/1SOHPCCR6yElQmVp96LGJhfTP46RxVwzF) and place it in `preatrained_models`.

The folder structure for the data is as follows:

```
SLCO
├── data
│   ├── annotations
│   │   ├── kbref
│   │   │   ├── kbref_test.pth
│   │   │   ├── kbref_train.pth
│   │   │   └── kbref_val.pth
│   │   └── pseudo_annotations.json
│   └── images
│       └── 1.jpg ...
├── knowledge_retrieval
│   └── data
│       ├── plm-checkpoint-125120
│       ├── knowledge_for_model_training.json
│       ├── kbref_val_anno_format.json
│       ├── kbref_test_anno_format.json
│       ├── gt_knowledge_category.json
│       ├── kbref_knowledge_category_vocab.txt
│       └── obj_label_frcn_conf025.json
└── pretrained_models
    └── detr-r101.pth
```

## Training

```
chmod +x ./train.sh
./train.sh
```

Our trained model can be found in [Google Drive](https://drive.google.com/drive/folders/1vF1XdT20_Z3XfEgUMcY23am2azRAO6QR?usp=drive_link).

## Inference

```chmod +x ./train.sh
chmod +x ./test.sh
./test.sh
```

## Citation

If you find this resource helpful, please cite our paper and share our work.

```bibtex
@inproceedings{conf/acl/BuWL00H23/slco,
  author={Yuqi Bu and Xin Wu and Liuwu Li and Yi Cai and Qiong Liu and Qingbao Huang},
  title={Segment-Level and Category-Oriented Network for Knowledge-Based Referring Expression Comprehension},
  booktitle    = {Findings of the Association for Computational Linguistics: {ACL} 2023, Toronto, Canada, July 9-14, 2023},
  pages        = {8745--8757},
  publisher    = {Association for Computational Linguistics},
  year         = {2023},
}
```

## Acknowledgement

We sincerely thank the authors of [KB-Ref](https://github.com/wangpengnorman/KB-Ref_dataset), [VLTVG](https://github.com/yangli18/VLTVG), [ReSC](https://github.com/zyang-ur/ReSC), and [LAMA](https://github.com/taylorshin/LAMA) for kindly sharing their datasets and codes.