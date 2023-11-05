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

## Fine-tune language model

The fine-tuning code is built upon the huggingface's transformers. More information (e.g., required dependencies and libraries) can be found in this [README documentation](https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/README.md).

In contrast to its original random masking approach, our method involves entity-level masking, which requires modifications to certain vocabulary and tokenizer files. Here are the details:

1. Clone Transformers (v4.22.0) repository. `git clone --branch v4.22.0 https://github.com/huggingface/transformers.git`
2. Download BERT model.
   1. Download three files `pytorch_model.bin, config.json, tokenizer_config.json` from [huggingface](https://huggingface.co/bert-large-uncased).
   2. Download our revised files `vocab.txt, tokenizer.json` from [Google Drive](https://drive.google.com/drive/folders/1RP4B987SZABEL8xQsZMCMwjY0ycnvcQc?usp=drive_link).
   3. Place these files in a new folder `transformers/bert_model/bert-large-uncased`.
3. Download fine-tuning data `train_concept_webchild_shuffle.txt` from [Google Drive](https://drive.google.com/drive/folders/1RP4B987SZABEL8xQsZMCMwjY0ycnvcQc?usp=drive_link) and place it in the folder `transformers/bert_model`.
4. Replace the file `transformers/examples/pytorch/language-modeling/run_mlm.py` with the corresponding file from [Google Drive](https://drive.google.com/drive/folders/17fcITpKXaTrW2gYEOsrcN9RFNr8Vmed0?usp=drive_link). In addition, download two files `mask_vocab.txt, mask_word_map_real_word.json` and place them in the same folder.
5. Replace the file `transformers/src/transformers/data/data_collator.py` with the corresponding file from [Google Drive](https://drive.google.com/drive/folders/193R4-wu-jbYQcGT3DfOifUbScA31s_4u?usp=drive_link).
6. Run the code. The output model in the folder `checkpoint-125120` is the same as the provided model in the folder `plm-checkpoint-125120`.
```
cd transformers
CUDA_VISIBLE_DEVICES=0 python examples/pytorch/language-modeling/run_mlm.py --model_name_or_path bert_model/bert-large-uncased --train_file bert_model/train_concept_webchild_shuffle.txt --per_device_train_batch_size 128 --per_device_eval_batch_size 128 --do_train --do_eval --output_dir bert_model/bert-large-uncased-finetuned --line_by_line --pad_to_max_length --max_seq_length 16 --overwrite_output_dir --save_strategy epoch --num_train_epochs 30 --evaluation_strategy epoch
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
