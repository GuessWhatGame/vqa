# VQA models

This repo aims at reproducing the results of VQA from the following paper:
-  Modulating early visual processing by language [1] https://arxiv.org/abs/1707.00683

The code was equally developed by Florian Strub (University of Lille) and Harm de Vries (University of Montreal)

The project is part of the CHISTERA - IGLU Project.

#### Summary:

* [Introduction](#introduction)
* [Installation](#installation)
    * [Download](#Download)
    * [Requirements](#requirements)
    * [File architecture](#file-architecture)
    * [Data](#data)
    * [Pretrained models](#pretrained-models)
* [Reproducing results](#reproducing-results)
    * [Process Data](#data)
    * [Train Model](#train-model)
* [FAQ](#faq)
* [Citation](#citation)

## Introduction

We introduce a new Visual Question Answering Baseline (VQA) based on Condtional Batch Normalization technique.
In a few words, A ResNet pipeline is altered by conditioning the Batch Normalization parameters on the question.
It differs from classic approach that mainly focus on developing new attention mechanism.
## Installation


### Download

Our code has internal dependencies called submodules. To properly clone the repository, please use the following git command:\

```
git clone --recursive git@github.com:vqa/vqa.git
```

### Requirements

The code works on both python 2 and 3. It relies on the tensorflow python API.
It requires the following python packages:

```
pip install \
    tensorflow-gpu \
    nltk \
    tqdm
```


### File architecture
In the following, we assume that the following file/folder architecture is respected:

```
vqa
├── config         # store the configuration file to create/train models
|   └── vqa
|
├── out            # store the output experiments (checkpoint, logs etc.)
|   └── vqa
|
├── data          # contains the VQA data
|
└── src            # source files
```

To complete the git-clone file arhictecture, you can do:

```
cd guesswhat
mkdir data;
mkdir out; mkdir out/vqa
```

Of course, one is free to change this file architecture!

### Data
VQA relies on two dataset:
 - VQAv1
 - VQAv2

Note that we ran all lour experiments on VQAv1 but the code is compatible with VQAv2 dataset.
To do so, change the year 2014 to 2017 in this tutorial.

To download the VQA dataset please use the script 'scripts/vqa_download.sh':
```
scripts/vqa_download.sh `pwd`/data
```


### Pretrained networks

TO COME

## Reproducing results

To launch the experiments in the local directory, you first have to set the python path:
```
export PYTHONPATH=src:${PYTHONPATH}
```
Note that you can also directly execute the experiments in the source folder.

### Process Data

Before starting the training, one needs to create a dictionary

#### Extract image features
You do not need to extract image feature for VQA + CBN.
Yet, this code does support any kind of image features as input. If you want to do so, please follow the instruction in the GuessWhat submodule.

#### Create dictionary

To create the VQA dictionary, you need to use the python script vqa/src/vqa/preprocess_data/create_dico.py .

```
python src/vqa/preprocess_data/create_dictionary.py -data_dir data -year 2014 -dict_file dict.json
```


#### Create GLOVE dictionary

Our model use GLOVE vectors (pre-computed word embedding) to perform well.
To create the GLOVE dictionary, you first need to download the original glove file and the you need to use the pythn script guesswhat/src/guesswhat/preprocess_data/create_gloves.py .

```
wget http://nlp.stanford.edu/data/glove.42B.300d.zip -P data/
python src/vqa/preprocess_data/create_gloves.py -data_dir data -glove_in data/glove.42B.300d.zip -glove_out data/glove_dict.pkl -year 2014
```

### Train Model
To train the network, you need to select/configure the kind of neural architecure you want.
To do so, you have update the file config/vqa/config.json

Once the config file is set, you can launch the training step:
```
python src/vqa/train/train_vqa.py \
   -data_dir data \
   -img_dir data/img \
   -config config/vqa/config.json \
   -exp_dir out/vqa \
   -year 2014
   -no_thread 2
```

After training, we obtained the following results:

TBD



## FAQ

 - When I start a python script, I have the following message: ImportError: No module named generic.data_provider.iterator (or equivalent module). It is likely that your python path is not correctly set. Add the "src" folder to your python path (PYTHONPATH=src)


## Citation


```
@inproceedings{guesswhat_game,
author = {Harm de Vries and Florian Strub and J\'er\'emie Mary and Hugo Larochelle and Olivier Pietquin and Aaron C. Courville},
title = {Modulating early visual processing by language},
booktitle = {Advances in Neural Information Processing Systems 30},
year = {2017}
url = {https://arxiv.org/abs/1707.00683}
}
```


## Acknowledgement
 - SequeL Team
 - Mila Team






