#!/usr/bin/env python
import os
import tensorflow as tf
from distutils.util import strtobool
import numpy as np
import argparse

import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.python.slim.nets.resnet_v1 as resnet_v1
import tensorflow.contrib.slim.python.slim.nets.resnet_utils as slim_utils

from generic.data_provider.image_loader import RawImageBuilder
from generic.preprocess_data.extract_img_features import extract_features

from vqa.data_provider.vqa_dataset import VQADataset
from vqa.data_provider.vqa_batchifier import VQABatchifier



parser = argparse.ArgumentParser('Feature extractor! ')

parser.add_argument("-img_dir", type=str, required=True, help="Input Image folder")
parser.add_argument("-data_dir", type=str, required=True,help="Dataset folder")
parser.add_argument("-out_dir", type=str, required=True, help="Output directory for h5 files")
parser.add_argument("-set_type", type=list, default=["val", "train", "test"], help='Select the dataset to dump')
parser.add_argument("-year", type=int, default=2014, help="VQA dataset year (2014/2017)")

parser.add_argument("-ckpt", type=str, required=True, help="Path for network checkpoint: ")
parser.add_argument("-feature_name", type=str, default="block4", help="Pick the name of the network features")

parser.add_argument("-subtract_mean", type=lambda x:bool(strtobool(x)), default="True", help="Preprocess the image by substracting the mean")
parser.add_argument("-img_size", type=int, default=224, help="image size (pixels)")
parser.add_argument("-batch_size", type=int, default=64, help="Batch size to extract features")

parser.add_argument("-gpu_ratio", type=float, default=1., help="How many GPU ram is required? (ratio)")
parser.add_argument("-no_thread", type=int, default=2, help="No thread to load batch")

args = parser.parse_args()


# define image
if args.subtract_mean:
    channel_mean = np.array([123.68, 116.779, 103.939])
else:
    channel_mean = None


# define the image loader
source = 'image'
images = tf.placeholder(tf.float32, [None, args.img_size, args.img_size, 3], name=source)
image_builder = RawImageBuilder(args.img_dir,
                                height=args.img_size,
                                width=args.img_size,
                                channel=channel_mean)



# create network
print("Create network...")
with slim.arg_scope(slim_utils.resnet_arg_scope(is_training=False)):
    _, end_points = resnet_v1.resnet_v1_152(images, 1000)  # 1000 is the number of softmax class

    ft_name = os.path.join("resnet_v1_152", args.feature_name) # define the feature name according slim standard
    ft_output = end_points[ft_name]


extract_features(
    img_input = images,
    ft_output = ft_output,
    dataset_cstor = VQADataset,
    dataset_args = {"folder": args.data_dir, "year": args.year, "image_builder":image_builder},
    batchifier_cstor = VQABatchifier,
    out_dir = args.out_dir,
    set_type = args.set_type,
    network_ckpt=args.ckpt,
    batch_size = args.batch_size,
    no_threads = args.no_thread,
    gpu_ratio = args.gpu_ratio,
)
