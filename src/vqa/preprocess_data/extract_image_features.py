#!/usr/bin/env python
import numpy
import os
import tensorflow as tf
from multiprocessing import Pool
from tqdm import tqdm
import numpy as np
import argparse

import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.python.slim.nets.resnet_v1 as resnet_v1
import tensorflow.contrib.slim.python.slim.nets.resnet_utils as slim_utils

from generic.data_provider.image_loader import RawImageBuilder

from generic.data_provider.iterator import Iterator


from vqa.data_provider.vqa_dataset import VQATestDataset
from vqa.data_provider.vqa_batchifier import VQABatchifier



parser = argparse.ArgumentParser('Feature extractor! ')

parser.add_argument("-img_dir", type=str, required=True, help="Input Image folder")
parser.add_argument("-data_dir", type=str, required=True,help="Dataset folder")
parser.add_argument("-set_type", type=list, default=["val", "train", "test", "test-dev"], help='Select the dataset to dump')

parser.add_argument("-data_out", type=str, required=True, help="Output folder")
parser.add_argument("-ckpt", type=str, required=True, help="Path for network checkpoint (resnet-152) ")
parser.add_argument("-feature_name", type=str, default="block4", help="Pick the name of the network features")

parser.add_argument("-subtract_mean", type=bool, default=True, help="Preprocess the image by substracting the mean")
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
    feature_name = os.path.join("resnet_v1_152", args.feature_name) # define the feature name according slim standard


# Define the output folder
out_file = "vqa_resnetV1_152_{feature_name}_{size}".format(feature_name=args.feature_name, size=args.img_size)

out_dir = os.path.join(args.data_dir, out_file)
if not os.path.exists(out_dir):
    os.makedirs(out_dir)


#Create a dummy tokenizer
class DummyTokenizer(object):
    def __init__(self):
        self.padding_token = 0

    def encode_question(self, _):
        return []

    def encode_answer(self, _):
        return []


dummy_tokenizer = DummyTokenizer()


# CPU/GPU option
cpu_pool = Pool(args.no_thread, maxtasksperchild=1000)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_ratio)

with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)) as sess:

    saver = tf.train.Saver()
    saver.restore(sess, args.ckpt)

    for one_set in args.set_type:

        print("Load dataset -> set: {}".format(one_set))
        dataset = VQATestDataset(args.data_dir, one_set, image_builder=image_builder)
        batchifier = VQABatchifier(tokenizer=dummy_tokenizer, sources=[source])
        iterator = Iterator(dataset,
                            batch_size=args.batch_size,
                            pool=cpu_pool,
                            batchifier=batchifier)

        for batch in tqdm(iterator):
            feat = sess.run(end_points[feature_name], feed_dict={images: numpy.array(batch[source])})
            for ft, game in zip(feat, batch["raw"]):
                filename = os.path.join(out_dir, "{}.npz".format(game.image.id))
                np.savez_compressed(filename, x=ft)


print("Done!")
