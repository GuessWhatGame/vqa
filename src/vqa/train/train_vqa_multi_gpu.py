#!/usr/bin/env python3
import argparse
import json
import pickle
import logging
import shutil
import os
import tensorflow as tf

import guesswhat.data_provider as provider
import guesswhat.train.utils as utils
import guesswhat.train.image_loader as img

from tensorflow.python.ops import control_flow_ops

from guesswhat.train.evaluator_listener import *

from guesswhat.models.vqa.vqa_network import VQANetwork
from guesswhat.models.vgg16 import channel_mean
from guesswhat.train.evaluators import *
from guesswhat.data_provider import Iterator
from guesswhat.data_provider.vqa_dataset import VQADataset,VQATestDataset
from guesswhat.data_provider.vg_dataset import VGDataset
from guesswhat.data_provider.dataset import DatasetMerger
from guesswhat.data_provider.vqa_batchifier import VQABatchifier

###############################
#  LOAD CONFIG
#############################

parser = argparse.ArgumentParser('Oracle network baseline!')

parser.add_argument("-data_dir", type=str, help="Directory with data")
parser.add_argument("-img_dir", type=str, help="Directory with image")
parser.add_argument("-year", type=str, help="VQA release year (either 2014 or 2017)")
parser.add_argument("-test_set", type=str, default="test-dev", help="VQA release year (either 2014 or 2017)")
parser.add_argument("-exp_dir", type=str, help="Directory in which experiments are stored")
parser.add_argument("-config", type=str, help='Config file')
parser.add_argument("-load_checkpoint", type=str, help="Load model parameters from specified checkpoint")
parser.add_argument("-continue_exp", type=bool, default=False, help="Continue previously started experiment?")
parser.add_argument("-no_thread", type=int, default=1, help="No thread to load batch")
parser.add_argument("-gpu_ratio", type=float, default=0.75, help="How many GPU ram is required? (ratio)")
parser.add_argument("-num_gpus", type=int, default=1, help="How many gpus?")

args = parser.parse_args()

config, exp_identifier, save_path = utils.load_config(args.config, args.exp_dir)
logger = logging.getLogger()

image_input = config["model"]["image_input"]

resnet_version = config['model'].get('resnet_version', 50)
lrt = config['optimizer']['learning_rate']
batch_size = config['optimizer']['batch_size']
clip_val = config['optimizer']['clip_val']
no_epoch = config["optimizer"]["no_epoch"]



logger.info('Loading picture...'.format(image_input))
#TODO move to a helper
use_resnet = False

logger.info('Loading images..')
image_loader = get_img_loader(config['model']['image'], args.image_dir)

if image_input == "fc8" or image_input == "fc7":
    image_loader = img.fcLoader("vqa", args.img_dir, args.year, image_input)
elif image_input == "raw":
    image_loader = img.RawImageLoader(os.path.join(args.img_dir,"img"),
                                      width=config['model']['image']['width'],
                                      height=config['model']['image']['height'])
    use_resnet = True
elif image_input.startswith("conv"):
    image_loader = img.ConvLoader("resnet","vqa", args.img_dir, args.year, image_input)
else:
    logger.info('No image input... use dummy image loader.')
    image_loader = img.DummyImgLoader(args.data_dir, size=config["model"]["fc8_dim"])


# Load data
logger.info('Loading data..')
trainset = VQADataset(args.data_dir, year=args.year, which_set="train", image_loader=image_loader)
validset = VQADataset(args.data_dir, year=args.year, which_set="val", image_loader=image_loader)
testdevset = VQATestDataset(args.data_dir, year=args.year, which_set="test-dev", image_loader=image_loader)
testset = VQATestDataset(args.data_dir, year=args.year, which_set="test", image_loader=image_loader)


if config["merge_dataset"]:
    trainset = DatasetMerger([trainset, validset])


# Load dictionary
logger.info('Loading dictionary..')
tokenizer = provider.VQATokenizer(os.path.join(args.data_dir, config["dico_name"]))

# Load glove
logger.info('Loading glove..')
glove = provider.GloveEmbeddings(os.path.join(args.data_dir, 'glove_dict.pkl'), glove_dim=config["model"]["glove_dim"])

###############################
#  START TRAINING
#############################
fined_tuned = []
if 'fine_tuned' in config["model"]:
    fined_tuned = config["model"]['fine_tuned']
optimizer = tf.train.AdamOptimizer(learning_rate=lrt, name="optimizer")

logger.info('Building network..')
num_gpus = args.num_gpus
logger.info("NUMBER OF GPUS: "+str(num_gpus))
tower_grads = []
for i in range(num_gpus):
    logger.info('Building network (' + str(i) + ')')
    device = 'gpu:'+str(i)
    with tf.device(device):
        with tf.name_scope('tower_'+str(i)) as gpu_scope:
            network = VQANetwork(config=config["model"],
                                 no_words=tokenizer.no_words,
                                 no_answers=tokenizer.no_answers,
                                 image_input=image_input,
                                 reuse=(i>0),
                                 device=i)
            tf.add_to_collection('loss', network.loss)
            tf.add_to_collection('accuracy', network.accuracy)
            train_vars = network.get_parameters(fine_tuned=fined_tuned)
            grads = optimizer.compute_gradients(network.loss, train_vars)
            tower_grads.append(grads)

avg_grads = utils.average_gradients(tower_grads)
clipped_grads = utils.gradient_clipper(avg_grads, clip_val=config['optimizer']['clip_val'])
optimizer = optimizer.apply_gradients(clipped_grads)

avg_loss = tf.reduce_mean(tf.stack(tf.get_collection('loss')))
avg_accuracy = tf.reduce_mean(tf.stack(tf.get_collection('accuracy')))
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
if update_ops:
    updates = tf.group(*update_ops)
    avg_loss = control_flow_ops.with_dependencies([updates], avg_loss)
loss = [avg_loss, avg_accuracy]

saver = tf.train.Saver()

if use_resnet:
    start = len(network.scope_name)+1
    resnet_vars = {v.name[start:-2]: v for v in network.get_resnet_parameters()}
    resnet_saver = tf.train.Saver(resnet_vars)

from multiprocessing import Pool
pool = Pool(args.no_thread, maxtasksperchild=1000)
#vqa_writer = VQATestWriter()

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_ratio)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)) as sess:

    # retrieve incoming sources
    sources = network.get_sources(sess)
    # logger.info("SOURCES:" + sources)

    # Create evaluation tools
    evaluator = MultiGPUEvaluator(sources, network.scope_name, ['tower_'+str(j) for j in range(args.num_gpus)])
    single_gpu_evaluator = Evaluator(sources, os.path.join(gpu_scope,network.scope_name), network=network, tokenizer=tokenizer)
    loss_spliter_eval_listener = VQASplitAnswerLossListener(require=network.extended_accuracy)

    vqa_file_name = "vqa_OpenEnded_mscoco_{}{}_cbn_results.json".format("test", args.year, config["model"]["name"])
    dumper_eval_listener_test = VQADumperListener(tokenizer, os.path.join(args.exp_dir, save_path.format(vqa_file_name)), use_counter=True,
                                             require=network.prediction)

    vqa_file_name = "vqa_OpenEnded_mscoco_{}{}_cbn_results.json".format("test-dev", args.year, config["model"]["name"])
    dumper_eval_listener_testdev = VQADumperListener(tokenizer, os.path.join(args.exp_dir, save_path.format(vqa_file_name)), use_counter=True,
                                                  require=network.prediction)




    dump_file = save_path.format('tmp.json')
    ques_file = os.path.join(args.data_dir, 'OpenEnded_mscoco_val2014_questions.json')
    ann_file = os.path.join(args.data_dir, 'mscoco_val2014_annotations.json')

    vqa_eval_listener = VQAEvaluator(tokenizer, dump_file, ann_file, ques_file, require=network.prediction)

    # define how to split/filter the data
    train_batchifier = VQABatchifier(tokenizer, sources, glove, remove_unknown=True)
    eval_batchifier = VQABatchifier(tokenizer, sources, glove, remove_unknown=False)

    # Load checkpoints or pre-trained networks
    done, start_epoch = utils.load_checkpoint(sess, saver, args, save_path)
    if not done:
        sess.run(tf.global_variables_initializer())
        if use_resnet:
            resnet_saver.restore(sess, os.path.join(args.data_dir,'resnet_v1_'+ str(resnet_version) +'.ckpt'))


    # start actual training
    best_val_acc, best_train_acc = 0, 0
    for t in range(start_epoch, no_epoch):

        logger.info('Epoch {}/{}..'.format(t + 1,no_epoch))

        train_iterator = Iterator(trainset,
                                  batch_size=batch_size,
                                  batchifier=train_batchifier,
                                  shuffle=True,
                                  pool=pool)
        [train_loss, train_accuracy] = evaluator.process(sess, train_iterator, outputs=loss + [optimizer])


        valid_loss, valid_accuracy = 0,0
        if not config["merge_dataset"]:
            valid_iterator = Iterator(validset,
                                      batch_size=batch_size*2,
                                      batchifier=eval_batchifier,
                                      shuffle=True,
                                      pool=pool)

            [valid_loss, valid_accuracy] = evaluator.process(sess, valid_iterator, outputs=loss)

        logger.info("Training loss: {}".format(train_loss))
        logger.info("Training accuracy: {}".format(train_accuracy))
        logger.info("Validation loss: {}".format(valid_loss))
        logger.info("Validation accuracy: {}".format(valid_accuracy))
        logger.info(loss_spliter_eval_listener.get_loss())

        if  t > 9 and t % 3 == 0    :
            logger.info("Compute vqa test-dev results...")
            testdev_iterator = Iterator(testdevset,
                                     batch_size=batch_size*2,
                                     batchifier=eval_batchifier,
                                     shuffle=False,
                                     pool=pool)
            single_gpu_evaluator.process(sess, testdev_iterator, outputs=[], listener=dumper_eval_listener_testdev)

            # logger.info("Compute vqa test results...")
            # test_iterator = Iterator(testset,
            #                          batch_size=batch_size*2,
            #                          batchifier=eval_batchifier,
            #                          shuffle=False,
            #                          pool=pool)
            # single_gpu_evaluator.process(sess, test_iterator, outputs=[], listener=dumper_eval_listener_test)

        if valid_accuracy >= best_val_acc:
            best_train_acc = train_accuracy
            best_val_acc = valid_accuracy
            saver.save(sess, save_path.format('params.ckpt'))
            logger.info("checkpoint saved...")

        pickle.dump({'epoch': t}, open(save_path.format('status.pkl'), 'wb'))


    saver.restore(sess, save_path.format('params.ckpt'))

    logger.info("Compute final test results...")
    dumper_eval_listener = VQADumperListener(tokenizer,
                                             os.path.join(args.exp_dir, save_path.format(vqa_file_name)),
                                             use_counter=False,
                                             require=network.prediction)
    test_iterator = Iterator(testdevset,
                             batch_size=batch_size*2,
                             batchifier=eval_batchifier,
                             shuffle=False,
                             pool=pool)
    single_gpu_evaluator.process(sess, test_iterator, outputs=[], listener=dumper_eval_listener)
