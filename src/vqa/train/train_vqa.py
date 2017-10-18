import argparse
import logging
import os
import tensorflow as tf
from distutils.util import strtobool

from multiprocessing import Pool

from generic.data_provider.iterator import Iterator
from generic.tf_utils.evaluator import Evaluator, MultiGPUEvaluator
from generic.tf_utils.optimizer import create_multi_gpu_optimizer
from generic.tf_utils.ckpt_loader import load_checkpoint, create_resnet_saver
from generic.utils.config import load_config
from generic.utils.file_handlers import pickle_dump
from generic.data_provider.image_loader import get_img_builder
from generic.data_provider.nlp_utils import GloveEmbeddings
from generic.data_provider.dataset import DatasetMerger

from vqa.data_provider.vqa_tokenizer import VQATokenizer
from vqa.data_provider.vqa_dataset import VQADataset
from vqa.data_provider.vqa_batchifier import VQABatchifier
from vqa.models.vqa_network import VQANetwork
from vqa.train.evaluator_listener import VQADumperListener, VQAEvaluator


###############################
#  LOAD CONFIG
#############################

parser = argparse.ArgumentParser('VQA network baseline!')

parser.add_argument("-data_dir", type=str, help="Directory with data")
parser.add_argument("-img_dir", type=str, help="Directory with image")
parser.add_argument("-img_buf", type=lambda x:bool(strtobool(x)), default="False", help="Store image in memory (faster but require a lot of RAM)")
parser.add_argument("-year", type=str, help="VQA release year (either 2014 or 2017)")
parser.add_argument("-test_set", type=str, default="test-dev", help="VQA release year (either 2014 or 2017)")
parser.add_argument("-exp_dir", type=str, help="Directory in which experiments are stored")
parser.add_argument("-config", type=str, help='Config file')
parser.add_argument("-load_checkpoint", type=str, help="Load model parameters from specified checkpoint")
parser.add_argument("-continue_exp", type=lambda x:bool(strtobool(x)), default="False", help="Continue previously started experiment?")
parser.add_argument("-no_thread", type=int, default=1, help="No thread to load batch")
parser.add_argument("-no_gpu", type=int, default=1, help="How many gpus?")
parser.add_argument("-gpu_ratio", type=float, default=0.95, help="How many GPU ram is required? (ratio)")

args = parser.parse_args()

config, exp_identifier, save_path = load_config(args.config, args.exp_dir)
logger = logging.getLogger()


# Load config
resnet_version = config['model']["image"].get('resnet_version', 50)
finetune = config["model"]["image"].get('finetune', list())
use_glove = config["model"]["glove"]
batch_size = config['optimizer']['batch_size']
no_epoch = config["optimizer"]["no_epoch"]
merge_dataset = config.get("merge_dataset", False)


# Load images
logger.info('Loading images..')
image_builder = get_img_builder(config['model']['image'], args.img_dir)
use_resnet = image_builder.is_raw_image()


# Load dictionary
logger.info('Loading dictionary..')
tokenizer = VQATokenizer(os.path.join(args.data_dir, config["dico_name"]))


# Load data
logger.info('Loading data..')
trainset = VQADataset(args.data_dir, year=args.year, which_set="train", image_builder=image_builder, preprocess_answers=tokenizer.preprocess_answers)
validset = VQADataset(args.data_dir, year=args.year, which_set="val", image_builder=image_builder, preprocess_answers=tokenizer.preprocess_answers)
testset = VQADataset(args.data_dir, year=args.year, which_set=args.test_set, image_builder=image_builder)

if merge_dataset:
    trainset = DatasetMerger([trainset, validset])


# Load glove
glove = None
if use_glove:
    logger.info('Loading glove..')
    glove = GloveEmbeddings(os.path.join(args.data_dir, config["glove_name"]))


# Build Network
logger.info('Building multi_gpu network..')
networks = []
for i in range(args.no_gpu):
    logging.info('Building network ({})'.format(i))

    with tf.device('gpu:{}'.format(i)):
        with tf.name_scope('tower_{}'.format(i)) as tower_scope:

            network = VQANetwork(
                config=config["model"],
                no_words=tokenizer.no_words,
                no_answers=tokenizer.no_answers,
                reuse=(i > 0), device=i)

            networks.append(network)

assert len(networks) > 0, "you need to set no_gpu > 0 even if you are using CPU"


# Build Optimizer
logger.info('Building optimizer..')
optimizer, outputs = create_multi_gpu_optimizer(networks, config, finetune=finetune)
#optimizer, outputs = create_optimizer(networks[0], config, finetune=finetune)


###############################
#  START  TRAINING
#############################

# create a saver to store/load checkpoint
saver = tf.train.Saver()
resnet_saver = None

# Retrieve only resnet variabes
if use_resnet:
    resnet_saver = create_resnet_saver(networks)


# CPU/GPU option
cpu_pool = Pool(args.no_thread, maxtasksperchild=1000)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_ratio)


with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)) as sess:

    # retrieve incoming sources
    sources = networks[0].get_sources(sess)
    scope_names = ['tower_{}/{}'.format(i, network.scope_name) for i, network in enumerate(networks)]
    logger.info("Sources: " + ', '.join(sources))


    # Create evaluation tools
    train_evaluator = MultiGPUEvaluator(sources, scope_names, networks=networks, tokenizer=tokenizer)
    #train_evaluator = Evaluator(sources, scope_names[0], network=networks[0], tokenizer=tokenizer)
    eval_evaluator = Evaluator(sources, scope_names[0], network=networks[0], tokenizer=tokenizer)


    # Load checkpoints or pre-trained networks
    sess.run(tf.global_variables_initializer())
    start_epoch = load_checkpoint(sess, saver, args, save_path)
    if use_resnet:
        resnet_saver.restore(sess, os.path.join(args.data_dir,'resnet_v1_{}.ckpt'.format(resnet_version)))




    train_batchifier = VQABatchifier(tokenizer, sources, glove, remove_unknown=True)
    eval_batchifier = VQABatchifier(tokenizer, sources, glove, remove_unknown=False)


    # Create listener to use VQA evaluation code
    dump_file = save_path.format('tmp.json')
    ques_file = os.path.join(args.data_dir, 'OpenEnded_mscoco_val2014_questions.json')
    ann_file = os.path.join(args.data_dir, 'mscoco_val2014_annotations.json')

    vqa_eval_listener = VQAEvaluator(tokenizer, dump_file, ann_file, ques_file, require=networks[0].prediction)


    # start actual training
    best_val_acc, best_train_acc = 0, 0
    for t in range(start_epoch, no_epoch):

        logger.info('Epoch {}/{}..'.format(t + 1,no_epoch))

        train_iterator = Iterator(trainset,
                                  batch_size=batch_size,
                                  batchifier=train_batchifier,
                                  shuffle=True,
                                  pool=cpu_pool)
        [train_loss, train_accuracy] = train_evaluator.process(sess, train_iterator, outputs=outputs + [optimizer])


        valid_loss, valid_accuracy = 0,0
        if not merge_dataset:
            valid_iterator = Iterator(validset,
                                      batch_size=batch_size*2,
                                      batchifier=eval_batchifier,
                                      shuffle=False,
                                      pool=cpu_pool)

            # Note : As we need to dump a compute VQA accuracy, we can only use a single-gpu evaluator
            [valid_loss, valid_accuracy] = eval_evaluator.process(sess, valid_iterator,
                                                                  outputs=[networks[0].loss, networks[0].accuracy],
                                                                  listener=vqa_eval_listener)

        logger.info("Training loss: {}".format(train_loss))
        logger.info("Training accuracy: {}".format(train_accuracy))
        logger.info("Validation loss: {}".format(valid_loss))
        logger.info("Validation accuracy: {}".format(valid_accuracy))
        logger.info(vqa_eval_listener.get_accuracy())

        if valid_accuracy >= best_val_acc:
            best_train_acc = train_accuracy
            best_val_acc = valid_accuracy
            saver.save(sess, save_path.format('params.ckpt'))
            logger.info("checkpoint saved...")

            pickle_dump({'epoch': t}, save_path.format('status.pkl'))

    # Dump test file to upload on VQA website
    logger.info("Compute final {} results...".format(args.test_set))

    vqa_file_name = "vqa_OpenEnded_mscoco_{}{}_cbn_results.json".format(args.test_set, args.year, config["model"]["name"])
    dumper_eval_listener = VQADumperListener(tokenizer, os.path.join(args.exp_dir, save_path.format(vqa_file_name)),
                                                  require=networks[0].prediction)

    saver.restore(sess, save_path.format('params.ckpt'))
    test_iterator = Iterator(testset,
                             batch_size=batch_size*2,
                             batchifier=eval_batchifier,
                             shuffle=False,
                             pool=cpu_pool)
    eval_evaluator.process(sess, test_iterator, outputs=[], listener=dumper_eval_listener)
    logger.info("File dump at {}".format(dumper_eval_listener.out_path))