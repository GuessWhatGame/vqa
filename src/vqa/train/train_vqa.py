import argparse
import logging
import os
import tensorflow as tf
from distutils.util import strtobool

from multiprocessing import Pool

from generic.data_provider.iterator import Iterator
from generic.tf_utils.evaluator import Evaluator
from generic.tf_utils.optimizer import create_optimizer
from generic.tf_utils.ckpt_loader import load_checkpoint
from generic.utils.config import load_config
from generic.utils.file_handlers import pickle_dump
from generic.data_provider.image_loader import get_img_loader
from generic.data_provider.nlp_utils import GloveEmbeddings
from generic.data_provider.dataset import DatasetMerger

from vqa.data_provider.vqa_tokenizer import VQATokenizer
from vqa.data_provider.vqa_dataset import VQADataset,VQATestDataset
from vqa.data_provider.vqa_batchifier import VQABatchifier
from vqa.models.vqa_network import VQANetwork
from vqa.train.evaluator_listener import VQADumperListener, VQAEvaluator

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
parser.add_argument("-continue_exp", type=lambda x:bool(strtobool(x)), default="False", help="Continue previously started experiment?")
parser.add_argument("-no_thread", type=int, default=1, help="No thread to load batch")
parser.add_argument("-gpu_ratio", type=float, default=0.75, help="How many GPU ram is required? (ratio)")
parser.add_argument("-num_gpus", type=int, default=1, help="How many gpus?")

args = parser.parse_args()

config, exp_identifier, save_path = load_config(args.config, args.exp_dir)
logger = logging.getLogger()


# Load config
resnet_version = config['model'].get('resnet_version', 50)
use_glove = config["model"]["glove"]
finetune = config["model"]["image"].get('finetune', list())
lrt = config['optimizer']['learning_rate']
batch_size = config['optimizer']['batch_size']
clip_val = config['optimizer']['clip_val']
no_epoch = config["optimizer"]["no_epoch"]
merge_dataset = config.get("merge_dataset", False)


# Load images
logger.info('Loading images..')
image_loader = get_img_loader(config['model']['image'], args.img_dir)
use_resnet = image_loader.is_raw_image()


# Load dictionary
logger.info('Loading dictionary..')
tokenizer = VQATokenizer(os.path.join(args.data_dir, config["dico_name"]))

# Load data
logger.info('Loading data..')
trainset = VQADataset(args.data_dir, year=args.year, which_set="train", image_loader=image_loader, preprocess_answers=tokenizer.preprocess_answers)
validset = VQADataset(args.data_dir, year=args.year, which_set="val", image_loader=image_loader, preprocess_answers=tokenizer.preprocess_answers)
testset = VQATestDataset(args.data_dir, year=args.year, which_set=args.test_set, image_loader=image_loader)

if merge_dataset:
    trainset = DatasetMerger([trainset, validset])

# Load glove
glove = None
if use_glove:
    logger.info('Loading glove..')
    glove = GloveEmbeddings(os.path.join(args.data_dir, config["glove_name"]))

# Build Network
logger.info('Building network..')
network = VQANetwork(config=config["model"],
                                 no_words=tokenizer.no_words,
                                 no_answers=tokenizer.no_answers)

# Build Optimizer
logger.info('Building optimizer..')
optimizer, loss = create_optimizer(network, network.loss, config, finetune=finetune)
outputs = [loss, network.accuracy]


###############################
#  START  TRAINING
#############################

# create a saver to store/load checkpoint
saver = tf.train.Saver()
resnet_saver = None

# Retrieve only resnet variabes
if use_resnet:
    start = len(network.scope_name)+1
    resnet_vars = {v.name[start:-2]: v for v in network.get_resnet_parameters()}
    resnet_saver = tf.train.Saver(resnet_vars)


# CPU/GPU option
cpu_pool = Pool(args.no_thread, maxtasksperchild=1000)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_ratio)


with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)) as sess:

    # retrieve incoming sources
    sources = network.get_sources(sess)
    logger.info("Sources: " + ', '.join(sources))


    # Load checkpoints or pre-trained networks
    sess.run(tf.global_variables_initializer())
    start_epoch = load_checkpoint(sess, saver, args, save_path)
    if use_resnet:
        resnet_saver.restore(sess, os.path.join(args.data_dir,'resnet_v1_{}.ckpt'.format(resnet_version)))


    # Create evaluation tools
    evaluator = Evaluator(sources, network.scope_name, network=network, tokenizer=tokenizer)
    train_batchifier = VQABatchifier(tokenizer, sources, glove, remove_unknown=True)
    eval_batchifier = VQABatchifier(tokenizer, sources, glove, remove_unknown=False)


    # Create listener to use VQA evaluation code
    dump_file = save_path.format('tmp.json')
    ques_file = os.path.join(args.data_dir, 'OpenEnded_mscoco_val2014_questions.json')
    ann_file = os.path.join(args.data_dir, 'mscoco_val2014_annotations.json')

    vqa_eval_listener = VQAEvaluator(tokenizer, dump_file, ann_file, ques_file, require=network.prediction)


    # start actual training
    best_val_acc, best_train_acc = 0, 0
    for t in range(start_epoch, no_epoch):

        logger.info('Epoch {}/{}..'.format(t + 1,no_epoch))

        train_iterator = Iterator(trainset,
                                  batch_size=batch_size,
                                  batchifier=train_batchifier,
                                  shuffle=True,
                                  pool=cpu_pool)
        [train_loss, train_accuracy] = evaluator.process(sess, train_iterator, outputs=outputs + [optimizer])


        valid_loss, valid_accuracy = 0,0
        if not merge_dataset:
            valid_iterator = Iterator(validset,
                                      batch_size=batch_size*2,
                                      batchifier=eval_batchifier,
                                      shuffle=True,
                                      pool=cpu_pool)

            [valid_loss, valid_accuracy] = evaluator.process(sess, valid_iterator, outputs=outputs, listener=vqa_eval_listener)

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
                                                  require=network.prediction)

    saver.restore(sess, save_path.format('params.ckpt'))
    test_iterator = Iterator(testset,
                             batch_size=batch_size*2,
                             batchifier=eval_batchifier,
                             shuffle=False,
                             pool=cpu_pool)
    evaluator.process(sess, test_iterator, outputs=[], listener=dumper_eval_listener)
    logger.info("File dump at {}".format(dumper_eval_listener.out_path))