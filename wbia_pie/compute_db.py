# -*- coding: utf-8 -*-
"""
===============================================================================
Script to compute embeddings for images with CNN to use later for re-identification.

USAGE:
    with required parameters only (defaults are from config):
    python compute_db.py -d <db_path>

    with optional parametres:
    python compute_db.py -d <db_path> -c <path_config> -o <path_output_folder> -p <prefix_string>

PARAMETERS:
    -d     path to a directory with images to process. Required argument.
    -c     path to configuration file. Default: configs/config-manta.json
           Preconfigured files: configs/config-manta.json for manta rays, configs/config-whale.json
    -o     path to save csv with embeddings files. Default is in config: prod.embeddings
    -p     prefix, string to add to csv filenames. Default is in config: prod.prefix

README:
Image directory should have the following structure:
image_dir
        - class1
              -img1.png
              -img2.png
              ...
        - class2
              -img3.png
              ...

===============================================================================
"""

import argparse
import os
import json
import numpy as np

from .model.triplet import TripletLoss
from .utils.utils import export_emb
from .utils.preprocessing import read_dataset

argparser = argparse.ArgumentParser(
    description='Compute embeddings for the database. No arguments are required if default values are used.'
)

argparser.add_argument(
    '-d', '--dbpath', required=True, help='Path to folder with localized database images'
)
argparser.add_argument(
    '-c', '--conf', help='Path to configuration file', default='configs/config-manta.json'
)
argparser.add_argument(
    '-o',
    '--output',
    help='Path to output csv files. Default is in config: prod.embeddings',
)
argparser.add_argument(
    '-p',
    '--prefix',
    help='String to add to embeddings file. Default is in config: prod.prefix',
)


def hello():
    print('yo yo yo!')


# This is a package-ified version of original _main_ func
def compute(dbpath, config_path, output_dir, prefix, export=False, augmentation_seed=None):

    # process inputs and load default values
    with open(config_path) as config_buffer:
        config = json.loads(config_buffer.read())

    # should output_dir be unique per augmentation seed?
    if output_dir is None:
        output_dir = config['prod']['embeddings']
        plugin_folder = os.path.dirname(os.path.realpath(__file__))
        output_dir = os.path.join(plugin_folder, output_dir)

    # output_dir is relative so (for now at least--TODO: decide) we'll prepend the plugin dir
    print('output_dir = %s' % output_dir)

    if prefix is None:
        prefix = config['prod']['prefix']

    # Read localized images from a folder with localized database images
    if os.path.exists(dbpath):
        print('Loading images from from {}'.format(dbpath))
        db_imgs, db_labels, lbl2names, db_files = read_dataset(
            dbpath, return_filenames=True, original_labels=False
        )
        db_names = np.array([lbl2names[lab] for lab in db_labels])
    else:
        print('Error! Path does not exist: {}'.format(dbpath))
        quit()

    # print('db_imgs   = %s' % db_imgs)
    # print('db_labels = %s' % db_labels)
    # print('lbl2names = %s' % lbl2names)
    # print('db_files  = %s' % db_files)

    ##############################
    #   Load the model
    ##############################

    INPUT_SHAPE = (config['model']['input_height'], config['model']['input_width'], 3)
    model_args = dict(
        backend=config['model']['backend'],
        frontend=config['model']['frontend'],
        input_shape=INPUT_SHAPE,
        embedding_size=config['model']['embedding_size'],
        connect_layer=config['model']['connect_layer'],
        train_from_layer=config['model']['train_from_layer'],
        loss_func=config['model']['loss'],
        weights='imagenet',
        optimizer=config['model'].get('optimizer', 'adam'),
        use_dropout=config['model'].get('use_dropout', False),
    )
    print('model_args  = %s' % model_args)

    if config['model']['type'] == 'TripletLoss':
        mymodel = TripletLoss(**model_args)
    else:
        raise Exception('Only TripletLoss model type is supported')

    # Define folder for experiment
    exp_folder = os.path.join(
        plugin_folder, config['train']['exp_dir'], config['train']['exp_id']
    )
    saved_weights = os.path.join(exp_folder, 'best_weights.h5')

    if os.path.exists(saved_weights):
        print('Loading saved weights in ', saved_weights)
        mymodel.load_weights(saved_weights)
    else:
        print('ERROR! No pre-trained weights are found in {} ', saved_weights)
        quit()

    # Compute embeddings
    print('Computing embeddings and saving as csv in {}'.format(output_dir))
    db_preds = mymodel.preproc_predict(db_imgs, 1024, augmentation_seed)
    # db_preds appears to be just the embeddings. So we can hook in here and export them

    if export:
        # Export embeddings
        print('Done computing embeddings, exporting to %s' % output_dir)
        export_emb(
            db_preds,
            info=[db_labels, db_files, db_names],
            folder=output_dir,
            prefix=prefix,
            info_header=['class,file,name'],
        )
    return db_preds, db_files


if __name__ == '__main__':
    args = argparser.parse_args()
    compute(
        dbpath=args.dbpath,
        config_path=args.conf,
        output_dir=args.output,
        prefix=args.prefix,
    )
