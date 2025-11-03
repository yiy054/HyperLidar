#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import argparse
import subprocess
import datetime
import yaml
from shutil import copyfile
import os
import shutil

if __name__ == '__main__':
    splits = ["train", "valid", "test"]
    parser = argparse.ArgumentParser("./infer.py")
    parser.add_argument(
        '--dataset', '-d',
        type=str,
        required=True,
        default=None,
        help='Dataset to train with. No Default',
    )
    parser.add_argument(
        '--model', '-m',
        type=str,
        required=True,
        default=None,
        help='Directory to get the trained model.'
    )
    parser.add_argument(
        '--train_seq', '-t', 
        type=str, 
        required=False,
        default=None,
        help='Comma-separated list of training sequences (e.g., "0,1,2")'
    )
    parser.add_argument(
        '--buffer_rate', '-b',
        type=float,
        required=False,
        default=None,
        help='Buffer rate setup (e.g., "0.1" or "0.1,0.3"). If multiple values provided, the first is used.'
    )
    FLAGS, unparsed = parser.parse_known_args()

    # print summary of what we will do
    print("----------")
    print("INTERFACE:")
    print("dataset", FLAGS.dataset)
    print("model", FLAGS.model)
    print("buffer rate", FLAGS.buffer_rate)
    print("----------\n")

    # open arch config file
    try:
        print("Opening arch config file from %s" % FLAGS.model)
        ARCH = yaml.safe_load(open(FLAGS.model + "/arch_cfg.yaml", 'r'))
        # ARCH = yaml.safe_load(open("config/arch/senet-512.yml", 'r'))
    except Exception as e:
        print(e)
        print("Error opening arch yaml file.")
        quit()

    # open data config file
    try:
        print("Opening data config file from %s" % FLAGS.model)
        DATA = yaml.safe_load(open(FLAGS.model + "/data_cfg.yaml", 'r'))
        # DATA = yaml.safe_load(open("config/labels/semantic-kitti.yaml", 'r'))
    except Exception as e:
        print(e)
        print("Error opening data yaml file.")
        quit()

    # does model folder exist?
    if os.path.isdir(FLAGS.model):
        print("model folder exists! Using model from %s" % (FLAGS.model))
    else:
        print("model folder doesnt exist! Can't infer...")
        quit()
    
    if FLAGS.train_seq is not None:
        try:
            DATA["split"]["train"] = [int(s.strip()) for s in FLAGS.train_seq.split(",")]
            print(f"[INFO] Overriding training sequences: {DATA['split']['train']}")
        except ValueError as e:
            print(f"[ERROR] Failed to parse --train_seq: {e}")
            exit(1)
    else:
        print("[INFO] Using default training sequences from data config.")

    from modules.Basic_HD import BasicHD
    BasicHD = BasicHD(ARCH, DATA, FLAGS.dataset, FLAGS.model, FLAGS.buffer_rate)
    BasicHD.start()

    # from modules.Basic_Conv import BasicConv
    # BasicConv = BasicConv(ARCH, DATA, FLAGS.dataset, FLAGS.model, None)
    # BasicConv.start()

