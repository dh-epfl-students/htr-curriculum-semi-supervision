#!/usr/bin/env python
# Compute CER dictionary for Curriculum Learning to have one rank per sample

from __future__ import absolute_import

import argparse
import os
import torch

import laia.common.logging as log
from laia.common.arguments import add_argument, args, add_defaults
from laia.common.arguments_types import str2bool
from laia.common.loader import ModelLoader, CheckpointLoader
from laia.data import ImageDataLoader, ImageFromListDataset
from laia.decoders import CTCGreedyDecoder
from laia.engine.feeders import ImageFeeder, ItemFeeder
from laia.experiments import Experiment
from laia.utils import SymbolsTable, ImageToTensor
from laia.data import ImageDataLoader, TextImageFromTextTableDataset, FixedSizeSampler, TestSampler
import laia.data.transforms as transforms
import multiprocessing
from laia.engine.feeders import ImageFeeder, ItemFeeder
from tqdm import tqdm
import pickle

import editdistance

def cer_score_dict(decoded,target,ids):
    cer_dict = {}
    for ref, hyp, x in zip(target, decoded, ids):
        cer_dict[x] = editdistance.eval(ref,hyp)/len(ref)
    return cer_dict


if __name__ == "__main__":
    add_defaults("batch_size", "gpu", "train_path", logging_level="WARNING")
    add_argument(
        "--syms",
        type=argparse.FileType("r"),
        help="Symbols table mapping from strings to integers",
    )
    add_argument(
        "--img_dirs", type=str, nargs="+", help="Directory containing word images"
    )
    add_argument(
        "--txt_table",
        type=argparse.FileType("r"),
        help="Character transcriptions of each image.",
    )
    add_argument(
        "--delimiters",
        type=str,
        nargs="+",
        default=["<space>"],
        help="Sequence of characters representing the word delimiters.",
    )
    add_argument(
        "--model_filename", type=str, default="model", help="File name of the model"
    )
    add_argument(
        "--checkpoint",
        type=str,
        default="experiment.ckpt.lowest-valid-cer*",
        help="Name of the model checkpoint to use, can be a glob pattern",
    )
    add_argument(
        "--score_function",
        type=str,
        help="Score function",
    )
    add_argument(
        "--source",
        type=str,
        default="experiment",
        choices=["experiment", "model"],
        help="Type of class which generated the checkpoint",
    )
    add_argument(
        "--save_dict_filename",
        type=str
    )

    # Loading of models and datasets
    args = args()

    syms = SymbolsTable(args.syms)
    device = torch.device("cuda:{}".format(args.gpu - 1) if args.gpu else "cpu")

    model = ModelLoader(
        args.train_path, filename=args.model_filename, device=device
    ).load()
    if model is None:
        log.error("Could not find the model")
        exit(1)
    state = CheckpointLoader(device=device).load_by(
        os.path.join(args.train_path, args.checkpoint)
    )
    model.load_state_dict(
        state if args.source == "model" else Experiment.get_model_state_dict(state)
    )
    model = model.to(device)
    model.eval()

    dataset = TextImageFromTextTableDataset(
        args.txt_table,
        args.img_dirs,
        img_transform=ImageToTensor(),
        txt_transform=transforms.text.ToTensor(syms),
    )

    dataset_loader = ImageDataLoader(
        dataset=dataset,
        image_channels=1,
        batch_size=args.batch_size,
        num_workers=multiprocessing.cpu_count(),
    )

    batch_input_fn = ImageFeeder(device=device, parent_feeder=ItemFeeder("img"))
    batch_target_fn = ItemFeeder("txt")
    batch_id_fn = ItemFeeder("id")

    if args.score_function == 'cer':
        score_fn = cer_score_dict

    decoder = CTCGreedyDecoder()

    decoded = []
    target = []
    ids = []

    counter = 0

    # Go through all the samples, compute the prediction, get the label
    for batch in tqdm(dataset_loader):

        counter += 1
        batch_input = batch_input_fn(batch)
        batch_target = batch_target_fn(batch)
        batch_id = batch_id_fn(batch)

        batch_input = batch_input_fn(batch)
        batch_output = model(batch_input)
        batch_decoded = decoder(batch_output)

        decoded.extend(batch_decoded)
        target.extend(batch_target)
        ids.extend(batch_id)

        if counter == 10:
            break
    
    # Compute the CER dictionnary
    cer_dict = cer_score_dict(decoded,target,ids)

    print(cer_dict)

    with open(args.save_dict_filename,'wb') as handle:
        pickle.dump(cer_dict,handle)