# htr-curriculum-semi-supervision
Curriculum Learning And Semi-Supervised Learning applied for Handwritten Text Recognition

## Basic information

Student: Karthigan Sinnathamby

Supervisor: Sofia Ares Oliveira 

Academic Year: Fall 2019

## Introduction
Handwritten Text Recognition is a challenging task when dealing with few samples only.  In this study, we tackle this difficult problem from the learning point of view. Transfer Learning has come to be very handy in such cases.  In order to improve the accuracy of this method, we explore two methods orthogonal to it:  Curriculum Learning and Semi-Supervised Learning.

## Research summary
Many benchmarks have been done using TL. Building on top of this method, we explore two types of learning orthogonal to TL: Curriculum Learning (CL) and Semi-Supervised Learning (SSL).

CL reshape how samples are presented to the learning algorithm. We compute a rank score using a cross-validation Transfer Learning taking the validation CER. Then, we use a single-step pacing function to choose the samples presented to the learning function.

SSL tries to use unlabelled data along with labelled data. We use the prediction of the model as "fake" label of the unlabelled sampled. We take only those which have a good confidence score. This score is computed based on the entropy of the prediction or based on the difference between the two best probabilities at each timesteps. We use the labelled dataset as well as another dynamic dataset which is constructed using these 'fake' labels.


## Installation and Usage


### Modification to PyLaia
We have implemented the function on top of [Pylaia Code](https://github.com/jpuigcerver/PyLaia). Some comments on the Pylaia Code and usage are given in [code/pylaia-explanation.md](code/pylaia-explanation.md). In the [code](code/) folder, you will have the Pylaia Code. We have mainly modified some files listed below:

- [pylaia-htr-train-ctc](code/pylaia-htr-train-ctc)
- [laia/engine/trainer.py](code/laia/engine/trainer.py)
- [laia/experiement/htr_experiment.py](code/laia/experiement/htr_experiment.py)

A new file has been added for the model training code:
[laia/data/karthigan_custom_sampler.py](code/laia/data/karthigan_custom_sampler.py)

All the other new files are present in [code/egs/iam-htr/src](code/egs/iam-htr/src). **These files must be run from [code/egs/iam-htr/](code/egs/iam-htr/)**

One package has been used additionally to the ones required by the Pylaia Code:

`pip install symspellpy`

Below you will find instructions to use the code and change the parameters for each topic.

Basically, we run the same python scripy pylaia-htr-train-ctc changing the different arguments.
Before that we need to prepare the different datasets and models. **Please run the sections Datasets, Models and Baseline even if you want to do only CL or SSL.**

### Datasets
For IAM, follow the instructions present in [code/egs/iam-htr/README.md](code/egs/iam-htr/README.md).

For Washington, follow the instructions below:
- ./download.sh (washington folder)
- ./prepare_dortmund.sh (washington folder)
- ./prepare_almazan_lines.sh (washington folder)
- rename images/original to original/lines
- rename almazan_lines folder to data_washington and copy to iam-htr/
- ./prepare_images_washington.sh (iam-htr folder)
- move data_washington/lang content to data_washington/lang/puigcerver/lines/

For ICFHR, download the files and put them [code/egs/iam-htr/data_icfhr/general_data](code/egs/iam-htr/data_icfhr/general_data). Run [code/egs/iam-htr/src/transform_icfhr.py](code/egs/iam-htr/src/transform_icfhr.py) with the name of the wanted subfolder. For instance, for Bentham collection, run: `python3 src/transform_icfhr.py 30887`

### Models
First you need to do a full training on IAM dataset as written in [code/egs/iam-htr/README.md](code/egs/iam-htr/README.md).
For Washington dataset, you can directly perform the Transfer Learning.

For ICFHR, as the alphabet can be different, you need to retrain the FC layer of the model to adapt to the new alphabet. To do so, run the following:

- `python3 src/create_alphabet.py data_icfhr`
- `src/create_model_icfhr.sh`
- `src/train_icfhr_alphabet.sh`

### Baseline

Run [code/egs/iam-htr/src/create_one_split.py](code/egs/iam-htr/src/create_one_split.py) with the wanted folder as argument (for instance `python3 src/create_one_split.py data_washington`)
to create one train/val/split of your dataset. 

If you want to sample a specific number of samples, run [code/egs/iam-htr/src/create_sampled_split.py](code/egs/iam-htr/src/create_sampled_split.py) by changing these [lines](https://github.com/karna2605/htr-curriculum-semi-supervision/blob/741b5207aaf2f6886654a76afdb33a2349030127/code/egs/iam-htr/src/create_sampled_split.py#L42-L52).  In particular, the variable filename_tr_unlabel is only for SSL. 

If you want a reduced dataset of the baseline, for instance to reproduce Baseline-30% of the report, run [code/egs/iam-htr/src/sample_baseline.py](code/egs/iam-htr/src/sample_baseline.py) (for instance `python3 src/sample_baseline.py data_washington 0.3 True`)

### Curriculum Learning
To perform one run of CL, run [code/egs/iam-htr/src/train_one_split.sh](code/egs/iam-htr/src/train_one_split.sh).

To create cross-validation splits, run [code/egs/iam-htr/src/cv_splits_creation.py](code/egs/iam-htr/src/cv_splits_creation.py) with the data folder as argument. Then run the script [code/egs/iam-htr/src/transfer.sh](code/egs/iam-htr/src/transfer.sh) to perform transfer learning cross-validation for CL. 

To get the final CER dictionnary, use [code/egs/iam-htr/src/merge_dict_score.py](code/egs/iam-htr/src/merge_dict_score.py) that merges the dictionnary that came from each of the cross-validation split.

Now, we can perform CL by using [code/egs/iam-htr/src/cl.sh](code/egs/iam-htr/src/cl.sh).

To change the pacing function, modifiy those [lines](https://github.com/dhlab-epfl-students/htr-curriculum-semi-supervision/blob/master/code/pylaia-htr-train-ctc#L343-L346).

To compute the Normalized Levenshtein Distance distribution, use the script [code/egs/iam-htr/src/hist_transfer.py](code/egs/iam-htr/src/hist_transfer.py).

### Semi-Supervised Learning

For the SSL, use the script [code/egs/iam-htr/src/semi_supervised_one_split.sh](code/egs/iam-htr/src/semi_supervised_one_split.sh).

In order to use the spell checker, you need to download from the [original github](https://github.com/mammothb/symspellpy), the corpus file corresponding to the language of the used dataset. This file needs to be put at [code/egs/iam-htr/](code/egs/iam-htr/) and you need to change this [line](https://github.com/dhlab-epfl-students/htr-curriculum-semi-supervision/blob/master/code/laia/engine/trainer.py#L117). 

The use of the spell checker during the training is at these [lines](https://github.com/dhlab-epfl-students/htr-curriculum-semi-supervision/blob/master/code/laia/engine/trainer.py#L271-L279).

To start the SSL after attaining a certain validation CER, change the threshold at this [line](https://github.com/dhlab-epfl-students/htr-curriculum-semi-supervision/blob/master/code/laia/experiments/htr_experiment.py#L100). If you want to do this, you still need to set non_decreasing_epochs_semi_supervised argument: you can set it to a high value, for instance 100.

To change the number of samples of dataset A used when using a dataset B as mentioned in the report, change this [line](https://github.com/dhlab-epfl-students/htr-curriculum-semi-supervision/blob/master/code/laia/engine/trainer.py#L204).

To change the metric used to rank the samples (entropy based or 'diff-proba'), change this [line](https://github.com/dhlab-epfl-students/htr-curriculum-semi-supervision/blob/master/code/laia/engine/trainer.py#L219).


### pylaia-htr-train-ctc and Shell

Basically the parameters of pylaia-htr-train-ctc need to be changed when launching a shell script. 
**You can take [code/egs/iam-htr/src/semi_supervised_one_split.sh](code/egs/iam-htr/src/semi_supervised_one_split.sh) as a reference script and modify it to launch the code for different usages.**

The other main thing to change in the shell scripts is the path to the right dataset for the training and validation split. You will also need to set where the weights of the right model are and the directory of the images. The parameters are self-explanatory, be sure to give the right value depending on the usage you want.

**Beware: Run setup.py install at each modification of a python file**

### Licence
htr-curriculum-semi-supervision - Karthigan Sinnathamby

Copyright (c) 2019 EPFL

This program is licensed under the terms of this [licence](LICENSE).
