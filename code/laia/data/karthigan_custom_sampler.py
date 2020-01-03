from __future__ import absolute_import

import torch
from torch.utils.data.sampler import Sampler
from collections import OrderedDict
import random
import math


class TestBatchSampler(Sampler):
    def __init__(self, n_batches, batch_size, data_source, val_data_source, dict_source, mode, starting_percent, step_length, inc, use_reverse):
        """
            n_batches: number of batch
            batch_size: size of the batch
            data_source: dataloader of the original dataset (train+validation)
            val_data_source: validation dataloader
            dict_source: dictionnary of CER from the cross-validation transfer Learning
            mode: single_step, fixed_exp or var_exp. The latter has not be implemented
            starting_percent: starting fraction of the dataset to be used
            step_length: length of a step before increasing the fraction
            inc: increment factor
            use_reverse: Whether we use reverse CL
        """
        self.batch_size = batch_size
        self.n_batches = n_batches

        self.data_source = data_source
        self.val_data_source = val_data_source

        self.mode = mode
        self.starting_percent = starting_percent
        self.step_length = step_length
        self.inc = inc

        self.iteration = 0
        self.size = 0
        self.use_reverse = use_reverse

        dict_source =  self.remove_val_filenames(dict_source)

        if self.use_reverse:
            self.samples_ordered = list(OrderedDict(sorted(dict_source.items(), key=lambda x: x[1],reverse=True)).keys())
        else:
            self.samples_ordered = list(OrderedDict(sorted(dict_source.items(), key=lambda x: x[1])).keys())

        self.total_number_samples = len(self.samples_ordered)

    def remove_val_filenames(self,dict_source):
        """Remove validation samples from the dataset dictionnary"""
        val_keys = self.val_data_source.imgs
        val_keys = [x.split("/")[-1] for x in val_keys]
        val_keys = [x.split(".")[0] for x in val_keys]
        
        new_dict = { your_key: dict_source[your_key] for your_key in dict_source.keys() if your_key not in val_keys }

        return new_dict
        
    def __iter__(self):
        ctr = 0
        while ctr < self.n_batches:
            size = self.g_theta(self.iteration)
            self.size = size
            idxs = self.get_samples(size)
            yield idxs

            # Increment batch counter
            self.iteration += 1
            ctr += 1

    def __len__(self):
        """Returns how many batches are inside."""
        return self.n_batches

    def get_samples(self,size):
        """Get samples from the ranked dataset. Randomly sample among the 'size' first samples"""
        tmp_samples = self.samples_ordered[:size]
        filenames = random.sample(tmp_samples,self.batch_size)
        idxs = []
        for filename in filenames:
            try:
                x = self.data_source._ids.index(filename)
                idxs.append(x)
            except:
                print("NOT FOUND",filename)
        return idxs

    def g_theta(self,iteration):
        """Pacing function"""
        if self.mode == 'fixed_exp':
            return self.fixed_exp(iteration)
        elif self.mode == 'var_exp':
            # Not DEFINED
            return self.var_exp(iteration)
        elif self.mode == 'single_step':
            return self.single_step(iteration)

    def fixed_exp(self,iteration):
        exponent = math.floor(iteration/self.step_length)
        return math.ceil(min(self.starting_percent*(self.inc**exponent),1)*self.total_number_samples)

    def single_step(self,iteration):
        exponent = 1 if iteration < self.step_length else 0
        return math.ceil(self.total_number_samples*(self.starting_percent**exponent))


class TestSampler(Sampler):
    """Randomly resample elements from a data source up to a fixed number of
    samples.

    In each iterator, the samples are randomly sorted. If `num_samples` is
    greater than the number of samples in the `data_source`, there will be
    some repetitions in the new dataset. Otherwise, the samples will be unique.

    Arguments:
        data_source (Dataset): dataset to sample from
        num_samples (int): number of samples to resample from the dataset
    """

    def __init__(self, data_source, num_samples, dict_source):
        super(TestSampler, self).__init__(data_source)
        self._data_source = data_source
        self._num_samples = num_samples
        self._dict_source = dict_source

    def __iter__(self):
        idxs = []
        while len(idxs) < self._num_samples:
            idxs.extend([self._data_source._ids.index('a01-000u-00')])
        return iter(idxs[: self._num_samples])

    def __len__(self):
        return self._num_samples
