from __future__ import absolute_import

from typing import Callable, Union, Iterable, Optional

import torch

import laia.common.logging as log
from laia.engine.engine import Engine, EPOCH_END, ITER_START, ITER_END
from laia.hooks import Hook, action
from laia.losses.loss import Loss
from laia.utils import check_inf, check_nan
from laia.decoders import CTCGreedyDecoder

_logger = log.get_logger(__name__)

from laia.losses.ctc_loss import transform_output
from tqdm import tqdm
from torch._six import string_classes
import numpy as np
import os
import torch.nn.functional as F

from symspellpy import SymSpell, Verbosity

class Trainer(Engine):
    r"""Wrapper class to train a model.

    See :class:`laia.engine.Engine` for more information.

    Args:
        model: model to train.
        criterion: used criterion to train the model.
        optimizer: optimizer object that will update the parameters of the model.
        data_loader: iterable object from which batches are read.
        batch_input_fn: function used to extract the input
            for the model (e.g. a ``torch.Tensor``), from the batch loaded by
            the ``data_loader``. If ``None``, the batch is fed as-is to the
            model.
        batch_target_fn: if given, this callable object
            is used to extract the targets from the batch, which are
            passed to the `ITER_START` and `ITER_END` hooks.
        batch_id_fn: if given, this callable object is
            used to extract the batch ids to be used in a possible exception.
        progress_bar: if ``True``, :mod:`tqdm` will be
            used to show a progress bar for each epoch. If a string is given,
            the content of the string will be shown before the progress bar.
        iterations_per_update: Number of successive mini-batch
            parameter gradients to accumulate before updating the parameters.
        cv_number: Display information to see which cross-validation we are doing
        use_baseline: Whether to perform the baseline (No CL nor SSL)
        use_cl: Whether to use curriculum learning (CL)
        use_transfer: Whether to use transfer learning (TL)
        use_semi_supervised: Whether to use semi-supervised learning (SSL)
        threshold_score_semi_supervised: Threshold on the rank of the samples for SSL
        data_semi_supervised_loader: unlabbeled dataset for SSL
        epoch_frequency_semi_supervision: Frequency of update of the SSL dataset B (the one which is not labelled), cf report
        syms: token-text table
        original_data_loader: Original dataset (dataset A) for SSL
    """

    def __init__(
        self,
        model,  # type: torch.nn.Module
        criterion,  # type: Optional[Callable]
        optimizer,  # type: torch.optim.Optimizer
        data_loader=None,  # type: Optional[Iterable]
        batch_input_fn=None,  # type: Optional[Callable]
        batch_target_fn=None,  # type: Optional[Callable]
        batch_id_fn=None,  # type: Optional[Callable]
        progress_bar=None,  # type: Optional[Union[bool, str]]
        iterations_per_update=1,  # type: int
        cv_number=None,
        use_baseline=None,
        use_cl=None,
        use_transfer=None,
        use_semi_supervised=None,
        threshold_score_semi_supervised=None,
        data_semi_supervised_loader=None,
        epoch_frequency_semi_supervision=None,
        syms=None,
        original_data_loader=None,
    ):
        # type: (...) -> None
        super(Trainer, self).__init__(
            model=model,
            data_loader=data_loader,
            batch_input_fn=batch_input_fn,
            batch_target_fn=batch_target_fn,
            batch_id_fn=batch_id_fn,
            progress_bar=progress_bar,
            use_baseline=use_baseline,
            use_cl=use_cl,
            use_transfer=use_transfer
        )
        self._criterion = criterion
        self._optimizer = optimizer
        self._iterations_per_update = iterations_per_update
        self._updates = 0
        self._cv_number = cv_number
        self._progress_bar = progress_bar

        self.data_loader = data_loader
        
        self.use_semi_supervised=use_semi_supervised
        self.threshold_score_semi_supervised=threshold_score_semi_supervised
        self.data_semi_supervised_loader=data_semi_supervised_loader
        self.epoch_frequency_semi_supervision = epoch_frequency_semi_supervision
        self.counter_epoch_semi_supervision = 0
        self.semi_supervision_started = False
        self.original_dataset = {'ids':data_loader.dataset._ids,'imgs':data_loader.dataset._imgs,'txts':data_loader.dataset._txts}
        self.decoder = CTCGreedyDecoder()
        self.syms = syms
        self.original_data_loader = original_data_loader

        # Load Spell Checker
        self.sym_spell = SymSpell(max_dictionary_edit_distance=5, prefix_length=7)
        dict_name = 'de_50k.txt' #"frequency_dictionary_en_82_765.txt"
        if not self.sym_spell.load_dictionary(dict_name,term_index=0, count_index=1, encoding='utf-8-sig'):
             print("error loading spell checker")

    @property
    def criterion(self):
        return self._criterion

    @criterion.setter
    def criterion(self, criterion):
        assert callable(criterion)
        self._criterion = criterion

    @property
    def optimizer(self):
        return self._optimizer

    def updates(self):
        return self._updates

    @property
    def logger(self):
        return _logger

    @property
    def iterations_per_update(self):
        return self._iterations_per_update

    @iterations_per_update.setter
    def iterations_per_update(self, num):
        if num is None:
            self._iterations_per_update = 1
        else:
            assert isinstance(num, int)
            assert num > 0
            self._iterations_per_update = num

    @action
    def start_semi_supervision(self):
        self.semi_supervision_started = True

    def score_semi_supervision(self, batch, mode='entropy'):
        """Compute the rank scores for SSL for the given batch depending on the mode: 'entropy' or 'diff' for 'diff-proba' metric"""
        batch_input, batch_target = self._prepare_input_and_target(batch)
        batch_ids = self.batch_id_fn(batch) if self.batch_id_fn else None
        
        # Batch timestep probabilities
        batch_output = self._model(batch_input)
        # Batch prediction that will be the label for the sample selected
        batch_decode = self.decoder(batch_output)
        
        # Compute the score of each sample: median of the difference between the top 2 probabilities per timestep
        x, xs = transform_output(batch_output)
        x = F.softmax(x, dim=2)
        xs = xs.numpy()
        
        if mode=='diff':
            best_probas, _ = x.topk(dim=2,k=2)
            scores = best_probas[:,:,0] - best_probas[:,:,1]

        elif mode=='entropy':
            sorted_probas, _ = x.topk(dim=2,k=x.shape[2])
            best_probas_entropy = (sorted_probas[:,:,0]*torch.log(sorted_probas[:,:,0]))[:,:,None]
            all_probas_but_best_entropy = (1-sorted_probas[:,:,1:])*torch.log(1-sorted_probas[:,:,1:])
            #scores = minus real entropy
            scores = torch.cat([best_probas_entropy,all_probas_but_best_entropy],dim=2).sum(dim=2)

        sizes = np.arange(scores.shape[0]) < xs[...,None]
        sizes = torch.tensor(sizes.T).cuda()
        semi_supervised_score = scores.sum(dim=0)/(sizes).float().sum(dim=0)
        semi_supervised_score = semi_supervised_score.cpu().detach().numpy()

        return semi_supervised_score, batch_ids, batch_decode, batch_target

    @action
    def compute_semi_supervision(self,epoch):
        if self.use_semi_supervised and self.semi_supervision_started:
            
            # Frequency of update
            if self.counter_epoch_semi_supervision == self.epoch_frequency_semi_supervision:
                self.counter_epoch_semi_supervision = 0

                # Get the dataset B cf report
                new_ids,new_imgs,new_txts = self.compute_score_semi_supervision(epoch)

                # Fraction of the original dataset
                # To be changed on your usage
                nb_samples = 0 #len(self.original_dataset['ids'])//2
                idx = np.random.randint(0,len(self.original_dataset['ids']),nb_samples)
                ori_ids = [self.original_dataset['ids'][i] for i in idx]
                ori_imgs = [self.original_dataset['imgs'][i] for i in idx]
                ori_txts = [self.original_dataset['txts'][i] for i in idx]
                
                self.data_loader.dataset._ids = ori_ids + new_ids
                self.data_loader.dataset._imgs = ori_imgs + new_imgs
                self.data_loader.dataset._txts = ori_txts + new_txts

            self.counter_epoch_semi_supervision += 1
                

    def compute_score_semi_supervision(self,epoch):
        # Choose mode
        mode = 'entropy' 
        # Batch iterator
        if self._progress_bar:
            batch_iterator = tqdm(
                self.data_semi_supervised_loader,
                desc=self._progress_bar
                if isinstance(self._progress_bar, string_classes)
                else None,
            )
        else:
            batch_iterator = self.data_semi_supervised_loader

        # Compute metric on labelled dataset A
        original_scores = []
        for it, batch in enumerate(self.original_data_loader,1):
            semi_supervised_score ,_ ,_, _ = self.score_semi_supervision(batch,mode)
            original_scores.append(semi_supervised_score)

        original_scores = np.concatenate(original_scores)
        print('Median Percentiles',np.percentile(original_scores,[25,50,75,90,95,99]))
        # Median score on the dataset A
        score_threshold = np.percentile(original_scores,50)

        # If the user has indicated a threshold on the rank score, we use it
        # If not we take the median of the score on the dataset A
        if self.threshold_score_semi_supervised != 0.0:
            score_threshold = self.threshold_score_semi_supervised
        print(score_threshold,"\n")

        # To store the samples to be added to the training set
        new_ids = []
        new_txts = []
        targets = []
        for it, batch in enumerate(batch_iterator, 1):
            semi_supervised_score ,batch_ids ,batch_decode, batch_target = self.score_semi_supervision(batch,mode)
                        
            print('Median Percentiles',np.percentile(semi_supervised_score,[25,50,75,90,95,99]))

            # Compute the ids of samples to be added to the training set
            idx = np.argwhere(semi_supervised_score>score_threshold).reshape(-1)
            # Add ids and new labels
            batch_ids = np.array(batch_ids)
            new_ids += batch_ids[idx].reshape(-1).tolist()
            new_txts += [ [str(self.syms[val]) for val in batch_decode[i] ] for i in idx]
            targets += [ [str(self.syms[val]) for val in batch_target[i] ] for i in idx]

        # Compute the filenames of the new labelled samples
        unlabelled_imgs = self.data_semi_supervised_loader.dataset._imgs
        file_format = unlabelled_imgs[0].split('.')[-1]
        head, tail = os.path.split(unlabelled_imgs[0])
        new_imgs = [os.path.join(head,i+"."+file_format) for i in new_ids]

        # Correction with Spell Checker
        corrected_new_txts = []
        for txt in new_txts:
            txt = ''.join(txt).split('@')
            correction = []
            for word in txt:
                suggestion = self.sym_spell.lookup(word, Verbosity.CLOSEST,max_edit_distance=5,include_unknown=True)[0]
                correction.append(suggestion.term.split(',')[0])
            corrected_new_txts.append(list('@'.join(correction)))

        # print few corrections
        # for txt, corr, target  in zip(new_txts[:50],corrected_new_txts[:50],targets[:50]):
        #     print('\n')
        #     print('output',''.join(txt).split('@'))
        #     print('corrected',''.join(corr).split('@'))
        #     print('target',''.join(target).split('@'))
            
        return new_ids,new_imgs,corrected_new_txts

    def add_evaluator(self, evaluator, when=EPOCH_END, condition=None):
        r"""Add an evaluator to run at the end of each epoch."""
        if evaluator is not None:
            self.add_hook(
                when,
                Hook(condition, evaluator.run)
                if condition is not None
                else evaluator.run,
            )
        return self

    @action
    def run(self):
        r"""Run training """
        assert callable(
            self._batch_input_fn
        ), "batch_input_fn (type: {!r}) is not callable".format(
            str(self._batch_target_fn)
        )
        assert callable(
            self._batch_target_fn
        ), "batch_target_fn (type: {!r}) is not callable".format(
            str(self._batch_target_fn)
        )
        while not self._must_stop:
            if self._cv_number !='':
                print("Cross_validation",self._cv_number)
            self._run_epoch()
        return self

    def _run_iteration(self, batch_n, batch, train_iterations=None):
        batch_input, batch_target = self._prepare_input_and_target(batch)

        action_kwargs = {
            "batch": batch,
            "batch_num": batch_n,
            "epoch": self._epochs,
            "iteration": self._iterations,
            "batch_input": batch_input,
            "batch_target": batch_target,
        }
        self._call_hooks(ITER_START, **action_kwargs)

        if self._must_stop:
            return

        # Make all parameter gradients equal to zero.
        # Note: IT % NIPU = the iteration after a step()
        if self._iterations % self.iterations_per_update == 0:
            self._optimizer.zero_grad()

        # Put model in training mode
        if hasattr(self._model, "train"):
            self._model.train()

        # Run model
        with self.exception_catcher(batch):
            batch_output = self._model(batch_input)

        # Note: These checks are only active when logging level <= DEBUG
        check_inf(
            tensor=batch_output,
            logger=__name__,
            msg="Found {abs_num} ({rel_num:.2%}) INF values in the "
            "model output at epoch {epoch}, batch {batch} (absolute "
            "iteration {iteration})",
            epoch=self._epochs,
            batch=self.batch_id_fn(batch) if self.batch_id_fn else batch,
            iteration=self._iterations,
        )
        check_nan(
            tensor=batch_output,
            logger=__name__,
            msg="Found {abs_num} ({rel_num:.2%}) NAN values in the "
            "model output at epoch {epoch}, batch {batch} (absolute "
            "iteration {iteration})",
            epoch=self._epochs,
            batch=self.batch_id_fn(batch) if self.batch_id_fn else batch,
            iteration=self._iterations,
        )
        
        batch_loss = self.compute_loss(batch, batch_output, batch_target)
        if batch_loss is None:
            return

        # Make the loss and gradients w.r.t. output independent of the number
        # of accumulated iterations.
        if self.iterations_per_update > 1:
            batch_loss /= self.iterations_per_update

        # Compute gradients w.r.t. parameters
        self.logger.debug(
            "Start backward at epoch {}, batch {} (absolute iteration {})",
            self._epochs,
            batch_n,
            self._iterations,
        )
        with self.exception_catcher(batch):
            batch_loss.backward()

        self._iterations += 1

        # Update model parameters.
        if self._iterations % self.iterations_per_update == 0:
            self._updates += 1
            self.logger.debug(
                "Updating parameters at epoch {}, batch {} (absolute iteration {})",
                self._epochs,
                batch_n,
                self._iterations,
            )
            self._optimizer.step()

        action_kwargs["train_iterations"] = self._iterations
        action_kwargs["batch_output"] = batch_output
        action_kwargs["batch_loss"] = batch_loss.item()
        action_kwargs["batch_id"] = self.batch_id_fn(batch) if self.batch_id_fn else None

        self._call_hooks(ITER_END, **action_kwargs)

    def compute_loss(self, batch, batch_output, batch_target):
        with self.exception_catcher(batch):
            kwargs = {}
            if isinstance(self._criterion, Loss) and self.batch_id_fn:
                kwargs = {"batch_ids": self.batch_id_fn(batch)}
            loss = self._criterion(batch_output, batch_target, **kwargs)
            if loss is not None:
                if torch.sum(torch.isnan(loss)).item() > 0:
                    raise ValueError("The loss is NaN")
                if torch.sum(torch.isinf(loss)).item() > 0:
                    raise ValueError("The loss is +/-Inf")
            return loss

    def state_dict(self):
        state = super(Trainer, self).state_dict()
        state["optimizer"] = self._optimizer.state_dict()
        state["updates"] = self._updates
        return state

    def load_state_dict(self, state):
        super(Trainer, self).load_state_dict(state)
        self._optimizer.load_state_dict(state["optimizer"])
        self._updates = state["updates"]
