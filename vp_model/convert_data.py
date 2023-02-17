# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 11:07:54 2021

@author: lansf
"""
import json
from logging import Logger
import os
from typing import Dict, List
from copy import deepcopy

import numpy as np
import pandas as pd
from tensorboardX import SummaryWriter
import torch
from tqdm import trange
from torch.optim.lr_scheduler import ExponentialLR

from chemprop.train.evaluate import evaluate, evaluate_predictions
from chemprop.train.predict import predict
from chemprop.train import train
from chemprop.spectra_utils import normalize_spectra, load_phase_mask
from chemprop.args import TrainArgs
from chemprop.constants import MODEL_FILE_NAME
from chemprop.data import get_class_sizes, get_data, MoleculeDataLoader, set_cache_graph, MoleculeDataset
from chemprop.models import MoleculeModel
from chemprop.nn_utils import param_count, param_count_all
from chemprop.utils import build_optimizer, build_lr_scheduler, get_loss_func, load_checkpoint, makedirs, \
    save_checkpoint, save_smiles_splits, load_frzn_model 
from numpy.random import default_rng

def split_antoine_data(input_path, save_dir, split_tuple, num_folds=1, seed=1
               , weights_path=None):
    if os.path.exists(save_dir) == False:
        os.mkdir(save_dir)
    data = pd.read_csv(input_path)
    seed_list = [seed + i for i in range(num_folds)]
    rng = default_rng(seed_list[0])
    indices = list(range(data.shape[0]))
    rng.shuffle(indices)
    train_size = int(split_tuple[0] * data.shape[0])
    train_val_size = int((split_tuple[0] + split_tuple[1]) * data.shape[0])
    test_data = data.iloc[indices[train_val_size:]]
    test_path = os.path.join(save_dir, 'antoine_test.csv')
    test_data.to_csv(test_path, index=False)
    data = data.iloc[indices[0:train_val_size]]
    if weights_path is not None:
        data_weights = pd.read_csv(weights_path)
        data_weights = data_weights.iloc[indices[0:train_val_size]]
    train_paths = []
    antoine_weights_paths = []
    val_paths = []
    fold_dirs = []
    for i in range(num_folds):
        fold_dirs.append(os.path.join(save_dir,'fold_'+str(i)))
        if os.path.exists(fold_dirs[i]) == False:
            os.mkdir(fold_dirs[i])
        rng = default_rng(seed_list[i])
        indices = list(range(data.shape[0]))
        rng.shuffle(indices)
        train = data.iloc[indices[0:train_size]]
        if weights_path is not None:
            weights_train = data_weights.iloc[indices[0:train_size]]
            antoine_weights_paths.append(os.path.join(fold_dirs[i],'antoine_weights.csv'))
            weights_train.to_csv(antoine_weights_paths[-1], index=False)
        val = data.iloc[indices[train_size:]]
        train_paths.append(os.path.join(fold_dirs[i],'antoine_train.csv'))
        val_paths.append(os.path.join(fold_dirs[i],'antoine_val.csv'))
        train.to_csv(train_paths[-1], index=False)
        val.to_csv(val_paths[-1], index=False)
    
def get_antoine_splits(antoine_dir, num_folds=1, weights=False):
    test_path = os.path.join(antoine_dir, 'antoine_test.csv')
    train_paths = []
    antoine_weights_paths = []
    val_paths = []
    fold_dirs = []
    for i in range(num_folds):
        fold_dirs.append(os.path.join(antoine_dir,'fold_'+str(i)))
        if weights == True:
            antoine_weights_paths.append(os.path.join(fold_dirs[i],'antoine_weights.csv'))
        train_paths.append(os.path.join(fold_dirs[i],'antoine_train.csv'))
        val_paths.append(os.path.join(fold_dirs[i],'antoine_val.csv'))
    if weights == False:
        return (train_paths, val_paths, test_path)
    else:
        return (train_paths, val_paths, test_path, antoine_weights_paths)
    
def get_vp_data_paths(save_dir, num_folds=1, weights=False):
    if os.path.exists(save_dir) == False:
        os.mkdir(save_dir)
    test_full = os.path.join(save_dir, 'test_full.csv')
    test_features = os.path.join(save_dir, 'test_features.csv')
    train_full = []
    train_features = []
    train_weights = []
    val_full = []
    val_features = []
    fold_dirs = []
    for i in range(num_folds):
        fold_dirs.append(os.path.join(save_dir,'fold_'+str(i)))
        if os.path.exists(fold_dirs[i]) == False:
            os.mkdir(fold_dirs[i])
        if weights == True:
            train_weights.append(os.path.join(fold_dirs[i],'train_weights.csv'))
        train_full.append(os.path.join(fold_dirs[i],'train_full.csv'))
        val_full.append(os.path.join(fold_dirs[i],'val_full.csv'))
        train_features.append(os.path.join(fold_dirs[i],'train_features.csv'))
        val_features.append(os.path.join(fold_dirs[i],'val_features.csv'))
    
    if weights == False:
        return (train_full, train_features, val_full, val_features
                , test_full, test_features, fold_dirs)
    else:
        return (train_full, train_features, val_full, val_features
                , test_full, test_features, fold_dirs, train_weights)

class VP_Data():
    """A :class:`VP_data` is a class for generating vapor pressure data."""

    def __init__(self, antoine_path: str, npoints: int
                 , data_path: str, features_path: str
                 , antoine_weights_path=None, train_weights_path=None
                 , extrap_lower=0, extrap_upper=0
                 , log10Pmin=-30, log10Pmax=5, extra_T=None):
        """
        
        Parameters
        ----------
        antoine_path : str
            antoine vp data of A, B, and C parameters.
        npoints : int
            number of pressure values to calculate per molecule.
        data_path : str
            output data path.
        features_path : str
            output features path.
        extra_T: str
            Extra temperature feature can be None, 'mid', 'exact'

        Returns
        -------
        None.

        """
        self.antoine_path = antoine_path
        self.antoine_weights_path = antoine_weights_path
        self.npoints = npoints
        self.data_path = data_path
        self.features_path = features_path
        self.train_weights_path = train_weights_path
        self.random = False
        self.extrap_lower = extrap_lower
        self.extrap_upper = extrap_upper
        self.log10Pmin = log10Pmin
        self.log10Pmax = log10Pmax
        self.extra_T = extra_T
        self.rng_temp_2 = default_rng(1)
        self.vp_data_from_antoine()
        
    def set_random_state(self, random_state: int):
        """
        
        Parameters
        ----------
        random : TYPE, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        None.

        """
        self.rng = default_rng(random_state)
        self.random = True
        
    def vp_data_from_antoine(self):
        df = pd.read_csv(self.antoine_path)
        if self.antoine_weights_path is not None:
            weights = pd.read_csv(self.antoine_weights_path)
            weights = weights[weights.columns[0]].to_list()
            weights_list = []
        SMILES = []
        TEMPERATURE = []
        EXTRA_T = []
        log10atm = []
        for count in range(df.shape[0]):
            TMIN = df['TMIN [C]'][count] + 273.15 + self.extrap_lower
            TMAX = df['TMAX [C]'][count]  + 273.15 + self.extrap_upper
            if self.random == True:
                T_LIST = self.rng.uniform(TMIN, TMAX, self.npoints)
            else:
                T_LIST = np.linspace(TMIN, TMAX, num=self.npoints, endpoint=True)
            for T in T_LIST:
                divider = T - 273.15 + df['C [C]'][count]
                if divider !=0:
                    P = (df['A [log10(atm)]'][count]
                        - df['B [C]'][count]
                        / divider)
                    if P > self.log10Pmin and P < self.log10Pmax:
                        TEMPERATURE.append(T)
                        SMILES.append(df['smiles'][count])
                        log10atm.append(P)
                        if self.extra_T == 'mixed':
                            r_n = self.rng_temp_2.integers(low=0, high=3)
                            if r_n == 0:
                                EXTRA_T.append(T)
                            elif r_n == 1:
                                EXTRA_T.append(0.5 * TMIN + 0.5 * TMAX)
                            else:
                                EXTRA_T.append(self.rng_temp_2.uniform(TMIN, TMAX))
                        elif self.extra_T == 'exact':
                            EXTRA_T.append(T)
                        elif self.extra_T == 'mid': 
                            EXTRA_T.append(0.5 * TMIN + 0.5 * TMAX)
                        elif self.extra_T == 'random':
                            EXTRA_T.append(self.rng_temp_2.uniform(TMIN, TMAX))
                        if self.antoine_weights_path is not None:
                            weights_list.append(weights[count])
        
        data = pd.DataFrame(zip(SMILES,log10atm), columns=['smiles','log10atm'])
        features = pd.DataFrame(TEMPERATURE, columns = ['Temp [K]'])
        if self.extra_T is not None:
            features['Extra T [K]'] = EXTRA_T
        data.to_csv(self.data_path, index=False)
        features.to_csv(self.features_path, index=False)
        if self.antoine_weights_path is not None:
            data_weights = pd.DataFrame(weights_list, columns=['weights'])
            data_weights.to_csv(self.train_weights_path, index=False)
        
    def run_training(self, args: TrainArgs,
                 data: MoleculeDataset,
                 logger: Logger = None,
                 model_list: List[MoleculeModel] = None) -> Dict[str, List[float]]:
        """
        Loads data, trains a Chemprop model, and returns test scores for the model checkpoint with the highest validation score.
    
        :param args: A :class:`~chemprop.args.TrainArgs` object containing arguments for
                     loading data and training the Chemprop model.
        :param data: A :class:`~chemprop.data.MoleculeDataset` containing the data.
        :param logger: A logger to record output.
        :param model_list: A list of :class:`~chemprop.models.model.MoleculeModel`.
        :return: A dictionary mapping each metric in :code:`args.metrics` to a list of values for each task.
    
        """
        if logger is not None:
            debug, info = logger.debug, logger.info
        else:
            debug = info = print
    
        # Set pytorch seed for random initial weights
        torch.manual_seed(args.pytorch_seed)
        # Split data
        debug(f'Splitting data with seed {args.seed}')

        train_data = data
        test_data = get_data(path=args.separate_test_path,
                             args=args,
                             features_path=args.separate_test_features_path,
                             atom_descriptors_path=args.separate_test_atom_descriptors_path,
                             bond_features_path=args.separate_test_bond_features_path,
                             phase_features_path=args.separate_test_phase_features_path,
                             smiles_columns=args.smiles_columns,
                             logger=logger)

        val_data = get_data(path=args.separate_val_path,
                            args=args,
                            features_path=args.separate_val_features_path,
                            atom_descriptors_path=args.separate_val_atom_descriptors_path,
                            bond_features_path=args.separate_val_bond_features_path,
                            phase_features_path=args.separate_val_phase_features_path,
                            smiles_columns = args.smiles_columns,
                            logger=logger)
        
    
        if args.dataset_type == 'classification':
            class_sizes = get_class_sizes(data)
            debug('Class sizes')
            for i, task_class_sizes in enumerate(class_sizes):
                debug(f'{args.task_names[i]} '
                      f'{", ".join(f"{cls}: {size * 100:.2f}%" for cls, size in enumerate(task_class_sizes))}')
    
        if args.save_smiles_splits:
            save_smiles_splits(
                data_path=args.data_path,
                save_dir=args.save_dir,
                task_names=args.task_names,
                features_path=args.features_path,
                train_data=train_data,
                val_data=val_data,
                test_data=test_data,
                smiles_columns=args.smiles_columns,
                logger=logger,
            )
    
        if args.features_scaling:
            features_scaler = train_data.normalize_features(replace_nan_token=0)
            val_data.normalize_features(features_scaler)
            test_data.normalize_features(features_scaler)
        else:
            features_scaler = None
    
        if args.atom_descriptor_scaling and args.atom_descriptors is not None:
            atom_descriptor_scaler = train_data.normalize_features(replace_nan_token=0, scale_atom_descriptors=True)
            val_data.normalize_features(atom_descriptor_scaler, scale_atom_descriptors=True)
            test_data.normalize_features(atom_descriptor_scaler, scale_atom_descriptors=True)
        else:
            atom_descriptor_scaler = None
    
        if args.bond_feature_scaling and args.bond_features_size > 0:
            bond_feature_scaler = train_data.normalize_features(replace_nan_token=0, scale_bond_features=True)
            val_data.normalize_features(bond_feature_scaler, scale_bond_features=True)
            test_data.normalize_features(bond_feature_scaler, scale_bond_features=True)
        else:
            bond_feature_scaler = None
    
        args.train_data_size = len(train_data)
    
        debug(f'Total size = {len(data):,} | '
              f'train size = {len(train_data):,} | val size = {len(val_data):,} | test size = {len(test_data):,}')
    
        # Initialize scaler and scale training targets by subtracting mean and dividing standard deviation (regression only)
        if args.dataset_type == 'regression':
            debug('Fitting scaler')
            scaler = train_data.normalize_targets()
        elif args.dataset_type == 'spectra':
            debug('Normalizing spectra and excluding spectra regions based on phase')
            args.spectra_phase_mask = load_phase_mask(args.spectra_phase_mask_path)
            for dataset in [train_data, test_data, val_data]:
                data_targets = normalize_spectra(
                    spectra=dataset.targets(),
                    phase_features=dataset.phase_features(),
                    phase_mask=args.spectra_phase_mask,
                    excluded_sub_value=None,
                    threshold=args.spectra_target_floor,
                )
                dataset.set_targets(data_targets)
            scaler = None
        else:
            scaler = None
    
        # Get loss function
        loss_func = get_loss_func(args)
    
        # Set up test set evaluation
        test_smiles, test_targets = test_data.smiles(), test_data.targets()
        if args.dataset_type == 'multiclass':
            sum_test_preds = np.zeros((len(test_smiles), args.num_tasks, args.multiclass_num_classes))
        else:
            sum_test_preds = np.zeros((len(test_smiles), args.num_tasks))
    
        # Automatically determine whether to cache
        if len(data) <= args.cache_cutoff:
            set_cache_graph(True)
            num_workers = 0
        else:
            set_cache_graph(False)
            num_workers = args.num_workers
    
        # Create data loaders
        train_data_loader = MoleculeDataLoader(
            dataset=train_data,
            batch_size=args.batch_size,
            num_workers=num_workers,
            class_balance=args.class_balance,
            shuffle=True,
            seed=args.seed
        )
        val_data_loader = MoleculeDataLoader(
            dataset=val_data,
            batch_size=args.batch_size,
            num_workers=num_workers
        )
        test_data_loader = MoleculeDataLoader(
            dataset=test_data,
            batch_size=args.batch_size,
            num_workers=num_workers
        )
    
        if args.class_balance:
            debug(f'With class_balance, effective train size = {train_data_loader.iter_size:,}')
    
        # Train ensemble of models
        for model_idx in range(args.ensemble_size):
            # Tensorboard writer
            save_dir = os.path.join(args.save_dir, f'model_{model_idx}')
            makedirs(save_dir)
            try:
                writer = SummaryWriter(log_dir=save_dir)
            except:
                writer = SummaryWriter(logdir=save_dir)
    
            # Load/build model
            if model_list is not None:
                model = deepcopy(model_list[model_idx])
            elif args.checkpoint_paths is not None:
                debug(f'Loading model {model_idx} from {args.checkpoint_paths[model_idx]}')
                model = load_checkpoint(args.checkpoint_paths[model_idx], logger=logger)
            else:
                debug(f'Building model {model_idx}')
                model = MoleculeModel(args)
                
            # Optionally, overwrite weights:
            if args.checkpoint_frzn is not None:
                debug(f'Loading and freezing parameters from {args.checkpoint_frzn}.')
                model = load_frzn_model(model=model,path=args.checkpoint_frzn, current_args=args, logger=logger)     
            
            debug(model)
            
            if args.checkpoint_frzn is not None:
                debug(f'Number of unfrozen parameters = {param_count(model):,}')
                debug(f'Total number of parameters = {param_count_all(model):,}')
            else:
                debug(f'Number of parameters = {param_count_all(model):,}')
            
            if args.cuda:
                debug('Moving model to cuda')
            model = model.to(args.device)
    
            # Ensure that model is saved in correct location for evaluation if 0 epochs
            save_checkpoint(os.path.join(save_dir, MODEL_FILE_NAME), model, scaler,
                            features_scaler, atom_descriptor_scaler, bond_feature_scaler, args)
    
            # Optimizers
            optimizer = build_optimizer(model, args)
    
            # Learning rate schedulers
            scheduler = build_lr_scheduler(optimizer, args)
    
            # Run training
            best_score = float('inf') if args.minimize_score else -float('inf')
            best_epoch, n_iter = 0, 0 
            for epoch in trange(args.epochs):
                if epoch > 0 and self.random == True:
                    self.vp_data_from_antoine()
                    train_data = get_data(
                        path=args.data_path,
                        args=args,
                        logger=logger,
                        skip_none_targets=True,
                        data_weights_path=args.data_weights_path
                    )
                    
                    if args.features_scaling:
                        train_data.normalize_features(features_scaler)
                
                    if args.atom_descriptor_scaling and args.atom_descriptors is not None:
                        train_data.normalize_features(atom_descriptor_scaler, scale_atom_descriptors=True)
                
                    if args.bond_feature_scaling and args.bond_features_size > 0:
                        train_data.normalize_features(bond_feature_scaler, scale_bond_features=True)
                    
                    if args.dataset_type == 'regression':
                        debug('Fitting scaler')
                        train_data.normalize_targets(scaler)
                    
                    train_data_loader = MoleculeDataLoader(
                        dataset=train_data,
                        batch_size=args.batch_size,
                        num_workers=num_workers,
                        class_balance=args.class_balance,
                        shuffle=True,
                        seed=args.seed
                    )
                debug(f'Epoch {epoch}')
                n_iter = train(
                    model=model,
                    data_loader=train_data_loader,
                    loss_func=loss_func,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    args=args,
                    n_iter=n_iter,
                    logger=logger,
                    writer=writer
                )
                if isinstance(scheduler, ExponentialLR):
                    scheduler.step()
                val_scores = evaluate(
                    model=model,
                    data_loader=val_data_loader,
                    num_tasks=args.num_tasks,
                    metrics=args.metrics,
                    dataset_type=args.dataset_type,
                    scaler=scaler,
                    logger=logger
                )
    
                for metric, scores in val_scores.items():
                    # Average validation score
                    avg_val_score = np.nanmean(scores)
                    debug(f'Validation {metric} = {avg_val_score:.6f}')
                    writer.add_scalar(f'validation_{metric}', avg_val_score, n_iter)
    
                    if args.show_individual_scores:
                        # Individual validation scores
                        for task_name, val_score in zip(args.task_names, scores):
                            debug(f'Validation {task_name} {metric} = {val_score:.6f}')
                            writer.add_scalar(f'validation_{task_name}_{metric}', val_score, n_iter)
    
                # Save model checkpoint if improved validation score
                avg_val_score = np.nanmean(val_scores[args.metric])
                if args.minimize_score and avg_val_score < best_score or \
                        not args.minimize_score and avg_val_score > best_score:
                    best_score, best_epoch = avg_val_score, epoch
                    save_checkpoint(os.path.join(save_dir, MODEL_FILE_NAME), model, scaler,
                            features_scaler, atom_descriptor_scaler, bond_feature_scaler, args)
    
            # Evaluate on test set using model with best validation score
            info(f'Model {model_idx} best validation {args.metric} = {best_score:.6f} on epoch {best_epoch}')
            model = load_checkpoint(os.path.join(save_dir, MODEL_FILE_NAME)
                                    , device=args.device, logger=logger)
    
            test_preds = predict(
                model=model,
                data_loader=test_data_loader,
                scaler=scaler
            )
            test_scores = evaluate_predictions(
                preds=test_preds,
                targets=test_targets,
                num_tasks=args.num_tasks,
                metrics=args.metrics,
                dataset_type=args.dataset_type,
                logger=logger
            )
    
            if len(test_preds) != 0:
                sum_test_preds += np.array(test_preds)
    
            # Average test score
            for metric, scores in test_scores.items():
                avg_test_score = np.nanmean(scores)
                info(f'Model {model_idx} test {metric} = {avg_test_score:.6f}')
                writer.add_scalar(f'test_{metric}', avg_test_score, 0)
    
                if args.show_individual_scores and args.dataset_type != 'spectra':
                    # Individual test scores
                    for task_name, test_score in zip(args.task_names, scores):
                        info(f'Model {model_idx} test {task_name} {metric} = {test_score:.6f}')
                        writer.add_scalar(f'test_{task_name}_{metric}', test_score, n_iter)
            writer.close()
    
        # Evaluate ensemble on test set
        avg_test_preds = (sum_test_preds / args.ensemble_size).tolist()
    
        ensemble_scores = evaluate_predictions(
            preds=avg_test_preds,
            targets=test_targets,
            num_tasks=args.num_tasks,
            metrics=args.metrics,
            dataset_type=args.dataset_type,
            logger=logger
        )
    
        for metric, scores in ensemble_scores.items():
            # Average ensemble score
            avg_ensemble_test_score = np.nanmean(scores)
            info(f'Ensemble test {metric} = {avg_ensemble_test_score:.6f}')
    
            # Individual ensemble scores
            if args.show_individual_scores:
                for task_name, ensemble_score in zip(args.task_names, scores):
                    info(f'Ensemble test {task_name} {metric} = {ensemble_score:.6f}')
    
        # Save scores
        with open(os.path.join(args.save_dir, 'test_scores.json'), 'w') as f:
            json.dump(ensemble_scores, f, indent=4, sort_keys=True)
    
        # Optionally save test preds
        if args.save_preds:
            test_preds_dataframe = pd.DataFrame(data={'smiles': test_data.smiles()})
    
            for i, task_name in enumerate(args.task_names):
                test_preds_dataframe[task_name] = [pred[i] for pred in avg_test_preds]
    
            test_preds_dataframe.to_csv(os.path.join(args.save_dir, 'test_preds.csv'), index=False)
    
        return ensemble_scores