# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 11:07:54 2021

@author: lansf
"""
import os
from chemprop.args import TrainArgs
from chemprop.train import cross_validate
from vp_model.convert_data import get_antoine_splits
from vp_model.convert_data import get_vp_data_paths
from vp_model.convert_data import VP_Data
from chemprop.train.make_predictions import make_predictions
from chemprop.args import PredictArgs
num_folds=3
save_dir = './vp_splits'
(train_paths, val_paths
 , test_path, antoine_weights) = get_antoine_splits('./splits_o_w_weights', num_folds
                                               , weights=True)
                                                    
(train_full, train_features, val_full, val_features, test_full, test_features
 , fold_dirs, train_weights) = get_vp_data_paths(save_dir, num_folds, weights=True)

npoints = 15
if __name__ == '__main__':
    VP_Data(test_path, npoints
                          , test_full, test_features, extra_T='exact')
    
    for count in range(num_folds):
        VP_Data(val_paths[count], npoints
                          , val_full[count], val_features[count], extra_T='exact')
        vp_function = VP_Data(train_paths[count], npoints
                          , train_full[count], train_features[count]
                          , antoine_weights_path=antoine_weights[count]
                          , train_weights_path=train_weights[count]
                          , extra_T='mid')
        # training arguments 
        train_args = [
            '--save_preds',
            '--show_individual_scores',
            '--metric', 'mse',
            '--extra_metrics', 'r2','rmse',
            '--dataset_type', 'regression',
            '--depth', '4',
            '--seed', '1',
            '--num_workers', '0',
            '--aggregation', 'sum',
            '--epochs', '15', #15
            '--batch_size', '5', #5
            '--final_lr', '0.0001',# '0.0001'
            '--init_lr', '0.00001', #.00001
            '--max_lr', '0.001', #0.001
            '--ffn_hidden_size', '300','300','300',
            '--hidden_size', '300',
            '--data_path', train_full[count],
            '--separate_val_path', val_full[count],
            '--separate_test_path', test_full,
            '--features_path', train_features[count],
            '--data_weights_path', train_weights[count],
            '--separate_val_features_path', val_features[count],
            '--separate_test_features_path', test_features,
            '--save_dir', fold_dirs[count],
            '--no_features_scaling',
            '--custom_func_dir', '.'
        ]
        args=TrainArgs().parse_args(train_args)
        
        mean_score, std_score = cross_validate(args, train_func=vp_function.run_training)
        
        
        predict_args = ['--checkpoint_dir', fold_dirs[count]
                            , '--test_path', val_full[count]
                            , '--features_path', val_features[count]
                            , '--preds_path', os.path.join(fold_dirs[count], 'val_preds.csv')
                            , '--num_workers', '0'
                            , '--no_features_scaling'
                            , '--custom_func_dir', '.'
                            ]
        prediction_args = PredictArgs().parse_args(predict_args)
        make_predictions(args=prediction_args)
        
        predict_args = ['--checkpoint_dir', fold_dirs[count]
                        , '--test_path', train_full[count]
                        , '--features_path', train_features[count]
                        , '--preds_path', os.path.join(fold_dirs[count], 'train_preds.csv')
                        , '--num_workers', '0'
                        , '--no_features_scaling'
                        , '--custom_func_dir', '.'
                        ]
        prediction_args = PredictArgs().parse_args(predict_args)
        make_predictions(args=prediction_args)
        
    predict_args = ['--checkpoint_dir', save_dir
                    , '--test_path', test_full
                    , '--features_path', test_features
                    , '--preds_path', os.path.join(save_dir, 'ensemble_preds.csv')
                    , '--num_workers', '0'
                    , '--no_features_scaling'
                    , '--custom_func_dir', '.'
                    ]
    prediction_args = PredictArgs().parse_args(predict_args)
    make_predictions(args=prediction_args)
    
