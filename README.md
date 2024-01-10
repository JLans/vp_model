# The Vapor Pressure Toolkit (VPT)

VPT is a toolkit for building Chemprop message-passing neural network models to predict vapor pressure via vp_model.
The necessary chemprop package can be obtained via the command "git clone --branch transfer_learning https://github.com/JLans/chemprop.git".

## Table of Contents
[Background and description](#bckgrd_dscrpts)

[Install package](#install_package)

[Examples](#examples)

* [Split data](#split)

* [Run model](#run)

[Credits](#credits)

[License](#license)

## <a name="bckgrd_dscrpts"/></a>Background and description
This packages enables building physics-informed vapor pressure models.
## <a name="install_package"/></a>Install package
A setup.py file is provided for installation from source.
```
cd vp_model
pip install .
```

## <a name="examples"/></a>Examples
See *examples* folder.

### <a name="split"/></a>Splitting data
Split the data into training, validation, and test sets
```python
from vp_model.convert_data import split_antoine_data
num_folds = 3
Yaw_liquid_path = '../data/Yaw_vp_o_data_liquids.csv'
save_dir = './splits_o_w_weights'
split_antoine_data(Yaw_liquid_path, save_dir, (0.8, 0.1, 0.1), num_folds, seed=1
                   , weights_path='../vp_model/data/sim_max_liquids_o.csv')
```

### <a name="runt"/></a>Run a vapor pressure model
Import relevant modules
```python
import os
from chemprop.args import TrainArgs
from chemprop.train import cross_validate
from vp_model.convert_data import get_antoine_splits
from vp_model.convert_data import get_vp_data_paths
from vp_model.convert_data import VP_Data
from chemprop.train.make_predictions import make_predictions
from chemprop.args import PredictArgs
```

Get relevant data paths
```python
import os
(train_paths, val_paths
 , test_path, antoine_weights) = get_antoine_splits('./splits_o_w_weights', num_folds
                                               , weights=True)
                                                    
(train_full, train_features, val_full, val_features, test_full, test_features
 , fold_dirs, train_weights) = get_vp_data_paths(save_dir, num_folds, weights=True)

```

Generate a vapor pressure data class and run the model
```python
npoints = 15 #Number of temperature values to compute vapor pressure per molecule
num_folds=3
if __name__ == '__main__':
    #Generate data from Antoine parameters
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
            '--dataset_type', 'regression',
            '--depth', '4',
            '--aggregation', 'sum',
            '--epochs', '15',
            '--batch_size', '5',
            '--final_lr', '0.0001',
            '--init_lr', '0.00001',
            '--max_lr', '0.001', 
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
            '--no_features_scaling', #should not scale temperature
            '--custom_func_dir', '.' #custom_func.py is imported by the model.
        ]
        args=TrainArgs().parse_args(train_args)
        mean_score, std_score = cross_validate(args, train_func=vp_function.run_training)

```

## <a name="credits"/></a>Credits
See publication for details.

Contributors:

Joshua L. Lansford <br />
Brian C. Barnes

## <a name="license"/></a>License
This project is licensed under the [MIT](https://opensource.org/licenses/MIT) license.
