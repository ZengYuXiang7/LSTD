```python
|2024-10-28 01:49:31| {
     'Ablation': 0, 'L_windows': 10, 'S_windows': 10, 'bs': 1024,
     'classification': False, 'dataset': cpu, 'debug': False, 'decay': 0.0001,
     'density': 0.1, 'device': cuda, 'epochs': 200, 'experiment': False,
     'log': <utils.logger.Logger object at 0x7f67af948f20>, 'logger': None, 'loss_func': L1Loss, 'lr': 0.001,
     'model': ours, 'num_servs': 4500, 'num_times': 64, 'num_users': 142,
     'optim': AdamW, 'path': ./datasets/, 'patience': 20, 'program_test': False,
     'rank': 20, 'record': True, 'retrain': False, 'rounds': 5,
     'seed': 0, 'train_size': 500, 'verbose': 0,

|2024-10-28 01:49:31| ********************Experiment Start********************
|2024-10-28 02:09:25| Round=1 BestEpoch= 90 MAE=1.0128 RMSE=3.3702 NMAE=0.3187 NRMSE=0.4882 Training_time=877.4 s 
|2024-10-28 02:09:42| MAE=1.0551 RMSE=3.4282 NMAE=0.3320 NRMSE=0.4966
|2024-10-28 02:09:58| MAE=1.0906 RMSE=3.5273 NMAE=0.3433 NRMSE=0.5111
|2024-10-28 02:10:14| MAE=1.0462 RMSE=3.4377 NMAE=0.3292 NRMSE=0.4980
|2024-10-28 02:10:31| MAE=1.0198 RMSE=3.3713 NMAE=0.3210 NRMSE=0.4885
|2024-10-28 02:10:31| ********************Experiment Results:********************
|2024-10-28 02:10:31| MAE: 1.0449 ± 0.0278
|2024-10-28 02:10:31| RMSE: 3.4269 ± 0.0575
|2024-10-28 02:10:31| NMAE: 0.3289 ± 0.0087
|2024-10-28 02:10:31| NRMSE: 0.4965 ± 0.0083
|2024-10-28 02:10:31| Acc_1: 0.1035 ± 0.0026
|2024-10-28 02:10:31| Acc_5: 0.2052 ± 0.0099
|2024-10-28 02:10:31| Acc_10: 0.3131 ± 0.0153
|2024-10-28 02:10:31| train_time: 1377.0153 ± 485.7922
|2024-10-28 02:10:31| ********************Experiment Success********************
