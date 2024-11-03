```python
|2024-10-28 16:17:29| {
     'Ablation': 0, 'L_windows': 10, 'S_windows': 10, 'bs': 1024,
     'classification': False, 'dataset': cpu, 'debug': False, 'decay': 0.0001,
     'density': 0.1, 'device': cuda, 'epochs': 200, 'experiment': False,
     'log': <utils.logger.Logger object at 0x7f9618fc7da0>, 'logger': None, 'loss_func': L1Loss, 'lr': 0.001,
     'model': ours, 'num_servs': 4500, 'num_times': 64, 'num_users': 142,
     'optim': AdamW, 'path': ./datasets/, 'patience': 20, 'program_test': False,
     'rank': 20, 'record': True, 'retrain': False, 'rounds': 5,
     'seed': 0, 'train_size': 500, 'verbose': 0,
}
|2024-10-28 16:17:29| ********************Experiment Start********************
|2024-10-28 16:37:02| Round=1 BestEpoch= 87 MAE=0.9137 RMSE=3.1257 NMAE=0.2875 NRMSE=0.4528 Training_time=859.7 s 
|2024-10-28 16:37:19| MAE=1.0551 RMSE=3.4282 NMAE=0.3320 NRMSE=0.4966
|2024-10-28 16:37:35| MAE=1.0906 RMSE=3.5273 NMAE=0.3433 NRMSE=0.5111
|2024-10-28 16:37:52| MAE=1.0462 RMSE=3.4377 NMAE=0.3292 NRMSE=0.4980
|2024-10-28 16:38:08| MAE=1.0198 RMSE=3.3713 NMAE=0.3210 NRMSE=0.4885
|2024-10-28 16:38:08| ********************Experiment Results:********************
|2024-10-28 16:38:08| MAE: 1.0251 ± 0.0601
|2024-10-28 16:38:08| RMSE: 3.3780 ± 0.1357
|2024-10-28 16:38:08| NMAE: 0.3226 ± 0.0189
|2024-10-28 16:38:08| NRMSE: 0.4894 ± 0.0197
|2024-10-28 16:38:08| Acc_1: 0.1051 ± 0.0054
|2024-10-28 16:38:08| Acc_5: 0.2098 ± 0.0182
|2024-10-28 16:38:08| Acc_10: 0.3191 ± 0.0264
|2024-10-28 16:38:08| train_time: 1373.4760 ± 489.4696
|2024-10-28 16:38:08| ********************Experiment Success********************
