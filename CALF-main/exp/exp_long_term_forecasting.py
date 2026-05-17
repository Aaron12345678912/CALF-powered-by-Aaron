from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual, visual_advanced, test_params_flop
from utils.metrics import metric
from utils.cmLoss import cmLoss
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args, self.device).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag, vali_test=False):
        data_set, data_loader = data_provider(self.args, flag, vali_test)
        return data_set, data_loader

    def _select_optimizer(self):
        proj_params = []
        tq_params = []
        main_params = []
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                if '_proj' in n:
                    proj_params.append(p)
                elif 'temporalQuery' in n or 'channelAggregator' in n:
                    tq_params.append(p)
                else:
                    main_params.append(p)
        model_optim = optim.Adam([
            {"params": main_params, "lr": self.args.learning_rate},
            {"params": tq_params, "lr": 0.001},
        ])
        loss_optim = optim.Adam(proj_params, lr=1e-4)

        return model_optim, loss_optim

    def _select_criterion(self):
        criterion = cmLoss(self.args.feature_loss, 
                           self.args.output_loss, 
                           self.args.task_loss, 
                           self.args.task_name, 
                           self.args.feature_w, 
                           self.args.output_w, 
                           self.args.task_w)
        return criterion

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test', vali_test=True)

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim, loss_optim = self._select_optimizer()
        criterion = self._select_criterion()
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=self.args.tmax, eta_min=1e-8)
        scaler = torch.cuda.amp.GradScaler(enabled=self.args.use_amp)#初始化梯度缩放器

        epoch_times = []
        best_val = np.Inf
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            
            max_memory = 0
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                loss_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_cycle = batch_cycle.to(self.device)

                with torch.cuda.amp.autocast(enabled=self.args.use_amp):#训练前向和反向改为AMP流程
                    outputs_dict = self.model(batch_x, cycle_index=batch_cycle)
                    loss_dict = criterion(outputs_dict, batch_y)
                    loss = loss_dict['total_loss']

                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | total_loss: {2:.7f} | task_loss: {3:.7f} | output_loss: {4:.7f} | feature_loss: {5:.7f}".format(
                        i + 1, epoch + 1,
                        loss_dict['total_loss'].item(),
                        loss_dict['task_loss'].item(),
                        loss_dict['output_loss'].item(),
                        loss_dict['feature_loss'].item()
                    ))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                scaler.scale(loss).backward()#反向优化
                scaler.step(model_optim)
                scaler.step(loss_optim)
                scaler.update()
                
                current_memory = torch.cuda.max_memory_allocated() / 1024 ** 2
                max_memory = max(max_memory, current_memory)
            
            t = time.time() - epoch_time
            print("Epoch: {} cost time: {:4f} s".format(epoch + 1, t))
            epoch_times.append(t)
            
            train_loss = np.average(train_loss)

            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            # save best model in memory only (no disk I/O)
            if vali_loss < best_val:
                best_val = vali_loss
                self.best_model_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                if early_stopping.verbose:
                    print(f"Best model updated in memory (Vali Loss: {best_val:.6f})")

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))

            if self.args.cos:
                scheduler.step()
                print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
            else:
                adjust_learning_rate(model_optim, epoch + 1, self.args)

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
        
        print("平均训练时间: {:4f} s".format(np.average(epoch_times)))
        
        # load best model from memory (no disk I/O)
        if hasattr(self, 'best_model_state') and self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            if early_stopping.verbose:
                print("Best model loaded from memory.")
        else:
            if early_stopping.verbose:
                print("No best model in memory; using current model state.")
        
        print(f"Max Memory (MB): {max_memory}")
        
        return self.model

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []

        self.model.eval()

        with torch.no_grad(): 
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_cycle = batch_cycle.to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                outputs = self.model(batch_x, cycle_index=batch_cycle)
                outputs_ensemble = outputs['outputs_time'] 
                outputs_ensemble = outputs_ensemble[:, -self.args.pred_len:, :]
                
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)

                pred = outputs_ensemble.detach().cpu()
                true = batch_y.detach().cpu()

                loss = F.mse_loss(pred, true)

                total_loss.append(loss)

        total_loss = np.average(total_loss)

        self.model.train()

        return total_loss

    def test(self, setting, test=0):
        # zero shot
        if self.args.zero_shot:
            self.args.data = self.args.target_data
            self.args.data_path = f"{self.args.data}.csv"

        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            # load best model from memory (no disk I/O)
            if hasattr(self, 'best_model_state') and self.best_model_state is not None:
                print('Loading best model from memory...')
                self.model.load_state_dict(self.best_model_state)
            else:
                print("No best model in memory; using current model state.")

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_cycle = batch_cycle.to(self.device)

                outputs = self.model(batch_x[:, -self.args.seq_len:, :], cycle_index=batch_cycle)

                outputs_ensemble = outputs['outputs_time']
                outputs_ensemble = outputs_ensemble[:, -self.args.pred_len:, :]
                
                batch_y = batch_y[:, -self.args.pred_len:, :]

                pred = outputs_ensemble.detach().cpu().numpy()
                true = batch_y.detach().cpu().numpy()

                preds.append(pred)
                trues.append(true)

        # # # 模型参数量：
        # x = batch_x[0,:,:].unsqueeze(0)
        # test_params_flop(self.model, (x,))
        # # x = batch_x[0,:,:].unsqueeze(0).unsqueeze(0)
        # # test_params_flop(self.model, x) 
        y = (batch_x.shape[-2], batch_x.shape[-1])
        test_params_flop(self.model, y) 
        
        # preds = np.array(preds)
        # trues = np.array(trues)
        preds = np.concatenate(preds, axis=0) # without the "drop-last" trick
        trues = np.concatenate(trues, axis=0) # without the "drop-last" trick
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)
        

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()

        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        # np.save(folder_path + 'pred.npy', preds)
        # np.save(folder_path + 'true.npy', trues)

        # === 可视化：对第一个电站(channel 0)画出高级预测比对图 ===
        vis_folder = os.path.join(folder_path, 'visualization')
        if not os.path.exists(vis_folder):
            os.makedirs(vis_folder)

        windows = [0, 25, 50]
        trues_windows = []
        preds_windows = []
        
        for idx in windows:
            if idx < preds.shape[0]:
                if hasattr(test_data, 'inverse_transform'):
                    p = test_data.inverse_transform(preds[idx])[:, 0]
                    t = test_data.inverse_transform(trues[idx])[:, 0]
                else:
                    p = preds[idx][:, 0]
                    t = trues[idx][:, 0]
                trues_windows.append(t)
                preds_windows.append(p)
            else:
                # Mock if fewer samples
                trues_windows.append(np.zeros(preds.shape[1]))
                preds_windows.append(np.zeros(preds.shape[1]))

        if len(trues_windows) > 0:
            save_path = os.path.join(vis_folder, 'advanced_ablation_station_0.pdf')
            visual_advanced(trues_windows, preds_windows, metrics_full=(mse, mae), name=save_path)
            print(f'Advanced Visualization saved: {save_path}')

        # 额外画一个包含多个子图的综合图（4个电站）
        fig, axes = plt.subplots(2, 2, figsize=(14, 8))
        stations = [0, 1, 2, 3]  # 画前4个电站
        sample_idx = 0
        for idx, ch in enumerate(stations):
            ax = axes[idx // 2][idx % 2]
            if hasattr(test_data, 'inverse_transform'):
                pred_ch = test_data.inverse_transform(preds[sample_idx])[:, ch]
                true_ch = test_data.inverse_transform(trues[sample_idx])[:, ch]
            else:
                pred_ch = preds[sample_idx][:, ch]
                true_ch = trues[sample_idx][:, ch]
            ax.plot(true_ch, label='GroundTruth', linewidth=2, color='#2ca02c')
            ax.plot(pred_ch, label='Prediction', linewidth=2, color='#d62728', linestyle='--')
            ax.set_title(f'Station {ch}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        multi_save_path = os.path.join(vis_folder, f'multi_station_sample_{sample_idx}.pdf')
        plt.savefig(multi_save_path, bbox_inches='tight')
        plt.close()
        print(f'Multi-station visualization saved: {multi_save_path}')
        # ================================================================

        return
