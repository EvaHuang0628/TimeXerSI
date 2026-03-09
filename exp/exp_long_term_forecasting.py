from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from utils.dtw_metric import dtw, accelerated_dtw
from utils.augmentation import run_augmentation, run_augmentation_single

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        if self.args.loss == 'MAE':
            criterion = nn.L1Loss()
        elif self.args.loss == 'Huber':
            criterion = nn.HuberLoss(delta=1.0, reduction='mean')
        else:
            criterion = nn.MSELoss()
        return criterion

    # new function
    def _get_loss(self, pred, true, criterion):
        """
        Calculates the loss. It can be a standard loss or a custom weighted loss
        based on command-line arguments.
        """
        # Check for custom weighted loss for pred_len=336 and if weights are provided
        if self.args.pred_len == 336 and self.args.loss_weights and len(self.args.loss_weights) == 2:
            # Define the split point
            split_pos = self.args.pred_len // 2
            alpha = self.args.loss_weights[0]
            beta = self.args.loss_weights[1]

            # Calculate loss for each part
            loss_first_half = criterion(pred[:, :split_pos, :], true[:, :split_pos, :])
            loss_second_half = criterion(pred[:, split_pos:, :], true[:, split_pos:, :])

            # Return the weighted combination
            return alpha * loss_first_half + beta * loss_second_half
        else:
            # Default behavior: standard loss over the whole prediction window
            return criterion(pred, true)

    def _calculate_dynamic_loss(self, pred, true, criterion, epoch):
        """
        Calculates loss with dynamic weighting based on the current epoch.
        The weights transition from focusing on short-term predictions to
        balancing short-term and long-term predictions.
        """
        # 1. Determine current weights based on epoch
        total_epochs = self.args.train_epochs
        warmup_epochs = total_epochs * self.args.loss_warmup_ratio
        transition_duration = total_epochs * self.args.loss_transition_ratio
        transition_start_epoch = warmup_epochs
        transition_end_epoch = transition_start_epoch + transition_duration

        if epoch < transition_start_epoch:
            # Stage 1: Warm-up phase with initial weights
            w_short, w_long = self.args.loss_weights_initial
        elif epoch >= transition_end_epoch:
            # Stage 3: Final weights phase
            w_short, w_long = self.args.loss_weights_final
        else:
            # Stage 2: Transition phase with linear interpolation
            w_s_initial, w_l_initial = self.args.loss_weights_initial
            w_s_final, w_l_final = self.args.loss_weights_final

            progress = (epoch - transition_start_epoch) / transition_duration if transition_duration > 0 else 1.0

            w_short = w_s_initial + (w_s_final - w_s_initial) * progress
            w_long = w_l_initial + (w_l_final - w_l_initial) * progress

        # 2. Define the split point for short-term and long-term
        split_len = self.args.loss_split_len if self.args.loss_split_len is not None else self.args.pred_len // 2

        # 3. Calculate weighted loss if split is valid
        if 0 < split_len < self.args.pred_len:
            loss_short = criterion(pred[:, :split_len, :], true[:, :split_len, :])
            loss_long = criterion(pred[:, split_len:, :], true[:, split_len:, :])
            return w_short * loss_short + w_long * loss_long
        else:
            # Fallback to standard loss if split is not valid
            return criterion(pred, true)

    def vali(self, vali_data, vali_loader, criterion, epoch=0):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            # TimeXer_Adp_aug: batch_x_future_exog is added to the data loader tuple
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_x_future_exog) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                batch_x_future_exog = batch_x_future_exog.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        # Conditional model call based on model name
                        if 'TimeXer_' in self.args.model:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,
                                                 x_future_exog=batch_x_future_exog)
                        else:  # Default behavior for other models
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    # Conditional model call based on model name
                    if 'TimeXer_' in self.args.model:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,
                                             x_future_exog=batch_x_future_exog)
                    else:  # Default behavior for other models
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                if self.args.dynamic_loss:
                    loss = self._calculate_dynamic_loss(outputs, batch_y, criterion, epoch)
                else:
                    loss = self._get_loss(outputs, batch_y, criterion)

                total_loss.append(loss.item())

                # total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)

        early_stopping = EarlyStopping(
            patience=self.args.patience,
            verbose=True,
            save_checkpoints=not self.args.no_save_checkpoints
        )

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            # batch_x_future_exog is added to the data loader tuple
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_x_future_exog) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                batch_x_future_exog = batch_x_future_exog.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'TimeXer_' in self.args.model:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,
                                                 x_future_exog=batch_x_future_exog)
                        else:  # Default behavior for other models
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                        if self.args.dynamic_loss:
                            loss = self._calculate_dynamic_loss(outputs, batch_y, criterion, epoch)
                        else:
                            loss = self._get_loss(outputs, batch_y, criterion)

                        train_loss.append(loss.item())
                else:
                    if 'TimeXer_' in self.args.model:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,
                                             x_future_exog=batch_x_future_exog)
                    else:  # Default behavior for other models
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                    if self.args.dynamic_loss:
                        loss = self._calculate_dynamic_loss(outputs, batch_y, criterion, epoch)
                    else:
                        loss = self._get_loss(outputs, batch_y, criterion)

                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion, epoch)
            test_loss = self.vali(test_data, test_loader, criterion, epoch)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        if not self.args.no_save_checkpoints:
            self.model.load_state_dict(torch.load(best_model_path))
        else:
            print("Checkpoint saving is disabled. Skipping loading the best model. Using model state from last epoch.")

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            if not self.args.no_save_checkpoints:  # 确保在测试模式下也检查该参数
                self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))
            else:
                print("Checkpoint saving was disabled during training. Cannot load best model for testing.")

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            # batch_x_future_exog is added to the data loader tuple
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_x_future_exog) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                batch_x_future_exog = batch_x_future_exog.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'TimesNet_output' in self.args.model:
                            # TimesNet_output: Pass return_frequencies=True to get the selected periods and indices
                            outputs, periods, freq_indices = self.model(
                                batch_x, batch_x_mark, dec_inp, batch_y_mark, return_frequencies=True
                            )
                        elif 'TimeXer_' in self.args.model:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,
                                                 x_future_exog=batch_x_future_exog)
                        else:  # Default behavior for other models
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'TimesNet_output' in self.args.model:
                        # TimesNet_output: Pass return_frequencies=True to get the selected periods and indices
                        outputs, periods, freq_indices = self.model(
                            batch_x, batch_x_mark, dec_inp, batch_y_mark, return_frequencies=True
                        )
                    elif 'TimeXer_' in self.args.model:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,
                                             x_future_exog=batch_x_future_exog)
                    else:  # Default behavior for other models
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                if 'TimesNet_output' in self.args.model and i == 0:
                    print("--- Top-k Frequencies Analysis (First Batch) ---")
                    # `periods` and `freq_indices` are lists, each element corresponds to a TimesBlock layer
                    for layer_idx, (p, f) in enumerate(zip(periods, freq_indices)):
                        print(f"TimesBlock Layer {layer_idx}:")
                        print(
                            f"  - Selected Periods: {p}")  # p[0] is a numpy array of shape (k,) for the first sample in the batch
                        print(f"  - Corresponding FFT Freq Indices: {f}")  # f[0] is the same for frequency indices

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if i == 0:
                    print("test_data.scale: ", test_data.scale)
                    print("self.args.inverse: ", self.args.inverse)
                if test_data.scale and self.args.inverse:
                    shape = batch_y.shape
                    if outputs.shape[-1] != batch_y.shape[-1]:
                        outputs = np.tile(outputs, [1, 1, int(batch_y.shape[-1] / outputs.shape[-1])])
                    outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)

                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)

        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # dtw calculation
        if self.args.use_dtw:
            dtw_list = []
            manhattan_distance = lambda x, y: np.abs(x - y)
            for i in range(preds.shape[0]):
                x = preds[i].reshape(-1, 1)
                y = trues[i].reshape(-1, 1)
                if i % 100 == 0:
                    print("calculating dtw iter:", i)
                d, _, _, _ = accelerated_dtw(x, y, dist=manhattan_distance)
                dtw_list.append(d)
            dtw = np.array(dtw_list).mean()
        else:
            dtw = 'Not calculated'

        if self.args.pred_len == 336 and self.args.loss_weights and len(self.args.loss_weights) == 2:
            split_pos = self.args.pred_len // 2
            alpha = self.args.loss_weights[0]
            beta = self.args.loss_weights[1]

            mae_1, mse_1, rmse_1, mape_1, mspe_1 = metric(preds[:, :split_pos, :], trues[:, :split_pos, :])
            mae_2, mse_2, rmse_2, mape_2, mspe_2 = metric(preds[:, split_pos:, :], trues[:, split_pos:, :])

            mae = alpha * mae_1 + beta * mae_2
            mse = alpha * mse_1 + beta * mse_2
            rmse = np.sqrt(mse)

            mape = alpha * mape_1 + beta * mape_2
            mspe = alpha * mspe_1 + beta * mspe_2

            print('Weighted Indicators --> mse:{:.4f}, mae:{:.4f}, rmse:{:.4f}'.format(mse, mae, rmse))
        else:
            mae, mse, rmse, mape, mspe = metric(preds, trues)
            print('mae:{}, rmse:{}'.format(mae, rmse))

        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mae:{}, rmse:{}'.format(mae, rmse))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return