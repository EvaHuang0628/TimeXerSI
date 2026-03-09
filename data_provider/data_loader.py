import os
import numpy as np
import pandas as pd
import glob
import re
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
from data_provider.m4 import M4Dataset, M4Meta
from data_provider.uea import subsample, interpolate_missing, Normalizer
from sktime.datasets import load_from_tsfile_to_dataframe
import warnings
from utils.augmentation import run_augmentation_single

warnings.filterwarnings('ignore')

class Dataset_Custom(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        self.args = args
        self.train_ratio = getattr(self.args, 'train_ratio', 0.7)
        self.test_ratio = getattr(self.args, 'test_ratio', 0.2)
        print("train_ratio: ", self.train_ratio)
        print("test_ratio: ", self.test_ratio)

        self.flag = flag

        if size is None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        self.prediction_offset = getattr(self.args, 'prediction_offset', 0)
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features

        processed_targets = []
        if isinstance(target, list):
            for t in target:
                if isinstance(t, str) and ',' in t:
                    processed_targets.extend([st.strip() for st in t.split(',')])
                else:
                    processed_targets.append(t)
        elif isinstance(target, str):
            if ',' in target:
                processed_targets.extend([st.strip() for st in target.split(',')])
            else:
                processed_targets.append(target)

        self.target = processed_targets

        if not self.target:
            raise ValueError("No target columns were processed. Please check the --target argument.")

        self.scale = scale
        print("self.scale: ", self.scale)
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

        print("=" * 50)
        print(f"Dataset: {self.root_path}/{self.data_path}")
        exog_cols = self.cols[:-len(self.target)]
        for i, col_name in enumerate(exog_cols):
            print(f"  index {i}: {col_name}")
        print("=" * 50)

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        cols_to_keep = list(df_raw.columns)
        cols_to_keep.remove('date')
        for t in self.target:
            if t in cols_to_keep:
                cols_to_keep.remove(t)
            else:
                raise ValueError(
                    f"FATAL: The specified target column '{t}' was not found in the data columns: {df_raw.columns.tolist()}. "
                    f"Please check the --target argument and your CSV file header."
                )

        final_order = ['date'] + cols_to_keep + self.target
        df_raw = df_raw[final_order]
        self.cols = cols_to_keep + self.target

        num_train = int(len(df_raw) * self.train_ratio)
        num_test = int(len(df_raw) * self.test_ratio)
        num_vali = len(df_raw) - num_train - num_test

        if num_vali < 0:
            raise ValueError("The sum of the training and test sets exceeds 1.0, resulting in a negative validation set size.")

        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.flag == 'train':
            output_path = os.path.join(self.root_path, 'split_data')
            print("output_path:", output_path)
            if not os.path.exists(output_path):
                os.makedirs(output_path)

            train_df = df_raw.iloc[border1s[0]:border2s[0]]
            train_df.to_csv(os.path.join(output_path, 'train.csv'), index=False)

            val_df = df_raw.iloc[border1s[1]:border2s[1]]
            val_df.to_csv(os.path.join(output_path, 'val.csv'), index=False)

            test_df = df_raw.iloc[border1s[2]:border2s[2]]
            test_df.to_csv(os.path.join(output_path, 'test.csv'), index=False)


        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[self.target]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        if self.set_type == 0 and self.args.augmentation_ratio > 0:
            self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)

        self.data_stamp = data_stamp

    def save_test_data(self, folder_path):
        if self.flag != 'test':
            return

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        test_data = self.data_y
        test_df = pd.DataFrame(self.data_stamp)

        if hasattr(self, 'date'):
            test_df['date'] = self.date

        test_csv_path = os.path.join(folder_path, 'test_data.csv')
        test_df.to_csv(test_csv_path, index=False)

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len + self.prediction_offset
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        x_future_exog = np.zeros((self.pred_len, 0))
        if 'TimeXer_' in self.args.model:
            num_exog_features = self.data_x.shape[1] - len(self.target) if self.features in ['S', 'MS'] else 0

            if num_exog_features > 0:
                future_part_of_y = self.data_y[r_end - self.pred_len:r_end]
                x_future_exog = future_part_of_y[:, :-len(self.target)]

        return seq_x, seq_y, seq_x_mark, seq_y_mark, x_future_exog

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len - self.prediction_offset + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
