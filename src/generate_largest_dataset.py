import os
import pickle
from typing import Optional

import networkx as nx
import numpy as np
import pandas as pd
from huggingface_hub import HfApi, hf_hub_download, HfFileSystem
from dotenv import load_dotenv
from huggingface_hub.utils import enable_progress_bars

from tools import StandardScaler
from transforms import to_adj_matrix


def generate_data_and_idx(df, x_offsets, y_offsets, add_time_of_day: bool, add_day_of_week: bool):
    num_samples, num_nodes = df.shape
    data = np.expand_dims(df.values, axis=-1)

    feature_list = [data]
    if add_time_of_day:
        time_ind = (df.index.values - df.index.values.astype('datetime64[D]')) / np.timedelta64(1, 'D')
        time_of_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
        feature_list.append(time_of_day)
    if add_day_of_week:
        dow = df.index.dayofweek
        dow_tiled = np.tile(dow, [1, num_nodes, 1]).transpose((2, 1, 0))
        day_of_week = dow_tiled / 7
        feature_list.append(day_of_week)

    data = np.concatenate(feature_list, axis=-1)

    min_t = abs(min(x_offsets))
    max_t = abs(num_samples - abs(max(y_offsets)))  # Exclusive
    print('idx min & max:', min_t, max_t)
    idx = np.arange(min_t, max_t, 1)
    return data, idx

def new_and_dying_sensors(df: pd.DataFrame) -> tuple[list, list, list]:
    df = df.sort_index()

    isna   = df.isna()
    valid  = ~isna

    has_before = valid.cumsum(axis=0).gt(0)
    has_after  = valid[::-1].cumsum(axis=0)[::-1].gt(0)

    leading_nan = isna & ~has_before
    trailing_nan = isna & ~has_after
    internal_nan = isna & has_before & has_after

    newborn_cols = leading_nan.any(axis=0)
    dying_cols   = trailing_nan.any(axis=0)
    invalid_cols = internal_nan.any(axis=0) | isna.all(axis=0)

    newborn_sensors = df.columns[newborn_cols].tolist()
    dying_sensors   = df.columns[dying_cols].tolist()
    invalid_sensors = df.columns[invalid_cols].tolist()

    return newborn_sensors, dying_sensors, invalid_sensors

# Parameters
output_folder: str = 'output'
years: list = [2024]
sensors: Optional[list] = None
allow_non_complete_years: bool = False
sequence_length: int = 12
x_offsets = np.sort(np.arange(-(sequence_length - 1), 1, 1))
y_offsets = np.sort(np.arange(1, sequence_length + 1, 1))
splits = {
    'train': 0.6,
    'val': 0.2,
}

enable_progress_bars()
# load_dotenv()
HF_TOKEN = os.getenv("HF_API_TOKEN_GOMO") or False
repo_id = "dmariaa70/GO-MO"

# Download and transform the data for the given years
all_data = pd.DataFrame()

for year in years:
    file_name = f"traffic_data_{year}.csv"
    download_path = hf_hub_download(repo_id=repo_id,
                                    filename=file_name,
                                    repo_type="dataset",
                                    token=HF_TOKEN,
                                    force_download=True,
                                    dry_run=False)

    df = pd.read_csv(
        download_path,
        parse_dates=['entry_date']
    )

    if sensors is not None:
        df = df[df['sensor_id'].isin(sensors)]

    df['traffic_intensity'] = df['traffic_intensity'] / 4

    df_pivot = df.pivot(index='entry_date', columns='sensor_id', values='traffic_intensity')
    all_data = pd.concat([all_data, df_pivot], axis=1)

if not allow_non_complete_years:
    newborn, dying, temporary_off = new_and_dying_sensors(all_data)

    if len(temporary_off) > 0:
        print(f"Temporary off sensors found: {temporary_off}")

    to_drop = set(newborn) | set(dying) | set(temporary_off)
    all_data = all_data.drop(columns=to_drop)

data, idx = generate_data_and_idx(all_data, x_offsets, y_offsets, add_time_of_day=True, add_day_of_week=True)

# generate splits
num_samples = len(idx)
num_train = int(num_samples * splits['train'])
num_val = int(num_samples * splits['val'])
num_test = num_samples - num_train - num_val
idx_train = idx[:num_train]
idx_val = idx[num_train:num_train + num_val]
idx_test = idx[num_train + num_val:]

x_train = data[:idx_val[0] - sequence_length, :, 0]
scaler = StandardScaler(mean=x_train.mean(), std=x_train.std())
data[..., 0] = scaler.transform(data[..., 0])

os.makedirs(output_folder, exist_ok=True)
np.savez_compressed(os.path.join(output_folder, 'his.npz'), data=data, mean=scaler.mean, std=scaler.std)
np.save(os.path.join(output_folder, 'idx_train.npy'), idx_train)
np.save(os.path.join(output_folder, 'idx_val.npy'), idx_val)
np.save(os.path.join(output_folder, 'idx_test.npy'), idx_test)

# Download and transform the route based graph
hf_file = hf_hub_download(repo_id=repo_id,
                          filename="routes-graph.pkl",
                          repo_type="dataset",
                          token=HF_TOKEN,
                          dry_run=False)

with open(hf_file, 'rb') as f:
    G: nx.Graph = pickle.load(f)

    adj = to_adj_matrix(G, sensors=sensors)

    node_num = adj.shape[0]
    print('adj shape:', adj.shape)
    print('edge number:', np.count_nonzero(adj))
    print('node degree:', np.mean(np.count_nonzero(adj, axis=-1)))
    print('sparsity', np.count_nonzero(adj) / (node_num ** 2) * 100)

    # save adj matrix
    np.save(os.path.join(output_folder, 'metraq_rn_adj.npy'), adj)