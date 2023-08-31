import os.path

import pandas as pd
import numpy as np

dataset = 'baby_2'

train_df = pd.read_csv(f'./data/{dataset}/train.tsv', sep='\t')
val_df = pd.read_csv(f'./data/{dataset}/val.tsv', sep='\t')
test_df = pd.read_csv(f'./data/{dataset}/test.tsv', sep='\t')

train_df.columns = [0, 1]
val_df.columns = [0, 1]
test_df.columns = [0, 1]
train_df[2] = pd.Series([1.0] * len(train_df))
test_df[2] = pd.Series([1.0] * len(test_df))
val_df[2] = pd.Series([1.0] * len(val_df))

train_df.to_csv(f'./data/{dataset}/train.tsv', sep='\t', header=None, index=None)
test_df.to_csv(f'./data/{dataset}/test.tsv', sep='\t', header=None, index=None)
val_df.to_csv(f'./data/{dataset}/val.tsv', sep='\t', header=None, index=None)

train_items = train_df[1].unique().tolist()

image_feat = np.load(f'./data/{dataset}/image_feat.npy')
text_feat = np.load(f'./data/{dataset}/text_feat.npy')

if not os.path.exists(f'./data/{dataset}/image_feat/'):
    os.makedirs(f'./data/{dataset}/image_feat/')
if not os.path.exists(f'./data/{dataset}/text_feat/'):
    os.makedirs(f'./data/{dataset}/text_feat/')

for v in range(image_feat.shape[0]):
    np.save(f'./data/{dataset}/image_feat/{v}.npy', image_feat[v])

for t in range(text_feat.shape[0]):
    np.save(f'./data/{dataset}/text_feat/{t}.npy', text_feat[t])
