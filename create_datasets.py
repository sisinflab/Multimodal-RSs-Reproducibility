import os.path

import pandas as pd
import json
import numpy as np

dataset = 'tiktok'

train_path = f'./data/{dataset}/train.json'
val_path = f'./data/{dataset}/val.json'
test_path = f'./data/{dataset}/test.json'

train_json = json.load(open(train_path))
val_json = json.load(open(val_path))
test_json = json.load(open(test_path))

train_df = pd.DataFrame([], columns=[0, 1, 2])
val_df = pd.DataFrame([], columns=[0, 1, 2])
test_df = pd.DataFrame([], columns=[0, 1, 2])

# training set
users = []
items = []
ratings = []
for u, its in train_json.items():
    for i in its:
        users.append(u)
        items.append(i)
        ratings.append(1.0)

train_df[0] = pd.Series(users).astype(int)
train_df[1] = pd.Series(items).astype(int)
train_df[2] = pd.Series(ratings)
train_df = train_df.sort_values(by=0).reset_index(drop=True)

# validation set
users = []
items = []
ratings = []
for u, its in val_json.items():
    for i in its:
        users.append(u)
        items.append(i)
        ratings.append(1.0)

val_df[0] = pd.Series(users).astype(int)
val_df[1] = pd.Series(items).astype(int)
val_df[2] = pd.Series(ratings)
val_df = val_df.sort_values(by=0).reset_index(drop=True)

# test set
users = []
items = []
ratings = []
for u, its in test_json.items():
    for i in its:
        users.append(u)
        items.append(i)
        ratings.append(1.0)

test_df[0] = pd.Series(users).astype(int)
test_df[1] = pd.Series(items).astype(int)
test_df[2] = pd.Series(ratings)
test_df = test_df.sort_values(by=0).reset_index(drop=True)

train_df.to_csv(f'./data/{dataset}/train.tsv', sep='\t', header=None, index=None)
val_df.to_csv(f'./data/{dataset}/val.tsv', sep='\t', header=None, index=None)
test_df.to_csv(f'./data/{dataset}/test.tsv', sep='\t', header=None, index=None)

train_items = train_df[1].unique().tolist()

if dataset == 'tiktok':
    audio_feat = np.load(f'./data/{dataset}/audio_feat.npy')
    if not os.path.exists(f'./data/{dataset}/audio_feat/'):
        os.makedirs(f'./data/{dataset}/audio_feat/')

    for a in range(audio_feat.shape[0]):
        if a in train_items:
            np.save(f'./data/{dataset}/audio_feat/{a}.npy', audio_feat[a])
        else:
            print(a)

image_feat = np.load(f'./data/{dataset}/image_feat.npy')
text_feat = np.load(f'./data/{dataset}/text_feat.npy')

if not os.path.exists(f'./data/{dataset}/image_feat/'):
    os.makedirs(f'./data/{dataset}/image_feat/')
if not os.path.exists(f'./data/{dataset}/text_feat/'):
    os.makedirs(f'./data/{dataset}/text_feat/')

for v in range(image_feat.shape[0]):
    if v in train_items:
        np.save(f'./data/{dataset}/image_feat/{v}.npy', image_feat[v])
    else:
        print(v)

for t in range(text_feat.shape[0]):
    if t in train_items:
        np.save(f'./data/{dataset}/text_feat/{t}.npy', text_feat[t])
    else:
        print(t)

