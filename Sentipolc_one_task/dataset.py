import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer


class SentipolcDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer_name, task):
        df['text'] = df['text'].str.lower()
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.encodings = tokenizer(
            df['text'].tolist(),
            max_length=64,
            add_special_tokens=True,
            return_attention_mask=True,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

        self.labels = np.array(df[task]).reshape(-1, 1)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
    
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)

def train_val_split(df, tok_name,  val_perc=0.2):
    """
    It takes a dataframe, a tokenizer name, a validation percentage and a subsample flag. It then splits
    the dataframe into a training and validation set, and returns a HyperionDataset object for each
    
    :param df: the dataframe containing the data
    :param val_perc: the percentage of the data that will be used for validation
    :return: A tuple of two datasets, one for training and one for validation.
    """
    # Validation set creation
    val = df.sample(frac=val_perc)
    train = pd.concat([df,val]).drop_duplicates(keep=False)

    return SentipolcDataset(train, tok_name), SentipolcDataset(val, tok_name)