from re import S
import yaml
import os
import sys 
sys.path.append(os.path.dirname(sys.path[0]))
#sys.path.append(os.path.dirname(os.path.dirname(sys.path[0])))

import pandas as pd
import torch
import neptune.new as neptune

from model import ModelWithProperties
from transformers import AutoTokenizer
from dataset import HyperionDataset
from dataset import train_val_split
from trainer import BertClsTrainer
from utils.utils import seed_everything
from loggers.neptune_logger import NeptuneLogger
from utils.utils import plot_confusion_matrix, plot_f1, plot_loss

if len(sys.argv) != 2:
    print("ERROR:  config_file path not provided")
    sys.exit(1)

# Repository paths
# './' local
# './RepML/ cluster

try: 
    with open (sys.argv[1] + 'config.yml', 'r') as file:
        config = yaml.safe_load(file)        
except Exception as e:
    print('Error reading the config file')
    print(e)
    sys.exit(1)
print('config file loaded!')

seed_everything(config['seed'])

df = pd.read_csv(sys.argv[1] + 'hyperion_train.csv', na_filter=False)
test_df = pd.read_csv(sys.argv[1] + 'hyperion_test.csv', na_filter=False)

logger = NeptuneLogger()
logger.run['config'] = config


model_name = config['model']

train_dataset, val_dataset = train_val_split(df, model_name, val_perc=0.25, subsample=False)
test_dataset = HyperionDataset(test_df, model_name)

trainer = BertClsTrainer()

model = ModelWithProperties(model_name)
logger.run['model'] = model.state_dict()
# model.name = model_name
loss1 = torch.nn.BCEWithLogitsLoss(pos_weight = torch.Tensor(config['loss_weights1']))
loss2 = torch.nn.CrossEntropyLoss(weight = torch.Tensor(config['loss_weights2']))

history = trainer.fit(model,
            train_dataset, 
            val_dataset,
            config['batch_size'],
            float(config['learning_rate']),
            config['n_epochs'],
            loss1,
            loss2)

logger.run['history'] = history

"""
for param in model.bert.parameters():
    param.requires_grad = False

history = trainer.fit(model,
            train_dataset, 
            val_dataset,
            config['batch_size'],
            float(config['learning_rate']),
            config['n_epochs'],
            loss1,
            loss2)

logger.run['history_class'] = history
"""

out = trainer.test(model,test_dataset, config['batch_size'], loss1, loss2)
logger.run['test/metrics'] = out['metrics']
logger.run['test/loss'] = out['loss']

cm = plot_confusion_matrix(out['gt'], out['pred'], test_dataset.labels_list())
logger.run["confusion_matrix"].upload(neptune.types.File.as_image(cm))

fig = plot_loss(history['train_loss'], history['val_loss'])
logger.run["loss_plot"].upload(neptune.types.File.as_image(fig))

"""
hf_token = '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'
if config['save']:
    model.push_to_hub("RepML", use_temp_dir=True, use_auth_token=hf_token)
    AutoTokenizer.from_pretrained(model_name).push_to_hub("RepML", use_temp_dir=True, use_auth_token=hf_token)
"""