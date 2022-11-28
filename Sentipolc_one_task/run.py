import yaml
import os
import sys 
sys.path.append(os.path.dirname(sys.path[0]))

import pandas as pd
import torch
import neptune.new as neptune

from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from dataset import SentipolcDataset
from trainer import BertClsTrainer
from utils import seed_everything
from neptune_logger import NeptuneLogger
from utils import plot_confusion_matrix, plot_f1, plot_loss

if len(sys.argv) != 2:
    print("ERROR:  config_file path not provided")
    sys.exit(1)

try: 
    with open (sys.argv[1] + 'config.yml', 'r') as file:
        config = yaml.safe_load(file)        
except Exception as e:
    print('Error reading the config file')
    print(e)
    sys.exit(1)
print('config file loaded!')

seed_everything(config['seed'])

df = pd.read_csv(sys.argv[1] + 'train.csv', na_filter=False)
val_df = pd.read_csv(sys.argv[1] + 'valid.csv', na_filter=False)
test_df = pd.read_csv(sys.argv[1] + 'test.csv', na_filter=False)

logger = NeptuneLogger()
logger.run['config'] = config


model_name = config['model']
task = config['task']

train_dataset = SentipolcDataset(df, model_name, task)
val_dataset = SentipolcDataset(val_df, model_name, task)
test_dataset = SentipolcDataset(test_df, model_name, task)

trainer = BertClsTrainer()

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
# model.name = model_name
loss = torch.nn.BCEWithLogitsLoss(pos_weight = torch.Tensor(config['loss_weights']))

history = trainer.fit(model,
            train_dataset, 
            val_dataset,
            config['batch_size'],
            float(config['learning_rate']),
            config['n_epochs'],
            loss)

logger.run['history'] = history

out = trainer.test(model,test_dataset, config['batch_size'], loss)
logger.run['test/metrics/positive'] = out['metrics/positive']
logger.run['test/metrics/negative'] = out['metrics/negative']
logger.run['test/loss'] = out['loss']

fig = plot_loss(history['train_loss'], history['val_loss'])
logger.run["loss_plot"].upload(neptune.types.File.as_image(fig))


hf_token = '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'
if config['save']:
    model.push_to_hub("sentipolc", use_temp_dir=True, use_auth_token=hf_token)
    AutoTokenizer.from_pretrained(model_name).push_to_hub("sentipolc", use_temp_dir=True, use_auth_token=hf_token)
