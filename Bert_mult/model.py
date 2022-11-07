from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch import nn

class ModelWithProperties(nn.Module):
    def __init__(self, model_name):
        super(ModelWithProperties, self).__init__()
        self.bert = AutoModelForSequenceClassification.from_pretrained(model_name)
        #self.linear0 = nn.Linear(768*4, 768)
        self.linear1 = nn.Linear(4*768, 15)
        self.linear2 = nn.Linear(4*768, 30)
        self.output = nn.Linear(45, 23)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, input_ids, attention_mask):
        sequence_output = self.bert(input_ids, attention_mask=attention_mask, output_hidden_states=True, return_dict=False)
        hidden_states = sequence_output[1]
        pooled_output = torch.cat(tuple(hidden_states[i] for i in [-4, -3, -2, -1]), dim=-1)
        pooled_output = pooled_output[:,0,:]
        pooled_output = self.dropout(pooled_output)
        linear1 = self.linear1(pooled_output)
        linear2 = self.linear2(pooled_output)

        #output1 = nn.functional.relu(linear1)
        #utput2 = nn.functional.relu(linear2)
        joined = torch.cat((linear1, linear2), dim=1)
        final_output = self.output(joined)
        return linear1, final_output

def joined_loss(output1, output2, target1, target2, w1, w2):
    l1 = nn.BCELoss(pos_weights=w1)(output1, target1)
    l2 = nn.CrossEntropyLoss(weight = w2)(output2, target2)
    return l1 + l2
