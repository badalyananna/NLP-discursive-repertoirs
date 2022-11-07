from transformers import AutoModelForSequenceClassification
from torch import nn

class ModelWithRepertoirs(nn.Module):
    def __init__(self, model_name, num_labels=6):
        super(ModelWithRepertoirs, self).__init__()
        self.bert = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.linear = nn.Linear(768, 64)
        self.output = nn.Linear(64, num_labels)
        self.dropout = nn.Dropout(0.2)

    def forward(self, input_ids, attention_mask):
        sequence_output = self.bert(input_ids, attention_mask=attention_mask, output_hidden_states=True, return_dict=False)
        hidden_states = sequence_output[1]
        pooled_output = hidden_states[-1]
        pooled_output = pooled_output[:,0,:]
        pooled_output = self.dropout(pooled_output)
        
        linear = self.linear(pooled_output)
        linear = nn.functional.relu(linear)
        output = self.output(linear)
        return output
