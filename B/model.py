from transformers import BertModel
from torch.nn import Linear, Dropout, Module

class ABSABert(Module):
    def __init__(self, args):
        super().__init__()

        pretrained_model = args.pretrained_model
        dropout_rate = args.dropout_rate

        self.bert = BertModel.from_pretrained(pretrained_model)
        self.dropout = Dropout(dropout_rate)
        self.linear = Linear(self.bert.config.hidden_size, 5)

    def forward(self, ids, mask, token_type_ids):

        bert_output = self.bert(input_ids=ids, attention_mask=mask, token_type_ids=token_type_ids)
        x = self.dropout(bert_output.pooler_output)
        x = self.linear(x)
        
        return x