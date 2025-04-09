import copy
import torch
import torch.nn as nn
from models.components._abstract_model import SequentialRecModel
from models.components._modules import LayerNorm, BSARecBlock

class BSARecEncoder(nn.Module):
    def __init__(self, args):
        super(BSARecEncoder, self).__init__()
        self.args = args
        block = BSARecBlock(args)
        self.blocks = nn.ModuleList([copy.deepcopy(block) for _ in range(args.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=False):
        all_encoder_layers = [ hidden_states ]
        for layer_module in self.blocks:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states) # hidden_states => torch.Size([256, 50, 64])
        return all_encoder_layers

class BSARecModel(SequentialRecModel):
    def __init__(self,
                 hidden_dropout_prob: float,
                 item_size: int,
                 hidden_size: int,
                 max_seq_length: int,
                 initializer_range: float,
                 transformer,                 
                 ) -> None:
        super(BSARecModel, self).__init__(item_size, hidden_size, max_seq_length, initializer_range)

        self.LayerNorm = LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.item_encoder = BSARecEncoder(transformer) # the variable's name is transformer for time being.
        self.apply(self.init_weights)

    def forward(self, input_ids, user_ids=None, all_sequence_output=False):
        extended_attention_mask = self.get_attention_mask(input_ids)
        sequence_emb = self.add_position_embedding(input_ids)
        item_encoded_layers = self.item_encoder(sequence_emb,
                                                extended_attention_mask,
                                                output_all_encoded_layers=True,
                                                )               
        if all_sequence_output:
            sequence_output = item_encoded_layers
        else:
            sequence_output = item_encoded_layers[-1]

        return sequence_output

    def calculate_loss(self, seq_out, answers, neg_answers, same_target, user_ids):
        # seq_out is the forward output of the model
        # no need for negative answers here, interesting.
        seq_out = seq_out[:, -1, :]
        item_emb = self.item_embeddings.weight
        logits = torch.matmul(seq_out, item_emb.transpose(0, 1))
        loss = nn.CrossEntropyLoss()(logits, answers)

        return loss

