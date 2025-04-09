import torch
import torch.nn as nn
import copy
from models.components._abstract_model import SequentialRecModel
from models.components._modules import TransformerEncoder, LayerNorm

## needs to be nn.Module and feed it into seqrec_module
## the calculate loss needs to be separate from this class

"""
[Paper]
Author: Wang-Cheng Kang et al. 
Title: "Self-Attentive Sequential Recommendation."
Conference: ICDM 2018

[Code Reference]
https://github.com/kang205/SASRec
Rehash from BSARec github repo
"""

class SASRecModel(SequentialRecModel):
    def __init__(self,
                 hidden_dropout_prob: float,
                 item_size: int,
                 hidden_size: int,
                 max_seq_length: int,
                 initializer_range: float,
                 transformer,
                 ) -> None:
        # need to do this for initialization of the sequentialRecModel (abstract class)
        # this is because SASRecModel is a subclass of SequentialRecModel
        # but we can also use nn.Module and then call the SequentialRecModel separately as object
        super(SASRecModel,self).__init__(item_size, hidden_size, max_seq_length, initializer_range)

        self.LayerNorm = LayerNorm(self.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(hidden_dropout_prob) # this dropout will be used by the seqrec_module

        self.item_encoder = TransformerEncoder(transformer) # should be similar to args
        self.apply(self.init_weights)

    def forward(self, input_ids:torch.Tensor, user_ids=None, all_sequence_output=False) -> torch.Tensor:
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
        # seq_out: [batch seq_len hidden_size] is the predicted sequence output
        seq_out = seq_out[:, -1, :] # this means we have [batch hidden_size]
        pos_ids, neg_ids = answers, neg_answers

        # [batch seq_len hidden_size]
        pos_emb = self.item_embeddings(pos_ids) # should be only 1 item hence [batch hidden_size]
        neg_emb = self.item_embeddings(neg_ids)

        # [batch hidden_size]
        seq_emb = seq_out # [batch hidden_size]
        pos_logits = torch.sum(pos_emb * seq_emb, -1) # [batch*seq_len] the dot product
        neg_logits = torch.sum(neg_emb * seq_emb, -1)

        pos_labels, neg_labels = torch.ones(pos_logits.shape, device=seq_out.device), torch.zeros(neg_logits.shape, device=seq_out.device)
        indices = (pos_ids != 0).nonzero().reshape(-1)
        bce_criterion = torch.nn.BCEWithLogitsLoss()
        loss = bce_criterion(pos_logits[indices], pos_labels[indices])
        loss += bce_criterion(neg_logits[indices], neg_labels[indices])

        return loss


def main():
    import sys
    import os
    sys.path.append(os.path.abspath('/Users/refaldi/Documents/Work/llm_id_recsys/src/'))
    class transformer:
        def __init__(self):
            self.hidden_size= 64 # need to be called again
            self.hidden_dropout_prob = 0.1 # this too
            self.num_attention_heads = 4
            self.num_hidden_layers= 2
            self.attention_probs_dropout_prob= 0.5
            self.hidden_act = 'gelu'
    args = transformer()
    print(args.hidden_size)
    sasrec = SASRecModel(0.1, 100, 64, 50, 0.02, args)
    x = torch.randint(0, 100, (256, 25)) # ranging from 0 to 100, 256 samples, 50 sequence length
    # print('Before,', sasrec.embed().shape)
    out = sasrec(x)
    # print(sasrec(x).shape)
    print('After,', sasrec.embed.shape)

if __name__ == "__main__":
    main()