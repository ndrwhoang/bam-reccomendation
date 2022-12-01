import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel, PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import logging as t_logging

t_logging.set_verbosity_error()


class EncoderModule(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.encoder = AutoModel.from_pretrained(self.config.pretrained_name)

    def _init_weights(self, module):
        # Code from hf itself so this plays nice with the ecosystem/trainer
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    # def _set_gradient_checkpointing(self, module, value=False):
    #     if isinstance(module, AutoModel):
    #         module.gradient_checkpointing = value
    
    def loss_fn(self, scores, labels):
        return F.cross_entropy(scores, labels)

    def score_candidates(self, query_embeddings, candidate_embeddings):
        return torch.matmul(
            query_embeddings, torch.transpose(candidate_embeddings, 0, 1)
        )

    def forward(
        self,
        query_ids,
        query_attention_mask,
        candidate_ids,
        candidate_attention_mask,
        labels,
    ):
        query_out = self.encoder(
            input_ids=query_ids, attention_mask=query_attention_mask
        )
        query_embedding = torch.mean(query_out.last_hidden_state, dim=1)

        candidate_out = self.encoder(
            input_ids=candidate_ids, attention_mask=candidate_attention_mask
        )
        candidate_embedding = torch.mean(candidate_out.last_hidden_state, dim=1)

        scores = self.score_candidates(query_embedding, candidate_embedding)
        loss = self.loss_fn(scores, labels) if torch.is_tensor(labels) else None

        return SequenceClassifierOutput(logits=scores, loss=loss)
