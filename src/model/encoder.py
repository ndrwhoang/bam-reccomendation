import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel, PreTrainedModel, AutoConfig
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import logging as t_logging

t_logging.set_verbosity_error()


class EncoderModule(PreTrainedModel):
    def __init__(self, autoconfig, config=None):
        super().__init__(autoconfig)
        self.encoder = AutoModel.from_config(autoconfig)

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
        candidate_embedding = torch.mean(candidate_embedding.last_hidden_state, dim=1)

        scores = self.score_candidates(query_embedding, candidate_embedding)
        loss = self.loss_fn(scores, labels) if torch.is_tensor(labels) else None

        return SequenceClassifierOutput(logits=scores, loss=loss)
