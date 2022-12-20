import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel, PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import logging as t_logging

# t_logging.set_verbosity_error()


class Reranker(PreTrainedModel):
    def __init__(self, config, inference=False):
        super().__init__(config)
        self.config = config
        
        if not inference:
            self.pretrained_encoder = AutoModel.from_pretrained(self.config.pretrained_name)
        else:
            self.pretrained_encoder = AutoModel.from_config(self.config)

    def loss_fn(self, scores, labels):
        return F.cross_entropy(scores, labels)

    def score_candidates(self, query_embedding, candidate_embedding):
        return torch.matmul(
            query_embedding, torch.transpose(candidate_embedding, 0, 1)
        )

    def forward(
        self,
        query_ids,
        query_attention_mask,
        candidate_ids,
        candidate_attention_mask,
        labels,
    ):
        input_ids = torch.cat((query_ids, candidate_ids), dim=0)
        attention_mask = torch.cat(
            (query_attention_mask, candidate_attention_mask), dim=0
        )
        encoder_out = self.pretrained_encoder(
            input_ids=input_ids, attention_mask=attention_mask
        )

        query_embedding = encoder_out.last_hidden_state[: query_ids.size(0), :, :]
        query_embedding = torch.mean(query_embedding, dim=1)
        query_embedding = F.normalize(query_embedding, dim=1)

        candidate_embedding = encoder_out.last_hidden_state[query_ids.size(0) :, :, :]
        candidate_embedding = torch.mean(candidate_embedding, dim=1)
        candidate_embedding = F.normalize(candidate_embedding, dim=1)

        scores = self.score_candidates(query_embedding, candidate_embedding)
        loss = self.loss_fn(scores, labels) if torch.is_tensor(labels) else None

        return SequenceClassifierOutput(
            hidden_states=query_embedding, logits=scores, loss=loss
        )
