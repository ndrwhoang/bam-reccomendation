import torch
from transformers import Pipeline


class RecommendationPipeline(Pipeline):
    def _sanitize_parameters(self, **kwargs):
        preprocesses_kwargs = {}
        if "max_length" in kwargs:
            preprocesses_kwargs["max_length"] = kwargs["max_length"]

        postprocesses_kwargs = {}
        if "embedding_index" in kwargs:
            postprocesses_kwargs["embedding_index"] = kwargs["embedding_index"]
        if "top_k" in kwargs:
            postprocesses_kwargs["top_k"] = kwargs["top_k"]

        return preprocesses_kwargs, {}, postprocesses_kwargs

    def preprocess(self, text, max_length=512):
        candidate_ids = torch.empty(0)
        candidate_attention_mask = torch.empty(0)

        input_encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
            max_length=max_length,
        )

        return {
            "query_ids": input_encoding["input_ids"],
            "query_attention_mask": input_encoding["attention_mask"],
            "candidate_ids": candidate_ids,
            "candidate_attention_mask": candidate_attention_mask,
            "labels": None,
        }

    def _forward(self, model_inputs):
        return self.model(**model_inputs)

    def postprocess(self, model_outputs, embedding_index, top_k=4):
        hidden_states = model_outputs.hidden_states.detach().numpy()
        _, I = embedding_index.search(hidden_states, top_k)

        return {"top_k_indices": I}
