from transformers import TokenClassificationPipeline
import numpy as np

class MyTokenClassificationPipeline(TokenClassificationPipeline):
    def postprocess(self, all_outputs, aggregation_strategy, ignore_labels=None):
        if ignore_labels is None:
            ignore_labels = ["O"]
        all_entities = []
        for model_outputs in all_outputs:
            logits = model_outputs["logits"][0].numpy()
            sentence = all_outputs[0]["sentence"]
            input_ids = model_outputs["input_ids"][0]
            offset_mapping = (
                model_outputs["offset_mapping"][0] if model_outputs["offset_mapping"] is not None else None
            )
            special_tokens_mask = model_outputs["special_tokens_mask"][0].numpy()

            maxes = np.max(logits, axis=-1, keepdims=True)
            shifted_exp = np.exp(logits - maxes)
            scores = shifted_exp / shifted_exp.sum(axis=-1, keepdims=True)

            all_entities.append({
                'logits': logits,
                'sentence': sentence,
                'input_ids': input_ids,
                'offset_mapping': offset_mapping,
                'special_tokens_mask': special_tokens_mask,
                'scores': scores
            })

        return all_entities
