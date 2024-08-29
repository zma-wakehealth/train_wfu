from transformers import TokenClassificationPipeline
import numpy as np
from typing import List, Optional, Tuple, Union
from transformers.pipelines.token_classification import AggregationStrategy 

class MyTokenClassificationPipeline_GPU(TokenClassificationPipeline):
    def postprocess(self, all_outputs, aggregation_strategy, ignore_labels=None):
        if ignore_labels is None:
            ignore_labels = ["O"]
        return all_outputs

class MyTokenClassificationPipeline_CPU(TokenClassificationPipeline):
    def __init__(self, model, tokenizer):
        self.tokenizer = tokenizer
        self.framework = 'pt'
        self.id2label = model.config.id2label
    
    def aggregate_word(self, entities: List[dict], aggregation_strategy: AggregationStrategy) -> dict:
        word = self.tokenizer.convert_tokens_to_string([entity["word"] for entity in entities])
        if aggregation_strategy == AggregationStrategy.FIRST:
            scores = entities[0]["scores"]
            idx = scores.argmax()
            score = scores[idx]
            entity = self.id2label[idx]
        elif aggregation_strategy == AggregationStrategy.MAX:
            max_entity = max(entities, key=lambda entity: entity["scores"].max())
            scores = max_entity["scores"]
            idx = scores.argmax()
            score = scores[idx]
            entity = self.id2label[idx]
        elif aggregation_strategy == AggregationStrategy.AVERAGE:
            scores = np.stack([entity["scores"] for entity in entities])
            average_scores = np.nanmean(scores, axis=0)
            entity_idx = average_scores.argmax()
            entity = self.id2label[entity_idx]
            score = average_scores[entity_idx]
        else:
            raise ValueError("Invalid aggregation_strategy")
        new_entity = {
            "entity": entity,
            "score": score,
            "word": word,
            "start": entities[0]["start"],
            "end": entities[-1]["end"],
        }
        return new_entity
