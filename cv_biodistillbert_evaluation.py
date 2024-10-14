import datasets
import numpy as np
from transformers import AutoTokenizer
from utils_fold import PreProcess, compute_metrics
from transformers import DataCollatorForTokenClassification
from sklearn.metrics import classification_report, f1_score
from transformers import AutoModelForTokenClassification, TrainingArguments
from mytrainer import MyTrainer 
from sklearn.utils.class_weight import compute_class_weight
import os
from wfudata_fold import id2label, label2id
import sys

device = 'cuda:0'

def reset_model(num_labels, model_path):
    model = AutoModelForTokenClassification.from_pretrained(
                  model_path,
                  num_labels=num_labels,
                  id2label=id2label,
                  label2id=label2id).to(device)
    # need to force contiguous otherwise won't be able to save the model, strange behavior
    for _, param in model.named_parameters():
        param.data = param.data.contiguous()
    return model 


if (__name__ == '__main__'):

    fold = sys.argv[1]
    checkpoint_id = sys.argv[2]

    
    tokenizer = AutoTokenizer.from_pretrained('nlpie/bio-distilbert-uncased')

    # force the tokenizer model length to be biomert, this is from model.config['max_position_embeddings']
    # the default tokenizer.model_max_length is way too large
    # 128 reduce the memory issue
    tokenizer.model_max_length = 128   
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    model_dir = f'with_balanced_weighting_biodistilbert_fold_{fold}'
    model_path = os.path.join(model_dir, f'checkpoint-{checkpoint_id}')
    log_file = os.path.join(model_dir, 'eval.txt')
    if os.path.exists(log_file):
        os.remove(log_file)

    for text_class in ['all_types', 'H_and_P', 'progress', 'radiology', 'pathology']:

        # select out the note in that class
        # for now i just reload it, probably better way to do this
        wfu_dataset = datasets.load_dataset(f'wfudata_fold_{fold}', trust_remote_code=True)
        preprocess = PreProcess(tokenizer, wfu_dataset['train'].features['label'].str2int, stride=16)
        class_labels = wfu_dataset['train'].features['label'].names
        num_labels = len(class_labels)
        if text_class != 'all_types':
            wfu_dataset['test'] = wfu_dataset['test'].filter(lambda x:x['text_class']==text_class)
        wfu_dataset_tokenized = wfu_dataset.map(preprocess, batched=True, batch_size=16,
                                    remove_columns=wfu_dataset['train'].column_names)

        model = reset_model(num_labels, model_path)

        with open(log_file, 'a') as fid:
            fid.write(f'doing ==== {text_class} ======')
            fid.write('\n')

        # set up the trainer and run 1 evaluate step
        training_args = TrainingArguments(
          output_dir='tmp.dir',
          learning_rate=2e-5,
          per_device_train_batch_size=16,
          per_device_eval_batch_size=16,
          num_train_epochs=10,
          weight_decay=0.01,
          eval_strategy="epoch",
          save_strategy="epoch",
          push_to_hub=False,
          log_level='info',
          logging_steps=50,
          save_only_model=True,
          save_total_limit=5,
          load_best_model_at_end=True,
          metric_for_best_model='f1_weighted',
          greater_is_better=True
        )

        trainer = MyTrainer(
          class_weights = np.ones(num_labels, dtype=np.float32),
          log_file = log_file, 
          model=model,
          args=training_args,
          train_dataset=wfu_dataset_tokenized["train"],
          eval_dataset=wfu_dataset_tokenized["test"],
          tokenizer=tokenizer,
          data_collator=data_collator,
          compute_metrics=compute_metrics
        )

        trainer.evaluate()
