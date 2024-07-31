import datasets
import numpy as np
from transformers import AutoTokenizer
from utils import PreProcess, compute_metrics
from transformers import DataCollatorForTokenClassification
from sklearn.metrics import classification_report, f1_score
from transformers import AutoModelForTokenClassification, TrainingArguments
from mytrainer import MyTrainer 
from sklearn.utils.class_weight import compute_class_weight
import os

def reset_model(num_labels):
    model = AutoModelForTokenClassification.from_pretrained(
                  "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
                  num_labels=num_labels).to(device)
    # need to force contiguous otherwise won't be able to save the model, strange behavior
    for _, param in model.named_parameters():
        param.data = param.data.contiguous()
    return model 


if (__name__ == '__main__'):
    wfu_dataset = datasets.load_dataset('wfudata', trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext")
    preprocess = PreProcess(tokenizer, wfu_dataset['train'].features['label'].str2int, max_length=128, stride=10)

    wfu_dataset_tokenized = wfu_dataset.map(preprocess, batched=True, batch_size=5,
                                        remove_columns=wfu_dataset['train'].column_names)
    
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    device = 'cuda:0'
    class_labels = wfu_dataset['train'].features['label'].names
    num_labels = len(class_labels)

    for iexp in (1, 0):
        if iexp == 0:
            output_dir = 'no_balanced'
            class_weights = np.ones(num_labels, dtype=np.float32)
        else:
            output_dir = 'with_balanced_weighting'
            tokens_stat = []
            for x in wfu_dataset_tokenized['train']['labels']:
                tokens_stat += [i for i in x if i != -100]
            class_weights = compute_class_weight('balanced',
                                  classes=np.array(range(num_labels)),
                                  y=tokens_stat)
            class_weights = class_weights.astype(np.float32)
            print(class_weights)
        
        # default is append
        log_file = f'{output_dir}.txt'
        if os.path.exists(log_file):
            os.remove(log_file)

        model = reset_model(num_labels)
        training_args = TrainingArguments(
          output_dir=output_dir,
          learning_rate=2e-5,
          per_device_train_batch_size=16,
          per_device_eval_batch_size=16,
          num_train_epochs=10,
          weight_decay=0.01,
          eval_strategy="steps",
          eval_steps=200,
          save_strategy="epoch",
          push_to_hub=False,
          log_level='info',
          logging_steps=50
        )
        trainer = MyTrainer(
          class_weights = class_weights,
          log_file = log_file, 
          model=model,
          args=training_args,
          train_dataset=wfu_dataset_tokenized["train"],
          eval_dataset=wfu_dataset_tokenized["test"],
          tokenizer=tokenizer,
          data_collator=data_collator,
          compute_metrics=compute_metrics
        )
        trainer.train()
