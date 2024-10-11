from transformers import Trainer
import torch
from torch import nn

class MyTrainer(Trainer):
    '''
      overwrite the trainer to have 
      1: class_weight in the compute_loss function
      2: logging to a file
    '''
    def __init__(self, class_weights, log_file, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = torch.tensor(class_weights).to(self.model.device)
        self.log_file = log_file
    
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

    def log(self, logs):
        ''' over write the log to output things to file '''

        print('inside log:', logs, self.state.global_step)

        with open(self.log_file, 'a') as fid:
            if 'eval_loss' in logs:
                fid.write(f'evaliation: step={self.state.global_step}, f1_macro={logs["eval_f1_macro"]}, f1_weighted={logs["eval_f1_weighted"]}')
                fid.write('\n')
                fid.write(logs['eval_report'])
                fid.write('\n')
            elif 'loss' in logs:
                fid.write(f'training: step={self.state.global_step}, loss={logs["loss"]}')
                fid.write('\n')
            else:
                pass
        
        if self.state.epoch is not None:
            logs["epoch"] = self.state.epoch
        if self.args.include_num_input_tokens_seen:
            logs["num_input_tokens_seen"] = self.state.num_input_tokens_seen

        output = {**logs, **{"step": self.state.global_step}}
        self.state.log_history.append(output)
        self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)
