from transformers import AutoTokenizer
# from transformers import AutoModelForTokenClassification
from optimum.onnxruntime import ORTModelForTokenClassification
from transformers import TokenClassificationPipeline
from onnxconverter_common import float16

from datetime import datetime

model_id = './checkpoint-8600'
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.model_max_length = 128
# model = AutoModelForTokenClassification.from_pretrained(wfumodel)
model = ORTModelForTokenClassification.from_pretrained(model_id, export=True, use_io_binding=True)
clf = TokenClassificationPipeline(model=model, tokenizer=tokenizer, device=0)

model_fp16 = float16.convert_float_to_float16(model)
clf_fp16 = TokenClassificationPipeline(model=model, tokenizer=tokenizer, device=0)

with open('actual_note_test.txt', 'r') as fid:
    line = fid.read()
all_texts = line.split('---')

results = clf(all_texts[:5])
print(results)

results = clf_fp16(all_texts[:5])
print(results)

exit()

print(all_texts[:5])

print(datetime.now())
results = clf(all_texts, stride=16, ignore_labels=['NORMAL'], aggregation_strategy='max', batch_size=16)
print(datetime.now())

print(results[:5])