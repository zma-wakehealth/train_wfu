from transformers import AutoTokenizer
from transformers import AutoModelForTokenClassification
from optimum.onnxruntime import ORTModelForTokenClassification
#from transformers import TokenClassificationPipeline
from mypipeline import MyTokenClassificationPipeline
from optimum.onnxruntime import ORTOptimizer
from optimum.onnxruntime.configuration import OptimizationConfig
import onnxruntime
from datetime import datetime
from transformers.utils import logging
import cProfile


def load_onnx_model(onnx_model_path, onnx_model_name, num_threads=16, device_id=0):
    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = num_threads   # needs this on slurm
    model = ORTModelForTokenClassification.from_pretrained(
        onnx_model_path,
        file_name = onnx_model_name,
        session_options = options,
        provider = 'CUDAExecutionProvider',
        provider_options = {'device_id': device_id}   # needs this on the server for multiple gpu
    )
    return model

def test(model, tokenizer, device_id, all_texts, batch_size):
    clf = MyTokenClassificationPipeline(model=model, tokenizer=tokenizer, device=device_id)
    print(datetime.now())
    results = clf(all_texts, stride=16, ignore_labels=['NORMAL'], aggregation_strategy='max', batch_size=batch_size)
    print(datetime.now())
    print(results[4])
    return results

if (__name__ == '__main__'):
    original_model_path = './checkpoint-8600'
    distill_model_path = './checkpoint-9360'
    original_onnx_path = original_model_path + '-onnx'
    distill_onnx_path = distill_model_path + '-onnx'
    device_id = 0

    logging.set_verbosity_info()
    logger = logging.get_logger("transformers")
    logger.info("INFO")
    logger.warning("WARN")

    with open('actual_note_test.txt', 'r') as fid:
        line = fid.read()
    all_texts = line.split('---')
    all_texts = all_texts[:1000]

    tokenizer = AutoTokenizer.from_pretrained(distill_onnx_path)
    tokenizer.model_max_length = 128
    model = load_onnx_model(distill_onnx_path, 'model_optimized.onnx')

    for batch_size in [1024]:
        print(f'batch_size={batch_size}')
        #cProfile.run('test(model, tokenizer, 0, all_texts, batch_size)')
        results = test(model, tokenizer, 0, all_texts, batch_size)
        print(results[5])
