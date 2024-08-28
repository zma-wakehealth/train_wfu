from transformers import AutoTokenizer
from transformers import AutoModelForTokenClassification
from optimum.onnxruntime import ORTModelForTokenClassification
from transformers import TokenClassificationPipeline
from optimum.onnxruntime import ORTOptimizer
from optimum.onnxruntime.configuration import OptimizationConfig
import onnxruntime
from datetime import datetime

def optimize_model(input_model_path, optimized_model_path):
    '''
      given a input huggingface model, convert it first to onnx model, then optimize it
    '''
    model_onnx = ORTModelForTokenClassification.from_pretrained(input_model_path, export=True)
    model_onnx.save_pretrained(optimized_model_path)

    optimizer = ORTOptimizer.from_pretrained(optimized_model_path)
    optimization_config = OptimizationConfig(optimization_level=99,
                                             optimize_for_gpu=True,
                                             fp16=True
                                             )
    optimizer.optimize(optimization_config, save_dir=optimized_model_path, file_suffix='optimized')

def load_onnx_model(onnx_model_path, onnx_model_name, num_threads=4, device_id=0):
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
    clf = TokenClassificationPipeline(model=model, tokenizer=tokenizer, device=device_id)
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

    #optimize_model(original_model_path, original_onnx_path)
    #optimize_model(distill_model_path, distill_onnx_path)

    with open('actual_note_test.txt', 'r') as fid:
        line = fid.read()
    all_texts = line.split('---')

    # # test original model and the optimized version of it
    # tokenizer = AutoTokenizer.from_pretrained(original_model_path)
    # tokenizer.model_max_length = 128
    # model = AutoModelForTokenClassification.from_pretrained(original_model_path)
    # results = test(model, tokenizer, 0, all_texts, 16)

    # tokenizer = AutoTokenizer.from_pretrained(original_onnx_path)
    # tokenizer.model_max_length = 128
    # model = load_onnx_model(original_onnx_path, 'model.onnx')
    # results = test(model, tokenizer, 0, all_texts, 16)

    # model = load_onnx_model(original_onnx_path, 'model_optimized.onnx')
    # results = test(model, tokenizer, 0, all_texts, 16)

    # # distill
    # tokenizer = AutoTokenizer.from_pretrained(distill_model_path)
    # tokenizer.model_max_length = 128
    # model = AutoModelForTokenClassification.from_pretrained(distill_model_path)
    # results = test(model, tokenizer, 0, all_texts, 16)

    tokenizer = AutoTokenizer.from_pretrained(distill_onnx_path)
    tokenizer.model_max_length = 128
    # model = load_onnx_model(distill_onnx_path, 'model.onnx')
    # results = test(model, tokenizer, 0, all_texts, 16)

    model = load_onnx_model(distill_onnx_path, 'model_optimized.onnx')
    results = test(model, tokenizer, 0, all_texts, 16)
