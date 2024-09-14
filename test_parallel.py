from mypipeline import MyTokenClassificationPipeline_GPU, MyTokenClassificationPipeline_CPU
from optimum.onnxruntime import ORTModelForTokenClassification
from transformers import TokenClassificationPipeline
from transformers import AutoTokenizer
import onnxruntime
from transformers.pipelines.token_classification import AggregationStrategy
from multiprocessing.pool import Pool
from functools import partial
from datetime import datetime   

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

if (__name__ == '__main__'):

    device_id = 0

    with open('actual_note_test.txt', 'r') as fid:
        line = fid.read()
    all_texts = line.split('---')
    all_texts = all_texts[:2000]
    # all_texts = sorted(all_texts, key=lambda x:len(x))

    distill_model_path = './checkpoint-9360'
    distill_onnx_path = distill_model_path + '-onnx'

    tokenizer = AutoTokenizer.from_pretrained(distill_onnx_path)
    tokenizer.model_max_length = 128
    model = load_onnx_model(distill_onnx_path, 'model_optimized.onnx')
    clf = MyTokenClassificationPipeline_GPU(model=model, tokenizer=tokenizer, device=device_id)
    pool = Pool(processes=2)
    tmp = MyTokenClassificationPipeline_CPU(model=model, tokenizer=tokenizer)
    func = partial(tmp.postprocess, aggregation_strategy=AggregationStrategy.MAX, ignore_labels=['NORMAL'])

    rr = None
    rr_all = []

    print(f'before running: {datetime.now()}')

    for k in range(3):
        print(f'===== new round {k} =====')
        print('before gpu:', datetime.now())
        results = clf(all_texts, stride=16, ignore_labels=['NORMAL'], aggregation_strategy='max', batch_size=16)
        print('after gpu:', datetime.now())

        if rr is not None:
            print('before wait', datetime.now())
            rr.wait()
            print('after wait', datetime.now())
            rr_all.extend(list(rr.get()))

        print('before map async:', datetime.now())
        rr = pool.map_async(func, results, chunksize=50)
        # rr.wait()
        # rr_all.extend(list(rr.get()))
        print('after map async:', datetime.now())

    # last chunk of cpu process
    print('before last trunk:', datetime.now())
    rr.wait()
    rr_all.extend(list(rr.get()))
    print('after last trunk:', datetime.now())

    print(f'after running: {datetime.now()}')

    # print(rr[:10])
    pool.close()
    print(rr_all[7])
    print(rr_all[2007])
    print(rr_all[4007])


