import multiprocessing
from multiprocessing import Process
import os
import time
from datetime import datetime
from multiprocessing.pool import Pool
from transformers import AutoTokenizer
from transformers import AutoModelForTokenClassification
# from optimum.onnxruntime import ORTModelForTokenClassification
from transformers import TokenClassificationPipeline

a = 100

def test1():
    time.sleep(3)
    print(f'inside test1 {datetime.now()} my id = {os.getpid()} my parent id = {os.getppid()}')
    try:
        print(f'I can see {a}')
    except:
        print('i cannot see a')
    return (os.getppid(), os.getpid())

def test2(arg):
    '''
      pool seems require an arg
    '''
    time.sleep(3)
    print(f'inside test1 {datetime.now()} my id = {os.getpid()} my parent id = {os.getppid()}')
    try:
        print(f'I can see {a} and with arg={arg}')
    except:
        print('i cannot see a')
    return (os.getpid(), arg)

if __name__ == '__main__':
    # multiprocessing.set_start_method('spawn')

    # p_list = []
    # for _ in range(3):
    #     p_list.append(Process(target=test1))
    
    # # this will run them sequentially
    # for p in p_list:
    #     p.start()
    #     p.join()
    #     p.terminate()
    #     p.close()
    
    # # process actually can't be reused even if i don't close them
    # # so need to do these pieces again
    # p_list = []
    # for _ in range(3):
    #     p_list.append(Process(target=test1))

    # # this will run them in parallel
    # for p in p_list:
    #     p.start()
    # for p in p_list:
    #     p.join()
    # for p in p_list:
    #     p.terminate()
    #     p.close()

    wfumodel = './checkpoint-8600'  # full model
    # wfumodel = './checkpoint-9360'  # bio distill model
    tokenizer = AutoTokenizer.from_pretrained(wfumodel)
    tokenizer.model_max_length = 128
    model = AutoModelForTokenClassification.from_pretrained(wfumodel)
    model.eval()
    
    # model = ORTModelForTokenClassification.from_pretrained(wfumodel, export=True, use_io_binding=False)
    clf = TokenClassificationPipeline(model=model, tokenizer=tokenizer, device=0)

    pool = Pool(processes=3)

    # be sure to examine the output pid number, which tells how it is scheduled!
    results = pool.map(test2, range(5))
    print(results)

    results = pool.imap(test2, range(5))
    print(list(results))

    # this give chunk to each processer, instead of sequentially assigning jobs
    # this should be faster than chunksize=1
    results = pool.imap(test2, range(10), chunksize=4) 
    print(list(results))

    pool.close()