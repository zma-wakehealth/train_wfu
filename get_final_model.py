# get the final model 

from optimum.onnxruntime import ORTModelForTokenClassification
from optimum.onnxruntime import ORTOptimizer
from optimum.onnxruntime.configuration import OptimizationConfig
import os

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

if (__name__ == '__main__'):
    # see cv_biodistilbert_plot.ipynb, all models except fold_0 seems just fine
    model_path = 'with_balanced_weighting_biodistilbert_fold_2/checkpoint-3752'
    onnx_model_path = 'wfumodel_no_ipaddress/'
    optimize_model(model_path, onnx_model_path)

