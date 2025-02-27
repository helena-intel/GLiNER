#!/usr/bin/env python
# coding: utf-8

#%pip install gliner==0.2.5 onnx==1.16.2 openvino torch


# based on https://github.com/urchade/GLiNER/blob/main/examples/convert_to_onnx.ipynb

# pip install gliner==0.2.5 onnx==1.16.2 openvino torch 
# see https://github.com/urchade/GLiNER/issues/178 for gliner version: latest gliner errors on ONNX export

import torch
import openvino as ov
from gliner import GLiNER

import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import torch
import torch.onnx
from torch.onnx import register_custom_op_symbolic


# Define the custom symbolic function for prim::PackPadded
def prim_pad_packed_symbolic(g, input, lengths):
    # Implement the shape inference logic here
    # For demonstration purposes, let's assume the output shape is the same as the input shape
    # You need to adjust this based on the actual behavior of prim::PackPadded
    output_shape = input.type().sizes()

    # Create the ONNX node for prim::PackPadded
    return g.op("prim::PadPacked", input, lengths, outputs=1).setType(input.type().with_sizes(output_shape))

def prim_pack_padded_symbolic(g, input, lengths):
    print("Custom symbolic function for prim::PackPadded is called")

    # Calculate the output shape
    input_shape = input.type().sizes()
    batch_size = input_shape[0] if input_shape else 0 
    input_size = input_shape[2] if len(input_shape) > 2 else 0
    total_length = lengths.sum().item() if lengths.is_cuda else lengths.sum().cpu().item()
    output_shape = [total_length, input_size]
    # Create the ONNX node for prim::PackPadded 
    return g.op("prim::PackPadded", input, lengths, outputs=1).setType(input.type().with_sizes(output_shape))  

# Register the custom symbolic function for prim::PackPadded                                                                                                                                                                        
register_custom_op_symbolic("prim::PackPadded", prim_pack_padded_symbolic, 20)
register_custom_op_symbolic("prim::PadPacked", prim_pad_packed_symbolic, 20)   



# model = GLiNER.from_pretrained("urchade/gliner_medium")
model = GLiNER.from_pretrained("gliner-community/gliner_large-v2.5")


# save

model.save_pretrained("gliner_large")
gliner_model = GLiNER.from_pretrained("gliner_large", load_tokenizer=True)



# text = "ONNX is an open-source format designed to enable the interoperability of AI models across various frameworks and tools."
# labels = ['format', 'model', 'tool', 'cat']
# orig_inputs, _ = gliner_model.prepare_model_inputs([text], labels)


import os

onnx_save_path = "gliner_large-v2.5.onnx"


import numpy as np
import json

with open("inputs.json") as f:
    inputs = json.load(f)
# inputs.pop("text_lengths", None)
inputs = {key:torch.as_tensor(value) for key,value in inputs.items()}

# for i in range(inputs1.input_ids.shape[-1]):
#     if inputs1.input_ids[0,i] != inputs2.input_ids[0,i]:
#         print(i, inputs1.input_ids[0,i], inputs2.input_ids[0,i])


if gliner_model.config.span_mode == 'token_level':
    all_inputs =  (inputs['input_ids'], inputs['attention_mask'], 
                    inputs['words_mask'], inputs['text_lengths'])
    input_names = ['input_ids', 'attention_mask', 'words_mask', 'text_lengths']
    dynamic_axes={
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "sequence_length"},
        "words_mask": {0: "batch_size", 1: "sequence_length"},
        "text_lengths": {0: "batch_size", 1: "value"},
        "logits": {0: "position", 1: "batch_size", 2: "sequence_length", 3: "num_classes"},
    }
else:
    all_inputs =  (inputs['input_ids'], inputs['attention_mask'], 
                    inputs['words_mask'], inputs['text_lengths'],
                    inputs['span_idx'], inputs['span_mask'])
    input_names = ['input_ids', 'attention_mask', 'words_mask', 'text_lengths', 'span_idx', 'span_mask']
    dynamic_axes={
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "sequence_length"},
        "words_mask": {0: "batch_size", 1: "sequence_length"},
        "text_lengths": {0: "batch_size", 1: "value"},
        "span_idx": {0: "batch_size", 1: "num_spans", 2: "idx"},
        "span_mask": {0: "batch_size", 1: "num_spans"},
        "logits": {0: "batch_size", 1: "sequence_length", 2: "num_spans", 3: "num_classes"},
    }
print('Converting the model...')
torch.onnx.export(
    gliner_model.model,
    all_inputs,
    f=onnx_save_path,
    input_names=input_names,
    output_names=["logits"],
    # dynamic_axes=dynamic_axes,
    opset_version=14,
    do_constant_folding=True
)


ov_model = ov.convert_model(gliner_model.model,input=[(name,) for name in input_names] , example_input=all_inputs)


ov.save_model(ov_model, "gliner_large.xml")

print(ov_model)

print(ov.Core().read_model(onnx_save_path))
# import nncf
# compressed_model = nncf.compress_weights(ov_model, mode=nncf.CompressWeightsMode.INT8_ASYM)

# ov.save_model(compressed_model, "gliner_large_int8_nightly.xml")




