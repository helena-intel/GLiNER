#!/usr/bin/env python
# coding: utf-8

import onnx
from onnx import helper, numpy_helper
import numpy as np
import json


# onnx_model_path = "gliner_large-v2.5_static_slice_cumsum.onnx"# this file should exist
# new_onnx_model_path = "gliner_large-v2.5-static_slice_cumsum_reduced.onnx"
# ov_model_path = "gliner_large_static_cumsum.xml"
# int8_ov_model_path = "gliner_large_static_int8_cumsum.xml"

onnx_model_path = "gliner_large-v2.5.onnx"
new_onnx_model_path = "gliner_large-v2.5-reduced.onnx"
ov_model_path = "gliner_large_static.xml"
int8_ov_model_path = "gliner_large_static_int8.xml"

new_shape = [1,102,12,57]
text_lengths = np.array([[102]])


# ## Reduce ONNX model
# 
# Keep only logits output layer, replace text_lengths parameter with constant

onnx_model = onnx.load(onnx_model_path)
graph = onnx_model.graph
# replace text_lengths parameter with constant
#graph.input.remove(next(i for i in graph.input if i.name == "text_lengths"))
#constant_tensor = numpy_helper.from_array(text_lengths, name="text_lengths")
#constant_node = helper.make_node(
#    'Constant',
#    inputs=[],
#    outputs=["text_lengths"],
#    value=constant_tensor
#)
#graph.node.insert(0, constant_node)
#
new_output=[output for output in graph.output if output.name == "logits"][0]
graph.ClearField("output")
shape = new_output.type.tensor_type.shape
shape.ClearField('dim')
for dim in new_shape:
    shape.dim.add().dim_value = dim
graph.output.append(new_output)

#for input in graph.input:
#    input_type = input.type.tensor_type
#    if input_type.HasField("shape"):
#        for dim  in input_type.shape.dim:
#            dim.dim_param = "None"
#


shapes = {}
with open("inputs.json") as f:
    inputs = json.load(f)
    inputs.pop("text_lengths")
shapes = {key: np.array(value).shape for key,value in inputs.items()}


## Modify model inputs to have static shapes
#for input_tensor in graph.input:
#    input_name = input_tensor.name
#    if input_name in shapes:
#        new_shape = shapes[input_name]  # Get static shape from the dict
#        print(f"Updating input {input_name} to shape {new_shape}")
#
#        # Update the input tensor shape
#        input_tensor.type.tensor_type.shape.ClearField("dim")  # Clear existing shape
#        for dim_size in new_shape:
#            input_tensor.type.tensor_type.shape.dim.add().dim_value = dim_size  # Set static shape
#

# Identify NonZero nodes
for node in graph.node:
    if node.op_type == "NonZero":
        print(f"Replacing NonZero Node: {node.name}")

        # Define a fixed-size output tensor (adjust shape as needed)
        max_elements = 1224 # Set a safe upper bound
        dummy_indices = np.zeros((2, max_elements), dtype=np.int64)  # Assuming 2D indices

        # Create a constant tensor to replace NonZero
        static_output = numpy_helper.from_array(dummy_indices, name=node.output[0] + "_static")

        # Replace all references to NonZero output with the static tensor
        for next_node in graph.node:
            for i, inp in enumerate(next_node.input):
                if inp == node.output[0]:
                    next_node.input[i] = static_output.name

        # Add the constant tensor to the graph
        graph.initializer.append(static_output)

        # Remove the NonZero node
        graph.node.remove(node)

onnx.save(onnx_model, new_onnx_model_path)


import openvino as ov

ov.Core().read_model(new_onnx_model_path)


from openvino_devtools.ov2py import ov2py
import openvino as ov

model = ov.Core().read_model(new_onnx_model_path)
ov.save_model(model, ov_model_path)

ov2py_model = ov2py(model)
# number of dynamic layers
len([line for line in ov2py_model.splitlines() if "?" in line or ".." in line])
for line in ov2py_model.splitlines():
    if "?" in line or ".." in line:
        print(line)

