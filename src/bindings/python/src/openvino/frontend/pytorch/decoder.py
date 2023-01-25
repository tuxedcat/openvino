# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# flake8: noqa
# mypy: ignore-errors

from openvino.frontend.pytorch.py_pytorch_frontend import _FrontEndPytorchDecoder as Decoder
from openvino.frontend.pytorch.py_pytorch_frontend import _Type as DecoderType
from openvino.runtime import op, PartialShape, Type as OVType, OVAny, Shape

import warnings
import torch


def get_type_from_py_type(value):
    if isinstance(value, float):
        return OVType.f32
    if isinstance(value, int):
        return OVType.i32
    if isinstance(value, bool):
        return OVType.boolean
    return OVType.dynamic


def ivalue_to_constant(ivalue):
    ov_type = get_type_from_py_type(ivalue)
    if ov_type.is_static():
        return op.Constant(ov_type, Shape([]), [ivalue]).outputs()

    if isinstance(ivalue, list):
        assert len(ivalue) > 0, "Can't deduce type for empty list"
        ov_type = get_type_from_py_type(ivalue[0])
        assert ov_type.is_static(), "Can't deduce type for list"
        return op.Constant(ov_type, Shape([len(ivalue)]), ivalue).outputs()

    if ivalue.type() in pt_to_ov_type_map:
        try:
            ovshape = PartialShape(ivalue.size())
            ovtype = pt_to_ov_type_map[ivalue.type()]
            ov_const = op.Constant(ovtype, ovshape.get_shape(), ivalue.data_ptr())
        except Exception:
            # old variant that makes a slow data copying
            warnings.warn("[ WARNING ] Constant wasn't able to convert from data_ptr.")
            nvalues = ivalue.numpy()
            ovtype = np_to_ov_type_map[str(nvalues.dtype)]
            ovshape = PartialShape(nvalues.shape)
            ov_const = op.Constant(ovtype, ovshape.get_shape(), nvalues.flatten().tolist())
        return ov_const.outputs()


def get_value_from_getattr(getattr_node, self_module):
    assert getattr_node.kind() == "prim::GetAttr", "Got node of kind not equal to prim::GetAttr"
    # GetAttr nodes can be nested
    stack = []
    while getattr_node.kind() == "prim::GetAttr":
        stack.append(getattr_node)
        inputs = list(getattr_node.inputs())
        if len(inputs) == 0:
            break
        getattr_node = inputs[0].node()
    module = self_module
    while len(stack) > 0:
        node = stack.pop()
        assert (hasattr(module, node.s("name")))
        module = getattr(module, node.s("name"))
    return module


pt_to_ov_type_map = {
    "float": OVType.f32,
    "int": OVType.i32,
    "torch.float32": OVType.f32,
    "torch.int32": OVType.i32,
    "torch.bool": OVType.boolean,
    "torch.int64": OVType.i64,
    "torch.FloatTensor": OVType.f32,
    "torch.IntTensor": OVType.i32,
    "torch.LongTensor": OVType.i64,
    "torch.BoolTensor": OVType.boolean,
}

pt_to_py_type_map = {
    "float": "float",
    "int": "int",
    "torch.float32": "float",
    "torch.int32": "int",
    "torch.int64": "int",
    "torch.bool": "bool",
}

np_to_ov_type_map = {
    "float32": OVType.f32,
    "int32": OVType.i32,
}


class TorchScriptPythonDecoder (Decoder):
    def __init__(self, pt_module, graph_element=None):
        Decoder.__init__(self)
        # We store every decoder created by this decoder so that all them are not deleted until the first decoder is deleted
        self.m_decoders = []
        if graph_element is None:
            assert hasattr(pt_module, "inlined_graph"), "graph_element must have inlined_graph"
            self.graph_element = pt_module.inlined_graph
        else:
            self.graph_element = graph_element
        self.pt_module = pt_module

    def inputs(self):
        return [x.unique() for x in self.graph_element.inputs()]

    def get_input(self, index):
        return self.inputs()[index]

    def get_input_shape(self, index):
        raw_input = self._raw_input(index)
        return self.get_shape_for_value(raw_input)

    def get_input_type(self, index):
        raw_input = self._raw_input(index)
        return self.get_type_for_value(raw_input)

    def get_output_shape(self, index):
        output = self._raw_output(index)
        return self.get_shape_for_value(output)

    def get_output_type(self, index):
        output = self._raw_output(index)
        return self.get_type_for_value(output)

    def _get_known_type_for_value(self, pt_type):
        """Returns known/unknown types wrapped as OVAny."""
        # Check for simple scalar types first
        if pt_type is None:
            return OVAny(OVType.dynamic)
        # TODO: Don't use str, use native types
        if str(pt_type) in pt_to_ov_type_map:
            return OVAny(pt_to_ov_type_map[str(pt_type)])
        elif pt_type.__class__ is torch.TensorType:
            # Tensor type, parse element type
            return OVAny(DecoderType.Tensor(self._get_known_type_for_value(pt_type.dtype())))
        elif pt_type.__class__ is torch.ListType:
            element_type = pt_type.getElementType()
            return OVAny(DecoderType.List(self._get_known_type_for_value(element_type)))
        else:
            # Not yet recognized
            return OVAny(OVType.dynamic)

    def get_shape_for_value(self, value):
        if value.isCompleteTensor():
            ps = PartialShape(value.type().sizes())
            return ps
        else:
            # TODO: Recognize types that we can represent as a nested constructs with objects from DecoderType
            # If recognized, return scalar instead of dynamic. Scalar means a single value of that custom type.
            # See get_type_for_value for reference
            pass
        return PartialShape.dynamic()

    def get_type_for_value(self, value):
        full_type = self._get_known_type_for_value(value.type())
        return full_type

    def get_input_transpose_order(self, index):
        raw_input = self._raw_input(index)
        if raw_input.type() is not None and raw_input.type().kind() == "TensorType":
            strides = raw_input.type().strides()
            if strides is not None:
                return [s[0] for s in sorted(enumerate(strides), key=lambda x:x[1], reverse=True)]
        return []

    def get_output_transpose_order(self, index):
        output = self._raw_output(index)
        if output.type() is not None and output.type().kind() == "TensorType":
            strides = output.type().strides()
            if strides is not None:
                return [s[0] for s in sorted(enumerate(strides), key=lambda x:x[1], reverse=True)]
        return []

    def get_subgraph_size(self):
        return len(self.get_subgraphs()) if hasattr(self.graph_element, "blocks") else 1

    def visit_subgraph(self, node_visitor):
        # make sure topological order is satisfied
        for node in self.graph_element.nodes():
            decoder = TorchScriptPythonDecoder(self.pt_module, node)
            self.m_decoders.append(decoder)
            node_visitor(decoder)

    def get_subgraphs(self):
        return list(self.graph_element.blocks())

    def get_subgraph_decoder(self, index):
        decoder = TorchScriptPythonDecoder(self.pt_module, self.get_subgraphs()[index])
        self.m_decoders.append(decoder)
        return decoder

    def get_op_type(self):
        return self.graph_element.kind()

    def get_schema(self):
        return self.graph_element.schema()

    def outputs(self):
        return [x.unique() for x in self.graph_element.outputs()]

    def _raw_outputs(self):
        return list(self.graph_element.outputs())

    def _raw_output(self, index):
        return self._raw_outputs()[index]

    def _raw_inputs(self):
        return list(self.graph_element.inputs())

    def _raw_input(self, index):
        return self._raw_inputs()[index]

    def num_of_outputs(self):
        return len(self.outputs())

    def output(self, index):
        return self.outputs()[index]

    def mark_node(self, node):
        return node

    def try_decode_get_attr(self):
        pt_value = get_value_from_getattr(self.graph_element, self.pt_module)
        assert pt_value is not None, "Couldn't retrieve value from prim::GetAttr"
        if not isinstance(pt_value, torch.jit.ScriptModule) or isinstance(pt_value, torch.jit.TracedModule):
            return ivalue_to_constant(pt_value)
        else:
            return []

    def as_constant(self):
        if not self.get_op_type() == "prim::Constant":
            return None
        pt_value = self._raw_output(0)

        pt_type_class = pt_value.type().__class__
        if pt_type_class is torch.TensorType:
            return self.as_constant_tensor(pt_value)
        if pt_type_class is torch.ListType:
            return self.as_constant_list(pt_value)
        if str(pt_value.type()) in ["torch.int32", "int"]:
            return op.Constant(OVType.i32, Shape([]), [pt_value.toIValue()]).outputs()
        if str(pt_value.type()) in ["torch.float", "torch.FloatType", "float"]:
            return op.Constant(OVType.f32, Shape([]), [pt_value.toIValue()]).outputs()
        if str(pt_value.type()) in ["torch.bool", "bool"]:
            return op.Constant(OVType.boolean, Shape([]), [pt_value.toIValue()]).outputs()

        return None

    def as_string(self):
        if not self.get_op_type() == "prim::Constant":
            return None
        pt_value = self._raw_output(0)

        if str(pt_value.type()) in ["torch.StringType", "str"]:
            return pt_value.toIValue()
        return None

    def as_constant_tensor(self, pt_value):
        ivalue = pt_value.toIValue()
        if pt_value.isCompleteTensor():
            try:
                ivalue = ivalue.to(memory_format=torch.contiguous_format).detach().cpu()
            except Exception:
                warnings.warn("[ WARNING ] Tensor couldn't detach")
            if str(pt_value.type().dtype()) in pt_to_ov_type_map:
                # Constant interpretation doesn't respect new-full type of PT
                # It recognizes only tensors, and give lists as 1D tensors, and scalars as Tensor scalars
                # So only tensor-type constants are supported
                ovshape = PartialShape(pt_value.type().sizes())
                ovtype = pt_to_ov_type_map[str(pt_value.type().dtype())]

                # TODO: try-except here is a temporary WA for issues with data_ptr that we currently cannot predict; provide better solution
                try:
                    # this is only possible with adding a new ctor for Constant Python binding
                    # TODO Check strides and pass them somehow
                    values = ivalue.data_ptr()
                    ov_const = op.Constant(ovtype, ovshape.get_shape(), values)
                except Exception:
                    # old variant that makes a slow data copying
                    warnings.warn("[ WARNING ] Constant wasn't able to convert from data_ptr.")
                    values = ivalue.flatten().tolist()
                    ov_const = op.Constant(ovtype, ovshape.get_shape(), values)
                return ov_const.outputs()
        else:
            return ivalue_to_constant(ivalue)
        return None

    def as_constant_list(self, pt_value):
        # For now it is treat a list as a 1D tensor; it is required by converters to avoid need to massively
        # rewrite them in that part where constant attributes are queried
        pt_element_type = str(pt_value.type().getElementType())
        ivalue = pt_value.toIValue()
        is_known_type = pt_element_type in pt_to_ov_type_map

        if is_known_type:
            ovtype = pt_to_ov_type_map[pt_element_type]
            ovshape = PartialShape([len(ivalue)])
            ov_const = op.Constant(ovtype, ovshape.get_shape(), ivalue)
            return ov_const.outputs()

    def input_is_none(self, index):
        if index >= len(self.inputs()) or self._raw_input(index) is None:
            return True
        else:
            r_input = self._raw_input(index)
            if str(r_input.type()) in ["torch.NoneType", "NoneType"]:
                return True
            else:
                in_node = r_input.node()
                if in_node.kind() == "prim::GetAttr":
                    pt_value = get_value_from_getattr(in_node, self.pt_module)
                    return pt_value is None
        return False
