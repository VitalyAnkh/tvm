# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=invalid-name
"""Default legalization function for index operators."""
from tvm import topi, tir, te
from ...op import call_pure_packed
from ...block_builder import BlockBuilder
from ...expr import Call, Expr
from ...struct_info import ShapeStructInfo, PrimStructInfo
from .common import register_legalize


@register_legalize("relax.take")
def _take(bb: BlockBuilder, call: Call) -> Expr:
    # Currently "fast" is the default mode, which leads to segmentation faults
    # when there are out-of-bounds indices.
    return bb.call_te(topi.take, call.args[0], call.args[1], call.attrs.axis, mode=call.attrs.mode)


@register_legalize("relax.strided_slice")
def _strided_slice(bb: BlockBuilder, call: Call) -> Expr:
    def _relax_tuple_to_tir(relax_tuple):
        output = []
        for field in relax_tuple.struct_info.fields:
            assert isinstance(field, PrimStructInfo)
            assert field.value is not None
            output.append(field.value)
        return output

    if len(call.args) == 4:
        data, axes, begin, end = call.args
        strides = [tir.IntImm("int64", 1)] * len(axes.struct_info.fields)
    elif len(call.args) == 5:
        data, axes, begin, end, strides = call.args
        strides = _relax_tuple_to_tir(strides)
    else:
        raise ValueError(
            f"Expression {call} provides {len(call.args)} arguments, "
            f"but {call.op} requires either 4 or 5 arguments."
        )

    axes = _relax_tuple_to_tir(axes)
    begin = _relax_tuple_to_tir(begin)
    end = _relax_tuple_to_tir(end)

    return bb.call_te(
        topi.strided_slice,
        data,
        begin,
        end,
        strides,
        axes,
        slice_mode="end",
        assume_inbound=call.attrs.assume_inbound,
    )


@register_legalize("relax.dynamic_strided_slice")
def _dynamic_strided_slice(bb: BlockBuilder, call: Call) -> Expr:
    assert len(call.args) == 4
    data, begin, end, strides = call.args

    # 1. Insert shape function
    def shape_func(data, begin, end, strides):
        def _compute(i):
            def canonicalize_index(index, extent, strides):
                begin_range = tir.Select(strides < 0, tir.const(-1, "int64"), tir.const(0, "int64"))
                end_range = tir.Select(strides < 0, extent - 1, extent)
                index = tir.Select(index < 0, index + extent, index)
                return tir.Min(tir.Max(index, begin_range), end_range)

            def get_length(begin, end, strides, length):
                begin = canonicalize_index(begin, length, strides)
                end = canonicalize_index(end, length, strides)
                len1 = tir.ceildiv(begin - end, -strides)
                len2 = tir.ceildiv(end - begin, strides)
                return tir.Select(strides < 0, len1, len2)

            length = tir.const(-1, "int64")
            for idx in range(data.ndim):
                length = tir.Select(i == tir.const(idx, "int64"), data.shape[idx], length)

            return get_length(begin[i], end[i], strides[i], length)

        return te.compute((begin.shape[0],), _compute, name="T_shape_func_strided_slice_dynamic")

    output_shape = bb.normalize(
        bb.call_te(
            shape_func,
            data,
            begin,
            end,
            strides,
        )
    )

    # 2. Convert tensor to shape and match cast with new symbolic vars
    # Get shape length
    ndim = int(output_shape.struct_info.shape[0])
    output_shape = bb.emit(
        # TODO(@relax-team): Ideally, we should use the tensor_to_shape op here to
        # address the issue with purity, but that introduces a staging issue:
        # we need to apply DecomposeOpsForInference in that case
        # and it's unclear when in the build it should happen
        call_pure_packed(
            "vm.builtin.tensor_to_shape", output_shape, sinfo_args=ShapeStructInfo(ndim=ndim)
        )
    )
    output_shape_vars = [tir.Var("s", "int64") for i in range(ndim)]
    bb.match_cast(output_shape, ShapeStructInfo(output_shape_vars))

    # 3. Pass the output shape vars to TOPI
    return bb.call_te(
        topi.dynamic_strided_slice,
        call.args[0],
        call.args[1],
        call.args[2],
        call.args[3],
        output_shape=output_shape_vars,
    )
