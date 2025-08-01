/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file tvm/tir/expr.h
 * \brief TIR expressions.
 */
// Acknowledgement: Many low-level IR nodes originate from Halide.
#ifndef TVM_TIR_EXPR_H_
#define TVM_TIR_EXPR_H_

#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/map.h>
#include <tvm/ffi/string.h>
#include <tvm/ir/expr.h>
#include <tvm/node/functor.h>
#include <tvm/node/node.h>
#include <tvm/runtime/base.h>
#include <tvm/runtime/data_type.h>
#include <tvm/tir/buffer.h>
#include <tvm/tir/var.h>

#include <algorithm>
#include <iostream>
#include <limits>
#include <string>
#include <unordered_map>
#include <utility>

namespace tvm {
namespace tir {

using IntImmNode = tvm::IntImmNode;
using FloatImmNode = tvm::FloatImmNode;

/*! \brief String constants, only used in asserts. */
class StringImmNode : public PrimExprNode {
 public:
  /*! \brief The constant value content. */
  String value;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<StringImmNode>().def_ro("value", &StringImmNode::value);
  }

  static constexpr const char* _type_key = "tir.StringImm";
  TVM_DECLARE_FINAL_OBJECT_INFO(StringImmNode, PrimExprNode);
};

/*!
 * \brief Managed reference to StringImmNode.
 * \sa StringImmNode
 */
class StringImm : public PrimExpr {
 public:
  TVM_DLL StringImm(String value, Span span = Span());
  TVM_DEFINE_OBJECT_REF_METHODS(StringImm, PrimExpr, StringImmNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(StringImmNode);
};

/*!
 * \brief Cast value from one data type to another.
 * \note The lanes of value should keep fixed.
 */
class CastNode : public PrimExprNode {
 public:
  /*! \brief Original data type. */
  PrimExpr value;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<CastNode>().def_ro("value", &CastNode::value);
  }

  static constexpr const char* _type_key = "tir.Cast";
  TVM_DECLARE_FINAL_OBJECT_INFO(CastNode, PrimExprNode);
};

/*!
 * \brief Managed reference to CastNode
 * \sa CastNode
 */
class Cast : public PrimExpr {
 public:
  TVM_DLL Cast(DataType dtype, PrimExpr value, Span span = Span());
  TVM_DEFINE_OBJECT_REF_METHODS(Cast, PrimExpr, CastNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(CastNode);
};

/*!
 * \brief Base template to implement binary ops.
 * \tparam T The type of the child class.
 */
template <typename T>
class BinaryOpNode : public PrimExprNode {
 public:
  /*! \brief The left operand. */
  PrimExpr a;
  /*! \brief The right operand. */
  PrimExpr b;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<T>().def_ro("a", &T::a).def_ro("b", &T::b);
  }

  TVM_DECLARE_FINAL_OBJECT_INFO(T, PrimExprNode);
};

/*! \brief a + b */
class AddNode : public BinaryOpNode<AddNode> {
 public:
  static constexpr const char* _type_key = "tir.Add";
};

/*!
 * \brief Managed reference to AddNode
 * \sa AddNode
 */
class Add : public PrimExpr {
 public:
  TVM_DLL Add(PrimExpr a, PrimExpr b, Span span = Span());
  TVM_DEFINE_OBJECT_REF_METHODS(Add, PrimExpr, AddNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(AddNode);
};

/*! \brief a - b */
class SubNode : public BinaryOpNode<SubNode> {
 public:
  static constexpr const char* _type_key = "tir.Sub";
};

/*!
 * \brief Managed reference to SubNode
 * \sa SubNode
 */
class Sub : public PrimExpr {
 public:
  TVM_DLL Sub(PrimExpr a, PrimExpr b, Span span = Span());

  TVM_DEFINE_OBJECT_REF_METHODS(Sub, PrimExpr, SubNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(SubNode);
};

/*! \brief a * b */
class MulNode : public BinaryOpNode<MulNode> {
 public:
  static constexpr const char* _type_key = "tir.Mul";
};

/*!
 * \brief Managed reference to MulNode
 * \sa MulNode
 */
class Mul : public PrimExpr {
 public:
  TVM_DLL Mul(PrimExpr a, PrimExpr b, Span span = Span());
  TVM_DEFINE_OBJECT_REF_METHODS(Mul, PrimExpr, MulNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(MulNode);
};

/*!
 * \brief a / b in the C semnatics.
 * \note For integer division, C standard uses trunc div.
 */
class DivNode : public BinaryOpNode<DivNode> {
 public:
  static constexpr const char* _type_key = "tir.Div";
};

/*!
 * \brief Managed reference to DivNode
 * \sa DivNode
 */
class Div : public PrimExpr {
 public:
  TVM_DLL Div(PrimExpr a, PrimExpr b, Span span = Span());
  TVM_DEFINE_OBJECT_REF_METHODS(Div, PrimExpr, DivNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(DivNode);
};

/*!
 * \brief a % b in the C semnatics.
 * \note For integer division, C standard uses trunc div.
 */
class ModNode : public BinaryOpNode<ModNode> {
 public:
  static constexpr const char* _type_key = "tir.Mod";
};

/*!
 * \brief Managed reference to ModNode
 * \sa ModNode
 */
class Mod : public PrimExpr {
 public:
  TVM_DLL Mod(PrimExpr a, PrimExpr b, Span span = Span());
  TVM_DEFINE_OBJECT_REF_METHODS(Mod, PrimExpr, ModNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(ModNode);
};

/*! \brief Floor division, floor(a/b) */
class FloorDivNode : public BinaryOpNode<FloorDivNode> {
 public:
  static constexpr const char* _type_key = "tir.FloorDiv";
};

/*!
 * \brief Managed reference to FloorDivNode
 * \sa FloorDivNode
 */
class FloorDiv : public PrimExpr {
 public:
  TVM_DLL FloorDiv(PrimExpr a, PrimExpr b, Span span = Span());
  TVM_DEFINE_OBJECT_REF_METHODS(FloorDiv, PrimExpr, FloorDivNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(FloorDivNode);
};

/*! \brief The remainder of the floordiv */
class FloorModNode : public BinaryOpNode<FloorModNode> {
 public:
  static constexpr const char* _type_key = "tir.FloorMod";
};

/*!
 * \brief Managed reference to FloorModNode
 * \sa FloorModNode
 */
class FloorMod : public PrimExpr {
 public:
  TVM_DLL FloorMod(PrimExpr a, PrimExpr b, Span span = Span());
  TVM_DEFINE_OBJECT_REF_METHODS(FloorMod, PrimExpr, FloorModNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(FloorModNode);
};

/*! \brief min(a, b) */
class MinNode : public BinaryOpNode<MinNode> {
 public:
  static constexpr const char* _type_key = "tir.Min";
};

/*!
 * \brief Managed reference to MinNode
 * \sa MinNode
 */
class Min : public PrimExpr {
 public:
  TVM_DLL Min(PrimExpr a, PrimExpr b, Span span = Span());
  TVM_DEFINE_OBJECT_REF_METHODS(Min, PrimExpr, MinNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(MinNode);
};

/*! \brief max(a, b) */
class MaxNode : public BinaryOpNode<MaxNode> {
 public:
  static constexpr const char* _type_key = "tir.Max";
};

/*!
 * \brief Managed reference to MaxNode
 * \sa MaxNode
 */
class Max : public PrimExpr {
 public:
  TVM_DLL Max(PrimExpr a, PrimExpr b, Span span = Span());
  TVM_DEFINE_OBJECT_REF_METHODS(Max, PrimExpr, MaxNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(MaxNode);
};

/*!
 * \brief Base template to implement comparison ops.
 * \tparam T The type of the child class.
 */
template <typename T>
class CmpOpNode : public PrimExprNode {
 public:
  /*! \brief The left operand. */
  PrimExpr a;
  /*! \brief The right operand. */
  PrimExpr b;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<T>().def_ro("a", &T::a).def_ro("b", &T::b);
  }

  TVM_DECLARE_FINAL_OBJECT_INFO(T, PrimExprNode);
};

/*! \brief a == b */
class EQNode : public CmpOpNode<EQNode> {
 public:
  static constexpr const char* _type_key = "tir.EQ";
};

/*!
 * \brief Managed reference to EQNode
 * \sa EQNode
 */
class EQ : public PrimExpr {
 public:
  TVM_DLL EQ(PrimExpr a, PrimExpr b, Span span = Span());
  TVM_DEFINE_OBJECT_REF_METHODS(EQ, PrimExpr, EQNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(EQNode);
};

/*! \brief a != b */
class NENode : public CmpOpNode<NENode> {
 public:
  static constexpr const char* _type_key = "tir.NE";
};

/*!
 * \brief Managed reference to NENode
 * \sa NENode
 */
class NE : public PrimExpr {
 public:
  TVM_DLL NE(PrimExpr a, PrimExpr b, Span span = Span());
  TVM_DEFINE_OBJECT_REF_METHODS(NE, PrimExpr, NENode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(NENode);
};

/*! \brief a < b */
class LTNode : public CmpOpNode<LTNode> {
 public:
  static constexpr const char* _type_key = "tir.LT";
};

/*!
 * \brief Managed reference to LTNode
 * \sa LTNode
 */
class LT : public PrimExpr {
 public:
  TVM_DLL LT(PrimExpr a, PrimExpr b, Span span = Span());
  TVM_DEFINE_OBJECT_REF_METHODS(LT, PrimExpr, LTNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(LTNode);
};

/*! \brief a <= b */
struct LENode : public CmpOpNode<LENode> {
 public:
  static constexpr const char* _type_key = "tir.LE";
};

/*!
 * \brief Managed reference to LENode
 * \sa LENode
 */
class LE : public PrimExpr {
 public:
  TVM_DLL LE(PrimExpr a, PrimExpr b, Span span = Span());
  TVM_DEFINE_OBJECT_REF_METHODS(LE, PrimExpr, LENode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(LENode);
};

/*! \brief a > b */
class GTNode : public CmpOpNode<GTNode> {
 public:
  static constexpr const char* _type_key = "tir.GT";
};

/*!
 * \brief Managed reference to GTNode
 * \sa GTNode
 */
class GT : public PrimExpr {
 public:
  TVM_DLL GT(PrimExpr a, PrimExpr b, Span span = Span());
  TVM_DEFINE_OBJECT_REF_METHODS(GT, PrimExpr, GTNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(GTNode);
};

/*! \brief a >= b */
class GENode : public CmpOpNode<GENode> {
 public:
  static constexpr const char* _type_key = "tir.GE";
};

/*!
 * \brief Managed reference to GENode
 * \sa GENode
 */
class GE : public PrimExpr {
 public:
  TVM_DLL GE(PrimExpr a, PrimExpr b, Span span = Span());
  TVM_DEFINE_OBJECT_REF_METHODS(GE, PrimExpr, GENode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(GENode);
};

/*! \brief a && b */
class AndNode : public PrimExprNode {
 public:
  /*! \brief The left operand. */
  PrimExpr a;
  /*! \brief The right operand. */
  PrimExpr b;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<AndNode>().def_ro("a", &AndNode::a).def_ro("b", &AndNode::b);
  }

  static constexpr const char* _type_key = "tir.And";
  TVM_DECLARE_FINAL_OBJECT_INFO(AndNode, PrimExprNode);
};

/*!
 * \brief Managed reference to AndNode
 * \sa AndNode
 */
class And : public PrimExpr {
 public:
  TVM_DLL And(PrimExpr a, PrimExpr b, Span span = Span());
  TVM_DEFINE_OBJECT_REF_METHODS(And, PrimExpr, AndNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(AndNode);
};

/*! \brief a || b */
class OrNode : public PrimExprNode {
 public:
  /*! \brief The left operand. */
  PrimExpr a;
  /*! \brief The right operand. */
  PrimExpr b;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<OrNode>().def_ro("a", &OrNode::a).def_ro("b", &OrNode::b);
  }

  static constexpr const char* _type_key = "tir.Or";
  TVM_DECLARE_FINAL_OBJECT_INFO(OrNode, PrimExprNode);
};

/*!
 * \brief Managed reference to OrNode
 * \sa OrNode
 */
class Or : public PrimExpr {
 public:
  TVM_DLL Or(PrimExpr a, PrimExpr b, Span span = Span());
  TVM_DEFINE_OBJECT_REF_METHODS(Or, PrimExpr, OrNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(OrNode);
};

/*! \brief !a */
class NotNode : public PrimExprNode {
 public:
  /*! \brief The input operand. */
  PrimExpr a;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<NotNode>().def_ro("a", &NotNode::a);
  }

  static constexpr const char* _type_key = "tir.Not";
  TVM_DECLARE_FINAL_OBJECT_INFO(NotNode, PrimExprNode);
};

/*!
 * \brief Managed reference to NotNode
 * \sa NotNode
 */
class Not : public PrimExpr {
 public:
  TVM_DLL Not(PrimExpr a, Span span = Span());
  TVM_DEFINE_OBJECT_REF_METHODS(Not, PrimExpr, NotNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(NotNode);
};

/*!
 * \brief return true_value if condition is true, otherwise return false_value.
 * \note Both true_value and false_value could be evaluated
 *       regardless of the condition value.
 *       Do not use it to guard against out of bound access,
 *       please use if_then_else instead.
 */
class SelectNode : public PrimExprNode {
 public:
  /*! \brief The condition */
  PrimExpr condition;
  /*! \brief value to be returned when condition is true. */
  PrimExpr true_value;
  /*! \brief value to be returned when condition is false. */
  PrimExpr false_value;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<SelectNode>()
        .def_ro("condition", &SelectNode::condition)
        .def_ro("true_value", &SelectNode::true_value)
        .def_ro("false_value", &SelectNode::false_value);
  }

  static constexpr const char* _type_key = "tir.Select";
  TVM_DECLARE_FINAL_OBJECT_INFO(SelectNode, PrimExprNode);
};

/*!
 * \brief Managed reference to SelectNode
 * \sa SelectNode
 */
class Select : public PrimExpr {
 public:
  TVM_DLL Select(PrimExpr condition, PrimExpr true_value, PrimExpr false_value, Span span = Span());

  TVM_DEFINE_OBJECT_REF_METHODS(Select, PrimExpr, SelectNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(SelectNode);
};

/*!
 * \brief Load value from the high dimension buffer.
 *
 * \code
 *
 *  value = buffer[i, j];
 *
 * \endcode
 * \sa BufferStore
 */
class BufferLoadNode : public PrimExprNode {
 public:
  /*! \brief The buffer variable. */
  Buffer buffer;
  /*! \brief The indices location to be loaded. */
  Array<PrimExpr> indices;
  /*! \brief The predicate mask for loading values. */
  Optional<PrimExpr> predicate;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<BufferLoadNode>()
        .def_ro("buffer", &BufferLoadNode::buffer)
        .def_ro("indices", &BufferLoadNode::indices)
        .def_ro("predicate", &BufferLoadNode::predicate);
  }

  static constexpr const char* _type_key = "tir.BufferLoad";
  TVM_DECLARE_FINAL_OBJECT_INFO(BufferLoadNode, PrimExprNode);

 private:
  /*! \brief Set the dtype based on the buffer/indices
   *
   * Usually, the BufferLoad's dtype will be the same dtype as the
   * buffer.  This may have a different number of lanes than the
   * buffer's dtype if index values have more than 1 lane.
   *
   * This function should only be called during construction and after
   * CopyOnWrite.  Friend class used here to restrict usage.
   */
  void LegalizeDType();
  friend class BufferLoad;
  friend class CustomDatatypesLowerer;
  friend class VectorTypeRewriter;
  friend class Vectorizer;
};

/*!
 * \brief Managed reference to BufferLoadNode.
 * \sa BufferLoadNode
 */
class BufferLoad : public PrimExpr {
 public:
  TVM_DLL explicit BufferLoad(Buffer buffer, Array<PrimExpr> indices,
                              Optional<PrimExpr> predicate = std::nullopt, Span span = Span());
  TVM_DEFINE_OBJECT_REF_METHODS(BufferLoad, PrimExpr, BufferLoadNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(BufferLoadNode);
};

/*!
 * \brief Load value from the result produced by the producer.
 *
 * \note This node only appears in high-level DSLs that are built on top of the TIR.
 *       It should not appear in a valid TIR PrimFunc. A high-level DSL needs to lower
 *       this node before TIR transformations.
 *
 * \sa ProducerLoad, DataProducerNode
 */
class ProducerLoadNode : public PrimExprNode {
 public:
  /*! \brief The buffer producer. */
  DataProducer producer;
  /*! \brief The location arguments. */
  Array<PrimExpr> indices;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<ProducerLoadNode>()
        .def_ro("producer", &ProducerLoadNode::producer)
        .def_ro("indices", &ProducerLoadNode::indices);
  }

  static constexpr const char* _type_key = "tir.ProducerLoad";
  TVM_DECLARE_FINAL_OBJECT_INFO(ProducerLoadNode, PrimExprNode);
};

/*!
 * \brief Managed reference to ProducerLoadNode.
 * \sa ProducerLoadNode
 */
class ProducerLoad : public PrimExpr {
 public:
  TVM_DLL explicit ProducerLoad(DataProducer producer, Array<PrimExpr> indices, Span span = Span());

  TVM_DEFINE_OBJECT_REF_METHODS(ProducerLoad, PrimExpr, ProducerLoadNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(ProducerLoadNode);
};

/*!
 * \brief Construct a vector with lanes elements
 *        where its i-th element equals base + i * stride.
 *  This is useful to construct a index for a continuous vector load.
 *
 *  Examples:
 *  - ramp(0, 1, 3) = [0, 1, 2]
 *  - ramp(1, 2, 4) = [1, 3, 5, 7]
 */
class RampNode : public PrimExprNode {
 public:
  /*! \brief The base value. */
  PrimExpr base;
  /*! \brief The stride of each step. */
  PrimExpr stride;
  /*! \brief Total number of lanes. */
  PrimExpr lanes;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<RampNode>()
        .def_ro("base", &RampNode::base)
        .def_ro("stride", &RampNode::stride)
        .def_ro("lanes", &RampNode::lanes);
  }

  static constexpr const char* _type_key = "tir.Ramp";
  TVM_DECLARE_FINAL_OBJECT_INFO(RampNode, PrimExprNode);
};

/*!
 * \brief Managed reference to RampNode
 * \sa RampNode
 */
class Ramp : public PrimExpr {
 public:
  TVM_DLL Ramp(PrimExpr base, PrimExpr stride, PrimExpr lanes, Span span = Span());
  TVM_DEFINE_OBJECT_REF_METHODS(Ramp, PrimExpr, RampNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(RampNode);
};

/*! \brief Create a vector where all the elements are value. */
class BroadcastNode : public PrimExprNode {
 public:
  /*! \brief The base value. */
  PrimExpr value;
  /*! \brief The number of lanes. */
  PrimExpr lanes;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<BroadcastNode>()
        .def_ro("value", &BroadcastNode::value)
        .def_ro("lanes", &BroadcastNode::lanes);
  }

  static constexpr const char* _type_key = "tir.Broadcast";
  TVM_DECLARE_FINAL_OBJECT_INFO(BroadcastNode, PrimExprNode);
};

/*!
 * \brief Managed reference to BroadcastNode
 * \sa BroadcastNode
 */
class Broadcast : public PrimExpr {
 public:
  TVM_DLL Broadcast(PrimExpr value, PrimExpr lanes, Span span = Span());
  TVM_DEFINE_OBJECT_REF_METHODS(Broadcast, PrimExpr, BroadcastNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(BroadcastNode);
};

/*!
 * \brief Let binding. Bind var to value then evaluate body.
 */
class LetNode : public PrimExprNode {
 public:
  /*! \brief The variable. */
  Var var;
  /*! \brief The value to be binded. */
  PrimExpr value;
  /*! \brief The result expression. */
  PrimExpr body;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<LetNode>()
        .def_ro("var", &LetNode::var, refl::AttachFieldFlag::SEqHashDef())
        .def_ro("value", &LetNode::value)
        .def_ro("body", &LetNode::body);
  }

  static constexpr const char* _type_key = "tir.Let";
  TVM_DECLARE_FINAL_OBJECT_INFO(LetNode, PrimExprNode);
};

/*!
 * \brief Managed reference to LetNode
 * \sa LetNode
 */
class Let : public PrimExpr {
 public:
  TVM_DLL Let(Var var, PrimExpr value, PrimExpr body, Span span = Span());
  TVM_DEFINE_OBJECT_REF_METHODS(Let, PrimExpr, LetNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(LetNode);
};

/*!
 * \brief Call node.
 */
class CallNode : public PrimExprNode {
 public:
  /*!
   * \brief The operator(function) being invoked
   *
   *  - It can be tvm::Op which corresponds to the primitive operators(intrinsics).
   *  - It can also be another function in the IRModule (GlobalVar).
   */
  RelaxExpr op;

  /*! \brief The arguments. */
  Array<PrimExpr> args;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<CallNode>().def_ro("op", &CallNode::op).def_ro("args", &CallNode::args);
  }

  static constexpr const char* _type_key = "tir.Call";
  TVM_DECLARE_FINAL_OBJECT_INFO(CallNode, PrimExprNode);
};

/*!
 * \brief Managed reference to CallNode
 * \sa CallNode
 */
class Call : public PrimExpr {
 public:
  TVM_DLL Call(DataType dtype, RelaxExpr op, Array<PrimExpr> args, Span span = Span());
  TVM_DEFINE_OBJECT_REF_METHODS(Call, PrimExpr, CallNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(CallNode);
};

/*!
 * \brief Shuffle instruction.
 *  vec = concat(vectors)
 *  result = (vec[indices[0]], vec[indices[1]] ...)
 */
class ShuffleNode : public PrimExprNode {
 public:
  /*! \brief the input vectors. */
  Array<PrimExpr> vectors;
  /*! \brief The indices of each element. */
  Array<PrimExpr> indices;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<ShuffleNode>()
        .def_ro("vectors", &ShuffleNode::vectors)
        .def_ro("indices", &ShuffleNode::indices);
  }

  static constexpr const char* _type_key = "tir.Shuffle";
  TVM_DECLARE_FINAL_OBJECT_INFO(ShuffleNode, PrimExprNode);
};

/*!
 * \brief Managed reference to ShuffleNode
 * \sa ShuffleNode
 */
class Shuffle : public PrimExpr {
 public:
  TVM_DLL Shuffle(Array<PrimExpr> vectors, Array<PrimExpr> indices, Span span = Span());
  TVM_DLL static PrimExpr Concat(Array<PrimExpr> vectors, Span span = Span());
  TVM_DLL static PrimExpr ExtractElement(PrimExpr vector, int index, Span span = Span());

  TVM_DEFINE_OBJECT_REF_METHODS(Shuffle, PrimExpr, ShuffleNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(ShuffleNode);
};

// Reduce operator
/*!
 * \brief A commutative reducer node to represent a commutative
 *  binary operator with identity element
 */
class CommReducerNode : public Object {
 public:
  /*! \brief The left argument of reducer */
  Array<Var> lhs;
  /*! \brief The right argument of reducer */
  Array<Var> rhs;
  /*! \brief The result of reducer */
  Array<PrimExpr> result;
  /*!
   * \brief The identity element of reducer, which leaves other
   *  elements unchanged when combined with it, with respect to
   *  the binary operation of this reducer uses.
   */
  Array<PrimExpr> identity_element;
  /*! \brief Function call operator to combine a and b */
  Array<PrimExpr> operator()(Array<PrimExpr> a, Array<PrimExpr> b) const;
  /*!
   * \brief Span that points to the original source code.
   *        Reserved debug information.
   */
  mutable Span span;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<CommReducerNode>()
        .def_ro("lhs", &CommReducerNode::lhs, refl::AttachFieldFlag::SEqHashDef())
        .def_ro("rhs", &CommReducerNode::rhs, refl::AttachFieldFlag::SEqHashDef())
        .def_ro("result", &CommReducerNode::result)
        .def_ro("identity_element", &CommReducerNode::identity_element)
        .def_ro("span", &CommReducerNode::span, refl::AttachFieldFlag::SEqHashIgnore());
  }

  static constexpr const char* _type_key = "tir.CommReducer";
  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindTreeNode;
  TVM_DECLARE_FINAL_OBJECT_INFO(CommReducerNode, Object);
};

/*!
 * \brief Managed reference to CommReducerNode
 * \sa CommReducerNode
 */
class CommReducer : public ObjectRef {
 public:
  TVM_DLL CommReducer(Array<Var> lhs, Array<Var> rhs, Array<PrimExpr> result,
                      Array<PrimExpr> identity_element, Span span = Span());

  TVM_DEFINE_OBJECT_REF_METHODS(CommReducer, ObjectRef, CommReducerNode);
};

/*! \brief Reduction operator */
class ReduceNode : public PrimExprNode {
 public:
  /*! \brief The commutative combiner */
  CommReducer combiner;
  /*! \brief The source operand */
  Array<PrimExpr> source;
  /*! \brief The init operand */
  Array<PrimExpr> init;
  /*! \brief The reduction axis */
  Array<IterVar> axis;
  /*!
   * \brief Predicate on the reduction
   *  Only add the body to reduction if condition is true.
   */
  PrimExpr condition;
  /*! \brief the index of this reduce node */
  int value_index;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<ReduceNode>()
        .def_ro("combiner", &ReduceNode::combiner)
        .def_ro("source", &ReduceNode::source)
        .def_ro("init", &ReduceNode::init)
        .def_ro("axis", &ReduceNode::axis)
        .def_ro("condition", &ReduceNode::condition)
        .def_ro("value_index", &ReduceNode::value_index);
  }

  static constexpr const char* _type_key = "tir.Reduce";
  TVM_DECLARE_FINAL_OBJECT_INFO(ReduceNode, PrimExprNode);
};

/*!
 * \brief Managed reference to ReduceNode
 * \sa ReduceNode
 */
class Reduce : public PrimExpr {
 public:
  TVM_DLL Reduce(CommReducer combiner, Array<PrimExpr> src, Array<IterVar> rdom, PrimExpr condition,
                 int value_index, Array<PrimExpr> init, Span span = Span());

  TVM_DEFINE_OBJECT_REF_METHODS(Reduce, PrimExpr, ReduceNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(ReduceNode);
};

/*
 * \brief Template function to convert Map to unordered_map
 *  Sometimes useful for API gluing when internal uses unordered_map
 * \param dmap The container map
 * \return The corresponding unordered_map.
 * \tparam K the key of the Map.
 * \tparam V the value of the Map.
 */
template <typename K, typename V>
inline std::unordered_map<K, V> as_unordered_map(const Map<K, V>& dmap) {
  std::unordered_map<K, V> ret;
  for (auto kv : dmap) {
    ret[kv.first] = kv.second;
  }
  return ret;
}
}  // namespace tir

namespace ffi {

template <>
inline constexpr bool use_default_type_traits_v<tvm::tir::StringImm> = false;

template <>
struct TypeTraits<tvm::tir::StringImm>
    : public ObjectRefWithFallbackTraitsBase<tvm::tir::StringImm, String> {
  TVM_FFI_INLINE static tvm::tir::StringImm ConvertFallbackValue(String value) {
    return tvm::tir::StringImm(value);
  }
};
}  // namespace ffi
}  // namespace tvm

namespace std {
template <>
struct hash<::tvm::tir::IterVar> : public ::tvm::ObjectPtrHash {};
}  // namespace std
#endif  // TVM_TIR_EXPR_H_
