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
 * \file codegen_cpu.cc
 */
#ifdef TVM_LLVM_VERSION

#include "codegen_cpu.h"

#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/IR/Argument.h>
#include <llvm/IR/Attributes.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/CallingConv.h>
#include <llvm/IR/Comdat.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DIBuilder.h>
#include <llvm/IR/DebugInfoMetadata.h>
#include <llvm/IR/DebugLoc.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/MDBuilder.h>
#include <llvm/IR/Metadata.h>
#include <llvm/IR/Module.h>
#include <tvm/ffi/reflection/registry.h>
#if TVM_LLVM_VERSION >= 100
#include <llvm/Support/Alignment.h>
#endif
#include <llvm/Support/raw_ostream.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Transforms/Utils/ModuleUtils.h>
#include <tvm/runtime/base.h>
#include <tvm/runtime/module.h>
#include <tvm/tir/analysis.h>

#include <algorithm>
#include <memory>
#include <tuple>
#include <unordered_map>
#include <unordered_set>

#include "llvm_instance.h"

namespace tvm {
namespace codegen {

// Make these non-inline because of std::unique_ptr. See comment in
// codegen_llvm.cc for more information.
CodeGenCPU::CodeGenCPU() = default;
CodeGenCPU::~CodeGenCPU() = default;

void CodeGenCPU::Init(const std::string& module_name, LLVMTarget* llvm_target,
                      Optional<String> system_lib_prefix, bool dynamic_lookup,
                      bool target_c_runtime) {
  CodeGenLLVM::Init(module_name, llvm_target, system_lib_prefix, dynamic_lookup, target_c_runtime);
  system_lib_prefix_ = system_lib_prefix;
  dbg_info_ = CreateDebugInfo(module_.get());
  func_handle_map_.clear();
  export_system_symbols_.clear();

  // Runtime types.
  t_tvm_shape_index_ =
      llvm::Type::getIntNTy(*llvm_target_->GetContext(), DataType::ShapeIndex().bits());
  // Defined in 3rdparty/dlpack/include/dlpack/dlpack.h:
  // typedef struct { DLDeviceType device_type; int device_id; } DLDevice;
  t_tvm_device_ = llvm::StructType::create({t_int_, t_int_});
  // Defined in 3rdparty/dlpack/include/dlpack/dlpack.h:
  // typedef struct { uint8_t code; uint8_t bits; uint16_t lanes; } DLDataType;
  t_tvm_type_ = llvm::StructType::create({t_int8_, t_int8_, t_int16_});
  // Defined in include/tvm/runtime/base.h:
  // typedef void* TVMFunctionHandle;
  t_tvm_func_handle_ = t_void_p_;
  // Defined in 3rdparty/dlpack/include/dlpack/dlpack.h:
  // typedef struct { ... } DLTensor;
  t_tvm_array_ = llvm::StructType::create({t_void_p_, t_tvm_device_, t_int_, t_tvm_type_,
                                           llvmGetPointerTo(t_tvm_shape_index_, 0),
                                           llvmGetPointerTo(t_tvm_shape_index_, 0), t_int64_});
  // Defined in include/tvm/ffi/c_api.h:
  t_tvm_ffi_any_ = llvm::StructType::create({t_int32_, t_int32_, t_float64_});
  // Defined in include/tvm/runtime/c_backend_api.h:
  // typedef struct { void* sync_handle; int32_t num_task; } TVMParallelGroupEnv;
  t_tvm_parallel_group_env_ = llvm::StructType::create({llvmGetPointerTo(t_int32_, 0), t_int32_});
  // Defined in include/tvm/ffi/c_api.h:
  // typedef int (*)(void* self, const TVMFFIAny* args, int32_t num_args,
  //                 TVMFFIAny* result);
  ftype_tvm_ffi_c_func_ = llvm::FunctionType::get(
      t_int_,
      {t_void_p_, llvmGetPointerTo(t_tvm_ffi_any_, 0), t_int_, llvmGetPointerTo(t_tvm_ffi_any_, 0)},
      false);
  // Defined in include/tvm/runtime/c_backend_api.h:
  // typedef int (*FTVMParallelLambda)(int task_id, TVMParallelGroupEnv* penv, void* cdata);
  ftype_tvm_parallel_lambda_ = llvm::FunctionType::get(
      t_int_, {t_int_, llvmGetPointerTo(t_tvm_parallel_group_env_, 0), t_void_p_}, false);
  md_tbaa_ctx_ptr_ = md_builder_->createTBAAScalarTypeNode("ctx_ptr", md_tbaa_root_);

  // Runtime functions.
  // Defined in include/tvm/ffi/c_api.h:
  // int TVMFFIFunctionCall(TVMFunctionHandle func, TVMFFIAny* args, int32_t num_args,
  //                    TVMFFIAny* result);
  ftype_tvm_ffi_func_call_ = ftype_tvm_ffi_c_func_;
  // Defined in include/tvm/ffi/c_api.h:
  // void TVMFFIErrorSetRaisedFromCStr(const char *kind, const char* msg);
  ftype_tvm_ffi_error_set_raised_by_c_str_ = llvm::FunctionType::get(
      t_void_, {llvmGetPointerTo(t_char_, 0), llvmGetPointerTo(t_char_, 0)}, false);
  // Defined in include/tvm/runtime/c_backend_api.h:
  // int TVMBackendGetFuncFromEnv(void* mod_node, const char* func_name, TVMFunctionHandle* out);
  ftype_tvm_get_func_from_env_ = llvm::FunctionType::get(
      t_int_, {t_void_p_, llvmGetPointerTo(t_char_, 0), llvmGetPointerTo(t_tvm_func_handle_, 0)},
      false);
  // Defined in include/tvm/runtime/c_backend_api.h:
  // int TVMBackendParallelLaunch(FTVMParallelLambda flambda, void* cdata, int num_task);
  ftype_tvm_parallel_launch_ = llvm::FunctionType::get(
      t_int_, {llvmGetPointerTo(ftype_tvm_parallel_lambda_, 0), t_void_p_, t_int_}, false);
  // Defined in include/tvm/runtime/c_backend_api.h:
  // int TVMBackendParallelBarrier(int task_id, TVMParallelGroupEnv* penv);
  ftype_tvm_parallel_barrier_ = llvm::FunctionType::get(
      t_int_, {t_int_, llvmGetPointerTo(t_tvm_parallel_group_env_, 0)}, false);
  ftype_tvm_static_init_callback_ = llvm::FunctionType::get(t_int_, {t_void_p_}, false);
  ftype_tvm_static_init_ = llvm::FunctionType::get(
      t_int_,
      {llvmGetPointerTo(t_void_p_, 0), llvmGetPointerTo(ftype_tvm_static_init_callback_, 0),
       t_void_p_, t_int_},
      false);
  // initialize TVM runtime API
  if (system_lib_prefix_.has_value() && !target_c_runtime) {
    // We will need this in environment for backward registration.
    // Defined in include/tvm/runtime/c_backend_api.h:
    // int TVMBackendRegisterSystemLibSymbol(const char* name, void* ptr);
    f_tvm_register_system_symbol_ = llvm::Function::Create(
        llvm::FunctionType::get(t_int_, {llvmGetPointerTo(t_char_, 0), t_void_p_}, false),
        llvm::Function::ExternalLinkage, "TVMBackendRegisterSystemLibSymbol", module_.get());
  } else {
    f_tvm_register_system_symbol_ = nullptr;
  }
  if (dynamic_lookup || system_lib_prefix_.has_value()) {
    f_tvm_ffi_func_call_ =
        llvm::Function::Create(ftype_tvm_ffi_func_call_, llvm::Function::ExternalLinkage,
                               "TVMFFIFunctionCall", module_.get());
    f_tvm_ffi_set_raised_by_c_str_ = llvm::Function::Create(
        ftype_tvm_ffi_error_set_raised_by_c_str_, llvm::Function::ExternalLinkage,
        "TVMFFIErrorSetRaisedFromCStr", module_.get());
    f_tvm_get_func_from_env_ =
        llvm::Function::Create(ftype_tvm_get_func_from_env_, llvm::Function::ExternalLinkage,
                               "TVMBackendGetFuncFromEnv", module_.get());
    f_tvm_parallel_launch_ =
        llvm::Function::Create(ftype_tvm_parallel_launch_, llvm::Function::ExternalLinkage,
                               "TVMBackendParallelLaunch", module_.get());
    f_tvm_parallel_barrier_ =
        llvm::Function::Create(ftype_tvm_parallel_barrier_, llvm::Function::ExternalLinkage,
                               "TVMBackendParallelBarrier", module_.get());
  }
  target_c_runtime_ = target_c_runtime;
  InitGlobalContext(dynamic_lookup);
}

llvm::DISubprogram* CodeGenCPU::CreateDebugFunction(llvm::StringRef name,
                                                    const Array<Type>& param_types,
                                                    const Type& return_type) {
#if TVM_LLVM_VERSION < 50
  return nullptr;
#else

  llvm::SmallVector<llvm::Metadata*, 4> paramTys;

  paramTys.push_back(GetDebugType(return_type));
  for (const auto& param_type : param_types) {
    paramTys.push_back(GetDebugType(param_type));
  }

  auto* DIFunctionTy = dbg_info_->di_builder_->createSubroutineType(
      dbg_info_->di_builder_->getOrCreateTypeArray(paramTys));

  bool local_to_unit = llvm::GlobalVariable::isLocalLinkage(llvm::GlobalValue::InternalLinkage);

#if TVM_LLVM_VERSION >= 80
  auto SPFlags = llvm::DISubprogram::toSPFlags(local_to_unit, /*IsDefinition=*/true,
                                               /*IsOptimized=*/true);
#else
  bool SPFlags = /*IsOptimized=*/true;
#endif

  auto* DIFunction = dbg_info_->di_builder_->createFunction(
      /*Scope=*/dbg_info_->file_, /*Name=*/name, /*LinkageName=*/"",
      /*File=*/dbg_info_->file_, /*LineNo=*/0, /*Ty=*/DIFunctionTy,
      /*ScopeLine=*/0, /*Flags=*/llvm::DINode::FlagPrototyped, /*SPFlags=*/SPFlags);

  return DIFunction;

#endif
}

llvm::DISubprogram* CodeGenCPU::CreateDebugFunction(const GlobalVar& gvar, const PrimFunc& func) {
  std::string name = func->GetAttr<String>(tvm::attr::kGlobalSymbol).value_or(gvar->name_hint);
  return CreateDebugFunction(name, func->params.Map(GetType), func->ret_type);
}

void CodeGenCPU::AddFunction(const GlobalVar& gvar, const PrimFunc& func) {
  di_subprogram_ = CreateDebugFunction(gvar, func);
  EmitDebugLocation(func->span);
  CodeGenLLVM::AddFunction(gvar, func);
  if (f_tvm_register_system_symbol_ != nullptr) {
    if (auto global_symbol = func->GetAttr<String>(tvm::attr::kGlobalSymbol)) {
      export_system_symbols_.emplace_back(
          std::make_pair(global_symbol.value().operator std::string(), function_));
    }
  }
  AddDebugInformation(function_, func->params.Map(GetType));
}

void CodeGenCPU::AddMainFunction(const std::string& entry_func_name) {
  llvm::Function* f = module_->getFunction(entry_func_name);
  ICHECK(f) << "Function " << entry_func_name << "does not in module";
  llvm::Type* type = llvm::ArrayType::get(t_char_, entry_func_name.length() + 1);
  llvm::GlobalVariable* global =
      new llvm::GlobalVariable(*module_, type, true, llvm::GlobalValue::WeakAnyLinkage, nullptr,
                               runtime::symbol::tvm_module_main);
#if TVM_LLVM_VERSION >= 100
  global->setAlignment(llvm::Align(1));
#else
  global->setAlignment(1);
#endif
  // comdat is needed for windows select any linking to work
  // set comdat to Any(weak linking)
  if (llvm_target_->GetOrCreateTargetMachine()->getTargetTriple().isOSWindows()) {
    llvm::Comdat* comdat = module_->getOrInsertComdat(runtime::symbol::tvm_module_main);
    comdat->setSelectionKind(llvm::Comdat::Any);
    global->setComdat(comdat);
  }

  global->setInitializer(
      llvm::ConstantDataArray::getString(*llvm_target_->GetContext(), entry_func_name));
  global->setDLLStorageClass(llvm::GlobalVariable::DLLExportStorageClass);
}

std::unique_ptr<llvm::Module> CodeGenCPU::Finish() {
  // link modules
  if (dbg_info_ != nullptr) {
    dbg_info_->di_builder_->finalize();
  }
  return CodeGenLLVM::Finish();
}

CodeGenLLVM::TypedPointer CodeGenCPU::CreateStructRefPtr(DataType t, llvm::Value* buf,
                                                         llvm::Value* index, int kind) {
  if (kind < builtin::kArrKindBound_) {
    if (buf->getType() == t_void_p_) {
      buf = builder_->CreatePointerCast(buf, llvmGetPointerTo(t_tvm_array_, 0));
    } else {
      ICHECK_EQ(buf->getType(), llvmGetPointerTo(t_tvm_array_, 0));
    }
  }
  switch (kind) {
    case builtin::kArrAddr: {
      return TypedPointer(t_tvm_array_, builder_->CreateInBoundsGEP(t_tvm_array_, buf, index));
    }
    case builtin::kArrData: {
      llvm::Type* member_type = t_tvm_array_->getStructElementType(0);
      llvm::Value* member_addr =
          builder_->CreateInBoundsGEP(t_tvm_array_, buf, {index, ConstInt32(0)});
      return TypedPointer(member_type, member_addr);
    }
    case builtin::kArrShape: {
      llvm::Type* member_type = t_tvm_array_->getStructElementType(4);
      llvm::Value* member_addr =
          builder_->CreateInBoundsGEP(t_tvm_array_, buf, {index, ConstInt32(4)});
      return TypedPointer(member_type, member_addr);
    }
    case builtin::kArrStrides: {
      llvm::Type* member_type = t_tvm_array_->getStructElementType(5);
      llvm::Value* member_addr =
          builder_->CreateInBoundsGEP(t_tvm_array_, buf, {index, ConstInt32(5)});
      return TypedPointer(member_type, member_addr);
    }
    case builtin::kArrNDim: {
      llvm::Type* member_type = t_tvm_array_->getStructElementType(2);
      llvm::Value* member_addr =
          builder_->CreateInBoundsGEP(t_tvm_array_, buf, {index, ConstInt32(2)});
      return TypedPointer(member_type, member_addr);
    }
    case builtin::kArrTypeCode: {
      llvm::Type* member_type = t_tvm_array_->getStructElementType(3)->getStructElementType(0);
      llvm::Value* member_addr =
          builder_->CreateInBoundsGEP(t_tvm_array_, buf, {index, ConstInt32(3), ConstInt32(0)});
      return TypedPointer(member_type, member_addr);
    }
    case builtin::kArrTypeBits: {
      llvm::Type* member_type = t_tvm_array_->getStructElementType(3)->getStructElementType(1);
      llvm::Value* member_addr =
          builder_->CreateInBoundsGEP(t_tvm_array_, buf, {index, ConstInt32(3), ConstInt32(1)});
      return TypedPointer(member_type, member_addr);
    }
    case builtin::kArrTypeLanes: {
      llvm::Type* member_type = t_tvm_array_->getStructElementType(3)->getStructElementType(2);
      llvm::Value* member_addr =
          builder_->CreateInBoundsGEP(t_tvm_array_, buf, {index, ConstInt32(3), ConstInt32(2)});
      return TypedPointer(member_type, member_addr);
    }
    case builtin::kArrByteOffset: {
      llvm::Type* member_type = t_tvm_array_->getStructElementType(6);
      llvm::Value* member_addr =
          builder_->CreateInBoundsGEP(t_tvm_array_, buf, {index, ConstInt32(6)});
      return TypedPointer(member_type, member_addr);
    }
    case builtin::kArrDeviceId: {
      llvm::Type* member_type = t_tvm_array_->getStructElementType(1)->getStructElementType(1);
      llvm::Value* member_addr =
          builder_->CreateInBoundsGEP(t_tvm_array_, buf, {index, ConstInt32(1), ConstInt32(1)});
      return TypedPointer(member_type, member_addr);
    }
    case builtin::kArrDeviceType: {
      llvm::Type* member_type = t_tvm_array_->getStructElementType(1)->getStructElementType(0);
      llvm::Value* member_addr =
          builder_->CreateInBoundsGEP(t_tvm_array_, buf, {index, ConstInt32(1), ConstInt32(0)});
      return TypedPointer(member_type, member_addr);
    }
    case builtin::kTVMFFIAnyTypeIndex: {
      buf = builder_->CreatePointerCast(buf, llvmGetPointerTo(t_tvm_ffi_any_, 0));
      buf = builder_->CreateInBoundsGEP(t_tvm_ffi_any_, buf, {index, ConstInt32(0)});
      return TypedPointer(t_int32_, buf);
    }
    case builtin::kTVMFFIAnyUnionValue: {
      ICHECK_EQ(t.lanes(), 1);
      buf = builder_->CreatePointerCast(buf, llvmGetPointerTo(t_tvm_ffi_any_, 0));
      // field 2 is the union value
      buf = builder_->CreateInBoundsGEP(t_tvm_ffi_any_, buf, {index, ConstInt32(2)});
      if (t.is_bool()) {
        // it should be safe to set the pointer to the first byte of the union value
        buf = builder_->CreatePointerCast(buf, llvmGetPointerTo(DTypeToLLVMType(t), 0));
        return TypedPointer(t_int8_, buf);
      } else if (t.is_int() && t.bits() == 64) {
        buf = builder_->CreatePointerCast(buf, llvmGetPointerTo(t_int64_, 0));
        return TypedPointer(t_int64_, buf);
      } else if (t.is_float() && t.bits() == 64) {
        buf = builder_->CreatePointerCast(buf, llvmGetPointerTo(t_float64_, 0));
        return TypedPointer(t_float64_, buf);
      } else if (t.is_handle()) {
        buf = builder_->CreatePointerCast(buf, llvmGetPointerTo(t_void_p_, 0));
        return TypedPointer(t_void_p_, buf);
      } else {
        LOG(DEBUG) << "DataType " << t << " cannot be stored into a TVMFFIAny's value field";
      }
    }
    default:
      LOG(FATAL) << "unknown field code";
  }
}

llvm::Value* CodeGenCPU::CreateCallExtern(Type ret_type, String global_symbol,
                                          const Array<PrimExpr>& args, bool skip_first_arg) {
  std::vector<llvm::Value*> arg_values;
  for (size_t i = static_cast<size_t>(skip_first_arg); i < args.size(); ++i) {
    arg_values.push_back(MakeValue(args[i]));
  }
  std::vector<llvm::Type*> arg_types;
  for (llvm::Value* v : arg_values) {
    arg_types.push_back(v->getType());
  }
  llvm::FunctionType* ftype = llvm::FunctionType::get(GetLLVMType(ret_type), arg_types, false);
  // Check if it is available in global function table as injected function.

  auto callee = [&]() -> llvm::Value* {
    if (auto it = gv_func_map_.find(global_symbol); it != gv_func_map_.end()) {
      if (it->second == nullptr) {
        it->second = InitContextPtr(llvmGetPointerTo(ftype, 0), "__" + global_symbol);
      }
      return GetContextPtr(it->second);
    } else if (llvm::Function* f = module_->getFunction(MakeStringRef(global_symbol))) {
      return f;
    } else {
      return llvm::Function::Create(ftype, llvm::Function::ExternalLinkage,
                                    MakeStringRef(global_symbol), module_.get());
    }
  }();

  if (callee->getType() != llvmGetPointerTo(ftype, 0)) {
    callee = builder_->CreatePointerCast(callee, llvmGetPointerTo(ftype, 0));
  }
  return builder_->CreateCall(ftype, callee, arg_values);
}

llvm::GlobalVariable* CodeGenCPU::InitContextPtr(llvm::Type* p_type, std::string name) {
  llvm::GlobalVariable* gv = new llvm::GlobalVariable(
      *module_, p_type, false, llvm::GlobalValue::LinkOnceAnyLinkage, nullptr, name);
#if TVM_LLVM_VERSION >= 100
  gv->setAlignment(llvm::Align(data_layout_->getTypeAllocSize(p_type)));
#else
  gv->setAlignment(data_layout_->getTypeAllocSize(p_type));
#endif
  gv->setInitializer(llvm::Constant::getNullValue(p_type));
  gv->setDLLStorageClass(llvm::GlobalValue::DLLStorageClassTypes::DLLExportStorageClass);
  // comdat is needed for windows select any linking to work
  // set comdat to Any(weak linking)
  if (llvm_target_->GetOrCreateTargetMachine()->getTargetTriple().isOSWindows()) {
    llvm::Comdat* comdat = module_->getOrInsertComdat(name);
    comdat->setSelectionKind(llvm::Comdat::Any);
    gv->setComdat(comdat);
  }
  return gv;
}

llvm::Value* CodeGenCPU::GetContextPtr(llvm::GlobalVariable* gv) {
  ICHECK(gv != nullptr);
#if TVM_LLVM_VERSION >= 110
  llvm::LoadInst* faddr =
      builder_->CreateAlignedLoad(gv->getValueType(), gv, llvm::Align(gv->getAlignment()));
#elif TVM_LLVM_VERSION >= 80
  llvm::LoadInst* faddr = builder_->CreateAlignedLoad(gv->getValueType(), gv, gv->getAlignment());
#else
  llvm::LoadInst* faddr = builder_->CreateAlignedLoad(gv, gv->getAlignment());
#endif
  faddr->setMetadata("tbaa",
                     md_builder_->createTBAAStructTagNode(md_tbaa_ctx_ptr_, md_tbaa_ctx_ptr_, 0));
  return faddr;
}

void CodeGenCPU::InitGlobalContext(bool dynamic_lookup) {
  std::string ctx_symbol =
      system_lib_prefix_.value_or("") + tvm::runtime::symbol::tvm_ffi_library_ctx;
  // Module context
  gv_mod_ctx_ = InitContextPtr(t_void_p_, ctx_symbol);
  // Register back the locations.
  if (f_tvm_register_system_symbol_ != nullptr && !target_c_runtime_) {
    export_system_symbols_.emplace_back(std::make_pair(ctx_symbol, gv_mod_ctx_));
  } else {
    if (!dynamic_lookup) {
      gv_tvm_ffi_func_call_ =
          InitContextPtr(llvmGetPointerTo(ftype_tvm_ffi_func_call_, 0), "__TVMFFIFunctionCall");
      gv_tvm_get_func_from_env_ = InitContextPtr(llvmGetPointerTo(ftype_tvm_get_func_from_env_, 0),
                                                 "__TVMBackendGetFuncFromEnv");
      gv_tvm_ffi_set_last_error_c_str_ =
          InitContextPtr(llvmGetPointerTo(ftype_tvm_ffi_error_set_raised_by_c_str_, 0),
                         "__TVMFFIErrorSetRaisedFromCStr");
      gv_tvm_parallel_launch_ = InitContextPtr(llvmGetPointerTo(ftype_tvm_parallel_launch_, 0),
                                               "__TVMBackendParallelLaunch");
      gv_tvm_parallel_barrier_ = InitContextPtr(llvmGetPointerTo(ftype_tvm_parallel_barrier_, 0),
                                                "__TVMBackendParallelBarrier");
      // Mark as context functions
      gv_func_map_["TVMBackendAllocWorkspace"] = nullptr;
      gv_func_map_["TVMBackendFreeWorkspace"] = nullptr;
    }
  }
}

llvm::BasicBlock* CodeGenCPU::CheckCallSuccess(llvm::Value* retcode) {
  // create emit codes that checks and load the function.
  llvm::LLVMContext* ctx = llvm_target_->GetContext();
  auto* fail_block = llvm::BasicBlock::Create(*ctx, "call_fail", function_);
  auto* end_block = llvm::BasicBlock::Create(*ctx, "call_end", function_);
  auto* succ = builder_->CreateICmpEQ(retcode, llvm::ConstantInt::get(t_int_, 0));
  builder_->CreateCondBr(succ, end_block, fail_block, md_very_likely_branch_);
  builder_->SetInsertPoint(fail_block);
  // return the code.
  builder_->CreateRet(retcode);
  // otherwise set it to be new end.
  builder_->SetInsertPoint(end_block);
  return end_block;
}

void CodeGenCPU::CreateComputeScope(const AttrStmtNode* op) {
  EmitDebugLocation(op);
  /*! \brief maintain states that should be guarded when step into compute scope */
  struct ComputeScopeStates {
    explicit ComputeScopeStates(CodeGenCPU* parent) : parent_(parent) {}

    void EnterWithScope() {
      std::swap(function_, parent_->function_);
      std::swap(analyzer_, parent_->analyzer_);
      std::swap(var_map_, parent_->var_map_);
      std::swap(di_subprogram_, parent_->di_subprogram_);
    }

    void ExitWithScope() {
      std::swap(function_, parent_->function_);
      std::swap(analyzer_, parent_->analyzer_);
      std::swap(var_map_, parent_->var_map_);
      std::swap(di_subprogram_, parent_->di_subprogram_);
    }

    llvm::Function* function_{nullptr};
    llvm::DISubprogram* di_subprogram_{nullptr};
    std::unordered_map<const VarNode*, llvm::Value*> var_map_;
    std::unique_ptr<arith::Analyzer> analyzer_{std::make_unique<arith::Analyzer>()};
    CodeGenCPU* parent_;
  };

  // There are two reasons why we create another function for compute_scope
  // - Make sure the generated compute function is clearly separately(though it can get inlined)
  // - Set noalias on all the pointer arguments, some of them are loaded from ffi::PackedArgs.
  //   This is easier than set the alias scope manually.
  Array<Var> vargs = tir::UndefinedVars(op->body, {});
  std::vector<llvm::Value*> arg_values;
  std::vector<llvm::Type*> arg_types;
  for (Var v : vargs) {
    llvm::Value* value = MakeValue(v);
    value->setName(v->name_hint.c_str());
    arg_values.push_back(value);
    arg_types.push_back(value->getType());
  }
  llvm::FunctionType* ftype = llvm::FunctionType::get(t_int_, arg_types, false);
  // $xxx_compute_ functions are not global. They should be marked as static (via InternalLinkage)
  // to call them correctly on MIPS platform (CALL16 issue)
  // Linkage ld Error: CALL16 reloc at 0x290 not against global symbol
  const StringImmNode* value = op->value.as<StringImmNode>();
  ICHECK(value != nullptr);
  llvm::Function* fcompute = llvm::Function::Create(ftype, llvm::Function::InternalLinkage,
                                                    MakeStringRef(value->value), module_.get());
  SetTargetAttributes(fcompute);
  for (auto it = fcompute->arg_begin(); it != fcompute->arg_end(); it++) {
    const Var& var = vargs[std::distance(fcompute->arg_begin(), it)];
    it->setName(std::string(var->name_hint));
  }

  llvm::BasicBlock* compute_call_end = CheckCallSuccess(builder_->CreateCall(fcompute, arg_values));
  llvm::LLVMContext* ctx = llvm_target_->GetContext();
  // enter compute scope and setup compute function.
  With<ComputeScopeStates> scope_states_guard(this);
  size_t idx = 0;
  for (auto it = fcompute->arg_begin(); it != fcompute->arg_end(); ++it, ++idx) {
    llvm::Argument* v = &(*it);
    const Var& var = vargs[idx];
    var_map_[var.get()] = v;
    if (var.dtype().is_handle() && !alias_var_set_.count(var.get())) {
      // set non alias.
#if TVM_LLVM_VERSION >= 50
      fcompute->addParamAttr(idx, llvm::Attribute::NoAlias);
      // always not inline compute function to make the code structure clean
#else
      fcompute->setDoesNotAlias(idx + 1);
#endif
      fcompute->addFnAttr(llvm::Attribute::NoInline);
    }
    // Add alignment attribute if needed.
#if TVM_LLVM_VERSION >= 50
    auto f = alloc_storage_info_.find(var.get());
    if (f != alloc_storage_info_.end()) {
      unsigned align = f->second.alignment;
      if (align > 1) {
        auto attr = llvm::Attribute::get(*ctx, llvm::Attribute::Alignment, align);
        fcompute->addParamAttr(idx, attr);
      }
    }
#endif
  }

  function_ = fcompute;
  di_subprogram_ = CreateDebugFunction(MakeStringRef(value->value), vargs.Map(GetType),
                                       PrimType(DataType::Int(32)));
  auto* compute_entry = llvm::BasicBlock::Create(*ctx, "entry", function_);
  builder_->SetInsertPoint(compute_entry);
  this->VisitStmt(op->body);
  builder_->CreateRet(ConstInt32(0));
  builder_->SetInsertPoint(compute_call_end);

  AddDebugInformation(fcompute, vargs.Map(GetType));
}

CodeGenLLVM::TypedPointer CodeGenCPU::PackClosureData(const Array<Var>& vfields,
                                                      uint64_t* num_bytes,
                                                      std::string struct_name) {
  if (vfields.size() == 0) {
    *num_bytes = 0U;
    return TypedPointer(t_void_p_, llvm::Constant::getNullValue(t_void_p_));
  }
  std::vector<llvm::Type*> fields;
  for (Var v : vfields) {
    auto it = var_map_.find(v.get());
    ICHECK(it != var_map_.end());
    fields.push_back(it->second->getType());
  }
  llvm::StructType* ctype = struct_name.size() ? llvm::StructType::create(fields, struct_name)
                                               : llvm::StructType::create(fields);
  llvm::AllocaInst* cvalue =
      WithFunctionEntry([&]() { return builder_->CreateAlloca(ctype, ConstInt32(1)); });
  llvm::Value* zero = ConstInt32(0);
  for (size_t i = 0; i < vfields.size(); ++i) {
    builder_->CreateStore(var_map_.at(vfields[i].get()),
                          builder_->CreateInBoundsGEP(ctype, cvalue, {zero, ConstInt32(i)}));
  }
  *num_bytes = data_layout_->getTypeAllocSize(ctype);
  return TypedPointer(ctype, cvalue);
}

void CodeGenCPU::UnpackClosureData(TypedPointer cdata, const Array<Var>& vfields,
                                   std::unordered_map<const VarNode*, llvm::Value*>* vmap) {
  for (size_t i = 0; i < vfields.size(); ++i) {
    llvm::Type* field_type = cdata.type->getStructElementType(i);
    llvm::Value* field_addr =
        builder_->CreateInBoundsGEP(cdata.type, cdata.addr, {ConstInt32(0), ConstInt32(i)});
    llvm::Value* load =
        builder_->CreateLoad(field_type, field_addr, std::string(vfields[i]->name_hint));
    (*vmap)[vfields[i].get()] = load;
  }
}

void CodeGenCPU::CreateParallelLaunch(const Stmt& body, int num_task, std::string name) {
  // closure data
  llvm::Function* f =
      llvm::Function::Create(ftype_tvm_parallel_lambda_, llvm::Function::PrivateLinkage,
                             "__tvm_parallel_lambda", module_.get());
  SetTargetAttributes(f);

  // allocate and setup the closure, call the closure.
  Array<Var> vfields = tir::UndefinedVars(body, {});
  uint64_t nbytes;
  TypedPointer cdata = PackClosureData(vfields, &nbytes, "closure_" + name);
#if TVM_LLVM_VERSION >= 90
  auto launch_callee = llvm::FunctionCallee(ftype_tvm_parallel_launch_, RuntimeTVMParallelLaunch());
#else
  auto launch_callee = RuntimeTVMParallelLaunch();
#endif
  llvm::BasicBlock* par_launch_end = CheckCallSuccess(builder_->CreateCall(
      launch_callee,
      {f, builder_->CreatePointerCast(cdata.addr, t_void_p_), ConstInt32(num_task)}));
  // Setup the closure function.
  auto* lambda_entry =
      llvm::BasicBlock::Create(*llvm_target_->GetContext(), "parallel_closure_entry", f);
  builder_->SetInsertPoint(lambda_entry);
  auto it = f->arg_begin();
  llvm::Value* task_id = &(*it++);
  task_id->setName("task_id");
  llvm::Value* penv = &(*it++);
  cdata.addr = builder_->CreatePointerCast(&(*it++), cdata.addr->getType());
  // setup new variable map, swap it with current var context.
  std::unordered_map<const VarNode*, llvm::Value*> new_vmap;
  UnpackClosureData(cdata, vfields, &new_vmap);
  // setup parallel env
  ParallelEnv par_env;
  par_env.task_id = Var("task_id", DataType::Int(32));
  par_env.num_task = Var("num_task", DataType::Int(32));
  new_vmap[par_env.task_id.get()] = task_id;
  new_vmap[par_env.num_task.get()] = builder_->CreateLoad(
      t_int32_,
      builder_->CreateInBoundsGEP(t_tvm_parallel_group_env_, penv, {ConstInt32(0), ConstInt32(1)}),
      "num_task");
  par_env.penv = penv;
  auto new_analyzer = std::make_unique<arith::Analyzer>();
  std::swap(function_, f);
  std::swap(parallel_env_, par_env);
  std::swap(analyzer_, new_analyzer);
  std::swap(var_map_, new_vmap);
  this->VisitStmt(body);
  builder_->CreateRet(ConstInt32(0));
  // swap the var map back, now we are back on track.
  std::swap(var_map_, new_vmap);
  std::swap(analyzer_, new_analyzer);
  std::swap(parallel_env_, par_env);
  std::swap(function_, f);
  ICHECK_NE(par_env.parallel_loop_count, 0) << "Cannot find parallel loop within parallel launch";
  builder_->SetInsertPoint(par_launch_end);
}

llvm::Value* CodeGenCPU::CreateStaticHandle() {
  llvm::GlobalVariable* gv =
      new llvm::GlobalVariable(*module_, t_void_p_, false, llvm::GlobalValue::PrivateLinkage,
                               nullptr, "__tvm_static_handle");
#if TVM_LLVM_VERSION >= 100
  gv->setAlignment(llvm::Align(data_layout_->getTypeAllocSize(t_void_p_)));
#else
  gv->setAlignment(data_layout_->getTypeAllocSize(t_void_p_));
#endif
  gv->setInitializer(llvm::Constant::getNullValue(t_void_p_));
  return gv;
}

void CodeGenCPU::CreateStaticInit(const std::string& init_fname, const Stmt& body) {
  // closure data
  llvm::Function* f =
      llvm::Function::Create(ftype_tvm_static_init_callback_, llvm::Function::PrivateLinkage,
                             "__tvm_static_init_lambda", module_.get());
  SetTargetAttributes(f);
  llvm::Value* gv = CreateStaticHandle();
  llvm::Function* finit = module_->getFunction(init_fname);
  if (finit == nullptr) {
    finit = llvm::Function::Create(ftype_tvm_static_init_, llvm::Function::ExternalLinkage,
                                   init_fname, module_.get());
  }
  // allocate and setup the closure, call the closure.
  uint64_t nbytes;
  Array<Var> vfields = tir::UndefinedVars(body, {});
  TypedPointer cdata = PackClosureData(vfields, &nbytes);
  llvm::BasicBlock* init_end = CheckCallSuccess(builder_->CreateCall(
      finit, {gv, f, builder_->CreatePointerCast(cdata.addr, t_void_p_), ConstInt32(nbytes)}));
  // Setup the closure function.
  auto* lambda_entry = llvm::BasicBlock::Create(*llvm_target_->GetContext(), "entry", f);
  builder_->SetInsertPoint(lambda_entry);
  auto it = f->arg_begin();
  cdata.addr = builder_->CreatePointerCast(&(*it++), cdata.addr->getType());
  // setup new variable map, swap it with current var context.
  std::unordered_map<const VarNode*, llvm::Value*> new_vmap;
  UnpackClosureData(cdata, vfields, &new_vmap);
  ICHECK(parallel_env_.penv == nullptr);
  auto new_analyzer = std::make_unique<arith::Analyzer>();
  std::swap(function_, f);
  std::swap(analyzer_, new_analyzer);
  std::swap(var_map_, new_vmap);
  this->VisitStmt(body);
  builder_->CreateRet(ConstInt32(0));
  // swap the var map back, now we are back on track.
  std::swap(var_map_, new_vmap);
  std::swap(analyzer_, new_analyzer);
  std::swap(function_, f);
  builder_->SetInsertPoint(init_end);
}

llvm::Value* CodeGenCPU::GetPackedFuncHandle(const std::string& fname) {
  // We will store the packed function handle in global space.
  // Initialize it during the first call.
#if TVM_LLVM_VERSION >= 200
  llvm::DataLayout layout(module_.get()->getDataLayout());
#else
  llvm::DataLayout layout(module_.get());
#endif
  uint64_t align = layout.getTypeAllocSize(t_tvm_func_handle_);
  auto it = func_handle_map_.find(fname);

  llvm::GlobalVariable* hptr;
  if (it == func_handle_map_.end()) {
    // create global location for the handle
    // create the function handle
    hptr =
        new llvm::GlobalVariable(*module_, t_tvm_func_handle_, false,
                                 llvm::GlobalValue::InternalLinkage, nullptr, ".tvm_func." + fname);
#if TVM_LLVM_VERSION >= 100
    hptr->setAlignment(llvm::Align(align));
#else
    hptr->setAlignment(align);
#endif
    hptr->setInitializer(llvm::Constant::getNullValue(t_tvm_func_handle_));
    func_handle_map_[fname] = hptr;
  } else {
    hptr = it->second;
  }
  // create emit codes that checks and load the function.
  llvm::LLVMContext* ctx = llvm_target_->GetContext();
  llvm::BasicBlock* pre_block = builder_->GetInsertBlock();
  auto* init_block = llvm::BasicBlock::Create(*ctx, "handle_init", function_);
  auto* end_block = llvm::BasicBlock::Create(*ctx, "handle_init_end", function_);
#if TVM_LLVM_VERSION >= 110
  llvm::Value* handle = builder_->CreateAlignedLoad(hptr->getValueType(), hptr, llvm::Align(align));
#elif TVM_LLVM_VERSION >= 80
  llvm::Value* handle = builder_->CreateAlignedLoad(hptr->getValueType(), hptr, align);
#else
  llvm::Value* handle = builder_->CreateAlignedLoad(hptr, align);
#endif
  llvm::Value* handle_not_null =
      builder_->CreateICmpNE(handle, llvm::Constant::getNullValue(t_tvm_func_handle_));
  builder_->CreateCondBr(handle_not_null, end_block, init_block, md_very_likely_branch_);
  // Initialize the handle if needed.
  builder_->SetInsertPoint(init_block);
  llvm::Value* out =
      WithFunctionEntry([&]() { return builder_->CreateAlloca(t_tvm_func_handle_); });
#if TVM_LLVM_VERSION >= 110
  llvm::LoadInst* ctx_load = builder_->CreateAlignedLoad(gv_mod_ctx_->getValueType(), gv_mod_ctx_,
                                                         llvm::Align(gv_mod_ctx_->getAlignment()));
#elif TVM_LLVM_VERSION >= 80
  llvm::LoadInst* ctx_load = builder_->CreateAlignedLoad(gv_mod_ctx_->getValueType(), gv_mod_ctx_,
                                                         gv_mod_ctx_->getAlignment());
#else
  llvm::LoadInst* ctx_load = builder_->CreateAlignedLoad(gv_mod_ctx_, gv_mod_ctx_->getAlignment());
#endif
  ctx_load->setMetadata(
      "tbaa", md_builder_->createTBAAStructTagNode(md_tbaa_ctx_ptr_, md_tbaa_ctx_ptr_, 0));
#if TVM_LLVM_VERSION >= 90
  auto env_callee = llvm::FunctionCallee(ftype_tvm_get_func_from_env_, RuntimeTVMGetFuncFromEnv());
#else
  auto env_callee = RuntimeTVMGetFuncFromEnv();
#endif
  llvm::Value* retcode = builder_->CreateCall(env_callee, {ctx_load, GetConstString(fname), out});
  init_block = CheckCallSuccess(retcode);
#if TVM_LLVM_VERSION >= 110
  llvm::Value* loaded_handle =
      builder_->CreateAlignedLoad(t_tvm_func_handle_, out, llvm::Align(align));
#elif TVM_LLVM_VERSION >= 80
  llvm::Value* loaded_handle = builder_->CreateAlignedLoad(t_tvm_func_handle_, out, align);
#else
  llvm::Value* loaded_handle = builder_->CreateAlignedLoad(out, align);
#endif
  // Store the handle
  builder_->CreateStore(loaded_handle, hptr);
  builder_->CreateBr(end_block);
  // end block
  builder_->SetInsertPoint(end_block);
  llvm::PHINode* phi = builder_->CreatePHI(t_tvm_func_handle_, 2);
  phi->addIncoming(handle, pre_block);
  phi->addIncoming(loaded_handle, init_block);
  return phi;
}

CodeGenCPU::PackedCall CodeGenCPU::MakeCallPackedLowered(const Array<PrimExpr>& args,
                                                         const DataType& r_type,
                                                         const int64_t begin, const int64_t end,
                                                         bool use_env_lookup) {
  std::string func_name = [&]() {
    auto ptr = args[0].as<StringImmNode>();
    ICHECK(ptr) << "Expected first argument of tir::Call to be "
                << "a string containing the callee's name, "
                << "but instead contained " << args[0];
    return ptr->value;
  }();
  // call the function
  int64_t nargs = end - begin;
  ICHECK_GE(nargs, 0);
  llvm::Value* stack_args = MakeValue(args[1]);
  llvm::Value* packed_args = builder_->CreateInBoundsGEP(
      t_tvm_ffi_any_, builder_->CreatePointerCast(stack_args, llvmGetPointerTo(t_tvm_ffi_any_, 0)),
      ConstInt32(begin));
  llvm::Value* result = builder_->CreateInBoundsGEP(
      t_tvm_ffi_any_, builder_->CreatePointerCast(stack_args, llvmGetPointerTo(t_tvm_ffi_any_, 0)),
      ConstInt32(end));

  llvm::FunctionType* callee_ftype = nullptr;
  llvm::Value* callee_value = nullptr;
  std::vector<llvm::Value*> call_args;

  if (use_env_lookup) {
    callee_ftype = ftype_tvm_ffi_func_call_;
    callee_value = RuntimeTVMFFIFunctionCall();
    call_args.push_back(GetPackedFuncHandle(func_name));
    call_args.insert(call_args.end(), {packed_args, ConstInt32(nargs), result});
  } else {
    callee_ftype = ftype_tvm_ffi_c_func_;
    callee_value = module_->getFunction(func_name);
    if (callee_value == nullptr) {
      callee_value = llvm::Function::Create(ftype_tvm_ffi_c_func_, llvm::Function::ExternalLinkage,
                                            func_name, module_.get());
    }
    call_args.push_back(llvm::ConstantPointerNull::get(t_void_p_));
    call_args.insert(call_args.end(), {packed_args, ConstInt32(nargs), result});
  }
#if TVM_LLVM_VERSION >= 90
  auto call_callee = llvm::FunctionCallee(callee_ftype, callee_value);
#else
  (void)callee_ftype;  // use callee_ftype to avoid unused variable warning when using older LLVM.
  auto call_callee = callee_value;
#endif
  llvm::Value* call = builder_->CreateCall(call_callee, call_args);

  llvm::BasicBlock* end_block = CheckCallSuccess(call);

  PackedCall pc = {nullptr};

  if (!r_type.is_void()) {
    // Load the return value and cast it to the designated type (r_type).
    DataType r_api_type = tir::APIType(r_type);
    llvm::Type* llvm_r_api_type = DTypeToLLVMType(r_api_type);
    llvm::Value* result_value =
        builder_->CreateInBoundsGEP(t_tvm_ffi_any_, result, {ConstInt32(0), ConstInt32(2)});
    llvm::Value* load_ptr =
        builder_->CreatePointerCast(result_value, llvmGetPointerTo(llvm_r_api_type, 0));
#if TVM_LLVM_VERSION >= 110
    llvm::Value* rvalue = builder_->CreateAlignedLoad(llvm_r_api_type, load_ptr, llvm::Align(8));
#elif TVM_LLVM_VERSION >= 80
    llvm::Value* rvalue = builder_->CreateAlignedLoad(llvm_r_api_type, load_ptr, 8);
#else
    llvm::Value* rvalue = builder_->CreateAlignedLoad(load_ptr, 8);
#endif

    pc.ret_value = CreateCast(r_api_type, r_type, rvalue);
    llvm::Value* result_type_index =
        builder_->CreateInBoundsGEP(t_tvm_ffi_any_, result, {ConstInt32(0), ConstInt32(0)});

    // Load the return type code.
#if TVM_LLVM_VERSION >= 110
    pc.ret_type_index = builder_->CreateAlignedLoad(t_int32_, result_type_index, llvm::Align(4));
#elif TVM_LLVM_VERSION >= 80
    pc.ret_type_index = builder_->CreateAlignedLoad(t_int32_, result_type_index, 8);
#else
    pc.ret_type_index = builder_->CreateAlignedLoad(result_type_index, 8);
#endif
  }

  pc.end_block = end_block;
  return pc;
}

llvm::Value* CodeGenCPU::CreateCallPacked(const CallNode* op) {
  ICHECK_EQ(op->args.size(), 4U);
  bool use_string_lookup = op->op.same_as(builtin::tvm_call_packed_lowered());
  PackedCall pc = MakeCallPackedLowered(op->args, op->dtype, op->args[2].as<IntImmNode>()->value,
                                        op->args[3].as<IntImmNode>()->value, use_string_lookup);
  return pc.ret_value;
}

llvm::Value* CodeGenCPU::CreateCallTracePacked(const CallNode* op) {
  ICHECK_EQ(op->args.size(), 5U);
  PackedCall pc = MakeCallPackedLowered(op->args, op->dtype, op->args[2].as<IntImmNode>()->value,
                                        op->args[3].as<IntImmNode>()->value, true);
  llvm::LLVMContext* ctx = llvm_target_->GetContext();
  // Get traced value.
  llvm::Value* traced_value = MakeValue(op->args[4]);
  // The update_block handles case when we need to update the return value.
  llvm::BasicBlock* update_block = llvm::BasicBlock::Create(*ctx, "update_block", function_);
  // The continue_block handles case when we need to return original
  // traced value.
  llvm::BasicBlock* continue_block = llvm::BasicBlock::Create(*ctx, "continue_block", function_);

  // Check the ret_type_code and create cmp instruction.
  llvm::Value* cmp = builder_->CreateICmpNE(
      pc.ret_type_index, llvm::ConstantInt::get(t_int_, ffi::TypeIndex::kTVMFFINone));
  builder_->CreateCondBr(cmp, update_block, continue_block);
  builder_->SetInsertPoint(update_block);
  builder_->CreateBr(continue_block);
  builder_->SetInsertPoint(continue_block);
  // The return value depends on from what bb we come from.
  llvm::PHINode* phi_rvalue = builder_->CreatePHI(traced_value->getType(), 2);
  phi_rvalue->addIncoming(pc.ret_value, update_block);
  phi_rvalue->addIncoming(traced_value, pc.end_block);
  return phi_rvalue;
}

llvm::Value* CodeGenCPU::RuntimeTVMFFIFunctionCall() {
  if (f_tvm_ffi_func_call_ != nullptr) return f_tvm_ffi_func_call_;
  return GetContextPtr(gv_tvm_ffi_func_call_);
}

llvm::Value* CodeGenCPU::RuntimeTVMGetFuncFromEnv() {
  if (f_tvm_get_func_from_env_ != nullptr) return f_tvm_get_func_from_env_;
  return GetContextPtr(gv_tvm_get_func_from_env_);
}
llvm::Value* CodeGenCPU::RuntimeTVMFFIErrorSetRaisedFromCStr() {
  if (f_tvm_ffi_set_raised_by_c_str_ != nullptr) return f_tvm_ffi_set_raised_by_c_str_;
  return GetContextPtr(gv_tvm_ffi_set_last_error_c_str_);
}
llvm::Value* CodeGenCPU::RuntimeTVMParallelLaunch() {
  if (f_tvm_parallel_launch_ != nullptr) return f_tvm_parallel_launch_;
  return GetContextPtr(gv_tvm_parallel_launch_);
}

llvm::Value* CodeGenCPU::RuntimeTVMParallelBarrier() {
  if (f_tvm_parallel_barrier_ != nullptr) return f_tvm_parallel_barrier_;
  return GetContextPtr(gv_tvm_parallel_barrier_);
}

void CodeGenCPU::AddStartupFunction() {
  if (!target_c_runtime_) {
    llvm::FunctionType* ftype = llvm::FunctionType::get(t_void_, {}, false);
    function_ = llvm::Function::Create(ftype, llvm::Function::InternalLinkage,
                                       "__tvm_module_startup", module_.get());
    SetTargetAttributes(function_);
    llvm::BasicBlock* startup_entry =
        llvm::BasicBlock::Create(*llvm_target_->GetContext(), "entry", function_);
    builder_->SetInsertPoint(startup_entry);
    for (const auto& kv : export_system_symbols_) {
      llvm::Value* name = GetConstString(kv.first);
      builder_->CreateCall(f_tvm_register_system_symbol_,
                           {name, builder_->CreateBitCast(kv.second, t_void_p_)});
    }
    llvm::appendToGlobalCtors(*module_, function_, 65535);
    builder_->CreateRet(nullptr);
  }
}

llvm::Value* CodeGenCPU::CreateIntrinsic(const CallNode* op) {
  if (op->op.same_as(builtin::tvm_call_packed_lowered())) {
    return CreateCallPacked(op);
  } else if (op->op.same_as(builtin::tvm_call_trace_packed_lowered())) {
    return CreateCallTracePacked(op);
  } else if (op->op.same_as(builtin::tvm_call_cpacked_lowered())) {
    return CreateCallPacked(op);
  } else if (op->op.same_as(builtin::tvm_static_handle())) {
    return CreateStaticHandle();
  } else if (op->op.same_as(builtin::tvm_throw_last_error())) {
    builder_->CreateRet(ConstInt32(-1));
    auto next_block = std::next(builder_->GetInsertBlock()->getIterator());
    llvm::BasicBlock* new_bb =
        llvm::BasicBlock::Create(*llvm_target_->GetContext(), "cont", function_, &*next_block);
    builder_->SetInsertPoint(new_bb);
    return ConstInt32(-1);
  } else if (op->op.same_as(builtin::tvm_struct_get())) {
    ICHECK_EQ(op->args.size(), 3U);
    int kind = op->args[2].as<IntImm>().value()->value;
    TypedPointer ref =
        CreateStructRefPtr(op->dtype, MakeValue(op->args[0]), MakeValue(op->args[1]), kind);
    if (kind == builtin::kArrAddr) {
      return builder_->CreatePointerCast(ref.addr, t_void_p_);
    }

    llvm::Value* struct_value = builder_->CreateLoad(ref.type, ref.addr);

    if (op->dtype == DataType::Bool()) {
      struct_value = CreateCast(DataType::Int(64), op->dtype, struct_value);
    }

    return struct_value;
  } else if (op->op.same_as(builtin::tvm_struct_set())) {
    ICHECK_EQ(op->args.size(), 4U);
    int kind = op->args[2].as<IntImm>().value()->value;
    llvm::Value* value = MakeValue(op->args[3]);
    TypedPointer ref = CreateStructRefPtr(op->args[3].dtype(), MakeValue(op->args[0]),
                                          MakeValue(op->args[1]), kind);
    ICHECK(kind != builtin::kArrAddr);
    if (value->getType()->isPointerTy()) {
      value = builder_->CreatePointerCast(value, ref.type);
    }

    if (kind == builtin::kTVMFFIAnyUnionValue) {
      // when we set any union value, we need to be careful to
      // clear off the union value to zero if the set size is less than 64 bits
      if (data_layout_->getTypeAllocSize(ref.type) != 8) {
        llvm::Value* i64_addr =
            builder_->CreatePointerCast(ref.addr, llvmGetPointerTo(t_int64_, 0));
        builder_->CreateStore(ConstInt64(0), i64_addr);
      }
    }
    builder_->CreateStore(value, ref.addr);
    return ConstInt32(0);
  } else if (op->op.same_as(builtin::tvm_stack_alloca())) {
    ICHECK_EQ(op->args.size(), 2U);
    std::string type = op->args[0].as<StringImm>().value()->value;
    return WithFunctionEntry([&]() -> llvm::AllocaInst* {
      const int64_t* pval = as_const_int(op->args[1]);
      ICHECK(pval) << "require stack alloca to contain constant value";
      llvm::Value* num = ConstInt32(pval[0]);
      if (type == "shape") {
        return builder_->CreateAlloca(t_tvm_shape_index_, num);
      } else if (type == "tvm_ffi_any") {
        return builder_->CreateAlloca(t_tvm_ffi_any_, num);
      } else if (type == "array") {
        return builder_->CreateAlloca(t_tvm_array_, num);
      } else if (type == "tensormap") {
        auto* alloca = builder_->CreateAlloca(t_tvm_tensormap_, num);
        alloca->setAlignment(llvm::Align(64));
        return alloca;
      } else {
        LOG(FATAL) << "Unknown stack alloca type " << type;
      }
    });
  } else {
    return CodeGenLLVM::CreateIntrinsic(op);
  }
}

void CodeGenCPU::VisitStmt_(const AssertStmtNode* op) {
  EmitDebugLocation(op);
  llvm::Value* cond = MakeValue(op->condition);
  std::ostringstream os;
  os << "Assert fail: " << op->condition;
  if (op->message.as<StringImmNode>()) {
    os << ", " << op->message.as<StringImmNode>()->value;
  }
  llvm::Value* msg = GetConstString(os.str());
  llvm::LLVMContext* ctx = llvm_target_->GetContext();
  auto* fail_block = llvm::BasicBlock::Create(*ctx, "assert_fail", function_);
  auto* end_block = llvm::BasicBlock::Create(*ctx, "assert_end", function_);
  builder_->CreateCondBr(cond, end_block, fail_block, md_very_likely_branch_);
  // fail condition.
  builder_->SetInsertPoint(fail_block);

#if TVM_LLVM_VERSION >= 90
  auto err_callee = llvm::FunctionCallee(ftype_tvm_ffi_error_set_raised_by_c_str_,
                                         RuntimeTVMFFIErrorSetRaisedFromCStr());
#else
  auto err_callee = RuntimeTVMFFIErrorSetRaisedFromCStr();
#endif
  builder_->CreateCall(err_callee, {GetConstString("RuntimeError"), msg});
  builder_->CreateRet(ConstInt32(-1));
  // otherwise set it to be new end.
  builder_->SetInsertPoint(end_block);
  CodeGenLLVM::VisitStmt_(op);
}

void CodeGenCPU::VisitStmt_(const AttrStmtNode* op) {
  EmitDebugLocation(op);
  if (op->attr_key == tir::attr::coproc_uop_scope) {
    const StringImmNode* value = op->value.as<StringImmNode>();
    ICHECK(value != nullptr);
    this->CreateStaticInit(value->value, op->body);
  } else if (op->attr_key == tir::attr::compute_scope) {
    this->CreateComputeScope(op);
  } else if (tir::attr::IsPragmaKey(op->attr_key)) {
    if (op->attr_key == "pragma_parallel_stride_pattern") {
      ICHECK(parallel_env_.penv != nullptr)
          << "Pragma parallel_stride_pattern only valid in parallel launch";
      parallel_env_.stride_pattern = true;
      this->VisitStmt(op->body);
    } else if (op->attr_key == "pragma_parallel_launch_point") {
      CreateParallelLaunch(op->body, 0, "pragma_parallel");
    } else if (op->attr_key == "pragma_parallel_barrier_when_finish") {
      ICHECK(parallel_env_.penv != nullptr) << "Cannot run barrier without parallel environment";
      ICHECK(!parallel_env_.in_parallel_loop)
          << "Cannot not place within parallel loop as the workload may differ, "
          << " place it between parallel and parallel_launch_point";
      this->VisitStmt(op->body);
#if TVM_LLVM_VERSION >= 90
      auto bar_callee =
          llvm::FunctionCallee(ftype_tvm_parallel_barrier_, RuntimeTVMParallelBarrier());
#else
      auto bar_callee = RuntimeTVMParallelBarrier();
#endif
      builder_->CreateCall(bar_callee, {MakeValue(parallel_env_.task_id), parallel_env_.penv});
    } else if (op->attr_key == tir::attr::pragma_import_llvm) {
      const StringImmNode* value = op->value.as<StringImmNode>();
      ICHECK(value != nullptr);
      this->HandleImport(value->value);
      this->VisitStmt(op->body);
    } else {
      LOG(WARNING) << "Unknown pragma " << op->attr_key;
      this->VisitStmt(op->body);
    }
  } else {
    CodeGenLLVM::VisitStmt_(op);
  }
}

void CodeGenCPU::VisitStmt_(const ForNode* op) {
  EmitDebugLocation(op);
  ICHECK(is_zero(op->min));
  if (op->kind == ForKind::kSerial || op->kind == ForKind::kUnrolled) {
    CodeGenLLVM::VisitStmt_(op);
  } else if (op->kind == ForKind::kParallel) {
    if (parallel_env_.penv == nullptr) {
      CreateParallelLaunch(For(op->loop_var, op->min, op->extent, op->kind, op->body,
                               op->thread_binding, op->annotations),
                           0, std::string("loop_parallel_") + op->loop_var->name_hint.c_str());
    } else {
      // already in parallel env.
      ICHECK(parallel_env_.task_id.defined());
      ICHECK(parallel_env_.num_task.defined());
      ICHECK(parallel_env_.penv != nullptr);
      DataType t = op->extent.dtype();
      PrimExpr num_task = cast(t, parallel_env_.num_task);
      PrimExpr task_id = cast(t, parallel_env_.task_id);
      ICHECK(!parallel_env_.in_parallel_loop)
          << "Nested parallel loop is not supported by threadpool, try fuse them instead";
      parallel_env_.in_parallel_loop = true;
      if (parallel_env_.stride_pattern) {
        CreateSerialFor(MakeValue(task_id), MakeValue(op->extent), MakeValue(num_task),
                        op->loop_var, op->body);
      } else {
        PrimExpr step = (op->extent + num_task - make_const(t, 1)) / num_task;
        PrimExpr begin = min(task_id * step, op->extent);
        PrimExpr end = min((task_id + make_const(t, 1)) * step, op->extent);
        CreateSerialFor(MakeValue(begin), MakeValue(end),
                        llvm::ConstantInt::getSigned(GetLLVMType(end), 1), op->loop_var, op->body);
      }
      parallel_env_.in_parallel_loop = false;
      ++parallel_env_.parallel_loop_count;
    }
  } else {
    LOG(FATAL) << "cannot handle for type " << op->kind;
  }
}

TVM_FFI_STATIC_INIT_BLOCK({
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def_packed("tvm.codegen.llvm.target_cpu",
                               [](const ffi::PackedArgs& targs, ffi::Any* rv) {
                                 *rv = static_cast<void*>(new CodeGenCPU());
                               });
});

}  // namespace codegen
}  // namespace tvm

#endif  // TVM_LLVM_VERSION
