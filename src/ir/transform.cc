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
 * \file src/ir/transform.cc
 * \brief Infrastructure for transformation passes.
 */
#include <dmlc/thread_local.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ffi/rvalue_ref.h>
#include <tvm/ir/transform.h>
#include <tvm/node/repr_printer.h>
#include <tvm/node/structural_hash.h>
#include <tvm/relax/expr.h>
#include <tvm/runtime/device_api.h>

#include <chrono>
#include <iomanip>
#include <stack>
#include <unordered_set>

#include "../runtime/regex.h"

namespace tvm {
namespace transform {

using tvm::ReprPrinter;
using tvm::ffi::Any;
using tvm::ffi::PackedArgs;

TVM_REGISTER_PASS_CONFIG_OPTION("testing.immutable_module", Bool);

struct PassContextThreadLocalEntry {
  /*! \brief The default pass context. */
  PassContext default_context;

  /*! \brief The current pass context. */
  std::stack<PassContext> context_stack;

  PassContextThreadLocalEntry() { default_context = PassContext(make_object<PassContextNode>()); }
};

/*! \brief Thread local store to hold the pass context. */
typedef dmlc::ThreadLocalStore<PassContextThreadLocalEntry> RelayPassContextThreadLocalStore;

void PassContext::EnterWithScope() {
  InstrumentEnterPassContext();

  PassContextThreadLocalEntry* entry = RelayPassContextThreadLocalStore::Get();
  entry->context_stack.push(*this);
}

void PassContext::ExitWithScope() {
  PassContextThreadLocalEntry* entry = RelayPassContextThreadLocalStore::Get();
  ICHECK(!entry->context_stack.empty());
  ICHECK(entry->context_stack.top().same_as(*this));
  entry->context_stack.pop();

  InstrumentExitPassContext();
}

PassContext PassContext::Current() {
  PassContextThreadLocalEntry* entry = RelayPassContextThreadLocalStore::Get();
  if (!entry->context_stack.empty()) {
    return entry->context_stack.top();
  } else {
    return entry->default_context;
  }
}

// linearly scan the pass array to match pass_name
bool PassArrayContains(const Array<String>& pass_array, const std::string& pass_name) {
  for (auto x : pass_array) {
    if (x == pass_name) return true;
  }
  return false;
}

bool PassContext::PassEnabled(const PassInfo& info) const {
  if (PassArrayContains(operator->()->disabled_pass, info->name)) {
    return false;
  }

  if (PassArrayContains(operator->()->required_pass, info->name)) {
    return true;
  }

  return operator->()->opt_level >= info->opt_level;
}

class PassConfigManager {
 public:
  void Register(std::string key, String value_type_str,
                std::function<ffi::Any(ffi::Any)> legalization) {
    ICHECK_EQ(key2vtype_.count(key), 0U);
    ValueTypeInfo info;
    info.type_str = value_type_str;
    info.legalization = legalization;
    key2vtype_[key] = info;
  }

  // Trying to validate and legalize a config.
  void Legalize(Map<String, ffi::Any>* config) {
    std::vector<std::pair<std::string, ffi::Any>> update;
    for (auto [key, value] : *config) {
      auto it = key2vtype_.find(key);
      if (it == key2vtype_.end()) {
        std::ostringstream os;
        os << "AttributeError: Invalid config option \'" << key << "\' candidates are:";
        int counter = 0;
        for (const auto& [key, value] : key2vtype_) {
          os << ' ';
          if (counter++ != 0) os << ',';
          os << key;
        }
        LOG(FATAL) << os.str();
      }
      const auto& info = it->second;

      ICHECK(value != nullptr) << "AttributeError: " << key << " is None";

      ICHECK(info.legalization) << "AttributeError: "
                                << "Config option \'" << key
                                << "\' was defined without a legalization function.";
      auto legalized = info.legalization(value);
      if (!legalized.same_as(value)) {
        update.emplace_back(key, legalized);
      }
    }
    for (auto&& kv : update) {
      config->Set(kv.first, kv.second);
    }
  }

  Map<String, Map<String, String>> ListConfigs() {
    Map<String, Map<String, String>> configs;
    for (const auto& kv : key2vtype_) {
      Map<String, String> metadata;
      metadata.Set("type", kv.second.type_str);
      configs.Set(kv.first, metadata);
    }
    return configs;
  }

  static PassConfigManager* Global() {
    static auto* inst = new PassConfigManager();
    return inst;
  }

 private:
  struct ValueTypeInfo {
    std::string type_str;
    std::function<ffi::Any(ffi::Any)> legalization;
  };

  std::unordered_map<std::string, ValueTypeInfo> key2vtype_;
};

void PassContext::RegisterConfigOption(const char* key, String value_type_str,
                                       std::function<ffi::Any(ffi::Any)> legalization) {
  PassConfigManager::Global()->Register(key, value_type_str, legalization);
}

Map<String, Map<String, String>> PassContext::ListConfigs() {
  return PassConfigManager::Global()->ListConfigs();
}

PassContext PassContext::Create() { return PassContext(make_object<PassContextNode>()); }

namespace {
struct ClearOnError {
  Array<instrument::PassInstrument>* instruments{nullptr};

  ~ClearOnError() {
    if (instruments) {
      LOG(INFO) << "Pass instrumentation enter/exti failed.";
      LOG(INFO) << "Disabling pass instrumentation.";
      instruments->clear();
    }
  }
};
struct ExitContextOnError {
  std::vector<instrument::PassInstrument> successes;

  ~ExitContextOnError() {
    for (auto it = successes.rbegin(); it != successes.rend(); it++) {
      LOG(INFO) << (*it)->name << " exiting PassContext ...";
      (*it)->ExitPassContext();
      LOG(INFO) << (*it)->name << " exited PassContext.";
    }
  }
};
}  // namespace

void PassContext::InstrumentEnterPassContext() {
  auto pass_ctx_node = this->operator->();
  if (pass_ctx_node->instruments.defined()) {
    ClearOnError clear_context{&pass_ctx_node->instruments};
    ExitContextOnError exit_context;
    for (instrument::PassInstrument pi : pass_ctx_node->instruments) {
      pi->EnterPassContext();
      exit_context.successes.push_back(pi);
    }
    exit_context.successes.clear();
    clear_context.instruments = nullptr;
  }
}

namespace {

struct ExitPassSuccesses {
  ~ExitPassSuccesses() {
    if (all_initialized) {
      return;
    }

    LOG(INFO) << "Pass instrumentation entering pass context failed.";
    LOG(INFO) << "Disable pass instrumentation.";
    instruments->clear();

    for (auto it = successes.rbegin(); it != successes.rend(); it++) {
      LOG(INFO) << (*it)->name << " exiting PassContext ...";
      (*it)->ExitPassContext();
      LOG(INFO) << (*it)->name << " exited PassContext.";
    }
  }

  bool all_initialized{false};
  std::vector<instrument::PassInstrument> successes;
  Array<instrument::PassInstrument>* instruments{nullptr};
};
}  // namespace

void PassContext::InstrumentExitPassContext() {
  auto pass_ctx_node = this->operator->();
  if (pass_ctx_node->instruments.defined()) {
    ClearOnError clear_context{&pass_ctx_node->instruments};
    for (instrument::PassInstrument pi : pass_ctx_node->instruments) {
      pi->ExitPassContext();
    }
    clear_context.instruments = nullptr;
  }
}

bool PassContext::InstrumentBeforePass(const IRModule& ir_module, const PassInfo& pass_info) const {
  auto pass_ctx_node = this->operator->();
  if (!pass_ctx_node->instruments.defined()) {
    return true;
  }

  const bool pass_required = PassArrayContains(pass_ctx_node->required_pass, pass_info->name);
  bool should_run = true;
  if (!pass_required) {
    for (instrument::PassInstrument pi : pass_ctx_node->instruments) {
      should_run &= pi->ShouldRun(ir_module, pass_info);
    }
  }

  if (should_run) {
    for (instrument::PassInstrument pi : pass_ctx_node->instruments) {
      pi->RunBeforePass(ir_module, pass_info);
    }
  }
  return should_run;
}

void PassContext::InstrumentAfterPass(const IRModule& ir_module, const PassInfo& pass_info) const {
  auto pass_ctx_node = this->operator->();
  if (pass_ctx_node->instruments.defined()) {
    for (instrument::PassInstrument pi : pass_ctx_node->instruments) {
      pi->RunAfterPass(ir_module, pass_info);
    }
  }
}

IRModule Pass::operator()(IRModule mod) const {
  return this->operator()(std::move(mod), PassContext::Current());
}

IRModule Pass::operator()(IRModule mod, const PassContext& pass_ctx) const {
  const PassNode* node = operator->();
  ICHECK(node != nullptr);
  const PassInfo& pass_info = node->Info();
  if (!pass_ctx.InstrumentBeforePass(mod, pass_info)) {
    DLOG(INFO) << "Skipping pass : " << pass_info->name
               << " with opt level: " << pass_info->opt_level;
    return mod;
  }
  IRModule ret;
  if (pass_ctx->GetConfig<Bool>("testing.immutable_module", Bool(false)).value()) {
    ret = Pass::AssertImmutableModule(mod, node, pass_ctx);
  } else {
    ret = node->operator()(std::move(mod), pass_ctx);
  }
  pass_ctx.InstrumentAfterPass(ret, pass_info);
  return ret;
}

IRModule Pass::AssertImmutableModule(const IRModule& mod, const PassNode* node,
                                     const PassContext& pass_ctx) {
  size_t before_pass_hash = tvm::StructuralHash()(mod);
  IRModule copy_mod = mod;
  IRModule ret = node->operator()(mod, pass_ctx);
  size_t after_pass_hash = tvm::StructuralHash()(copy_mod);
  if (before_pass_hash != after_pass_hash) {
    // The chance of getting a hash conflict between a module and the same module but mutated
    // must be very low.
    LOG_FATAL << "Immutable module has been modified in pass: " << node->Info()->name;
  }
  return ret;
}

/*!
 * \brief Module-level passes are designed to implement global
 * analysis/optimizations, i.e. interprocedural optimizations (IPO), etc. Passes
 * at this level have the full control of a given Relax program including
 * addition and deletion of functions.
 */
class ModulePassNode : public PassNode {
 public:
  /* \brief The pass meta data.*/
  PassInfo pass_info;

  /*! \brief The pass function sketches the real optimization. For example,
   * we may need to perform dead code elimination on the module level. We could
   * implement the algorithm in the `pass_func` and let it run on a module. It
   * will then remove the dead code including the unused functions in the module.
   */
  std::function<IRModule(IRModule, PassContext)> pass_func;

  ModulePassNode() = default;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<ModulePassNode>().def_ro("pass_info", &ModulePassNode::pass_info);
  }

  /*!
   * \brief Run a module pass on given pass context.
   *
   * \param mod The module that an optimization pass is applied on.
   * \param mod The context that an optimization pass executes on.
   *
   * \return Return the updated module.
   */
  IRModule operator()(IRModule mod, const PassContext& pass_ctx) const final;

  /*!
   * \brief Get the pass information/meta data.
   */
  PassInfo Info() const override { return pass_info; }

  static constexpr const char* _type_key = "transform.ModulePass";
  TVM_DECLARE_FINAL_OBJECT_INFO(ModulePassNode, PassNode);
};

class ModulePass : public Pass {
 public:
  ModulePass(std::function<IRModule(IRModule, PassContext)> pass_func, PassInfo pass_info);

  TVM_DEFINE_OBJECT_REF_METHODS(ModulePass, Pass, ModulePassNode);
};

PassInfo::PassInfo(int opt_level, String name, tvm::Array<String> required, bool traceable) {
  auto pass_info = make_object<PassInfoNode>();
  pass_info->opt_level = opt_level;
  pass_info->name = std::move(name);
  pass_info->required = std::move(required);
  pass_info->traceable = std::move(traceable);
  data_ = std::move(pass_info);
}

ModulePass::ModulePass(std::function<IRModule(IRModule, PassContext)> pass_func,
                       PassInfo pass_info) {
  auto n = make_object<ModulePassNode>();
  n->pass_func = std::move(pass_func);
  n->pass_info = std::move(pass_info);
  data_ = std::move(n);
}

// Module -> Module optimizations.
IRModule ModulePassNode::operator()(IRModule mod, const PassContext& pass_ctx) const {
  DiagnosticContext previous = DiagnosticContext::Default(mod);

  if (pass_ctx->diag_ctx) {
    DiagnosticContext tmp = pass_ctx->diag_ctx.value();
    pass_ctx->diag_ctx = previous;
    previous = tmp;
  } else {
    pass_ctx->diag_ctx = previous;
  }

  ICHECK(pass_ctx->diag_ctx)
      << "The diagnostic context was set at the top of this block this is a bug.";

  const PassInfo& pass_info = Info();
  ICHECK(mod.defined()) << "The input module must be set.";

  VLOG_CONTEXT << pass_info->name;
  VLOG(0) << "Executing module pass with opt level: " << pass_info->opt_level;

  mod = pass_func(std::move(mod), pass_ctx);

  ICHECK(mod.defined()) << "The return value of a module pass must be set.";

  ICHECK(pass_ctx->diag_ctx)
      << "The diagnostic context was set at the top of this block this is a bug.";

  pass_ctx->diag_ctx.value().Render();
  pass_ctx->diag_ctx = previous;

  return mod;
}

Sequential::Sequential(tvm::Array<Pass> passes, PassInfo pass_info) {
  auto n = make_object<SequentialNode>();
  n->passes = std::move(passes);
  n->pass_info = std::move(pass_info);
  data_ = std::move(n);
}

Sequential::Sequential(tvm::Array<Pass> passes, String name) {
  auto n = make_object<SequentialNode>();
  n->passes = std::move(passes);
  PassInfo pass_info = PassInfo(0, std::move(name), {}, /* traceable */ false);
  n->pass_info = std::move(pass_info);
  data_ = std::move(n);
}

const SequentialNode* Sequential::operator->() const {
  return static_cast<const SequentialNode*>(get());
}

void SequentialNode::ResolveDependency(const IRModule& mod) {
  // TODO(zhiics) Implement it.
  // 1. Consider the required passes for each pass.
  // 2. Only resolve the enabled passes.
  // 3. Build a dependency graph. Probably we need to update the pass list.
  LOG(FATAL) << "Pass dependency has not been resolved yet."
             << "\n";
}

Pass GetPass(const String& pass_name) {
  std::optional<tvm::ffi::Function> f;
  if (pass_name.operator std::string().find("transform.") != std::string::npos) {
    f = tvm::ffi::Function::GetGlobal(pass_name);
  } else {
    f = tvm::ffi::Function::GetGlobal("transform." + pass_name);
  }
  ICHECK(f.has_value()) << "Cannot use " << pass_name << " to create the pass";
  return (*f)().cast<Pass>();
}

// TODO(zhiics): we currently only sequentially execute each pass in
// a Sequential without the consideration of their orders. The phase
// ordering problem needs to be handled in the future.
IRModule SequentialNode::operator()(IRModule mod, const PassContext& pass_ctx) const {
  for (const Pass& pass : passes) {
    VLOG(0) << "Running pass " << pass->Info()->name;
    ICHECK(pass.defined()) << "Found undefined pass for optimization.";
    const PassInfo& pass_info = pass->Info();
    if (!pass_ctx.PassEnabled(pass_info)) {
      VLOG(0) << "skipping disabled pass '" << pass_info->name << "'";
      continue;
    }

    // resolve dependencies
    for (const auto& it : pass_info->required) {
      mod = GetPass(it)(std::move(mod), pass_ctx);
    }

    mod = pass(std::move(mod), pass_ctx);
  }
  return mod;
}

Pass CreateModulePass(std::function<IRModule(IRModule, PassContext)> pass_func, int opt_level,
                      String name, tvm::Array<String> required, bool traceable) {
  PassInfo pass_info = PassInfo(opt_level, name, required, traceable);
  return ModulePass(std::move(pass_func), pass_info);
}

TVM_REGISTER_NODE_TYPE(PassInfoNode);

TVM_FFI_STATIC_INIT_BLOCK({
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("transform.PassInfo",
           [](int opt_level, String name, tvm::Array<String> required, bool traceable) {
             return PassInfo(opt_level, name, required, traceable);
           })
      .def_packed("transform.Info", [](ffi::PackedArgs args, ffi::Any* ret) {
        Pass pass = args[0].cast<Pass>();
        *ret = pass->Info();
      });
});

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<PassInfoNode>([](const ObjectRef& ref, tvm::ReprPrinter* p) {
      auto* node = static_cast<const PassInfoNode*>(ref.get());
      p->stream << "The meta data of the pass - ";
      p->stream << "pass name: " << node->name;
      p->stream << ", opt_level: " << node->opt_level;
      if (node->required.empty()) {
        p->stream << ", required passes: []\n";
      } else {
        p->stream << ", required passes: ["
                  << "\n";
        for (const auto& it : node->required) {
          p->stream << it << ", ";
        }
        p->stream << "]\n";
      }
    });

TVM_FFI_STATIC_INIT_BLOCK({
  PassContextNode::RegisterReflection();
  PassInfoNode::RegisterReflection();
  SequentialNode::RegisterReflection();
  ModulePassNode::RegisterReflection();
});

TVM_REGISTER_NODE_TYPE(ModulePassNode);

TVM_FFI_STATIC_INIT_BLOCK({
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("transform.MakeModulePass",
           [](ffi::TypedFunction<IRModule(ffi::RValueRef<IRModule>, PassContext)> pass_func,
              PassInfo pass_info) {
             auto wrapped_pass_func = [pass_func](IRModule mod, PassContext ctx) {
               return pass_func(ffi::RValueRef<IRModule>(std::move(mod)), ctx);
             };
             return ModulePass(wrapped_pass_func, pass_info);
           })
      .def("transform.RunPass",
           [](Pass pass, ffi::RValueRef<IRModule> mod) { return pass(*std::move(mod)); });
});

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<ModulePassNode>([](const ObjectRef& ref, ReprPrinter* p) {
      auto* node = static_cast<const ModulePassNode*>(ref.get());
      const PassInfo info = node->Info();
      p->stream << "Run Module pass: " << info->name << " at the optimization level "
                << info->opt_level;
    });

TVM_REGISTER_NODE_TYPE(SequentialNode);

TVM_FFI_STATIC_INIT_BLOCK({
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def_packed("transform.Sequential", [](ffi::PackedArgs args, ffi::Any* ret) {
    auto passes = args[0].cast<tvm::Array<Pass>>();
    int opt_level = args[1].cast<int>();
    std::string name = args[2].cast<std::string>();
    auto required = args[3].cast<tvm::Array<String>>();
    bool traceable = args[4].cast<bool>();
    PassInfo pass_info = PassInfo(opt_level, name, required, /* traceable */ traceable);
    *ret = Sequential(passes, pass_info);
  });
});

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<SequentialNode>([](const ObjectRef& ref, ReprPrinter* p) {
      auto* node = static_cast<const SequentialNode*>(ref.get());
      const PassInfo info = node->Info();
      p->stream << "Run Sequential pass: " << info->name << " at the optimization level "
                << info->opt_level << ". ";
      p->stream << "The passes will be executed are: [";
      for (const auto& it : node->passes) {
        const PassInfo pass_info = it->Info();
        p->stream << pass_info->name << " ";
      }
      p->stream << "]";
    });

TVM_REGISTER_NODE_TYPE(PassContextNode);

TVM_FFI_STATIC_INIT_BLOCK({
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def(
      "transform.PassContext",
      [](int opt_level, Array<String> required, Array<String> disabled,
         Array<instrument::PassInstrument> instruments, Optional<Map<String, ffi::Any>> config) {
        auto pctx = PassContext::Create();
        pctx->opt_level = opt_level;

        pctx->required_pass = std::move(required);
        pctx->disabled_pass = std::move(disabled);
        pctx->instruments = std::move(instruments);

        if (config.defined()) {
          pctx->config = config.value();
        }
        PassConfigManager::Global()->Legalize(&(pctx->config));
        return pctx;
      });
});

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<PassContextNode>([](const ObjectRef& ref, ReprPrinter* p) {
      auto* node = static_cast<const PassContextNode*>(ref.get());
      p->stream << "Pass context information: "
                << "\n";
      p->stream << "\topt_level: " << node->opt_level << "\n";

      p->stream << "\trequired passes: " << node->required_pass << "\n";
      p->stream << "\tdisabled passes: " << node->disabled_pass << "\n";
      p->stream << "\tinstruments: " << node->instruments << "\n";

      p->stream << "\tconfig: " << node->config << "\n";
    });

class PassContext::Internal {
 public:
  static void EnterScope(PassContext pass_ctx) { pass_ctx.EnterWithScope(); }

  static void ExitScope(PassContext pass_ctx) { pass_ctx.ExitWithScope(); }
};

TVM_FFI_STATIC_INIT_BLOCK({
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("transform.GetCurrentPassContext", PassContext::Current)
      .def("transform.EnterPassContext", PassContext::Internal::EnterScope)
      .def("transform.ExitPassContext", PassContext::Internal::ExitScope)
      .def("transform.OverrideInstruments",
           [](PassContext pass_ctx, Array<instrument::PassInstrument> instruments) {
             pass_ctx.InstrumentExitPassContext();
             pass_ctx->instruments = instruments;
             pass_ctx.InstrumentEnterPassContext();
           });
});

Pass PrintIR(String header, bool show_meta_data) {
  auto pass_func = [header, show_meta_data](IRModule mod, const PassContext& ctx) {
    LOG(INFO) << "PrintIR(" << header << "):\n" << mod;
    return mod;
  };
  return CreateModulePass(pass_func, 0, "PrintIR", {}, /* traceable */ false);
}

TVM_FFI_STATIC_INIT_BLOCK({
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("transform.PrintIR", PrintIR)
      .def("transform.ListConfigs", PassContext::ListConfigs);
});

}  // namespace transform
}  // namespace tvm
