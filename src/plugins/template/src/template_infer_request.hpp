// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <array>
#include <chrono>
#include <cpp_interfaces/interface/ie_iinfer_request_internal.hpp>
#include <executable.hpp>
#include <ie_input_info.hpp>
#include <map>
#include <memory>
#include <ngraph/runtime/tensor.hpp>
#include <openvino/itt.hpp>
#include <string>
#include <vector>

namespace TemplatePlugin {

// forward declaration
class CompiledModel;

// ! [infer_request:header]
class TemplateInferRequest : public InferenceEngine::IInferRequestInternal {
public:
    typedef std::shared_ptr<TemplateInferRequest> Ptr;

    TemplateInferRequest(const std::vector<std::shared_ptr<const ov::Node>>& inputs,
                         const std::vector<std::shared_ptr<const ov::Node>>& outputs,
                         const std::shared_ptr<const TemplatePlugin::CompiledModel>& compiled_model);
    ~TemplateInferRequest();

    void InferImpl() override;
    std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> GetPerformanceCounts() const override;

    // pipeline methods-stages which are used in async infer request implementation and assigned to particular executor
    void inferPreprocess();
    void startPipeline();
    void waitPipeline();
    void inferPostprocess();

    InferenceEngine::Blob::Ptr GetBlob(const std::string& name) override;
    void SetBlob(const std::string& name, const InferenceEngine::Blob::Ptr& userBlob) override;

    void SetBlobsImpl(const std::string& name, const InferenceEngine::BatchedBlob::Ptr& batchedBlob) override;

private:
    void createInferRequest();
    void allocateDeviceBuffers();
    void allocateBlobs();

    enum { Preprocess, Postprocess, StartPipeline, WaitPipeline, numOfStages };

    std::shared_ptr<const CompiledModel> m_compiled_model;
    std::array<openvino::itt::handle_t, numOfStages> _profilingTask;
    // for performance counters
    std::array<std::chrono::duration<float, std::micro>, numOfStages> _durations;

    InferenceEngine::BlobMap _networkOutputBlobs;

    std::vector<std::shared_ptr<ngraph::runtime::Tensor>> _inputTensors;
    std::vector<std::shared_ptr<ngraph::runtime::Tensor>> _outputTensors;
    std::shared_ptr<ngraph::runtime::Executable> _executable;
};
// ! [infer_request:header]

}  // namespace TemplatePlugin
