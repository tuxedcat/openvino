// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <threading/ie_cpu_streams_executor.hpp>

#include <memory>
#include <future>
#include <iostream>

void example_itask_executor() {
// ! [itask_executor:define_pipeline]
    // std::promise is move only object so to satisfy copy callable constraint we use std::shared_ptr
    auto promise = std::make_shared<std::promise<void>>();
    // When the promise is created we can get std::future to wait the result
    auto future = promise->get_future();
    // Rather simple task
    InferenceEngine::Task task = [] {std::cout << "Some Output" << std::endl; };
    // Create an executor
    InferenceEngine::ITaskExecutor::Ptr taskExecutor = std::make_shared<InferenceEngine::CPUStreamsExecutor>();
    if (taskExecutor == nullptr) {
        // ProcessError(e);
        return;
    }
    // We capture the task and the promise. When the task is executed in the task executor context
    // we munually call std::promise::set_value() method
    taskExecutor->run([task, promise] {
        std::exception_ptr currentException;
        try {
            task();
        } catch(...) {
            // If there is some exceptions store the pointer to current exception
            currentException = std::current_exception();
        }

        if (nullptr == currentException) {
            promise->set_value();  //  <-- If there is no problems just call std::promise::set_value()
        } else {
            promise->set_exception(currentException);    //  <-- If there is an exception forward it to std::future object
        }
    });
    // To wait the task completion we call std::future::wait method
    future.wait();  //  The current thread will be blocked here and wait when std::promise::set_value()
                    //  or std::promise::set_exception() method will be called.

    // If the future store the exception it will be rethrown in std::future::get method
    try {
        future.get();
    } catch(std::exception& /*e*/) {
        // ProcessError(e);
    }
// ! [itask_executor:define_pipeline]
}
