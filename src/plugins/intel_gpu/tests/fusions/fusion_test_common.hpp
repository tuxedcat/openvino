// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "test_utils.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/data.hpp>
#include <intel_gpu/graph/network.hpp>

#include <cmath>

using namespace cldnn;
using namespace ::tests;

template<typename T>
class BaseFusingTest : public ::testing::TestWithParam<T> {
public:
#ifdef ENABLE_ONEDNN_FOR_GPU
    cldnn::engine& engine = get_onednn_test_engine();
#else
    cldnn::engine& engine = get_test_engine();
#endif
    cldnn::topology topology_fused;
    cldnn::topology topology_non_fused;
    cldnn::build_options bo_fused;
    cldnn::build_options bo_not_fused;

    float tolerance = 0.0f;

    static const int min_random = -200;
    static const int max_random = 200;

    void SetUp() override {
        bo_fused.set_option(build_option::optimize_data(true));
        bo_not_fused.set_option(build_option::optimize_data(false));
        bo_not_fused.set_option(build_option::allow_static_input_reorder(true));
    }

    void compare(network& net_ref, network& net_opt, T& p, bool count_reorder = false) {
        auto outputs_ref = net_ref.execute();
        auto outputs_opt = net_opt.execute();
        auto output_prim_id_ref = outputs_ref.begin()->first;
        auto output_prim_id_opt = outputs_opt.begin()->first;
        auto get_reorders_count = [](network& net) -> size_t {
            size_t count = 0;
            for (auto& pi : net.get_primitives_info()) {
                if (pi.type_id == "reorder") {
                    auto exec_prims = net.get_executed_primitives();
                    auto it = std::find_if(exec_prims.begin(), exec_prims.end(), [&](const std::pair<primitive_id, event::ptr>& e) -> bool {
                        return e.first == pi.original_id;
                    });
                    // We count executed reorders only
                    if (it != exec_prims.end())
                        count++;
                }
            }
            return count;
        };

        size_t reorders_count_fused = get_reorders_count(net_opt);
        size_t reorders_count_not_fused = get_reorders_count(net_ref);

        std::stringstream description;
        description << std::endl << "net_ref: " << std::endl;
        for (auto i : net_ref.get_primitives_info()) {
            description << "  " << i.original_id << " " << i.kernel_id << std::endl;
        }
        description << "net_opt: " << std::endl;
        for (auto i : net_opt.get_primitives_info()) {
            description << "  " << i.original_id << " " << i.kernel_id << std::endl;
        }
        SCOPED_TRACE(description.str());
        // Subtract reorders count to handle execution in different layouts when input/output reorders can be added in the graph
        ASSERT_EQ(net_opt.get_executed_primitives().size() - (count_reorder ? 0 : reorders_count_fused), p.expected_fused_primitives);
        ASSERT_EQ(net_ref.get_executed_primitives().size() - (count_reorder ? 0 : reorders_count_not_fused), p.expected_not_fused_primitives);
        ASSERT_EQ(outputs_ref.size(), outputs_opt.size());
        ASSERT_EQ(outputs_ref.size(), size_t(1));
        std::vector<float> refvals;
        std::vector<float> optvals;
        if (net_ref.get_output_layout(output_prim_id_ref).data_type == data_types::f32) {
            refvals=net_ref.get_output_values<float>(output_prim_id_ref);
        } else {
            for(auto i:net_ref.get_output_values<FLOAT16>(output_prim_id_ref))
                refvals.push_back(i);
        }
        if (net_opt.get_output_layout(output_prim_id_opt).data_type == data_types::f32) {
            optvals=net_opt.get_output_values<float>(output_prim_id_opt);
        } else {
            for(auto i:net_opt.get_output_values<FLOAT16>(output_prim_id_opt))
                optvals.push_back(i);
        }
        ASSERT_EQ(refvals.size(), optvals.size());
        for (size_t i = 0; i < refvals.size(); i++) {
            //abs err less than tolerance = (a-b)<tol
            //rel err less than tolerance = (a-b)/a<tol = (a-b)<tol*a
            //abs err or rel err less than tolerance = (a-b)<max(tol,tol*a)=tol*max(1,a)

            //case1: Two very large value a,b
            //small different behavior cause very large absolute error. relative error makes sense
            //case2: Two very small value a,b
            //small different behavior cause very large relative error. absolute error makes sense

            //max(abs,rel) is common criterion in competitive programming.
            ASSERT_NEAR(refvals[i], optvals[i], tolerance*std::max(1.f,refvals[i])) << "i = " << i;
        }
        auto sqx_accumulator = [](float acc, float x) {
            return acc + x * x;
        };
        float E_X = std::accumulate(refvals.begin(), refvals.end(), 0.f) / refvals.size();
        float E_SQX = std::accumulate(refvals.begin(), refvals.end(), 0.f, sqx_accumulator) / refvals.size();
        float SD = std::sqrt(E_SQX - E_X * E_X);
        float epsilon = default_tolerance(data_types::f32);
        if (SD < epsilon * refvals.size()) {
            std::cout << "WARNING: output variance is too low" << std::endl;
        }
    }

    void check_fusions_correctness(network& network_fused, std::map<std::string, std::vector<std::string>> expected_fused_primitives_ids = {}) {
        if (expected_fused_primitives_ids.size()) {
            auto primitives_info = network_fused.get_primitives_info();
            for (auto& prim : expected_fused_primitives_ids) {
                auto info = std::find_if(primitives_info.begin(), primitives_info.end(),
                                         [&prim](const primitive_info& info) -> bool { return info.original_id == prim.first; });
                if (info != primitives_info.end()) {
                    auto fused_primitives = info->c_fused_ids;
                    for (auto& expected_fused_prim : prim.second)
                        if (std::find(fused_primitives.begin(), fused_primitives.end(), expected_fused_prim) == fused_primitives.end())
                            FAIL() << "Couldn't find requested net_opt primitive id " + prim.first;
                } else {
                    FAIL() << "Couldn't find requested primitive id " + prim.first;
                }
            }
        }
    }

    cldnn::memory::ptr get_mem(cldnn::layout l) {
        auto prim = engine.allocate_memory(l);
        tensor s = l.get_tensor();
        if (l.data_type == data_types::bin) {
            VF<int32_t> rnd_vec = generate_random_1d<int32_t>(s.count() / 32, min_random, max_random);
            set_values(prim, rnd_vec);
        } else if (l.data_type == data_types::i8 || l.data_type == data_types::u8) {
            VF<uint8_t> rnd_vec = generate_random_1d<uint8_t>(s.count(), min_random, max_random);
            set_values(prim, rnd_vec);
        } else if (l.data_type == data_types::f16) {
            VF<uint16_t> rnd_vec = generate_random_1d<uint16_t>(s.count(), -1, 1);
            set_values(prim, rnd_vec);
        } else {
            VF<float> rnd_vec = generate_random_1d<float>(s.count(), -1, 1);
            set_values(prim, rnd_vec);
        }

        return prim;
    }

    cldnn::memory::ptr get_mem(cldnn::layout l, float fill_value) {
        auto prim = engine.allocate_memory(l);
        tensor s = l.get_tensor();
        if (l.data_type == data_types::bin) {
            VF<int32_t> rnd_vec(s.count() / 32, static_cast<int32_t>(fill_value));
            set_values(prim, rnd_vec);
        } else if (l.data_type == data_types::f16) {
            VF<uint16_t> rnd_vec(s.count(), float_to_half(fill_value));
            set_values(prim, rnd_vec);
        } else if (l.data_type == data_types::f32) {
            VF<float> rnd_vec(s.count(), fill_value);
            set_values(prim, rnd_vec);
        } else if (l.data_type == data_types::u8) {
            VF<uint8_t> rnd_vec(s.count(), static_cast<uint8_t>(fill_value));
            set_values(prim, rnd_vec);
        } else if (l.data_type == data_types::i8) {
            VF<int8_t> rnd_vec(s.count(), static_cast<int8_t>(fill_value));
            set_values(prim, rnd_vec);
        } else {
            throw std::runtime_error("get_mem: Unsupported precision");
        }

        return prim;
    }

    cldnn::memory::ptr get_repeatless_mem(cldnn::layout l, int min, int max) {
        auto prim = engine.allocate_memory(l);
        tensor s = l.get_tensor();
        if (l.data_type == data_types::f32) {
            VF<float> rnd_vec = generate_random_norepetitions_1d<float>(s.count(), min, max);
            set_values(prim, rnd_vec);
        } else if (l.data_type == data_types::f16) {
            VF<FLOAT16> rnd_vec = generate_random_norepetitions_1d<FLOAT16>(s.count(), min, max);
            set_values(prim, rnd_vec);
        } else if (l.data_type == data_types::i8) {
            VF<int8_t> rnd_vec = generate_random_norepetitions_1d<int8_t>(s.count(), min, max);
            set_values(prim, rnd_vec);
        }
        else if (l.data_type == data_types::bin) {
            VF<int32_t> rnd_vec = generate_random_norepetitions_1d<int32_t>(s.count(), min, max);
            set_values(prim, rnd_vec);
        }

        return prim;
    }

    cldnn::memory::ptr get_mem(cldnn::layout l, int min, int max) {
        auto prim = engine.allocate_memory(l);
        tensor s = l.get_tensor();
        if (l.data_type == data_types::f32) {
            VF<float> rnd_vec = generate_random_1d<float>(s.count(), min, max);
            set_values(prim, rnd_vec);
        } else if (l.data_type == data_types::f16) {
            VF<FLOAT16> rnd_vec = generate_random_1d<FLOAT16>(s.count(), min, max);
            set_values(prim, rnd_vec);
        } else if (l.data_type == data_types::i8) {
            VF<int8_t> rnd_vec = generate_random_1d<int8_t>(s.count(), min, max);
            set_values(prim, rnd_vec);
        } else if (l.data_type == data_types::u8) {
            VF<uint8_t> rnd_vec = generate_random_1d<uint8_t>(s.count(), min, max);
            set_values(prim, rnd_vec);
        } else if (l.data_type == data_types::bin) {
            VF<int32_t> rnd_vec = generate_random_1d<int32_t>(s.count() / 32, min, max);
            set_values(prim, rnd_vec);
        }

        return prim;
    }

    layout get_output_layout(T& p) {
        return layout{ p.data_type, p.input_format, p.out_shape };
    }

    layout get_weights_layout(T& p, const int32_t /* split */ = 1) {
        cldnn::tensor weights_tensor;
        if (p.groups == 1) {
            weights_tensor = cldnn::tensor(batch(p.out_shape.feature[0]), feature(p.in_shape.feature[0]),
                                           spatial(p.kernel.spatial[0], p.kernel.spatial[1], p.kernel.spatial[2]));
        } else {
            weights_tensor = cldnn::tensor(group(p.groups), batch(p.out_shape.feature[0] / p.groups), feature(p.in_shape.feature[0] / p.groups),
                                           spatial(p.kernel.spatial[0], p.kernel.spatial[1], p.kernel.spatial[2]));
        }
        return layout{p.weights_type, p.weights_format, weights_tensor};
    }

    layout get_weights_layout(T& p, const int32_t /* split */, cldnn::format f) {
        cldnn::tensor weights_tensor;
        weights_tensor = cldnn::tensor(batch(p.out_shape.feature[0]), feature(static_cast<int32_t>(p.in_shape.feature[0] / p.groups)),
                                       spatial(p.kernel.spatial[0], p.kernel.spatial[1], p.kernel.spatial[2]));
        return layout{p.weights_type, f, weights_tensor};
    }

    layout get_bias_layout(T& p) {
        return layout{ p.default_type, format::bfyx, tensor{1, p.out_shape.feature[0], 1, 1} };
    }

    layout get_weights_zp_layout(T& p) {
        return layout{ p.weights_type, p.default_format, tensor{p.out_shape.feature[0], 1, 1, 1} };
    }

    layout get_activations_zp_layout(T& p) {
        return layout{ p.data_type, p.default_format, tensor{1, p.in_shape.feature[0], 1, 1} };
    }

    layout get_single_element_layout(T& p) {
        return layout{ p.default_type, p.default_format, tensor{1, 1, 1, 1} };
    }

    template <class... Args>
    void create_topologies(Args const&... args) {
        topology_fused.add(args...);
        topology_non_fused.add(args...);
    }

    template <class... Args>
    void add_topologies(Args const&... args) {
        topology_fused.add(args...);
        topology_non_fused.add(args...);
    }
};
