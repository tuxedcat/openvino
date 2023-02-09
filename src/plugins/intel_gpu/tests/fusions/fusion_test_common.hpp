// Copyright (C) 2018-2023 Intel Corporation
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
    cldnn::engine& engine = get_test_engine();
    cldnn::topology topology_fused;
    cldnn::topology topology_non_fused;

    ExecutionConfig cfg_fused;
    ExecutionConfig cfg_not_fused;

    float tolerance_abs = 0.0f;
    float tolerance_rel = 0.01f;

    static const int min_random = -200;
    static const int max_random = 200; //TODO: remove?

    BaseFusingTest() : cfg_fused(ov::device::id("1")), cfg_not_fused(ov::device::id("1")) {}
    void SetUp() override {
        cfg_fused.set_property(ov::intel_gpu::enable_memory_pool(false));
        cfg_not_fused.set_property(ov::intel_gpu::enable_memory_pool(false));

        cfg_fused.set_property(ov::intel_gpu::optimize_data(true));
        cfg_not_fused.set_property(ov::intel_gpu::optimize_data(false));
        cfg_not_fused.set_property(ov::intel_gpu::allow_static_input_reorder(true));

        if (engine.get_device_info().supports_immad) {
            cfg_fused.set_property(ov::intel_gpu::queue_type(QueueTypes::in_order));
            cfg_not_fused.set_property(ov::intel_gpu::queue_type(QueueTypes::in_order));
            engine.create_onednn_engine(cfg_fused);
        }
    }

    void compare(network& net_ref, network& net_opt, T& p, bool count_reorder = false) {
        auto outnodes_ref = net_ref.execute();
        auto outnodes_opt = net_opt.execute();
        auto out_id_ref = outnodes_ref.begin()->first;
        auto out_id_opt = outnodes_opt.begin()->first;

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

        size_t reorders_count_ref = get_reorders_count(net_ref);
        size_t reorders_count_opt = get_reorders_count(net_opt);

        std::stringstream description;
        description << std::endl << "net_ref: " << std::endl;
        for (auto i : net_ref.get_primitives_info()) {
            description << "  " << i.original_id << " " << i.kernel_id << std::endl;
        }
        description << "net_opt: " << std::endl;
        for (auto i : net_opt.get_primitives_info()) {
            description << "  " << i.original_id << " " << i.kernel_id << std::endl;
        }

        std::cout << description.str() << std::endl;
        print_primitive<FLOAT16>(net_ref,"input",true);
        print_primitive<FLOAT16>(net_ref,"weights_generic_layer_0",true,32);
        print_primitive<FLOAT16>(net_opt,"weights_generic_layer_0",true,32);
        // print_primitive<FLOAT16>(net_ref,"reduce",true);
        // print_primitive<FLOAT16>(net_opt,"reduce",true);
        print_primitive<float>(net_ref,out_id_ref,true);
        print_primitive<float>(net_opt,out_id_opt,true);

        std::vector<float> val_ref;
        std::vector<float> val_opt;
        auto lay_ref=net_ref.get_output_layout(out_id_ref);
        auto lay_opt=net_ref.get_output_layout(out_id_opt);
        if (lay_ref.data_type == data_types::f32) {
            val_ref = net_ref.get_output_values_to_float<float>(out_id_ref);
        } else {
            val_ref = net_ref.get_output_values_to_float<FLOAT16>(out_id_ref);
        }
        if (lay_opt.data_type == data_types::f32) {
            val_opt = net_opt.get_output_values_to_float<float>(out_id_opt);
        } else {
            val_opt = net_opt.get_output_values_to_float<FLOAT16>(out_id_opt);
        }

        ASSERT_EQ(val_ref.size(), val_opt.size());
        // ASSERT_TRUE(format::is_simple_data_format(lay_ref.format));
        // ASSERT_TRUE(format::is_simple_data_format(lay_opt.format));
        ASSERT_TRUE(lay_ref.format == lay_opt.format);
        ASSERT_TRUE(lay_ref.count() == lay_opt.count());
        // NOTE:
        // val_ref.size()==memory size
        // lay_ref.count()==shape size
        // val_ref.size()>=lay_ref.count()
        // This loop is valid only when lay_ref is planar(simple) format
        for (size_t i = 0; i < lay_ref.count(); i++) {
            float err = abs(val_opt[i] - val_ref[i]);
            ASSERT_TRUE(err <= std::max(tolerance_abs, tolerance_rel * abs(val_ref[i])) + 1e-8)
                << "i = " << i << "\ntolerance = " << std::max(tolerance_abs, tolerance_rel * abs(val_ref[i]))
                << "\ndiff = " << err << "\nref[i] = " << val_ref[i] << "\nopt[i] = " << val_opt[i];
        }
        // Subtract reorders count to handle execution in different layouts when input/output reorders can be added in the graph
        ASSERT_EQ(net_opt.get_executed_primitives().size() - (count_reorder ? 0 : reorders_count_opt), p.expected_fused_primitives);
        ASSERT_EQ(net_ref.get_executed_primitives().size() - (count_reorder ? 0 : reorders_count_ref), p.expected_not_fused_primitives);
        ASSERT_TRUE(outnodes_ref.size() == 1);
        ASSERT_TRUE(outnodes_opt.size() == 1);

        GPU_DEBUG_GET_INSTANCE(debug_config);
        GPU_DEBUG_IF(debug_config->verbose >= 2) {
            float E_X = 0;
            float E_SQX = 0;
            float abs_diff_sum = 0;
            float max_abs_X = 0;
            for (size_t i = 0; i < val_ref.size(); i++) {
                E_X += val_ref[i];
                E_SQX += val_ref[i] * val_ref[i];
                abs_diff_sum += std::abs(val_ref[i] - val_opt[i]);
                max_abs_X = std::max(max_abs_X, val_ref[i]);
            }
            E_X /= val_ref.size();
            E_SQX /= val_ref.size();
            float SD = std::sqrt((E_SQX - E_X * E_X));
            if (SD < tolerance_abs * val_ref.size())
                GPU_DEBUG_INFO << "WARNING: output variance is too low" << std::endl;
            if (abs_diff_sum / val_ref.size() > tolerance_abs * val_ref.size())
                GPU_DEBUG_INFO << "WARNING: output average difference is too high" << std::endl;
            if (max_abs_X >= 1e6)
                GPU_DEBUG_INFO << "WARNING: output absolute value is too high" << std::endl;
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
                            FAIL() << "Couldn't find requested fused primitive id " + prim.first;
                } else {
                    FAIL() << "Couldn't find requested primitive id " + prim.first;
                }
            }
        }
    }

    cldnn::memory::ptr get_mem(cldnn::layout l, FillType ft = FillType::OnlyInsideShape) {
        auto prim = engine.allocate_memory(l);
        int cnt_base = ft == FillType::All ? prim->size() / data_type_traits::size_of(l.data_type) : l.count();
        if (l.data_type == data_types::bin) {
            set_values(prim, generate_random_1d<int32_t>(cnt_base / 32, min_random, max_random), ft);
        } else if (l.data_type == data_types::i8 || l.data_type == data_types::u8) {
            set_values(prim, generate_random_1d<uint8_t>(cnt_base, min_random, max_random), ft);
        } else if (l.data_type == data_types::f16) {
            set_values(prim, generate_random_1d<FLOAT16>(cnt_base, -1, 1), ft);
        } else if (l.data_type == data_types::f32) {
            set_values(prim, generate_random_1d<float>(cnt_base, -1, 1), ft);
        } else {
            IE_THROW() << "Unimplemented data_types";
        }
        return prim;
    }

    cldnn::memory::ptr get_mem(cldnn::layout l, float fill_value, FillType ft = FillType::OnlyInsideShape) {
        auto prim = engine.allocate_memory(l);
        int cnt_base = ft == FillType::All ? prim->size() / data_type_traits::size_of(l.data_type) : l.count();
        if (l.data_type == data_types::bin) {
            set_values(prim, VF<int32_t>(cnt_base / 32, fill_value), ft);
        } else if (l.data_type == data_types::i8 || l.data_type == data_types::u8) {
            set_values(prim, VF<uint8_t>(cnt_base, fill_value), ft);
        } else if (l.data_type == data_types::f16) {
            set_values(prim, VF<FLOAT16>(cnt_base, fill_value), ft);
        } else if (l.data_type == data_types::f32) {
            set_values(prim, VF<float>(cnt_base, fill_value), ft);
        } else {
            IE_THROW() << "Unimplemented data_types";
        }
        return prim;
    }

    cldnn::memory::ptr get_repeatless_mem(cldnn::layout l, int min, int max, FillType ft = FillType::OnlyInsideShape) {
        auto prim = engine.allocate_memory(l);
        int cnt_base = ft == FillType::All ? prim->size() / data_type_traits::size_of(l.data_type) : l.count();
        if (l.data_type == data_types::f32) {
            set_values(prim, generate_random_norepetitions_1d<float>(cnt_base, min, max), ft);
        } else if (l.data_type == data_types::f16) {
            set_values(prim, generate_random_norepetitions_1d<FLOAT16>(cnt_base, min, max), ft);
        } else if (l.data_type == data_types::i8) {
            set_values(prim, generate_random_norepetitions_1d<int8_t>(cnt_base, min, max), ft);
        } else if (l.data_type == data_types::bin) {
            set_values(prim, generate_random_norepetitions_1d<int32_t>(cnt_base / 32, min, max), ft);
        } else {
            IE_THROW() << "Unimplemented data_types";
        }

        return prim;
    }

    cldnn::memory::ptr get_mem(cldnn::layout l, int min, int max, FillType ft = FillType::OnlyInsideShape) {
        auto prim = engine.allocate_memory(l);
        int cnt_base = ft == FillType::All ? prim->size() / data_type_traits::size_of(l.data_type) : l.count();
        if (l.data_type == data_types::f32) {
            set_values(prim, generate_random_1d<float>(cnt_base, min, max), ft);
        } else if (l.data_type == data_types::f16) {
            set_values(prim, generate_random_1d<FLOAT16>(cnt_base, min, max), ft);
        } else if (l.data_type == data_types::i8) {
            set_values(prim, generate_random_1d<int8_t>(cnt_base, min, max), ft);
        } else if (l.data_type == data_types::u8) {
            set_values(prim, generate_random_1d<uint8_t>(cnt_base, min, max), ft);
        } else if (l.data_type == data_types::bin) {
            set_values(prim, generate_random_1d<int32_t>(cnt_base / 32, min, max), ft);
        } else {
            IE_THROW() << "Unimplemented data_types";
        }
        return prim;
    }

    layout get_output_layout(T& p) {
        return layout{ p.data_type, p.input_format, p.out_shape };
    }

    layout get_weights_layout(T& p) {
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

    layout get_weights_layout(T& p, cldnn::format f) {
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
