// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <vector>

// clang-format off
#ifdef ${BACKEND_NAME}_FLOAT_TOLERANCE_BITS
#define DEFAULT_FLOAT_TOLERANCE_BITS ${BACKEND_NAME}_FLOAT_TOLERANCE_BITS
#endif
#ifdef ${BACKEND_NAME}_DOUBLE_TOLERANCE_BITS
#define DEFAULT_DOUBLE_TOLERANCE_BITS ${BACKEND_NAME}_DOUBLE_TOLERANCE_BITS
#endif
// clang-format on

#include "common_test_utils/file_utils.hpp"
#include "default_opset.hpp"
#include "engines_util/test_case.hpp"
#include "engines_util/test_engines.hpp"
#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "ngraph/pass/constant_folding.hpp"
#include "ngraph/pass/manager.hpp"
#include "onnx_import/core/null_node.hpp"
#include "onnx_import/onnx.hpp"
#include "onnx_import/onnx_utils.hpp"
#include "util/all_close.hpp"
#include "util/all_close_f.hpp"
#include "util/ndarray.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"
#include "util/type_prop.hpp"

NGRAPH_SUPPRESS_DEPRECATED_START

using namespace ngraph;

static std::string s_manifest = "${MANIFEST}";
static std::string s_device = test::backend_name_to_device("${BACKEND_NAME}");

using Inputs = std::vector<std::vector<float>>;
using Outputs = std::vector<std::vector<float>>;

NGRAPH_TEST(${BACKEND_NAME}, onnx_prior_box) {
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/prior_box.onnx"));
    auto test_case = test::TestCase(function, s_device);
    std::vector<float> A(3 * 2 * 2);
    std::vector<float> B(3 * 6 * 6);
    std::vector<float> output = {
        -2.3200002,  -2.3200002,  3.6533334, 3.6533334, -3.7053659, -3.7053659, 5.0386992, 5.0386992,
        -0.98666668, -2.3200002,  4.9866667, 3.6533334, -2.3720326, -3.7053659, 6.3720322, 5.0386992,
        -2.3200002,  -0.98666668, 3.6533334, 4.9866667, -3.7053659, -2.3720326, 5.0386992, 6.3720322,
        -0.98666668, -0.98666668, 4.9866667, 4.9866667, -2.3720326, -2.3720326, 6.3720322, 6.3720322,
        0.1,         0.1,         0.2,       0.2,       0.1,        0.1,        0.2,       0.2,
        0.1,         0.1,         0.2,       0.2,       0.1,        0.1,        0.2,       0.2,
        0.1,         0.1,         0.2,       0.2,       0.1,        0.1,        0.2,       0.2,
        0.1,         0.1,         0.2,       0.2,       0.1,        0.1,        0.2,       0.2,
    };
    test_case.add_input<float>(A);
    test_case.add_input<float>(B);
    test_case.add_expected_output<float>(Shape{1, 2, 32}, output);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_priorbox_clustered) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                                        SERIALIZED_ZOO,
                                                                        "onnx/priorbox_clustered.onnx"));

    auto test_case = test::TestCase(function, s_device);
    std::vector<float> A{15.0};
    std::vector<float> B{10.0};
    std::vector<float> output = {
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        0.1, 0.1, 0.2, 0.2, 0.1, 0.1, 0.2, 0.2, 0.1, 0.1, 0.2, 0.2, 0.1, 0.1, 0.2, 0.2,
    };
    test_case.add_input<float>(A);
    test_case.add_input<float>(B);
    test_case.add_expected_output<float>(Shape{1, 2, 16}, output);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_priorbox_clustered_most_attrs_default) {
    auto function =
        onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                            SERIALIZED_ZOO,
                                                            "onnx/priorbox_clustered_most_attrs_default.onnx"));

    auto test_case = test::TestCase(function, s_device);
    std::vector<float> A(1 * 1 * 2 * 1);
    std::iota(std::begin(A), std::end(A), 0.0f);
    std::vector<float> B(1 * 1 * 3 * 3);
    std::iota(std::begin(B), std::end(B), 0.0f);
    std::vector<float> output = {-0.1666666716337203979,
                                 -0.1666666716337203979,
                                 0.1666666716337203979,
                                 0.1666666716337203979,
                                 -0.1666666716337203979,
                                 0.3333333432674407959,
                                 0.1666666716337203979,
                                 0.6666666865348815918,
                                 0.1,
                                 0.1,
                                 0.2,
                                 0.2,
                                 0.1,
                                 0.1,
                                 0.2,
                                 0.2};
    test_case.add_input<float>(A);
    test_case.add_input<float>(B);
    test_case.add_expected_output<float>(Shape{1, 2, 8}, output);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_priorbox_clustered_first_input_bad_shape) {
    try {
        auto function =
            onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                                SERIALIZED_ZOO,
                                                                "onnx/priorbox_clustered_first_input_bad_shape.onnx"));
        FAIL() << "Expected exception was not thrown";
    } catch (const ngraph::ngraph_error& e) {
        EXPECT_HAS_SUBSTRING(e.what(), std::string("Only 4D inputs are supported. First input rank: 5 (should be 4)"));
    } catch (...) {
        FAIL() << "Expected OnnxNodeValidationFailure exception was not thrown";
    }
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_priorbox_clustered_second_input_bad_shape) {
    try {
        auto function =
            onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                                SERIALIZED_ZOO,
                                                                "onnx/priorbox_clustered_second_input_bad_shape.onnx"));
        FAIL() << "Expected exception was not thrown";
    } catch (const ngraph::ngraph_error& e) {
        EXPECT_HAS_SUBSTRING(e.what(), std::string("Only 4D inputs are supported. Second input rank: 5 (should be 4)"));
    } catch (...) {
        FAIL() << "Expected OnnxNodeValidationFailure exception was not thrown";
    }
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_detection_output) {
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/detection_output.onnx"));
    auto test_case = test::TestCase(function, s_device);

    auto gen_vector = [](size_t size, float min, float max) -> std::vector<float> {
        float step = (max - min) / size;
        float next = min - step;

        std::vector<float> out(size);
        std::generate(out.begin(), out.end(), [&next, &step] {
            return next += step;
        });
        return out;
    };

    std::vector<float> logits = gen_vector(12, -2, 2);
    std::vector<float> class_preds = gen_vector(9, 0, 1);
    std::vector<float> proposals = gen_vector(12 * 2, 0, 1);
    std::vector<float> output = {0, 1, 0.777778, 0.279849,   0.283779,   0.562743,   0.695387,
                                 0, 1, 0.444444, 0.12963,    0.176075,   0.212963,   0.284573,
                                 0, 2, 0.888889, 0.279849,   0.283779,   0.562743,   0.695387,
                                 0, 2, 0.555556, 0.12963,    0.176075,   0.212963,   0.284573,
                                 0, 2, 0.222222, -0.0608094, -0.0142007, -0.0225239, 0.0304044};
    test_case.add_input<float>(logits);
    test_case.add_input<float>(class_preds);
    test_case.add_input<float>(proposals);
    test_case.add_expected_output<float>(Shape{1, 1, 5, 7}, output);
    int tolerance_bits = 6;
    test_case.run(tolerance_bits);
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_group_norm) {
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/group_norm.onnx"));
    auto test_case = test::TestCase(function, s_device);
    Shape shape{2, 8, 2, 2};
    int size = shape_size(shape);
    std::vector<float> data(size);
    std::iota(data.begin(), data.end(), 0);
    std::vector<float> output = {
        -0.52752507, -0.09108937, 0.3453464, 0.78178215, 2.4364357, 3.309307,  4.1821785, 5.05505,
        -1.5825753,  -0.27326822, 1.0360391, 2.3453465,  4.8728714, 6.618614,  8.364357,  10.1101,
        -2.6376252,  -0.45544672, 1.726732,  3.9089108,  7.309307,  9.927921,  12.546536, 15.165151,
        -3.6926756,  -0.6376257,  2.4174247, 5.472475,   9.745743,  13.237228, 16.728714, 20.2202,
        -0.52752507, -0.09108937, 0.3453464, 0.78178215, 2.4364357, 3.309307,  4.1821785, 5.05505,
        -1.5825753,  -0.27326822, 1.0360391, 2.3453465,  4.8728714, 6.618614,  8.364357,  10.1101,
        -2.6376252,  -0.45544672, 1.726732,  3.9089108,  7.309307,  9.927921,  12.546536, 15.165151,
        -3.6926756,  -0.6376257,  2.4174247, 5.472475,   9.745743,  13.237228, 16.728714, 20.2202,
    };

    test_case.add_input<float>(data);
    test_case.add_expected_output<float>(shape, output);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_group_norm_5d) {
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/group_norm_5d.onnx"));
    auto test_case = test::TestCase(function, s_device);
    Shape shape{2, 8, 1, 2, 1};
    int size = shape_size(shape);
    std::vector<float> data(size);
    std::iota(data.begin(), data.end(), 0);
    std::vector<float> output = {
        -0.34163546562, 0.55278813838,  2.89442372322,  4.68327093124,  -1.02490639686, 1.65836453437,  5.78884744644,
        9.36654186248,  -1.70817732810, 2.76394081115,  8.68327140808,  14.04981231689, -2.39144825935, 3.86951708793,
        11.57769489288, 18.73308372497, -0.34163546562, 0.55278813838,  2.89442372322,  4.68327093124,  -1.02490639686,
        1.65836453437,  5.78884744644,  9.36654186248,  -1.70817732810, 2.76394081115,  8.68327140808,  14.04981231689,
        -2.39144825935, 3.86951708793,  11.57769489288, 18.73308372497};

    test_case.add_input<float>(data);
    test_case.add_expected_output<float>(shape, output);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_normalize) {
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/normalize.onnx"));
    auto test_case = test::TestCase(function, s_device);
    std::vector<float> data(12);
    std::iota(data.begin(), data.end(), 1);
    std::vector<float> output = {
        0.19334731,
        0.33806169,
        0.44846106,
        0.53452247,
        1.4501048,
        1.5212777,
        1.5696137,
        1.6035674,
        3.4802516,
        3.3806169,
        3.2887144,
        3.2071347,
    };
    test_case.add_input<float>(data);
    test_case.add_expected_output<float>(Shape{1, 3, 2, 2}, output);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_swish_with_beta) {
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/swish_with_beta.onnx"));

    const Shape expected_output_shape{3};
    auto test_case = test::TestCase(function, s_device);
    std::vector<float> input_data{-0.5f, 0, 0.5f};
    test_case.add_input<float>(input_data);
    test_case.add_expected_output<float>(expected_output_shape, {-0.2036667, 0.0, 0.2963333});

    test_case.run_with_tolerance_as_fp(2.0e-5f);
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_swish_without_beta) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                                        SERIALIZED_ZOO,
                                                                        "onnx/swish_without_beta.onnx"));

    const Shape expected_output_shape{3};
    auto test_case = test::TestCase(function, s_device);
    std::vector<float> input_data{-0.5f, 0, 0.5f};
    test_case.add_input<float>(input_data);
    test_case.add_expected_output<float>(expected_output_shape, {-0.18877034, 0.0, 0.31122968});

    test_case.run_with_tolerance_as_fp(2.0e-5f);
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_experimental_detectron_detection_output) {
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                             SERIALIZED_ZOO,
                             "onnx/org.openvinotoolkit/experimental_detectron/detection_output.onnx"));

    auto test_case = test::TestCase(function, s_device);
    // rois
    test_case.add_input<float>({1.0f, 1.0f, 10.0f, 10.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                                1.0f, 1.0f, 1.0f,  4.0f,  1.0f, 8.0f, 5.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                                1.0f, 1.0f, 1.0f,  1.0f,  1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                                1.0f, 1.0f, 1.0f,  1.0f,  1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                                1.0f, 1.0f, 1.0f,  1.0f,  1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f});
    // deltas
    test_case.add_input<float>(
        {5.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
         1.0f, 1.0f, 1.0f, 1.0f, 4.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
         1.0f, 1.0f, 8.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
         1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,

         1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
         1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
         1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
         1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f});
    // scores
    test_case.add_input<float>({1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                                1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                                1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f});
    // im_info
    test_case.add_input<float>({1.0f, 1.0f, 1.0f});

    test_case.add_expected_output<float>(Shape{5, 4},
                                         {
                                             0.8929862f,
                                             0.892986297607421875,
                                             12.10701370239257812,
                                             12.10701370239257812,
                                             0,
                                             0.0f,
                                             0.0f,
                                             0.0f,
                                             0.0f,
                                             0.0f,
                                             0.0f,
                                             0.0f,
                                             0.0f,
                                             0.0f,
                                             0.0f,
                                             0.0f,
                                             0.0f,
                                             0.0f,
                                             0.0f,
                                             0.0f,
                                         });
    test_case.add_expected_output<int>(Shape{5}, {1, 0, 0, 0, 0});
    test_case.add_expected_output<float>(Shape{5}, {1.0f, 0.0f, 0.0f, 0.0f, 0.0f});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_experimental_detectron_detection_output_most_attrs_default) {
    auto function =
        onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                            SERIALIZED_ZOO,
                                                            "onnx/org.openvinotoolkit/experimental_detectron/"
                                                            "detection_output_most_attrs_default.onnx"));

    auto test_case = test::TestCase(function, s_device);
    // rois
    test_case.add_input<float>({1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                                1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                                1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                                1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                                1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f});
    // deltas
    test_case.add_input<float>(
        {5.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
         1.0f, 1.0f, 1.0f, 1.0f, 4.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
         1.0f, 1.0f, 8.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
         1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,

         1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
         1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
         1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
         1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f});
    // scores
    test_case.add_input<float>({1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                                1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                                1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f});
    // im_info
    test_case.add_input<float>({1.0f, 1.0f, 1.0f});

    test_case.add_expected_output<float>(Shape{5, 4},
                                         {
                                             0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                                             0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                                         });
    test_case.add_expected_output<int>(Shape{5}, {0, 0, 0, 0, 0});
    test_case.add_expected_output<float>(Shape{5}, {0.0f, 0.0f, 0.0f, 0.0f, 0.0f});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_experimental_detectron_generate_proposals_single_image) {
    auto function =
        onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                            SERIALIZED_ZOO,
                                                            "onnx/org.openvinotoolkit/experimental_detectron/"
                                                            "generate_proposals_single_image.onnx"));

    auto test_case = test::TestCase(function, s_device);
    // im_info
    test_case.add_input<float>({1.0f, 1.0f, 1.0f});
    // anchors
    test_case.add_input<float>(
        {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
         1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
         1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
         1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,

         1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
         1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
         1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
         1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,

         1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f});
    // deltas
    test_case.add_input<float>(
        {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
         1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
         1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
         1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,

         1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
         1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
         1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
         1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,

         1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f});
    // scores
    test_case.add_input<float>({
        5.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 4.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 8.0f, 1.0f,
    });

    test_case.add_expected_output<float>(Shape{6, 4},
                                         {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                                          0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f});
    test_case.add_expected_output<float>(Shape{6}, {8.0f, 5.0f, 4.0f, 1.0f, 1.0f, 1.0f});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_experimental_detectron_group_norm) {
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                             SERIALIZED_ZOO,
                             "onnx/org.openvinotoolkit/experimental_detectron/group_norm.onnx"));

    auto test_case = test::TestCase(function, s_device);
    Shape shape{2, 8, 2, 2};
    int size = shape_size(shape);
    std::vector<float> data(size);
    std::iota(data.begin(), data.end(), 0);
    std::vector<float> output = {
        -0.52752507, -0.09108937, 0.3453464, 0.78178215, 2.4364357, 3.309307,  4.1821785, 5.05505,
        -1.5825753,  -0.27326822, 1.0360391, 2.3453465,  4.8728714, 6.618614,  8.364357,  10.1101,
        -2.6376252,  -0.45544672, 1.726732,  3.9089108,  7.309307,  9.927921,  12.546536, 15.165151,
        -3.6926756,  -0.6376257,  2.4174247, 5.472475,   9.745743,  13.237228, 16.728714, 20.2202,
        -0.52752507, -0.09108937, 0.3453464, 0.78178215, 2.4364357, 3.309307,  4.1821785, 5.05505,
        -1.5825753,  -0.27326822, 1.0360391, 2.3453465,  4.8728714, 6.618614,  8.364357,  10.1101,
        -2.6376252,  -0.45544672, 1.726732,  3.9089108,  7.309307,  9.927921,  12.546536, 15.165151,
        -3.6926756,  -0.6376257,  2.4174247, 5.472475,   9.745743,  13.237228, 16.728714, 20.2202,
    };

    test_case.add_input<float>(data);
    test_case.add_expected_output<float>(shape, output);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_experimental_detectron_prior_grid_generator) {
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                             SERIALIZED_ZOO,
                             "onnx/org.openvinotoolkit/experimental_detectron/prior_grid_generator.onnx"));

    auto test_case = test::TestCase(function, s_device);

    std::vector<float> priors(shape_size(Shape{3, 4}));
    std::iota(priors.begin(), priors.end(), 0);

    std::vector<float> feature_map(shape_size(Shape{1, 1, 1, 3}));
    std::iota(feature_map.begin(), feature_map.end(), 0);

    std::vector<float> im_data(shape_size(Shape{1, 3, 4, 7}));
    std::iota(im_data.begin(), im_data.end(), 0);

    test_case.add_input<float>(priors);
    test_case.add_input<float>(feature_map);
    test_case.add_input<float>(im_data);

    test_case.add_expected_output<float>(Shape{9, 4},
                                         {2,  3, 4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 6,  3, 8,  5,  10, 7,
                                          12, 9, 14, 11, 16, 13, 10, 3, 12, 5,  14, 7,  16, 9, 18, 11, 20, 13});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_experimental_detectron_roi_feature_extractor) {
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                             SERIALIZED_ZOO,
                             "onnx/org.openvinotoolkit/experimental_detectron/roi_feature_extractor.onnx"));

    auto test_case = test::TestCase(function, s_device);

    std::vector<float> rois(shape_size(Shape{2, 4}));
    std::iota(rois.begin(), rois.end(), 0);

    std::vector<float> pyramid_layer_0(shape_size(Shape{1, 2, 2, 3}));
    std::iota(pyramid_layer_0.begin(), pyramid_layer_0.end(), 0);

    test_case.add_input<float>(rois);
    test_case.add_input<float>(pyramid_layer_0);

    test_case.add_expected_output<float>(Shape{2, 2, 3, 3},
                                         {1.416666746139526367,
                                          1.750000119209289551,
                                          2.083333492279052734,
                                          2.416666746139526367,
                                          2.75,
                                          3.083333492279052734,
                                          3.166666507720947266,
                                          3.5,
                                          3.833333492279052734,
                                          7.416666507720947266,
                                          7.75,
                                          8.083333015441894531,
                                          8.416666984558105469,
                                          8.75,
                                          9.083333969116210938,
                                          9.166666030883789062,
                                          9.5,
                                          9.833333969116210938,
                                          4.166666984558105469,
                                          4.5,
                                          4.833333492279052734,
                                          4.166666984558105469,
                                          4.5,
                                          4.833333492279052734,
                                          2.083333492279052734,
                                          2.25,
                                          2.416666746139526367,
                                          10.16666603088378906,
                                          10.5,
                                          10.83333206176757812,
                                          10.16666603088378906,
                                          10.5,
                                          10.83333206176757812,
                                          5.083333015441894531,
                                          5.25,
                                          5.416666507720947266});

    test_case.add_expected_output<float>(Shape{2, 4}, {0, 1, 2, 3, 4, 5, 6, 7});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_experimental_detectron_topk_rios) {
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                             SERIALIZED_ZOO,
                             "onnx/org.openvinotoolkit/experimental_detectron/topk_rios.onnx"));

    auto test_case = test::TestCase(function, s_device);

    test_case.add_input<float>({1.0f, 1.0f, 3.0f, 4.0f, 2.0f, 1.0f, 5.0f, 7.0f});
    test_case.add_input<float>({0.5f, 0.3f});

    test_case.add_expected_output<float>(Shape{1, 4}, {1, 1, 3, 4});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_deformable_conv_2d) {
    auto function =
        onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                            SERIALIZED_ZOO,
                                                            "onnx/org.openvinotoolkit/deformable_conv_2d.onnx"));

    auto test_case = test::TestCase(function, s_device);

    // data
    test_case.add_input<float>(
        {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f});

    // deformations
    test_case.add_input<float>({0.5f, -0.5f, 0.0f, 1.0f});

    test_case.add_expected_output<float>(
        Shape{1, 1, 3, 3},
        {4.5999999f, 5.2000003f, 6.4000001f, 8.4000006f, 9.8000002f, 9.6999998f, 11.5f, 13.4000006f, 14.3999996f});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_generate_proposals) {
    auto function =
        onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                            SERIALIZED_ZOO,
                                                            "onnx/org.openvinotoolkit/generate_proposals.onnx"));

    auto test_case = test::TestCase(function, s_device);

    // scores
    test_case.add_input<float>(
        Shape{1, 3, 2, 6},
        {0.56637216, 0.90457034, 0.69827306, 0.4353543,  0.47985056, 0.42658508, 0.14516132, 0.08081771, 0.1799732,
         0.9229515,  0.42420176, 0.50857586, 0.82664067, 0.4972319,  0.3752427,  0.56731623, 0.18241242, 0.33252355,
         0.30608943, 0.6572437,  0.69185436, 0.88646156, 0.36985755, 0.5590753,  0.5256446,  0.03342898, 0.1344396,
         0.68642473, 0.37953874, 0.32575172, 0.21108444, 0.5661886,  0.45378175, 0.62126315, 0.26799858, 0.37272978});
    // deltas
    test_case.add_input<float>(
        Shape{1, 12, 2, 6},
        {0.5337073,  0.86607957, 0.55151343, 0.21626699, 0.4462629,  0.03985678, 0.5157072,  0.9932138,  0.7565954,
         0.43803605, 0.802818,   0.14834064, 0.53932905, 0.14314,    0.3817048,  0.95075196, 0.05516243, 0.2567484,
         0.25508744, 0.77438325, 0.43561,    0.2094628,  0.8299043,  0.44982538, 0.95615596, 0.5651084,  0.11801951,
         0.05352486, 0.9774733,  0.14439464, 0.62644225, 0.14370479, 0.54161614, 0.557915,   0.53102225, 0.0840179,
         0.7249888,  0.9843559,  0.5490522,  0.53788143, 0.822474,   0.3278008,  0.39688024, 0.3286012,  0.5117038,
         0.04743988, 0.9408995,  0.29885054, 0.81039643, 0.85277915, 0.06807619, 0.86430097, 0.36225632, 0.16606331,
         0.5401001,  0.7541649,  0.11998601, 0.5131829,  0.40606487, 0.327888,   0.27721855, 0.6378373,  0.22795396,
         0.4961256,  0.3215895,  0.15607187, 0.14782153, 0.8908137,  0.8835288,  0.834191,   0.29907143, 0.7983525,
         0.755875,   0.30837986, 0.0839176,  0.26624718, 0.04371626, 0.09472824, 0.20689541, 0.37622106, 0.1083321,
         0.1342548,  0.05815459, 0.7676379,  0.8105144,  0.92348766, 0.26761323, 0.7183306,  0.8947588,  0.19020908,
         0.42731014, 0.7473663,  0.85775334, 0.9340091,  0.3278848,  0.755993,   0.05307213, 0.39705503, 0.21003333,
         0.5625373,  0.66188884, 0.80521655, 0.6125863,  0.44678232, 0.97802377, 0.0204936,  0.02686367, 0.7390654,
         0.74631,    0.58399844, 0.5988792,  0.37413648, 0.5946692,  0.6955776,  0.36377597, 0.7891322,  0.40900692,
         0.99139464, 0.50169915, 0.41435778, 0.17142445, 0.26761186, 0.31591868, 0.14249913, 0.12919712, 0.5418711,
         0.6523203,  0.50259084, 0.7379765,  0.01171071, 0.94423133, 0.00841132, 0.97486794, 0.2921785,  0.7633071,
         0.88477814, 0.03563205, 0.50833166, 0.01354555, 0.535081,   0.41366324, 0.0694767,  0.9944055,  0.9981207});
    // im_info
    test_case.add_input<float>(Shape{1, 3}, {200, 200, 0});
    // anchors
    test_case.add_input<float>(Shape{3, 4}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});

    test_case.add_expected_output<float>(
        Shape{6, 4},
        {0.12904608, 1.3703424, 3.6230984, 3.4675088, 0.9725206, 0.,        4.4917974, 4.9623675,
         4.882682,   5.1236916, 7.1700497, 10.213073, 4.4913187, 4.305372,  8.750267,  8.803502,
         0.9777608,  1.0317986, 3.228293,  4.495021,  4.125554,  5.4091997, 6.35439,   10.124915});
    test_case.add_expected_output<float>(Shape{6},
                                         {0.9229515, 0.90457034, 0.88646156, 0.82664067, 0.69827306, 0.69185436});
    test_case.add_expected_output<int64_t>(Shape{1}, {6});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_generate_proposals_batch) {
    auto function =
        onnx_import::import_onnx_model(file_util::path_join(CommonTestUtils::getExecutableDirectory(),
                                                            SERIALIZED_ZOO,
                                                            "onnx/org.openvinotoolkit/generate_proposals_batch2.onnx"));

    auto test_case = test::TestCase(function, s_device);

    // scores
    test_case.add_input<float>(Shape{2, 3, 2, 3}, {5, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 7, 1, 1, 1, 1,
                                                   1, 1, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 8, 1});
    // deltas
    test_case.add_input<float>(Shape{2, 12, 2, 3}, std::vector<float>(144, 1));
    // im_info
    test_case.add_input<float>(Shape{2, 3}, {1, 1, 0, 1, 1, 0});
    // anchors
    test_case.add_input<float>(Shape{3, 4}, std::vector<float>(12, 1));

    test_case.add_expected_output<float>(Shape{10, 4}, std::vector<float>(40, 1));
    test_case.add_expected_output<float>(Shape{10}, {7, 5, 3, 1, 1, 8, 4, 2, 1, 1});
    test_case.add_expected_output<int64_t>(Shape{2}, {5, 5});
    test_case.run();
}
