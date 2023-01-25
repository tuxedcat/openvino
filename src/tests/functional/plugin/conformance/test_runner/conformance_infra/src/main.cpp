// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <signal.h>
#ifdef _WIN32
#include <process.h>
#endif

#include "gtest/gtest.h"

#include "common_test_utils/file_utils.hpp"
#include "functional_test_utils/skip_tests_config.hpp"
#include "functional_test_utils/summary/environment.hpp"

#include "read_ir_test/read_ir.hpp"
#include "gflag_config.hpp"
#include "conformance.hpp"

#include "common_test_utils/crash_handler.hpp"

using namespace ov::test::conformance;

int main(int argc, char* argv[]) {
    // Workaround for Gtest + Gflag
    std::vector<char*> argv_gflags_vec;
    int argc_gflags = 0;
    for (int i = 0; i < argc; ++i) {
        std::string arg(argv[i]);
        if (arg.find("--gtest") == std::string::npos) {
            argv_gflags_vec.emplace_back(argv[i]);
            argc_gflags++;
        }
    }
    char** argv_gflags = argv_gflags_vec.data();

    // ---------------------------Parsing and validation of input args--------------------------------------
    gflags::ParseCommandLineNonHelpFlags(&argc_gflags, &argv_gflags, true);
    if (FLAGS_h) {
        showUsage();
        return 0;
    }
    if (FLAGS_extend_report && FLAGS_report_unique_name) {
        throw std::runtime_error("Using mutually exclusive arguments: --extend_report and --report_unique_name");
    }

    FuncTestUtils::SkipTestsConfig::disable_tests_skipping = FLAGS_disable_test_config;
    ov::test::utils::OpSummary::setExtendReport(FLAGS_extend_report);
    ov::test::utils::OpSummary::setExtractBody(FLAGS_extract_body);
    ov::test::utils::OpSummary::setSaveReportWithUniqueName(FLAGS_report_unique_name);
    ov::test::utils::OpSummary::setOutputFolder(FLAGS_output_folder);
    ov::test::utils::OpSummary::setSaveReportTimeout(FLAGS_save_report_timeout);
    {
        auto &apiSummary = ov::test::utils::ApiSummary::getInstance();
        apiSummary.setDeviceName(FLAGS_device);
    }
    if (FLAGS_shape_mode == std::string("static")) {
        ov::test::subgraph::shapeMode = ov::test::subgraph::ShapeMode::STATIC;
    } else if (FLAGS_shape_mode == std::string("dynamic")) {
        ov::test::subgraph::shapeMode = ov::test::subgraph::ShapeMode::DYNAMIC;
    } else if (FLAGS_shape_mode != std::string("")) {
        throw std::runtime_error("Incorrect value for `--shape_mode`. Should be `dynamic`, `static` or ``. Current value is `" + FLAGS_shape_mode + "`");
    }

    CommonTestUtils::CrashHandler::SetUpTimeout(FLAGS_test_timeout);

    // ---------------------------Initialization of Gtest env -----------------------------------------------
    ov::test::conformance::targetDevice = FLAGS_device.c_str();
    ov::test::conformance::IRFolderPaths = CommonTestUtils::splitStringByDelimiter(FLAGS_input_folders);
    if (!FLAGS_plugin_lib_name.empty()) {
        ov::test::conformance::targetPluginName = FLAGS_plugin_lib_name.c_str();
    }
    if (!FLAGS_skip_config_path.empty()) {
        ov::test::conformance::disabledTests = CommonTestUtils::readListFiles(
                CommonTestUtils::splitStringByDelimiter(FLAGS_skip_config_path));
    }
    if (!FLAGS_config_path.empty()) {
        ov::test::conformance::pluginConfig = ov::test::conformance::readPluginConfig(FLAGS_config_path);
    }

    ::testing::InitGoogleTest(&argc, argv);
    ::testing::AddGlobalTestEnvironment(new ov::test::utils::TestEnvironment);

    auto exernalSignalHandler = [](int errCode) {
        std::cerr << "Unexpected application crash with code: " << errCode << std::endl;

        auto& op_summary = ov::test::utils::OpSummary::getInstance();
        auto& api_summary = ov::test::utils::ApiSummary::getInstance();
        op_summary.saveReport();
        api_summary.saveReport();

        // set default handler for crash
        signal(SIGABRT, SIG_DFL);
        signal(SIGSEGV, SIG_DFL);
        signal(SIGINT, SIG_DFL);
        signal(SIGTERM, SIG_DFL);

        exit(1);
    };

    // killed by external
    signal(SIGINT, exernalSignalHandler);
    signal(SIGTERM , exernalSignalHandler);
    signal(SIGSEGV, exernalSignalHandler);
    signal(SIGABRT, exernalSignalHandler);
    return RUN_ALL_TESTS();
}
