# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set_property(GLOBAL PROPERTY JOB_POOLS four_jobs=4)

if(ENABLE_OV_ONNX_FRONTEND)
    # if requirements are not installed automatically, we need to checks whether they are here
    ov_check_pip_packages(REQUIREMENTS_FILE "${OpenVINO_SOURCE_DIR}/src/frontends/onnx/tests/requirements.txt"
                          RESULT_VAR onnx_FOUND
                          WARNING_MESSAGE "ONNX frontend tests will be skipped"
                          MESSAGE_MODE WARNING)
endif()

function(ov_model_convert SRC DST OUT)
    set(onnx_gen_script ${OpenVINO_SOURCE_DIR}/src/frontends/onnx/tests/onnx_prototxt_converter.py)

    file(GLOB_RECURSE xml_models RELATIVE "${SRC}" "${SRC}/*.xml")
    file(GLOB_RECURSE bin_models RELATIVE "${SRC}" "${SRC}/*.bin")
    file(GLOB_RECURSE onnx_models RELATIVE "${SRC}" "${SRC}/*.onnx")
    file(GLOB_RECURSE data_models RELATIVE "${SRC}" "${SRC}/*.data")

    if(onnx_FOUND)
        file(GLOB_RECURSE prototxt_models RELATIVE "${SRC}" "${SRC}/*.prototxt")
    endif()

    foreach(in_file IN LISTS prototxt_models xml_models bin_models onnx_models data_models)
        get_filename_component(ext "${in_file}" EXT)
        get_filename_component(rel_dir "${in_file}" DIRECTORY)
        get_filename_component(name_we "${in_file}" NAME_WE)
        set(model_source_dir "${SRC}/${rel_dir}")

        if(ext STREQUAL ".prototxt")
            # convert model
            set(rel_out_name "${name_we}.onnx")
            if(rel_dir)
                set(rel_out_name "${rel_dir}/${rel_out_name}")
            endif()
        else()
            # copy as is
            set(rel_out_name "${in_file}")
        endif()

        set(full_out_name "${DST}/${rel_out_name}")
        file(MAKE_DIRECTORY "${DST}/${rel_dir}")

        if(ext STREQUAL ".prototxt")
            # convert .prototxt models to .onnx binary
            add_custom_command(OUTPUT ${full_out_name}
                COMMAND ${PYTHON_EXECUTABLE} ${onnx_gen_script}
                    "${SRC}/${in_file}" ${full_out_name}
                DEPENDS ${onnx_gen_script} "${SRC}/${in_file}"
                COMMENT "Generate ${rel_out_name}"
                JOB_POOL four_jobs
                WORKING_DIRECTORY "${model_source_dir}")
        else()
            add_custom_command(OUTPUT ${full_out_name}
                COMMAND "${CMAKE_COMMAND}" -E copy_if_different
                    "${SRC}/${in_file}" ${full_out_name}
                DEPENDS ${onnx_gen_script} "${SRC}/${in_file}"
                COMMENT "Copy ${rel_out_name}"
                JOB_POOL four_jobs
                WORKING_DIRECTORY "${model_source_dir}")
        endif()
        list(APPEND files "${full_out_name}")
    endforeach()

    set(${OUT} ${files} PARENT_SCOPE)
endfunction()

ov_model_convert("${CMAKE_CURRENT_SOURCE_DIR}/src/core/tests"
                 "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/test_model_zoo/core"
                  core_tests_out_files)

set(rel_path "src/tests/functional/plugin/shared/models")
ov_model_convert("${OpenVINO_SOURCE_DIR}/${rel_path}"
                 "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/test_model_zoo/func_tests/models"
                 ft_out_files)

set(rel_path "src/frontends/onnx/tests/models")
ov_model_convert("${OpenVINO_SOURCE_DIR}/${rel_path}"
                 "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/test_model_zoo/onnx"
                 onnx_fe_out_files)

if(ENABLE_TESTS)
    add_custom_target(test_model_zoo DEPENDS ${core_tests_out_files}
                                             ${ft_out_files}
                                             ${onnx_fe_out_files})

    if (ENABLE_OV_PADDLE_FRONTEND)
        add_dependencies(test_model_zoo paddle_test_models)
    endif()

    install(DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/test_model_zoo"
            DESTINATION tests COMPONENT tests EXCLUDE_FROM_ALL)

    set(TEST_MODEL_ZOO "./test_model_zoo" CACHE PATH "Path to test model zoo")
endif()
