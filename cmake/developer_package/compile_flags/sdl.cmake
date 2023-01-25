# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if(UNIX)
    set(IE_C_CXX_FLAGS "${IE_C_CXX_FLAGS} -Wformat -Wformat-security")
    if (NOT ENABLE_SANITIZER)
        if(EMSCRIPTEN)
            # emcc does not support fortification, see:
            # https://stackoverflow.com/questions/58854858/undefined-symbol-stack-chk-guard-in-libopenh264-so-when-building-ffmpeg-wit
        else()
            # ASan does not support fortification https://github.com/google/sanitizers/issues/247
            set(IE_C_CXX_FLAGS "${IE_C_CXX_FLAGS} -D_FORTIFY_SOURCE=2")
        endif()
    endif()
    if(NOT APPLE)
        set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} -pie")
    endif()

    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        set(IE_LINKER_FLAGS "${IE_LINKER_FLAGS} -z noexecstack -z relro -z now")
        set(IE_C_CXX_FLAGS "${IE_C_CXX_FLAGS} -fno-strict-overflow -fno-delete-null-pointer-checks -fwrapv")
        if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 4.9)
            set(IE_C_CXX_FLAGS "${IE_C_CXX_FLAGS} -fstack-protector-all")
        else()
            set(IE_C_CXX_FLAGS "${IE_C_CXX_FLAGS} -fstack-protector-strong")
        endif()
        if (NOT ENABLE_SANITIZER)
            set(IE_C_CXX_FLAGS "${IE_C_CXX_FLAGS} -s")
        endif()
    elseif(OV_COMPILER_IS_CLANG)
        if(EMSCRIPTEN)
            # emcc does not support fortification 
            # https://stackoverflow.com/questions/58854858/undefined-symbol-stack-chk-guard-in-libopenh264-so-when-building-ffmpeg-wit
        else()
            set(IE_C_CXX_FLAGS "${IE_C_CXX_FLAGS} -fstack-protector-all")
        endif()
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
        if (NOT ENABLE_SANITIZER)
            set(IE_C_CXX_FLAGS "${IE_C_CXX_FLAGS} -Wl,--strip-all")
        endif()
        set(IE_C_CXX_FLAGS "${IE_C_CXX_FLAGS} -fstack-protector-strong")
        set(IE_LINKER_FLAGS "${IE_LINKER_FLAGS} -z noexecstack -z relro -z now")
    endif()
else()
    if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        set(IE_C_CXX_FLAGS "${IE_C_CXX_FLAGS} /sdl")
    endif()
    set(IE_C_CXX_FLAGS "${IE_C_CXX_FLAGS} /guard:cf")
    if(ENABLE_INTEGRITYCHECK)
        set(CMAKE_SHARED_LINKER_FLAGS_RELEASE "${CMAKE_SHARED_LINKER_FLAGS_RELEASE} /INTEGRITYCHECK")
    endif()
endif()

set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} ${IE_C_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${IE_C_CXX_FLAGS}")
set(CMAKE_SHARED_LINKER_FLAGS_RELEASE "${CMAKE_SHARED_LINKER_FLAGS_RELEASE} ${IE_LINKER_FLAGS}")
set(CMAKE_MODULE_LINKER_FLAGS_RELEASE "${CMAKE_MODULE_LINKER_FLAGS_RELEASE} ${IE_LINKER_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} ${IE_LINKER_FLAGS}")
