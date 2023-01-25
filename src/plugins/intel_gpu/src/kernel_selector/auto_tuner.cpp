﻿// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "auto_tuner.h"
#include <iostream>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <string>
#include "istreamwrapper.h"
#include "stringbuffer.h"
#include "prettywriter.h"
#include <memory>
#include <utility>
#include <tuple>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <SetupAPI.h>
#include <devguid.h>
#include <cstring>
#else
#include <unistd.h>
#include <limits.h>
#include <link.h>
#include <dlfcn.h>
#endif

namespace kernel_selector {

TuningCache::TuningCache(const std::string& cacheFilePath, bool createMode)
    : cache(), needsSave(false) {
    // Read cache file
    std::ifstream tuningFile(cacheFilePath);

    if (tuningFile && tuningFile.good()) {
        std::stringstream buffer;
        buffer << tuningFile.rdbuf();
        cache.Parse(buffer.str().c_str());
    } else {
        if (!createMode) {
            throw std::runtime_error("Tuning file: " + cacheFilePath +
                                     " could not be read! Must provide a valid cache file in USE_CACHE mode.");
        }

        cache.SetObject();
        needsSave = true;
    }

    if (cache.IsNull()) {
        cache.SetObject();
    } else if (!cache.IsObject()) {
        throw std::runtime_error("Tuning file: " + cacheFilePath + " has incorrect format.");
    }

    auto cacheObj = cache.GetObject();

    // Update to new format with version markers
    if (!cacheObj.HasMember(version2Marker)) {
        auto newName = rapidjson::Value(version2Marker, cache.GetAllocator());
        auto newObj = rapidjson::Value(rapidjson::Type::kObjectType);
        cacheObj.AddMember(newName, newObj, cache.GetAllocator());
    }

    bool needsV1 = false;
    for (auto& member : cacheObj) {
        std::string nameStr = member.name.GetString();
        if (nameStr != version1Marker && nameStr != version2Marker) {
            needsV1 = true;
        }
    }

    if (needsV1) {
        if (!cacheObj.HasMember(version1Marker)) {
            auto newName = rapidjson::Value(version1Marker, cache.GetAllocator());
            auto newObj = rapidjson::Value(rapidjson::Type::kObjectType);
            cacheObj.AddMember(newName, newObj, cache.GetAllocator());
        }

        for (auto it = cacheObj.begin(); it != cacheObj.end();) {
            auto& member = *it;
            std::string nameStr = member.name.GetString();
            if (nameStr != version1Marker && nameStr != version2Marker) {
                auto newName = rapidjson::Value(rapidjson::Type::kStringType);
                auto newValue = rapidjson::Value(rapidjson::Type::kObjectType);
                newName.Swap(member.name);
                newValue.Swap(member.value);
                cache[version1Marker].AddMember(newName, newValue, cache.GetAllocator());
                it = cacheObj.EraseMember(it);
            } else {
                it++;
            }
        }
        needsSave = true;
    }
    //
}

TuningCache::TuningCache()
    : cache(), needsSave(true) {
    cache.SetObject();
    auto v2Name = rapidjson::Value(version2Marker, cache.GetAllocator());
    auto v2Obj = rapidjson::Value(rapidjson::Type::kObjectType);
    cache.AddMember(v2Name, v2Obj, cache.GetAllocator());
}

TuningCache::Entry TuningCache::LoadKernel(const Params& params, bool update) {
    return LoadKernel(params, params.engineInfo.computeUnitsCount, update);
}

TuningCache::Entry TuningCache::LoadKernel(const Params& params, uint32_t computeUnitsCount, bool update) {
    bool oldVersion = false;
    // Try to load from version 2
    auto result = LoadKernel_v2(params, computeUnitsCount);
    // Try to load from version 1
    if (std::get<0>(result).empty() || update) {
        auto result_v1 = LoadKernel_v1(params, computeUnitsCount);
        oldVersion = !std::get<0>(result_v1).empty();
        if (oldVersion && std::get<0>(result).empty()) {
            result = result_v1;
        }
    }
    // Move cache from old version to newer
    if (oldVersion && update) {
        StoreKernel(params, computeUnitsCount, std::get<0>(result), std::get<1>(result));
    }

    return result;
}

TuningCache::Entry TuningCache::LoadKernel_v1(const Params& params, uint32_t computeUnitsCount) {
    Entry result = std::make_tuple<std::string, int>("", 0);

    auto hashStr = std::to_string(create_hash(params.to_string()));
    auto computeUnitsStr = std::to_string(computeUnitsCount);

    auto v1It = cache.FindMember(version1Marker);
    if (v1It == cache.MemberEnd())
        return result;

    auto computeUnitsIt = v1It->value.FindMember(computeUnitsStr.c_str());
    if (computeUnitsIt == v1It->value.MemberEnd())
        return result;

    auto hashIt = computeUnitsIt->value.FindMember(hashStr.c_str());
    if (hashIt == computeUnitsIt->value.MemberEnd())
        return result;

    auto& prog = hashIt->value;
    return std::make_tuple(prog[0].GetString(), prog[1].GetInt());
}

TuningCache::Entry TuningCache::LoadKernel_v2(const Params& params, uint32_t computeUnitsCount) {
    Entry result = std::make_tuple<std::string, int>("", 0);

    auto kTypeStr = toString(params.GetType());
    auto paramStr = params.to_cache_string_v2();
    auto computeUnitsStr = std::to_string(computeUnitsCount);

    auto v2It = cache.FindMember(version2Marker);
    if (v2It == cache.MemberEnd())
        return result;

    auto computeUnitsIt = v2It->value.FindMember(computeUnitsStr.c_str());
    if (computeUnitsIt == v2It->value.MemberEnd())
        return result;

    auto kTypeIt = computeUnitsIt->value.FindMember(kTypeStr.c_str());
    if (kTypeIt == computeUnitsIt->value.MemberEnd())
        return result;

    auto paramIt = kTypeIt->value.FindMember(paramStr.c_str());
    if (paramIt == kTypeIt->value.MemberEnd())
        return result;

    auto& prog = paramIt->value;
    return std::make_tuple(prog[0].GetString(), prog[1].GetInt());
}

void TuningCache::StoreKernel(const Params& params, const std::string& implementationName, int tuneIndex) {
    StoreKernel(params, params.engineInfo.computeUnitsCount, implementationName, tuneIndex);
}

void TuningCache::StoreKernel(const Params& params, uint32_t computeUnitsCount, const std::string& implementationName, int tuneIndex) {
    auto kTypeStr = toString(params.GetType());
    auto paramStr = params.to_cache_string_v2();
    auto computeUnitsStr = std::to_string(computeUnitsCount);
    auto& v2Cache = cache[version2Marker];

    if (!v2Cache.HasMember(computeUnitsStr.c_str())) {
        auto newName = rapidjson::Value(computeUnitsStr.c_str(), cache.GetAllocator());
        auto newObj = rapidjson::Value(rapidjson::Type::kObjectType);
        v2Cache.AddMember(newName, newObj, cache.GetAllocator());
    }

    if (!v2Cache[computeUnitsStr.c_str()].HasMember(kTypeStr.c_str())) {
        auto newName = rapidjson::Value(kTypeStr.c_str(), cache.GetAllocator());
        auto newObj = rapidjson::Value(rapidjson::Type::kObjectType);
        v2Cache[computeUnitsStr.c_str()].AddMember(newName, newObj, cache.GetAllocator());
    }

    auto& deviceCache = v2Cache[computeUnitsStr.c_str()][kTypeStr.c_str()];

    auto paramName = rapidjson::Value(paramStr.c_str(), cache.GetAllocator());
    auto implDetails = rapidjson::Value(rapidjson::Type::kArrayType);
    auto implName = rapidjson::Value(implementationName.c_str(), cache.GetAllocator());
    auto implIndex = rapidjson::Value(tuneIndex);
    implDetails.PushBack(implName, cache.GetAllocator());
    implDetails.PushBack(implIndex, cache.GetAllocator());

    deviceCache.AddMember(paramName, implDetails, cache.GetAllocator());

    // Remove from old version if present
    RemoveKernel_v1(params, computeUnitsCount);

    needsSave = true;
}

void TuningCache::RemoveKernel(const Params& params) {
    bool removed = false;
    // Remove from version 2
    removed |= RemoveKernel_v2(params, params.engineInfo.computeUnitsCount);
    // Remove from version 1
    removed |= RemoveKernel_v1(params, params.engineInfo.computeUnitsCount);

    needsSave |= removed;
}

bool TuningCache::RemoveKernel_v1(const Params& params, uint32_t computeUnitsCount) {
    auto hashStr = std::to_string(create_hash(params.to_string()));
    auto computeUnitsStr = std::to_string(computeUnitsCount);

    auto v1It = cache.FindMember(version1Marker);
    if (v1It == cache.MemberEnd())
        return false;

    auto computeUnitsIt = v1It->value.FindMember(computeUnitsStr.c_str());
    if (computeUnitsIt == v1It->value.MemberEnd())
        return false;

    auto hashIt = computeUnitsIt->value.FindMember(hashStr.c_str());
    if (hashIt == computeUnitsIt->value.MemberEnd())
        return false;

    computeUnitsIt->value.RemoveMember(hashIt);
    return true;
}

bool TuningCache::RemoveKernel_v2(const Params& params, uint32_t computeUnitsCount) {
    auto kTypeStr = toString(params.GetType());
    auto paramStr = params.to_cache_string_v2();
    auto computeUnitsStr = std::to_string(computeUnitsCount);

    auto v2It = cache.FindMember(version2Marker);
    if (v2It == cache.MemberEnd())
        return false;

    auto computeUnitsIt = v2It->value.FindMember(computeUnitsStr.c_str());
    if (computeUnitsIt == v2It->value.MemberEnd())
        return false;

    auto kTypeIt = computeUnitsIt->value.FindMember(kTypeStr.c_str());
    if (kTypeIt == computeUnitsIt->value.MemberEnd())
        return false;

    auto paramIt = kTypeIt->value.FindMember(paramStr.c_str());
    if (paramIt == kTypeIt->value.MemberEnd())
        return false;

    kTypeIt->value.RemoveMember(paramIt);
    return true;
}

void TuningCache::Save(const std::string& cacheFilePath) {
    std::ofstream cachedKernelsFile(cacheFilePath);
    rapidjson::StringBuffer buffer(0, 1024);
    rapidjson::PrettyWriter<rapidjson::StringBuffer> writer(buffer);
    writer.SetFormatOptions(rapidjson::PrettyFormatOptions::kFormatSingleLineArray);
    cache.Accept(writer);
    auto temp = buffer.GetString();
    cachedKernelsFile << temp;
    cachedKernelsFile.close();

    needsSave = false;
}

std::tuple<std::string, int> AutoTuner::LoadKernelOffline(const Params& params) {
    std::lock_guard<std::mutex> lock(mutex);
    static const uint32_t defaultComputeUnits = 24;
    TuningCache* deviceCache = TuningCache::get();
    if (!deviceCache)
        return {};
    auto result = deviceCache->LoadKernel(params, false);
    if (std::get<0>(result).empty() && params.engineInfo.computeUnitsCount != defaultComputeUnits) {
        result = deviceCache->LoadKernel(params, defaultComputeUnits);
    }
    return result;
}

TuningCache* TuningCache::get() {
    static std::mutex m;
    static std::shared_ptr<TuningCache> cache_instance = nullptr;
    std::lock_guard<std::mutex> lock(m);
    std::string path = "cache.json";
#ifdef _WIN32
    char module_path[MAX_PATH];
    HMODULE hm = NULL;
    GetModuleHandleEx(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
        (LPCSTR)&TuningCache::get,
        &hm);
    GetModuleFileName(hm, module_path, sizeof(module_path));
    std::string bin_path(module_path);
    path = bin_path.substr(0, bin_path.find_last_of("\\")) + "\\cache.json";
#else
    const char* device_info_failed_msg = "Device lookup failed";
    Dl_info dl_info;
    dladdr((void*)(device_info_failed_msg), &dl_info);  // NOLINT
    std::string bin_path(dl_info.dli_fname);
    path = bin_path.substr(0, bin_path.find_last_of("/")) + "/cache.json";
#endif

    if (!cache_instance) {
        try {
            cache_instance = std::make_shared<kernel_selector::TuningCache>(path, false);
        } catch (...) {
            cache_instance = std::make_shared<kernel_selector::TuningCache>();
        }
    }

    return cache_instance.get();
}

}  // namespace kernel_selector
