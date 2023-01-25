# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#! [dataset]
import nncf
import torch

calibration_loader = torch.utils.data.DataLoader(...)

def transform_fn(data_item):
    images, _ = data_item
    return images.numpy()

calibration_dataset = nncf.Dataset(calibration_loader, transform_fn)
#! [dataset]

#! [quantization]
model = ... # openvino.runtime.Model object

quantized_model = nncf.quantize(model, calibration_dataset)
#! [quantization]
