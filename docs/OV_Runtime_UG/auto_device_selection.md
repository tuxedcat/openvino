# Automatic Device Selection {#openvino_docs_OV_UG_supported_plugins_AUTO}

@sphinxdirective

.. toctree::
   :maxdepth: 1
   :hidden:

   Debugging Auto-Device Plugin <openvino_docs_OV_UG_supported_plugins_AUTO_debugging>

@endsphinxdirective

This article introduces how Automatic Device Selection works and how to use it for inference.

## <a name="how-auto-works"></a> How AUTO Works

The Automatic Device Selection mode, or AUTO for short, uses a "virtual" or a "proxy" device, 
which does not bind to a specific type of hardware, but rather selects the processing unit for inference automatically. 
It detects available devices, picks the one best-suited for the task, and configures its optimization settings. 
This way, you can write the application once and deploy it anywhere.

The selection also depends on your performance requirements, defined by the “hints” configuration API, as well as device priority list limitations, if you choose to exclude some hardware from the process.

The logic behind the choice is as follows: 
1. Check what supported devices are available. 
2. Check precisions of the input model (for detailed information on precisions read more on the `ov::device::capabilities`) 
3. Select the highest-priority device capable of supporting the given model, as listed in the table below. 
4. If model’s precision is FP32 but there is no device capable of supporting it, offload the model to a device supporting FP16. 

@sphinxdirective
+----------+------------------------------------------------------+-------------------------------------+
| Device   || Supported                                           || Supported                          |
| Priority || Device                                              || model precision                    |
+==========+======================================================+=====================================+
| 1        || dGPU                                                | FP32, FP16, INT8, BIN               |
|          || (e.g. Intel® Iris® Xe MAX)                          |                                     |
+----------+------------------------------------------------------+-------------------------------------+
| 2        || iGPU                                                | FP32, FP16, BIN                     |
|          || (e.g. Intel® UHD Graphics 620 (iGPU))               |                                     |
+----------+------------------------------------------------------+-------------------------------------+
| 3        || Intel® CPU                                          | FP32, FP16, INT8, BIN               |
|          || (e.g. Intel® Core™ i7-1165G7)                       |                                     |
+----------+------------------------------------------------------+-------------------------------------+
@endsphinxdirective

To put it simply, when loading the model to the first device on the list fails, AUTO will try to load it to the next device in line, until one of them succeeds. 
What is important, **AUTO always starts inference with the CPU of the system**, as it provides very low latency and can start inference with no additional delays. 
While the CPU is performing inference, AUTO continues to load the model to the device best suited for the purpose and transfers the task to it when ready.
This way, the devices which are much slower in compiling models, GPU being the best example, do not impede inference at its initial stages.
For example, if you use a CPU and a GPU, the first-inference latency of AUTO will be better than that of using GPU alone.

Note that if you choose to exclude CPU from the priority list, it will be unable to support the initial model compilation stage.
     
![](../img/autoplugin_accelerate.svg)

This mechanism can be easily observed in the [Using AUTO with Benchmark app sample](#using-auto-with-openvino-samples-and-benchmark-app) section, showing how the first-inference latency (the time it takes to compile the model and perform the first inference) is reduced when using AUTO. For example: 

```sh
benchmark_app -m ../public/alexnet/FP32/alexnet.xml -d GPU -niter 128
```

```sh
benchmark_app -m ../public/alexnet/FP32/alexnet.xml -d AUTO -niter 128
```


@sphinxdirective
.. note::

   The longer the process runs, the closer realtime performance will be to that of the best-suited device.
@endsphinxdirective

## Using AUTO 

Following the OpenVINO™ naming convention, the Automatic Device Selection mode is assigned the label of “AUTO.” It may be defined with no additional parameters, resulting in defaults being used, or configured further with the following setup options: 

@sphinxdirective

+--------------------------------+----------------------------------------------------------------------+
| | Property                     | | Values and Description                                             |
+================================+======================================================================+
| | <device candidate list>      | | **Values**:                                                        |
| |                              | |       empty                                                        |
| |                              | |       `AUTO`                                                       |
| |                              | |       `AUTO: <device names>` (comma-separated, no spaces)          |
| |                              | |                                                                    |
| |                              | | Lists the devices available for selection.                         |
| |                              | | The device sequence will be taken as priority from high to low.    |
| |                              | | If not specified, `AUTO` will be used as default,                  |
| |                              | | and all devices will be "viewed" as candidates.                    |
+--------------------------------+----------------------------------------------------------------------+
| | `ov::device:priorities`      | | **Values**:                                                        |
| |                              | |       `<device names>` (comma-separated, no spaces)                |
| |                              | |                                                                    |
| |                              | | Specifies the devices for AUTO to select.                          |
| |                              | | The device sequence will be taken as priority from high to low.    |
| |                              | | This configuration is optional.                                    |
+--------------------------------+----------------------------------------------------------------------+
| | `ov::hint::performance_mode` | | **Values**:                                                        |
| |                              | |       `ov::hint::PerformanceMode::LATENCY`                         |
| |                              | |       `ov::hint::PerformanceMode::THROUGHPUT`                      |
| |                              | |       `ov::hint::PerformanceMode::CUMULATIVE_THROUGHPUT`           |
| |                              | |                                                                    |
| |                              | | Specifies the performance option preferred by the application.     |
+--------------------------------+----------------------------------------------------------------------+
| | `ov::hint::model_priority`   | | **Values**:                                                        |
| |                              | |       `ov::hint::Priority::HIGH`                                   |
| |                              | |       `ov::hint::Priority::MEDIUM`                                 |
| |                              | |       `ov::hint::Priority::LOW`                                    |
| |                              | |                                                                    |
| |                              | | Indicates the priority for a model.                                |
| |                              | | IMPORTANT: This property is not fully supported yet.               |
+--------------------------------+----------------------------------------------------------------------+

@endsphinxdirective

Inference with AUTO is configured similarly to when device plugins are used:
you compile the model on the plugin with configuration and execute inference.

### Device Candidates and Priority
The device candidate list enables you to customize the priority and limit the choice of devices available to AUTO. 
- If <device candidate list> is not specified, AUTO assumes all the devices present in the system can be used. 
- If `AUTO` without any device names is specified, AUTO assumes all the devices present in the system can be used, and will load the network to all devices and run inference based on their default priorities, from high to low.

To specify the priority of devices, enter the device names in the priority order (from high to low) in `AUTO: <device names>`, or use the `ov::device:priorities` property.

See the following code for using AUTO and specifying devices: 

@sphinxdirective

.. tab:: C++

    .. doxygensnippet:: docs/snippets/AUTO0.cpp
       :language: cpp
       :fragment: [part0]

.. tab:: Python

    .. doxygensnippet:: docs/snippets/ov_auto.py
       :language: python
       :fragment: [part0]

@endsphinxdirective

Note that OpenVINO Runtime lets you use “GPU” as an alias for “GPU.0” in function calls. More details on enumerating devices can be found in [Working with devices](supported_plugins/Device_Plugins.md).

#### Checking Available Devices

To check what devices are present in the system, you can use Device API, as listed below. For information on how to use it, see [Query device properties and configuration](supported_plugins/config_properties.md).

@sphinxdirective

.. tab:: C++   

   .. code-block:: sh

      ov::runtime::Core::get_available_devices() 

   See the Hello Query Device C++ Sample for reference.

.. tab:: Python

   .. code-block:: sh

      openvino.runtime.Core.available_devices

   See the Hello Query Device Python Sample for reference.

@endsphinxdirective

#### Excluding Devices from Device Candidate List

You can also exclude hardware devices from AUTO, for example, to reserve CPU for other jobs. AUTO will not use the device for inference then. To do that, add a minus sign (-) before CPU in `AUTO: <device names>`, as in the following example:

@sphinxdirective

.. tab:: C++

   .. code-block:: sh

      ov::CompiledModel compiled_model = core.compile_model(model, "AUTO:-CPU"); 

.. tab:: Python

   .. code-block:: sh

      compiled_model = core.compile_model(model=model, device_name="AUTO:-CPU")

@endsphinxdirective

AUTO will then query all available devices and remove CPU from the candidate list. 

Note that if you choose to exclude CPU from device candidate list, CPU will not be able to support the initial model compilation stage. See more information in [How AUTO Works](#how-auto-works).

### Performance Hints for AUTO
The `ov::hint::performance_mode` property enables you to specify a performance option for AUTO to be more efficient for particular use cases.

> **NOTE**: Currently, the `ov::hint` property is supported by CPU and GPU devices only.

#### THROUGHPUT
This option prioritizes high throughput, balancing between latency and power. It is best suited for tasks involving multiple jobs, such as inference of video feeds or large numbers of images.

> **NOTE**: If no performance hint is set explicitly, AUTO will set THROUGHPUT for devices that have not set `ov::device::properties`. For example, if you have both a CPU and a GPU in the system, this command `core.compile_model("AUTO", ov::device::properties("CPU", ov::enable_profiling(true)))` will set THROUGHPUT for the GPU only. No hint will be set for the CPU although it's the selected device.

#### LATENCY
This option prioritizes low latency, providing short response time for each inference job. It performs best for tasks where inference is required for a single input image, e.g. a medical analysis of an ultrasound scan image. It also fits the tasks of real-time or nearly real-time applications, such as an industrial robot's response to actions in its environment or obstacle avoidance for autonomous vehicles.

@sphinxdirective

.. _cumulative throughput:

@endsphinxdirective

#### CUMULATIVE_THROUGHPUT
While `LATENCY` and `THROUGHPUT` can select one target device with your preferred performance option, the `CUMULATIVE_THROUGHPUT` option enables running inference on multiple devices for higher throughput. With `CUMULATIVE_THROUGHPUT`, AUTO loads the network model to all available devices in the candidate list, and then runs inference on them based on the default or specified priority. 

CUMULATIVE_THROUGHPUT has similar behavior as [the Multi-Device execution mode (MULTI)](./multi_device.md). The only difference is that CUMULATIVE_THROUGHPUT uses the devices specified by AUTO, which means that it's not mandatory to add devices manually, while with MULTI, you need to specify the devices before inference. 

With the CUMULATIVE_THROUGHPUT option:
- If `AUTO` without any device names is specified, and the system has more than two GPU devices, AUTO will remove CPU from the device candidate list to keep GPU running at full capacity.
- If device priority is specified, AUTO will run inference requests on devices based on the priority. In the following example, AUTO will always try to use GPU first, and then use CPU if GPU is busy:
   ```sh
   ov::CompiledModel compiled_model = core.compile_model(model, "AUTO:GPU,CPU", ov::hint::performance_mode(ov::hint::PerformanceMode::CUMULATIVE_THROUGHPUT));
   ```

#### Code Examples

To enable performance hints for your application, use the following code: 
@sphinxdirective

.. tab:: C++

    .. doxygensnippet:: docs/snippets/AUTO3.cpp
       :language: cpp
       :fragment: [part3]
 
.. tab:: Python

    .. doxygensnippet:: docs/snippets/ov_auto.py
       :language: python
       :fragment: [part3]

@endsphinxdirective

#### Disabling Auto-Batching for THROUGHPUT and CUMULATIVE_THROUGHPUT

The `ov::hint::PerformanceMode::THROUGHPUT` mode and the `ov::hint::PerformanceMode::CUMULATIVE_THROUGHPUT` mode will trigger Auto-Batching (for example, for the GPU device) by default. You can disable it by setting `ov::hint::allow_auto_batching(false)`, or change the default timeout value to a large number, e.g. `ov::auto_batch_timeout(1000)`. See [Automatic Batching](./automatic_batching.md) for more details.

### Configuring Model Priority

The `ov::hint::model_priority` property enables you to control the priorities of models in the Auto-Device plugin. A high-priority model will be loaded to a supported high-priority device. A lower-priority model will not be loaded to a device that is occupied by a higher-priority model.

@sphinxdirective

.. tab:: C++

    .. doxygensnippet:: docs/snippets/AUTO4.cpp
       :language: cpp
       :fragment: [part4]
 
.. tab:: Python

    .. doxygensnippet:: docs/snippets/ov_auto.py
       :language: python
       :fragment: [part4]

@endsphinxdirective

## Configuring Individual Devices and Creating the Auto-Device plugin on Top

Although the methods described above are currently the preferred way to execute inference with AUTO, the following steps can be also used as an alternative. It is currently available as a legacy feature and used if AUTO is uncapable of utilizing the Performance Hints option. 


@sphinxdirective

.. tab:: C++

    .. doxygensnippet:: docs/snippets/AUTO5.cpp
       :language: cpp
       :fragment: [part5]
 
.. tab:: Python

    .. doxygensnippet:: docs/snippets/ov_auto.py
       :language: python
       :fragment: [part5]

@endsphinxdirective

## <a name="using-auto-with-openvino-samples-and-benchmark-app"></a> Using AUTO with OpenVINO Samples and Benchmark app

To see how the Auto-Device plugin is used in practice and test its performance, take a look at OpenVINO™ samples. All samples supporting the "-d" command-line option (which stands for "device") will accept the plugin out-of-the-box. The Benchmark Application will be a perfect place to start – it presents the optimal performance of the plugin without the need for additional settings, like the number of requests or CPU threads. To evaluate the AUTO performance, you can use the following commands:

For unlimited device choice:

```sh
benchmark_app –d AUTO –m <model> -i <input> -niter 1000
```

For limited device choice:

```sh
benchmark_app –d AUTO:CPU,GPU,GNA –m <model> -i <input> -niter 1000
```

For more information, refer to the [C++](../../samples/cpp/benchmark_app/README.md) or [Python](../../tools/benchmark_tool/README.md) version instructions.	

@sphinxdirective
.. note::

   The default CPU stream is 1 if using “-d AUTO”.

   You can use the FP16 IR to work with auto-device.

   No demos are yet fully optimized for AUTO, by means of selecting the most suitable device, using the GPU streams/throttling, and so on.
@endsphinxdirective

## Additional Resources

- [Debugging AUTO](AutoPlugin_Debugging.md)
- [Running on Multiple Devices Simultaneously](./multi_device.md)
- [Supported Devices](supported_plugins/Supported_Devices.md)

