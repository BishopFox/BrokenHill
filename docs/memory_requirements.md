# Broken Hill PyTorch device memory requirements

This document currently lists general guidelines for requirement device memory based on ad hoc testing. It will be updated with additional information and more specific figures as we collect them.

*Note:* the memory required for testing is on the device where processing occurs. e.g. to perform a GCG attack against a model with 2 billion parameters, for `--device cuda` and `--model-data-type default` or `--model-data-type float16`, you would need an Nvidia GPU with at least 24 GiB of VRAM. For `--device cpu`, system memory is used instead. If you specify `--model-data-type float32`, be aware that the memory requirements are doubled. [Please see the "Selecting a model data type" document for more details on that option in relation to CPU processing](selecting_a_model_data_type.md).

*Note:* there are quite a few factors in determining how much memory PyTorch will attempt to use when Broken Hill runs. [The "I keep running out of memory" section in the Troubleshooting guide discusses this in more detail](troubleshooting.md). The figures below are for typical uses with default Broken Hill values.

## General guidelines

Performing the GCG attack requires representing the model's weights in a floating-point format. The smallest floating-point formats supported by PyTorch are 16 bits, or two bytes. Broken Hill performs all work using 16-bit floating point values by default, but supports 32-bit by specifying `--model-data-type float32`. For a 16-bit format, this means that the model will generally occupy a number of bytes in memory slightly over two times the number of parameters. For 32-bits, double all of the example values (e.g. the model itself will occupy about four times as many bytes in memory as its parameter count).

Some examples of device memory required just to load a model in 16-bit format:

* Qwen1.5-0.5B-Chat (463,987,712 parameters): 1,139,090,520 bytes
* gemma-2-2b-it (2,614,341,888 parameters): 5,228,674,048 bytes
* Phi-3-mini-128k-instruct (3,821,079,552 parameters): 7,642,165,248 bytes
* gpt-j-6b (6,050,882,784 parameters): 12,219,105,336 bytes

Performing the GCG attack also requires a gradient in PyTorch. The gradient is created at the beginning of the second iteration in a Broken Hill run (the first iteration is performed using the operator-specified adversarial content, so no gradient is required). The gradient generally occupies almost as much space in memory as the model itself.

Some examples of total device memory in use by the model and the gradient in 16-bit format:

* Qwen1.5-0.5B-Chat: 2,702,728,272 bytes
* gemma-2-2b-it: 12,851,731,394 bytes
* Phi-3-mini-128k-instruct: 15,302,306,960 bytes
* gpt-j-6b: more bytes than will fit into memory on a device with 24 GiB of VRAM

On top of that baseline, each iteration of the attack requires steps such as forward and backward propagation that generally require significant blocks of memory, if only very briefly. This amount can vary dramatically, and depends to some extent on the code for the specific model in the Transformers Python library. Some examples from testing with very simple Broken Hill configurations in 16-bit format:

* Qwen1.5-0.5B-Chat: 1,705,329,664 bytes
* gemma-2-2b-it: 2,457,600,000 bytes
* Phi-3-mini-128k-instruct: 289,089,024 bytes (yes, this is much smaller than the others)
* gpt-j-6b: unknown

## What size models can I use?

### Models with 500 million parameters or fewer

* 8 GiB should be sufficient to perform typical testing using Broken Hill if the model is loaded in 16-bit format

### Models with more than about 500 million, but less than 2 billion parameters

* There is probably a cutoff between 1 and 2 billion parameters where testing can be performed using 16 GiB of memory if the model is loaded in 16-bit format, but we haven't tested in that configuration yet
* 24 GiB should be sufficient to perform just about any testing you like using Broken Hill if the model is loaded in 16-bit format

### Models with 2 - 5 billion parameters

* 24 GiB is generally sufficient to perform the GCG attack if the model is loaded in 16-bit format
** Reducing the number of adversarial candidates using `--new-adversarial-value-candidate-count` to 16 or even 8 may be required for lengthy target strings or large numbers of adversarial tokens, especially with models at the high end of this parameter count range

### Models with 6 - 7 billion parameters

* 40 GiB should be sufficient to perform the GCG attack if the model is loaded in 16-bit format

### Models with 8 billion parameters

* 48 GiB should be sufficient to perform the GCG attack if the model is loaded in 16-bit format

### Models with 13 billion parameters

* 64 GiB should be sufficient to perform the GCG attack if the model is loaded in 16-bit format

### Models with 20 billion parameters

* 96 GiB should be sufficient to perform the GCG attack if the model is loaded in 16-bit format

### Models with 46 billion parameters

* 224 GiB should be sufficient to perform the GCG attack if the model is loaded in 16-bit format

### Models with more than 46 billion parameters

* We're going to need a <s>bigger boat</s>device with more memory