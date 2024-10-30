# Broken Hill PyTorch device memory requirements

This document currently lists general guidelines for requirement device memory based on ad hoc testing. It will be updated with additional information and more specific figures as we collect them.

*Note:* as discussed elsewhere in the documentation, Broken Hill is currently only really usable with CUDA devices, meaning that the memory required is on an Nvidia GPU. i.e. to use Broken Hill to perform a GCG attack against a model with 2 billion parameters, you will need an Nvidia GPU with at least 24 GiB of VRAM. You can technically perform testing using `--device cpu` to take advantage of RAM instead of VRAM, but the length of time per iteration makes it impractical to perform more than a few iterations using the CPU and system RAM.

*Note:* there are quite a few factors in determining how much memory PyTorch will attempt to use on a CUDA device when Broken Hill runs. [The "I keep running out of memory" section in the Troubleshooting guide discusses this in more detail](troubleshooting.md). The figures below are for typical uses with default Broken Hill values.

## General guidelines

Performing the GCG attack requires representing the model's weights in a floating-point format. The smallest floating-point formats are 16 bits, or two bytes, and Broken Hill currently performs all work using 16-bit floating point values, even if the model's weights are stored in a larger format. This means that the model will generally occupy a number of bytes in memory slightly over two times the number of parameters.

Some examples of device memory required just to load a model:

* Qwen1.5-0.5B-Chat (463,987,712 parameters): 1,139,090,520 bytes
* gemma-2-2b-it (2,614,341,888 parameters): 5,228,674,048 bytes
* Phi-3-mini-128k-instruct (3,821,079,552 parameters): 7,642,165,248 bytes
* gpt-j-6b (6,050,882,784 parameters): 12,219,105,336 bytes

Performing the GCG attack also requires a gradient in PyTorch. The gradient is created at the beginning of the second iteration in a Broken Hill run (the first iteration is performed using the operator-specified adversarial content, so no gradient is required). The gradient generally occupies almost as much space in memory as the model itself.

Some examples of total device memory in use by the model and the gradient:

* Qwen1.5-0.5B-Chat: 2,702,728,272 bytes
* gemma-2-2b-it: 12,851,731,394 bytes
* Phi-3-mini-128k-instruct: 15,302,306,960 bytes
* gpt-j-6b: more bytes than will fit into memory on a device with 24 GiB of VRAM

On top of that baseline, each iteration of the attack requires steps such as forward and backward propagation that generally require significant blocks of memory, if only very briefly. This amount can vary dramatically, and depends to some extent on the code for the specific model in the Transformers Python library. Some examples from testing with very simple Broken Hill configurations:

* Qwen1.5-0.5B-Chat: 1,705,329,664 bytes
* gemma-2-2b-it: 2,457,600,000 bytes
* Phi-3-mini-128k-instruct: 289,089,024 bytes (yes, this is much smaller than the others)
* gpt-j-6b: unknown

## What size models can I use?

### Models with 500 million parameters or fewer

* 8 GiB should be sufficient to perform typical testing using Broken Hill

### Models with more than about 500 million, but less than 2 billion parameters

* There is probably a cutoff between 1 and 2 billion parameters where testing can be performed using 16 GiB  of memory, but we haven't tested in that configuration yet
* 24 GiB should be sufficient to perform just about any testing you like using Broken Hill

### Models with 2 - 5 billion parameters

* 24 GiB is generally sufficient to perform the GCG attack
** Reducing the number of adversarial candidates using `--new-adversarial-value-candidate-count` to 16 or even 8 may be required for lengthy target strings or large numbers of adversarial tokens, especially with models at the high end of this parameter count range.

### Models with 6 - 11 billion parameters

* 24 GiB should be sufficient to perform Broken Hill self-tests, but generally not perform the GCG attack

### Models with 12 billion parameters or more

* We're going to need a <s>bigger boat</s>device with more than 24 GiB of VRAM