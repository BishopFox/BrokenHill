# Frequently-asked questions

## Why "Broken Hill"?

It's a reference to [*Lexicon*, by Max Barry](https://maxbarry.com/lexicon/). Saying any more than that would involve spoiling key plot elements from that novel.

## I keep running out of memory!

The GCG attack uses a *lot* of PyTorch device memory. A CUDA device with 24 GiB of memory, such as a GeForce RTX 4090, can easily load an LLM like `Llama-2-7b-chat-hf`, because seven billion parameters represented as 16-bit floating point values occupy about 14 GiB of memory. But (among other things) the GCG attack involves very memory-intensive operations like creating a PyTorch gradient and running backpropagation, and those operations are too large to perform for a seven-billion-parameter model on a device with 24 GiB of memory. If you want to test models of that size, you'll need to use [a device with more memory](other_graphics_hardware.md).

If you are already testing using [models with sizes in-line with the values in the "Broken Hill PyTorch device memory requirements" document](memory_requirements.md) and still running out of memory, here are a few things to try:

* If you are running Broken Hill on a device with one or more screens connected to the same GPU that you're using for Broken Hill:
** If you are using a GUI to interact with the OS (X, Weyland, Gnome, KDE, etc.), close out of the GUI or suspend it.
*** e.g. on Debian and most derived distributions, Ctrl-Alt-F2 to switch to the second virtual console, then Ctrl-Alt-F1 to switch back to the GUI session when you're done. This can free up to several GiB of VRAM depending on how many displays are attached, what resolution the GPU is rendering content for them at, etc.
* Specify a value lower than the default (48) for `--max-new-adversarial-value-candidate-count`, e.g. `--max-new-adversarial-value-candidate-count 16`. This will reduce the efficiency of the search for new values, so consider setting a `--required-loss-threshold` value [as discussed in the the "Content-generation size controls" section of the command-line options documentation](all_command-line_options.md#content-generation-size-controls).
* If you're specifying a non-default value for `--batch-size-get-logits`, stop doing that.

## Can I run this using something other than an Nvidia GPU?

You can use the PyTorch `cpu` device if you really want to, but it will literally take hundreds of times longer to run, e.g. 2+ hours for a single iteration of the main loop.

[The `mps` (Metal for Mac OS) PyTorch back-end does not support some of the necessary features at this time](https://github.com/pytorch/pytorch/issues/127743) ([e.g. nested tensors](https://github.com/pytorch/pytorch/blob/3855ac5a5d53fd4d2d6521744eaf80c2a95a4d54/aten/src/ATen/NestedTensorImpl.cpp#L183), so using `--device mps` will cause the script to crash not long after loading the model.

## Can I quantize the models to use less VRAM?

No (for any type of integer quantization, which is the main type of quantization that inspires this question). The source code for Broken Hill is littered with grumpy comments related to lack of support for quantized integer values in PyTorch for some of the features this tool depends upon.

In the unlikely event that you are asking about the corner case where you have a model with weights in a larger format than 16-bit floating point that hasn't already also been released in a 16-bit floating point format, you could quantize that model to a 16-bit floating point format, and it will occupy less memory when you load it into the tool. Most publicly-available models that you'd want to use with this tool are already available in such a format.
