# Frequently-asked questions

## Why "Broken Hill"?

It's a reference to [*Lexicon*, by Max Barry](https://maxbarry.com/lexicon/). Saying any more than that would involve spoiling key plot elements from that novel.

## I keep running out of memory!

[Please see the discussion in the Troubleshooting guide](troubleshooting.md).

## Can I run this using something other than an Nvidia GPU?

As of version 0.34, yes, you can specify `--device cpu` to perform processing on a CPU instead of an Nvidia GPU. Processing times will be slower than on CUDA hardware, but generally tolerable. [Please see the "Selecting a model data type" document for more details](selecting_a_model_data_type.md).

[The `mps` (Metal for Mac OS) PyTorch back-end does not support some of the necessary features at this time](https://github.com/pytorch/pytorch/issues/127743) ([e.g. nested tensors](https://github.com/pytorch/pytorch/blob/3855ac5a5d53fd4d2d6521744eaf80c2a95a4d54/aten/src/ATen/NestedTensorImpl.cpp#L183), so using `--device mps` will cause the script to crash not long after loading the model, with an error similar to the following:

```
[2024-11-14@05:37:42][X] Broken Hill encountered an unhandled exception during the GCG attack: storage_device.is_cpu() || storage_device.is_cuda() || storage_device.is_xpu() || storage_device.is_privateuseone() INTERNAL ASSERT FAILED at "/Users/runner/work/pytorch/pytorch/pytorch/aten/src/ATen/NestedTensorImpl.cpp":185, please report a bug to PyTorch. NestedTensorImpl storage must be either CUDA, CPU, XPU or privateuseone but got mps:0. The exception details will be displayed below this message for troubleshooting purposes.
Traceback (most recent call last):
  File "/Users/benlincoln/Documents/BrokenHill/./BrokenHill/brokenhill.py", line 720, in main
    logits, ids = get_logits(attack_state,
                  ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/benlincoln/Documents/BrokenHill/BrokenHill/llm_attacks_bishopfox/minimal_gcg/opt_utils.py", line 961, in get_logits
    nested_ids = torch.nested.nested_tensor(test_ids)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/benlincoln/Documents/BrokenHill/lib/python3.11/site-packages/torch/nested/__init__.py", line 226, in nested_tensor
    return _nested.nested_tensor(
           ^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: storage_device.is_cpu() || storage_device.is_cuda() || storage_device.is_xpu() || storage_device.is_privateuseone() INTERNAL ASSERT FAILED at "/Users/runner/work/pytorch/pytorch/pytorch/aten/src/ATen/NestedTensorImpl.cpp":185, please report a bug to PyTorch. NestedTensorImpl storage must be either CUDA, CPU, XPU or privateuseone but got mps:0
```

## What about Intel XPU devices, like that MPS-related error message above mentions?

Broken Hill does not currently support XPU devices, or [the Intel extension for PyTorch](https://github.com/intel/intel-extension-for-pytorch) in general, but a future version may.

## Can I quantize the models to use less memory?

No (for any type of integer quantization, which is the main type of quantization that inspires this question). The source code for Broken Hill is littered with grumpy comments related to lack of support for quantized integer values in PyTorch for some of the features Broken Hill depends upon.

## Why do you keep referring to the `fschat` library instead of calling it `fastchat` or FastChat like everyone else?

[Please see the "On the naming of FastChat's `fschat` (AKA `fastchat` in some contexts) Python library" document](fschat.md).
