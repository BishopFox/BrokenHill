# Troubleshooting

## I keep running out of memory!

The GCG attack uses a *lot* of PyTorch device memory. A CUDA device with 24 GiB of memory, such as a GeForce RTX 4090, can easily load an LLM like `Llama-2-7b-chat-hf`, because seven billion parameters represented as 16-bit floating point values occupy about 14 GiB of memory. But (among other things) [the GCG attack involves very memory-intensive operations like creating a PyTorch gradient and running backpropagation, and those operations are too large to perform for a seven-billion-parameter model on a device with 24 GiB of memory](memory_requirements.md). If you want to test models of that size, you'll need to use [a device with more memory](other_graphics_hardware.md), or the CPU.

If you are already testing using [models with sizes in-line with the values in the "Broken Hill PyTorch device memory requirements" document](memory_requirements.md) and still running out of memory, here are a few things to try:

* If you're using `--model-data-type float32`, using `--model-data-type bfloat16` will reduce the size of almost everything in memory by half.
* If you are running Broken Hill on a device with one or more screens connected to the same GPU that you're using for Broken Hill:
** If you are using a GUI to interact with the OS (X, Weyland, Gnome, KDE, etc.), log out of the GUI and then switch to a text console.
*** e.g. on Debian and most derived distributions, Ctrl-Alt-F2 to switch to the second virtual console, then Ctrl-Alt-F1 to switch back to the GUI session when you're done. This can free up to several GiB of VRAM depending on how many displays are attached, what resolution the GPU is rendering content for them at, etc.
* Specify a value lower than the default for `--max-new-adversarial-value-candidate-count`, e.g. `--max-new-adversarial-value-candidate-count 8`. This will reduce the efficiency of the search for new values, so consider setting a `--required-loss-threshold` value [as discussed in the the "Content-generation size controls" section of the command-line options documentation](all_command-line_options.md#content-generation-size-controls).
* If you're really pressed for memory, you can try specifying `--batch-size-get-logits 1`, but we don't recommend doing so.
* Examine the length of the system prompt (if any) and template messages (if any). During the gradient-generation phase of the GCG attack, calculations are performed using data derived from the *entire* conversation, including the system prompt and template messages. Lengthy messages of either kind can consume significant amounts of CUDA device memory.

## Broken Hill is running very slowly

### For CPU processing, make sure you're using a model data type that is acceptable for your hardware

[This is discussed in the "Selecting a model data type" document](selecting_a_model_data_type.md).

### Use Linux for the best performance

[As documented in the "Selecting a model data type" document](selecting_a_model_data_type.md), using Windows roughly doubles processing times versus the same device running Linux, for both CUDA and CPU processing.

## Dependency order conflict when installing using setup.py

If you are trying to install Broken Hill version 0.33 or older, or are installing 0.34 or newer using `setup.py` instead of the recommended method, you may receive an error like the following:

```
Collecting flash-attn==2.6.3 (from llm_attacks_bishopfox==0.0.2)
  Downloading flash_attn-2.6.3.tar.gz (2.6 MB)
...omitted for brevity...
  error: subprocess-exited-with-error
...omitted for brevity...
      ModuleNotFoundError: No module named 'torch'
```

This is because `flash-attn` and `causal-conv1d` have dependencies on `torch` that may not be declared properly, and they won't install if you don't already have `torch` installed system-wide or in the virtual environment. Comment out these lines in `requirements.txt`:

```
flash_attn==2.6.3
causal-conv1d==1.4.0
```

Re-run the installation:

```
$ bin/pip install ./BrokenHill/
```

Uncomment the lines in `requirements.txt` that you commented out previously, then run the installation one more time:

```
$ bin/pip install ./BrokenHill/
```

## TypeError: Couldn't build proto file into descriptor pool

Sometimes, everything will seem to be working fine, and then the next time you run Broken Hill, you'll get an error like this:

```
[load_model_and_tokenizer] Warning: unable to load standard tokenizer from '/mnt/md0/Machine_Learning/LLMs/Microsoft/Phi-3-mini-128k-instruct', attempting to fall back to fast tokenizer. The exception thrown when loading the standard tokenizer was: Couldn't build proto file into descriptor pool: Invalid default '0.9995' for field sentencepiece.TrainerSpec.character_coverage of type 2
```

...or like this, if you disable the associated exception handler:

```
  File "/mnt/md0/Machine_Learning/lib/python3.11/site-packages/transformers/models/auto/tokenization_auto.py", line 889, in from_pretrained
    return tokenizer_class.from_pretrained(pretrained_model_name_or_path, *inputs, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/md0/Machine_Learning/lib/python3.11/site-packages/transformers/tokenization_utils_base.py", line 2163, in from_pretrained
    return cls._from_pretrained(
           ^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/md0/Machine_Learning/lib/python3.11/site-packages/transformers/tokenization_utils_base.py", line 2397, in _from_pretrained
    tokenizer = cls(*init_inputs, **init_kwargs)
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/md0/Machine_Learning/lib/python3.11/site-packages/transformers/models/llama/tokenization_llama.py", line 171, in __init__
    self.sp_model = self.get_spm_processor(kwargs.pop("from_slow", False))
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/md0/Machine_Learning/lib/python3.11/site-packages/transformers/models/llama/tokenization_llama.py", line 203, in get_spm_processor
    model_pb2 = import_protobuf(f"The new behaviour of {self.__class__.__name__} (with `self.legacy = False`)")
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/md0/Machine_Learning/lib/python3.11/site-packages/transformers/convert_slow_tokenizer.py", line 40, in import_protobuf
    from transformers.utils import sentencepiece_model_pb2_new as sentencepiece_model_pb2
  File "/mnt/md0/Machine_Learning/lib/python3.11/site-packages/transformers/utils/sentencepiece_model_pb2_new.py", line 17, in <module>
    DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
TypeError: Couldn't build proto file into descriptor pool: Invalid default '0.9995' for field sentencepiece.TrainerSpec.character_coverage of type 2
```

This shouldn't happen anymore, but if it does, you can try the following workaround to remove the Python `protobuf` library, then install the pure Python version of the same library.

```
$ bin/pip uninstall protobuf
$ bin/pip install --no-binary protobuf protobuf
```
