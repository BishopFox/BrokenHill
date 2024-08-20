# Troubleshooting

## TypeError: Couldn't build proto file into descriptor pool

Sometimes, everything will seem to be working fine, and then the next time you run the tool, you'll get an error like this:

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
