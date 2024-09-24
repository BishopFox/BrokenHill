# Model notes

This document describes high-level results of testing the GCG attack using various publicly-available large language models.

## Table of contents

1. [](#)

## Model families with publicly-available versions capable of handling chat interaction

### Falcon

* Conversation template name: `falcon`
* [TII's Falcon LLM website](https://falconllm.tii.ae/)
* Trained to avoid discussing a variety of potentially-dangerous and controversial topics: **Yes**
  * Tool can generate adversarial content that defeats those restrictions: TBD
* Will generally follow system prompt instructions that restrict information given to the user: TBD
  * Tool can generate adversarial content that defeats those restrictions: TBD

#### Specific versions tested using this tool:

* [falcon-7b-instruct](https://huggingface.co/tiiuae/falcon-7b-instruct)

### Gemma

* Conversation template name: `gemma`
* [Google's Gemma model family documentation](https://ai.google.dev/gemma/docs)
* Trained to avoid discussing a variety of potentially-dangerous and controversial topics: **Yes**
  * Tool can generate adversarial content that defeats those restrictions: TBD
* Will generally follow system prompt instructions that restrict information given to the user: **Yes**
  * Tool can generate adversarial content that defeats those restrictions: **Yes**

Broken Hill includes a custom `gemma` chat template because `fschat` seems to go back and forth between including one and not including one.

#### Special considerations

Gemma is strongly conditioned to avoid discussing certain topics. We'll be adding a separate discussion about this.

#### Specific versions tested using this tool:

* [gemma-2b-it](https://huggingface.co/google/gemma-2b-it)

### Gemma 2

* Conversation template name: `gemma`
* [Google's Gemma model family documentation](https://ai.google.dev/gemma/docs)
* Trained to avoid discussing a variety of potentially-dangerous and controversial topics: **Yes**
  * Tool can generate adversarial content that defeats those restrictions: TBD
* Will generally follow system prompt instructions that restrict information given to the user: **Yes**
  * Tool can generate adversarial content that defeats those restrictions: **Yes**

Broken Hill includes a custom `gemma` chat template because `fschat` seems to go back and forth between including one and not including one. Gemma 2 *seems* to use the same template format.

#### Special considerations

Gemma 2 is strongly conditioned to avoid discussing certain topics. We'll be adding a separate discussion about this.

#### Specific versions tested using this tool:

* [gemma-2-2b](https://huggingface.co/google/gemma-2b)
* [gemma-2-2b-it](https://huggingface.co/google/gemma-2b-it)

### Guanaco

* Conversation template name: `guanaco`
* [Guanaco-7B at Hugging Face](https://huggingface.co/timdettmers/guanaco-7b)
* Trained to avoid discussing a variety of potentially-dangerous and controversial topics: **Yes**
  * Tool can generate adversarial content that defeats those restrictions: TBD
* Will generally follow system prompt instructions that restrict information given to the user: TBD
  * Tool can generate adversarial content that defeats those restrictions: TBD

#### Special considerations

Guanaco is a PEFT pre-trained model built on top of the original Llama. To you use, you'll need to specify the corresponding Llama model using the `--model` option, and refer to Guanaco using the `--peft-adapter` option, e.g.:

```
--model /mnt/md0/Machine_Learning/LLMs/huggyllama/llama-7b \
--peft-adapter /mnt/md0/Machine_Learning/LLMs/timdettmers/guanaco-7b \
```

Even though Guanaco is a model layered on top of Llama, it uses its own conversation template. The format is similar to the `fschat` `zero_shot` template, but not identical, so Broken Hill includes a custom `guanaco` template.

#### Specific versions tested using this tool:

* [guanaco-7b](https://huggingface.co/timdettmers/guanaco-7b)

### Llama

Broken Hill can successfully load the original Llama model, but we haven't been able to find any documentation on the specific format it expects conversation messages in. Using the templates that seem like they'd work (`llama2`, `zero_shot`, `guanaco`) produces output similar to other models when given input using a conversation template that doesn't match the data the model was trained with. In other words, it's unclear how useful the results are. If you have reliable information on the correct conversation format, please let us know.

### Llama-2

* Conversation template name: `llama2` or `llama-2` (see discussion below)
* [Meta's Llama LLM family website](https://www.llama.com/llama-downloads/)
* Trained to avoid discussing a variety of potentially-dangerous and controversial topics: **Yes**
  * Tool can generate adversarial content that defeats those restrictions: TBD
* Will generally follow system prompt instructions that restrict information given to the user: TBD
  * Tool can generate adversarial content that defeats those restrictions: TBD

#### Special considerations

`fschat` includes a template for Llama-2 named `llama-2`, but it is slightly incorrect (for example, it does not add the leading `<s>` at the beginning of the conversation, and it adds a trailing `<s>` to the conversation. Fixing the template completely seems like it will require code changes to `fschat`. Broken Hill includes a modified version of the template named `llama2` that can be used as a workaround. The custom template has a different name in this case to allow operators to easy choose which option they believe is the "least worst option" for their purposes.

The custom template is also slightly incorrect, but seems to be "less wrong" regarding the parts of the output that are more likely to affect Broken Hill's results. Specifically, it adds the leading `<s>` at the beginning of the conversation when a system prompt is present, and sets a default empty system message to cause the system message block to be included in all conversations. It still leaves a trailing `<s>` at the end of the conversation.

Until this issue is resolved, Broken Hill will report one or more warnings when the Llama-2 templates are used.

#### Specific versions tested using this tool:

* [Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)

### Llama-3

* Conversation template name: `llama-3` (see instructions below)
* [Meta's Llama LLM family website](https://www.llama.com/llama-downloads/)
* Trained to avoid discussing a variety of potentially-dangerous and controversial topics: **Yes**
  * Tool can generate adversarial content that defeats those restrictions: TBD
* Will generally follow system prompt instructions that restrict information given to the user: TBD
  * Tool can generate adversarial content that defeats those restrictions: TBD

#### Special considerations

As of this writing, the current *release* of `fschat` (0.2.36 - from February 11th, 2024) did not support Llama-3, and the template requires custom logic. [You can install `fschat` from source](https://github.com/lm-sys/FastChat?tab=readme-ov-file#method-2-from-source) to enable the `llama-3` template, e.g. from the base directory where you created the Python virtual environment for Broken Hill:

```
git clone https://github.com/lm-sys/FastChat.git
cd FastChat
../bin/pip install -e ".[model_worker,webui]"
cd ..
```

As with the Llama-2 conversation template, the `fschat` template for Llama-3 does not exactly match the output of the tokenizer's `apply_chat_template` function (for example, `fschat` adds an extra `<|eot_id|>` at the end of the prompt), but the differences shouldn't be enough to materially effect Broken Hill's test results. Until `fschat` is updated, Broken Hill will display a brief warning when the `llama-3` template is used.

#### Specific versions tested using this tool:

* [Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)

### MPT

* Conversation template name: `mpt`
* [Databricks' Mosaic Research website, which includes MPT](https://www.databricks.com/research/mosaic)
* Trained to avoid discussing a variety of potentially-dangerous and controversial topics: **Yes**
  * Tool can generate adversarial content that defeats those restrictions: TBD
* Will generally follow system prompt instructions that restrict information given to the user: TBD
  * Tool can generate adversarial content that defeats those restrictions: TBD

`fschat` includes a template for MPT, but for some reason there are two templates named `mpt-7b-chat` and `mpt-30b-chat`, which are identical except for the system prompt. Broken Hill includes a shortcut template definition that points to `mpt-7b-chat`.

#### Specific versions tested using this tool:

* [mpt-7b-chat](https://huggingface.co/mosaicml/mpt-7b-chat)

### Phi-2

* Conversation template name: `phi2`
* [Microsoft's Phi-2 model at Hugging Face](https://huggingface.co/microsoft/phi-2)
* Trained to avoid discussing a variety of potentially-dangerous and controversial topics: **Yes**
  * Tool can generate adversarial content that defeats those restrictions: **Yes**
* Will generally follow system prompt instructions that restrict information given to the user: **Yes**
  * Tool can generate adversarial content that defeats those restrictions: **Yes**

Broken Hill includes a custom `phi2` chat template because `fschat` does not currently include one.

#### Specific versions tested using this tool:

* [phi-2](https://huggingface.co/microsoft/phi-2)

### Phi-3

* Conversation template name: `phi3`
* [Microsoft's Phi-3 model collection at Hugging Face](https://huggingface.co/collections/microsoft/phi-3-6626e15e9585a200d2d761e3)
* Trained to avoid discussing a variety of potentially-dangerous and controversial topics: **Yes**
  * Tool can generate adversarial content that defeats those restrictions: **Yes**
* Will generally follow system prompt instructions that restrict information given to the user: **Yes**
  * Tool can generate adversarial content that defeats those restrictions: **Yes**

Broken Hill includes a custom `phi3` chat template because `fschat` does not currently include one.

Phi-3 is one of the most frequent test candidates when developing this tool. Everything should work as expected. 

#### Specific versions tested using this tool:

* [Phi-3-mini-128k-instruct](https://huggingface.co/microsoft/Phi-3-mini-128k-instruct)

#### Specific versions tested using tool output in `ollama`:

* phi3
* phi3:3.8b-mini-128k-instruct-q8_0
* phi3:3.8b-mini-128k-instruct-q2_K
* phi3:3.8b-mini-128k-instruct-q4_0
* phi3:3.8b-mini-128k-instruct-fp16

### Qwen

* Conversation template name: `qwen`
* [Alibaba's Qwen model family page at Hugging Face](https://huggingface.co/Qwen)
* Trained to avoid discussing a variety of potentially-dangerous and controversial topics: **Yes**
  * Tool can generate adversarial content that defeats those restrictions: TBD
* Will generally follow system prompt instructions that restrict information given to the user: **Yes**
  * Tool can generate adversarial content that defeats those restrictions: **Yes**

`fschat` includes a template for Qwen and Qwen 2, but for some reason it's named `qwen-7b-chat` specifically, so Broken Hill includes a shortcut template definition that points to that.

#### Specific versions tested using this tool:

* [Qwen1.5-0.5B-Chat](https://huggingface.co/Qwen/Qwen1.5-0.5B-Chat)

### Qwen 2

* Conversation template name: `qwen2`
* [Alibaba's Qwen model family page at Hugging Face](https://huggingface.co/Qwen)
* Trained to avoid discussing a variety of potentially-dangerous and controversial topics: **Yes**
  * Tool can generate adversarial content that defeats those restrictions: TBD
* Will generally follow system prompt instructions that restrict information given to the user: **Yes**
  * Tool can generate adversarial content that defeats those restrictions: **Yes**

`fschat` includes a template for Qwen and Qwen 2, but for some reason it's named `qwen-7b-chat` specifically, so Broken Hill includes a shortcut template definition that points to that.

#### Specific versions tested using this tool:

* [Qwen2-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2-0.5B-Instruct)

### SmolLM

* Conversation template name: `smollm`
* [Hugging Face blog post introducing SmolLM](https://huggingface.co/blog/smollm)
* Trained to avoid discussing a variety of potentially-dangerous and controversial topics: TBD
  * Tool can generate adversarial content that defeats those restrictions: TBD
* Will generally follow system prompt instructions that restrict information given to the user: TBD
  * Tool can generate adversarial content that defeats those restrictions: TBD

Broken Hill includes a custom `smollm` chat template because `fschat` does not currently include one.

#### Specific versions tested using this tool:

* [SmolLM-1.7B-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM-1.7B-Instruct)

### StableLM

* Conversation template name: `stablelm`
* [Stability AI StableLM family GitHub repository](https://github.com/Stability-AI/StableLM)
* Trained to avoid discussing a variety of potentially-dangerous and controversial topics: **No**
* Will generally follow system prompt instructions that restrict information given to the user: **No**

As discussed in [the documentation for stablelm-2-1_6b-chat](https://huggingface.co/stabilityai/stablelm-2-1_6b-chat) and [the documentation for stablelm-2-zephyr-1_6b](https://huggingface.co/stabilityai/stablelm-2-zephyr-1_6b), this model family doesn't have any built-in restrictions regarding controversial topics. It is supported by this tool in order to test models built on top of StabilityAI's foundation.

#### Specific versions tested using this tool:

* [stablelm-tuned-alpha-3b](https://huggingface.co/stabilityai/stablelm-tuned-alpha-3b)
* [stablelm-zephyr-3b](https://huggingface.co/stabilityai/stablelm-zephyr-3b)

### StableLM 2

* Conversation template name: `stablelm2`
* [Stability AI StableLM family GitHub repository](https://github.com/Stability-AI/StableLM)
* Trained to avoid discussing a variety of potentially-dangerous and controversial topics: **No**
* Will generally follow system prompt instructions that restrict information given to the user: **No**

Broken Hill includes a custom `stablelm2` chat template because `fschat` does not currently include one.
As discussed in [the documentation for stablelm-2-1_6b-chat](https://huggingface.co/stabilityai/stablelm-2-1_6b-chat) and [the documentation for stablelm-2-zephyr-1_6b](https://huggingface.co/stabilityai/stablelm-2-zephyr-1_6b), this model family doesn't have any built-in restrictions regarding controversial topics. It is supported by this tool in order to test models built on top of StabilityAI's foundation. 

#### Specific versions tested using this tool:

* [stablelm-2-1_6b-chat](https://huggingface.co/stabilityai/stablelm-2-1_6b-chat)
* [stablelm-2-zephyr-1_6b](https://huggingface.co/stabilityai/stablelm-2-zephyr-1_6b)

### TinyLlama

* Conversation template name: `TinyLlama`
* [TinyLlama GitHub repository](https://github.com/jzhang38/TinyLlama)
* Trained to avoid discussing a variety of potentially-dangerous and controversial topics: **Yes**
  * Tool can generate adversarial content that defeats those restrictions: TBD
* Will generally follow system prompt instructions that restrict information given to the user: **Yes**
  * Tool can generate adversarial content that defeats those restrictions: **Yes**

#### Specific versions tested using this tool:

* [TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)

### Vicuna 1.1

* Conversation template name: `vicuna_v1.1`
* [The Large Model Systems Organization's Vicuna web page](https://lmsys.org/blog/2023-03-30-vicuna/)
* Trained to avoid discussing a variety of potentially-dangerous and controversial topics: TBD
  * Tool can generate adversarial content that defeats those restrictions: TBD
* Will generally follow system prompt instructions that restrict information given to the user: TBD
  * Tool can generate adversarial content that defeats those restrictions: TBD

#### Specific versions tested using this tool:

* [vicuna-7b-v1.1](https://huggingface.co/lmsys/vicuna-7b-v1.1)

## Other model families that can be used in the tool

These model families can be used in the tool, but publicly-available versions are not trained to handle chat-type interactions. The tool can handle them in case someone runs across a derived model that's been trained for chat-like interaction. If you encounter a derived model, you'll likely need to add a custom chat template to the code to generate useful results.

### BART

* [BART GitHub page](https://github.com/facebookresearch/fairseq/tree/main/examples/bart)

#### Special considerations

BART currently requires currently requires `--max-new-tokens-final 512` (or lower) to be manually specified.

#### Specific versions tested using this tool:

* [bart-large-cnn](https://huggingface.co/facebook/bart-large-cnn)

### BigBird / BigBirdPegasus

* [BigBird GitHub page](https://github.com/google-research/bigbird)

#### Specific versions tested using this tool:

* [bigbird-pegasus-large-pubmed](https://huggingface.co/google/bigbird-pegasus-large-pubmed)

### GPT-2

* [GPT-2 GitHub repository](https://github.com/openai/gpt-2)

#### Special considerations

GPT-2 currently requires `--max-new-tokens-final 512` (or lower) to be manually specified.

#### Specific versions tested using this tool:

* [gpt2-medium](https://huggingface.co/openai-community/gpt2-medium)

### GPT-J

* [Eleuther AI's GPT-J 6B page at Hugging Face](https://huggingface.co/EleutherAI/gpt-j-6b)

#### Specific versions tested using this tool:

* [GPT-J 6B](https://huggingface.co/EleutherAI/gpt-j-6b)

### GPT-Neo

* [GPT-Neo 1.3B Hugging Face page](https://huggingface.co/EleutherAI/gpt-neo-1.3B)

#### Specific versions tested using this tool:

* [gpt-neo-1.3B](https://huggingface.co/EleutherAI/gpt-neo-1.3B)

### OPT

* [Facebook's "OPT: Open Pre-trained Transformer Language Models" paper](https://arxiv.org/abs/2205.01068)

#### Special considerations

OPT currently requires `--max-new-tokens-final 512` (or lower) to be explicitly specified.

#### Specific versions tested using this tool:

* [opt-2.7b](https://huggingface.co/microsoft/phi-1_5)

### Phi-1

* [Microsoft's Phi-1 model collection at Hugging Face](https://huggingface.co/collections/microsoft/phi-1-6626e29134744e94e222d572)

#### Specific versions tested using this tool:

* [phi-1_5](https://huggingface.co/facebook/opt-2.7b)

### Pythia

* [Pythia GitHub repository](https://github.com/EleutherAI/pythia)

#### Specific versions tested using this tool:

* [pythia-1.4b](https://huggingface.co/EleutherAI/pythia-1.4b)

### RoBERTa

* [RoBERTa GitHub page](https://github.com/facebookresearch/fairseq/tree/main/examples/roberta)

#### Specific versions tested using this tool:

* [roberta-base](https://huggingface.co/FacebookAI/roberta-base)

## Other model families that do not currently work in the tool

### BlenderBot

* [BlenderBot 3B Hugging Face page](https://huggingface.co/facebook/blenderbot-3B)

This used to work in an early pre-release version, now it doesn't. We'll try to make it work again in a future release.

#### Special considerations

When Broken Hill can handle this model again, BlenderBot may require `--max-new-tokens 32` (or lower) and `--max-new-tokens-final 32` (or lower) depending on whether or not we add some other additional code.

#### Specific versions tested using this tool:

* [blenderbot-3B](https://huggingface.co/facebook/blenderbot-3B)

### GPT-NeoX

* [Eleuther AI's GPT-NeoX page at Hugging Face](https://huggingface.co/EleutherAI/gpt-neox-20b)

GPT-NeoX is currently unsupported because it only supports use of a fast tokenizer, and the fast tokenizer logic in Broken Hill needs additional work. We plan to have this operational again in the near future.

#### Special considerations

Some models based on GPT-NeoX do not include their own tokenizer, e.g. [tiny-random-GPTNeoXForCausalLM-safetensors](https://huggingface.co/trl-internal-testing/tiny-random-GPTNeoXForCausalLM-safetensors). If you receive a "Can't load tokenizer" error, try explicitly specifying the path to the GPT-NeoX 20B tokenizer, e.g. `--tokenizer LLMs/EleutherAI/gpt-neox-20b`

#### Specific versions tested using this tool:

* [tiny-random-GPTNeoXForCausalLM-safetensors](https://huggingface.co/trl-internal-testing/tiny-random-GPTNeoXForCausalLM-safetensors)


### Pegasus

* [Pegasus documentation](https://huggingface.co/docs/transformers/main/model_doc/pegasus)

This used to work in an early pre-release version, now it doesn't. We'll try to make it work again in a future release.

#### Special considerations

When Broken Hill can handle this model again, Pegasus may require `--max-new-tokens-final 512` (or lower) depending on whether or not we add some other additional code.

#### Specific versions tested using this tool:

* [pegasus-wikihow](https://huggingface.co/google/pegasus-wikihow)
