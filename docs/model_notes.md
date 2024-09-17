# Model notes

This document describes high-level results of testing using various publicly-available large language models.

## Model families with publicly-available versions capable of handling chat interaction


### Gemma

* Chat template name: `gemma`
* Minimum recommended PyTorch device memory: 24GB
* [Google's Gemma model family documentation](https://ai.google.dev/gemma/docs)
* Trained to avoid discussing a variety of potentially-dangerous and controversial topics: **Yes**
** Tool can generate adversarial content that defeats those restrictions: TBD
* Will generally follow system prompt instructions that restrict information given to the user: **Yes**
** Tool can generate adversarial content that defeats those restrictions: **Yes**

Broken Hill includes a custom `gemma` chat template because `fschat` seems to go back and forth between including one and not including one.

#### Special considerations

[Gemma is strongly conditioned to avoid discussing certain topics. See the "Slippery models" document for a lengthier discussion.](slippery_models.md)

#### Specific versions tested using this tool:

* [gemma-2b-it](https://huggingface.co/google/gemma-2b-it)

### Gemma 2

* Chat template name: `gemma`
* Minimum recommended PyTorch device memory: 24GB
* [Google's Gemma model family documentation](https://ai.google.dev/gemma/docs)
* Trained to avoid discussing a variety of potentially-dangerous and controversial topics: **Yes**
** Tool can generate adversarial content that defeats those restrictions: TKTK
* Will generally follow system prompt instructions that restrict information given to the user: **Yes**
** Tool can generate adversarial content that defeats those restrictions: **Yes**

Broken Hill includes a custom `gemma` chat template because `fschat` seems to go back and forth between including one and not including one. Gemma 2 *seems* to use the same template format.

#### Special considerations

[Gemma 2 is strongly conditioned to avoid discussing certain topics. See the "Slippery models" document for a lengthier discussion.](slippery_models.md)

#### Specific versions tested using this tool:

* [gemma-2-2b](https://huggingface.co/google/gemma-2b)
* [gemma-2-2b-it](https://huggingface.co/google/gemma-2b-it)

### Llama-2

* Chat template name: `llama2` (not: *not* `llama-2`)
* Minimum recommended PyTorch device memory: TBD
* [TKTK](TKTK)
* Trained to avoid discussing a variety of potentially-dangerous and controversial topics: **Yes**
** Tool can generate adversarial content that defeats those restrictions: TBD
* Will generally follow system prompt instructions that restrict information given to the user: TBD
** Tool can generate adversarial content that defeats those restrictions: TBD

`fschat` includes a template for Llama-2 named `llama-2`, but it is slightly incorrect. Broken Hill includes a slightly-less-wrong template definition  named `llama2`that matches the output of the Llama-2 tokenizer's `apply_chat_template` method in terms of the beginning of the text, which is more likely to affect Broken Hill results. Fixing the template completely seems like it will require code changes to `fschat`.

### MPT

* Chat template name: `mpt`
* Minimum recommended PyTorch device memory: TBD
* [TKTK](TKTK)
* Trained to avoid discussing a variety of potentially-dangerous and controversial topics: **Yes**
** Tool can generate adversarial content that defeats those restrictions: TBD
* Will generally follow system prompt instructions that restrict information given to the user: TBD
** Tool can generate adversarial content that defeats those restrictions: TBD

`fschat` includes a template for MPT, but for some reason there are two templates named `mpt-7b-chat` and `mpt-30b-chat`, which are identical except for the system prompt. Broken Hill includes a shortcut template definition that points to `mpt-7b-chat`.

### Phi-2

* Chat template name: `phi2`
* Minimum recommended PyTorch device memory: TBD
* [Microsoft's Phi-2 model at Hugging Face](https://huggingface.co/microsoft/phi-2)
* Trained to avoid discussing a variety of potentially-dangerous and controversial topics: **Yes**
** Tool can generate adversarial content that defeats those restrictions: **Yes**
* Will generally follow system prompt instructions that restrict information given to the user: **Yes**
** Tool can generate adversarial content that defeats those restrictions: **Yes**

Broken Hill includes a custom `phi2` chat template because `fschat` does not currently include one.

#### Specific versions tested using this tool:

* [phi-2](https://huggingface.co/microsoft/phi-2)

### Phi-3

* Chat template name: `phi3`
* Minimum recommended PyTorch device memory: TBD
* [Microsoft's Phi-3 model collection at Hugging Face](https://huggingface.co/collections/microsoft/phi-3-6626e15e9585a200d2d761e3)
* Trained to avoid discussing a variety of potentially-dangerous and controversial topics: **Yes**
** Tool can generate adversarial content that defeats those restrictions: **Yes**
* Will generally follow system prompt instructions that restrict information given to the user: **Yes**
** Tool can generate adversarial content that defeats those restrictions: **Yes**

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

* Chat template name: `qwen`
* Minimum recommended PyTorch device memory: TBD
* [Qwen collection at Hugging Face](https://huggingface.co/collections/Qwen/qwen-65c0e50c3f1ab89cb8704144)
* [Qwen 1.5 collection at Hugging Face](https://huggingface.co/collections/Qwen/qwen15-65c0a2f577b1ecb76d786524)
* Trained to avoid discussing a variety of potentially-dangerous and controversial topics: **Yes**
** Tool can generate adversarial content that defeats those restrictions: TKTK
* Will generally follow system prompt instructions that restrict information given to the user: **Yes**
** Tool can generate adversarial content that defeats those restrictions: **Yes**

`fschat` includes a template for Qwen and Qwen 2, but for some reason it's named `qwen-7b-chat` specifically, so Broken Hill includes a shortcut template definition that points to that.

#### Specific versions tested using this tool:

* [Qwen1.5-0.5B-Chat](https://huggingface.co/Qwen/Qwen1.5-0.5B-Chat)

### Qwen 2

* Chat template name: `qwen2`
* Minimum recommended PyTorch device memory: TBD
* [Qwen 2 collection at Hugging Face](https://huggingface.co/collections/Qwen/qwen2-6659360b33528ced941e557f)
* Trained to avoid discussing a variety of potentially-dangerous and controversial topics: **Yes**
** Tool can generate adversarial content that defeats those restrictions: TKTK
* Will generally follow system prompt instructions that restrict information given to the user: **Yes**
** Tool can generate adversarial content that defeats those restrictions: **Yes**

`fschat` includes a template for Qwen and Qwen 2, but for some reason it's named `qwen-7b-chat` specifically, so Broken Hill includes a shortcut template definition that points to that.

#### Specific versions tested using this tool:

* [Qwen2-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2-0.5B-Instruct)

### StableLM

* Chat template name: `stablelm`
* Minimum recommended PyTorch device memory: TBD
* Trained to avoid discussing a variety of potentially-dangerous and controversial topics: **No**
* Will generally follow system prompt instructions that restrict information given to the user: **No**

As discussed in [the documentation for stablelm-2-1_6b-chat](https://huggingface.co/stabilityai/stablelm-2-1_6b-chat) and [the documentation for stablelm-2-zephyr-1_6b](https://huggingface.co/stabilityai/stablelm-2-zephyr-1_6b), this model family doesn't have any built-in restrictions regarding controversial topics. It is supported by this tool in order to test models built on top of StabilityAI's foundation.

#### Specific versions tested using this tool:

* [stablelm-tuned-alpha-3b](https://huggingface.co/stabilityai/stablelm-tuned-alpha-3b)
* [stablelm-zephyr-3b](https://huggingface.co/stabilityai/stablelm-zephyr-3b)

### StableLM 2

* Chat template name: `stablelm2`
* Minimum recommended PyTorch device memory: TBD
* Trained to avoid discussing a variety of potentially-dangerous and controversial topics: **No**
* Will generally follow system prompt instructions that restrict information given to the user: **No**

Broken Hill includes a custom `stablelm2` chat template because `fschat` does not currently include one.
As discussed in [the documentation for stablelm-2-1_6b-chat](https://huggingface.co/stabilityai/stablelm-2-1_6b-chat) and [the documentation for stablelm-2-zephyr-1_6b](https://huggingface.co/stabilityai/stablelm-2-zephyr-1_6b), this model family doesn't have any built-in restrictions regarding controversial topics. It is supported by this tool in order to test models built on top of StabilityAI's foundation. 

#### Specific versions tested using this tool:

* [stablelm-2-1_6b-chat](https://huggingface.co/stabilityai/stablelm-2-1_6b-chat)
* [stablelm-2-zephyr-1_6b](https://huggingface.co/stabilityai/stablelm-2-zephyr-1_6b)

### TinyLlama

* Chat template name: `TinyLlama`
* Minimum recommended PyTorch device memory: TBD
* [TinyLlama GitHub repository](https://github.com/jzhang38/TinyLlama)
* Trained to avoid discussing a variety of potentially-dangerous and controversial topics: TBD
** Tool can generate adversarial content that defeats those restrictions: TBD
* Will generally follow system prompt instructions that restrict information given to the user: **Yes**
** Tool can generate adversarial content that defeats those restrictions: **Yes**

#### Specific versions tested using this tool:

* [TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)

## Other model families that can be used in the tool

These model families can be used in the tool, but publicly-available versions are not trained to handle chat-type interactions. The tool can handle them in case someone runs across a derived model that's been trained for chat-like interaction. If you encounter a derived model, you'll likely need to add a custom chat template to the code to generate useful results.

### BART

* [BART GitHub page](https://github.com/facebookresearch/fairseq/tree/main/examples/bart)

#### Specific versions tested using this tool:

* [bart-large-cnn](https://huggingface.co/facebook/bart-large-cnn)

### BigBird / BigBirdPegasus

* [BigBird GitHub page](https://github.com/google-research/bigbird)

#### Specific versions tested using this tool:

* [bigbird-pegasus-large-pubmed](https://huggingface.co/google/bigbird-pegasus-large-pubmed)

### GPT-2

* [GPT-2 GitHub repository](https://github.com/openai/gpt-2)

#### Specific versions tested using this tool:

* [gpt2-medium](https://huggingface.co/openai-community/gpt2-medium)

### GPT-Neo

* [GPT-Neo 1.3B Hugging Face page](https://huggingface.co/EleutherAI/gpt-neo-1.3B)

#### Specific versions tested using this tool:

* [gpt-neo-1.3B](https://huggingface.co/EleutherAI/gpt-neo-1.3B)

### Phi-1

* [Microsoft's Phi-1 model collection at Hugging Face](https://huggingface.co/collections/microsoft/phi-1-6626e29134744e94e222d572)

#### Specific versions tested using this tool:

* [phi-1_5](https://huggingface.co/microsoft/phi-1_5)

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

#### Specific versions tested using this tool:

* [blenderbot-3B](https://huggingface.co/facebook/blenderbot-3B)

### Pegasus

* [Pegasus documentation](https://huggingface.co/docs/transformers/main/model_doc/pegasus)

This used to work in an early pre-release version, now it doesn't. We'll try to make it work again in a future release.

#### Specific versions tested using this tool:

* [pegasus-wikihow](https://huggingface.co/google/pegasus-wikihow)
