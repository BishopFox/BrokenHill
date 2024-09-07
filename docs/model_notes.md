# Model notes

This document describes high-level results of testing using various publicly-available large language models.

## Model families with publicly-available versions capable of handling chat interaction


### Gemma

* [Google's Gemma model family documentation](https://ai.google.dev/gemma/docs)
* Trained to avoid discussing a variety of potentially-dangerous and controversial topics: **Yes**
** Tool can generate adversarial content that defeats those restrictions: TKTK
* Will generally follow system prompt instructions that restrict information given to the user: **Yes**
** Tool can generate adversarial content that defeats those restrictions: **Yes**
* `fschat` template name: `gemma`

The tool includes a custom `gemma` chat template because `fschat` seems to go back and forth between including one and not including one.

#### Specific versions tested using this tool:

* [gemma-2b-it](https://huggingface.co/google/gemma-2b-it)

### Gemma 2

* [Google's Gemma model family documentation](https://ai.google.dev/gemma/docs)
* Trained to avoid discussing a variety of potentially-dangerous and controversial topics: **Yes**
** Tool can generate adversarial content that defeats those restrictions: TKTK
* Will generally follow system prompt instructions that restrict information given to the user: **Yes**
** Tool can generate adversarial content that defeats those restrictions: **Yes**
* `fschat` template name: `gemma`

The tool includes a custom `gemma` chat template because `fschat` seems to go back and forth between including one and not including one. Gemma 2 *seems* to use the same template format.

#### Specific versions tested using this tool:

* [gemma-2-2b](https://huggingface.co/google/gemma-2b)
* [gemma-2-2b-it](https://huggingface.co/google/gemma-2b-it)

### Phi-2

* [Microsoft's Phi-2 model at Hugging Face](https://huggingface.co/microsoft/phi-2)
* Trained to avoid discussing a variety of potentially-dangerous and controversial topics: **Yes**
** Tool can generate adversarial content that defeats those restrictions: **Yes**
* Will generally follow system prompt instructions that restrict information given to the user: **Yes**
** Tool can generate adversarial content that defeats those restrictions: **Yes**
* `fschat` template name: `phi2`

The tool includes a custom `phi2` chat template because `fschat` does not currently include one.

#### Specific versions tested using this tool:

* [phi-2](https://huggingface.co/microsoft/phi-2)

### Phi-3

* [Microsoft's Phi-3 model collection at Hugging Face](https://huggingface.co/collections/microsoft/phi-3-6626e15e9585a200d2d761e3)
* Trained to avoid discussing a variety of potentially-dangerous and controversial topics: **Yes**
** Tool can generate adversarial content that defeats those restrictions: **Yes**
* Will generally follow system prompt instructions that restrict information given to the user: **Yes**
** Tool can generate adversarial content that defeats those restrictions: **Yes**
* `fschat` template name: `phi3`

Phi-3 is one of the most frequent test candidates when developing this tool. Everything should work as expected. The tool includes a custom `phi3` chat template because `fschat` does not currently include one.

#### Specific versions tested using this tool:

* [Phi-3-mini-128k-instruct](https://huggingface.co/microsoft/Phi-3-mini-128k-instruct)

#### Specific versions tested using tool output in `ollama`:

* phi3
* phi3:3.8b-mini-128k-instruct-q8_0
* phi3:3.8b-mini-128k-instruct-q2_K
* phi3:3.8b-mini-128k-instruct-q4_0
* phi3:3.8b-mini-128k-instruct-fp16

### Qwen

* [Qwen collection at Hugging Face](https://huggingface.co/collections/Qwen/qwen-65c0e50c3f1ab89cb8704144)
* [Qwen 1.5 collection at Hugging Face](https://huggingface.co/collections/Qwen/qwen15-65c0a2f577b1ecb76d786524)
* Trained to avoid discussing a variety of potentially-dangerous and controversial topics: **Yes**
** Tool can generate adversarial content that defeats those restrictions: TKTK
* Will generally follow system prompt instructions that restrict information given to the user: **Yes**
** Tool can generate adversarial content that defeats those restrictions: **Yes**
* `fschat` template name: `qwen`

#### Specific versions tested using this tool:

* [Qwen1.5-0.5B-Chat](https://huggingface.co/Qwen/Qwen1.5-0.5B-Chat)

### Qwen 2

* [Qwen 2 collection at Hugging Face](https://huggingface.co/collections/Qwen/qwen2-6659360b33528ced941e557f)
* Trained to avoid discussing a variety of potentially-dangerous and controversial topics: **Yes**
** Tool can generate adversarial content that defeats those restrictions: TKTK
* Will generally follow system prompt instructions that restrict information given to the user: **Yes**
** Tool can generate adversarial content that defeats those restrictions: **Yes**
* `fschat` template name: `qwen2`

#### Specific versions tested using this tool:

* [Qwen2-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2-0.5B-Instruct)

### StableLM and StableLM 2

* Trained to avoid discussing a variety of potentially-dangerous and controversial topics: **No**
* Will generally follow system prompt instructions that restrict information given to the user: **No**
* `fschat` template names: `stablelm` and `stablelm2`

As discussed in [the documentation for stablelm-2-1_6b-chat](https://huggingface.co/stabilityai/stablelm-2-1_6b-chat) and [the documentation for stablelm-2-zephyr-1_6b](https://huggingface.co/stabilityai/stablelm-2-zephyr-1_6b), this model family doesn't have any built-in restrictions regarding controversial topics. It is supported by this tool in order to test models built on top of StabilityAI's foundation. The tool includes a custom `stablelm2` chat template because `fschat` does not currently include one.

#### Specific versions tested using this tool:

* [stablelm-tuned-alpha-3b](https://huggingface.co/stabilityai/stablelm-tuned-alpha-3b)
* [stablelm-zephyr-3b](https://huggingface.co/stabilityai/stablelm-zephyr-3b)
* [stablelm-2-1_6b-chat](https://huggingface.co/stabilityai/stablelm-2-1_6b-chat)
* [stablelm-2-zephyr-1_6b](https://huggingface.co/stabilityai/stablelm-2-zephyr-1_6b)

### TinyLlama

* [TinyLlama GitHub repository](https://github.com/jzhang38/TinyLlama)
* Trained to avoid discussing a variety of potentially-dangerous and controversial topics: TKTK
** Tool can generate adversarial content that defeats those restrictions: TKTK
* Will generally follow system prompt instructions that restrict information given to the user: **Yes**
** Tool can generate adversarial content that defeats those restrictions: **Yes**
* `fschat` template name: `TinyLlama`

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
