# Model notes

This document describes high-level results of testing the GCG attack using various publicly-available large language models.

## Table of contents

* [Model families with publicly-available versions capable of handling chat interaction](#model-families-with-publicly-available-versions-capable-of-handling-chat-interaction)
  * [APT](#apt)
  * [ChatGLM and GLM](#chatglm-and-glm)
  * [Falcon](#falcon)
  * [FalconMamba](#falconmamba)
  * [Gemma](#gemma)
  * [Gemma 2](#gemma-2)
  * [GPT-NeoX](#gpt-neox)
  * [Guanaco](#guanaco)
  * [Llama](#llama)
  * [Llama-2](#llama-2)
  * [Llama-3](#llama-3)
  * [MiniCPM](#minicpm)
  * [Mistral](#mistral)
  * [Mixtral](#mixtral)
  * [MPT](#mpt)
  * [Orca](#orca)
  * [Phi-2](#phi-2)
  * [Phi-3](#phi-3)
  * [Pythia](#pythia)
  * [Qwen](#qwen)
  * [Qwen 2](#qwen-2)
  * [SmolLM](#smollm)
  * [SOLAR](#solar)
  * [StableLM](#stablelm)
  * [StableLM 2](#stablelm-2)
  * [TinyLlama](#tinyllama)
  * [Vicuna](#vicuna)
* [Other model families that can be used with Broken Hill](#other-model-families-that-can-be-used-with-broken-hill)
  * [BART](#bart)
  * [BigBird / BigBirdPegasus](#bigbird--bigbirdpegasus)
  * [BlenderBot](#blenderbot)
  * [GPT-1](#gpt-1)
  * [GPT-2](#gpt-2)
  * [GPT-J](#gpt-j)
  * [GPT-Neo](#gpt-neo)
  * [Mamba](#mamba)
  * [OPT](#opt)
  * [Pegasus](#pegasus)
  * [Phi-1](#phi-1)
  * [RoBERTa](#roberta)
  * [XLNet](#xlnet)
* [Other model families that do not currently work with Broken Hill](#other-model-families-that-do-not-currently-work-with-broken-hill)
  * [T5](#t5)

## Model families with publicly-available versions capable of handling chat interaction

### APT

* Conversation template name: `llama2`
* [Azurro APT3 collection at Hugging Face](https://huggingface.co/collections/Azurro/apt3-66fa965b5eea43a116b1c545)

#### Specific models tested using Broken Hill:

* [APT-1B-Base](https://huggingface.co/Azurro/APT-1B-Base)
* [APT2-1B-Base](https://huggingface.co/Azurro/APT2-1B-Base)
* [APT3-1B-Base](https://huggingface.co/Azurro/APT3-1B-Base)
* [APT3-1B-Instruct-v1](https://huggingface.co/Azurro/APT3-1B-Instruct-v1)

### ChatGLM and GLM

* Conversation template names:
  * For GLM-4: `glm4`
* [Zhipu AI's GLM-4-9B-Chat page at Hugging Face](https://huggingface.co/THUDM/glm-4-9b-chat/blob/main/README_en.md)
* Trained to avoid discussing a variety of potentially-dangerous and controversial topics: TBD
  * Tool can generate adversarial content that defeats those restrictions: TBD
* Will generally follow system prompt instructions that restrict information given to the user: TBD
  * Tool can generate adversarial content that defeats those restrictions: TBD

#### Special considerations

You will need to specify the `--trust-remote-code` option to use these models.

Broken Hill includes a custom `glm4` conversation template based on the `fschat` `chatglm3` template, due to differences in the system prompt formatting.

Currently, only GLM-4 works in Broken Hill due to code provided with earlier versions of the model that is incompatible with modern versions of Transformers. We've tried coming up with instructions to make the earlier versions work, but it looks like a deep rabbit hole.

You may encounter the following error when using GLM-4, depending on whether or not the GLM developers have updated their code by the time you've cloned their repository:

```
TypeError: ChatGLM4Tokenizer._pad() got an unexpected keyword argument 'padding_side'
```

As a workaround, you can make the following change to the `tokenization_chatglm.py` file included with the model:

Locate the following code:

```
def _pad(
		self,
		encoded_inputs: Union[Dict[str, EncodedInput], BatchEncoding],
		max_length: Optional[int] = None,
		padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
		pad_to_multiple_of: Optional[int] = None,
		return_attention_mask: Optional[bool] = None,
) -> dict:
```

Add a definition for the missing parameter, so that the method signature looks like this:

```
def _pad(
		self,
		encoded_inputs: Union[Dict[str, EncodedInput], BatchEncoding],
		max_length: Optional[int] = None,
		padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
		pad_to_multiple_of: Optional[int] = None,
		return_attention_mask: Optional[bool] = None,
		padding_side: Optional[str] = "left",
) -> dict:
```

Locate this line:

```
assert self.padding_side == "left"
```

Comment it out, so it looks like the following:
    
```
#assert padding_side == "left"
```

#### Specific models tested using Broken Hill:

* [glm-4-9b-chat](https://huggingface.co/THUDM/glm-4-9b-chat)

### Falcon

* Conversation template name: `falcon`
* [TII's Falcon LLM website](https://falconllm.tii.ae/)
* Trained to avoid discussing a variety of potentially-dangerous and controversial topics: **Yes**
  * Tool can generate adversarial content that defeats those restrictions: TBD
* Will generally follow system prompt instructions that restrict information given to the user: TBD
  * Tool can generate adversarial content that defeats those restrictions: TBD

#### Specific models tested using Broken Hill:

* [falcon-7b-instruct](https://huggingface.co/tiiuae/falcon-7b-instruct)

### FalconMamba

* Conversation template name: `falcon-mamba`
* [TII's FalconMamba 7B page at Hugging Face](https://huggingface.co/tiiuae/falcon-mamba-7b-instruct)
* Trained to avoid discussing a variety of potentially-dangerous and controversial topics: **Yes**
  * Tool can generate adversarial content that defeats those restrictions: TBD
* Will generally follow system prompt instructions that restrict information given to the user: TBD
  * Tool can generate adversarial content that defeats those restrictions: TBD

#### Special considerations

Broken Hill includes a custom `falcon-mamba` conversation template for this model.

#### Specific models tested using Broken Hill:

* [falcon-mamba-7b-instruct](https://huggingface.co/tiiuae/falcon-mamba-7b-instruct)

### Gemma

#### First-party models

* Conversation template name: `gemma`
* [Google's Gemma model family documentation](https://ai.google.dev/gemma/docs)
* Trained to avoid discussing a variety of potentially-dangerous and controversial topics: **Yes**
  * Tool can generate adversarial content that defeats those restrictions: TBD
* Will generally follow system prompt instructions that restrict information given to the user: **Yes**
  * Tool can generate adversarial content that defeats those restrictions: **Yes**

Broken Hill includes a custom `gemma` chat template because `fschat` seems to go back and forth between including one and not including one, and the current version the last time I checked added a spurious extra `\n<end_of_turn>` to the end of the conversation.

#### Special considerations

Gemma is strongly conditioned to avoid discussing certain topics. We'll be adding a separate discussion about this.

#### Specific models tested using Broken Hill:

* [gemma-2b-it](https://huggingface.co/google/gemma-2b-it)
* [gemma-1.1-2b-it](https://huggingface.co/google/gemma-1.1-2b-it)
* [gemma-1.1-7b-it](https://huggingface.co/google/gemma-1.1-7b-it)

#### Other third-party variations

* Conversation template name: `gemma`
* Trained to avoid discussing a variety of potentially-dangerous and controversial topics: Depends on variation
* Will generally follow system prompt instructions that restrict information given to the user: Depends on variation

##### Specific models tested using Broken Hill:

* [Vikhr-Gemma-2B-instruct](https://huggingface.co/Vikhrmodels/Vikhr-Gemma-2B-instruct)

### Gemma 2

* Conversation template name: `gemma`
* [Google's Gemma model family documentation](https://ai.google.dev/gemma/docs)
* Trained to avoid discussing a variety of potentially-dangerous and controversial topics: **Yes**
  * Tool can generate adversarial content that defeats those restrictions: TBD
* Will generally follow system prompt instructions that restrict information given to the user: **Yes**
  * Tool can generate adversarial content that defeats those restrictions: **Yes**

Broken Hill includes a custom `gemma` chat template because `fschat` seems to go back and forth between including one and not including one, and the current version the last time I checked added a spurious extra `\n<end_of_turn>` to the end of the conversation.

#### Special considerations

Gemma 2 is strongly conditioned to avoid discussing certain topics. We'll be adding a separate discussion about this.

#### Specific models tested using Broken Hill:

* [gemma-2-2b](https://huggingface.co/google/gemma-2b)
* [gemma-2-2b-it](https://huggingface.co/google/gemma-2b-it)
* [gemma-2-9b-it](https://huggingface.co/google/gemma-2-9b-it)

### GPT-NeoX

As with their [GPT-J](#gpt-j), [GPT-Neo](#gpt-neo), and [Pythia](#pythia) models, Eleuther AI only publishes GPT-NeoX as base models, and (as of this writing) all GPT-NeoX variations fine-tuned for chat have been published by unrelated third parties.

#### First-party models

* Conversation template name: `gptneox`
* Trained to avoid discussing a variety of potentially-dangerous and controversial topics: TBD
  * Tool can generate adversarial content that defeats those restrictions: TBD
* Will generally follow system prompt instructions that restrict information given to the user: TBD
  * Tool can generate adversarial content that defeats those restrictions: TBD
* [Eleuther AI's GPT-NeoX page at Hugging Face](https://huggingface.co/EleutherAI/gpt-neox-20b)

##### Specific models tested using Broken Hill:

* [gpt-neox-20b](https://huggingface.co/EleutherAI/gpt-neox-20b)

#### Kobold AI "Erebus" uncensored GPT-NeoX model

* Conversation template name: `gptneox`
* Trained to avoid discussing a variety of potentially-dangerous and controversial topics: TBD
* Will generally follow system prompt instructions that restrict information given to the user: TBD

##### Specific models tested using Broken Hill:

* [GPT-NeoX-20B-Erebus](https://huggingface.co/KoboldAI/GPT-NeoX-20B-Erebus)

#### Other third-party variations

* Conversation template name: `gptneox`
* Trained to avoid discussing a variety of potentially-dangerous and controversial topics: Depends on variation
* Will generally follow system prompt instructions that restrict information given to the user: Depends on variation

##### Special considerations

Some models based on GPT-NeoX do not include their own tokenizer, e.g. [tiny-random-GPTNeoXForCausalLM-safetensors](https://huggingface.co/trl-internal-testing/tiny-random-GPTNeoXForCausalLM-safetensors). If you receive a "Can't load tokenizer" error, try explicitly specifying the path to the GPT-NeoX 20B tokenizer, e.g. `--tokenizer LLMs/EleutherAI/gpt-neox-20b`. However, `tiny-random-GPTNeoXForCausalLM-safetensors` specifically will still cause Broken Hill to crash, so don't use that model unless your goal is to make Broken Hill crash.

### Guanaco

#### Base model

* Conversation template name: `guanaco`
* [Guanaco-7B at Hugging Face](https://huggingface.co/timdettmers/guanaco-7b)
* Trained to avoid discussing a variety of potentially-dangerous and controversial topics: **Yes**
  * Tool can generate adversarial content that defeats those restrictions: TBD
* Will generally follow system prompt instructions that restrict information given to the user: TBD
  * Tool can generate adversarial content that defeats those restrictions: TBD

##### Special considerations

Guanaco is a PEFT pre-trained model built on top of the original Llama. To use [Tim Dettmers' canonical version](https://huggingface.co/timdettmers/guanaco-7b), you'll need to specify the corresponding Llama model using the `--model` option, and refer to Guanaco using the `--peft-adapter` option, e.g.:

```
--model /mnt/md0/Machine_Learning/LLMs/huggyllama/llama-7b \
--peft-adapter /mnt/md0/Machine_Learning/LLMs/timdettmers/guanaco-7b \
```

["TheBloke"'s guanaco-7B-HF version unifies Guanaco into a single model](https://huggingface.co/TheBloke/guanaco-7B-HF). Using the `--peft-adapter` option is unnecessary with that variation.

Even though Guanaco is a model layered on top of Llama, it uses its own conversation template. The format is similar to the `fschat` `zero_shot` template, but not identical, so Broken Hill includes a custom `guanaco` template.

##### Specific models tested using Broken Hill:

* [guanaco-7b](https://huggingface.co/timdettmers/guanaco-7b)
* [guanaco-7B-HF](https://huggingface.co/TheBloke/guanaco-7B-HF)

#### "Fredithefish" Guanaco-3B-Uncensored-v2

* Conversation template name: `guanaco`
* [Guanaco-3B-Uncensored-v2 at Hugging Face](https://huggingface.co/Fredithefish/Guanaco-3B-Uncensored-v2)
* Trained to avoid discussing a variety of potentially-dangerous and controversial topics: **No**
* Will generally follow system prompt instructions that restrict information given to the user: TBD
  * Tool can generate adversarial content that defeats those restrictions: TBD

##### Specific models tested using Broken Hill:

* [Guanaco-3B-Uncensored-v2](https://huggingface.co/Fredithefish/Guanaco-3B-Uncensored-v2)

### Llama

#### First-party models

Broken Hill can successfully load the original Llama model, but we haven't been able to find any documentation on the specific format it expects conversation messages in. Using the templates that seem like they'd work (`llama2`, `zero_shot`, `guanaco`) produces output similar to other models when given input using a conversation template that doesn't match the data the model was trained with. In other words, it's unclear how useful the results are. If you have reliable information on the correct conversation format, please let us know.

##### Specific models tested using Broken Hill:

* [llama-7b](https://huggingface.co/huggyllama/llama-7b)

#### Alpaca

* Conversation template name: `alpaca`

##### Specific models tested using Broken Hill:

* [alpaca-13b](https://huggingface.co/chavinlo/alpaca-13b)

#### Llama-68M-Chat

* Conversation template name: `felladrin-llama-chat`

##### Specific models tested using Broken Hill:

* [Llama-68M-Chat-v1](https://huggingface.co/meta-llama/Felladrin/Llama-68M-Chat-v1)

#### Vikhr-7B-instruct_merged

* Conversation template name: `vikhr`

##### Specific models tested using Broken Hill:

* [Vikhr-7B-instruct_merged](https://huggingface.co/Vikhrmodels/Vikhr-7B-instruct_merged)

### Llama-2

#### First-party models

* Conversation template name: `llama2` or `llama-2` (see discussion below)
* [Meta's Llama LLM family website](https://www.llama.com/llama-downloads/)
* Trained to avoid discussing a variety of potentially-dangerous and controversial topics: **Yes**
  * Tool can generate adversarial content that defeats those restrictions: TBD
* Will generally follow system prompt instructions that restrict information given to the user: TBD
  * Tool can generate adversarial content that defeats those restrictions: TBD

##### Special considerations

`fschat` includes a template for Llama-2 named `llama-2`, but it is slightly incorrect (for example, it does not add the leading `<s>` at the beginning of the conversation, and it adds a trailing `<s>` to the conversation. Fixing the template completely seems like it will require code changes to `fschat`. Broken Hill includes a modified version of the template named `llama2` that can be used as a workaround. The custom template has a different name in this case to allow operators to easy choose which option they believe is the "least worst option" for their purposes.

The custom template is also slightly incorrect, but seems to be "less wrong" regarding the parts of the output that are more likely to affect Broken Hill's results. Specifically, it adds the leading `<s>` at the beginning of the conversation when a system prompt is present, and sets a default empty system message to cause the system message block to be included in all conversations. It still leaves a trailing `<s>` at the end of the conversation.

Until this issue is resolved, Broken Hill will report one or more warnings when the Llama-2 templates are used.

##### Specific models tested using Broken Hill:

* [Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)
* [Meta-Llama-Guard-2-8B](https://huggingface.co/meta-llama/Meta-Llama-Guard-2-8B)

#### Other third-party variations

* Conversation template name: `llama2` or `llama-2` (see discussion above)
* Trained to avoid discussing a variety of potentially-dangerous and controversial topics: Depends on variation
* Will generally follow system prompt instructions that restrict information given to the user: Depends on variation

##### Specific models tested using Broken Hill:

* [Swallow-7b-instruct-hf](https://huggingface.co/tokyotech-llm/Swallow-7b-instruct-hf)
* [Swallow-MS-7b-instruct-v0.1](https://huggingface.co/tokyotech-llm/Swallow-MS-7b-instruct-v0.1)
* [Vikhr-7B-instruct_0.4](https://huggingface.co/Vikhrmodels/Vikhr-7B-instruct_0.4)
* [youri-7b-chat](https://huggingface.co/rinna/youri-7b-chat)

### Llama-3

#### First-party models

* Conversation template name: `llama-3` (see instructions below)
* [Meta's Llama LLM family website](https://www.llama.com/llama-downloads/)
* Trained to avoid discussing a variety of potentially-dangerous and controversial topics: **Yes**
  * Tool can generate adversarial content that defeats those restrictions: TBD
* Will generally follow system prompt instructions that restrict information given to the user: TBD
  * Tool can generate adversarial content that defeats those restrictions: TBD

##### Specific models tested using Broken Hill:

* [Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)
* [Meta-Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct)
* [Llama-Guard-3-8B](https://huggingface.co/meta-llama/Llama-Guard-3-8B)

#### Other third-party variations

* Conversation template name: `llama-3` (see instructions above)
* Trained to avoid discussing a variety of potentially-dangerous and controversial topics: Depends on variation
* Will generally follow system prompt instructions that restrict information given to the user: Depends on variation

##### Specific models tested using Broken Hill:

* [Llama-3-Swallow-8B-Instruct-v0.1](tokyotech-llm/Llama-3-Swallow-8B-Instruct-v0.1)
* [llama-3-youko-8b-instruct](https://huggingface.co/rinna/llama-3-youko-8b-instruct)

### MiniCPM

#### First-party models

* Conversation template name: `minicpm`
* Trained to avoid discussing a variety of potentially-dangerous and controversial topics: **Yes**
* Will generally follow system prompt instructions that restrict information given to the user: TBD

##### Special considerations

Broken Hill includes a custom `minicpm` conversation template for first-party MiniCPM models.

##### Specific models tested using Broken Hill:

* [MiniCPM-2B-sft-fp32](https://huggingface.co/openbmb/MiniCPM-2B-sft-fp32)

#### Vikhr-tiny

* Conversation template name: `vikhr`
* [Vikhr-tiny-0.1 model page at Hugging Face](https://huggingface.co/Vikhrmodels/Vikhr-tiny-0.1)
* Trained to avoid discussing a variety of potentially-dangerous and controversial topics: TBD
* Will generally follow system prompt instructions that restrict information given to the user: TBD

##### Specific models tested using Broken Hill:

* [Vikhr-tiny-0.1](https://huggingface.co/Vikhrmodels/Vikhr-tiny-0.1)

### Mistral

#### First-party models

* Conversation template name: `mistral`
* [Mistral AI homepage](https://mistral.ai/)
* Trained to avoid discussing a variety of potentially-dangerous and controversial topics: TBD
  * Tool can generate adversarial content that defeats those restrictions: TBD
* Will generally follow system prompt instructions that restrict information given to the user: TBD
  * Tool can generate adversarial content that defeats those restrictions: TBD

##### Specific models tested using Broken Hill:

* [Mistral-7B-Instruct-v0.3](https://huggingface.co/MistralAI/Mistral-7B-Instruct-v0.3)
* [Mistral-Nemo-Instruct-2407](https://huggingface.co/MistralAI/Mistral-Nemo-Instruct-2407)

#### Daredevil / NeuralDaredevil

* Conversation template name: `daredevil`
* [Daredevil-7B model page at Hugging Face](https://huggingface.co/mlabonne/Daredevil-7B)
* [NeuralDaredevil-7B model page at Hugging Face](https://huggingface.co/mlabonne/NeuralDaredevil-7B)
* Trained to avoid discussing a variety of potentially-dangerous and controversial topics: TBD
* Will generally follow system prompt instructions that restrict information given to the user: TBD

##### Special considerations

These models are derived from Mistral (and other models), and their chat format is similar, but not identical. Broken Hill includes a custom `daredevil` conversation template to use with them.

##### Specific models tested using Broken Hill:

* [Daredevil-7B](https://huggingface.co/mlabonne/Daredevil-7B)
* [NeuralDaredevil-7B](https://huggingface.co/mlabonne/NeuralDaredevil-7B)

#### Intel Neural Chat

* Conversation template name: `mistral`
* [Intel Neural-Chat-v3-3 model page at Hugging Face](https://huggingface.co/Intel/neural-chat-7b-v3-3)
* Trained to avoid discussing a variety of potentially-dangerous and controversial topics: **No**
* Will generally follow system prompt instructions that restrict information given to the user: TBD
  * Tool can generate adversarial content that defeats those restrictions: TBD

##### Specific models tested using Broken Hill:

* [neural-chat-7b-v3-3](https://huggingface.co/Intel/neural-chat-7b-v3-3)

### Mixtral

#### Base models

* Conversation template name: `mistral`
* [Mistral AI homepage](https://mistral.ai/)
* Trained to avoid discussing a variety of potentially-dangerous and controversial topics: **Yes**
  * Tool can generate adversarial content that defeats those restrictions: TBD
* Will generally follow system prompt instructions that restrict information given to the user: TBD
  * Tool can generate adversarial content that defeats those restrictions: TBD

##### Specific models tested using Broken Hill:

* [Mixtral-8x7B-Instruct-v0.1](https://huggingface.co/MistralAI/Mixtral-8x7B-Instruct-v0.1)

### MPT

* Conversation template name: `mpt`
* [Databricks' Mosaic Research website, which includes MPT](https://www.databricks.com/research/mosaic)
* Trained to avoid discussing a variety of potentially-dangerous and controversial topics: **Yes**
  * Tool can generate adversarial content that defeats those restrictions: TBD
* Will generally follow system prompt instructions that restrict information given to the user: TBD
  * Tool can generate adversarial content that defeats those restrictions: TBD

`fschat` includes a template for MPT, but for some reason there are two templates named `mpt-7b-chat` and `mpt-30b-chat`, which are completely different. Broken Hill includes a shortcut template definition for `mpt` that points to `mpt-7b-chat`.

#### Special considerations

One might assume that - because Broken Hill supports a model named `MPT` - it would also support the very similarly named model [mpt-1b-redpajama-200b-dolly](https://huggingface.co/mosaicml/mpt-1b-redpajama-200b-dolly). That assumption would be incorrect. `mpt-1b-redpajama-200b-dolly` has its own custom template in Broken Hill (`mpt-redpajama` - because it uses a completely different conversation format), but a GCG attack cannot currently be performed against it because the interface for the model doesn't support generation using an `inputs_embeds` keyword (or equivalent): 

"inputs_embeds is not implemented for MosaicGPT yet"
		-- `mosaic_gpt.py`:400

Maybe someday someone will add that model to the Transformers library and give it the appropriate code.

#### Specific models tested using Broken Hill:

* [mpt-7b-chat](https://huggingface.co/mosaicml/mpt-7b-chat)

### Orca-2

* Conversation template name: `zero_shot`
* [Microsoft's Orca-2-7b model at Hugging Face](https://huggingface.co/Microsoft/Orca-2-7b)
* Trained to avoid discussing a variety of potentially-dangerous and controversial topics: TBD
  * Tool can generate adversarial content that defeats those restrictions: TBD
* Will generally follow system prompt instructions that restrict information given to the user: TBD
  * Tool can generate adversarial content that defeats those restrictions: TBD

#### Specific models tested using Broken Hill:

* [Orca-2-7b](https://huggingface.co/Microsoft/Orca-2-7b)

### Phi-2

* Conversation template name: `phi2`
* [Microsoft's Phi-2 model at Hugging Face](https://huggingface.co/microsoft/phi-2)
* Trained to avoid discussing a variety of potentially-dangerous and controversial topics: **Yes**
  * Tool can generate adversarial content that defeats those restrictions: **Yes**
* Will generally follow system prompt instructions that restrict information given to the user: **Yes**
  * Tool can generate adversarial content that defeats those restrictions: **Yes**

Broken Hill includes a custom `phi2` chat template because `fschat` does not currently include one.

#### Specific models tested using Broken Hill:

* [phi-2](https://huggingface.co/microsoft/phi-2)

### Phi-3

* Conversation template name: `phi3`
* [Microsoft's Phi-3 model collection at Hugging Face](https://huggingface.co/collections/microsoft/phi-3-6626e15e9585a200d2d761e3)
* Trained to avoid discussing a variety of potentially-dangerous and controversial topics: **Yes**
  * Tool can generate adversarial content that defeats those restrictions: **Yes**
* Will generally follow system prompt instructions that restrict information given to the user: **Yes**
  * Tool can generate adversarial content that defeats those restrictions: **Yes**

Broken Hill includes a custom `phi3` chat template because `fschat` does not currently include one.

#### Special considerations

##### Phi-3-small-128k-instruct

The `Phi-3-small-128k-instruct` version of Phi-3 requires `--trust-remote-code`, even though other versions of the model (such as `Phi-3-medium-128k-instruct`) no longer require it. Additionally, that version will cause Broken Hill to crash when processed on the CPU instead of a CUDA device, with the following error:

```
Pointer argument (at 0) cannot be accessed from Triton (cpu tensor?)
```

We're researching a workaround for this.

#### Specific models tested using Broken Hill:

* [Phi-3-mini-128k-instruct](https://huggingface.co/Microsoft/Phi-3-mini-128k-instruct)
* [Phi-3-medium-128k-instruct](https://huggingface.co/Microsoft/Phi-3-medium-128k-instruct)

#### Specific models tested using tool output in `ollama`:

* phi3
* phi3:3.8b-mini-128k-instruct-q8_0
* phi3:3.8b-mini-128k-instruct-q2_K
* phi3:3.8b-mini-128k-instruct-q4_0
* phi3:3.8b-mini-128k-instruct-fp16

### Pythia

As with their [GPT-J](#gpt-j), [GPT-Neo](#gpt-neo), and [GPT-NeoX](#gpt-neox) models, Eleuther AI only publishes Pythia as base models, and (as of this writing) all Pythia variations fine-tuned for chat have been published by unrelated third parties.

#### First-party models

* Conversation template name: ``
* Trained to avoid discussing a variety of potentially-dangerous and controversial topics: TBD
  * Tool can generate adversarial content that defeats those restrictions: TBD
* Will generally follow system prompt instructions that restrict information given to the user: TBD
  * Tool can generate adversarial content that defeats those restrictions: TBD
* [Pythia GitHub repository](https://github.com/EleutherAI/pythia)

##### Specific models tested using Broken Hill:

* [pythia-14m](https://huggingface.co/EleutherAI/pythia-14m)
* [pythia-70m](https://huggingface.co/EleutherAI/pythia-70m)
* [pythia-70m-deduped](https://huggingface.co/EleutherAI/pythia-70m-deduped)
* [pythia-160m](https://huggingface.co/EleutherAI/pythia-160m)
* [pythia-410m](https://huggingface.co/EleutherAI/pythia-410m)
* [pythia-1b](https://huggingface.co/EleutherAI/pythia-1b)
* [pythia-1b-deduped](https://huggingface.co/EleutherAI/pythia-1b-deduped)
* [pythia-1.4b](https://huggingface.co/EleutherAI/pythia-1.4b)
* [pythia-2.8b](https://huggingface.co/EleutherAI/pythia-2.8b)

#### OpenAssistant (and derived) variations fine-tuned for chat purposes

* Conversation template name: `oasst_pythia`
* Trained to avoid discussing a variety of potentially-dangerous and controversial topics: Depends on variation
* Will generally follow system prompt instructions that restrict information given to the user: Depends on variation

##### Specific models tested using Broken Hill:

* [huge-roadrunner-pythia-1b-deduped-oasst](https://huggingface.co/csimokat/huge-roadrunner-pythia-1b-deduped-oasst)
  * Does not appear to be trained to avoid discussing topics
* [oasst_pythia-70m-deduped_webgpt](https://huggingface.co/WKLI22/oasst_pythia-70m-deduped_webgpt)
  * Does not appear to be trained to avoid discussing topics

### Qwen

#### First-party models

* Conversation template name: `qwen`
* [Alibaba's Qwen model family page at Hugging Face](https://huggingface.co/Qwen)
* Trained to avoid discussing a variety of potentially-dangerous and controversial topics: **Yes**
  * Tool can generate adversarial content that defeats those restrictions: TBD
* Will generally follow system prompt instructions that restrict information given to the user: **Yes**
  * Tool can generate adversarial content that defeats those restrictions: **Yes**

`fschat` includes a template for Qwen and Qwen 2, but for some reason it's named `qwen-7b-chat` specifically, and it specifies the use of an `<|endoftext|>` stop string that the models' `apply_chat_template` function does not add. As a result, Broken Hill includes a custom `qwen` template definition.

##### Specific models tested using Broken Hill:

* [Qwen-1_8B-Chat](https://huggingface.co/Qwen/Qwen-1_8B-Chat)
* [Qwen-7B-Chat](https://huggingface.co/Qwen/Qwen-7B-Chat)
* [Qwen-14B-Chat](https://huggingface.co/Qwen/Qwen-14B-Chat)
* [Qwen1.5-0.5B-Chat](https://huggingface.co/Qwen/Qwen1.5-0.5B-Chat)
* [Qwen1.5-1.8B-Chat](https://huggingface.co/Qwen/Qwen1.5-1.8B-Chat)

#### Nekomata

* Conversation template name: `qwen`

##### Specific models tested using Broken Hill:

* [nekomata-7b-instruction](https://huggingface.co/rinna/nekomata-7b-instruction)

### Qwen 2

* Conversation template name: `qwen2`
* [Alibaba's Qwen model family page at Hugging Face](https://huggingface.co/Qwen)
* Trained to avoid discussing a variety of potentially-dangerous and controversial topics: **Yes**
  * Tool can generate adversarial content that defeats those restrictions: TBD
* Will generally follow system prompt instructions that restrict information given to the user: **Yes**
  * Tool can generate adversarial content that defeats those restrictions: **Yes**

`fschat` includes a template for Qwen and Qwen 2, but for some reason it's named `qwen-7b-chat` specifically, and it specifies the use of an `<|endoftext|>` stop string that the models' `apply_chat_template` function does not add. As a result, Broken Hill includes a custom `qwen` template definition.

#### Specific models tested using Broken Hill:

* [Qwen2-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2-0.5B-Instruct)
* [Qwen2-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2-1.5B-Instruct)

### RedPajama-INCITE

* Conversation template name: `redpajama-incite`
* [RedPajama-INCITE-7B-Chat model page at Hugging Face](https://huggingface.co/togethercomputer/RedPajama-INCITE-7B-Chat)
* Trained to avoid discussing a variety of potentially-dangerous and controversial topics: TBD
  * Tool can generate adversarial content that defeats those restrictions: TBD
* Will generally follow system prompt instructions that restrict information given to the user: TBD
  * Tool can generate adversarial content that defeats those restrictions: TBD

#### Specific models tested using Broken Hill:

* [RedPajama-INCITE-7B-Chat](https://huggingface.co/togethercomputer/RedPajama-INCITE-7B-Chat)
* [RedPajama-INCITE-7B-Instruct](https://huggingface.co/togethercomputer/RedPajama-INCITE-7B-Instruct)
* [RedPajama-INCITE-Chat-3B-v1](https://huggingface.co/togethercomputer/RedPajama-INCITE-Chat-3B-v1)

### SmolLM

* Conversation template name: `smollm`
* [Hugging Face blog post introducing SmolLM](https://huggingface.co/blog/smollm)
* Trained to avoid discussing a variety of potentially-dangerous and controversial topics: TBD
  * Tool can generate adversarial content that defeats those restrictions: TBD
* Will generally follow system prompt instructions that restrict information given to the user: TBD
  * Tool can generate adversarial content that defeats those restrictions: TBD

Broken Hill includes a custom `smollm` chat template because `fschat` does not currently include one.

#### Specific models tested using Broken Hill:

* [SmolLM-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM-135M-Instruct)
* [SmolLM-360M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM-360M-Instruct)
* [SmolLM-1.7B-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM-1.7B-Instruct)

### SOLAR

* Conversation template name: `solar`
* [Hugging Face blog post introducing SmolLM](https://huggingface.co/blog/smollm)
* Trained to avoid discussing a variety of potentially-dangerous and controversial topics: TBD
  * Tool can generate adversarial content that defeats those restrictions: TBD
* Will generally follow system prompt instructions that restrict information given to the user: TBD
  * Tool can generate adversarial content that defeats those restrictions: TBD

`fschat` includes a `solar` template, but its output is missing the `### System:` header for the system prompt, so Broken Hill includes a custom `solar` chat template with that issue corrected.

#### Specific models tested using Broken Hill:

* [SOLAR-10.7B-Instruct-v1.0](https://huggingface.co/upstage/SOLAR-10.7B-Instruct-v1.0)
* [TinySolar-248m-4k-code-instruct](https://huggingface.co/upstage/TinySolar-248m-4k-code-instruct)

### StableLM

* Conversation template name: `stablelm`
* [Stability AI StableLM family GitHub repository](https://github.com/Stability-AI/StableLM)
* Trained to avoid discussing a variety of potentially-dangerous and controversial topics: **Depends**
* Will generally follow system prompt instructions that restrict information given to the user: TBD

The smaller versions of this model family don't seem to have any built-in restrictions regarding controversial topics. However, the larger versions (e.g. `stablelm-tuned-alpha-7b`) do exhibit restrictions.

#### Specific models tested using Broken Hill:

* [stablelm-tuned-alpha-3b](https://huggingface.co/stabilityai/stablelm-tuned-alpha-3b)
* [stablelm-zephyr-3b](https://huggingface.co/stabilityai/stablelm-zephyr-3b)
* [stablelm-tuned-alpha-7b](https://huggingface.co/stabilityai/stablelm-tuned-alpha-7b)

### StableLM 2

* Conversation template name: `stablelm2`
* [Stability AI StableLM family GitHub repository](https://github.com/Stability-AI/StableLM)
* Trained to avoid discussing a variety of potentially-dangerous and controversial topics: **Depends**
* Will generally follow system prompt instructions that restrict information given to the user: TBD

As discussed in [the documentation for stablelm-2-1_6b-chat](https://huggingface.co/stabilityai/stablelm-2-1_6b-chat) and [the documentation for stablelm-2-zephyr-1_6b](https://huggingface.co/stabilityai/stablelm-2-zephyr-1_6b), the smaller versions of this model family don't have any built-in restrictions regarding controversial topics.

However, the larger versions (e.g. `stablelm-2-12b-chat`) do exhibit restrictions.

#### Specific models tested using Broken Hill:

* [stablelm-2-1_6b-chat](https://huggingface.co/stabilityai/stablelm-2-1_6b-chat)
* [stablelm-2-zephyr-1_6b](https://huggingface.co/stabilityai/stablelm-2-zephyr-1_6b)
* [stablelm-2-12b-chat](https://huggingface.co/stabilityai/stablelm-2-12b-chat)

### TinyLlama

* Conversation template name: `TinyLlama`
* [TinyLlama GitHub repository](https://github.com/jzhang38/TinyLlama)
* Trained to avoid discussing a variety of potentially-dangerous and controversial topics: **Yes**
  * Tool can generate adversarial content that defeats those restrictions: TBD
* Will generally follow system prompt instructions that restrict information given to the user: **Yes**
  * Tool can generate adversarial content that defeats those restrictions: **Yes**

#### Specific models tested using Broken Hill:

* [TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)

### Vicuna

Vicuna is based on Llama, but has so many sub-variations that it's been given its own section.

#### First-party models

Despite `fschat`'s overly-specific `vicuna_v1.1` template name, versions up to 1.5 have been successfully tested in Broken Hill.

* Conversation template name: `vicuna_v1.1`
* [The Large Model Systems Organization's Vicuna web page](https://lmsys.org/blog/2023-03-30-vicuna/)
* Trained to avoid discussing a variety of potentially-dangerous and controversial topics: **Yes**
  * Tool can generate adversarial content that defeats those restrictions: TBD
* Will generally follow system prompt instructions that restrict information given to the user: TBD
  * Tool can generate adversarial content that defeats those restrictions: TBD

##### Specific models tested using Broken Hill:

* [vicuna-7b-v1.1](https://huggingface.co/lmsys/vicuna-7b-v1.1)
* [vicuna-7b-v1.3](https://huggingface.co/lmsys/vicuna-7b-v1.3)
* [vicuna-7b-v1.5](https://huggingface.co/lmsys/vicuna-7b-v1.5)
* [vicuna-13b-v1.1](https://huggingface.co/lmsys/vicuna-13b-v1.1)

#### StableVicuna

* Conversation template name: `stable-vicuna`
* [CarperAI's original version of StableVicuna](https://huggingface.co/CarperAI/stable-vicuna-13b-delta)
* ["TheBloke"'s pre-merged version of StableVicuna](https://huggingface.co/TheBloke/stable-vicuna-13B-HF)
* Trained to avoid discussing a variety of potentially-dangerous and controversial topics: **Yes**
  * Tool can generate adversarial content that defeats those restrictions: TBD
* Will generally follow system prompt instructions that restrict information given to the user: TBD
  * Tool can generate adversarial content that defeats those restrictions: TBD

##### Special considerations

StableVicuna was originally released by CarperAI as a set of weight differences to be applied to the Llama model. We have only tested it using ["TheBloke"'s pre-merged version of the model](https://huggingface.co/TheBloke/stable-vicuna-13B-HF).

##### Specific models tested using Broken Hill:

* [stable-vicuna-13B-HF](https://huggingface.co/TheBloke/stable-vicuna-13B-HF)


## Other model families that can be used with Broken Hill

These model families can be used in the tool, but publicly-available versions are not trained to handle chat-type interactions. Broken Hill can handle them in case someone runs across a derived model that's been trained for chat-like interaction. If you encounter a derived model, you'll likely need to add a custom chat template to the code to generate useful results.

### Arctic-embed

* [Snowflake's Arctic-embed collection at Hugging Face](https://huggingface.co/collections/Snowflake/arctic-embed-661fd57d50fab5fc314e4c18)

#### Specific models tested using Broken Hill:

* [snowflake-arctic-embed-s](https://huggingface.co/Snowflake/snowflake-arctic-embed-s)

### BART

* [BART GitHub page](https://github.com/facebookresearch/fairseq/tree/main/examples/bart)

#### Special considerations

BART currently requires currently requires `--max-new-tokens-final 512` (or lower) to be manually specified.

#### Specific models tested using Broken Hill:

* [bart-large-cnn](https://huggingface.co/facebook/bart-large-cnn)

### BigBird / BigBirdPegasus

* [BigBird GitHub page](https://github.com/google-research/bigbird)

#### Specific models tested using Broken Hill:

* [bigbird-pegasus-large-arxiv](https://huggingface.co/google/bigbird-pegasus-large-arxiv)
* [bigbird-pegasus-large-pubmed](https://huggingface.co/google/bigbird-pegasus-large-pubmed)

### BlenderBot

* [BlenderBot Small / 90M Hugging Face page](https://huggingface.co/facebook/blenderbot_small-90M)

Currently, only the `blenderbot_small-90M` version of BlenderBot works in Broken Hill. We are unsure why other versions of the model cause it to crash, but plan to investigate the issue.

#### Special considerations

BlenderBot currently requires `--max-new-tokens 32` (or lower) and `--max-new-tokens-final 32` (or lower).

#### Specific models tested using Broken Hill:

* [blenderbot_small-90M](https://huggingface.co/facebook/blenderbot_small-90M)
* [blenderbot-400M-distill](https://huggingface.co/facebook/blenderbot-400M-distill)
* [blenderbot-1B-distill](https://huggingface.co/facebook/blenderbot-1B-distill)
* [blenderbot-3B](https://huggingface.co/facebook/blenderbot-3B)

### GPT-1

* [Open AI's GPT-1 page at Hugging Face](https://huggingface.co/openai-community/openai-gpt)

#### Specific models tested using Broken Hill:

* [openai-gpt](https://huggingface.co/openai-community/openai-gpt)

### GPT-2

* [GPT-2 GitHub repository](https://github.com/openai/gpt-2)

#### Special considerations

GPT-2 currently requires `--max-new-tokens-final 512` (or lower) to be manually specified.

#### Specific models tested using Broken Hill:

* (OpenAI) [gpt2](https://huggingface.co/OpenAI/gpt2) (not the same model as the next entry)
* (openai-community) [gpt2](https://huggingface.co/openai-community/gpt2) (not the same model as the previous entry)
* [gpt2-medium](https://huggingface.co/openai-community/gpt2-medium)
* [gpt2-large](https://huggingface.co/openai-community/gpt2-large)
* [gpt2-xl](https://huggingface.co/openai-community/gpt2-xl)
* [tiny-gpt2](https://huggingface.co/sshleifer/tiny-gpt2)

### GPT-J

* [Eleuther AI's GPT-J 6B page at Hugging Face](https://huggingface.co/EleutherAI/gpt-j-6b)

#### Specific models tested using Broken Hill:

* [GPT-J 6B](https://huggingface.co/EleutherAI/gpt-j-6b)

### GPT-Neo

* [GPT-Neo 1.3B Hugging Face page](https://huggingface.co/EleutherAI/gpt-neo-1.3B)

#### Specific models tested using Broken Hill:

* [gpt-neo-125m](https://huggingface.co/EleutherAI/gpt-neo-125m)
* [gpt-neo-1.3B](https://huggingface.co/EleutherAI/gpt-neo-1.3B)

### Mamba

* Conversation template name: `zero_shot`
* [State Spaces' "Transformers-compatible Mamba" page at Hugging Face](https://huggingface.co/collections/state-spaces/transformers-compatible-mamba-65e7b40ab87e5297e45ae406)

#### Special considerations

* `--suppress-attention-mask` is required.
* You *must* use the "-hf" ("Transformers-compatible") variations of Mamba.
* A better conversation template to use instead of `zero_shot` would likely improve results.

#### Specific models tested using Broken Hill:

* [mamba-1.4b-hf](https://huggingface.co/state-spaces/mamba-1.4b-hf)

### OPT

* [Facebook's "OPT: Open Pre-trained Transformer Language Models" paper](https://arxiv.org/abs/2205.01068)

#### Special considerations

OPT currently requires `--max-new-tokens-final 512` (or lower) to be explicitly specified.

#### Specific models tested using Broken Hill:

* [opt-125m](https://huggingface.co/Facebook/opt-125m)
* [opt-1.3b](https://huggingface.co/Facebook/opt-1.3b)
* [opt-2.7b](https://huggingface.co/Facebook/opt-2.7b)

### Pegasus

* [Pegasus documentation](https://huggingface.co/docs/transformers/main/model_doc/pegasus)

Broken Hill can work with Pegasus again as of version 0.32, but don't expect useful results unless you're working with a trained derivative.

#### Special considerations

Pegasus requires `--max-new-tokens-final 512` (or lower).

Pegasus-X models are not currently supported.

#### Specific models tested using Broken Hill:

* [pegasus-arxiv](https://huggingface.co/google/pegasus-arxiv)
* [pegasus-cnn_dailymail](https://huggingface.co/google/pegasus-cnn_dailymail)
* [pegasus-large](https://huggingface.co/google/pegasus-large)
* [pegasus-reddit_tifu](https://huggingface.co/google/pegasus-reddit_tifu)

### Phi-1

* [Microsoft's Phi-1 model collection at Hugging Face](https://huggingface.co/collections/microsoft/phi-1-6626e29134744e94e222d572)

#### Specific models tested using Broken Hill:

* [phi-1_5](https://huggingface.co/facebook/opt-2.7b)

### RoBERTa

* [RoBERTa GitHub page](https://github.com/facebookresearch/fairseq/tree/main/examples/roberta)

#### Specific models tested using Broken Hill:

* [roberta-base](https://huggingface.co/FacebookAI/roberta-base)

### XLNet

* [xlnet-large-cased page at Hugging Face](https://huggingface.co/xlnet/xlnet-large-cased)

#### Specific models tested using Broken Hill:

* [xlnet-large-cased](https://huggingface.co/xlnet/xlnet-large-cased)

## Other model families that do not currently work with Broken Hill


### T5

* [Google T5 page at Hugging Face](https://huggingface.co/google-t5/t5-base)

No model based on T5 can currently be tested with Broken Hill, because Broken Hill does not yet support its architecture.
