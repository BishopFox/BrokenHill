# Thoughts on testing remote LLMs

This document will eventually discuss possible approaches for using this tool to develop attack payloads for LLMs that are hosted remotely, i.e. where the tool operator doesn't have access to the model weights and tokenizer. It is currently only a sketch.

## Table of contents

1. [Convince the remote LLM to describe itself, replicate its configuration locally, then generate payloads](#convince-the-remote-llm-to-describe-itself-replicate-its-configuration-locally-then-generate-payloads)
1. [Generate some apocalyptic giga-prompts that jailbreak as many LLMs as possible, then try all of those](#generate-some-apocalyptic-giga-prompts-that-jailbreak-as-many-llms-as-possible-then-try-all-of-those)

## Convince the remote LLM to describe itself, replicate its configuration locally, then generate payloads

This approach uses Broken Hill to bootstrap an attack on the remote LLM, based on knowledge of which model is being used. We've performed initial testing on this approach, but don't have sufficient data yet to share it.

### Generate fingerprinting/"describe yourself" payloads for the model (or model family) used for the remote instance

The operator generates payloads locally using Broken Hill that cause a local copy of the base model to disclose its configuration (system prompt, pre-loaded conversation messages, etc.). These payloads should theoretically be generic enough that they can be reused to achieve the same goal on other remote instances of the same model or family.

### Send the fingerprinting payloads to the remote LLM

### Use the results obtained so far to replicate the LLM locally

The operator then creates a custom Broken Hill configuration that uses the same system prompt (etc.) information.

### Generate a second set of payloads to target the replicated LLM

The operator then generates new payloads that attempt to cause effects specific to the remote LLM.

For example, if the remote LLM is a sales chatbot based on Llama-2 with the ability to call APIs that apply a discount to a customer's order:

1. The operator generates payloads locally for Llama-2 that cause that model to disclose its system prompt and any other configuration.
2. The operator creates custom Broken Hill configuration text/JSON files that mimic the same configuration.
3. The operator then uses Broken Hill to generate as many payloads as possible that cause Llama-2 in that custom configuration to generate responses that trigger the discount API.

### Test the second set of payloads against the original LLM

Finally, the operator sends the new payloads to the remote LLM.

## Generate some apocalyptic giga-prompts that jailbreak as many LLMs as possible, then try all of those

This is a subset of the first approach. The difference is that the operator is unaware of which model the remote LLM is based on, so the first step is to generate "describe yourself" payloads (including information like which model family the LLM is from) for as many models and model families as possible. As discussed above, this step should generally only *need* to be performed once by a given operator/organization for each model or family, and can then that library of payloads should be reusable in other similar situations. Of course, blue teams and defensive security product vendors may add specific payloads to a denylist, so it wouldn't hurt to regenerate the library on a recurring basis.

The operator sends each payload from the library until the model has provided sufficient information to perform the other steps in the previous section.
