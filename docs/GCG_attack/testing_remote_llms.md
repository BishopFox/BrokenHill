# Thoughts on testing remote LLMs

This document discusses possible approaches for using this tool to develop attack payloads for LLMs that are hosted remotely, i.e. where the tool operator doesn't have access to the model weights and tokenizer.

## Convince the remote LLM to describe itself, then replicate it locally

This approach uses the tool itself to bootstrap an attack on the remote LLM.

### Generate fingerprinting/"describe yourself" payloads for as many LLMs as possible

### Send the fingerprinting payloads to the remote LLM

### Use the results obtained so far to replicate the LLM locally

### Generate a second set of payloads to target the replicated LLM

### Test the second set of payloads against the original LLM

## Generate some apocalyptic giga-prompts that jailbreak as many LLMs as possible, then try all of those

TKTK.

But really, this is actually a subset of the first approach. Just follow the first part, generating prompts for as many LLMs as possible.


