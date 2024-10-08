# Re-encoding at every iteration (the `--reencode-every-iteration` option in Broken Hill)

## Introduction

[The original demonstration produced by the authors of the "Universal and Transferable Adversarial Attacks on Aligned Language Models" paper](https://github.com/llm-attacks/llm-attacks/) maintained the adversarial content between attack iterations as a string, which necessitated encoding the string to token IDs at every iteration, so that those IDs could be used in combination with the gradient to select a variation for further testing.

In other words, at every iteration, the original demonstration performed the following operations:

1. Use the LLM tokenizer to convert the current adversarial string into a list of token IDs.
2. Use the token IDs to generate a set of variations.
3. Select the variation with the lowest loss value.
4. Convert that variation to a string and test it for jailbreak success.
5. Go back to the first step, using the new variation as the current adversarial string.

Broken Hill maintains the adversarial content as token IDs between iterations, so it doesn't *have* to perform that encode => decode => encode => decode cycle. However, you can emulate the original behaviour in Broken Hill by adding the `--reencode-every-iteration` option, because it does affect the results. Is it better, worse, or just different? We don't have enough data to be comfortable making a claim yet.

Re-encoding at every iteration is disabled by default because that approach has side effects. If all you do is add `--reencode-every-iteration`, you're likely to either run out of memory or run out of candidate adversarial content to test.

## Transforming between a string and tokens is lossy

The recurring encode => decode transformation has a side-effect: decoding a list of token IDs to a string and then encoding that string back to token IDs is not guaranteed to be a symmetric operation, and frequently results in the tokenizer producing a different list of token IDs than the initial set contained. For example, the tokenizer for Microsoft's Phi-3 LLM includes the token `<|assistant|>`, with ID 32001. If you ask the tokenizer to decode token ID 32001 to a string, it will return `<|assistant|>`. But if you include the string `<|assistant|>` in a prompt that's sent to Phi-3, it could at least theoretically be encoded as any of the following, or another variation:

* `[ "<|assistant|>" ]` / `[ 32001 ]`
* `[ "<", "|", "ass", "istant", "|", ">" ]` / `[ 29966, 29989, 465, 22137, 29989, 29958 ]`
* `[ "<", "|", "ass", "is", "tan", "t", "|", ">" ]` / `[ 29966, 29989, 465, 275, 13161, 29873, 29989, 29958 ]`
* `[ "<", "|", "ass", "is", "ta", "nt", "|", ">" ]` / `[ 29966, 29989, 465, 275, 941, 593, 29989, 29958 ]`
* `[ "<", "|", "ass", "is", "ta", "n", "t", "|", ">" ]` / `[ 29966, 29989, 465, 275, 941, 29876, 29873, 29989, 29958 ]`

The context the text is found in can also cause the encoded version to change. For example, a tokenizer might encode the word "fire" to one token ID if the previous token is a space, but a different token ID (or multiple token IDs) if the previous token is not a space; multiple adjacent tokens may become a string that is encoded to a single word. If the current adversarial content includes these two adjacent tokens:

```
[ 'office', 'immediate' ]
```

...but during the current iteration, Broken Hill finds that replacing `immediate` with `r` results in the lowest loss value:

```
[ 'office', 'r' ]
```

...the resulting decoded string may contain "officer", which the tokenizer could encode back to a single token (`officer`), reducing the number of tokens in the adversarial content by one.

## Why this can be a problem

The GCG attack is (in the absence of other controls) generally more likely to select adversarial content with more tokens for the next iteration than content with fewer tokens. In a few hundred iterations, the number can grow so high that attempting to generate the data for the next iteration will exceed the amount of device memory available to PyTorch.

## Options for preventing uncontrolled growth of the adversarial content

If you want to use `--reencode-every-iteration`, you'll generally want to limit the maximum number of tokens in the adversarial content. There are currently two options for doing that.

### Mimic the original demonstration code's approach

[During the candidate adversarial content generation step, the original demonstration code decodes each candidate to a string, then re-encodes that string back to token IDs, and requires that the length of the two lists of token IDs match](https://github.com/llm-attacks/llm-attacks/blob/098262edf85f807224e70ecd87b9d83716bf6b73/llm_attacks/minimal_gcg/opt_utils.py#L101).

You can emulate that behaviour by adding the `--attempt-to-keep-token-count-consistent` option.

### Set bounds on the number of tokens

Alternatively (or in addition to the previous option), you can use the `--adversarial-candidate-filter-tokens-min` and `--adversarial-candidate-filter-tokens-max` options to set lower and upper bounds on the number of tokens that candidate adversarial content is allowed to include. For example, to require that candidate values have between 12 and 20 tokens, `--adversarial-candidate-filter-tokens-min 12 --adversarial-candidate-filter-tokens-max 20`.

## Options for avoiding a condition where no adversarial content candidates are available

Both of the options for preventing excessive token counts are applied after candidate values are generated. This means that it's possible for every candidate value to be excluded, in which case Broken Hill may indicate that it can't find an acceptable value and exit prematurely. To keep the iterations going, the best current approach is to specify the `--add-token-when-no-candidates-returned` and/or `--delete-token-when-no-candidates-returned` options. If no valid candidates are found during a given iteration, those options will cause Broken Hill to randomly duplicate an existing token (increasing the number of tokens by one), or randomly delete an existing token (reducing the number of tokens by one), as long as the resulting count would be within the bounds defined by `--adversarial-candidate-filter-tokens-min` and `--adversarial-candidate-filter-tokens-max`.

A future version of Broken Hill will incorporate a "gamma garden" feature for applying other changes when this condition occurs, as well as other conditions.
