# Broken Hill version history

# 0.34 - 2024-10-xx

* Added support for the following models:
  * 
* Added the following custom chat templates:
  * `felladrin-llama-chat`
* Added yet another method for finding a list of tokens within a larger list of tokens after observing additional possible encoding variations by some LLMs. This allows some scenarios to be tested that would previously cause Broken Hill to crash, such as `--target-output` values beginning with "Wonderful" for Phi-3.
* Added experimental `--ignore-prologue-during-gcg-operations` option.
* A few additions to the default jailbreak detection rules.
* Changed the default value for `--max-new-tokens-final` from 16384 back to 1024 to avoid excessive delays when some LLMs go way off the rails.
* Bug fixes:
  * Fixed incorrect prompt parsing logic for fast tokenizer output that would output invalid information if the base prompt, target output, or adversarial content appeared more than once in the prompt (e.g. if the target output was a specific message from the system prompt).
  * Fixed a bug that could cause a crash instead of displaying a warning during jailbreak self-testing.
  * Fixed a bug that would cause a crash when testing for jailbreaks if the LLM output was an empty string or consisted only of whitespace.
  * Fixed a bug that caused `--adversarial-candidate-filter-regex` to not be applied correctly.
  * Fixed a bug that incorrectly reported a chat template mismatch when a system message and/or template messages were specified.
  * Updated `ollama_wrapper.sh` to correctly handle adversarial content containing shell metacharacters.
  * Minor update to the logic used to generate the "Current input string:" output text so that trailing delimiters between the input and the LLM role indicator are omitted.
  * Minor correction to the custom `stablelm2` chat template.

# 0.33 - 2024-10-08

* Significant performance improvements, especially when using `--random-seed-comparisons`
  * Some output data has been moved around in the output JSON file to greatly reduce redundant storage of data, in order to make JSON files much smaller and faster to write to disk
  * [the "Extracting information from Broken Hill result JSON files" guide](docs/extracting_result_information.md) has been updated to reflect the changes
* Fixed a bug that caused `--rollback-on-jailbreak-count-threshold` to not actually roll back, introduced during a previous refactoring
* When `--display-failure-output` is omitted, the `unique_results` field of result data now excludes strings that are the shorter versions of longer LLM output generated during the first stage of a successful jailbreak detection, to save users the time of filtering them out manually
* `--attempt-to-keep-token-count-consistent` now more closely mimics the check used in [the original demonstration produced by the authors of the "Universal and Transferable Adversarial Attacks on Aligned Language Models" paper](https://github.com/llm-attacks/llm-attacks/)
* Improvements to profanity list

# 0.32 - 2024-10-02

* Added the following options:
  * `--exclude-language-names-except` and `--list-language-tags`
  * `--suppress-attention-mask`
  * `--temperature-range`
* Improved handling of conversation templates with priming messages
* Re-enabled fast tokenizer support after fixing bugs in related code
* Improved handling of generated conversation data
* Performance improvements, especially during start-up
* Improved randomization handling
* Added basic support for Mamba
* Verified that Pegasus more or less works again
* Reorganized some of the code

# 0.31 - 2024-09-26  

* Added Ollama wrapper and associated example
* Improved randomization handling
* Bug fixes

# 0.30 - 2024-09-23

First public release.