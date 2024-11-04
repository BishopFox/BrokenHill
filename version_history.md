# Broken Hill version history

# 0.34 - 2024-10-xx

This is the most significant update to Broken Hill since its public release in September 2024. We performed extensive refactoring in order to add the additional features and functionality. We think we've prevented any regression bugs from making their way into this release, but due to the number of changes it's possible something slipped through.

* [Broken Hill now automatically backs up the attack state at each iteration so that tests can be resumed if they're ended early, or the operator wants to continue iterating on existing results](docs/all_command-line_options.md#options-related-to-attack-state backup-and-resumption).
  * This is a much more robust option than the "start with the adversarial content from the previous test" approach documented for earlier versions of Broken Hill.
  * The state preserves e.g. all of the random number generator states, so interrupting and resuming a test should be (and is, in our testing) produces results identical to one uninterrupted test.
  * However, it *is* a brand new feature, so please let us know if you find any discrepancies.
  * If you really want to disable this feature, there is also a `--disable-state-backup` option.
* [Greatly improved console output and built-in log-generation capabilities](docs/all_command-line_options.md#output-options), so no more need to pipe to `tee`.
  * The default console output uses ANSI formatting for colour. You can disable this with `--no-ansi`.
  * Log to a file using `--log <path>`.
  * Set the message level to include in console output using `--console-level` (default: `info`).
  * Set the message level to include in log file output using `--log-level` (default: `info`).
* Added support for the following models and families. The linked models are specific versions tested, but other models from the same series or family should also work just fine. We'll be testing more of them over time.
  * [Azurro's APT family](https://huggingface.co/Azurro)
      * [APT-1B-Base](https://huggingface.co/Azurro/APT-1B-Base)
      * [APT2-1B-Base](https://huggingface.co/Azurro/APT2-1B-Base)
      * [APT3-1B-Base](https://huggingface.co/Azurro/APT3-1B-Base)
      * [APT3-1B-Instruct-v1](https://huggingface.co/Azurro/APT3-1B-Instruct-v1)  
  * GPT-NeoX
    * [gpt-neox-20b](https://huggingface.co/EleutherAI/gpt-neox-20b)
  * Mamba
    * [mamba-1.4b-hf](https://huggingface.co/state-spaces/mamba-1.4b-hf)
  * Mistral
    * [Mistral-7B-Instruct-v0.3](https://huggingface.co/MistralAI/Mistral-7B-Instruct-v0.3)
    * [Mistral-Nemo-Instruct-2407](https://huggingface.co/MistralAI/Mistral-Nemo-Instruct-2407)
	* [Daredevil-7B](https://huggingface.co/mlabonne/Daredevil-7B)
	* [Intel neural-chat-7b-v3-3](https://huggingface.co/Intel/neural-chat-7b-v3-3)
    * [NeuralDaredevil-7B](https://huggingface.co/mlabonne/NeuralDaredevil-7B)
  * Mixtral
    * [Mixtral-8x7B-Instruct-v0.1](https://huggingface.co/MistralAI/Mixtral-8x7B-Instruct-v0.1)
  * RedPajama-INCITE
    * [RedPajama-INCITE-7B-Chat](https://huggingface.co/togethercomputer/RedPajama-INCITE-7B-Chat)
    * [RedPajama-INCITE-Chat-3B-v1](https://huggingface.co/togethercomputer/RedPajama-INCITE-Chat-3B-v1)
  * Snowflake Arctic
    * [snowflake-arctic-embed-s](https://huggingface.co/Snowflake/snowflake-arctic-embed-s)
  * SOLAR
    * [SOLAR-10.7B-Instruct-v1.0](https://huggingface.co/upstage/SOLAR-10.7B-Instruct-v1.0)
    * [TinySolar-248m-4k-code-instruct](https://huggingface.co/upstage/TinySolar-248m-4k-code-instruct)
* Added support for the following models derived from model families already supported by Broken Hill:
  * Gemma-derived models
    * [Vikhr-Gemma-2B-instruct](https://huggingface.co/Vikhrmodels/Vikhr-Gemma-2B-instruct)
  * Llama-2-derived models
    * [Swallow-7b-instruct-hf](https://huggingface.co/tokyotech-llm/Swallow-7b-instruct-hf)
	* [Swallow-MS-7b-instruct-v0.1](https://huggingface.co/tokyotech-llm/Swallow-MS-7b-instruct-v0.1)
    * [youri-7b-chat](https://huggingface.co/rinna/youri-7b-chat) (derived from Llama-2)
  * Llama-3-derived models
    * [Llama-3-Swallow-8B-Instruct-v0.1](tokyotech-llm/Llama-3-Swallow-8B-Instruct-v0.1)
    * [llama-3-youko-8b-instruct](https://huggingface.co/rinna/llama-3-youko-8b-instruct) (derived from Llama-3)
  * OpenAssistant (and derived) fine-tuned variations of [Eleuther AI's Pythia model family](https://github.com/EleutherAI/pythia)
    * [huge-roadrunner-pythia-1b-deduped-oasst](https://huggingface.co/csimokat/huge-roadrunner-pythia-1b-deduped-oasst)
  * [stable-vicuna-13B-HF](https://huggingface.co/TheBloke/stable-vicuna-13B-HF) (derived from Vicuna and therefore from Llama as well)  
* Added the following custom chat templates:
  * `felladrin-llama-chat`
  * `gemma`
  * `gptneox`
  * `mistral`
  * `mistral-nemo`
  * `solar`
  * `vikhr`
* System resource utilization information is now collected at many more locations in the code than previous versions, but the default behaviour is to only display it at a few key steps.
  * Use `--verbose-resource-info` to display the data every time it's collected. This is probably only useful when debugging Broken Hill.
  * The resource utilization information is used to compile and display statistical information at the end of a test.
    * By default, a relatively short list of statistics are displayed.
    * Use `--verbose-stats` to output the complete list instead.
  * The console output formatting for resource utilization has been changed slightly to make it match the end-of-run information's language.
  * Additionally, a `--torch-cuda-memory-history-file` option has been added that uses [PyTorch's built-in CUDA profiling feature to generate a pickled blob of information that you can use to visualize details of data in memory on your CUDA device(s)](https://pytorch.org/docs/stable/torch_cuda_memory.html) during a Broken Hill run
* Replaced the `--override-fschat-templates` option with `--do-not-override-fschat-templates`, and changed the default behaviour to "use the custom Broken Hill template" due to the increasing number of custom templates that fix issues with `fschat` templates.
* Updated several custom chat templates to more closely match their models' `apply_chat_template` output or other documentation.
* Added some basic automated tests to make validating additional models and families easier and avoid regression bugs.
* Added yet another method for finding a list of tokens within a larger list of tokens after observing additional possible encoding variations by some LLMs. This allows some scenarios to be tested that would previously cause Broken Hill to crash, such as `--target-output` values beginning with "Wonderful" for Phi-3.
* Added the following additional [options, which are discussed in more detail in the "All command-line options" document](docs/all_command-line_options.md):
  * `--performance-output-file`, which writes resource/performance data at key stages in JSON format to the specified file to help tune memory utilization and other factors.  
  * `--model-parameter-info`, which displays more detailed information about the model's parameters.
  * [Option-saving and loading](docs/all_command-line_options.md#saving-and-reusing-broken-hill-options):
    * `--save-options`
	* `--load-options`
	* `--load-options-from-state`
  * *Experimental* `--model-device`, `--gradient-device`, and `--forward-device`, which may allow testing larger models more efficiently on systems with limited CUDA device memory.
  * *Experimental* `--ignore-prologue-during-gcg-operations`, which attempts to reduce the amount of memory used for creating the gradient, but may also reduce the quality of results.
  * *Experimental* `--model-data-type`, which theoretically would allow more accurate testing using models' native weight formats, but currently will probably just cause Broken Hill to crash.
* Changed the default value for `--batch-size-get-logits` from 1 to 512, based on analysis of CUDA memory profile information.
  * This significantly improves performance in some cases, and seems to use approximately the same amount of memory as a value of 1.
  * If you are trying to minimize CUDA device memory utilization as much as possible, you can still try reducing this value to 1 manually.
* A few additions to the default jailbreak detection rules.
* Changed the default value for `--max-new-tokens-final` from 16384 back to 1024 to avoid excessive delays when some LLMs go way off the rails.
* Bug fixes:
  * Fixed incorrect prompt parsing logic for fast tokenizer output that would output invalid information if the base prompt, target output, or adversarial content appeared more than once in the prompt (e.g. if the target output was a specific message from the system prompt).
  * Corrected the default behaviour for tokenizers without a padding token defined to default to the "end of string" token, and to pad from the left.
    * The new `--missing-pad-token-padding-side` option can be used to override this, e.g. `--missing-pad-token-padding-side right`, but you probably don't want to do that.
	* If you want to default to whatever token Torch would pick (which is usually "end of string" anyway), use `--missing-pad-token-replacement default`.
  * Added special corner-case handling for Qwen models that define non-standard, redundant, proprietary equivalents of the `dtype` parameter when loading a model.
    * The affected models default to converting the weights to `bfloat16` on the fly if the proprietary additional parameters are not included.
	* The corner-case handling overrides this behaviour and loads the weights in the correct format.
  * Fixed a bug that could cause a crash instead of displaying a warning during jailbreak self-testing.
  * Fixed a bug that would cause a crash when testing for jailbreaks if the LLM output was an empty string or consisted only of whitespace.
  * Fixed a bug that caused `--adversarial-candidate-filter-regex` to not be applied correctly.
  * Fixed a bug that incorrectly reported a chat template mismatch when a system message and/or template messages were specified.
  * Updated `ollama_wrapper.sh` to correctly handle adversarial content containing shell metacharacters.
  * Memory utilization statistics now correctly display process-level virtual memory instead of duplicating the physical memory figure.
  * Minor update to the logic used to generate the "Current input string:" output text so that trailing delimiters between the input and the LLM role indicator are omitted.
  * Minor correction to the custom `stablelm2` chat template.
  * Many other lesser fixes.
* Removed a bunch of leftover code that was no longer used to make it slightly easier to understand what was going on.


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