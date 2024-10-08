# Broken Hill version history

# 0.33 - 2024-10-07

* Significant performance improvements, especially when using `--random-seed-comparisons`
  * Some output data has been moved around in the output JSON file to greatly reduce redundant storage of data, in order to make JSON files much smaller and faster to write to disk
  * [the "Extracting information from Broken Hill result JSON files" guide](docs/extracting_result_information.md) has been updated to reflect the changes
* Fixed a bug that caused `--rollback-on-jailbreak-count-threshold` to not actually roll back, introduced during a previous refactoring
* When `--display-failure-output` is omitted, the `unique_results` field of result data now excludes strings that are the shorter versions of longer LLM output generated during the first stage of a successful jailbreak detection, to save users the time of filtering them out manually
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