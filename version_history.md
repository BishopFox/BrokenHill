# Broken Hill version history

# 0.33 - 2024-10-07

* Fixed a bug related to `--rollback-on-jailbreak-count-threshold`
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