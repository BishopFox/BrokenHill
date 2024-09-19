# Broken Hill PyTorch device memory requirements

This document currently lists rough estimates for requirement device memory based on ad hoc testing. It will be updated with additional information and more specific figures as we collect them.

Note: as discussed elsewhere in the documentation, Broken Hill currently only supports CUDA devices, meaning that the memory required is on an Nvidia GPU. i.e. to use Broken Hill to perform a GCG attack against a model with 2 billion parameters, you will need an Nvidia GPU with at least 24 GiB of VRAM.

## Models with less than 2 billion parameters

* 24 GiB should be more than sufficient to perform any testing using Broken Hill

## Models with 2 billion parameters

* 24 GiB is generally sufficient to perform the GCG attack
** Reducing the number of adversarial candidates using `--new-adversarial-value-candidate-count` to 16 or even 8 may be required for lengthy target strings or large numbers of adversarial tokens

## Models with 7 billion parameters

* 24 GiB is sufficient to perform Broken Hill self-tests, but generally not perform the GCG attack
