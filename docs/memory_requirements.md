# Broken Hill PyTorch device memory requirements

This document currently lists rough estimates for requirement device memory based on ad hoc testing. It will be updated with more specific figures as we collect additional data.

Note: as discussed elsewhere in the documentation, Broken Hill currently only supports CUDA devices, meaning that the memory required is on an Nvidia GPU. i.e. to use Broken Hill to perform a GCG attack against a model with 2 billion parameters, you will need an Nvidia GPU with at least 24GiB of VRAM.

## 2 billion parameter models

* 24GiB is generally sufficient to perform the GCG attack
** Reducing the number of adversarial candidates using `--new-adversarial-value-candidate-count` to 16 or even 8 may be required for lengthy target strings or large numbers of adversarial tokens

## 7 billion parameter models

* 24GiB is sufficient to perform Broken Hill self-tests
