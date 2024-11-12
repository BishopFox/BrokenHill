# Results on other graphics hardware

So far, Broken Hill has only been tested on a GeForce RTX 4090 and a GeForce RTX 4070. This document contains placeholders for some other hardware of potential interest.

## Nvidia GPUs with more than 24 GiB of VRAM

### Nvidia A100

This is a cloud/hosting-provider GPU that costs a lot of money and is for computation *only* - you won't be playing *Returnal* or *Alan Wake 2* with the settings maxed out on this device.

Broken Hill should run fine on this hardware, but we haven't tried it yet. We'll try it on a rented cloud instance some time soon and update this page.

### Nvidia H100

This is a cloud/hosting-provider GPU that costs a lot of money and is for computation *only* - you won't be playing *Returnal* or *Alan Wake 2* with the settings maxed out on this device.

Broken Hill should run fine on this hardware, but we haven't tried it yet. We'll try it on a rented cloud instance some time soon and update this page.

### Nvidia RTX 5000 / 5880 / 6000

Nvidia's workstation-class GPU line includes several options with relatively large amounts of VRAM. For the Ampere generation of hardware (same generation as the GeForce RTX 3xxx consumer GPUs), the RTX 6000 included 48 GB of VRAM. For the Ada generation (same generation as the GeForce RTX 4xxx consumer GPUs), the RTX 5000 includes 32 GiB of VRAM, while the 5880 and 6000 include 48 GiB. The Ada-based cards are (as of this writing) US$4,000 - $7,200.

Broken Hill should run fine on this hardware, but we haven't tried it yet.

### Nvidia TITAN V CEO Edition

This is an extremely rare older-generation (GeForce GT/GTX 1xxx era) consumer GPU that included 32 GiB of VRAM. [Performance-wise, it should sit somewhere in-between high-end GTX 2xxx and low-end GTX 3xxx era GPUs](https://www.techpowerup.com/gpu-specs/titan-v-ceo-edition.c3277), but the extra 8 GiB of VRAM would likely allow attacks against some models that are too big for 24 GiB.

Because it's built on technology three generations older than the reference RTX 4090 hardware, we're not sure if modern versions of PyTorch support it correctly. But if you're in possession of one of the ~20 of these cards in existence and want to try it out, please let us know how it goes!

## Nvidia GPUs with less than 24 GiB of VRAM

Most LLMs of interest plus the associated data for the attack won't fit into 16 GiB or less of VRAM. But if we discover any - and test it on a GPU with that much VRAM - we'll update this section. If you find any yourself and test it successfully before we do, we'd love to hear about it.

## Non-Nvidia GPUs

[The GCG attack itself (at least as implemented in Broken Hill) cannot currently run on any non-Nvidia GPUs, due to lack of support in PyTorch](FAQ.md).
