#!/bin/env python3

script_name = "gcg-attack.py"
script_version = "0.13"
script_date = "2024-08-09"

def get_script_description():
    result = 'Performs a "Greedy Coordinate Gradient" (GCG) attack against various large language models (LLMs), as described in the paper "Universal and Transferable Adversarial Attacks on Aligned Language Models" by Andy Zou1, Zifan Wang, Nicholas Carlini, Milad Nasr, J. Zico Kolter, and Matt Fredrikson, representing Carnegie Mellon University, the Center for AI Safety, Google DeepMind, and the Bosch Center for AI.'
    result += "\n"
    result += "Originally based on the demo.ipynb notebook included in the https://github.com/llm-attacks/llm-attacks repository."
    result += "\n"
    result += "*Heavily* modified by Ben Lincoln, Bishop Fox."
    result += "\n"
    result += f"version {script_version}, {script_date}"    
    return result

def get_short_script_description():
    result = 'Based on code by Andy Zou1, Zifan Wang, Nicholas Carlini, Milad Nasr, J. Zico Kolter, and Matt Fredrikson.'
    result += "\n"
    result += "*Heavily* modified by Ben Lincoln, Bishop Fox."
    return result

import argparse
import datetime
import fastchat.conversation as fcc
import gc
import locale
import json
import math
import numpy as np
import os
import pathlib
import psutil
import re
import shutil
import sys
import tempfile
import time
import torch
import torch.nn as nn
import torch.quantization as tq
import traceback

from llm_attacks_bishopfox import get_effective_max_token_value_for_model_and_tokenizer
from llm_attacks_bishopfox import get_embedding_layer
from llm_attacks_bishopfox import get_nonascii_token_list
from llm_attacks_bishopfox import get_token_denylist
from llm_attacks_bishopfox import get_token_list_as_tensor
from llm_attacks_bishopfox.attack.attack_classes import AttackResultInfo
from llm_attacks_bishopfox.attack.attack_classes import FakeException
from llm_attacks_bishopfox.attack.attack_classes import gcg_attack_params
from llm_attacks_bishopfox.attack.attack_classes import PyTorchDevice
from llm_attacks_bishopfox.minimal_gcg.opt_utils import get_decoded_token
from llm_attacks_bishopfox.minimal_gcg.opt_utils import get_decoded_tokens
from llm_attacks_bishopfox.minimal_gcg.opt_utils import get_filtered_cands
from llm_attacks_bishopfox.minimal_gcg.opt_utils import get_logits
from llm_attacks_bishopfox.minimal_gcg.opt_utils import get_missing_pad_token_names
from llm_attacks_bishopfox.minimal_gcg.opt_utils import load_model_and_tokenizer
from llm_attacks_bishopfox.minimal_gcg.opt_utils import sample_control
from llm_attacks_bishopfox.minimal_gcg.opt_utils import target_loss
from llm_attacks_bishopfox.minimal_gcg.opt_utils import token_gradients
from llm_attacks_bishopfox.minimal_gcg.string_utils import SuffixManager
from llm_attacks_bishopfox.minimal_gcg.string_utils import get_default_generic_role_indicator_template
from llm_attacks_bishopfox.minimal_gcg.string_utils import load_conversation_template
from llm_attacks_bishopfox.util.util_functions import get_elapsed_time_string
from llm_attacks_bishopfox.util.util_functions import get_file_content
from llm_attacks_bishopfox.util.util_functions import get_file_content
from llm_attacks_bishopfox.util.util_functions import get_file_content
from llm_attacks_bishopfox.util.util_functions import get_now
from llm_attacks_bishopfox.util.util_functions import get_time_string
from llm_attacks_bishopfox.util.util_functions import numeric_string_to_int
from llm_attacks_bishopfox.util.util_functions import safely_write_text_output_file
from llm_attacks_bishopfox.util.util_functions import str2bool
from llm_attacks_bishopfox.util.util_functions import update_elapsed_time_string
from torch.quantization import quantize_dynamic
from torch.quantization.qconfig import float_qparams_weight_only_qconfig

# threshold for warning the user if the specified PyTorch device already has more than this percent of its memory reserved
# 0.1 = 10%
torch_device_reserved_memory_warning_threshold = 0.1

locale.setlocale(locale.LC_ALL, '')

# Workaround for glitchy Protobuf code somewhere
# See https://stackoverflow.com/questions/75042153/cant-load-from-autotokenizer-from-pretrained-typeerror-duplicate-file-name
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"]="python"

def check_pytorch_devices(attack_params):
    all_devices = {}
    devices_above_threshold = {}
    if "cuda"  in attack_params.device:
        for i in range(torch.cuda.device_count()):
            d = PyTorchDevice.from_cuda_device_number(i)
            all_devices[d.device_name] = d
            if d.total_memory_utilization > torch_device_reserved_memory_warning_threshold:
                devices_above_threshold[d.device_name] = d
    device_names = list(all_devices.keys())
    if len(device_names) > 0:
        device_names.sort()
        message = f"Available PyTorch devices for the back-end in use:\n"
        for dn in device_names:
            d = all_devices[dn]
            message += f"\t{d.device_name} - {d.device_display_name}\n"
            message += f"\t\tTotal memory: {d.total_memory:n}\n"
            message += f"\t\tMemory in use across the entire device: {d.gpu_used_memory:n}\n"
            message += f"\t\tCurrent memory utilization for the device as a whole: {d.total_memory_utilization:.0%}\n"
        print(message)
    above_threshold_device_names = list(devices_above_threshold.keys())
    if len(above_threshold_device_names) > 0:
        above_threshold_device_names.sort()
        warning_message = f"Warning: the following PyTorch devices have more than {torch_device_reserved_memory_warning_threshold:.0%} of their memory reserved:\n"
        for dn in above_threshold_device_names:
            d = devices_above_threshold[dn]
            warning_message += f"\t{d.device_name} ({d.device_display_name}): {d.total_memory_utilization:.0%}\n"
        warning_message += f"If you encounter out-of-memory errors when using this tool, consider suspending other processes that use GPU resources, to maximize the amount of memory available to PyTorch. For example, on Linux desktops with a GUI enabled, consider switching to a text-only console to suspend the display manager and free up the associated VRAM. On Debian, Ctrl-Alt-F2 switches to the second console, and Ctrl-Alt-F1 switches back to the default console.\n"
        print(warning_message)

def print_stats(attack_params):
    display_string = "---\n"
    print(f"System resource statistics")
    process_mem_info = psutil.Process().memory_full_info()
    process_physical_memory = process_mem_info.rss
    process_virtual_memory = process_mem_info.vms
    process_swap = None
    if hasattr(process_mem_info, "swap"):
        process_swap = process_mem_info.swap
    system_mem_info = psutil.virtual_memory()
    system_physical_memory = system_mem_info.total
    system_available_memory = system_mem_info.available
    system_in_use_memory = system_physical_memory - system_available_memory
    #system_memory_util_percent = system_mem_info.percent
    system_memory_util_percent = float(system_in_use_memory) / float(system_physical_memory)
    display_string += f"System:\n"
    display_string += f"\tTotal physical memory: {system_physical_memory:n} bytes\n"
    display_string += f"\tMemory in use: {system_in_use_memory:n} bytes\n"
    display_string += f"\tAvailable memory: {system_available_memory:n} bytes\n"
    display_string += f"\tMemory utilization: {system_memory_util_percent:.0%} bytes\n"

    if "cuda"  in attack_params.device:
        for i in range(torch.cuda.device_count()):
            d = PyTorchDevice.from_cuda_device_number(i)            
            display_string += f"CUDA device {d.device_name} - {d.device_display_name}\n"
            display_string += f"\tTotal memory: {d.total_memory:n} bytes\n"
            display_string += f"\tMemory in use across the entire device: {d.gpu_used_memory:n}\n"
            display_string += f"\tMemory available across the entire device: {d.available_memory:n}\n"
            display_string += f"\tCurrent memory utilization for the device as a whole: {d.total_memory_utilization:.0%}\n"
            display_string += f"\tDevice memory reserved by this process: {d.process_reserved_memory:n} bytes\n"
            display_string += f"\tDevice memory utilization for this process: {d.process_memory_utilization:.0%}\n"
            display_string += f"\tDevice reserved allocated memory for this process: {d.process_reserved_allocated_memory:n} bytes\n"
            display_string += f"\tDevice reserved unallocated memory for this process: {d.process_reserved_unallocated_memory:n} bytes\n"
            display_string += f"\tReserved memory utilization within this process' reserved memory space: {d.process_reserved_utilization:.0%}\n"
    # TKTK mps equivalent of the CUDA code for the day when the PyTorch Metal back-end supports the necessary functionality
    if process_swap is not None:
        if process_swap > 0:
            display_string += f"Warning: this process has {process_swap:n} bytes swapped to disk. If you are encountering poor performance, it may be due to insufficient system RAM.\n"
    display_string += "---\n"
    print(display_string)


def generate(attack_params,model, tokenizer, input_ids, assistant_role_slice, gen_config=None):
    if gen_config is None:
        gen_config = model.generation_config
        gen_config.max_new_tokens = attack_params.generation_max_new_tokens

    #if gen_config.max_new_tokens > 32:
    #    print('WARNING: max_new_tokens > 32 may cause testing to slow down.')
        
    input_ids = input_ids[:assistant_role_slice.stop].to(model.device).unsqueeze(0)
    attn_masks = torch.ones_like(input_ids).to(model.device)
    output_ids = model.generate(input_ids, 
                                attention_mask=attn_masks, 
                                generation_config=gen_config,
                                pad_token_id=tokenizer.pad_token_id)[0]

    return output_ids[assistant_role_slice.stop:]

def check_for_attack_success(attack_params, model, tokenizer, input_ids, assistant_role_slice, test_prefixes, gen_config=None):
    generated_tokens = generate(attack_params,
                                        model, 
                                        tokenizer, 
                                        input_ids, 
                                        assistant_role_slice, 
                                        gen_config=gen_config)
                                        
    gen_str = tokenizer.decode(generated_tokens).strip()
                                        
    jailbroken = True
    # if the output contains one of the negative test strings, flag it as not jailbroken
    for prefix in attack_params.negative_output_strings:
        if prefix in gen_str:
            jailbroken = False
            break
    # ...but if it contains one of the positive test strings, flag it as jailbroken again
    for prefix in attack_params.positive_output_strings:
        if prefix in gen_str:
            jailbroken = True
            break
    # Additional checks to filter out garbage
    if jailbroken and not attack_params.jailbreak_minimum_sequential_letters_regex.search(gen_str):
        jailbroken = False
    #print(f"Jailbroken: {jailbroken} for generated string '{gen_str}'")
    return jailbroken, gen_str, generated_tokens

def get_current_input_and_output_tokens(attack_params, model, tokenizer, suffix_manager, adversarial_string):
    gen_config = model.generation_config
    current_max_new_tokens = gen_config.max_new_tokens
    gen_config.max_new_tokens = attack_params.full_decoding_max_new_tokens

    #input_ids = suffix_manager.get_input_ids(adv_string = adversarial_string, update_self_values = False).to(attack_params.device)
    #print(f"[get_current_input_and_output_tokens] Debug: calling get_input_ids with adversarial_string = '{adversarial_string}'")
    input_id_data = suffix_manager.get_input_ids(adv_string = adversarial_string, force_python_tokenizer = attack_params.force_python_tokenizer)
    input_ids = input_id_data.get_input_ids_as_tensor().to(attack_params.device)
    #input_ids = input_ids[:input_id_data.slice_data.assistant_role.stop].to(model.device).unsqueeze(0)
    input_ids_sliced = input_ids[:input_id_data.slice_data.assistant_role.stop]
    input_ids_converted = input_ids_sliced.to(model.device).unsqueeze(0)
    attn_masks = torch.ones_like(input_ids_converted).to(model.device)
    output_ids = model.generate(input_ids_converted, 
                                attention_mask=attn_masks, 
                                generation_config=gen_config,
                                pad_token_id=tokenizer.pad_token_id)[0]

    #input_tokens = output_ids[suffix_manager._goal_slice.start:suffix_manager._goal_slice.stop]
    #input_tokens = output_ids[suffix_manager._goal_slice.start:suffix_manager._control_slice.stop]
    #output_tokens_all_decoded = get_decoded_tokens(tokenizer, output_ids)
    #print(f"[get_current_input_and_output_tokens] Debug: output_ids = '{output_ids}', output_tokens_all_decoded = '{output_tokens_all_decoded}'")

    input_tokens = output_ids[input_id_data.slice_data.goal.start:input_id_data.slice_data.control.stop]
    #input_tokens = output_ids[suffix_manager._goal_slice.start:(suffix_manager._assistant_role_slice.start - 1)]
    #output_tokens = output_ids[suffix_manager._assistant_role_slice.stop:]

    # Using target.start will cause one token to be discarded if the generated prompt contains a newline after the assistant role stop and the model-generated full output does not, which is the case with Gemma
    #output_ids_output_only = output_ids[input_id_data.slice_data.target.start:]
    output_ids_output_only = output_ids[input_id_data.slice_data.assistant_role.stop:]
    
    #output_tokens_decoded = get_decoded_tokens(tokenizer, output_tokens)
    #input_tokens_decoded = get_decoded_tokens(tokenizer, input_tokens)
    #print(f"[get_current_input_and_output_tokens] Debug: input_tokens = '{input_tokens}', input_tokens_decoded = '{input_tokens_decoded}', output_tokens = '{output_tokens}', output_tokens_decoded = '{output_tokens_decoded}'")
    return input_id_data.input_ids, input_tokens, output_ids_output_only, output_ids

def get_input_and_output_strings(attack_params, model, tokenizer, suffix_manager, adversarial_string, input_label = "Current input", output_label = "Current output"):
    full_input_token_ids, generation_input_token_ids, output_token_ids, full_generation_token_ids = get_current_input_and_output_tokens(attack_params, model, tokenizer, suffix_manager, adversarial_string)
    #decoded_full_input = tokenizer.decode(full_input_token_ids).strip()
    #decoded_input = tokenizer.decode(generation_input_token_ids).strip()
    #decoded_output = tokenizer.decode(output_token_ids).strip()
    #decoded_full_generation_tokens = tokenizer.decode(full_generation_token_ids).strip()
    #decoded_full_input = tokenizer.decode(full_input_token_ids)
    #decoded_full_input_array = get_decoded_tokens(tokenizer, full_input_token_ids)
    decoded_input = tokenizer.decode(generation_input_token_ids).strip()
    decoded_output = tokenizer.decode(output_token_ids).strip()
    #decoded_full_generation_tokens = tokenizer.decode(full_generation_token_ids)
    #decoded_full_generation_tokens_array = get_decoded_tokens(tokenizer, full_generation_token_ids)

    result = f"{input_label}: '{decoded_input}'\n\n{output_label}: '{decoded_output}'"
    #result = f"{input_label}: '{decoded_input}'\n\n{output_label}: '{decoded_output}'\n\nGeneration full input: '{decoded_full_input}'\n\nFull generation text: '{decoded_full_generation_tokens}'"
    #result = f"{input_label}: '{decoded_input}'\n\n{output_label}: '{decoded_output}'\n\nGeneration full input: '{decoded_full_input}'\n\nFull generation text: '{decoded_full_generation_tokens}'\n\nGeneration full input as array: '{decoded_full_input_array}'\n\nFull generation text as array: '{decoded_full_generation_tokens_array}', full_generation_token_ids = '{full_generation_token_ids}'"
    return result

def main(attack_params):

    # Parameter validation, warnings, and errors
    device_warning = False
    if len(attack_params.device) > 2 and attack_params.device[0:3] == "cpu":
        device_warning = True
    if len(attack_params.device) < 4 or attack_params.device[0:4] != "cuda":
        device_warning = True
    if device_warning:
        print(f"Warning: the specified device ('{attack_params.device}') is not recommended. This tool is heavily optimized for CUDA. It will run very slowly or not at all on other hardware. Expect run times about 100 times slower on CPU hardware, for example.")

    if (not attack_params.base_prompt) or (not attack_params.target_output):
        print(f"Error: a base prompt and a target must be specified, either as distinct values, or using the --auto-target option to set both.")
        sys.exit(1)

    user_aborted = False


    # Initial setup based on configuration
    # Set random seeds
    # NumPy
    np.random.seed(attack_params.np_random_seed)
    # PyTorch
    torch.manual_seed(attack_params.torch_manual_seed)
    # CUDA
    torch.cuda.manual_seed_all(attack_params.torch_cuda_manual_seed_all)

    start_dt = get_now()
    start_ts = get_time_string(start_dt)
    print(f"Starting at {start_ts}")

    print_stats(attack_params)
    
    successful_attacks = []
    adversarial_string = attack_params.initial_adversarial_string
    model = None
    tokenizer = None
    suffix_manager = None
    # keep two arrays to avoid having to convert every item to JSON every iteration
    json_data = []
    attack_data = []
    # keep another array to track adversarial values 
    adversarial_values = []
    
    try:
        print(f"Loading model from '{attack_params.model_path}'")
        model, tokenizer = load_model_and_tokenizer(attack_params.model_path, 
                                tokenizer_path = attack_params.tokenizer_path,
                                low_cpu_mem_usage = attack_params.low_cpu_mem_usage, 
                                use_cache = attack_params.use_cache,
                                dtype = torch.float16,
                                trust_remote_code = attack_params.load_options_trust_remote_code,
                                ignore_mismatched_sizes = attack_params.load_options_ignore_mismatched_sizes,
                                enable_hardcoded_tokenizer_workarounds = attack_params.enable_hardcoded_tokenizer_workarounds,
                                missing_pad_token_replacement = attack_params.missing_pad_token_replacement,
                                device=attack_params.device)
        print_stats(attack_params)
        
        attack_params.generation_max_new_tokens = get_effective_max_token_value_for_model_and_tokenizer("--max-new-tokens", model, tokenizer, attack_params.generation_max_new_tokens)
        attack_params.full_decoding_max_new_tokens = get_effective_max_token_value_for_model_and_tokenizer("--max-new-tokens-final", model, tokenizer, attack_params.full_decoding_max_new_tokens)

        
        token_denylist = get_token_denylist(tokenizer, attack_params.not_allowed_token_list, device=attack_params.device, filter_nonascii_tokens = attack_params.exclude_nonascii_tokens, filter_special_tokens = attack_params.exclude_special_tokens, filter_additional_special_tokens = attack_params.exclude_additional_special_tokens, filter_whitespace_tokens = attack_params.exclude_whitespace_tokens, token_regex = attack_params.get_token_filter_regex())        
        
        #print(f"Debug: token_denylist = '{token_denylist}'")
        not_allowed_tokens = None
        if len(token_denylist) > 0:
            #not_allowed_tokens = get_token_list_as_tensor(token_denylist, device=attack_params.device)
            not_allowed_tokens = get_token_list_as_tensor(token_denylist, device='cpu')
        #not_allowed_tokens = get_token_list_as_tensor(tokenizer, token_denylist, device='cpu')
        #print(f"Debug: not_allowed_tokens = '{not_allowed_tokens}'")

        original_model_size = 0

        if attack_params.display_model_size:
            original_model_size = get_model_size(model)
            print(f"Model size: {original_model_size}")

        if attack_params.quantization_dtype:
            if attack_params.enable_static_quantization:
                print("This script only supports quantizing using static or dynamic approaches, not both at once")
                sys.exit(1)
            print(f"Quantizing model to '{attack_params.quantization_dtype}'")    
            #model = quantize_dynamic(model=model, qconfig_spec={nn.LSTM, nn.Linear}, dtype=quantization_dtype, inplace=False)
            model = quantize_dynamic(model=model, qconfig_spec={nn.LSTM, nn.Linear}, dtype=attack_params.quantization_dtype, inplace=True)
            #print_stats(attack_params)

        if attack_params.enable_static_quantization:
            backend = "qnnpack"
            print(f"Quantizing model using static backend {backend}") 
            torch.backends.quantized.engine = backend
            model.qconfig = tq.get_default_qconfig(backend)
            #model.qconfig = float_qparams_weight_only_qconfig
            # disable quantization of embeddings because quantization isn't really supported for them
            model_embeds = get_embedding_layer(model)
            model_embeds.qconfig = float_qparams_weight_only_qconfig
            model = tq.prepare(model, inplace=True)
            model = tq.convert(model, inplace=True)

        if attack_params.conversion_dtype:
            model = model.to(attack_params.conversion_dtype)

        if attack_params.quantization_dtype or attack_params.enable_static_quantization or attack_params.conversion_dtype:
            print("Warning: you've enabled quantization and/or type conversion, which are unlikely to work for the foreseeable future due to PyTorch limitations. Please see my comments in the source code for this tool.")
            if attack_params.display_model_size:
                quantized_model_size = get_model_size(model)
                size_factor = float(quantized_model_size) / float(original_model_size) * 100.0
                size_factor_formatted = f"{size_factor:.2f}%"
                print(f"Model size after reduction: {quantized_model_size} ({size_factor_formatted} of original size)")
            
        #print(f"Loading conversation template '{attack_params.template_name}'")
        #conv_template = load_conversation_template(attack_params.template_name, generic_role_indicator_template = attack_params.generic_role_indicator_template, system_prompt=attack_params.custom_system_prompt, clear_existing_template_conversation=attack_params.clear_existing_template_conversation, conversation_template_messages=attack_params.conversation_template_messages)
        conv_template = load_conversation_template(attack_params.model_path, template_name = attack_params.template_name, generic_role_indicator_template = attack_params.generic_role_indicator_template, system_prompt=attack_params.custom_system_prompt, clear_existing_template_conversation=attack_params.clear_existing_template_conversation, conversation_template_messages=attack_params.conversation_template_messages)
        if attack_params.template_name is not None:
            if conv_template.name != attack_params.template_name:
                print(f"Warning: the template '{attack_params.template_name}' was specified, but fastchat returned the template '{conv_template.name}' in response to that value.")
        print(f"Conversation template: '{conv_template.name}'")
        print(f"Conversation template sep: '{conv_template.sep}'")
        print(f"Conversation template sep2: '{conv_template.sep2}'")
        print(f"Conversation template roles: '{conv_template.roles}'")
        print(f"Conversation template system message: '{conv_template.system_message}'")
        messages = json.dumps(conv_template.messages)
        print(f"Conversation template messages: '{messages}'")
        #print_stats(attack_params)

        #print(f"Creating suffix manager")
        suffix_manager = SuffixManager(tokenizer=tokenizer, 
                      conv_template=conv_template, 
                      instruction=attack_params.base_prompt, 
                      target=attack_params.target_output, 
                      adv_string=attack_params.initial_adversarial_string)
        #print_stats(attack_params)
         
        #print(f"Debug: Model dtype: {model.dtype}")
        
        #import pdb; pdb.Pdb(nosigint=True).set_trace()

        print(f"Starting main loop")

        for main_loop_iteration_number in range(attack_params.max_iterations):
            if user_aborted:
                break
            else:
                try:
                    print(f"---------")
                    current_dt = get_now()
                    current_ts = get_time_string(current_dt)
                    current_elapsed_string = get_elapsed_time_string(start_dt, current_dt)
                    print(f"{current_ts} - Main loop iteration {main_loop_iteration_number + 1} of {attack_params.max_iterations} - elapsed time {current_elapsed_string} - successful attack count: {len(successful_attacks)}")                    
                    print_stats(attack_params)
                    print(f"---------")
                    
                    adversarial_values.append(adversarial_string)
                    
                    # Step 1. Encode user prompt (behavior + adv suffix) as tokens and return token ids.
                    #print(f"[main - encoding user prompt + adversarial data] Debug: calling get_input_ids with adversarial_string = '{adversarial_string}'")
                    #input_ids = suffix_manager.get_input_ids(adv_string = adversarial_string, force_python_tokenizer = attack_params.force_python_tokenizer)
                    input_id_data = suffix_manager.get_input_ids(adv_string = adversarial_string, force_python_tokenizer = attack_params.force_python_tokenizer)
                    #print_stats(attack_params)
                    
                    #decoded_input_ids = get_decoded_tokens(tokenizer, input_id_data.input_ids)
                    #decoded_full_prompt_ids = get_decoded_tokens(tokenizer, input_id_data.full_prompt_ids)
                    #decoded_control_slice = get_decoded_tokens(tokenizer, input_id_data.full_prompt_ids[input_id_data.slice_data.control])
                    #decoded_target_slice = get_decoded_tokens(tokenizer, input_id_data.full_prompt_ids[input_id_data.slice_data.target])
                    #decoded_loss_slice = get_decoded_tokens(tokenizer, input_id_data.full_prompt_ids[input_id_data.slice_data.loss])
                    #print(f"[input ID generation for token_gradients] Debug: decoded_input_ids = '{decoded_input_ids}', decoded_full_prompt_ids = '{decoded_full_prompt_ids}', decoded_control_slice = '{decoded_control_slice}', decoded_target_slice = '{decoded_target_slice}', decoded_loss_slice = '{decoded_loss_slice}'")
                    
                    #print(f"Converting input IDs to device")
                    input_ids = input_id_data.get_input_ids_as_tensor().to(attack_params.device)
                    #print(f"Debug: input_ids after conversion = '{input_ids}'")
                    #print_stats(attack_params)

                    # Step 2. Compute Coordinate Gradient
                    #print(f"Computing coordinate gradient")
                    # coordinate_grad = token_gradients(model, 
                                    # input_ids, 
                                    # suffix_manager._control_slice, 
                                    # suffix_manager._target_slice, 
                                    # suffix_manager._loss_slice)
                    coordinate_grad = token_gradients(model, 
                                    input_ids, 
                                    input_id_data.slice_data.control, 
                                    input_id_data.slice_data.target, 
                                    input_id_data.slice_data.loss)
                    #print_stats(attack_params)
                    
                    is_success = False
                    current_check_string = None
                    current_check_token_ids = None
                    control_slice_decoded = None
                    best_new_adversarial_string = None
                    
                    # Step 3. Sample a batch of new tokens based on the coordinate gradient.
                    # Notice that we only need the one that minimizes the loss.
                    with torch.no_grad():
                        
                        # Step 3.1 Slice the input to locate the adversarial suffix.
                        #print(f"Slicing input")
                        control_slice = input_ids[input_id_data.slice_data.control]
                        
                        control_slice_decoded = get_decoded_tokens(tokenizer, control_slice)
                        #print(f"[main - Slicing input] Debug: control_slice_decoded = '{control_slice_decoded}'")

                        adversarial_string_tokens = control_slice.to(attack_params.device)
                        #print_stats(attack_params)
                        #print(f"adversarial_string_tokens: {adversarial_string_tokens}")
                        
                        # Step 3.2 Randomly sample a batch of replacements.
                        #print(f"Randomly sampling a batch of replacements")
                        new_adversarial_string_toks = sample_control(adversarial_string_tokens, 
                                       coordinate_grad, 
                                       attack_params.batch_size_new_adversarial_tokens, 
                                       topk=attack_params.topk, 
                                       temp=attack_params.model_temperature, 
                                       not_allowed_tokens=not_allowed_tokens)
                        #print_stats(attack_params)
                        #print(f"new_adversarial_string_toks: {new_adversarial_string_toks}")
                        #decoded_ast = get_decoded_tokens(tokenizer, adversarial_string_tokens)
                        #nast_data = new_adversarial_string_toks
                        #if isinstance(new_adversarial_string_toks, torch.Tensor):
                            #print(f"Debug: new_adversarial_string_toks.tolist() = '{new_adversarial_string_toks.tolist()}', dir(new_adversarial_string_toks) = '{dir(new_adversarial_string_toks)}'")
                        #    nast_data = new_adversarial_string_toks.tolist()
                        #decoded_nast = get_decoded_tokens(tokenizer, nast_data)
                        #print(f"Debug: adversarial_string_tokens = '{adversarial_string_tokens}', new_adversarial_string_toks = '{new_adversarial_string_toks}', decoded_ast = '{decoded_ast}', decoded_nast = '{decoded_nast}'")
                        
                        # Note: I'm leaving this explanation here for historical reference
                        # Step 3.3 This step ensures all adversarial candidates have the same number of tokens. 
                        # This step is necessary because tokenizers are not invertible
                        # so Encode(Decode(tokens)) may produce a different tokenization.
                        # We ensure the number of token remains to prevent the memory keeps growing and run into OOM.
                        #print(f"Getting filtered candidates")
                        new_adversarial_string_list = get_filtered_cands(tokenizer, 
                                                            new_adversarial_string_toks, 
                                                            adversarial_values,
                                                            filter_cand=True, 
                                                            curr_control=adversarial_string,
                                                            filter_regex = attack_params.get_candidate_filter_regex(),
                                                            filter_repetitive_tokens = attack_params.candidate_filter_repetitive_tokens,
                                                            filter_repetitive_lines = attack_params.candidate_filter_repetitive_lines,
                                                            filter_newline_limit = attack_params.candidate_filter_newline_limit,
                                                            replace_newline_characters = attack_params.candidate_replace_newline_characters,
                                                            attempt_to_keep_token_count_consistent = attack_params.attempt_to_keep_token_count_consistent, 
                                                            candidate_filter_tokens_min = attack_params.candidate_filter_tokens_min, 
                                                            candidate_filter_tokens_max = attack_params.candidate_filter_tokens_max)
                        if len(new_adversarial_string_list) == 0:
                            print(f"Error: the attack appears to have failed to generate any adversarial string data at this iteration. This may be due to excessive post-generation filtering options. The tool will likely crash immediately after this condition occurs.")
                        #print_stats(attack_params)
                        #print(f"new_adversarial_string_list: '{new_adversarial_string_list}'")
                        
                        # Step 3.4 Compute loss on these candidates and take the argmin.
                        #print(f"Getting logits")
                        logits, ids = get_logits(model=model, 
                                                 tokenizer=tokenizer,
                                                 input_ids=input_ids,
                                                 #control_slice=suffix_manager._control_slice, 
                                                 control_slice=input_id_data.slice_data.control, 
                                                 test_controls=new_adversarial_string_list, 
                                                 return_ids=True,
                                                 batch_size=attack_params.batch_size_get_logits) # decrease this number if you run into OOM.
                        #print_stats(attack_params)

                        #print(f"Calculating target loss")
                        #losses = target_loss(logits, ids, suffix_manager._target_slice)
                        losses = target_loss(logits, ids, input_id_data.slice_data.target)
                        #print_stats(attack_params)

                        #print(f"Getting losses argmin")
                        best_new_adversarial_string_id = losses.argmin()
                        #print_stats(attack_params)

                        #print(f"Setting best new adversarial string")
                        best_new_adversarial_string = new_adversarial_string_list[best_new_adversarial_string_id]
                        #print_stats(attack_params)

                        #print(f"Getting current loss")
                        current_loss = losses[best_new_adversarial_string_id]
                        #print_stats(attack_params)

                        # Update the running adversarial_string with the best candidate
                        #print(f"Updating adversarial string - was '{adversarial_string}', now '{best_new_adversarial_string}'")
                        #adversarial_string = best_new_adversarial_string
                        #print_stats(attack_params)
                        #print(f"Checking for success")
                        #is_success_input_ids = suffix_manager.get_input_ids(adv_string = adversarial_string).to(attack_params.device)
                        #print(f"[main - checking for success] Debug: calling get_input_ids with adversarial_string = '{adversarial_string}'")
                        is_success_input_id_data = suffix_manager.get_input_ids(adv_string = adversarial_string, force_python_tokenizer = attack_params.force_python_tokenizer)
                        is_success_input_ids = is_success_input_id_data.get_input_ids_as_tensor().to(attack_params.device)
                        is_success, current_check_string, current_check_token_ids = check_for_attack_success(attack_params, model, 
                                                 tokenizer,
                                                 is_success_input_ids, 
                                                 is_success_input_id_data.slice_data.assistant_role, 
                                                 attack_params.negative_output_strings)            
                        #print_stats(attack_params)

                    # Create a dynamic plot for the loss.
                    #plotlosses.update({'Loss': current_loss.detach().cpu().numpy()})
                    #plotlosses.send() 
                    
                    current_loss_as_float = float(f"{current_loss.detach().cpu().numpy()}")
                    print(f"Loss: {current_loss_as_float}")
                    print(f"Jailbroken: {is_success} for generated string '{current_check_string}'\nCurrent best new adversarial string: '{best_new_adversarial_string}'")
                    #print(f"Passed:{is_success}\nCurrent best new adversarial string: '{best_new_adversarial_string}'")
                    #json_data_current_iteration = {}
                    attack_data_current_iteration = AttackResultInfo()

                    attack_data_current_iteration.date_time_utc = get_time_string()
                    attack_data_current_iteration.jailbreak_detected = is_success
                    attack_data_current_iteration.loss = current_loss_as_float
                    attack_data_current_iteration.adversarial_tokens = control_slice_decoded
                    #attack_data_current_iteration.adversarial_value = tokenizer.decode(control_slice)
                    attack_data_current_iteration.adversarial_value = adversarial_string
                    attack_data_current_iteration.best_new_adversarial_value = best_new_adversarial_string
                    attack_data_current_iteration.jailbreak_check_output_string = current_check_string
                    is_success_input_ids_list = is_success_input_ids.tolist()
                    attack_data_current_iteration.jailbreak_check_input_token_ids = is_success_input_ids_list
                    attack_data_current_iteration.jailbreak_check_input_tokens = get_decoded_tokens(tokenizer, is_success_input_ids_list)
                    attack_data_current_iteration.jailbreak_check_input_string = tokenizer.decode(is_success_input_ids_list)
                    current_check_token_ids_list = current_check_token_ids.tolist()
                    attack_data_current_iteration.jailbreak_check_generation_token_ids = current_check_token_ids_list
                    attack_data_current_iteration.jailbreak_check_generation_tokens = get_decoded_tokens(tokenizer, current_check_token_ids_list)
                    attack_data_current_iteration.jailbreak_check_generation_string = tokenizer.decode(current_check_token_ids_list)
                            
                    if is_success or attack_params.display_full_failed_output:
                        #current_string = get_input_and_output_strings(attack_params, model, tokenizer, suffix_manager, adversarial_string)
                        full_input_token_ids, generation_input_token_ids, output_token_ids, full_generation_token_ids = get_current_input_and_output_tokens(attack_params, model, tokenizer, suffix_manager, adversarial_string)
                        decoded_input = tokenizer.decode(generation_input_token_ids).strip()
                        decoded_output = tokenizer.decode(output_token_ids).strip()
                        current_string = f"Current input: '{decoded_input}'\n\nCurrent output: '{decoded_output}'"
                        print(current_string)
                        
                        attack_data_current_iteration.full_test_token_ids = full_input_token_ids
                        #attack_data_current_iteration.full_generation_token_ids = full_generation_token_ids
                        full_generation_token_ids_list = full_generation_token_ids.tolist()
                        attack_data_current_iteration.full_generation_token_ids = full_generation_token_ids_list
                        #attack_data_current_iteration.input_token_ids = generation_input_token_ids]
                        generation_input_token_ids_list = generation_input_token_ids.tolist()
                        attack_data_current_iteration.input_token_ids = generation_input_token_ids_list
                        #attack_data_current_iteration.output_token_ids = output_token_ids
                        output_token_ids_list = output_token_ids.tolist()
                        attack_data_current_iteration.output_token_ids = output_token_ids_list
                        attack_data_current_iteration.decoded_full_test_tokens = get_decoded_tokens(tokenizer, full_input_token_ids)
                        attack_data_current_iteration.decoded_full_test_string = tokenizer.decode(full_input_token_ids)
                        attack_data_current_iteration.decoded_full_generation_tokens = get_decoded_tokens(tokenizer, full_generation_token_ids)
                        attack_data_current_iteration.decoded_full_generation_string = tokenizer.decode(full_generation_token_ids)
                        attack_data_current_iteration.decoded_input_tokens = get_decoded_tokens(tokenizer, generation_input_token_ids)
                        attack_data_current_iteration.decoded_output_token_ids = get_decoded_tokens(tokenizer, output_token_ids)
                        attack_data_current_iteration.input_string = decoded_input
                        attack_data_current_iteration.output_string = decoded_output
                            
                        if is_success:
                            successful_attacks.append(current_string)
                            if attack_params.break_on_success:
                                break
                    
                    attack_data.append(attack_data_current_iteration)
                    if attack_params.json_output_file is not None:
                        json_data.append(attack_data_current_iteration.to_dict())
                        safely_write_text_output_file(attack_params.json_output_file, json.dumps(json_data))
                    
                    if attack_params.rollback_on_loss_increase and main_loop_iteration_number > 0:
                        previous_data = attack_data[main_loop_iteration_number - 1]                        
                        if attack_data_current_iteration.loss > previous_data.loss:
                            print(f"The loss value for adversarial data '{attack_data_current_iteration.adversarial_tokens}' ({attack_data_current_iteration.loss}) is greater than for the previous adversarial data '{previous_data.adversarial_tokens}' ({previous_data.loss}). Rolling back to the previous adversarial data ('{previous_data.best_new_adversarial_value}') for the next iteration instead of using '{best_new_adversarial_string}'.")
                            best_new_adversarial_string = previous_data.best_new_adversarial_value
                    
                    # Update the running adversarial_string with the best candidate
                    # Moved down here to make full output generation consistent and allow rollback
                    #print(f"Updating adversarial string - was '{adversarial_string}', now '{best_new_adversarial_string}'")
                    adversarial_string = best_new_adversarial_string
                        
                    # (Optional) Clean up the cache.
                    #print(f"Cleaning up the cache")
                    del coordinate_grad, adversarial_string_tokens ; gc.collect()
                    #if "cuda" in attack_params.device:
                    #    torch.cuda.empty_cache()
                    
                except KeyboardInterrupt:
                    #import pdb; pdb.Pdb(nosigint=True).post_mortem()
                    print(f"Exiting main loop early by request")
                    user_aborted = True

    except KeyboardInterrupt:
        #import pdb; pdb.Pdb(nosigint=True).post_mortem()
        print(f"Exiting early by request")
        user_aborted = True

    if not user_aborted:
        print(f"Main loop complete")
    print_stats(attack_params)

    len_successful_attacks = len(successful_attacks)
    if len_successful_attacks > 1:
        success_list_string = f"Successful attacks ({len_successful_attacks}):"
        for i in range(0, len_successful_attacks):
            success_list_string += f"\n{successful_attacks[i]}\n"
        print(success_list_string)

    if model is not None and tokenizer is not None and suffix_manager is not None:        
        current_string = get_input_and_output_strings(attack_params, model, tokenizer, suffix_manager, adversarial_string, input_label = "Final input", output_label = "Final output")
        print(current_string)

    end_dt = get_now()
    end_ts = get_time_string(end_dt)
    total_elapsed_string = get_elapsed_time_string(start_dt, end_dt)
    print(f"Finished at {end_ts} - elapsed time {total_elapsed_string}")


if __name__=='__main__':
    short_description = get_short_script_description()
    print(f"{script_name} version {script_version}, {script_date}\n{short_description}")
    
    attack_params = gcg_attack_params()
    
    cuda_available = torch.cuda.is_available()
    mps_available = torch.backends.mps.is_available()
    
    if not cuda_available:
        print(f"Warning: this host does not appear to have a PyTorch CUDA back-end available. The default --device option has therefore been changed from '{attack_params.device}' to '{attack_params.device_fallback}'. Using CPU processing will result in significantly longer run times for this tool. Expect each iteration to take several hours instead of tens of seconds on a modern GPU with support for CUDA. If your host has CUDA hardware, you should investigate why PyTorch is not detecting it.")        
        attack_params.device = attack_params.device_fallback
    if mps_available:
        print(f"Warning: this host appears to be an Apple device with support for the Metal ('mps') PyTorch back-end. At the time this version of {script_name} was developed, the Metal back-end did not support some features that were critical to the attack code, such as nested tensors. If you believe that you are using a newer version of PyTorch that has those features implemented, you can try enabling the Metal back-end by specifying the --device mps command-line option. However, it is unlikely to succeed. This message will be removed when Bishop Fox has verified that the Metal back-end supports the necessary features.")  
    
    parser = argparse.ArgumentParser(description=get_script_description(),formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument("-m", "--model", required=True, type=str, 
        help="Path to the base directory for the large language model you want to attack, e.g. /home/blincoln/LLMs/StabilityAI/stablelm-2-1_6b-chat")
        
    parser.add_argument("--tokenizer", type=str, 
        help="(optional) Path to the base directory for the LLM tokenizer you want to use with the model instead of any tokenizer that may be included with the model itself. Intended for use with models such as Mamba that do not include their own tokenizer.")
        
    template_name_list = ", ".join(attack_params.get_known_template_names())
    
    parser.add_argument("-t", "--template", type=str, 
        help=f"An optional model type name, for selecting the correct chat template. Use --list-templates to view available options. If this option is not specified, the fastchat library will attempt to load the correct template based on the base model directory contents.")

    parser.add_argument("--list-templates", type=str2bool, nargs='?',
        const=True,
        help="Output a list of all template names for the version of the fastchat library you have installed (to use with --template), then exit.")
        
    parser.add_argument("--system-prompt", type=str, 
        help=f"Specify a custom value to use instead of the default system prompt for the conversation template.")

    parser.add_argument("--system-prompt-from-file", type=str, 
        help=f"Specify a custom value to use instead of the default system prompt for the conversation template by reading it from a file. The entire file will be used as a single string system prompt.")

    parser.add_argument("--clear-existing-conversation", type=str2bool, nargs='?',
        const=True, default=attack_params.clear_existing_template_conversation,
        help="Removes any existing non-system messages from the conversation template.")
        
    parser.add_argument("--template-messages-from-file", type=str, 
        help=f"Load custom messages into the conversation template by reading them from a JSON file. If --clear-existing-conversation is not specified, the messages will be appended to any existing messages in the list for the template. The format of the file should be '[[<role ID or name>, <message>],[<role ID or name>, <message>][...][<role ID or name>, <message>]]', e.g. {attack_params.template_messages_from_file_example}")

    parser.add_argument("--base-prompt", type=str, 
        help=f"The prompt the tool should try to jailbreak, e.g. '{attack_params.base_prompt_example}'")
        
    parser.add_argument("--target-output", type=str, 
        help=f"The target output the tool should measure potential results against, e.g. '{attack_params.target_output_example}'")
    
    parser.add_argument("--auto-target", type=str, 
        help=f"Instead of manually specifying separate --base-prompt and --target-output values, specify a single goal (without a leading verb such as 'write'), and the tool will generate the base prompt and target output values, e.g. --auto-target '{attack_params.autotarget_example}'")
    
    parser.add_argument("-d", "--device", default=attack_params.device, type=str, 
        help="The device to use for the PyTorch operations ('cuda', 'cuda:0', etc.). Using anything other than CUDA is unlikely to produce satisfactory results.")

    parser.add_argument("--initial-adversarial-string", default=attack_params.initial_adversarial_string, type=str, 
        help="The initial string to iterate on. Leave this as the default to perform the standard attack. Specify the output of a previous run to continue iterating at that point. Specify a custom value to experiment. Specify an arbitrary number of space-delimited exclamation points to perform the standard attack, but using a different number of initial tokens.")
    
    parser.add_argument("--topk", type=numeric_string_to_int,
        default=attack_params.topk,
        help=f"'topk' value to pass to the sample_control function.")

    parser.add_argument("--temperature", type=numeric_string_to_int,
        default=attack_params.model_temperature,
        help=f"'Temperature' value to pass to the model. Use the default value for deterministic results.")

    parser.add_argument("--random-seed-numpy", type=numeric_string_to_int,
        default=attack_params.np_random_seed,
        help=f"Random seed for NumPy")
    parser.add_argument("--random-seed-torch", type=numeric_string_to_int,
        default=attack_params.torch_manual_seed,
        help=f"Random seed for PyTorch")
    parser.add_argument("--random-seed-cuda", type=numeric_string_to_int,
        default=attack_params.torch_cuda_manual_seed_all,
        help=f"Random seed for CUDA")

    parser.add_argument("--max-iterations", type=numeric_string_to_int,
        default=attack_params.max_iterations,
        help=f"Maximum number of times to iterate on the adversarial data")

    parser.add_argument("--batch-size-new-adversarial-tokens", type=numeric_string_to_int,
        default=attack_params.batch_size_new_adversarial_tokens,
        help=f"The PyTorch batch size to use when generating new adversarial tokens. If you are running out of memory and this value is greater than 1, try reducing it. If it still happens with all of the batch size values set to 1, you're probably out of luck without more VRAM. Alternatively, if you *aren't* running out of memory, you can try increasing this value for better performance.")

    parser.add_argument("--batch-size-get-logits", type=numeric_string_to_int,
        default=attack_params.batch_size_get_logits,
        help=f"The PyTorch batch size to use when calling the get_logits function, which is the most memory-intensive operation other than loading the model itself. If you are running out of memory and this value is greater than 1, try reducing it. If it still happens with all of the batch size values set to 1, you're probably out of luck without more VRAM. Alternatively, if you *aren't* running out of memory, you can try increasing this value for better performance.")
        
    parser.add_argument("--max-new-tokens", type=numeric_string_to_int,
        default=attack_params.generation_max_new_tokens,
        help=f"The maximum number of tokens to generate when testing output for a successful jailbreak.")

    parser.add_argument("--max-new-tokens-final", type=numeric_string_to_int,
        default=attack_params.full_decoding_max_new_tokens,
        help=f"The maximum number of tokens to generate when generating final output for display.")

    parser.add_argument("--exclude-nonascii-tokens", type=str2bool, nargs='?',
        const=True, default=attack_params.exclude_nonascii_tokens,
        help="Bias the adversarial content generation data to avoid using tokens that are not printable ASCII text.")

    parser.add_argument("--exclude-special-tokens", type=str2bool, nargs='?',
        const=True, default=attack_params.exclude_special_tokens,
        help="Bias the adversarial content generation data to avoid using basic special tokens (begin/end of string, padding, unknown).")

    parser.add_argument("--exclude-additional-special-tokens", type=str2bool, nargs='?',
        const=True, default=attack_params.exclude_additional_special_tokens,
        help="Bias the adversarial content generation data to avoid using additional special tokens defined in the tokenizer configuration.")

    parser.add_argument("--exclude-whitespace-tokens", type=str2bool, nargs='?',
        const=True, default=False,
        help="Bias the adversarial content generation data to avoid using tokens that consist solely of whitespace characters.")

    parser.add_argument("--exclude-token", action='append', nargs='*', required=False,
        help=f"Bias the adversarial content generation data to avoid using the specified token (if it exists as a discrete value in the model). May be specified multiple times to exclude multiple tokens.")

    parser.add_argument("--exclude-newline-tokens", type=str2bool, nargs='?',
        const=True, default=False,
        help="A shortcut equivalent to specifying just about any newline token variations using --exclude-token.")

    parser.add_argument("--exclude-three-hashtag-tokens", type=str2bool, nargs='?',
        const=True, default=False,
        help="A shortcut equivalent to specifying most variations on the token '###' using --exclude-token.")

    parser.add_argument("--token-filter-regex", type=str,
        help="If specified, biases the adversarial content generation to exclude tokens that don't match the specified regular expression.")

    parser.add_argument("--adversarial-candidate-filter-regex", type=str, 
        default=attack_params.candidate_filter_regex,
        help="The regular expression used to filter candidate adversarial strings. The default value is very forgiving and simply requires that the string contain at least one occurrence of two consecutive mixed-case alphanumeric characters.")
    
    parser.add_argument("--adversarial-candidate-repetitive-line-limit", type=numeric_string_to_int,
        help=f"If this value is specified, candidate adversarial strings will be filtered out if any one line is repeated more than this many times.")
        
    parser.add_argument("--adversarial-candidate-repetitive-token-limit", type=numeric_string_to_int,
        help=f"If this value is specified, candidate adversarial strings will be filtered out if any one token is repeated more than this many times.")
        
    parser.add_argument("--adversarial-candidate-newline-limit", type=numeric_string_to_int,
        help=f"If this value is specified, candidate adversarial strings will be filtered out if they contain more than this number of newline characters.")
        
    parser.add_argument("--adversarial-candidate-newline-replacement", type=str, 
        help="If this value is specified, it will be used to replace any newline characters in candidate adversarial strings. This can be useful if you want to avoid generating attacks that depend specifically on newline-based content, such as injecting different role names.")

    parser.add_argument("--adversarial-candidate-filter-tokens-min", type=numeric_string_to_int,
        help=f"If this value is specified, candidate adversarial strings will be filtered out if they contain fewer than this number of tokens.")
        
    parser.add_argument("--adversarial-candidate-filter-tokens-max", type=numeric_string_to_int,
        help=f"If this value is specified, candidate adversarial strings will be filtered out if they contain more than this number of tokens.")

    parser.add_argument("--attempt-to-keep-token-count-consistent", type=str2bool, nargs='?',
        const=True, default=attack_params.attempt_to_keep_token_count_consistent,
        help="Enable the check from the original authors' code that attempts to keep the number of tokens consistent between each adversarial string. This will cause all candidates to be excluded for some models, such as StableLM 2. If you want to limit the number of tokens (e.g. to prevent the attack from wasting time on single-token strings or to avoid out-of-memory conditions) --adversarial-candidate-filter-tokens-min and --adversarial-candidate-filter-tokens-max are generally much better options.")

    parser.add_argument("--generic-role-template", type=str, 
        help="The Python formatting string to use if fastchat defaults to a generic chat template. e.g --generic-role-template '[{role}]', '<|{role}|>'.")
    
    parser.add_argument("--trust-remote-code", type=str2bool, nargs='?',
        const=True, default=attack_params.load_options_trust_remote_code,
        help="When loading the model, pass 'trust_remote_code=True', which enables execution of arbitrary Python scripts included with the model. You should probably examine those scripts first before deciding if you're comfortable with this option. Currently required for some models, such as Phi-3.")
    parser.add_argument("--ignore-mismatched-sizes", type=str2bool, nargs='?',
        const=True, default=attack_params.load_options_ignore_mismatched_sizes,
        help="When loading the model, pass 'ignore_mismatched_sizes=True', which may allow you to load some models with mismatched size data. It will probably only let the tool get a little further before erroring out, though.")

    parser.add_argument("--break-on-success", type=str2bool, nargs='?',
        const=True, default=attack_params.break_on_success,
        help="Stop iterating upon the first detection of a potential successful jailbreak.")
    parser.add_argument("--rollback-on-loss-increase", type=str2bool, nargs='?',
        const=True, default=attack_params.rollback_on_loss_increase,
        help="If the loss value does not decrease between iterations, roll back to the last 'good' adversarial data.")
    parser.add_argument("--display-failure-output", type=str2bool, nargs='?',
        const=True, default=attack_params.display_full_failed_output,
        help="Output the full decoded input and output for failed jailbreak attempts (in addition to successful attempts, which are always output).")
    parser.add_argument("--low-cpu-mem-usage", type=str2bool, nargs='?',
        const=True, default=attack_params.low_cpu_mem_usage,
        help="When loading the model and tokenizer, pass 'low_cpu_mem_usage=True'. May or may not affect performance and results.")
    parser.add_argument("--use-cache", type=str2bool, nargs='?',
        const=True, default=attack_params.use_cache,
        help="When loading the model and tokenizer, pass 'use_cache=True'. May or may not affect performance and results.")
    parser.add_argument("--display-model-size", type=str2bool, nargs='?',
        const=True, default=attack_params.display_model_size,
        help="Displays size information for the selected model. Warning: will write the raw model data to a temporary file, which may double the load time.")
    parser.add_argument("--force-python-tokenizer", type=str2bool, nargs='?',
        const=True, default=attack_params.force_python_tokenizer,
        help="Use the Python tokenizer even if the model supports a (usually faster) non-Python tokenizer. May allow use of some models that include incomplete non-Python tokenizers.")
    parser.add_argument("--enable-hardcoded-tokenizer-workarounds", type=str2bool, nargs='?',
        help="Enable the undocumented, hardcoded tokenizer workarounds that the original developers introduced for some models.")
    padding_token_values = get_missing_pad_token_names()
    parser.add_argument("--missing-pad-token-replacement", type=str,
        help=f"If the tokenizer is missing a padding token definition, use an alternative special token instead. Must be one of: {padding_token_values}.")
    parser.add_argument("--json-output-file", type=str,
        help=f"If the tokenizer is missing a padding token definition, use an alternative special token instead. Must be one of: {padding_token_values}.")
    parser.add_argument("--overwrite-output", type=str2bool, nargs='?',
        const=True,
        help="Overwrite any existing output files (--json-output-file, etc.).")

    # TKTK: option to export attack_params as JSON or joad it as JSON
    # TKTK: ability to customize the positive and negative test strings
    # Related TKTK: ability to specify positive/negative test rules after 
    #       implementing the rule engine instead of positive/negative lists

    args = parser.parse_args()

    if args.list_templates:
        fc_template_list = []
        for fct_name in fcc.conv_templates.keys():
            fc_template_list.append(fct_name)
        fc_template_list.sort()
        list_string = "Templates tested more or less successfully by Bishop Fox:\n"
        for ttl in attack_params.get_known_template_names():
            list_string += f"{ttl}\n"
        list_string += "\nAll templates included with the version of the fastchat library in your environment:\n"
        for fctl in fc_template_list:
            list_string += f"{fctl}\n"
        list_string += "\nNote: many of the entries in the second list will produce unexpected or incorrect results.\n"
        print(list_string)
        sys.exit(0)

    attack_params.device = args.device

    if cuda_available:
        if len(attack_params.device) < 4 or attack_params.device[0:4] != "cuda":
            print(f"Warning: this appears to have a PyTorch CUDA back-end available, but the back-end has been set to '{attack_params.device}' instead. This is likely to result in significantly decreased performance versus using the CUDA back-end.")        

    check_pytorch_devices(attack_params)

    attack_params.model_path = os.path.abspath(args.model)
    if not os.path.isdir(attack_params.model_path):
        print(f"The specified model directory ('{attack_params.model_path}') does not appear to exist.")
        sys.exit(1)
        
    if args.tokenizer:
        attack_params.tokenizer_path = os.path.abspath(args.tokenizer)
        if not os.path.isdir(attack_params.tokenizer_path):
            print(f"The specified tokenizer directory ('{attack_params.tokenizer_path}') does not appear to exist.")
            sys.exit(1)
        
    if args.template:
        attack_params.template_name = args.template

    if args.clear_existing_conversation:
        attack_params.clear_existing_template_conversation = True
    
    if args.system_prompt:
        attack_params.custom_system_prompt = args.system_prompt
        
    if args.system_prompt_from_file:
        if args.system_prompt:
            print(f"Only one of --system-prompt-from-file and --system-prompt may be specified.")
            sys.exit(1)
        system_prompt_file = os.path.abspath(args.system_prompt_from_file)
        attack_params.custom_system_prompt = get_file_content(system_prompt_file, failure_is_critical = True)

    if args.template_messages_from_file:
        message_file = os.path.abspath(args.template_messages_from_file)
        message_file_content = get_file_content(message_file, failure_is_critical = True)
        try:
            attack_params.set_conversation_template_messages(json.loads(message_file_content))
        except Exception as e:
            print(f"Error loading conversation template messages from file '{message_file}', content '{message_file_content}': {e}.")
            sys.exit(1)

    attack_params.initial_adversarial_string = args.initial_adversarial_string

    if args.auto_target:
        if args.base_prompt or args.target_output:
            print(f"Error: cannot specify --auto-target when either --base-prompt or --target-output are also specified")
            sys.exit(1)
        attack_params.set_automatic_base_and_target(args.auto_target)
    
    if args.base_prompt:
        attack_params.base_prompt = args.base_prompt
        
    if args.target_output:
        attack_params.target_output = args.target_output

    if args.overwrite_output:
        attack_params.overwrite_output = True

    attack_params.topk = args.topk

    attack_params.model_temperature = args.temperature

    attack_params.np_random_seed = args.random_seed_numpy

    attack_params.torch_manual_seed = args.random_seed_torch

    attack_params.torch_cuda_manual_seed_all = args.random_seed_cuda

    attack_params.max_iterations = args.max_iterations

    attack_params.batch_size_new_adversarial_tokens = args.batch_size_new_adversarial_tokens

    attack_params.batch_size_get_logits = args.batch_size_get_logits
    
    attack_params.generation_max_new_tokens = args.max_new_tokens
    
    attack_params.full_decoding_max_new_tokens = args.max_new_tokens_final

    attack_params.exclude_nonascii_tokens = args.exclude_nonascii_tokens
    
    attack_params.exclude_special_tokens = args.exclude_special_tokens
    
    attack_params.exclude_additional_special_tokens = args.exclude_additional_special_tokens
    
    if args.token_filter_regex:
        attack_params.token_filter_regex = args.token_filter_regex
    
    attack_params.candidate_filter_regex = args.adversarial_candidate_filter_regex
    
    if args.adversarial_candidate_filter_tokens_min:
        if args.adversarial_candidate_filter_tokens_min < 1:
            print("--adversarial-candidate-filter-tokens-min must be a positive integer.")
            sys.exit(1)
        attack_params.candidate_filter_tokens_min = args.adversarial_candidate_filter_tokens_min
    
    if args.adversarial_candidate_filter_tokens_max:
        if args.adversarial_candidate_filter_tokens_max < 1:
            print("--adversarial-candidate-filter-tokens-max must be a positive integer.")
            sys.exit(1)
        attack_params.candidate_filter_tokens_max= args.adversarial_candidate_filter_tokens_max
    
    attack_params.attempt_to_keep_token_count_consistent = args.attempt_to_keep_token_count_consistent
    
    if args.adversarial_candidate_repetitive_line_limit:
        if args.adversarial_candidate_repetitive_line_limit < 1:
            print("--adversarial-candidate-repetitive-line-limit must be a positive integer.")
            sys.exit(1)
        attack_params.candidate_filter_repetitive_lines = args.adversarial_candidate_repetitive_line_limit
        
    if args.adversarial_candidate_repetitive_token_limit:
        if args.adversarial_candidate_repetitive_token_limit < 1:
            print("--adversarial-candidate-repetitive-token-limit must be a positive integer.")
            sys.exit(1)
        attack_params.candidate_filter_repetitive_tokens = args.adversarial_candidate_repetitive_token_limit
    
    if args.adversarial_candidate_newline_limit:
        if args.adversarial_candidate_newline_limit < 0:
            print("--adversarial-candidate-newline-limit must be an integer greater than or equal to 0.")
            sys.exit(1)
        attack_params.candidate_filter_newline_limit = args.adversarial_candidate_newline_limit
    
    if args.adversarial_candidate_newline_replacement:
        attack_params.candidate_replace_newline_characters = args.adversarial_candidate_newline_replacement
    
    if args.generic_role_template:
        attack_params.generic_role_indicator_template = args.generic_role_template

    attack_params.load_options_trust_remote_code = args.trust_remote_code
    
    attack_params.load_options_ignore_mismatched_sizes = args.ignore_mismatched_sizes
    
    attack_params.break_on_success = args.break_on_success
    
    attack_params.rollback_on_loss_increase = args.rollback_on_loss_increase
    
    attack_params.display_full_failed_output = args.display_failure_output
    
    attack_params.low_cpu_mem_usage = args.low_cpu_mem_usage
    
    attack_params.use_cache = args.use_cache
    
    attack_params.display_model_size = args.display_model_size
    
    attack_params.force_python_tokenizer = args.force_python_tokenizer

    if args.enable_hardcoded_tokenizer_workarounds:
        attack_params.enable_hardcoded_tokenizer_workarounds = True
        
    if args.missing_pad_token_replacement:
        if args.missing_pad_token_replacement not in padding_token_values:
            print(f"The value for --missing-pad-token-replacement must be one of: {padding_token_values}")
            sys.exit(1)
        attack_params.missing_pad_token_replacement = args.missing_pad_token_replacement

    # shortcut option processing
    if args.exclude_whitespace_tokens:
        attack_params.exclude_whitespace_tokens = True
                
    if args.exclude_three_hashtag_tokens:
        # If you want to disallow "###", you also have to disallow "#" and "##" or the generation algorithm will reconstruct "###" from them
        attack_params.not_allowed_token_list.append("#")
        attack_params.not_allowed_token_list.append("##")
        attack_params.not_allowed_token_list.append("###")

    if args.exclude_token:
        for elem in args.exclude_token:
            for et in elem:
                if et.strip() != "":
                    if et not in attack_params.not_allowed_token_list:
                        attack_params.not_allowed_token_list.append(et)

    if args.json_output_file:
        attack_params.json_output_file = os.path.abspath(args.json_output_file)
        # test output ability now so that the user doesn't have to wait to find out that it will fail
        if os.path.isfile(attack_params.json_output_file):
            if attack_params.overwrite_output:
                print(f"Warning: overwriting JSON output file '{attack_params.json_output_file}'")
            else:
                print(f"Error: the JSON output file '{attack_params.json_output_file}' already exists. Specify --overwrite-output to replace it.")
                sys.exit(1)
        try:
            safely_write_text_output_file(attack_params.json_output_file, "")
        #except Exception as e:
        except Exception as e:
            print(f"Could not validate the ability to write to the file '{attack_params.json_output_file}': {e}")
            sys.exit(1)
    
    main(attack_params)
    
    
