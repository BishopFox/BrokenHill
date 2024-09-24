#!/bin/env python3

script_name = "brokenhill.py"
script_version = "0.30"
script_date = "2024-09-23"

def get_logo():
    result =  "                                                                                \n"
    #result = "xxxxxxxxxxxxxxxxxxx|xxxxxxxxxxxxxxxxxxx||xxxxxxxxxxxxxxxxxxx|xxxxxxxxxxxxxxxxxxx\n"
    result += " .                                                                            . \n"
    result += "   .oO___________                                              ___________Oo.   \n"
    result += "   .             \____________________________________________/             .   \n"
    result += "    |                                                                      |    \n"
    result += "    |                                                                      |    \n"
    result += "    |                             Broken Hill                              |    \n"
    result += "    |                                                                      |    \n"
    result += "    |          a tool for attacking LLMs, presented by Bishop Fox          |    \n"
    result += "    |                                                                      |    \n"    
    result += "    |                https://github.com/BishopFox/BrokenHill               |    \n"
    result += "    |                                                                      |    \n"
    result += "    |                   ________________________________                   |    \n"
    result += "   '  _________________/                                \_________________  '   \n"
    result += "   '^O                                                                    O^'   \n"
    result += " '                                                                            ' \n"
    result += "                                                                                \n"
    return result


# def get_logo():
    # result =  "                                                                                \n"
    # #result = "xxxxxxxxxxxxxxxxxxx|xxxxxxxxxxxxxxxxxxx||xxxxxxxxxxxxxxxxxxx|xxxxxxxxxxxxxxxxxxx\n"
    # result += " .                                                                            . \n"
    # result += "   .oO_________________                                  _________________Oo.   \n"
    # result += "   .                   \________________________________/                   .   \n"
    # result += "    |                                                                      |    \n"
    # result += "    |                                                                      |    \n"
    # result += "    |                             Broken Hill                              |    \n"
    # result += "    |                                                                      |    \n"
    # result += "    |          a tool for attacking LLMs, presented by Bishop Fox          |    \n"
    # result += "    |                                                                      |    \n"    
    # result += "    |                https://github.com/BishopFox/BrokenHill               |    \n"
    # result += "    |                                                                      |    \n"
    # result += "    |             ____________________________________________             |    \n"
    # result += "   '  ___________/                                            \___________  '   \n"
    # result += "   '^O                                                                    O^'   \n"
    # result += " '                                                                            ' \n"
    # result += "                                                                                \n"
    # return result




def get_script_description():
    result = 'Performs a "Greedy Coordinate Gradient" (GCG) attack against various large language models (LLMs), as described in the paper "Universal and Transferable Adversarial Attacks on Aligned Language Models" by Andy Zou, Zifan Wang, Nicholas Carlini, Milad Nasr, J. Zico Kolter, and Matt Fredrikson, representing Carnegie Mellon University, the Center for AI Safety, Google DeepMind, and the Bosch Center for AI.'
    result += "\n"
    result += "Originally based on the demo.ipynb notebook and associated llm-attacks library from https://github.com/llm-attacks/llm-attacks"
    result += "\n"
    result += "Also incorporates gradient-sampling code and mellowmax loss function from nanoGCG - https://github.com/GraySwanAI/nanoGCG"
    result += "\n"
    result += "This tool created and all other post-fork changes to the associated library by Ben Lincoln, Bishop Fox."
    result += "\n"
    result += f"version {script_version}, {script_date}"    
    return result

def get_short_script_description():
    result = 'Based on code and research by Andy Zou, Zifan Wang, Nicholas Carlini, Milad Nasr, J. Zico Kolter, and Matt Fredrikson.'
    result += "\n"
    result += "Also incorporates gradient-sampling code and mellowmax loss function from nanoGCG - https://github.com/GraySwanAI/nanoGCG"
    result += "\n"
    result += "This tool created and all other post-fork changes to the associated library by Ben Lincoln, Bishop Fox."
    return result

import argparse
import base64
import datetime
# IMPORTANT: 'fastchat' is in the PyPi package 'fschat', not 'fastchat'!
import fastchat.conversation
import gc
import locale
import json
import logging
import math
import numpy
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

from llm_attacks_bishopfox.attack.attack_classes import AdversarialContent
from llm_attacks_bishopfox.attack.attack_classes import AdversarialContentList
from llm_attacks_bishopfox.attack.attack_classes import AdversarialContentPlacement
from llm_attacks_bishopfox.attack.attack_classes import AttackParams
from llm_attacks_bishopfox.attack.attack_classes import AttackResultInfo
from llm_attacks_bishopfox.attack.attack_classes import AttackResultInfoCollection
from llm_attacks_bishopfox.attack.attack_classes import AttackResultInfoData
from llm_attacks_bishopfox.attack.attack_classes import AttackState
from llm_attacks_bishopfox.attack.attack_classes import BrokenHillMode
from llm_attacks_bishopfox.attack.attack_classes import BrokenHillResultData
from llm_attacks_bishopfox.attack.attack_classes import FakeException
from llm_attacks_bishopfox.attack.attack_classes import GenerationResults
from llm_attacks_bishopfox.attack.attack_classes import InitialAdversarialContentCreationMode
from llm_attacks_bishopfox.attack.attack_classes import LossAlgorithm
from llm_attacks_bishopfox.attack.attack_classes import LossSliceMode
from llm_attacks_bishopfox.attack.attack_classes import OverallScoringFunction
from llm_attacks_bishopfox.attack.attack_classes import PyTorchDevice
from llm_attacks_bishopfox.base.attack_manager import get_decoded_token
from llm_attacks_bishopfox.base.attack_manager import get_decoded_tokens
from llm_attacks_bishopfox.base.attack_manager import get_effective_max_token_value_for_model_and_tokenizer
from llm_attacks_bishopfox.base.attack_manager import get_embedding_layer
from llm_attacks_bishopfox.base.attack_manager import get_encoded_token
from llm_attacks_bishopfox.base.attack_manager import get_nonascii_token_list
from llm_attacks_bishopfox.base.attack_manager import get_random_seed_list_for_comparisons
from llm_attacks_bishopfox.base.attack_manager import get_token_allow_and_deny_lists
from llm_attacks_bishopfox.base.attack_manager import get_token_list_as_tensor
from llm_attacks_bishopfox.dumpster_fires.conversation_templates import ConversationTemplateTester
from llm_attacks_bishopfox.dumpster_fires.conversation_templates import fschat_conversation_template_to_json
from llm_attacks_bishopfox.dumpster_fires.conversation_templates import fschat_conversation_template_from_json
from llm_attacks_bishopfox.dumpster_fires.offensive_tokens import get_profanity
from llm_attacks_bishopfox.dumpster_fires.offensive_tokens import get_slurs
from llm_attacks_bishopfox.dumpster_fires.offensive_tokens import get_other_highly_problematic_content
from llm_attacks_bishopfox.dumpster_fires.trash_fire_tokens import TrashFireTokenCollection
from llm_attacks_bishopfox.dumpster_fires.trash_fire_tokens import remove_empty_leading_and_trailing_tokens
from llm_attacks_bishopfox.jailbreak_detection.jailbreak_detection import JailbreakDetectionRuleResult
from llm_attacks_bishopfox.jailbreak_detection.jailbreak_detection import LLMJailbreakDetector
from llm_attacks_bishopfox.jailbreak_detection.jailbreak_detection import LLMJailbreakDetectorRuleSet
from llm_attacks_bishopfox.minimal_gcg.adversarial_content_utils import AdversarialContentManager
from llm_attacks_bishopfox.minimal_gcg.adversarial_content_utils import get_default_generic_role_indicator_template
from llm_attacks_bishopfox.minimal_gcg.adversarial_content_utils import load_conversation_template
from llm_attacks_bishopfox.minimal_gcg.adversarial_content_utils import register_missing_conversation_templates
from llm_attacks_bishopfox.minimal_gcg.opt_utils import get_filtered_cands
from llm_attacks_bishopfox.minimal_gcg.opt_utils import get_logits
from llm_attacks_bishopfox.minimal_gcg.opt_utils import get_missing_pad_token_names
from llm_attacks_bishopfox.minimal_gcg.opt_utils import load_model_and_tokenizer
from llm_attacks_bishopfox.minimal_gcg.opt_utils import get_adversarial_content_candidates
from llm_attacks_bishopfox.minimal_gcg.opt_utils import target_loss
from llm_attacks_bishopfox.minimal_gcg.opt_utils import token_gradients
from llm_attacks_bishopfox.util.util_functions import add_values_to_list_if_not_already_present
from llm_attacks_bishopfox.util.util_functions import comma_delimited_string_to_integer_array
from llm_attacks_bishopfox.util.util_functions import get_escaped_string
from llm_attacks_bishopfox.util.util_functions import get_elapsed_time_string
from llm_attacks_bishopfox.util.util_functions import get_file_content
from llm_attacks_bishopfox.util.util_functions import get_now
from llm_attacks_bishopfox.util.util_functions import get_random_token_id
from llm_attacks_bishopfox.util.util_functions import get_random_token_ids
from llm_attacks_bishopfox.util.util_functions import get_time_string
from llm_attacks_bishopfox.util.util_functions import numeric_string_to_float
from llm_attacks_bishopfox.util.util_functions import numeric_string_to_int
from llm_attacks_bishopfox.util.util_functions import safely_write_text_output_file
from llm_attacks_bishopfox.util.util_functions import str2bool
from llm_attacks_bishopfox.util.util_functions import update_elapsed_time_string
from peft import PeftModel
from torch.quantization import quantize_dynamic
from torch.quantization.qconfig import float_qparams_weight_only_qconfig
from transformers.generation import GenerationConfig

SAFETENSORS_WEIGHTS_FILE_NAME = "adapter_model.safetensors"

# threshold for warning the user if the specified PyTorch device already has more than this percent of its memory reserved
# 0.1 = 10%
torch_device_reserved_memory_warning_threshold = 0.1

locale.setlocale(locale.LC_ALL, '')

# Workaround for glitchy Protobuf code somewhere
# See https://stackoverflow.com/questions/75042153/cant-load-from-autotokenizer-from-pretrained-typeerror-duplicate-file-name
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"]="python"

loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
for logger in loggers:
    logger.setLevel(logging.WARNING)

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
    # display_string += f"System:\n"
    # display_string += f"\tTotal physical memory: {system_physical_memory:n} bytes\n"
    # display_string += f"\tMemory in use: {system_in_use_memory:n} bytes\n"
    # display_string += f"\tAvailable memory: {system_available_memory:n} bytes\n"
    # display_string += f"\tMemory utilization: {system_memory_util_percent:.0%}\n"

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

def generate(attack_params, model, tokenizer, adversarial_content_manager, adversarial_content, gen_config = None, do_sample = True, generate_full_output = False, include_target_content = False):
    working_gen_config = gen_config
    # Copy the generation config to avoid changing the original
    if gen_config is None:
        working_gen_config = GenerationConfig.from_dict(config_dict = model.generation_config.to_dict())
    else:
        working_gen_config = GenerationConfig.from_dict(config_dict = gen_config.to_dict())
    
    if attack_params.model_temperature != 1.0 and do_sample:
        working_gen_config.do_sample = True
        working_gen_config.temperature = attack_params.model_temperature
    #else:
    #    working_gen_config.do_sample = False
    #working_gen_config.temperature = attack_params.model_temperature
    if attack_params.display_full_failed_output or generate_full_output:
        working_gen_config.max_new_tokens = attack_params.full_decoding_max_new_tokens
    else:
        working_gen_config.max_new_tokens = attack_params.generation_max_new_tokens

    result = GenerationResults()
    result.max_new_tokens = working_gen_config.max_new_tokens

    result.input_token_id_data = adversarial_content_manager.get_prompt(adversarial_content = adversarial_content, force_python_tokenizer = attack_params.force_python_tokenizer)
    input_ids = result.input_token_id_data.get_input_ids_as_tensor().to(attack_params.device)
    input_ids_sliced = input_ids
    if not include_target_content:
        input_ids_sliced = input_ids[:result.input_token_id_data.slice_data.assistant_role.stop]
    input_ids_converted = input_ids_sliced.to(model.device).unsqueeze(0)
    attn_masks = torch.ones_like(input_ids_converted).to(model.device)
        
    result.output_token_ids = model.generate(input_ids_converted, 
                                attention_mask=attn_masks, 
                                generation_config=working_gen_config,
                                pad_token_id=tokenizer.pad_token_id)[0]
    
    result.output_token_ids_output_only = result.output_token_ids[result.input_token_id_data.slice_data.assistant_role.stop:]
    
    result.generation_input_token_ids = result.output_token_ids[result.input_token_id_data.slice_data.goal.start:result.input_token_id_data.slice_data.control.stop]
    
    #print(f"[generate] Debug: result.input_token_id_data = {result.input_token_id_data}, result.generation_input_token_ids = {result.generation_input_token_ids}, result.output_token_ids = {result.output_token_ids}, result.output_token_ids_output_only = {result.output_token_ids_output_only}")
    
    return result
    
def check_for_attack_success(attack_params, model, tokenizer, adversarial_content_manager, adversarial_content, jailbreak_detector, gen_config = None, do_sample = True, include_target_content = False):
    #input_id_data, input_token_ids, output_ids_llm_output_only, output_token_ids = generate(attack_params,
    generation_results = generate(attack_params,
                                        model, 
                                        tokenizer, 
                                        adversarial_content_manager, 
                                        adversarial_content, 
                                        gen_config=gen_config,
                                        do_sample = do_sample,
                                        include_target_content = include_target_content)
                                        
    #gen_str = tokenizer.decode(generated_tokens).strip()
    
    result_ar_info_data = AttackResultInfoData()
    result_ar_info_data.set_values(tokenizer, generation_results.max_new_tokens, generation_results.input_token_id_data.full_prompt_token_ids, generation_results.output_token_ids, generation_results.generation_input_token_ids, generation_results.output_token_ids_output_only)
    
    #print(f"[check_for_attack_success] Debug: result_ar_info_data = {result_ar_info_data.to_json()}")
    #print(f"[check_for_attack_success] Debug: result_ar_info_data.decoded_generated_prompt_string = '{result_ar_info_data.decoded_generated_prompt_string}', \nresult_ar_info_data.decoded_llm_generation_string = '{result_ar_info_data.decoded_llm_generation_string}', \nresult_ar_info_data.decoded_user_input_string = '{result_ar_info_data.decoded_user_input_string}', \nresult_ar_info_data.decoded_llm_output_string = '{result_ar_info_data.decoded_llm_output_string}', \nresult_ar_info_data.decoded_generated_prompt_tokens = '{result_ar_info_data.decoded_generated_prompt_tokens}', \nresult_ar_info_data.decoded_llm_generation_tokens = '{result_ar_info_data.decoded_llm_generation_tokens}', \nresult_ar_info_data.decoded_user_input_tokens = '{result_ar_info_data.decoded_user_input_tokens}', \nresult_ar_info_data.decoded_llm_output_tokens = '{result_ar_info_data.decoded_llm_output_tokens}'")
    
    jailbroken = False
    
    if result_ar_info_data.decoded_llm_output_string.strip() != "":
        jailbreak_check_result = jailbreak_detector.check_string(result_ar_info_data.decoded_llm_output_string)

    if jailbreak_check_result == JailbreakDetectionRuleResult.SUCCESS:
        jailbroken = True
    #print(f"Jailbroken: {jailbroken} for generated string '{result_ar_info_data.decoded_llm_output_string}'")
    
    return jailbroken, result_ar_info_data, generation_results

def main(attack_params):

    attack_state = AttackState()

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
    numpy.random.seed(attack_params.numpy_random_seed)
    # PyTorch
    torch.manual_seed(attack_params.torch_manual_seed)
    # CUDA
    torch.cuda.manual_seed_all(attack_params.torch_cuda_manual_seed_all)
    
    random_generator_attack_params_device = torch.Generator(device = attack_params.device).manual_seed(attack_params.torch_manual_seed)
    random_generator_cpu = torch.Generator(device = 'cpu').manual_seed(attack_params.torch_manual_seed)
    random_generator_gradient = None

    start_dt = get_now()
    start_ts = get_time_string(start_dt)
    print(f"Starting at {start_ts}")
    main_loop_iteration_number = 0

    print_stats(attack_params)
    
    #successful_attacks = []
    successful_attack_count = 0
    model = None
    tokenizer = None
    adversarial_content_manager = None
    jailbreak_detector = LLMJailbreakDetector()
    jailbreak_detector.rule_set = attack_params.jailbreak_detection_rule_set
    
    # keep two arrays to avoid having to convert every item to JSON every iteration
    overall_result_data = BrokenHillResultData()    
    overall_result_data.start_date_time = start_ts
    #attack_data = []
    # keep another array to track adversarial values
    current_adversarial_content = None
    tested_adversarial_content = AdversarialContentList()
    random_seed_values = get_random_seed_list_for_comparisons()
    
    try:
        model_load_message = f"Loading model and tokenizer from '{attack_params.model_path}'."
        if attack_params.tokenizer_path is not None:
            model_load_message = f"Loading model from '{attack_params.model_path}' and tokenizer from {attack_params.tokenizer_path}."
        print(model_load_message)
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
        if attack_params.peft_adapter_path is not None:
            f"Loading PEFT model from '{attack_params.peft_adapter_path}'."
            try:
                model = PeftModel.from_pretrained(model, attack_params.peft_adapter_path)
            except Exception as e:
                print(f"Error: loading PEFT model from '{attack_params.peft_adapter_path}': {e}.")
                sys.exit(1)
        print_stats(attack_params)
        
        if isinstance(tokenizer.pad_token_id, type(None)):
            if attack_params.loss_slice_mode == LossSliceMode.ASSISTANT_ROLE_PLUS_FULL_TARGET_SLICE:
                print("Error: the padding token is not set for the current tokenizer, but the current loss slice algorithm requires that the list of target tokens be padded. Please specify a replacement token using the --missing-pad-token-replacement option.")
                sys.exit(1)
        
        #print(f"[main] Debug: getting max effective token value for model and tokenizer.")
        
        attack_params.generation_max_new_tokens = get_effective_max_token_value_for_model_and_tokenizer("--max-new-tokens", model, tokenizer, attack_params.generation_max_new_tokens)
        attack_params.full_decoding_max_new_tokens = get_effective_max_token_value_for_model_and_tokenizer("--max-new-tokens-final", model, tokenizer, attack_params.full_decoding_max_new_tokens)
        
        #additional_token_strings_case_insensitive = attack_params.not_allowed_token_list_case_insensitive
        # TKTK: add a localization option for these
        if attack_params.exclude_slur_tokens:
            #print(f"[main] Debug: adding slurs to the list that will be used to build the token denylist.")
            attack_params.not_allowed_token_list_case_insensitive = add_values_to_list_if_not_already_present(attack_params.not_allowed_token_list_case_insensitive, get_slurs())
        if attack_params.exclude_profanity_tokens:
            #print(f"[main] Debug: adding profanity to the list that will be used to build the token denylist.")
            attack_params.not_allowed_token_list_case_insensitive = add_values_to_list_if_not_already_present(attack_params.not_allowed_token_list_case_insensitive, get_profanity())
        if attack_params.exclude_other_offensive_tokens:
            #print(f"[main] Debug: adding other highly-problematic content to the list that will be used to build the token denylist.")
            attack_params.not_allowed_token_list_case_insensitive = add_values_to_list_if_not_already_present(attack_params.not_allowed_token_list_case_insensitive, get_other_highly_problematic_content())
        
        #print(f"[main] Debug: building token allowlist and denylist.")
        token_allow_and_deny_lists = get_token_allow_and_deny_lists(tokenizer, 
            attack_params.not_allowed_token_list, 
            device=attack_params.device, 
            additional_token_strings_case_insensitive = attack_params.not_allowed_token_list_case_insensitive, 
            filter_nonascii_tokens = attack_params.exclude_nonascii_tokens, 
            filter_nonprintable_tokens = attack_params.exclude_nonprintable_tokens, 
            filter_special_tokens = attack_params.exclude_special_tokens, 
            filter_additional_special_tokens = attack_params.exclude_additional_special_tokens, 
            filter_whitespace_tokens = attack_params.exclude_whitespace_tokens, 
            token_regex = attack_params.get_token_filter_regex()
            )        
        
        #print(f"Debug: token_allow_and_deny_lists.denylist = '{token_allow_and_deny_lists.denylist}', token_allow_and_deny_lists.allowlist = '{token_allow_and_deny_lists.allowlist}'")
        not_allowed_tokens = None
        if len(token_allow_and_deny_lists.denylist) > 0:
            #print(f"[main] Debug: getting not_allowed_tokens from token allowlist and denylist.")
            not_allowed_tokens = get_token_list_as_tensor(token_allow_and_deny_lists.denylist, device='cpu')
        #print(f"Debug: not_allowed_tokens = '{not_allowed_tokens}'")

        original_model_size = 0

        if attack_params.display_model_size:
            #print(f"[main] Debug: Determining model size.")
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
            #print(f"[main] Debug: converting model dtype to {attack_params.conversion_dtype}.")
            model = model.to(attack_params.conversion_dtype)

        if attack_params.quantization_dtype or attack_params.enable_static_quantization or attack_params.conversion_dtype:
            print("Warning: you've enabled quantization and/or type conversion, which are unlikely to work for the foreseeable future due to PyTorch limitations. Please see my comments in the source code for this tool.")
            if attack_params.display_model_size:
                quantized_model_size = get_model_size(model)
                size_factor = float(quantized_model_size) / float(original_model_size) * 100.0
                size_factor_formatted = f"{size_factor:.2f}%"
                print(f"Model size after reduction: {quantized_model_size} ({size_factor_formatted} of original size)")
        
        #print(f"[main] Debug: registering missing conversation templates.")
        register_missing_conversation_templates(attack_params)
        #print(f"[main] Debug: loading conversation template '{attack_params.template_name}'.")
        #conv_template = load_conversation_template(attack_params.template_name, generic_role_indicator_template = attack_params.generic_role_indicator_template, system_prompt=attack_params.custom_system_prompt, clear_existing_template_conversation=attack_params.clear_existing_template_conversation, conversation_template_messages=attack_params.conversation_template_messages)
        conv_template = load_conversation_template(attack_params.model_path, template_name = attack_params.template_name, generic_role_indicator_template = attack_params.generic_role_indicator_template, system_prompt=attack_params.custom_system_prompt, clear_existing_template_conversation=attack_params.clear_existing_template_conversation, conversation_template_messages=attack_params.conversation_template_messages)
        if attack_params.template_name is not None:
            if conv_template.name != attack_params.template_name:
                print(f"Warning: the template '{attack_params.template_name}' was specified, but fschat returned the template '{conv_template.name}' in response to that value.")
        if conv_template is None:
            print(f"Error: got a null conversation template when trying to load '{attack_params.template_name}'. This should never happen.")
            sys.exit(1)
        #print(f"Conversation template: '{conv_template.name}'")
        #print(f"Conversation template sep: '{conv_template.sep}'")
        #print(f"Conversation template sep2: '{conv_template.sep2}'")
        #print(f"Conversation template roles: '{conv_template.roles}'")
        #print(f"Conversation template system message: '{conv_template.system_message}'")
        #messages = json.dumps(conv_template.messages, indent=4)
        #print(f"Conversation template messages: '{messages}'")
        #print_stats(attack_params)

        #print(f"[main] Debug: creating a meticulously-curated treasury of trash fire tokens.")
        trash_fire_token_treasury = TrashFireTokenCollection.get_meticulously_curated_trash_fire_token_collection(tokenizer, conv_template)

        #print(f"[main] Debug: setting initial adversarial content.")
        
        initial_adversarial_content = None
        if attack_params.initial_adversarial_content_creation_mode == InitialAdversarialContentCreationMode.FROM_STRING:
            initial_adversarial_content = AdversarialContent.from_string(tokenizer, trash_fire_token_treasury, attack_params.initial_adversarial_string)
        
        if attack_params.initial_adversarial_content_creation_mode == InitialAdversarialContentCreationMode.SINGLE_TOKEN:
            single_token_id = None
            try:
                single_token_id = get_encoded_token(tokenizer, attack_params.initial_adversarial_token_string)
            except Exception as e:
                print(f"Error: encoding string '{attack_params.initial_adversarial_token_string}' to token: {e}.")
                sys.exit(1)
            if isinstance(single_token_id, type(None)):
                    print(f"Error: the selected tokenizer encoded the string '{attack_params.initial_adversarial_token_string}' to a null value.")
                    sys.exit(1)
            if isinstance(single_token_id, list):
                decoded_tokens = get_decoded_tokens(tokenizer, single_token_id)
                single_token_id, decoded_tokens = remove_empty_leading_and_trailing_tokens(trash_fire_token_treasury, single_token_id, decoded_tokens)
                if len(single_token_id) > 1:
                    print(f"Error: the selected tokenizer encoded the string '{attack_params.initial_adversarial_token_string}' as more than one token: {decoded_tokens} / {single_token_id}. You must specify a string that encodes to only a single token when using this mode.")
                    sys.exit(1)
                else:
                    single_token_id = single_token_id[0]
            attack_params.initial_adversarial_token_ids = []
            for i in range(0, attack_params.initial_adversarial_token_count):
                attack_params.initial_adversarial_token_ids.append(single_token_id)
            
            initial_adversarial_content = AdversarialContent.from_token_ids(tokenizer, trash_fire_token_treasury, attack_params.initial_adversarial_token_ids)

        if attack_params.initial_adversarial_content_creation_mode == InitialAdversarialContentCreationMode.FROM_TOKEN_IDS:
            initial_adversarial_content = AdversarialContent.from_token_ids(tokenizer, trash_fire_token_treasury, attack_params.initial_adversarial_token_ids)
        
        if attack_params.initial_adversarial_content_creation_mode == InitialAdversarialContentCreationMode.RANDOM_TOKEN_IDS:
            token_ids = get_random_token_ids(token_allow_and_deny_lists, attack_params.initial_adversarial_token_count)
            initial_adversarial_content = AdversarialContent.from_token_ids(tokenizer, trash_fire_token_treasury, token_ids)
        
        post_self_test_initial_adversarial_content_creation_modes = [ InitialAdversarialContentCreationMode.LOSS_TOKENS, InitialAdversarialContentCreationMode.RANDOM_TOKEN_IDS_LOSS_TOKEN_COUNT, InitialAdversarialContentCreationMode.SINGLE_TOKEN_LOSS_TOKEN_COUNT ]
        
        if attack_params.initial_adversarial_content_creation_mode in post_self_test_initial_adversarial_content_creation_modes:
            initial_adversarial_content = AdversarialContent.from_string(tokenizer, trash_fire_token_treasury, attack_params.initial_adversarial_string)
        
        # This should never actually happen, but just in case
        if initial_adversarial_content is None:
            print("Error: no initial adversarial content was specified.")
            sys.exit(1)
        
        #print(f"[main] Debug: determining if any tokens in the adversarial content are also in the token denylist, or not in the tokenizer at all.")
        tokens_in_denylist = []
        tokens_not_in_tokenizer = []
        for i in range(0, len(initial_adversarial_content.token_ids)):
            token_id = initial_adversarial_content.token_ids[i]
            if token_id in token_allow_and_deny_lists.denylist:
                if token_id not in tokens_in_denylist:
                    tokens_in_denylist.append(token_id)
            else:
                if token_id not in token_allow_and_deny_lists.allowlist:
                    if token_id not in tokens_not_in_tokenizer:
                        tokens_not_in_tokenizer.append(token_id)
        
        if len(tokens_in_denylist) > 0:
            token_list_string = ""
            for i in range(0, len(tokens_in_denylist)):
                decoded_token = get_escaped_string(get_decoded_token(tokenizer, tokens_in_denylist[i]))
                formatted_token = f"'{decoded_token}' (ID {tokens_in_denylist[i]})"
                if token_list_string == "":
                    token_list_string = formatted_token
                else:
                    token_list_string += ", {formatted_token}"
            print(f"Warning: the following tokens were found in the initial adversarial content, but are also present in the user-configured list of disallowed tokens: {token_list_string}. These tokens will be removed from the denylist, because otherwise the attack cannot proceed.")
            new_denylist = []
            for existing_denylist_index in range(0, len(token_allow_and_deny_lists.denylist)):
                if token_allow_and_deny_lists.denylist[existing_denylist_index] in tokens_in_denylist:
                    token_allow_and_deny_lists.allowlist.append(token_allow_and_deny_lists.denylist[existing_denylist_index])
                else:
                    new_denylist.append(token_allow_and_deny_lists.denylist[existing_denylist_index])
            token_allow_and_deny_lists.denylist = new_denylist
        if len(tokens_not_in_tokenizer) > 0:
            print(f"Warning: the following token IDs were found in the initial adversarial content, but were not found by the selected tokenizer: {tokens_not_in_tokenizer}. This may cause this test to fail, the script to crash, or other unwanted behaviour. Please modify your choice of initial adversarial content to avoid the conflict.")
        
        print(f"Initial adversarial content: {initial_adversarial_content.get_full_description()}")

        current_adversarial_content = initial_adversarial_content.copy()

        #print(f"[main] Debug: creating suffix manager.")
        adversarial_content_manager = AdversarialContentManager(tokenizer=tokenizer, 
                      conv_template = conv_template, 
                      instruction = attack_params.base_prompt, 
                      target = attack_params.target_output, 
                      adversarial_content = initial_adversarial_content.copy(),
                      trash_fire_tokens = trash_fire_token_treasury,
                      loss_slice_mode = attack_params.loss_slice_mode,
                      adversarial_content_placement = attack_params.adversarial_content_placement)
        #print_stats(attack_params)
         
        #print(f"Debug: Model dtype: {model.dtype}")
        
        #import pdb; pdb.Pdb(nosigint=True).set_trace()

        # TKTK: move this stuff and similar values into an AttackState class to support true suspend/resume
        # There is only one "last known good" adversarial value tracked to avoid the following scenario:
        # User has multiple types of rollback enabled
        # Rollback type 1 is triggered, and the script rolls back to the last-known-good adversarial value associated with rollback type 1
        # The rollback type 2 last-known-good adversarial value is updated to the value that caused the rollback
        # In the next iteration, rollback type 2 is triggered, and the script "rolls sideways" to the data that caused the first rollback, making the script branch into bad values
        last_known_good_adversarial_content = AdversarialContent()
        last_known_good_adversarial_content.token_ids = None
        last_known_good_adversarial_content.tokens = None
        last_known_good_adversarial_content.as_string = None
        best_loss_value = None
        best_jailbreak_count = None
        original_new_adversarial_value_candidate_count = attack_params.new_adversarial_value_candidate_count
        original_topk = attack_params.topk
        is_first_iteration = True

        # Keep this out until things like denied tokens are file paths instead of inline, to keep from infecting result files with bad words
        #overall_result_data.attack_params = attack_params

        #print(f"Debug: testing conversation template")
        print(f"Testing conversation template '{conv_template.name}'")
        conversation_template_tester = ConversationTemplateTester(adversarial_content_manager, model)
        conversation_template_test_results = conversation_template_tester.test_templates(verbose = attack_params.verbose_self_test_output)
        if len(conversation_template_test_results.result_messages) > 0:
            for i in range(0, len(conversation_template_test_results.result_messages)):
                print(conversation_template_test_results.result_messages[i])
        else:
            print(f"Broken Hill did not detect any issues with the conversation template in use with the current model.")

        
        #print(f"Debug: testing for jailbreak with no adversarial content")
        empty_adversarial_content = AdversarialContent.from_string(tokenizer, trash_fire_token_treasury, "")
        nac_jailbreak_result, nac_jailbreak_check_data, nac_jailbreak_check_generation_results = check_for_attack_success(attack_params, 
                                                model, 
                                                tokenizer,
                                                adversarial_content_manager, 
                                                empty_adversarial_content,
                                                jailbreak_detector,
                                                do_sample = False)
        
        overall_result_data.self_test_results["GCG-no_adversarial_content"] = nac_jailbreak_check_data
        
        if nac_jailbreak_result:
            #print(f"Error: Broken Hill tested the specified request string with no adversarial content and the current jailbreak detection configuration indicated that a jailbreak occurred. This may indicate that the model being targeted has no restrictions on providing the requested type of response, or that jailbreak detection is not configured correctly for the specified attack. The full conversation generated during this test was:\n'{nac_jailbreak_check_data.decoded_generated_prompt_string.strip()}'")
            print(f"Error: Broken Hill tested the specified request string with no adversarial content and the current jailbreak detection configuration indicated that a jailbreak occurred. The model's response to '{attack_params.base_prompt}' was:\n'{nac_jailbreak_check_data.decoded_llm_output_string.strip()}'\nThis may indicate that the model being targeted has no restrictions on providing the requested type of response, or that jailbreak detection is not configured correctly for the specified attack. The full conversation generated during this test was:\n'{nac_jailbreak_check_data.decoded_generated_prompt_string.strip()}'")
        else:
            print(f"Validated that a jailbreak was not detected for the given configuration when adversarial content was not included. The model's response to '{attack_params.base_prompt}' was:\n'{nac_jailbreak_check_data.decoded_llm_output_string.strip()}'\nIf this output does not match your expectations, verify your jailbreak detection configuration.")
        
        #print(f"Debug: testing for jailbreak when the LLM is prompted with the target string")
        target_jailbreak_result, target_jailbreak_check_data, target_jailbreak_check_generation_results = check_for_attack_success(attack_params, 
                                                model, 
                                                tokenizer,
                                                adversarial_content_manager, 
                                                empty_adversarial_content,
                                                jailbreak_detector,
                                                do_sample = False,
                                                include_target_content = True)
        
        overall_result_data.self_test_results["GCG-simulated_ideal_adversarial_content"] = target_jailbreak_check_data
        
        if target_jailbreak_result:
            print(f"Validated that a jailbreak was detected when the model was given a prompt that simulated an ideal adversarial string, using the given configuration. The model's response to '{attack_params.base_prompt}' when given the prefix '{attack_params.target_output}' was:\n'{target_jailbreak_check_data.decoded_llm_output_string.strip()}'\nIf this output does not match your expectations, verify your jailbreak detection configuration.")
        else:            
            print(f"Warning: Broken Hill did not detect a jailbreak when the model was given a prompt that simulated an ideal adversarial string, using the given configuration. The model's response to '{attack_params.base_prompt}' when given the prefix '{attack_params.target_output}' was:\n'{target_jailbreak_check_data.decoded_llm_output_string.strip()}'\nIf this output does meet your expectations for a successful jailbreak, verify your jailbreak detection configuration. If the model's response truly does not appear to indicate a successful jailbreak, the current attack configuration is unlikely to succeed. This may be due to an incorrect attack configuration (such as a conversation template that does not match the format the model expects), or the model may have been hardened against this type of attack.")

        if attack_params.operating_mode != BrokenHillMode.GCG_ATTACK_SELF_TEST:
            if nac_jailbreak_result or not target_jailbreak_result:
                if not attack_params.ignore_jailbreak_self_tests:
                    print(f"Because the jailbreak detection self-tests indicated that the results of this attack would likely not be useful, Broken Hill will exit. If you wish to perform the attack anyway, add the --ignore-jailbreak-self-tests option.")
                    sys.exit(1)

        # TKTK: if possible, self-test to determine if a loss calculation for what should be an ideal value actually has a score that makes sense.
        
        # TKTK: self-test to count the number of tokens in a role-switching operation for the model.
        
        # TKTK: detect if the tokenizer has support for .apply_chat_template(). If it does, use that to create a simple conversation using random sentinels, create the same conversation using this tool's get_prompt() code, and make sure that the role-switching tokens actually match, in case the tokenizer doesn't tokenize the text form of those tokens back to the token IDs that it recognizes as a role switch.
        
        if attack_params.operating_mode == BrokenHillMode.GCG_ATTACK_SELF_TEST:
            print(f"Broken Hill has completed all self-test operations and will now exit.")
            end_ts = get_time_string(get_now())            
            overall_result_data.end_date_time = end_ts
            if attack_params.json_output_file is not None:
                safely_write_text_output_file(attack_params.json_output_file, overall_result_data.to_json())
            sys.exit(0)

        print(f"Starting main loop")

        while main_loop_iteration_number < attack_params.max_iterations:
            is_success = False
            if user_aborted:
                break
            else:
                try:
                    print(f"---------")
                    current_dt = get_now()
                    current_ts = get_time_string(current_dt)
                    current_elapsed_string = get_elapsed_time_string(start_dt, current_dt)
                    print(f"{current_ts} - Main loop iteration {main_loop_iteration_number + 1} of {attack_params.max_iterations} - elapsed time {current_elapsed_string} - successful attack count: {successful_attack_count}")
                    overall_result_data.end_date_time = current_ts
                    overall_result_data.elapsed_time_string = current_elapsed_string
    
                    print_stats(attack_params)
                    print(f"---------")
                    
                    attack_data_previous_iteration = None
                    if main_loop_iteration_number > 0:
                        #attack_data_previous_iteration = attack_data[len(attack_data) - 1]
                        attack_data_previous_iteration = overall_result_data.attack_results[len(overall_result_data.attack_results) - 1]
                    
                    tested_adversarial_content.append_if_new(current_adversarial_content)
                    
                    # TKTK: split the actual attack step out into a separate subclass of an attack class.
                    # Maybe TokenPermutationAttack => GreedyCoordinateGradientAttack?
                    
                    # if this is not the first iteration, and the user has enabled emulation of the original attack, encode the current string, then use those IDs for this round instead of persisting everything in token ID format
                    if main_loop_iteration_number > 0:
                        if attack_params.reencode_adversarial_content_every_iteration:
                            reencoded_token_ids = tokenizer.encode(current_adversarial_content.as_string)
                            current_adversarial_content = AdversarialContent.from_token_ids(tokenizer, trash_fire_token_treasury, reencoded_token_ids)
                                       
                    # Step 1. Encode user prompt (behavior + adv suffix) as tokens and return token ids.
                    #print(f"[main - encoding user prompt + adversarial data] Debug: calling get_input_ids with current_adversarial_content = '{current_adversarial_content.get_short_description()}'")
                    input_id_data = adversarial_content_manager.get_prompt(adversarial_content = current_adversarial_content, force_python_tokenizer = attack_params.force_python_tokenizer)
                    #print_stats(attack_params)
                    
                    decoded_input_tokens = get_decoded_tokens(tokenizer, input_id_data.input_token_ids)
                    decoded_full_prompt_token_ids = get_decoded_tokens(tokenizer, input_id_data.full_prompt_token_ids)
                    decoded_control_slice = get_decoded_tokens(tokenizer, input_id_data.full_prompt_token_ids[input_id_data.slice_data.control])
                    decoded_target_slice = get_decoded_tokens(tokenizer, input_id_data.full_prompt_token_ids[input_id_data.slice_data.target])
                    decoded_loss_slice = get_decoded_tokens(tokenizer, input_id_data.full_prompt_token_ids[input_id_data.slice_data.loss])
                    decoded_loss_slice_string = get_escaped_string(tokenizer.decode(input_id_data.full_prompt_token_ids[input_id_data.slice_data.loss]))
                    #print(f"[main loop - input ID generation for token_gradients] Debug: decoded_input_tokens = '{decoded_input_tokens}'\n decoded_full_prompt_token_ids = '{decoded_full_prompt_token_ids}'\n decoded_control_slice = '{decoded_control_slice}'\n decoded_target_slice = '{decoded_target_slice}'\n decoded_loss_slice = '{decoded_loss_slice}'\n input_id_data.slice_data.control = '{input_id_data.slice_data.control}'\n input_id_data.slice_data.target = '{input_id_data.slice_data.target}'\n input_id_data.slice_data.loss = '{input_id_data.slice_data.loss}'\n input_id_data.input_token_ids = '{input_id_data.input_token_ids}'\n input_id_data.full_prompt_token_ids = '{input_id_data.full_prompt_token_ids}'")
                    
                    #print(f"Converting input IDs to device")
                    input_ids = input_id_data.get_input_ids_as_tensor().to(attack_params.device)
                    #print(f"Debug: input_ids after conversion = '{input_ids}'")
                    #print_stats(attack_params)

                    best_new_adversarial_content = None
                    attack_results_current_iteration = AttackResultInfoCollection()
                    
                    # preserve the PyTorch RNG state because the code in this section is likely to reset it a bunch of times
                    torch_rng_state = torch.get_rng_state()
                    
                    # declare these here so they can be cleaned up later
                    coordinate_gradient = None
                    # during the first iteration, do not generate variations - test the value that was given                    
                    if main_loop_iteration_number == 0:
                        print(f"Testing initial adversarial value '{current_adversarial_content.get_short_description()}'")
                    else:
                        # Step 2. Compute Coordinate Gradient
                        #print(f"Computing coordinate gradient")
                        coordinate_gradient = token_gradients(attack_params,
                                        model, 
                                        tokenizer,
                                        input_ids,
                                        input_id_data)
                        #print_stats(attack_params)

                        if isinstance(random_generator_gradient, type(None)):
                            random_generator_gradient = torch.Generator(device = coordinate_gradient.device).manual_seed(attack_params.torch_manual_seed)

                        # TKTK: add another option (--require-loss-decrease-threshold) that loops through just this with torch.no_grad() section until a candidate with a loss less than the current value + some threshold is found.
                        # That should greatly improve the efficiency of testing candidates when multiple random seeds are tested against, because it doesn't require using the LLMs to generate any output text.
                        # It probably makes sense to actually replace the --rollback-on-loss-increase option with that, and only use rollbacks for jailbreak count.
                        #
                        # Step 3. Sample a batch of new tokens based on the coordinate gradient.
                        # Notice that we only need the one that minimizes the loss.

                        with torch.no_grad():
                            
                            #print_stats(attack_params)
                            
                            got_candidate_list = False
                            new_adversarial_candidate_list = None
                            new_adversarial_candidate_list_filtered = None
                            
                            losses = None
                            best_new_adversarial_content_id = None
                            best_new_adversarial_content = None
                            current_loss = None
                            current_loss_as_float = None

                            # BEGIN: wrap in loss threshold check
                            candidate_list_meets_loss_threshold = False
                            num_iterations_without_acceptable_loss = 0
                            
                            # store the best value from each attempt in case no value is found that meets the threshold
                            best_failed_attempts = AdversarialContentList()                            
                            
                            while not candidate_list_meets_loss_threshold:

                                got_candidate_list = False

                                while not got_candidate_list:                                
                                    
                                    # Step 3.2 Randomly sample a batch of replacements.
                                    #print(f"Randomly sampling a batch of replacements")
                                    new_adversarial_candidate_list = None
                                    
                                    try:
                                        new_adversarial_candidate_list = get_adversarial_content_candidates(attack_params,
                                                       adversarial_content_manager,
                                                       current_adversarial_content, 
                                                       coordinate_gradient, 
                                                       random_generator_gradient,
                                                       random_generator_cpu,
                                                       random_generator_attack_params_device,
                                                       not_allowed_tokens = not_allowed_tokens)
                                    except RuntimeError as e:
                                        print(f"Error: attempting to generate a new set of candidate adversarial data failed with a low-level error: '{e}'. This is typically caused by excessive or conflicting candidate-filtering options. For example, the operator may have specified a regular expression filter that rejects long strings, but also specified a long initial adversarial value. This error is unrecoverable. If you believe the error was not due to excessive/conflicting filtering options, please submit an issue.")
                                        sys.exit(1)
     
                                    #print_stats(attack_params)
                                    #print(f"new_adversarial_candidate_list: {new_adversarial_candidate_list.adversarial_content}")
                                    
                                    # Note: I'm leaving this explanation here for historical reference
                                    # Step 3.3 This step ensures all adversarial candidates have the same number of tokens. 
                                    # This step is necessary because tokenizers are not invertible
                                    # so Encode(Decode(tokens)) may produce a different tokenization.
                                    # We ensure the number of token remains to prevent the memory keeps growing and run into OOM.
                                    #print(f"Getting filtered candidates")
                                    new_adversarial_candidate_list_filtered = get_filtered_cands(attack_params, adversarial_content_manager, 
                                                                        new_adversarial_candidate_list, 
                                                                        tested_adversarial_content,
                                                                        filter_cand=True, 
                                                                        current_adversarial_content = current_adversarial_content)
                                    if len(new_adversarial_candidate_list_filtered.adversarial_content) > 0:
                                        got_candidate_list = True
                                    else:
                                        # try to find a way to increase the number of options available
                                        something_has_changed = False
                                        standard_explanation_intro = "The attack has failed to generate any adversarial values at this iteration that meet the specified filtering criteria and have not already been tested."
                                        standard_explanation_outro = "You can try specifying larger values for --max-batch-size-new-adversarial-tokens and/or --max-topk to avoid this error, or enabling --add-token-when-no-candidates-returned and/or --delete-token-when-no-candidates-returned if they are not already enabled."
                                        
                                        if attack_params.add_token_when_no_candidates_returned:
                                            token_count_limited = True
                                            if isinstance(attack_params.candidate_filter_tokens_max, type(None)):
                                                token_count_limited = False
                                            if token_count_limited:
                                                if len(current_adversarial_content.token_ids) < attack_params.candidate_filter_tokens_max:
                                                    token_count_limited = False
                                            current_short_description = current_adversarial_content.get_short_description()
                                            if token_count_limited:
                                                print(f"{standard_explanation_intro} The option to add an additional token is enabled, but the current adversarial content {current_short_description} is already at the limit of {attack_params.candidate_filter_tokens_max} tokens.")
                                            else:
                                                current_adversarial_content.duplicate_random_token(tokenizer)
                                                new_short_description = current_adversarial_content.get_short_description()
                                                something_has_changed = True
                                                print(f"{standard_explanation_intro} Because the option to add an additional token is enabled, the current adversarial content has been modified from {current_short_description} to {new_short_description}.")
                                        #else:
                                        #    print(f"[main loop] Debug: the option to add an additional token is disabled.")
                                        
                                        if not something_has_changed:
                                            if attack_params.delete_token_when_no_candidates_returned:
                                                token_count_limited = True
                                                minimum_token_count = 1
                                                if isinstance(attack_params.candidate_filter_tokens_min, type(None)):
                                                    token_count_limited = False
                                                else:
                                                    if attack_params.candidate_filter_tokens_min > 1:
                                                        minimum_token_count = attack_params.candidate_filter_tokens_min
                                                    if len(current_adversarial_content.token_ids) > attack_params.candidate_filter_tokens_min:
                                                        token_count_limited = False
                                                if not token_count_limited:
                                                    if len(current_adversarial_content.token_ids) < 2:
                                                        token_count_limited = True
                                                current_short_description = current_adversarial_content.get_short_description()
                                                if token_count_limited:
                                                    print(f"{standard_explanation_intro} The option to delete a random token is enabled, but the current adversarial content {current_short_description} is already at the minimum of {minimum_token_count} token(s).")
                                                else:
                                                    current_adversarial_content.delete_random_token(tokenizer)
                                                    new_short_description = current_adversarial_content.get_short_description()
                                                    something_has_changed = True
                                                    print(f"{standard_explanation_intro} Because the option to delete a random token is enabled, the current adversarial content has been modified from {current_short_description} to {new_short_description}.")
                                            #else:
                                            #    print(f"[main loop] Debug: the option to delete a random token is disabled.")
                                        
                                        if not something_has_changed:
                                            new_new_adversarial_value_candidate_count = attack_params.new_adversarial_value_candidate_count + original_new_adversarial_value_candidate_count
                                            increase_new_adversarial_value_candidate_count = True
                                            if not isinstance(attack_params.max_new_adversarial_value_candidate_count, type(None)):
                                                if new_new_adversarial_value_candidate_count > attack_params.max_new_adversarial_value_candidate_count:
                                                    new_new_adversarial_value_candidate_count = attack_params.max_new_adversarial_value_candidate_count
                                                    if new_new_adversarial_value_candidate_count <= attack_params.new_adversarial_value_candidate_count:
                                                        increase_new_adversarial_value_candidate_count = False
                                                    #else:
                                                    #    print(f"[main loop] Debug: new_new_adversarial_value_candidate_count > attack_params.new_adversarial_value_candidate_count.")
                                                #else:
                                                #    print(f"[main loop] Debug: new_new_adversarial_value_candidate_count <= attack_params.max_new_adversarial_value_candidate_count.")
                                            #else:
                                            #    print(f"[main loop] Debug: attack_params.max_new_adversarial_value_candidate_count is None.")
                                            if increase_new_adversarial_value_candidate_count:
                                                print(f"{standard_explanation_intro}  This may be due to excessive post-generation filtering options. The --batch-size-new-adversarial-tokens value is being increased from {attack_params.new_adversarial_value_candidate_count} to {new_new_adversarial_value_candidate_count} to increase the number of candidate values. {standard_explanation_outro}")
                                                attack_params.new_adversarial_value_candidate_count = new_new_adversarial_value_candidate_count
                                                something_has_changed = True
                                            #else:
                                            #    print(f"[main loop] Debug: not increasing the --batch-size-new-adversarial-tokens value.")
                                        
                                        if not something_has_changed:
                                            new_topk = attack_params.topk + original_topk
                                            increase_topk = True
                                            if not isinstance(attack_params.max_topk, type(None)):
                                                if new_topk > attack_params.max_topk:
                                                    new_topk = attack_params.max_topk
                                                    if new_topk <= attack_params.topk:
                                                        increase_topk = False
                                                    #else:
                                                    #    print(f"[main loop] Debug: new_topk > attack_params.topk.")
                                                #else:
                                                #    print(f"[main loop] Debug: new_topk <= attack_params.max_topk.")
                                            #else:
                                            #    print(f"[main loop] Debug: attack_params.max_topk is None.")
                                            if increase_topk:
                                                print(f"{standard_explanation_intro}  This may be due to excessive post-generation filtering options. The --topk value is being increased from {attack_params.topk} to {new_topk} to increase the number of candidate values. {standard_explanation_outro}")
                                                attack_params.topk = new_topk
                                                something_has_changed = True
                                            #else:
                                            #    print(f"[main loop] Debug: not increasing the --topk value.")
                                        
                                        if not something_has_changed:
                                            print(f"{standard_explanation_intro} This may be due to excessive post-generation filtering options. Because the 'topk' value has already reached or exceeded the specified maximum ({attack_params.max_topk}), and no other options for increasing the number of potential candidates is possible in the current configuration, the tool will now exit. {standard_explanation_outro}")
                                            sys.exit(1)
                                        

                                        
                                                
                                #print_stats(attack_params)
                                #print(f"new_adversarial_candidate_list_filtered: '{new_adversarial_candidate_list_filtered.to_dict()}'")
                                
                                # Step 3.4 Compute loss on these candidates and take the argmin.
                                #print(f"Getting logits")
                                logits, ids = get_logits(model = model, 
                                                         tokenizer = tokenizer,
                                                         input_ids = input_ids,
                                                         adversarial_content = current_adversarial_content, 
                                                         adversarial_candidate_list = new_adversarial_candidate_list_filtered, 
                                                         return_ids = True,
                                                         batch_size = attack_params.batch_size_get_logits) # decrease this number if you run into OOM.
                                #print_stats(attack_params)

                                #print(f"Calculating target loss")
                                losses = target_loss(attack_params, logits, ids, input_id_data, tokenizer)
                                #print_stats(attack_params)
                                #print(f"[main loop] Debug: losses = {losses}")

                                #print(f"Getting losses argmin")
                                best_new_adversarial_content_id = losses.argmin()
                                #print_stats(attack_params)
                                #print(f"[main loop] Debug: best_new_adversarial_content_id = {best_new_adversarial_content_id}")

                                #print(f"Setting best new adversarial content")
                                best_new_adversarial_content = new_adversarial_candidate_list_filtered.adversarial_content[best_new_adversarial_content_id].copy()
                                #print_stats(attack_params)

                                #print(f"Getting current loss")
                                current_loss = losses[best_new_adversarial_content_id]
                                #print_stats(attack_params)
                                current_loss_as_float = float(f"{current_loss.detach().cpu().numpy()}")
                                best_new_adversarial_content.original_loss = current_loss_as_float
                                
                                if isinstance(attack_params.required_loss_threshold, type(None)) or main_loop_iteration_number == 0:
                                    candidate_list_meets_loss_threshold = True
                                else:
                                    if attack_data_previous_iteration is None:
                                        candidate_list_meets_loss_threshold = True
                                    else:
                                        if isinstance(attack_data_previous_iteration.loss, type(None)):
                                            candidate_list_meets_loss_threshold = True
                                        if not candidate_list_meets_loss_threshold:
                                            if best_new_adversarial_content.original_loss <= (attack_data_previous_iteration.loss + attack_params.required_loss_threshold):
                                                candidate_list_meets_loss_threshold = True
                                if not candidate_list_meets_loss_threshold:
                                    num_iterations_without_acceptable_loss += 1
                                    best_failed_attempts.append_if_new(best_new_adversarial_content)
                                    loss_attempt_stats_message = f"{num_iterations_without_acceptable_loss} unsuccessful attempt(s) to generate a list of random candidates that has at least one candidate with a loss lower than {(attack_data_previous_iteration.loss + attack_params.required_loss_threshold)}"
                                    if not isinstance(attack_params.required_loss_threshold, type(None)):
                                        if attack_params.required_loss_threshold != 0.0:
                                            loss_attempt_stats_message += f" (previous loss of {attack_data_previous_iteration.loss} plus the specified threshold value {attack_params.required_loss_threshold})" 
                                    loss_attempt_stats_message += f". Best value during this attempt was {current_loss_as_float}."
                                    print(loss_attempt_stats_message)
                                    if not isinstance(attack_params.loss_threshold_max_attempts, type(None)):
                                        if num_iterations_without_acceptable_loss >= attack_params.loss_threshold_max_attempts:
                                            loss_attempt_result_message = f"{num_iterations_without_acceptable_loss} unsuccessful attempt(s) has reached the limit of {attack_params.loss_threshold_max_attempts} attempts."
                                            if attack_params.exit_on_loss_threshold_failure:
                                                loss_attempt_result_message += " Broken Hill has been configured to exit when this condition occurs."
                                                print(loss_attempt_result_message)
                                                sys.exit(1)
                                            else:
                                                best_new_adversarial_content = best_failed_attempts.get_content_with_lowest_loss()
                                                candidate_list_meets_loss_threshold = True
                                                loss_attempt_result_message += f" Broken Hill has been configured to use the adversarial content with the lowest loss discovered during this iteration when this condition occurs. Out of {len(best_failed_attempts.adversarial_content)} unique set(s) of tokens discovered during this iteration, the lowest loss value was {best_new_adversarial_content.original_loss} versus the previous loss of {attack_data_previous_iteration.loss}"
                                                if not isinstance(attack_params.required_loss_threshold, type(None)):
                                                    if attack_params.required_loss_threshold != 0.0:
                                                        loss_attempt_result_message += f" and threshold {attack_params.required_loss_threshold}"
                                                loss_attempt_result_message += ". The adversarial content with that loss value will be used."
                                                print(loss_attempt_result_message)
                                                
                                        
                            
                            # END: wrap in loss threshold check 

                            # Update the running current_adversarial_content with the best candidate
                            #print(f"Updating adversarial content - was '{current_adversarial_content.get_short_description()}', now '{best_new_adversarial_content.get_short_description()}'")
                            #print_stats(attack_params)
                            
                            print(f"Updating adversarial value to the best value out of the new permutation list and testing it.\nWas: {current_adversarial_content.get_short_description()} ({len(current_adversarial_content.token_ids)} tokens)\nNow: {best_new_adversarial_content.get_short_description()} ({len(best_new_adversarial_content.token_ids)} tokens)")
                            current_adversarial_content = best_new_adversarial_content
                            print(f"Loss value for the new adversarial value in relation to '{decoded_loss_slice_string}'\nWas: {attack_data_previous_iteration.loss}\nNow: {current_adversarial_content.original_loss}")
                            #print(f"Debug: decoded_loss_slice = '{decoded_loss_slice}'")
                            #print(f"Debug: input_id_data.full_prompt_token_ids[input_id_data.slice_data.loss] = '{input_id_data.full_prompt_token_ids[input_id_data.slice_data.loss]}'")
                        
                            #attack_results_current_iteration.loss = current_loss_as_float
                            attack_results_current_iteration.loss = current_adversarial_content.original_loss

                    # restore the PyTorch RNG state
                    #torch.set_rng_state(torch_rng_state)
                    # preserve the PyTorch RNG state because the code in this section is likely to reset it a bunch of times
                    # it's preserved twice because this is inside a block that may not occur
                    # but if it is tampered with, it will be in the next section
                    torch_rng_state = torch.get_rng_state()

                    attack_results_current_iteration.adversarial_content = current_adversarial_content.copy()
                    
                    # BEGIN: do for every random seed
                    prng_seed_index = -1
                    for randomized_test_number in range(0, attack_params.random_seed_comparisons + 1):
                        prng_seed_index += 1
                        attack_data_current_iteration = AttackResultInfo()
                        attack_data_current_iteration.model_path = attack_params.model_path
                        attack_data_current_iteration.tokenizer_path = attack_params.tokenizer_path
                        attack_data_current_iteration.numpy_random_seed = attack_params.numpy_random_seed
                        attack_data_current_iteration.torch_manual_seed = attack_params.torch_manual_seed
                        attack_data_current_iteration.torch_cuda_manual_seed_all = attack_params.torch_cuda_manual_seed_all
                        # For the first run, leave the model in its default do_sample configuration
                        do_sample = False
                        if randomized_test_number > 0:
                            # For all other runs, enable do_sample to randomize results
                            do_sample = True
                            # Pick the next random seed that's not equivalent to any of the initial values
                            got_random_seed = False
                            random_seed = random_seed_values[prng_seed_index]
                            while not got_random_seed:
                                seed_already_used = False
                                if random_seed == attack_params.numpy_random_seed:
                                    seed_already_used = True
                                if random_seed == attack_params.torch_manual_seed:
                                    seed_already_used = True
                                if random_seed == attack_params.torch_cuda_manual_seed_all:
                                    seed_already_used = True
                                if seed_already_used:
                                    prng_seed_index += 1
                                else:
                                    got_random_seed = True
                            #print(f"[main loop] Temporarily setting all random seeds to {random_seed} to compare results")
                            numpy.random.seed(random_seed)
                            torch.manual_seed(random_seed)
                            torch.cuda.manual_seed_all(random_seed)
                            attack_data_current_iteration.numpy_random_seed = random_seed
                            attack_data_current_iteration.torch_manual_seed = random_seed
                            attack_data_current_iteration.torch_cuda_manual_seed_all = random_seed
                    
                        #print(f"Checking for success")
                        is_success, jailbreak_check_data, jailbreak_check_generation_results = check_for_attack_success(attack_params, 
                                                model, 
                                                tokenizer,
                                                adversarial_content_manager, 
                                                current_adversarial_content,
                                                jailbreak_detector,
                                                do_sample = do_sample)            
                        #print_stats(attack_params)
                        if is_success:
                            attack_data_current_iteration.jailbreak_detected = True
                            attack_results_current_iteration.jailbreak_detection_count += 1
                        
                        # Get the current full user input from the first successful test
                        if attack_results_current_iteration.complete_user_input is None:
                            attack_results_current_iteration.complete_user_input = adversarial_content_manager.get_complete_input_string(jailbreak_check_generation_results.input_token_id_data)

                        #print(f"Passed:{is_success}\nCurrent best new adversarial content: '{current_adversarial_content.get_short_description()}'")
                        
                        full_output_dataset_name = "full_output"
                        
                        jailbreak_check_dataset_name = "jailbreak_check"
                        if attack_params.display_full_failed_output:
                            jailbreak_check_dataset_name = full_output_dataset_name
                        
                        attack_data_current_iteration.result_data_sets[jailbreak_check_dataset_name] = jailbreak_check_data
                        
                        # only generate full output if it hasn't already just been generated
                        if not attack_params.display_full_failed_output and is_success:
                            full_output_data = AttackResultInfoData()
                            # Note: for randomized variations where do_sample is True, the "full output" here will almost certainly differ from the values generated during jailbreak detection. I can't think of a great way around that.
                            generation_results = generate(attack_params, model, tokenizer, adversarial_content_manager, current_adversarial_content, do_sample = do_sample, generate_full_output = True)
                          
                            full_output_data.set_values(tokenizer, generation_results.max_new_tokens, generation_results.input_token_id_data.full_prompt_token_ids, generation_results.output_token_ids, generation_results.generation_input_token_ids, generation_results.output_token_ids_output_only)
                            
                            attack_data_current_iteration.result_data_sets[full_output_dataset_name] = full_output_data
                        
                        
                        attack_results_current_iteration.results.append(attack_data_current_iteration)
                        
                        # END: do for every random seed
                    
                    # reset back to specified random seeds if using extra tests
                    # only do this if using extra tests to avoid resetting the PRNG unnecessarily
                    if attack_params.random_seed_comparisons > 0:
                        #print(f"[main loop] Resetting random seeds back to {attack_params.numpy_random_seed}, {attack_params.torch_manual_seed}, and {attack_params.torch_cuda_manual_seed_all}.")
                        # NumPy
                        numpy.random.seed(attack_params.numpy_random_seed)
                        # PyTorch
                        torch.manual_seed(attack_params.torch_manual_seed)
                        # CUDA
                        torch.cuda.manual_seed_all(attack_params.torch_cuda_manual_seed_all)
                    
                    # restore the PyTorch RNG state
                    torch.set_rng_state(torch_rng_state)
                    
                    attack_results_current_iteration.update_unique_output_values()
                    iteration_status_message = f"-----------------\n"
                    iteration_status_message += f"Current input string:\n---\n{attack_results_current_iteration.results[0].get_first_result_data_set().decoded_user_input_string}\n---\n"
                    iteration_status_message += f"Successful jailbreak attempts detected: {attack_results_current_iteration.jailbreak_detection_count}, with {attack_results_current_iteration.unique_result_count} unique output(s) generated during testing:\n"
                    for uov_string in attack_results_current_iteration.unique_results.keys():
                        uov_count = attack_results_current_iteration.unique_results[uov_string]
                        iteration_status_message += f"--- {uov_count} occurrence(s): ---\n" 
                        iteration_status_message += uov_string
                        iteration_status_message += "\n"
                    iteration_status_message += f"---\n" 
                    iteration_status_message += f"Current best new adversarial content: '{current_adversarial_content.get_short_description()}'\n"
                    iteration_status_message += f"-----------------\n"                    
                    print(iteration_status_message)
                    
                    # TKTK: maybe make this a threshold
                    if attack_results_current_iteration.jailbreak_detection_count > 0:
                        successful_attack_count += 1
                    
                    #attack_data.append(attack_data_current_iteration)
                    #attack_data.append(attack_results_current_iteration)
                    overall_result_data.attack_results.append(attack_results_current_iteration)
                    if attack_params.json_output_file is not None:
                        #overall_result_data.attack_results.append(attack_results_current_iteration.to_dict())
                        safely_write_text_output_file(attack_params.json_output_file, overall_result_data.to_json())
                    
                    rollback_triggered = False
                    
                    if main_loop_iteration_number > 0:
                        rollback_message = ""
                        if attack_params.rollback_on_loss_increase:
                            if (attack_results_current_iteration.loss - attack_params.rollback_on_loss_threshold) > best_loss_value:
                                if attack_params.rollback_on_loss_threshold == 0.0:
                                    rollback_message += f"The loss value for the current iteration ({attack_results_current_iteration.loss}) is greater than the best value achieved during this run ({best_loss_value}). "
                                else:
                                    rollback_message += f"The loss value for the current iteration ({attack_results_current_iteration.loss}) is greater than the allowed delta of {attack_params.rollback_on_loss_threshold} from the best value achieved during this run ({best_loss_value}). "
                                rollback_triggered = True
                            #else:
                            #    print(f"[main loop] Debug: rollback not triggered by current loss value {attack_results_current_iteration.loss} versus current best value {best_loss_value} and threshold {attack_params.rollback_on_loss_threshold}.")
                        if attack_params.rollback_on_jailbreak_count_decrease:
                            if (attack_results_current_iteration.jailbreak_detection_count + attack_params.rollback_on_jailbreak_count_threshold) < best_jailbreak_count:
                                if attack_params.rollback_on_jailbreak_count_threshold == 0:
                                    rollback_message += f"The jailbreak detection count for the current iteration ({attack_results_current_iteration.jailbreak_detection_count}) is less than for the best count achieved during this run ({best_jailbreak_count}). "
                                else:
                                    rollback_message += f"The jailbreak detection count for the current iteration ({attack_results_current_iteration.jailbreak_detection_count}) is less than the allowed delta of {attack_params.rollback_on_jailbreak_count_threshold} from the best count achieved during this run ({best_jailbreak_count}). "
                                rollback_triggered = True
                            #else:
                            #    print(f"[main loop] Debug: rollback not triggered by current jailbreak count {attack_results_current_iteration.jailbreak_detection_count} versus current best value {best_jailbreak_count} and threshold {attack_params.rollback_on_jailbreak_count_threshold}.")
                        # TKTK: if use of a threshold has allowed a score to drop below the last best value for x iterations, roll all the way back to the adversarial value that resulted in the current best value
                        # maybe use a tree model, with each branch from a node allowed to decrease 50% the amount of the previous branch, and too many failures to reach the value of the previous branch triggers a rollback to that branch
                        # That would allow some random exploration of various branches, at least allowing for the possibility of discovering a strong value within them, but never getting stuck for too long
                        if rollback_triggered:
                            #rollback_message += f"Rolling back to the last-known-good adversarial data {last_known_good_adversarial_content.get_short_description()} for the next iteration instead of using this iteration's result {current_adversarial_content.get_short_description()}."
                            rollback_message += f"Rolling back to the last-known-good adversarial data for the next iteration instead of using this iteration's result.\nThis iteration:  '{current_adversarial_content.get_short_description()}'\nLast-known-good: {last_known_good_adversarial_content.get_short_description()}."
                            print(rollback_message)
                            # add the rejected result to the list of tested results to avoid getting stuck in a loop
                            tested_adversarial_content.append_if_new(current_adversarial_content)
                            # roll back
                            adversarial_content = last_known_good_adversarial_content.copy()

                    # only update the "last-known-good" results if no rollback was triggered (for any reason)
                    # otherwise, if someone has multiple rollback options enabled, and only one of them is tripped, the other path will end up containing bad data
                    if not rollback_triggered:
                        rollback_notification_message = f"Updating last-known-good adversarial value from {last_known_good_adversarial_content.get_short_description()} to {current_adversarial_content.get_short_description()}."
                        last_known_good_adversarial_content = current_adversarial_content.copy()
                        # only update these if they're improvements - they should work as a high water mark to avoid gradually decreasing quality over time when the rollback thresholds are enabled
                        update_loss = False
                        
                        if isinstance(best_loss_value, type(None)):
                            if not isinstance(attack_results_current_iteration.loss, type(None)):
                                update_loss = True
                        else:
                            if not isinstance(attack_results_current_iteration.loss, type(None)):
                                if attack_results_current_iteration.loss < best_loss_value:
                                    update_loss = True
                        if update_loss:
                            rollback_notification_message += f" Updating best loss value from {best_loss_value} to {attack_results_current_iteration.loss}."
                            best_loss_value = attack_results_current_iteration.loss
                        
                        update_jailbreak_count = False
                        if isinstance(best_jailbreak_count, type(None)):
                            if not isinstance(attack_results_current_iteration.jailbreak_detection_count, type(None)):
                                update_jailbreak_count = True
                        else:
                            if not isinstance(attack_results_current_iteration.jailbreak_detection_count, type(None)):
                                if attack_results_current_iteration.jailbreak_detection_count > best_jailbreak_count:
                                    update_jailbreak_count = True
                        if update_jailbreak_count:
                            rollback_notification_message += f" Updating best jailbreak count from {best_jailbreak_count} to {attack_results_current_iteration.jailbreak_detection_count}."
                            best_jailbreak_count = attack_results_current_iteration.jailbreak_detection_count
                        
                    # (Optional) Clean up the cache.
                    #print(f"Cleaning up the cache")
                    if coordinate_gradient is not None:
                        del coordinate_gradient
                    gc.collect()
                    #if "cuda" in attack_params.device:
                    #    torch.cuda.empty_cache()
                
                # Neither of the except KeyboardInterrupt blocks currently do anything because some inner code in another module is catching it first
                except KeyboardInterrupt:
                    #import pdb; pdb.Pdb(nosigint=True).post_mortem()
                    print(f"Exiting main loop early by request")
                    user_aborted = True
            if is_success and attack_params.break_on_success:
                break
            main_loop_iteration_number += 1
            overall_result_data.completed_iterations = main_loop_iteration_number

    except KeyboardInterrupt:
        #import pdb; pdb.Pdb(nosigint=True).post_mortem()
        print(f"Exiting early by request")
        user_aborted = True
    
    except torch.OutOfMemoryError as toome:
        print(f"Broken Hill ran out of memory on the specified PyTorch device. If you have not done so already, please consult the Broken Hill documentation regarding the sizes of models you can test given your device's memory. The list of command-line parameters contains several options you can use to reduce the amount of memory used during the attack as well. The exception details will be displayed below this message for troubleshooting purposes.")
        raise toome

    if not user_aborted:
        print(f"Main loop complete")
    print_stats(attack_params)

    end_dt = get_now()
    end_ts = get_time_string(end_dt)
    total_elapsed_string = get_elapsed_time_string(start_dt, end_dt)
    print(f"Finished after {main_loop_iteration_number} iterations at {end_ts} - elapsed time {total_elapsed_string} - successful attack count: {successful_attack_count}")
    overall_result_data.end_date_time = end_ts
    overall_result_data.elapsed_time_string = total_elapsed_string
    if attack_params.json_output_file is not None:
        safely_write_text_output_file(attack_params.json_output_file, overall_result_data.to_json())

def exit_if_unauthorized_overwrite(output_file_path, attack_params):
    if os.path.isfile(output_file_path):
        if attack_params.overwrite_output:
            print(f"Warning: overwriting output file '{output_file_path}'")
        else:
            print(f"Error: the output file '{output_file_path}' already exists. Specify --overwrite-output to replace it.")
            sys.exit(1)

if __name__=='__main__':
    print(get_logo())
    short_description = get_short_script_description()
    print(f"{script_name} version {script_version}, {script_date}\n{short_description}")
    
    attack_params = AttackParams()
    
    cuda_available = torch.cuda.is_available()
    mps_available = torch.backends.mps.is_available()
    
    if not cuda_available:
        print(f"Warning: this host does not appear to have a PyTorch CUDA back-end available. The default --device option has therefore been changed from '{attack_params.device}' to '{attack_params.device_fallback}'. Using CPU processing will result in significantly longer run times for this tool. Expect each iteration to take several hours instead of tens of seconds on a modern GPU with support for CUDA. If your host has CUDA hardware, you should investigate why PyTorch is not detecting it.")        
        attack_params.device = attack_params.device_fallback
    if mps_available:
        print(f"Warning: this host appears to be an Apple device with support for the Metal ('mps') PyTorch back-end. At the time this version of {script_name} was developed, the Metal back-end did not support some features that were critical to the attack code, such as nested tensors. If you believe that you are using a newer version of PyTorch that has those features implemented, you can try enabling the Metal back-end by specifying the --device mps command-line option. However, it is unlikely to succeed. This message will be removed when Bishop Fox has verified that the Metal back-end supports the necessary features.")  
    
    parser = argparse.ArgumentParser(description=get_script_description(),formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # TKTK: --mode full (currently the only behaviour) 
    # TKTK: --mode test-results - read an existing result file and test each of the generated values against a different processing engine / model / tokenizer / random seed / etc. combination. Should ignore any combinations that have already been performed
    # TKTK: --mode retest - rerun the same tests in an existing result file, without changing the model/tokenizer/seed/etc.
    # TKTK: --mode minimize-length - start with the specified adversarial content and remove any tokens that can be removed without causing the number of jailbreak successes to fall below a user-specified threshold, or the loss to increase above a user-specified threshold
    
    # TKTK: --mode test-jailbreak-rules - reads a result JSON file, apply a different set of jailbreak detection rules, and re-output the result.
    
    # TKTK: --mode stats - reads a result JSON file and outputs useful statistics, without having to remember jq syntax
        # Histogram of jailbreak success
        # Time-series graph of jailbreak count
        # Export adversarial content by jailbreak success and/or loss thresholds (without having to use jq)
        # etc.

    # TKTK: --attack gcg (currently the only behaviour, should remain the default)
    # TKTK: --attack tbtf ("token-by-token forward" brute-force attack that starts with one adversarial content token, tests literally every allowlisted token in the tokenizer, finds the one with the lowest loss, adds another token after the first, etc.)
    # TKTK: --attack tbtr ("token-by-token reverse" - same as tbtf, except it adds each adversarial content token before the previous token)
    
    #TKTK: load / save configuration from / to JSON
    #TKTK: save attack state to JSON at the beginning of every iteration
    #TKTK: load attack state from JSON
    #TKTK: save/load custom conversation template
    
    parser.add_argument("--model", required=True, type=str, 
        help="Path to the base directory for the large language model you want to attack, e.g. /home/blincoln/LLMs/StabilityAI/stablelm-2-1_6b-chat")
        
    parser.add_argument("--tokenizer", type=str, 
        help="(optional) Path to the base directory for the LLM tokenizer you want to use with the model instead of any tokenizer that may be included with the model itself. Intended for use with models such as Mamba that do not include their own tokenizer.")
        
    parser.add_argument("--peft-adapter", type=str, 
        help="(optional) Path to the base directory for a PEFT pre-trained model/adapter that is based on the model specified with --model. Used to load models such as Guanaco.")
        
    template_name_list = ", ".join(attack_params.get_known_template_names())
    
    parser.add_argument("--template", type=str, 
        help=f"An optional model type name, for selecting the correct chat template. Use --list-templates to view available options. If this option is not specified, the fastchat library will attempt to load the correct template based on the base model directory contents.")

    parser.add_argument("--list-templates", type=str2bool, nargs='?',
        const=True,
        help="Output a list of all template names for the version of the fastchat library you have installed (to use with --template), then exit.")
        
    parser.add_argument("--override-fschat-templates", type=str2bool, nargs='?',
        const=True,
        help="If .")
        
    parser.add_argument("--system-prompt", type=str, 
        help=f"Specify a custom value to use instead of the default system prompt for the conversation template.")

    parser.add_argument("--system-prompt-from-file", type=str, 
        help=f"Specify a custom value to use instead of the default system prompt for the conversation template by reading it from a file. The entire file will be used as a single string system prompt.")

    parser.add_argument("--clear-existing-conversation", type=str2bool, nargs='?',
        const=True, default=attack_params.clear_existing_template_conversation,
        help="Removes any existing non-system messages from the conversation template.")
        
    parser.add_argument("--template-messages-from-file", type=str, 
        help=f"Load custom messages into the conversation template by reading them from a JSON file. If --clear-existing-conversation is not specified, the messages will be appended to any existing messages in the list for the template. The format of the file should be '[[<role ID or name>, <message>],[<role ID or name>, <message>][...][<role ID or name>, <message>]]', e.g. [[0, \"Do you have any albums by Cindergarden or Mari Kattman?\"], [1, \"We have albums by both artists in stock!\"]] or [[\"Human\", \"Do you have any albums by The Birthday Massacre or Night Club?\"], [\"Assistant\", \"We have albums by both artists in stock!\"]]")

    parser.add_argument("--base-prompt", type=str, 
        help=f"The prompt the tool should try to jailbreak, e.g. 'Write a convincing anonymous death threat.'")
        
    parser.add_argument("--target-output", type=str, 
        help=f"The target output the tool should measure potential results against, e.g. 'Sure, here's a convincing anonymous death threat'")
    
    parser.add_argument("--auto-target", type=str, 
        help=f"Instead of manually specifying separate --base-prompt and --target-output values, specify a single goal (without a leading verb such as 'write'), and the tool will generate the base prompt and target output values, e.g. --auto-target 'a convincing anonymous death threat'")
    
    parser.add_argument("--device", default=attack_params.device, type=str, 
        help="The device to use for the PyTorch operations ('cuda', 'cuda:0', etc.). Using anything other than CUDA is unlikely to produce satisfactory results.")

    parser.add_argument("--self-test", type=str2bool, nargs='?',
        const=True,
        help="Performs self-testing only, then exits before performing an actual attack. Can be used to test basic model-handling support on systems without enough device memory to perform the GCG attack.")


    parser.add_argument("--initial-adversarial-string", default=attack_params.initial_adversarial_string, type=str, 
        help="The initial string to iterate on. Leave this as the default to perform the attack described in the original paper. Specify the output of a previous run to continue iterating at that point (more or less). Specify a custom value to experiment. Specify an arbitrary number of space-delimited exclamation points to perform the standard attack, but using a different number of initial tokens.")
    
    parser.add_argument("--initial-adversarial-string-base64", type=str, 
        help="Identical to --initial-adversarial-string, except that the value should be specified in Base64-encoded form. This allows easily specifying an initial adversarial string that includes newlines or other characters that are difficult to represent as arguments on the command line.")

    parser.add_argument("--initial-adversarial-token", type = str, 
        nargs = 2,
        metavar = ('token', 'count'),
        help="Specify the initial adversarial content as a single token repeated n times, e.g. for 24 copies of the token '?', --initial-adversarial-token-id '?' 24")

    parser.add_argument("--initial-adversarial-token-id", type = numeric_string_to_int, 
        nargs = 2,
        metavar = ('token_id', 'count'),
        help="Specify the initial adversarial content as a single token repeated n times, e.g. for 24 copies of the token with ID 71, --initial-adversarial-token-id 71 24")
    
    parser.add_argument("--initial-adversarial-token-ids", type=str, 
        help="Specify the initial adversarial content as a comma-delimited list of integer token IDs instead of a string. e.g. --initial-adversarial-token-ids '1,2,3,4,5'")
    
    parser.add_argument("--random-adversarial-tokens", type=numeric_string_to_int,
        help=f"Generate this many random tokens to use as the initial adversarial value instead of specifying explicit initial data. The list of possible tokens will be filtered using the same criteria defined by any of the candidate-filtering options that are also specified.")
    
    # TKTK: adversarial tokens are target tokens
    # TKTK: adversarial tokens are <character> * token count of target tokens
    # TKTK: adversarial tokens are loss tokens (this will require a little work)
    # TKTK: adversarial tokens are <character> * token count of loss tokens (this will require a little work)
    
    
    parser.add_argument("--reencode-every-iteration", type=str2bool, nargs='?',
        const=True,
        help="Emulate the original attack's behaviour of converting the adversarial content to a string and then back to token IDs at every iteration of the attack, instead of persisting it as token IDs only. Enabling this option will cause the number of tokens to change between iterations. Use the --adversarial-candidate-filter-tokens-min, --adversarial-candidate-filter-tokens-max, and/or --attempt-to-keep-token-count-consistent options if you want to try to control how widely the number of tokens varies.")
        
    parser.add_argument("--topk", type=numeric_string_to_int,
        default=attack_params.topk,
        help=f"The number of results assessed when determining the best possible candidate adversarial data for each iteration.")
        
    parser.add_argument("--max-topk", type=numeric_string_to_int,
        default = attack_params.max_topk,
        help=f"The maximum number to allow --topk to grow to when no candidates are found in a given iteration. Default: {attack_params.max_topk}.")

    parser.add_argument("--temperature", type=numeric_string_to_float,
        default=attack_params.model_temperature,
        help=f"'Temperature' value to pass to the model. Use the default value for deterministic results.")

    parser.add_argument("--random-seed-numpy", type=numeric_string_to_int,
        default=attack_params.numpy_random_seed,
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

    parser.add_argument("--number-of-tokens-to-update-every-iteration", type=numeric_string_to_int,
        default=attack_params.number_of_tokens_to_update_every_iteration,
        help="The number of tokens to randomly alter in candidate adversarial content during every iteration. If this option is set to 1 (the default), the gradient-sampling algorithm from Zou, Wang, Carlini, Nasr, Kolter, and Fredrikson's code is used. If it is set to any other value, the nanoGCG gradient-sampling algorithm is used instead.")
        
    parser.add_argument("--always-use-nanogcg-sampling-algorithm", type=str2bool, nargs='?',
        const=True, default=attack_params.always_use_nanogcg_sampling_algorithm,
        help="If this option is specified, the nanoGCG gradient-sampling algorithm is used even when --number-of-tokens-to-update-every-iteration is 1.")

    parser.add_argument("--new-adversarial-value-candidate-count", type=numeric_string_to_int,
        default=attack_params.new_adversarial_value_candidate_count,
        help=f"The number of candidate adversarial values to generate at every iteration. If you are running out of memory and this value is greater than 1, try reducing it. Alternatively, if you *aren't* running out of memory, you can try increasing this value for better performance.")
        
    parser.add_argument("--max-new-adversarial-value-candidate-count", type=numeric_string_to_int,
        default=attack_params.max_new_adversarial_value_candidate_count,
        help=f"The maximum amount that the number of candidate adversarial values is allowed to grow to when no new candidates are found.")

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
        help="Bias the adversarial content generation data to avoid using tokens that are not ASCII text.")

    parser.add_argument("--exclude-nonprintable-tokens", type=str2bool, nargs='?',
        const=True, default=attack_params.exclude_nonprintable_tokens,
        help="Bias the adversarial content generation data to avoid using tokens that are not printable.")

    # TKTK: make this --exclude-basic-special-tokens, add another --exclude-all-special-tokens if --exclude-additional-special-tokens doesn't do that already
    parser.add_argument("--exclude-special-tokens", type=str2bool, nargs='?',
        const=True, default=attack_params.exclude_special_tokens,
        help="Bias the adversarial content generation data to avoid using basic special tokens (begin/end of string, padding, unknown).")

    parser.add_argument("--exclude-additional-special-tokens", type=str2bool, nargs='?',
        const=True, default=attack_params.exclude_additional_special_tokens,
        help="Bias the adversarial content generation data to avoid using additional special tokens defined in the tokenizer configuration.")

    parser.add_argument("--exclude-whitespace-tokens", type=str2bool, nargs='?',
        const=True, default=False,
        help="Bias the adversarial content generation data to avoid using tokens that consist solely of whitespace characters.")
        
    parser.add_argument("--exclude-slur-tokens", type=str2bool, nargs='?',
        const=True, default=False,
        help="Bias the adversarial content generation data to avoid using tokens that are contained in a hardcoded list of slurs.")

    parser.add_argument("--exclude-profanity-tokens", type=str2bool, nargs='?',
        const=True, default=False,
        help="Bias the adversarial content generation data to avoid using tokens that are contained in a hardcoded list of profanity.")

    parser.add_argument("--exclude-other-offensive-tokens", type=str2bool, nargs='?',
        const=True, default=False,
        help="Bias the adversarial content generation data to avoid using tokens that are contained in a hardcoded list of other highly-offensive words.")

    parser.add_argument("--exclude-token", action='append', nargs='*', required=False,
        help=f"Bias the adversarial content generation data to avoid using the specified token (if it exists as a discrete value in the model). May be specified multiple times to exclude multiple tokens.")
        
    parser.add_argument("--excluded-tokens-from-file", type = str, required=False,
        help=f"Equivalent to calling --exclude-token for every line in the specified file.")

    parser.add_argument("--excluded-tokens-from-file-case-insensitive", type = str, required=False,
        help=f"Equivalent to calling --exclude-token for every line in the specified file, except that matching is performed without taking upper-/lower-case characters into account.")

    parser.add_argument("--exclude-newline-tokens", type=str2bool, nargs='?',
        const=True, default=False,
        help="A shortcut equivalent to specifying just about any newline token variations using --exclude-token.")

    #parser.add_argument("--exclude-three-hashtag-tokens", type=str2bool, nargs='?',
    #    const=True, default=False,
    #    help="A shortcut equivalent to specifying most variations on the token '###' using --exclude-token.")

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
        help=f"If this value is specified, *and* --reencode-every-iteration is also specified, candidate adversarial strings will be filtered out if they contain fewer than this number of tokens.")
        
    parser.add_argument("--adversarial-candidate-filter-tokens-max", type=numeric_string_to_int,
        help=f"If this value is specified, *and* --reencode-every-iteration is also specified, candidate adversarial strings will be filtered out if they contain more than this number of tokens.")

    parser.add_argument("--attempt-to-keep-token-count-consistent", type=str2bool, nargs='?',
        const=True, default=attack_params.attempt_to_keep_token_count_consistent,
        help="If this option is specified, *and* --reencode-every-iteration is also specified, enable the check from the original attack code that attempts to keep the number of tokens consistent between each adversarial string. This will cause all candidates to be excluded for some models, such as StableLM 2. If you want to limit the number of tokens (e.g. to prevent the attack from wasting time on single-token strings or to avoid out-of-memory conditions) --adversarial-candidate-filter-tokens-min and --adversarial-candidate-filter-tokens-max are generally much better options.")

    parser.add_argument("--add-token-when-no-candidates-returned", type=str2bool, nargs='?',
        const=True, default=attack_params.add_token_when_no_candidates_returned,
        help="If this option is specified, and the number of tokens in the adversarial content is below any restrictions specified by the operator, then a failure to generate any new/untested adversarial content variations will result in a random token in the content being duplicated, increasing the length of the adversarial content by one token.")

    parser.add_argument("--delete-token-when-no-candidates-returned", type=str2bool, nargs='?',
        const=True, default=attack_params.delete_token_when_no_candidates_returned,
        help="If this option is specified, and the number of tokens in the adversarial content is greater than any minimum specified by the operator, then a failure to generate any new/untested adversarial content variations will result in a random token in the content being deleted, reducing the length of the adversarial content by one token. If both this option and --add-token-when-no-candidates-returned are enabled, and the prequisites for both options apply, then a token will be added.")

    parser.add_argument("--random-seed-comparisons", type=numeric_string_to_int, default = attack_params.random_seed_comparisons,
        help=f"If this value is greater than zero, at each iteration, the tool will test results using the specified number of additional random seed values, to attempt to avoid focusing on fragile results. The sequence of random seeds is hardcoded to help make results deterministic.")
    
    # not currently used - see discussion in attack_classes.py
    #parser.add_argument("--scoring-mode", type=str, default="median", choices=[ "median", "average", "minimum", "maximum" ],
    #    help=f"If --random-seed-comparisons is set to 1 or more, use this statistical function to generate an overall score for the results. Default: median.")

    parser.add_argument("--generic-role-template", type=str, 
        help="The Python formatting string to use if fastchat defaults to a generic chat template. e.g --generic-role-template '[{role}]', '<|{role}|>'.")
    
    parser.add_argument("--trust-remote-code", type=str2bool, nargs='?',
        const=True, default=attack_params.load_options_trust_remote_code,
        help="When loading the model, pass 'trust_remote_code=True', which enables execution of arbitrary Python scripts included with the model. You should probably examine those scripts first before deciding if you're comfortable with this option. Currently required for some models, such as Phi-3.")
    parser.add_argument("--ignore-mismatched-sizes", type=str2bool, nargs='?',
        const=True, default=attack_params.load_options_ignore_mismatched_sizes,
        help="When loading the model, pass 'ignore_mismatched_sizes=True', which may allow you to load some models with mismatched size data. It will probably only let the tool get a little further before erroring out, though.")

    parser.add_argument("--jailbreak-detection-rules-file", type=str, 
        help=f"If specified, loads the jailbreak detection rule set from a JSON file instead of using the default rule set.")
    parser.add_argument("--write-jailbreak-detection-rules-file", type=str, 
        help=f"If specified, writes the jailbreak detection rule set to a JSON file and then exits. If --jailbreak-detection-rules-file is not specified, this will cause the default rules to be written to the file. If --jailbreak-detection-rules-file *is* specified, then the custom rules will be normalized and written in the current standard format to the output file.")

    parser.add_argument("--break-on-success", type=str2bool, nargs='?',
        const=True, default=attack_params.break_on_success,
        help="Stop iterating upon the first detection of a potential successful jailbreak.")
        
    parser.add_argument("--verbose-self-test-output", type=str2bool, nargs='?',
        const=True, default=attack_params.verbose_self_test_output,
        help="If self-test operations fail badly enough to output a comparison of strings generated, also output token ID and token information for debugging.")

    parser.add_argument("--ignore-jailbreak-self-tests", type=str2bool, nargs='?',
        const=True, default=attack_params.ignore_jailbreak_self_tests,
        help="Perform the attack even if the jailbreak self-tests indicate that the results will likely not be useful.")

    parser.add_argument("--required-loss-threshold", type=numeric_string_to_float,
        default=attack_params.required_loss_threshold,
        help=f"During the candidate adversarial content generation stage, require that the loss for the best value be lower than the previous loss plus this amount.")

    parser.add_argument("--loss-threshold-max-attempts", type=numeric_string_to_int,
        default=attack_params.loss_threshold_max_attempts,
        help=f"If --required-loss-threshold has been specified, make this many attempts at finding a value that meets the threshold before giving up. If --exit-on-loss-threshold-failure is *not* specified, Broken Hill will use the value with the lowest loss found during the attempt to find a value that met the threshold. If --exit-on-loss-threshold-failure is specified, Broken Hill will exit if it is unable to find a value that meets the requirement.")

    parser.add_argument("--exit-on-loss-threshold-failure", type=str2bool, nargs='?',
        const=True, default=attack_params.exit_on_loss_threshold_failure,
        help=f"During the candidate adversarial content generation stage, require that the loss for the best value be lower than the previous loss plus this amount.")

    parser.add_argument("--rollback-on-loss-increase", type=str2bool, nargs='?',
        const=True, default=attack_params.rollback_on_loss_increase,
        help="If the loss value increases between iterations, roll back to the last 'good' adversarial data. This option is not recommended, and included for experimental purposes only.")
    parser.add_argument("--rollback-on-loss-threshold", type=numeric_string_to_float,
        help=f"Equivalent to --rollback-on-loss-increase, but only if the loss value increases by more than the specified amount for a given iteration. Like --rollback-on-loss-increase, using this option is not recommended.")

    parser.add_argument("--rollback-on-jailbreak-count-decrease", type=str2bool, nargs='?',
        const=True, default=attack_params.rollback_on_jailbreak_count_decrease,
        help="If the number of jailbreaks detected decreases between iterations, roll back to the last 'good' adversarial data.")
    parser.add_argument("--rollback-on-jailbreak-count-threshold", type=numeric_string_to_int,
        help=f"Equivalent to --rollback-on-jailbreak-count-decrease, but only if the jailbreak count decreases by more than the specified amount for a given iteration. Can be useful for avoiding getting stuck in a local maximum.")
        
    # TKTK: branching tree rollback modes - see discussion in the main loop
    # Both of the existing rollback modes should be able to be handled as subsets of tree mode
    # Basic rollback would be a tree configuration that doesn't allow branching down to less-desirable values at all
    # Threshold rollback would need some kind of special handling, like "nodes inherit the overall best values instead of using their own values"
    
    # TKTK: --gamma-garden <integer> <floating-point value between 0.0 and 1.0>
    # If no results that meet the current "good" criteria are discovered for <integer> iterations, step through the list of token IDs for the current adversarial value
    # Use <floating-point value between 0.0 and 1.0> as a percent likelihood of randomly replacing each token ID
    # In the default mode, keep a list of all of the token IDs that have ever been selected by the GCG algorithm and select randomly from that list
    # TKTK: --neutron-garden mode that works identically, except that it selects the random token IDs from any in the tokenizer that are allowed by the other options, similiar to the --random-token-adversarial-string option
    # Both options should probably only allow a given token ID to be used as a random replacement once for each iteration unless there are no more to select from, to avoid results that are unlikely to be useful, like the corner case where a bunch of tokens are reverted back to "!".
    # For --gamma-garden mode, there should be a threshold value for minimum number of previously-generated tokens to select from before the mode can be triggered, for the same reason. If there are only three token IDs in the list, it doesn't make much sense to potentially replace several "good" tokens with "!". 
        
        
    # how to determine the loss slice
    
    parser.add_argument("--loss-slice-is-index-shifted-target-slice", type=str2bool, nargs='?',
        const=True, default=False,
        help="This option causes the loss slice to be determined by starting with the target slice, and decreasing the start and end indices by 1, so that the length remains identical to the target slice, but the loss slice sometimes includes at least part of the LLM-role-indicator token. This is the behaviour that the original GCG attack code used, and it is the default mode.")

    # TKTK: option to set the index shifting
    
    # TKTK: option to shift left, but pad the comparison values to the right, like with the LLM role options
        
    # TKTK: loss slice is index-shifted base prompt plus target (set to zero shift for no shift)

    # TKTK: loss slice is index-shifted base prompt plus target with padding
    
    parser.add_argument("--loss-slice-is-llm-role-and-truncated-target-slice", type=str2bool, nargs='?',
        const=True, default=False,
        help="This option causes the loss slice to be determined by starting with the token(s) that indicate the speaking role is switching from user to LLM, and includes as many of the tokens from the target string as will fit without the result exceeding the length of the target slice. This is similar to the original GCG attack code method (--loss-slice-is-index-shifted-target-slice), but should work with any LLM, even those that use multiple tokens to indicate a role change.")

    parser.add_argument("--loss-slice-is-llm-role-and-full-target-slice", type=str2bool, nargs='?',
        const=True, default=False,
        help="This option causes the loss slice to be determined by starting with the token(s) that indicate the speaking role is switching from user to LLM, and includes all of the target string.")

    parser.add_argument("--loss-slice-is-target-slice", type=str2bool, nargs='?',
        const=True, default=False,
        help="This option makes the loss slice identical to the target slice. This will break the GCG attack, so you should only use this option if you want to prove to yourself that shifting those indices really is a fundamental requirement for the GCG attack.")

    # parser.add_argument("--mellowmax", type=str2bool, nargs='?',
        # const=True, default=False,
        # help="If this option is specified, the attack will use the mellowmax loss algorithm (borrowed from nanoGCG) instead of the cross-entropy loss from Zou, Wang, Carlini, Nasr, Kolter, and Fredrikson's code.")

    # parser.add_argument("--mellowmax-alpha", type=numeric_string_to_float,
        # default=attack_params.mellowmax_alpha,
        # help=f"If --loss-mellowmax is specified, this setting controls the 'alpha' value (default: 1.0).")

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

    args = parser.parse_args()

    if args.list_templates:
        fc_template_list = []
        for fct_name in fastchat.conversation.conv_templates.keys():
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
            
    if args.peft_adapter:
        attack_params.peft_adapter_path = os.path.abspath(args.peft_adapter)
        if not os.path.isdir(attack_params.peft_adapter_path):
            print(f"The specified PEFT adapter directory ('{attack_params.peft_adapter_path}') does not appear to exist.")
            sys.exit(1)
        
    if args.template:
        attack_params.template_name = args.template

    if args.override_fschat_templates:
        attack_params.override_fschat_templates = args.override_fschat_templates

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

    initial_data_method_count = 0
    if args.initial_adversarial_string != attack_params.initial_adversarial_string:
        initial_data_method_count += 1
    attack_params.initial_adversarial_string = args.initial_adversarial_string
    
    if args.initial_adversarial_string_base64:
        initial_data_method_count += 1
        attack_params.initial_adversarial_string = base64.b64decode(args.initial_adversarial_string_base64).decode("utf-8")
    
    if args.initial_adversarial_token:
        initial_data_method_count += 1
        initial_token_string, attack_params.initial_adversarial_token_count = args.initial_adversarial_token
        attack_params.initial_adversarial_token_count = numeric_string_to_int(attack_params.initial_adversarial_token_count)
        attack_params.initial_adversarial_token_string = initial_token_string
        attack_params.initial_adversarial_content_creation_mode = InitialAdversarialContentCreationMode.SINGLE_TOKEN
        if attack_params.initial_adversarial_token_count < 1:
            print("Error: The number of tokens specified as the second parameter for  --initial-adversarial-token must be a positive integer.")
            sys.exit(1)
    
    if args.initial_adversarial_token_id:
        initial_data_method_count += 1
        initial_token_id, attack_params.initial_adversarial_token_count = args.initial_adversarial_token_id
        attack_params.initial_adversarial_token_ids = []
        for i in range(0, attack_params.initial_adversarial_token_count):
            attack_params.initial_adversarial_token_ids.append(initial_token_id)
        attack_params.initial_adversarial_content_creation_mode = InitialAdversarialContentCreationMode.FROM_TOKEN_IDS
        if attack_params.initial_adversarial_token_count < 1:
            print("Error: The number of tokens specified as the second parameter for  --initial-adversarial-token-id must be a positive integer.")
            sys.exit(1)

    if args.initial_adversarial_token_ids:
        initial_data_method_count += 1
        attack_params.initial_adversarial_token_ids = comma_delimited_string_to_integer_array(args.initial_adversarial_token_ids)
        attack_params.initial_adversarial_content_creation_mode = InitialAdversarialContentCreationMode.FROM_TOKEN_IDS
        if len(attack_params.initial_adversarial_token_ids) < 1:
            print("Error: At least one adversarial token ID must be specified when using --initial-adversarial-token-ids.")
            sys.exit(1)

    if args.random_adversarial_tokens:
        initial_data_method_count += 1
        attack_params.initial_adversarial_token_count = args.random_adversarial_tokens
        attack_params.initial_adversarial_content_creation_mode = InitialAdversarialContentCreationMode.RANDOM_TOKEN_IDS
        if attack_params.initial_adversarial_token_count < 1:
            print("Error: The value specified for --random-adversarial-tokens must be a positive integer.")
            sys.exit(1)

    if args.initial_adversarial_token_id:
        initial_data_method_count += 1
        attack_params.initial_adversarial_token_count = args.random_adversarial_tokens
        attack_params.initial_adversarial_content_creation_mode = InitialAdversarialContentCreationMode.RANDOM_TOKEN_IDS
        if attack_params.initial_adversarial_token_count < 1:
            print("Error: The value specified for --random-adversarial-tokens must be a positive integer.")
            sys.exit(1)

    
    if initial_data_method_count > 1:
        print("Error: only one of the following options may be specified: --initial-adversarial-string, --initial-adversarial-string-base64, --initial-adversarial-token, --initial-adversarial-token-id, --initial-adversarial-token-ids, --random-adversarial-tokens.")
        sys.exit(1)
    
    if args.self_test:
        attack_params.operating_mode = BrokenHillMode.GCG_ATTACK_SELF_TEST
        
    if args.reencode_every_iteration:
        attack_params.reencode_adversarial_content_every_iteration = args.reencode_every_iteration

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

    # TKTK: --evaluate-top-candidates <integer> to cause the top <integer> candidates generated from each gradient-based step to be evaluated for jailbreak success
    # TKTK: --evaluate-top-candidate-combinations <integer> to take the top <integer> candidates that occupy different positions in the list of token IDs, and evaluate every possible combination of those with the previous iteration's data for jailbreak success
    # If either or both options are enabled, the highest-scoring value will be passed on to the next iteration.
    # Q: how to handle multiple candidates with the same top score?
    # A: a quasi-queueing system that ties into the tree-based rollback mechanism:
    #   If there are multiple candidates that all have the same "best" score out of the set tested:
    #       Those candidates all become a chain of nodes with the same scoring data.
    #       The tool uses the last node as a starting point.
    #       Then if that branch doesn't result in any successes for enough iterations, the tool will "roll back" to the previous "best score" value in the chain.
    #   It's not perfect, because it's possible that some of the nodes will never be branched from again even though they're equally "good" starting points, but it would be fairly easy to implement.
    # 
    # Another option: a true queuing system, with support for handing off data to other instances of the tool
    #   e.g. there is a queue for each level of success score, and the user can set a threshold where results are only queued if they're within n of whatever scoring mechanisms are in use.
    #       a given instance of the tool will select the next value from the highest-value queue that has entries queued, process it, and if necessary return results to the queue.
    #   that would be really neat, but it's probably going to be awhile before I get around to implementing it.
    #   Seems like it would require a "queue server" that all of the instances would connect to.

    if not isinstance(args.loss_slice_is_llm_role_and_full_target_slice, type(None)) and args.loss_slice_is_llm_role_and_full_target_slice == True:
        attack_params.loss_slice_mode = LossSliceMode.ASSISTANT_ROLE_PLUS_FULL_TARGET_SLICE        
        
    if not isinstance(args.loss_slice_is_llm_role_and_truncated_target_slice, type(None)) and args.loss_slice_is_llm_role_and_truncated_target_slice == True:
        attack_params.loss_slice_mode = LossSliceMode.ASSISTANT_ROLE_PLUS_TRUNCATED_TARGET_SLICE

    if not isinstance(args.loss_slice_is_index_shifted_target_slice, type(None)) and args.loss_slice_is_index_shifted_target_slice == True:
        attack_params.loss_slice_mode = LossSliceMode.INDEX_SHIFTED_TARGET_SLICE
        #print("Warning: --loss-slice-is-index-shifted-target-slice was specified. This will work as expected with some LLMs, but likely fail to generate useful results for LLMs that have multi-token role indicators, such as Gemma and Llama.")

    if not isinstance(args.loss_slice_is_target_slice, type(None)) and args.loss_slice_is_target_slice == True:
        attack_params.loss_slice_mode = LossSliceMode.SAME_AS_TARGET_SLICE
        print("Warning: --loss-slice-is-target-slice was specified. This will prevent the GCG attack from working correctly. Expect poor results.")

    # if not isinstance(args.mellowmax, type(None)) and args.mellowmax == True:
        # attack_params.loss_algorithm = LossAlgorithm.MELLOWMAX  

    # if not isinstance(args.mellowmax_alpha, type(None)):
        # attack_params.mellowmax_alpha = args.mellowmax_alpha

    attack_params.topk = args.topk
    
    if not isinstance(args.max_topk, type(None)):
        attack_params.max_topk = args.max_topk

    attack_params.model_temperature = args.temperature

    attack_params.numpy_random_seed = args.random_seed_numpy

    attack_params.torch_manual_seed = args.random_seed_torch

    attack_params.torch_cuda_manual_seed_all = args.random_seed_cuda

    attack_params.max_iterations = args.max_iterations
    
    attack_params.number_of_tokens_to_update_every_iteration = args.number_of_tokens_to_update_every_iteration

    if not isinstance(args.always_use_nanogcg_sampling_algorithm, type(None)) and args.always_use_nanogcg_sampling_algorithm == True:
        attack_params.always_use_nanogcg_sampling_algorithm = True
        if attack_params.number_of_tokens_to_update_every_iteration == 1:
            print("Using the nanoGCG gradient-sampling algorithm even though only one token will be updated during each iteration.")

    attack_params.new_adversarial_value_candidate_count = args.new_adversarial_value_candidate_count
    
    attack_params.max_new_adversarial_value_candidate_count = args.max_new_adversarial_value_candidate_count
    
    if attack_params.max_new_adversarial_value_candidate_count < attack_params.new_adversarial_value_candidate_count:
        print(f"Warning: the value specified for --max-new-adversarial-token-candidate-count ({attack_params.max_new_adversarial_value_candidate_count}) was less than the value specified for --new-adversarial-token-candidate-count ({attack_params.new_adversarial_value_candidate_count}). Both values will be set to {attack_params.max_new_adversarial_value_candidate_count}.")
        attack_params.new_adversarial_value_candidate_count = attack_params.max_new_adversarial_value_candidate_count

    attack_params.batch_size_get_logits = args.batch_size_get_logits
    
    attack_params.generation_max_new_tokens = args.max_new_tokens
    
    attack_params.full_decoding_max_new_tokens = args.max_new_tokens_final

    attack_params.exclude_nonascii_tokens = args.exclude_nonascii_tokens
    
    attack_params.exclude_nonprintable_tokens = args.exclude_nonprintable_tokens
    
    attack_params.exclude_special_tokens = args.exclude_special_tokens
    
    attack_params.exclude_additional_special_tokens = args.exclude_additional_special_tokens
    
    if args.token_filter_regex:
        attack_params.token_filter_regex = args.token_filter_regex
    
    attack_params.candidate_filter_regex = args.adversarial_candidate_filter_regex
    
    if not isinstance(args.adversarial_candidate_filter_tokens_min, type(None)):
        if args.adversarial_candidate_filter_tokens_min < 1:
            print("--adversarial-candidate-filter-tokens-min must be a positive integer.")
            sys.exit(1)
        attack_params.candidate_filter_tokens_min = args.adversarial_candidate_filter_tokens_min
    
    if not isinstance(args.adversarial_candidate_filter_tokens_max, type(None)):
        if args.adversarial_candidate_filter_tokens_max < 1:
            print("--adversarial-candidate-filter-tokens-max must be a positive integer.")
            sys.exit(1)
        attack_params.candidate_filter_tokens_max= args.adversarial_candidate_filter_tokens_max
    
    attack_params.attempt_to_keep_token_count_consistent = args.attempt_to_keep_token_count_consistent
    
    attack_params.add_token_when_no_candidates_returned = args.add_token_when_no_candidates_returned
    
    attack_params.delete_token_when_no_candidates_returned = args.delete_token_when_no_candidates_returned
    
    if args.random_seed_comparisons < 0 or args.random_seed_comparisons > 253:
        print("--args-random-seed-comparisons must specify a value between 0 and 253.")
        sys.exit(1)
    attack_params.random_seed_comparisons = args.random_seed_comparisons
    if attack_params.random_seed_comparisons > 0 and attack_params.model_temperature == 1.0:
        print("--args-random-seed-comparisons can only be used if --temperature is set to a floating-point value greater than 1.0, because otherwise the seed values will be ignored.")
        sys.exit(1)
    
    # not currently used
    # if args.scoring_mode == "median":
        # attack_params.random_seed_scoring_mode = OverallScoringFunction.MEDIAN
    # if args.scoring_mode == "average":
        # attack_params.random_seed_scoring_mode = OverallScoringFunction.AVERAGE
    # if args.scoring_mode == "minimum":
        # attack_params.random_seed_scoring_mode = OverallScoringFunction.MINIMUM
    # if args.scoring_mode == "maximum":
        # attack_params.random_seed_scoring_mode = OverallScoringFunction.MAXIMUM    
    
    if not isinstance(args.adversarial_candidate_repetitive_line_limit, type(None)):
        if args.adversarial_candidate_repetitive_line_limit < 1:
            print("--adversarial-candidate-repetitive-line-limit must be a positive integer.")
            sys.exit(1)
        attack_params.candidate_filter_repetitive_lines = args.adversarial_candidate_repetitive_line_limit
        
    if not isinstance(args.adversarial_candidate_repetitive_token_limit, type(None)):
        if args.adversarial_candidate_repetitive_token_limit < 1:
            print("--adversarial-candidate-repetitive-token-limit must be a positive integer.")
            sys.exit(1)
        attack_params.candidate_filter_repetitive_tokens = args.adversarial_candidate_repetitive_token_limit
    
    if not isinstance(args.adversarial_candidate_newline_limit, type(None)):
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
    

    # jailbreak detection and related

    if args.jailbreak_detection_rules_file:
        rules_file = os.path.abspath(args.jailbreak_detection_rules_file)
        rules_file_content = get_file_content(rules_file, failure_is_critical = True)
        try:
            attack_params.jailbreak_detection_rule_set = LLMJailbreakDetectorRuleSet.from_json(rules_file_content)
        except FakeException as e:
            print(f"Error loading jailbreak detection rules from file '{rules_file}', content '{rules_file_content}': {e}.")
            sys.exit(1)
    else:
        attack_params.jailbreak_detection_rule_set = LLMJailbreakDetectorRuleSet.get_default_rule_set()
    
    if args.write_jailbreak_detection_rules_file:
        rules_output_file = os.path.abspath(args.write_jailbreak_detection_rules_file)
        exit_if_unauthorized_overwrite(rules_output_file, attack_params)
        try:
            rules_data = attack_params.jailbreak_detection_rule_set.to_dict()
            json_rules_data = json.dumps(rules_data, indent=4)
            safely_write_text_output_file(rules_output_file, json_rules_data)
            print(f"Wrote jailbreak detection rules to file '{rules_output_file}'.")
            sys.exit(0)
        except Exception as e:
            print(f"Error writing jailbreak detection rules to file '{rules_output_file}': {e}.")
            sys.exit(1)
    
    attack_params.verbose_self_test_output = args.verbose_self_test_output
    
    attack_params.ignore_jailbreak_self_tests = args.ignore_jailbreak_self_tests
    
    attack_params.break_on_success = args.break_on_success
    
    attack_params.rollback_on_loss_increase = args.rollback_on_loss_increase
    
    if not isinstance(args.required_loss_threshold, type(None)):
        attack_params.required_loss_threshold = args.required_loss_threshold

    if not isinstance(args.loss_threshold_max_attempts, type(None)):
        attack_params.loss_threshold_max_attempts = args.loss_threshold_max_attempts

    if not isinstance(args.required_loss_threshold, type(None)) and isinstance(args.loss_threshold_max_attempts, type(None)):
        print(f"Warning: --required-loss-threshold was specified without also specifying a maximum number of attempts using --loss-threshold-max-attempts. This will cause Broken Hill to potentially loop forever if it cannot find a value that meets the specified loss threshold.")

    if not isinstance(args.loss_threshold_max_attempts, type(None)) and isinstance(args.required_loss_threshold, type(None)):
        print(f"Warning: --loss-threshold-max-attempts was specified without also specifying --required-loss-threshold. --loss-threshold-max-attempts has no effect unless a threshold is also specified using --required-loss-threshold.")

    if args.exit_on_loss_threshold_failure:
        attack_params.exit_on_loss_threshold_failure = True

    if not isinstance(args.rollback_on_loss_threshold, type(None)):
        attack_params.rollback_on_loss_increase = True
        attack_params.rollback_on_loss_threshold = args.rollback_on_loss_threshold
    
    attack_params.rollback_on_jailbreak_count_decrease = args.rollback_on_jailbreak_count_decrease
    
    if not isinstance(args.rollback_on_jailbreak_count_threshold, type(None)):
        attack_params.rollback_on_jailbreak_count_decrease = True
        attack_params.rollback_on_jailbreak_count_threshold = args.rollback_on_jailbreak_count_threshold
    
    attack_params.display_full_failed_output = args.display_failure_output
    
    # other tweakable options
    
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

    if args.exclude_whitespace_tokens:
        attack_params.exclude_whitespace_tokens = True

    if args.exclude_slur_tokens:
        attack_params.exclude_slur_tokens = True

    if args.exclude_profanity_tokens:
        attack_params.exclude_profanity_tokens = True

    if args.exclude_other_offensive_tokens:
        attack_params.exclude_other_offensive_tokens = True
                
    # shortcut option processing
    #if args.exclude_three_hashtag_tokens:
        # If you want to disallow "###", you also have to disallow "#" and "##" or the generation algorithm will reconstruct "###" from them
    #    attack_params.not_allowed_token_list.append("#")
    #    attack_params.not_allowed_token_list.append("##")
    #    attack_params.not_allowed_token_list.append("###")

    if args.exclude_token:
        for elem in args.exclude_token:
            for et in elem:
                if et.strip() != "":
                    attack_params.not_allowed_token_list = add_value_to_list_if_not_already_present(attack_params.not_allowed_token_list, et)
                    #if et not in attack_params.not_allowed_token_list:
                    #    attack_params.not_allowed_token_list.append(et)

    if args.excluded_tokens_from_file:
        excluded_token_file = os.path.abspath(args.excluded_tokens_from_file)
        excluded_token_file_content = get_file_content(excluded_token_file, failure_is_critical = True)
        for l in excluded_token_file_content.splitlines():
            attack_params.not_allowed_token_list = add_value_to_list_if_not_already_present(attack_params.not_allowed_token_list, l.strip())

    if args.excluded_tokens_from_file_case_insensitive:
        excluded_token_file = os.path.abspath(args.excluded_tokens_from_file_case_insensitive)
        excluded_token_file_content = get_file_content(excluded_token_file, failure_is_critical = True)
        for l in excluded_token_file_content.splitlines():
            attack_params.not_allowed_token_list_case_insensitive = add_value_to_list_if_not_already_present(attack_params.not_allowed_token_list_case_insensitive, l.strip())


    # determine if any arbitrary code execution is possible during model load and handle accordingly
    if attack_params.peft_adapter_path is not None:
        if not os.path.isfile(os.path.join(attack_params.peft_adapter_path, SAFETENSORS_WEIGHTS_FILE_NAME)):
            peft_message = f"the specified PEFT adapter directory ('{attack_params.peft_adapter_path}') does not contain a safe tensors file ('{SAFETENSORS_WEIGHTS_FILE_NAME}'). Because the model weights are only available in Python 'pickle' format, loading the adapter could result in arbitrary code execution on your system. "
            if attack_params.load_options_trust_remote_code:
                peft_message += " The --trust-remote-code option was specified, so the model will be loaded."
                print(f"Warning: {peft_message}")
            else:
                peft_message += " If you trust the specified adapter and understand the implications of potentially running untrusted Python code, add the --trust-remote-code option to load the adapter."
                print(f"Error: {peft_message}")
                sys.exit(1)

    if args.json_output_file:
        attack_params.json_output_file = os.path.abspath(args.json_output_file)
        # test output ability now so that the user doesn't have to wait to find out that it will fail
        exit_if_unauthorized_overwrite(attack_params.json_output_file, attack_params)
        # if os.path.isfile(attack_params.json_output_file):
            # if attack_params.overwrite_output:
                # print(f"Warning: overwriting JSON output file '{attack_params.json_output_file}'")
            # else:
                # print(f"Error: the JSON output file '{attack_params.json_output_file}' already exists. Specify --overwrite-output to replace it.")
                # sys.exit(1)
        try:
            safely_write_text_output_file(attack_params.json_output_file, "")
        #except Exception as e:
        except Exception as e:
            print(f"Could not validate the ability to write to the file '{attack_params.json_output_file}': {e}")
            sys.exit(1)
    
    main(attack_params)
    
    
