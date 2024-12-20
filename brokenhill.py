#!/bin/env python3

script_name = "brokenhill.py"
script_version = "0.38"
script_date = "2024-12-18"

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
import copy
import datetime
# IMPORTANT: 'fastchat' is in the PyPi package 'fschat', not 'fastchat'!
#import fastchat.conversation
import fastchat as fschat
import fastchat.conversation as fschat_conversation
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
from llm_attacks_bishopfox.attack.attack_classes import AssociationRebuildException
from llm_attacks_bishopfox.attack.attack_classes import AttackInitializationException
from llm_attacks_bishopfox.attack.attack_classes import AttackParams
from llm_attacks_bishopfox.attack.attack_classes import AttackResultInfo
from llm_attacks_bishopfox.attack.attack_classes import AttackResultInfoCollection
from llm_attacks_bishopfox.attack.attack_classes import AttackResultInfoData
from llm_attacks_bishopfox.attack.attack_classes import BrokenHillMode
from llm_attacks_bishopfox.attack.attack_classes import BrokenHillRandomNumberGenerators
from llm_attacks_bishopfox.attack.attack_classes import BrokenHillResultData
from llm_attacks_bishopfox.attack.attack_classes import DecodingException
from llm_attacks_bishopfox.attack.attack_classes import EncodingException
from llm_attacks_bishopfox.attack.attack_classes import GenerationException
from llm_attacks_bishopfox.attack.attack_classes import InitialAdversarialContentCreationMode
from llm_attacks_bishopfox.attack.attack_classes import LossAlgorithm
from llm_attacks_bishopfox.attack.attack_classes import LossSliceMode
from llm_attacks_bishopfox.attack.attack_classes import LossThresholdException
from llm_attacks_bishopfox.attack.attack_classes import ModelDataFormatHandling
from llm_attacks_bishopfox.attack.attack_classes import MyCurrentMentalImageOfALargeValueShouldBeEnoughForAnyoneException
from llm_attacks_bishopfox.attack.attack_classes import OverallScoringFunction
from llm_attacks_bishopfox.attack.attack_classes import PersistableAttackState
from llm_attacks_bishopfox.attack.attack_classes import VolatileAttackState
from llm_attacks_bishopfox.attack.attack_classes import get_missing_pad_token_names
from llm_attacks_bishopfox.base.attack_manager import EmbeddingLayerNotFoundException
from llm_attacks_bishopfox.dumpster_fires.conversation_templates import SeparatorStyleConversionException
from llm_attacks_bishopfox.dumpster_fires.conversation_templates import ConversationTemplateSerializationException
from llm_attacks_bishopfox.dumpster_fires.trash_fire_tokens import TrashFireTokenException
from llm_attacks_bishopfox.dumpster_fires.trash_fire_tokens import get_decoded_token
from llm_attacks_bishopfox.dumpster_fires.trash_fire_tokens import get_decoded_tokens
from llm_attacks_bishopfox.jailbreak_detection.jailbreak_detection import LLMJailbreakDetector
from llm_attacks_bishopfox.jailbreak_detection.jailbreak_detection import LLMJailbreakDetectorRuleSet
from llm_attacks_bishopfox.json_serializable_object import JSONSerializationException
from llm_attacks_bishopfox.llms.large_language_models import LargeLanguageModelException
from llm_attacks_bishopfox.llms.large_language_models import LargeLanguageModelParameterException
from llm_attacks_bishopfox.logging import BrokenHillLogManager
from llm_attacks_bishopfox.logging import LoggingException
from llm_attacks_bishopfox.minimal_gcg.adversarial_content_utils import AdversarialContentManager
from llm_attacks_bishopfox.minimal_gcg.adversarial_content_utils import PromptGenerationException
from llm_attacks_bishopfox.minimal_gcg.adversarial_content_utils import register_custom_conversation_templates
from llm_attacks_bishopfox.minimal_gcg.opt_utils import GradientCreationException
from llm_attacks_bishopfox.minimal_gcg.opt_utils import GradientSamplingException
from llm_attacks_bishopfox.minimal_gcg.opt_utils import MellowmaxException
from llm_attacks_bishopfox.minimal_gcg.opt_utils import NullPaddingTokenException
from llm_attacks_bishopfox.minimal_gcg.opt_utils import PaddingException
from llm_attacks_bishopfox.minimal_gcg.opt_utils import get_adversarial_content_candidates
from llm_attacks_bishopfox.minimal_gcg.opt_utils import get_filtered_cands
from llm_attacks_bishopfox.minimal_gcg.opt_utils import get_logits
from llm_attacks_bishopfox.minimal_gcg.opt_utils import target_loss
from llm_attacks_bishopfox.minimal_gcg.opt_utils import token_gradients
from llm_attacks_bishopfox.statistics.statistical_tools import StatisticsException
from llm_attacks_bishopfox.teratogenic_tokens.language_names import HumanLanguageException
from llm_attacks_bishopfox.util.util_functions import BrokenHillFileIOException
from llm_attacks_bishopfox.util.util_functions import BrokenHillValueException
from llm_attacks_bishopfox.util.util_functions import FakeException
from llm_attacks_bishopfox.util.util_functions import PyTorchDevice
from llm_attacks_bishopfox.util.util_functions import comma_delimited_string_to_integer_array
from llm_attacks_bishopfox.util.util_functions import command_array_to_string
from llm_attacks_bishopfox.util.util_functions import delete_file
from llm_attacks_bishopfox.util.util_functions import get_broken_hill_state_file_name
from llm_attacks_bishopfox.util.util_functions import get_elapsed_time_string
from llm_attacks_bishopfox.util.util_functions import get_escaped_string
from llm_attacks_bishopfox.util.util_functions import get_file_content
from llm_attacks_bishopfox.util.util_functions import get_file_content_from_sys_argv
from llm_attacks_bishopfox.util.util_functions import get_log_level_names
from llm_attacks_bishopfox.util.util_functions import get_now
from llm_attacks_bishopfox.util.util_functions import get_random_token_id
from llm_attacks_bishopfox.util.util_functions import get_random_token_ids
from llm_attacks_bishopfox.util.util_functions import get_time_string
from llm_attacks_bishopfox.util.util_functions import load_json_from_file
from llm_attacks_bishopfox.util.util_functions import log_level_name_to_log_level
from llm_attacks_bishopfox.util.util_functions import numeric_string_to_float
from llm_attacks_bishopfox.util.util_functions import numeric_string_to_int
from llm_attacks_bishopfox.util.util_functions import safely_write_text_output_file
from llm_attacks_bishopfox.util.util_functions import str2bool
from llm_attacks_bishopfox.util.util_functions import update_elapsed_time_string
from llm_attacks_bishopfox.util.util_functions import verify_output_file_capability
from peft import PeftModel
from torch.quantization import quantize_dynamic
from torch.quantization.qconfig import float_qparams_weight_only_qconfig

logger = logging.getLogger(__name__)

SAFETENSORS_WEIGHTS_FILE_NAME = "adapter_model.safetensors"

# workarounds for f-strings
DOUBLE_QUOTE = '"'

# threshold for warning the user if the specified PyTorch device already has more than this percent of its memory reserved
# 0.1 = 10%
torch_device_reserved_memory_warning_threshold = 0.1

# Use the OS-level locale
locale.setlocale(locale.LC_ALL, '')

# Workaround for glitchy Protobuf code somewhere
# See https://stackoverflow.com/questions/75042153/cant-load-from-autotokenizer-from-pretrained-typeerror-duplicate-file-name
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"]="python"

# Workaround for overly-chatty-by-default PyTorch code
# all_loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
# for existing_logger in all_loggers:
    # existing_logger.setLevel(logging.WARNING)

def check_pytorch_devices(attack_params):
    all_devices = {}
    devices_above_threshold = {}
    if attack_params.using_cuda():
        cuda_devices = PyTorchDevice.get_all_cuda_devices()
        for i in range(0, len(cuda_devices)):
            d = cuda_devices[i]
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
            message += f"\t\tTotal memory: {d.total_memory:n} byte(s)\n"
            message += f"\t\tMemory in use across the entire device: {d.gpu_used_memory:n} byte(s)\n"
            message += f"\t\tCurrent memory utilization for the device as a whole: {d.total_memory_utilization:.0%}\n"
        logger.info(message)
    above_threshold_device_names = list(devices_above_threshold.keys())
    if len(above_threshold_device_names) > 0:
        above_threshold_device_names.sort()
        warning_message = f"The following PyTorch devices have more than {torch_device_reserved_memory_warning_threshold:.0%} of their memory reserved:\n"
        for dn in above_threshold_device_names:
            d = devices_above_threshold[dn]
            warning_message += f"\t{d.device_name} ({d.device_display_name}): {d.total_memory_utilization:.0%}\n"
        warning_message += f"If you encounter out-of-memory errors when using Broken Hill, consider suspending other processes that use GPU resources, to maximize the amount of memory available to PyTorch. For example, on Linux desktops with a GUI enabled, consider switching to a text-only console to suspend the display manager and free up the associated VRAM. On Debian, Ctrl-Alt-F2 switches to the second console, and Ctrl-Alt-F1 switches back to the default console.\n"
        logger.warning(warning_message)



def main(attack_params, log_manager):
    user_aborted = False
    abnormal_termination = False

    general_error_guidance_message = "\n\nIf this issue is due to a temporary or operator-correctable condition (such as another process using device memory that would otherwise be available to Broken Hill) you may be able to continue execution (after correcting any issues) using the instructions that Broken Hill should generate before exiting.\n"    
    bug_guidance_message = "\n\nIf this error occurred while using Broken Hill with an LLM in the list of officially tested models, using the recommended options for that model, is not discussed in the troubleshooting documentation, and is not the result of an operator-correctable condition (such as an invalid path, insufficient memory, etc.), please open an issue with the Broken Hill developers, including steps to reproduce the error.\nIf this error occurred while using Broken Hill with an LLM that is not in the list of tested models, you may submit a feature request to add support for the model.\nIf this error occurred while using Broken Hill with an LLM in the list of officially tested models, but *not* using the recommended options for that model, please try using the recommended options instead."    

    attack_state = VolatileAttackState()
    attack_state.log_manager = log_manager
    if attack_params.load_state_from_file:
        state_file_dict = None
        try:
            state_file_dict = load_json_from_file(attack_params.load_state_from_file)
        except Exception as e:
            logger.critical(f"Could not load state JSON data from '{attack_params.load_state_from_file}': {e}\n{traceback.format_exc()}")
            sys.exit(1)

        # Loading the AttackParams directly from the saved state won't work, because then the user wouldn't be able to override them with other explicit command-line options.
        # attack_params is already a merged version of whatever combination of sources was specified on the command line.
        #merged_attack_params = attack_params.copy()
        # Everything else is then loaded from the persistable state data.
        attack_state.persistable = PersistableAttackState.apply_dict(attack_state.persistable, state_file_dict)
        attack_state.restore_from_persistable_data()
        
        if attack_state.persistable.broken_hill_version != script_version:
            logger.warning(f"The state file '{attack_params.load_state_from_file}' was generated by Broken Hill version {attack_state.persistable.broken_hill_version}, but is being loaded using Broken Hill version {script_version}. If you need results that match as closely as possible, you should use Broken Hill version {attack_state.persistable.broken_hill_version} instead.")
        # Finally, the AttackParams are replaced with the merged version generated earlier, by the first line outside this if block
        #attack_state.persistable.attack_params = merged_attack_params
    
    attack_state.persistable.attack_params = attack_params
    attack_state.persistable.broken_hill_version = script_version
    
    # Saving the options is done here instead of earlier so that they're the merged result of all possible option sources
    if attack_state.persistable.attack_params.operating_mode == BrokenHillMode.SAVE_OPTIONS:
        try:
            options_data = attack_state.persistable.attack_params.to_dict()
            json_options_data = json.dumps(options_data, indent=4)
            safely_write_text_output_file(attack_state.persistable.attack_params.save_options_path, json_options_data)
            logger.info(f"The current Broken Hill configuration has been written to '{attack_state.persistable.attack_params.save_options_path}'.")
        except Exception as e:
            logger.critical(f"Could not write the current Broken Hill configuration to '{attack_state.persistable.attack_params.save_options_path}': {e}\n{traceback.format_exc()}")
            sys.exit(1)
        sys.exit(0)
    
    if attack_state.persistable.attack_params.torch_cuda_memory_history_file is not None and attack_state.persistable.attack_params.using_cuda():
        torch.cuda.memory._record_memory_history()
    
    if not attack_params.load_state_from_file:
        attack_state.initialize_devices()
        attack_state.initialize_random_number_generators()        
        attack_state.persistable.initialize_language_manager()
    
    attack_state.initialize_jailbreak_detector()
    
    if attack_state.persistable.attack_params.save_state:
        # if there is not an existing state file, create a new one in the specified directory
        create_new_state_file = True
        if attack_state.persistable.attack_params.state_file is not None:
            if attack_state.persistable.attack_params.overwrite_output and attack_state.persistable.attack_params.overwrite_existing_state:
                create_new_state_file = False
        if create_new_state_file:
            attack_state.persistable.attack_params.state_file = os.path.join(attack_state.persistable.attack_params.state_directory, get_broken_hill_state_file_name(attack_state))

        # only test write capability if there is not an existing state file, so it's not overwritten
        if not os.path.isfile(attack_state.persistable.attack_params.state_file):
            verify_output_file_capability(attack_state.persistable.attack_params.state_file, attack_state.persistable.attack_params.overwrite_output)
        
        logger.info(f"State information for this attack will be stored in '{attack_state.persistable.attack_params.state_file}'.")

    start_dt = get_now()
    start_ts = get_time_string(start_dt)
    logger.info(f"Starting at {start_ts}")

    # Parameter validation, warnings, and errors
    if attack_state.persistable.attack_params.using_cpu():
        using_ok_format = False
        if attack_state.persistable.attack_params.model_data_format_handling == ModelDataFormatHandling.DEFAULT:
            using_ok_format = True
        if attack_state.persistable.attack_params.model_data_format_handling == ModelDataFormatHandling.FORCE_BFLOAT16:
            using_ok_format = True
        if attack_state.persistable.attack_params.model_data_format_handling == ModelDataFormatHandling.FORCE_FLOAT32:
            using_ok_format = True
        if using_ok_format:
            logger.warning(f"You are using a CPU device for processing. Depending on your hardware, the default model data type may cause degraded performance. If you encounter poor performance, try specifying --model-data-type float32 when launching Broken Hill, as long as your device has sufficient system RAM. Consult the documentation for further details.")
        else:
            logger.warning(f"You are using a CPU device for processing, but Broken Hill is configured to use an unsupported PyTorch dtype when loading the model. In particular, if 'float16' is specified, it will also greatly increase runtimes. If you encounter unusual behaviour, such as iteration times of 10 hours, or incorrect output from the model, try specifying --model-data-type default. Consult the documentation for further details.")
    non_cuda_devices = attack_state.persistable.attack_params.get_non_cuda_devices()
    if len(non_cuda_devices) > 0:
        if attack_state.persistable.attack_params.model_data_format_handling == ModelDataFormatHandling.FORCE_FLOAT16:
            logger.warning(f"Using the following device(s) with the 'float16' model data format is not recommended: {non_cuda_devices}. Using this format on non-CUDA devices will cause Broken Hill to run extremely slowly. Expect performance about 100 times slower than using 'float16' on CUDA hardware, for example, and about 10 times slower than using 'float32' on CPU hardware.")

    ietf_tag_names = None
    ietf_tag_data = None
    try:        
        ietf_tag_names, ietf_tag_data = attack_state.persistable.language_manager.get_ietf_tags()
    except Exception as e:
        logger.critical(f"Could not load the human language data bundled with Broken Hill: {e}\n{traceback.format_exc()}")
        sys.exit(1)
    
    if attack_state.persistable.attack_params.operating_mode == BrokenHillMode.LIST_IETF_TAGS:
        ietf_message = "Supported IETF language tags in this version:"
        for i in range(0, len(ietf_tag_names)):
            ietf_message = f"{ietf_message}\n{ietf_tag_names[i]}\t{ietf_tag_data[ietf_tag_names[i]]}"
        logger.info(ietf_message)
        sys.exit(0)

    attack_state.persistable.overall_result_data.start_date_time = start_ts    
    attack_state.persistable.performance_data.collect_torch_stats(attack_state, is_key_snapshot_event = True, location_description = "before loading model and tokenizer")
    
    try:
        attack_state.load_model()
        
        if attack_state.model_type_name == "MosaicGPT":
            logger.warning("This model is of type MosaicGPT. At the time this version of Broken Hill was made, MosaicGPT did not support the 'inputs_embeds' keyword when calling the forward method. If that is still the case when you are reading this message, Broken Hill will likely crash during the GCG attack.")

        if attack_state.model_type_name == "GPTNeoXForCausalLM":
            warn_about_bad_gpt_neox_weight_type = True
            if attack_state.persistable.attack_params.model_data_format_handling == ModelDataFormatHandling.DEFAULT:
                warn_about_bad_gpt_neox_weight_type = False
            # still need to validate that AUTO will actually work
            if attack_state.persistable.attack_params.model_data_format_handling == ModelDataFormatHandling.AUTO:
                warn_about_bad_gpt_neox_weight_type = False
            if attack_state.persistable.attack_params.model_data_format_handling == ModelDataFormatHandling.FORCE_FLOAT32:
                warn_about_bad_gpt_neox_weight_type = False
            logger.warning("This model is of type GPTNeoXForCausalLM. At the time this version of Broken Hill was made, GPT-NeoX did not perform correctly in PyTorch/Transformers when loaded with weights in float16 format, possibly any other dtype besides float32. If you encounter very long processing times or incorrect output (such as the LLM responding with only the <|endoftext|> token), try using one of the following options instead of your current --model-data-type selection:\n--model-data-type default\n--model-data-type auto\n--model-data-type float32")

        # only build the token allowlist and denylist if this is a new run.
        # If the state has been loaded, the lists are already populated.
        # So is the tensor form of the list.
        if not attack_params.load_state_from_file:
            attack_state.build_token_allow_and_denylists()
        
        original_model_size = 0

        if attack_state.persistable.attack_params.display_model_size:
            # Only perform this work if the results will actually be logged, to avoid unnecessary performance impact
            # Assume the same comment for all instances of this pattern
            if attack_state.log_manager.get_lowest_log_level() <= logging.DEBUG:
                logger.debug(f"Determining model size.")
            original_model_size = get_model_size(attack_state.model)
            logger.info(f"Model size: {original_model_size}")

        # This code still doesn't do anything useful, so don't get your hopes up!
        attack_state.apply_model_quantization()
        
        attack_state.apply_model_dtype_conversion()

        # if attack_state.persistable.attack_params.conversion_dtype:
            # #logger.debug(f"converting model dtype to {attack_state.persistable.attack_params.conversion_dtype}.")
            # attack_state.model = attack_state.model.to(attack_state.persistable.attack_params.conversion_dtype)

        if attack_state.persistable.attack_params.quantization_dtype or attack_state.persistable.attack_params.enable_static_quantization or attack_state.persistable.attack_params.conversion_dtype:
            logger.warning(f"You've enabled quantization and/or type conversion, which are unlikely to work for the foreseeable future due to PyTorch limitations. Please see the comments in the source code for Broken Hill.")
            if attack_state.persistable.attack_params.display_model_size:
                quantized_model_size = get_model_size(attack_state.model)
                size_factor = float(quantized_model_size) / float(original_model_size) * 100.0
                size_factor_formatted = f"{size_factor:.2f}%"
                logger.info(f"Model size after reduction: {quantized_model_size} ({size_factor_formatted} of original size)")
        
        register_custom_conversation_templates(attack_state.persistable.attack_params)

        attack_state.load_conversation_template()

        attack_state.ignite_trash_fire_token_treasury()
        
        # If the state was loaded from a file, the initial adversarial content information should already be populated.
        if not attack_params.load_state_from_file:
            attack_state.create_initial_adversarial_content()
        
            attack_state.check_for_adversarial_content_token_problems()
        
            logger.info(f"Initial adversarial content: {attack_state.persistable.initial_adversarial_content.get_full_description()}")

            attack_state.persistable.current_adversarial_content = attack_state.persistable.initial_adversarial_content.copy()

        attack_state.persistable.performance_data.collect_torch_stats(attack_state, location_description = "before creating adversarial content manager")
        if attack_state.log_manager.get_lowest_log_level() <= logging.DEBUG:
            logger.debug(f"creating adversarial content manager.")        
        attack_state.adversarial_content_manager = AdversarialContentManager(attack_state = attack_state, 
            conv_template = attack_state.conversation_template, 
            #adversarial_content = attack_state.persistable.initial_adversarial_content.copy(),
            adversarial_content = attack_state.persistable.current_adversarial_content.copy(),
            trash_fire_tokens = attack_state.trash_fire_token_treasury)
            
        attack_state.persistable.performance_data.collect_torch_stats(attack_state, location_description = "after creating adversarial content manager")
         
        #import pdb; pdb.Pdb(nosigint=True).set_trace()

        attack_state.persistable.original_new_adversarial_value_candidate_count = attack_state.persistable.attack_params.new_adversarial_value_candidate_count
        attack_state.persistable.original_topk = attack_state.persistable.attack_params.topk

        # Keep this out until things like denied tokens are file paths instead of inline, to keep from infecting result files with bad words
        #attack_state.persistable.overall_result_data.attack_params = attack_state.persistable.attack_params

        attack_state.test_conversation_template()

        attack_state.perform_jailbreak_tests()
        
        if attack_state.persistable.attack_params.operating_mode == BrokenHillMode.GCG_ATTACK_SELF_TEST:
            logger.info(f"Broken Hill has completed all self-test operations and will now exit.")
            end_ts = get_time_string(get_now())            
            attack_state.persistable.overall_result_data.end_date_time = end_ts
            attack_state.write_output_files()
            sys.exit(0)
        
        attack_state.persistable.performance_data.collect_torch_stats(attack_state, location_description = "before creating embedding_matrix")
        
        # If loading the state from a file, reload the RNG states right before starting the loop, in case something during initialization has messed with them
        if attack_params.load_state_from_file:
            attack_state.restore_random_number_generator_states()
        
        logger.info(f"Starting main loop")

        while attack_state.persistable.main_loop_iteration_number < attack_state.persistable.attack_params.max_iterations:
            display_iteration_number = attack_state.persistable.main_loop_iteration_number + 1
            #attack_state.persistable.random_number_generator_states = attack_state.random_number_generators.get_current_states()
            is_success = False
            if user_aborted:
                break
            else:
                try:
                    attack_results_current_iteration = AttackResultInfoCollection()
                    attack_results_current_iteration.iteration_number = attack_state.persistable.main_loop_iteration_number + 1
                    iteration_start_dt = get_now()
                    current_ts = get_time_string(iteration_start_dt)
                    current_elapsed_string = get_elapsed_time_string(start_dt, iteration_start_dt)
                    logger.info(f"{current_ts} - Main loop iteration {display_iteration_number} of {attack_state.persistable.attack_params.max_iterations} - elapsed time {current_elapsed_string} - successful attack count: {attack_state.persistable.successful_attack_count}")
                    attack_state.persistable.overall_result_data.end_date_time = current_ts
                    attack_state.persistable.overall_result_data.elapsed_time_string = current_elapsed_string
    
                    attack_state.persistable.performance_data.collect_torch_stats(attack_state, is_key_snapshot_event = True, location_description = f"beginning of main loop iteration {display_iteration_number}")
                    
                    attack_data_previous_iteration = None
                    if attack_state.persistable.main_loop_iteration_number > 0:
                        #attack_data_previous_iteration = attack_data[len(attack_data) - 1]
                        attack_data_previous_iteration = attack_state.persistable.overall_result_data.attack_results[len(attack_state.persistable.overall_result_data.attack_results) - 1]
                    
                    attack_state.persistable.tested_adversarial_content.append_if_new(attack_state.persistable.current_adversarial_content)
                    
                    # TKTK: split the actual attack step out into a separate subclass of an attack class.
                    # Maybe TokenPermutationAttack => GreedyCoordinateGradientAttack?
                    
                    # if this is not the first iteration, and the user has enabled emulation of the original attack, encode the current string, then use those IDs for this round instead of persisting everything in token ID format
                    if attack_state.persistable.main_loop_iteration_number > 0:
                        if attack_state.persistable.attack_params.reencode_adversarial_content_every_iteration:
                            reencoded_token_ids = attack_state.tokenizer.encode(attack_state.persistable.current_adversarial_content.as_string)
                            attack_state.persistable.current_adversarial_content = AdversarialContent.from_token_ids(attack_state, attack_state.trash_fire_token_treasury, reencoded_token_ids)
                    
                    attack_state.adversarial_content_manager.adversarial_content = attack_state.persistable.current_adversarial_content
                                       
                    # Step 1. Encode user prompt (behavior + adv suffix) as tokens and return token ids.
                    attack_state.persistable.performance_data.collect_torch_stats(attack_state, location_description = f"main loop iteration {display_iteration_number} - before creating input_id_data")
                    if attack_state.log_manager.get_lowest_log_level() <= logging.DEBUG:
                        logger.debug(f"Calling get_input_ids with attack_state.persistable.current_adversarial_content = '{attack_state.persistable.current_adversarial_content.get_short_description()}'")
                    input_id_data = attack_state.adversarial_content_manager.get_prompt(adversarial_content = attack_state.persistable.current_adversarial_content, force_python_tokenizer = attack_state.persistable.attack_params.force_python_tokenizer)
                    attack_state.persistable.performance_data.collect_torch_stats(attack_state, location_description = f"main loop iteration {display_iteration_number} - after creating input_id_data")
                    
                    # Only perform this work if the results will actually be logged
                    decoded_loss_slice = None
                    if attack_state.log_manager.get_lowest_log_level() <= logging.DEBUG:
                        if attack_state.persistable.attack_params.generate_debug_logs_requiring_extra_tokenizer_calls:
                            decoded_input_tokens = get_decoded_tokens(attack_state, input_id_data.input_token_ids)
                            decoded_full_prompt_token_ids = get_decoded_tokens(attack_state, input_id_data.full_prompt_token_ids)
                            decoded_control_slice = get_decoded_tokens(attack_state, input_id_data.full_prompt_token_ids[input_id_data.slice_data.control])
                            decoded_target_slice = get_decoded_tokens(attack_state, input_id_data.full_prompt_token_ids[input_id_data.slice_data.target_output])
                            decoded_loss_slice = get_decoded_tokens(attack_state, input_id_data.full_prompt_token_ids[input_id_data.slice_data.loss])                    
                            logger.debug(f"decoded_input_tokens = '{decoded_input_tokens}'\n decoded_full_prompt_token_ids = '{decoded_full_prompt_token_ids}'\n decoded_control_slice = '{decoded_control_slice}'\n decoded_target_slice = '{decoded_target_slice}'\n decoded_loss_slice = '{decoded_loss_slice}'\n input_id_data.slice_data.control = '{input_id_data.slice_data.control}'\n input_id_data.slice_data.target_output = '{input_id_data.slice_data.target_output}'\n input_id_data.slice_data.loss = '{input_id_data.slice_data.loss}'\n input_id_data.input_token_ids = '{input_id_data.input_token_ids}'\n input_id_data.full_prompt_token_ids = '{input_id_data.full_prompt_token_ids}'")
                    
                    decoded_loss_slice_string = get_escaped_string(attack_state.tokenizer.decode(input_id_data.full_prompt_token_ids[input_id_data.slice_data.loss]))
                    
                    attack_state.persistable.performance_data.collect_torch_stats(attack_state, location_description = f"main loop iteration {display_iteration_number} - before creating input_ids")
                    if attack_state.log_manager.get_lowest_log_level() <= logging.DEBUG:
                        logger.debug(f"Converting input IDs to device")
                    input_ids = input_id_data.get_input_ids_as_tensor().to(attack_state.model_device)
                    if attack_state.log_manager.get_lowest_log_level() <= logging.DEBUG:
                        logger.debug(f"input_ids after conversion = '{input_ids}'")
                    attack_state.persistable.performance_data.collect_torch_stats(attack_state, location_description = f"main loop iteration {display_iteration_number} - after creating input_ids")
                    
                    input_id_data_gcg_ops = input_id_data
                    input_ids_gcg_ops = input_ids
                    
                    if attack_state.persistable.attack_params.ignore_prologue_during_gcg_operations:
                        conv_template_gcg_ops = attack_state.conversation_template.copy()
                        conv_template_gcg_ops.system_message=""
                        conv_template_gcg_ops.messages = []
                        adversarial_content_manager_gcg_ops = AdversarialContentManager(attack_state = attack_state, 
                            conv_template = conv_template_gcg_ops, 
                            adversarial_content = attack_state.persistable.current_adversarial_content.copy(),
                            trash_fire_tokens = attack_state.trash_fire_token_treasury)
                        input_id_data_gcg_ops = adversarial_content_manager_gcg_ops.get_prompt(adversarial_content = attack_state.persistable.current_adversarial_content, force_python_tokenizer = attack_state.persistable.attack_params.force_python_tokenizer)
                        input_ids_gcg_ops = input_id_data_gcg_ops.get_input_ids_as_tensor().to(attack_state.model_device)

                    best_new_adversarial_content = None                    
                    
                    # preserve the RNG states because the code in this section is likely to reset them a bunch of times
                    rng_states = attack_state.random_number_generators.get_current_states()
                    
                    # declare these here so they can be cleaned up later
                    coordinate_gradient = None
                    # during the first iteration, do not generate variations - test the value that was given                    
                    if attack_state.persistable.main_loop_iteration_number == 0:
                        logger.info(f"Testing initial adversarial value '{attack_state.persistable.current_adversarial_content.get_short_description()}'")
                    else:
                        # Step 2. Compute Coordinate Gradient
                        attack_state.persistable.performance_data.collect_torch_stats(attack_state, location_description = f"main loop iteration {display_iteration_number} - before creating coordinate gradient")
                        if attack_state.log_manager.get_lowest_log_level() <= logging.DEBUG:
                            logger.debug(f"Computing coordinate gradient")
                        try:
                            coordinate_gradient = token_gradients(attack_state,
                                input_ids_gcg_ops,
                                input_id_data_gcg_ops)
                        except GradientCreationException as e:
                            raise GradientCreationException(f"Attempting to generate a coordinate gradient failed: {e}. Please contact a developer with steps to reproduce this issue if it has not already been reported.\n{traceback.format_exc()}")
                        if attack_state.log_manager.get_lowest_log_level() <= logging.DEBUG:
                            logger.debug(f"coordinate_gradient.shape[0] = {coordinate_gradient.shape[0]}")
                        attack_state.persistable.performance_data.collect_torch_stats(attack_state, location_description = f"main loop iteration {display_iteration_number} - after creating coordinate gradient")

                        # if isinstance(random_generator_gradient, type(None)):
                            # random_generator_gradient = torch.Generator(device = coordinate_gradient.device).manual_seed(attack_state.persistable.attack_params.torch_manual_seed)

                        # Step 3. Sample a batch of new tokens based on the coordinate gradient.
                        # Notice that we only need the one that minimizes the loss.

                        with torch.no_grad():                            
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
                                    
                                    if attack_state.log_manager.get_lowest_log_level() <= logging.DEBUG:
                                        logger.debug(f"Generating new adversarial content candidates")
                                    new_adversarial_candidate_list = None
                                    
                                    try:
                                        new_adversarial_candidate_list = get_adversarial_content_candidates(attack_state, 
                                                       coordinate_gradient,
                                                       not_allowed_tokens = attack_state.get_token_denylist_as_cpu_tensor())
                                    except GradientSamplingException as e:
                                        raise GradientSamplingException(f"Attempting to generate a new set of candidate adversarial data failed: {e}. Please contact a developer with steps to reproduce this issue if it has not already been reported.\n{traceback.format_exc()}")
                                    except RuntimeError as e:
                                        raise GradientSamplingException(f"Attempting to generate a new set of candidate adversarial data failed with a low-level error: {e}. This is typically caused by excessive or conflicting candidate-filtering options. For example, the operator may have specified a regular expression filter that rejects long strings, but also specified a long initial adversarial value. This error is unrecoverable. If you believe the error was not due to excessive/conflicting filtering options, please submit an issue.\n{traceback.format_exc()}")
     
                                    attack_state.persistable.performance_data.collect_torch_stats(attack_state, location_description = f"main loop iteration {display_iteration_number} - before getting filtered candidates")
                                    if attack_state.log_manager.get_lowest_log_level() <= logging.DEBUG:
                                        logger.debug(f"new_adversarial_candidate_list: {new_adversarial_candidate_list.adversarial_content}")
                                    
                                    # Note: I'm leaving this explanation here for historical reference
                                    # Step 3.3 This step ensures all adversarial candidates have the same number of tokens. 
                                    # This step is necessary because tokenizers are not invertible so Encode(Decode(tokens)) may produce a different tokenization.
                                    # We ensure the number of token remains [constant -Ben] to prevent the memory keeps growing and run into OOM.
                                    if attack_state.log_manager.get_lowest_log_level() <= logging.DEBUG:
                                        logger.debug(f"Getting filtered adversarial content candidates")
                                    new_adversarial_candidate_list_filtered = get_filtered_cands(attack_state, new_adversarial_candidate_list, filter_cand = True)
                                    attack_state.persistable.performance_data.collect_torch_stats(attack_state, location_description = f"main loop iteration {display_iteration_number} - after getting filtered candidates")
                                    if len(new_adversarial_candidate_list_filtered.adversarial_content) > 0:
                                        got_candidate_list = True
                                    else:
                                        # try to find a way to increase the number of options available
                                        something_has_changed = False
                                        standard_explanation_intro = "The attack has failed to generate any adversarial values at this iteration that meet the specified filtering criteria and have not already been tested."
                                        standard_explanation_outro = "You can try specifying larger values for --max-batch-size-new-adversarial-tokens and/or --max-topk to avoid this error, or enabling --add-token-when-no-candidates-returned and/or --delete-token-when-no-candidates-returned if they are not already enabled."
                                        
                                        if attack_state.persistable.attack_params.add_token_when_no_candidates_returned:
                                            token_count_limited = True
                                            if isinstance(attack_state.persistable.attack_params.candidate_filter_tokens_max, type(None)):
                                                token_count_limited = False
                                            if token_count_limited:
                                                if len(attack_state.persistable.current_adversarial_content.token_ids) < attack_state.persistable.attack_params.candidate_filter_tokens_max:
                                                    token_count_limited = False
                                            current_short_description = attack_state.persistable.current_adversarial_content.get_short_description()
                                            if token_count_limited:
                                                logger.warning(f"{standard_explanation_intro} The option to add an additional token is enabled, but the current adversarial content {current_short_description} is already at the limit of {attack_state.persistable.attack_params.candidate_filter_tokens_max} tokens.")
                                            else:
                                                attack_state.persistable.current_adversarial_content.duplicate_random_token(numpy_random_generator, attack_state.tokenizer)
                                                new_short_description = attack_state.persistable.current_adversarial_content.get_short_description()
                                                something_has_changed = True
                                                logger.info(f"{standard_explanation_intro} Because the option to add an additional token is enabled, the current adversarial content has been modified from {current_short_description} to {new_short_description}.")
                                        else:
                                            if attack_state.log_manager.get_lowest_log_level() <= logging.DEBUG:
                                                logger.debug(f"The option to add an additional token is disabled.")
                                        
                                        if not something_has_changed:
                                            if attack_state.persistable.attack_params.delete_token_when_no_candidates_returned:
                                                token_count_limited = True
                                                minimum_token_count = 1
                                                if isinstance(attack_state.persistable.attack_params.candidate_filter_tokens_min, type(None)):
                                                    token_count_limited = False
                                                else:
                                                    if attack_state.persistable.attack_params.candidate_filter_tokens_min > 1:
                                                        minimum_token_count = attack_state.persistable.attack_params.candidate_filter_tokens_min
                                                    if len(attack_state.persistable.current_adversarial_content.token_ids) > attack_state.persistable.attack_params.candidate_filter_tokens_min:
                                                        token_count_limited = False
                                                if not token_count_limited:
                                                    if len(attack_state.persistable.current_adversarial_content.token_ids) < 2:
                                                        token_count_limited = True
                                                current_short_description = attack_state.persistable.current_adversarial_content.get_short_description()
                                                if token_count_limited:
                                                    logger.warning(f"{standard_explanation_intro} The option to delete a random token is enabled, but the current adversarial content {current_short_description} is already at the minimum of {minimum_token_count} token(s).")
                                                else:
                                                    attack_state.persistable.current_adversarial_content.delete_random_token(numpy_random_generator, attack_state.tokenizer)
                                                    new_short_description = attack_state.persistable.current_adversarial_content.get_short_description()
                                                    something_has_changed = True
                                                    logger.info(f"{standard_explanation_intro} Because the option to delete a random token is enabled, the current adversarial content has been modified from {current_short_description} to {new_short_description}.")
                                            else:
                                                if attack_state.log_manager.get_lowest_log_level() <= logging.DEBUG:
                                                    logger.debug(f"The option to delete a random token is disabled.")
                                        
                                        if not something_has_changed:
                                            new_new_adversarial_value_candidate_count = attack_state.persistable.attack_params.new_adversarial_value_candidate_count + attack_state.persistable.original_new_adversarial_value_candidate_count
                                            increase_new_adversarial_value_candidate_count = True
                                            if not isinstance(attack_state.persistable.attack_params.max_new_adversarial_value_candidate_count, type(None)):
                                                if new_new_adversarial_value_candidate_count > attack_state.persistable.attack_params.max_new_adversarial_value_candidate_count:
                                                    new_new_adversarial_value_candidate_count = attack_state.persistable.attack_params.max_new_adversarial_value_candidate_count
                                                    if new_new_adversarial_value_candidate_count <= attack_state.persistable.attack_params.new_adversarial_value_candidate_count:
                                                        increase_new_adversarial_value_candidate_count = False
                                                    else:
                                                        if attack_state.log_manager.get_lowest_log_level() <= logging.DEBUG:
                                                            logger.debug(f"new_new_adversarial_value_candidate_count > attack_state.persistable.attack_params.new_adversarial_value_candidate_count.")
                                                else:
                                                    if attack_state.log_manager.get_lowest_log_level() <= logging.DEBUG:
                                                        logger.debug(f"new_new_adversarial_value_candidate_count <= attack_state.persistable.attack_params.max_new_adversarial_value_candidate_count.")
                                            else:
                                                if attack_state.log_manager.get_lowest_log_level() <= logging.DEBUG:
                                                    logger.debug(f"attack_state.persistable.attack_params.max_new_adversarial_value_candidate_count is None.")
                                            if increase_new_adversarial_value_candidate_count:
                                                logger.warning(f"{standard_explanation_intro}  This may be due to excessive post-generation filtering options. The --batch-size-new-adversarial-tokens value is being increased from {attack_state.persistable.attack_params.new_adversarial_value_candidate_count} to {new_new_adversarial_value_candidate_count} to increase the number of candidate values. {standard_explanation_outro}")
                                                attack_state.persistable.attack_params.new_adversarial_value_candidate_count = new_new_adversarial_value_candidate_count
                                                something_has_changed = True
                                            else:
                                                if attack_state.log_manager.get_lowest_log_level() <= logging.DEBUG:
                                                    logger.debug(f"Not increasing the --batch-size-new-adversarial-tokens value.")
                                        
                                        if not something_has_changed:
                                            new_topk = attack_state.persistable.attack_params.topk + attack_state.persistable.original_topk
                                            increase_topk = True
                                            if not isinstance(attack_state.persistable.attack_params.max_topk, type(None)):
                                                if new_topk > attack_state.persistable.attack_params.max_topk:
                                                    new_topk = attack_state.persistable.attack_params.max_topk
                                                    if new_topk <= attack_state.persistable.attack_params.topk:
                                                        increase_topk = False
                                                    else:
                                                        if attack_state.log_manager.get_lowest_log_level() <= logging.DEBUG:
                                                            logger.debug(f"new_topk > attack_state.persistable.attack_params.topk.")
                                                else:
                                                    if attack_state.log_manager.get_lowest_log_level() <= logging.DEBUG:
                                                        logger.debug(f"new_topk <= attack_state.persistable.attack_params.max_topk.")
                                            else:
                                                if attack_state.log_manager.get_lowest_log_level() <= logging.DEBUG:
                                                    logger.debug(f"attack_state.persistable.attack_params.max_topk is None.")
                                            if increase_topk:
                                                logger.warning(f"{standard_explanation_intro}  This may be due to excessive post-generation filtering options. The --topk value is being increased from {attack_state.persistable.attack_params.topk} to {new_topk} to increase the number of candidate values. {standard_explanation_outro}")
                                                attack_state.persistable.attack_params.topk = new_topk
                                                something_has_changed = True
                                            else:
                                                if attack_state.log_manager.get_lowest_log_level() <= logging.DEBUG:
                                                    logger.debug(f"Not increasing the --topk value.")
                                        
                                        if not something_has_changed:
                                            raise GradientSamplingException(f"{standard_explanation_intro} This may be due to excessive post-generation filtering options. Because the 'topk' value has already reached or exceeded the specified maximum ({attack_state.persistable.attack_params.max_topk}), and no other options for increasing the number of potential candidates is possible in the current configuration, Broken Hill will now exit. {standard_explanation_outro}\n{traceback.format_exc()}")
                                                
                                attack_state.persistable.performance_data.collect_torch_stats(attack_state, location_description = f"main loop iteration {display_iteration_number} - after getting finalized filtered candidates")
                                if attack_state.log_manager.get_lowest_log_level() <= logging.DEBUG:
                                    logger.debug(f"new_adversarial_candidate_list_filtered: '{new_adversarial_candidate_list_filtered.to_dict()}'")
                                
                                # Step 3.4 Compute loss on these candidates and take the argmin.
                                if attack_state.log_manager.get_lowest_log_level() <= logging.DEBUG:
                                    logger.debug(f"Getting logits")
                                logits, ids = get_logits(attack_state,
                                    input_ids = input_ids_gcg_ops,
                                    adversarial_content = attack_state.persistable.current_adversarial_content, 
                                    adversarial_candidate_list = new_adversarial_candidate_list_filtered, 
                                    return_ids = True)
                                attack_state.persistable.performance_data.collect_torch_stats(attack_state, location_description = f"main loop iteration {display_iteration_number} - after getting logits")

                                if attack_state.log_manager.get_lowest_log_level() <= logging.DEBUG:
                                    logger.debug(f"Calculating target loss")
                                losses = target_loss(attack_state, logits, ids, input_id_data_gcg_ops)
                                attack_state.persistable.performance_data.collect_torch_stats(attack_state, location_description = f"main loop iteration {display_iteration_number} - after getting loss values")
                                # get rid of logits and ids immediately to save device memory, as it's no longer needed after the previous operation
                                # This frees about 1 GiB of device memory for a 500M model on CPU or CUDA
                                del logits
                                del ids
                                gc.collect()
                                attack_state.persistable.performance_data.collect_torch_stats(attack_state, location_description = f"main loop iteration {display_iteration_number} - after deleting logits and ids and running gc.collect")
                                if attack_state.log_manager.get_lowest_log_level() <= logging.DEBUG:
                                    logger.debug(f"losses = {losses}")
                                    logger.debug(f"Getting losses argmin")
                                best_new_adversarial_content_id = losses.argmin()
                                attack_state.persistable.performance_data.collect_torch_stats(attack_state, location_description = f"main loop iteration {display_iteration_number} - after getting best new adversarial content ID")
                                if attack_state.log_manager.get_lowest_log_level() <= logging.DEBUG:
                                    logger.debug(f"best_new_adversarial_content_id = {best_new_adversarial_content_id}")

                                if attack_state.log_manager.get_lowest_log_level() <= logging.DEBUG:
                                    logger.debug(f"Setting best new adversarial content")
                                best_new_adversarial_content = new_adversarial_candidate_list_filtered.adversarial_content[best_new_adversarial_content_id].copy()
                                attack_state.persistable.performance_data.collect_torch_stats(attack_state, location_description = f"main loop iteration {display_iteration_number} - after getting best new adversarial content")

                                if attack_state.log_manager.get_lowest_log_level() <= logging.DEBUG:
                                    logger.debug(f"Getting current loss")
                                current_loss = losses[best_new_adversarial_content_id]
                                del losses
                                gc.collect()
                                attack_state.persistable.performance_data.collect_torch_stats(attack_state, location_description = f"main loop iteration {display_iteration_number} - after deleting losses and running gc.collect")
                                current_loss_as_float = None
                                try:
                                    current_loss_as_float = float(f"{current_loss.detach().to(torch.float32).cpu().numpy()}")
                                except Exception as e:
                                    logger.error(f"Could not convert the current loss value '{current_loss}' to a floating-point number: {e}\n{traceback.format_exc()}\nThe value 100.0 will be used instead.")
                                    current_loss_as_float = 100.0
                                best_new_adversarial_content.original_loss = current_loss_as_float
                                
                                if isinstance(attack_state.persistable.attack_params.required_loss_threshold, type(None)) or attack_state.persistable.main_loop_iteration_number == 0:
                                    candidate_list_meets_loss_threshold = True
                                else:
                                    if attack_data_previous_iteration is None:
                                        candidate_list_meets_loss_threshold = True
                                    else:
                                        if isinstance(attack_data_previous_iteration.loss, type(None)):
                                            candidate_list_meets_loss_threshold = True
                                        if not candidate_list_meets_loss_threshold:
                                            if best_new_adversarial_content.original_loss <= (attack_data_previous_iteration.loss + attack_state.persistable.attack_params.required_loss_threshold):
                                                candidate_list_meets_loss_threshold = True
                                if not candidate_list_meets_loss_threshold:
                                    num_iterations_without_acceptable_loss += 1
                                    best_failed_attempts.append_if_new(best_new_adversarial_content)
                                    loss_attempt_stats_message = f"{num_iterations_without_acceptable_loss} unsuccessful attempt(s) to generate a list of random candidates that has at least one candidate with a loss lower than {(attack_data_previous_iteration.loss + attack_state.persistable.attack_params.required_loss_threshold)}"
                                    if not isinstance(attack_state.persistable.attack_params.required_loss_threshold, type(None)):
                                        if attack_state.persistable.attack_params.required_loss_threshold != 0.0:
                                            loss_attempt_stats_message += f" (previous loss of {attack_data_previous_iteration.loss} plus the specified threshold value {attack_state.persistable.attack_params.required_loss_threshold})" 
                                    loss_attempt_stats_message += f". Best value during this attempt was {current_loss_as_float}."
                                    logger.warning(loss_attempt_stats_message)
                                    if not isinstance(attack_state.persistable.attack_params.loss_threshold_max_attempts, type(None)):
                                        if num_iterations_without_acceptable_loss >= attack_state.persistable.attack_params.loss_threshold_max_attempts:
                                            loss_attempt_result_message = f"{num_iterations_without_acceptable_loss} unsuccessful attempt(s) has reached the limit of {attack_state.persistable.attack_params.loss_threshold_max_attempts} attempts."
                                            if attack_state.persistable.attack_params.exit_on_loss_threshold_failure:
                                                loss_attempt_result_message += " Broken Hill has been configured to exit when this condition occurs."
                                                raise LossThresholdException(loss_attempt_result_message)
                                            else:
                                                best_new_adversarial_content = best_failed_attempts.get_content_with_lowest_loss()
                                                candidate_list_meets_loss_threshold = True
                                                loss_attempt_result_message += f" Broken Hill has been configured to use the adversarial content with the lowest loss discovered during this iteration when this condition occurs. Out of {len(best_failed_attempts.adversarial_content)} unique set(s) of tokens discovered during this iteration, the lowest loss value was {best_new_adversarial_content.original_loss} versus the previous loss of {attack_data_previous_iteration.loss}"
                                                if not isinstance(attack_state.persistable.attack_params.required_loss_threshold, type(None)):
                                                    if attack_state.persistable.attack_params.required_loss_threshold != 0.0:
                                                        loss_attempt_result_message += f" and threshold {attack_state.persistable.attack_params.required_loss_threshold}"
                                                loss_attempt_result_message += ". The adversarial content with that loss value will be used."
                                                logger.warning(loss_attempt_result_message)
                                                
                                        
                            
                            # END: wrap in loss threshold check 

                            # Update the running attack_state.persistable.current_adversarial_content with the best candidate
                            attack_state.persistable.performance_data.collect_torch_stats(attack_state, location_description = f"main loop iteration {display_iteration_number} - before updating adversarial value")
                            
                            logger.info(f"Updating adversarial value to the best value out of the new permutation list and testing it.\nWas: {attack_state.persistable.current_adversarial_content.get_short_description()} ({len(attack_state.persistable.current_adversarial_content.token_ids)} tokens)\nNow: {best_new_adversarial_content.get_short_description()} ({len(best_new_adversarial_content.token_ids)} tokens)")
                            attack_state.persistable.current_adversarial_content = best_new_adversarial_content
                            logger.info(f"Loss value for the new adversarial value in relation to '{decoded_loss_slice_string}'\nWas: {attack_data_previous_iteration.loss}\nNow: {attack_state.persistable.current_adversarial_content.original_loss}")
                            if attack_state.log_manager.get_lowest_log_level() <= logging.DEBUG:
                                if attack_state.persistable.attack_params.generate_debug_logs_requiring_extra_tokenizer_calls:
                                    # this is in this block because the variable will be None if the previous check of the same type was bypassed
                                    logger.debug(f"decoded_loss_slice = '{decoded_loss_slice}'")
                                logger.debug(f"input_id_data.full_prompt_token_ids[input_id_data.slice_data.loss] = '{input_id_data.full_prompt_token_ids[input_id_data.slice_data.loss]}'")
                                logger.debug(f"input_id_data_gcg_ops.full_prompt_token_ids[input_id_data_gcg_ops.slice_data.loss] = '{input_id_data_gcg_ops.full_prompt_token_ids[input_id_data_gcg_ops.slice_data.loss]}'")
                        
                            #attack_results_current_iteration.loss = current_loss_as_float
                            attack_results_current_iteration.loss = attack_state.persistable.current_adversarial_content.original_loss

                    attack_state.persistable.performance_data.collect_torch_stats(attack_state, location_description = f"main loop iteration {display_iteration_number} - before creating best_new_adversarial_content_input_token_id_data")
                    best_new_adversarial_content_input_token_id_data = attack_state.adversarial_content_manager.get_prompt(adversarial_content = attack_state.persistable.current_adversarial_content, force_python_tokenizer = attack_state.persistable.attack_params.force_python_tokenizer)
                    attack_state.persistable.performance_data.collect_torch_stats(attack_state, location_description = f"main loop iteration {display_iteration_number} - after creating best_new_adversarial_content_input_token_id_data")

                    # preserve the RNG states because the code in this section is likely to reset them a bunch of times
                    # they're preserved twice because this is inside a block that may not occur
                    # but if they're altered, it will be in the next section
                    rng_states = attack_state.random_number_generators.get_current_states()

                    attack_results_current_iteration.adversarial_content = attack_state.persistable.current_adversarial_content.copy()
                    
                    # BEGIN: do for every random seed
                    prng_seed_index = -1
                    for randomized_test_number in range(0, attack_state.persistable.attack_params.random_seed_comparisons + 1):
                        prng_seed_index += 1
                        attack_data_current_iteration = AttackResultInfo()                        
                        attack_data_current_iteration.numpy_random_seed = attack_state.persistable.attack_params.numpy_random_seed
                        attack_data_current_iteration.torch_manual_seed = attack_state.persistable.attack_params.torch_manual_seed
                        attack_data_current_iteration.torch_cuda_manual_seed_all = attack_state.persistable.attack_params.torch_cuda_manual_seed_all
                        current_temperature = attack_state.persistable.attack_params.model_temperature_range_begin
                        # For the first run, leave the model in its default do_sample configuration
                        do_sample = False
                        if randomized_test_number == 0:
                            attack_data_current_iteration.is_canonical_result = True
                            attack_results_current_iteration.set_values(attack_state, best_new_adversarial_content_input_token_id_data.full_prompt_token_ids, best_new_adversarial_content_input_token_id_data.get_user_input_token_ids())
                        else:
                            if randomized_test_number == attack_state.persistable.attack_params.random_seed_comparisons or attack_state.persistable.attack_params.model_temperature_range_begin == attack_state.persistable.attack_params.model_temperature_range_end:
                                current_temperature = attack_state.persistable.attack_params.model_temperature_range_end
                            else:
                                current_temperature = attack_state.persistable.attack_params.model_temperature_range_begin + (((attack_state.persistable.attack_params.model_temperature_range_end - attack_state.persistable.attack_params.model_temperature_range_begin) / float(attack_state.persistable.attack_params.random_seed_comparisons) * randomized_test_number))
                            # For all other runs, enable do_sample to randomize results
                            do_sample = True
                            # Pick the next random seed that's not equivalent to any of the initial values
                            got_random_seed = False                            
                            while not got_random_seed:
                                random_seed = attack_state.random_seed_values[prng_seed_index]
                                seed_already_used = False
                                if random_seed == attack_state.persistable.attack_params.numpy_random_seed:
                                    seed_already_used = True
                                if random_seed == attack_state.persistable.attack_params.torch_manual_seed:
                                    seed_already_used = True
                                if random_seed == attack_state.persistable.attack_params.torch_cuda_manual_seed_all:
                                    seed_already_used = True
                                if seed_already_used:
                                    prng_seed_index += 1
                                    len_random_seed_values = len(attack_state.random_seed_values)
                                    if prng_seed_index > len_random_seed_values:
                                        raise MyCurrentMentalImageOfALargeValueShouldBeEnoughForAnyoneException(f"Exceeded the number of random seeds available({len_random_seed_values}).")
                                else:
                                    got_random_seed = True
                            if attack_state.log_manager.get_lowest_log_level() <= logging.DEBUG:
                                logger.debug(f"Temporarily setting all random seeds to {random_seed} to compare results")
                            numpy.random.seed(random_seed)
                            torch.manual_seed(random_seed)
                            if attack_state.persistable.attack_params.using_cuda():
                                torch.cuda.manual_seed_all(random_seed)
                            attack_data_current_iteration.numpy_random_seed = random_seed
                            attack_data_current_iteration.torch_manual_seed = random_seed
                            attack_data_current_iteration.torch_cuda_manual_seed_all = random_seed
                        attack_data_current_iteration.temperature = current_temperature
                    
                        attack_state.persistable.performance_data.collect_torch_stats(attack_state, location_description = f"main loop iteration {display_iteration_number} - before checking for jailbreak success")
                        if attack_state.log_manager.get_lowest_log_level() <= logging.DEBUG:
                            logger.debug(f"Checking for successful jailbreak")
                        is_success, jailbreak_check_data, jailbreak_check_generation_results = attack_state.check_for_attack_success(best_new_adversarial_content_input_token_id_data,
                                                current_temperature,
                                                do_sample = do_sample)            
                        attack_state.persistable.performance_data.collect_torch_stats(attack_state, location_description = f"main loop iteration {display_iteration_number} - after checking for jailbreak success")
                        if is_success:
                            if attack_data_current_iteration.is_canonical_result:
                                attack_data_current_iteration.canonical_llm_jailbroken = True
                            attack_data_current_iteration.jailbreak_detected = True
                            attack_results_current_iteration.jailbreak_detection_count += 1

                        if attack_state.log_manager.get_lowest_log_level() <= logging.DEBUG:
                            logger.debug(f"Passed:{is_success}\nCurrent best new adversarial content: '{attack_state.persistable.current_adversarial_content.get_short_description()}'")
                        
                        full_output_dataset_name = "full_output"
                        
                        jailbreak_check_dataset_name = "jailbreak_check"
                        if attack_state.persistable.attack_params.display_full_failed_output:
                            jailbreak_check_dataset_name = full_output_dataset_name
                        
                        attack_data_current_iteration.result_data_sets[jailbreak_check_dataset_name] = jailbreak_check_data
                        
                        # only generate full output if it hasn't already just been generated
                        if not attack_state.persistable.attack_params.display_full_failed_output and is_success:
                            full_output_data = AttackResultInfoData()
                            # Note: set random seeds for randomized variations where do_sample is True so that full output begins with identical output to shorter version
                            if do_sample:
                                if attack_state.log_manager.get_lowest_log_level() <= logging.DEBUG:
                                    logger.debug(f"Temporarily setting all random seeds to {random_seed} to generate full output")
                                numpy.random.seed(random_seed)
                                torch.manual_seed(random_seed)
                                if attack_state.persistable.attack_params.using_cuda():
                                    torch.cuda.manual_seed_all(random_seed)
                            generation_results = attack_state.generate(best_new_adversarial_content_input_token_id_data, current_temperature, do_sample = do_sample, generate_full_output = True)
                            full_output_data.set_values(attack_state, generation_results.max_new_tokens, generation_results.output_token_ids, generation_results.output_token_ids_output_only)
                            
                            attack_data_current_iteration.result_data_sets[full_output_dataset_name] = full_output_data
                        
                        
                        attack_results_current_iteration.results.append(attack_data_current_iteration)
                        
                        # END: do for every random seed
                    
                    # restore the RNG states
                    attack_state.random_number_generators.set_states(rng_states)
                    
                    attack_results_current_iteration.update_unique_output_values()
                    iteration_status_message = f"Status:\n"
                    iteration_status_message = f"{iteration_status_message}Current input string:\n---\n{attack_results_current_iteration.decoded_user_input_string}\n---\n"
                    iteration_status_message = f"{iteration_status_message}Successful jailbreak attempts detected: {attack_results_current_iteration.jailbreak_detection_count}."
                    if attack_results_current_iteration.canonical_llm_jailbroken:
                        iteration_status_message = f"{iteration_status_message} Canonical LLM instance was jailbroken."
                    else:
                        iteration_status_message= f"{iteration_status_message} Canonical LLM instance was not jailbroken."
                    iteration_status_message = f"{iteration_status_message}\n{attack_results_current_iteration.unique_result_count} unique output(s) generated during testing:\n"
                    for uov_string in attack_results_current_iteration.unique_results.keys():
                        uov_count = attack_results_current_iteration.unique_results[uov_string]
                        iteration_status_message = f"{iteration_status_message}--- {uov_count} occurrence(s): ---\n" 
                        iteration_status_message = f"{iteration_status_message}{uov_string}\n"
                    iteration_status_message = f"{iteration_status_message}---\n" 
                    iteration_status_message = f"{iteration_status_message}Current best new adversarial content: {attack_state.persistable.current_adversarial_content.get_short_description()}"               
                    logger.info(iteration_status_message)
                    
                    # TKTK: maybe make this a threshold
                    if attack_results_current_iteration.jailbreak_detection_count > 0:
                        attack_state.persistable.successful_attack_count += 1
                    
                    iteration_end_dt = get_now()
                    iteration_elapsed = iteration_end_dt - iteration_start_dt
                    attack_results_current_iteration.total_processing_time_seconds = iteration_elapsed.total_seconds()
                    
                    attack_state.persistable.overall_result_data.attack_results.append(attack_results_current_iteration)
                    if attack_state.persistable.attack_params.write_output_every_iteration:
                        attack_state.write_output_files()
                    
                    rollback_triggered = False
                    
                    if attack_state.persistable.main_loop_iteration_number > 0:
                        rollback_message = ""
                        if attack_state.persistable.attack_params.rollback_on_loss_increase:
                            if (attack_results_current_iteration.loss - attack_state.persistable.attack_params.rollback_on_loss_threshold) > attack_state.persistable.best_loss_value:
                                if attack_state.persistable.attack_params.rollback_on_loss_threshold == 0.0:
                                    rollback_message += f"The loss value for the current iteration ({attack_results_current_iteration.loss}) is greater than the best value achieved during this run ({attack_state.persistable.best_loss_value}). "
                                else:
                                    rollback_message += f"The loss value for the current iteration ({attack_results_current_iteration.loss}) is greater than the allowed delta of {attack_state.persistable.attack_params.rollback_on_loss_threshold} from the best value achieved during this run ({attack_state.persistable.best_loss_value}). "
                                rollback_triggered = True
                            else:
                                if attack_state.log_manager.get_lowest_log_level() <= logging.DEBUG:
                                    logger.debug(f"Rollback not triggered by current loss value {attack_results_current_iteration.loss} versus current best value {attack_state.persistable.best_loss_value} and threshold {attack_state.persistable.attack_params.rollback_on_loss_threshold}.")
                        if attack_state.persistable.attack_params.rollback_on_jailbreak_count_decrease:
                            if (attack_results_current_iteration.jailbreak_detection_count + attack_state.persistable.attack_params.rollback_on_jailbreak_count_threshold) < attack_state.persistable.best_jailbreak_count:
                                if attack_state.persistable.attack_params.rollback_on_jailbreak_count_threshold == 0:
                                    rollback_message += f"The jailbreak detection count for the current iteration ({attack_results_current_iteration.jailbreak_detection_count}) is less than for the best count achieved during this run ({attack_state.persistable.best_jailbreak_count}). "
                                else:
                                    rollback_message += f"The jailbreak detection count for the current iteration ({attack_results_current_iteration.jailbreak_detection_count}) is less than the allowed delta of {attack_state.persistable.attack_params.rollback_on_jailbreak_count_threshold} from the best count achieved during this run ({attack_state.persistable.best_jailbreak_count}). "
                                rollback_triggered = True
                            else:
                                if attack_state.log_manager.get_lowest_log_level() <= logging.DEBUG:
                                    logger.debug(f"Rollback not triggered by current jailbreak count {attack_results_current_iteration.jailbreak_detection_count} versus current best value {attack_state.persistable.best_jailbreak_count} and threshold {attack_state.persistable.attack_params.rollback_on_jailbreak_count_threshold}.")
                        # TKTK: if use of a threshold has allowed a score to drop below the last best value for x iterations, roll all the way back to the adversarial value that resulted in the current best value
                        # maybe use a tree model, with each branch from a node allowed to decrease 50% the amount of the previous branch, and too many failures to reach the value of the previous branch triggers a rollback to that branch
                        # That would allow some random exploration of various branches, at least allowing for the possibility of discovering a strong value within them, but never getting stuck for too long
                        if rollback_triggered:
                            #rollback_message += f"Rolling back to the last-known-good adversarial data {attack_state.persistable.last_known_good_adversarial_content.get_short_description()} for the next iteration instead of using this iteration's result {attack_state.persistable.current_adversarial_content.get_short_description()}."
                            rollback_message += f"Rolling back to the last-known-good adversarial data for the next iteration instead of using this iteration's result.\nThis iteration:  '{attack_state.persistable.current_adversarial_content.get_short_description()}'\nLast-known-good: {attack_state.persistable.last_known_good_adversarial_content.get_short_description()}."
                            logger.info(rollback_message)
                            # add the rejected result to the list of tested results to avoid getting stuck in a loop
                            attack_state.persistable.tested_adversarial_content.append_if_new(attack_state.persistable.current_adversarial_content)
                            # roll back
                            #adversarial_content = attack_state.persistable.last_known_good_adversarial_content.copy()
                            #attack_state.persistable.current_adversarial_content = adversarial_content
                            attack_state.persistable.current_adversarial_content = attack_state.persistable.last_known_good_adversarial_content.copy()
                            

                    # only update the "last-known-good" results if no rollback was triggered (for any reason)
                    # otherwise, if someone has multiple rollback options enabled, and only one of them is tripped, the other path will end up containing bad data
                    if not rollback_triggered:
                        rollback_notification_message = f"Updating last-known-good adversarial value from {attack_state.persistable.last_known_good_adversarial_content.get_short_description()} to {attack_state.persistable.current_adversarial_content.get_short_description()}."
                        attack_state.persistable.last_known_good_adversarial_content = attack_state.persistable.current_adversarial_content.copy()
                        # only update these if they're improvements - they should work as a high water mark to avoid gradually decreasing quality over time when the rollback thresholds are enabled
                        update_loss = False
                        
                        if isinstance(attack_state.persistable.best_loss_value, type(None)):
                            if not isinstance(attack_results_current_iteration.loss, type(None)):
                                update_loss = True
                        else:
                            if not isinstance(attack_results_current_iteration.loss, type(None)):
                                if attack_results_current_iteration.loss < attack_state.persistable.best_loss_value:
                                    update_loss = True
                        if update_loss:
                            rollback_notification_message += f" Updating best loss value from {attack_state.persistable.best_loss_value} to {attack_results_current_iteration.loss}."
                            attack_state.persistable.best_loss_value = attack_results_current_iteration.loss
                        
                        update_jailbreak_count = False
                        if isinstance(attack_state.persistable.best_jailbreak_count, type(None)):
                            if not isinstance(attack_results_current_iteration.jailbreak_detection_count, type(None)):
                                update_jailbreak_count = True
                        else:
                            if not isinstance(attack_results_current_iteration.jailbreak_detection_count, type(None)):
                                if attack_results_current_iteration.jailbreak_detection_count > attack_state.persistable.best_jailbreak_count:
                                    update_jailbreak_count = True
                        if update_jailbreak_count:
                            rollback_notification_message += f" Updating best jailbreak count from {attack_state.persistable.best_jailbreak_count} to {attack_results_current_iteration.jailbreak_detection_count}."
                            attack_state.persistable.best_jailbreak_count = attack_results_current_iteration.jailbreak_detection_count
                                            
                    if not attack_state.persistable.attack_params.preserve_gradient:
                        # (Optional) Clean up the cache.
                        if attack_state.log_manager.get_lowest_log_level() <= logging.DEBUG:
                            logger.debug(f"Deleting coordinate gradient")
                        attack_state.persistable.performance_data.collect_torch_stats(attack_state, location_description = f"main loop iteration {display_iteration_number} - before deleting coordinate gradient")
                        if coordinate_gradient is not None:
                            del coordinate_gradient
                        gc.collect()
                        attack_state.persistable.performance_data.collect_torch_stats(attack_state, location_description = f"main loop iteration {display_iteration_number} - after deleting coordinate gradient and running gc.collect")
                                    
                # Neither of the except KeyboardInterrupt blocks currently do anything because some inner code in another module is catching it first
                except KeyboardInterrupt:
                    #import pdb; pdb.Pdb(nosigint=True).post_mortem()
                    logger.warning(f"Exiting main loop early by request")
                    user_aborted = True
                
            if is_success and attack_state.persistable.attack_params.break_on_success:
                break
            attack_state.persistable.main_loop_iteration_number += 1
            attack_state.persistable.overall_result_data.completed_iterations = attack_state.persistable.main_loop_iteration_number
            if attack_state.persistable.attack_params.write_output_every_iteration:
                attack_state.write_persistent_state()

    except KeyboardInterrupt:
        #import pdb; pdb.Pdb(nosigint=True).post_mortem()
        logger.warning(f"Exiting early by request")
        user_aborted = True
    
    except torch.OutOfMemoryError as toome:
        logger.critical(f"Broken Hill ran out of memory on the specified PyTorch device. If you have not done so already, please consult the Broken Hill documentation regarding the sizes of models you can test given your device's memory. The list of command-line parameters contains several options you can use to reduce the amount of memory used during the attack as well. The exception details will be displayed below this message for troubleshooting purposes.\n{traceback.format_exc()}\n{general_error_guidance_message}")
        abnormal_termination = True

    except BrokenHillFileIOException as bfio:
        logger.critical(f"Broken Hill is unable to continue execution due to an error that occurred while reading or writing a file: {bfio}\n{traceback.format_exc()}\n{general_error_guidance_message}\n{bug_guidance_message}")
        abnormal_termination = True
        
    except JSONSerializationException as je:
        logger.critical(f"Broken Hill is unable to continue execution due to an error that occurred while performing JSON serialization or deserialation: {je}\n{traceback.format_exc()}\n{general_error_guidance_message}\n{bug_guidance_message}")
        abnormal_termination = True
        
    except BrokenHillValueException as bve:
        logger.critical(f"Broken Hill is unable to continue execution due to a value not meeting necessary criteria: {bve}\n{traceback.format_exc()}\n{general_error_guidance_message}\n{bug_guidance_message}")
        abnormal_termination = True
    
    except AttackInitializationException as aie:
        logger.critical(f"Broken Hill is unable to continue execution due to an error that occurred during the attack initialization phase: {aie}\n{bug_guidance_message}")
        abnormal_termination = True

    except (GradientCreationException, GradientSamplingException, MellowmaxException) as ge:
        logger.critical(f"Broken Hill is unable to continue execution due to an error that occurred while creating or sampling the gradient to create new adversarial content:\n{ge}\n{traceback.format_exc()}\n{general_error_guidance_message}\n{bug_guidance_message}")
        abnormal_termination = True
    
    except LossThresholdException as lte:
        logger.critical(f"Broken Hill is unable to continue execution due to contraints specified in the current configuration: {lte}\nSome potential options:\n* Begin a new attack with less-restrictive options.\n* Continue the attack using the saved state, but alter the configuration using additional command-line options to avoid this condition.\n* Begin a new attack or continue the current attack with the same configuration, but different random seed values.")
        abnormal_termination = True
        
    except PromptGenerationException as pge:
        logger.critical(f"Broken Hill is unable to continue execution due to an error that occurred while generating a prompt: {pge}\n{traceback.format_exc()}\n{bug_guidance_message}")
        abnormal_termination = True
        
    except MyCurrentMentalImageOfALargeValueShouldBeEnoughForAnyoneException as six_hundred_forty_kb:
        logger.critical(f"{six_hundred_forty_kb} If you requested that Broken Hill compare results against more than 16,381 randomized versions of the LLM, please specify a lower value for --random-seed-comparisons or add additional seed values to the hardcoded list in the get_random_seed_list_for_comparisons method. If you did not specify --random-seed-comparisons with a value greater than 16000, please notify the Broken Hill developers with steps to reproduce this condition.")
        abnormal_termination = True
        
    except TrashFireTokenException as tfte:
        logger.critical(f"Broken Hill is unable to continue execution due to an error that occurred while trying to bring law to the lawless frontier of LLM tokens: {tfte}\n{traceback.format_exc()}\n{bug_guidance_message}")
        abnormal_termination = True
        
    except (PaddingException, NullPaddingTokenException) as pe:
        logger.critical(f"Broken Hill is unable to continue execution due to an error that occurred while performing a padding operation: {pe}\n{traceback.format_exc()}\n{bug_guidance_message}")
        abnormal_termination = True
        
    except (SeparatorStyleConversionException, ConversationTemplateSerializationException) as cte:
        logger.critical(f"Broken Hill is unable to continue execution due to an error that occurred while performing an operation related to conversation templates: {cte}\n{traceback.format_exc()}\n{bug_guidance_message}")
        abnormal_termination = True
        
    except HumanLanguageException as hle:
        logger.critical(f"Broken Hill is unable to continue execution due to an error that occurred while working with its internal list of human language names: {tfte}\n{traceback.format_exc()}\n{bug_guidance_message}")
        abnormal_termination = True
        
    except EmbeddingLayerNotFoundException as ele:
        logger.critical(f"Broken Hill is unable to continue execution due to an error that occurred while obtaining the embedding layer from the specified model: {ele}\n{traceback.format_exc()}\n{bug_guidance_message}")
        abnormal_termination = True
        
    except GenerationException as gene:
        logger.critical(f"Broken Hill is unable to continue execution due to an error that occurred while generating content using the specified model: {gene}\n{traceback.format_exc()}\n{bug_guidance_message}")
        abnormal_termination = True
        
    except StatisticsException as se:
        logger.critical(f"Broken Hill is unable to continue execution due to an error that occurred while collecting or processing statistics: {se}\n{traceback.format_exc()}\n{bug_guidance_message}")
        abnormal_termination = True
        
    except LoggingException as le:
        logger.critical(f"Broken Hill is unable to continue execution due to an error that occurred while performing a logging operation: {le}\n{traceback.format_exc()}\n{bug_guidance_message}")
        abnormal_termination = True
        
    except (LargeLanguageModelParameterException, LargeLanguageModelException) as llme:
        logger.critical(f"Broken Hill is unable to continue execution due to an error that occurred while performing an operation related to large language models: {llme}\n{traceback.format_exc()}\n{bug_guidance_message}")
        abnormal_termination = True
        
    except AssociationRebuildException as are:
        logger.critical(f"Broken Hill is unable to continue execution due to an error that occurred while rebuilding associations between search nodes: {are}\n{traceback.format_exc()}\n{bug_guidance_message}")
        abnormal_termination = True

    # Most of the rest of this section uses an extra-cautious approach of wrapping almost everything in try/except logic that will allow the script to continue even if describing an exception results in another exception.
    # This is to help ensure that any output files really are written to persistent storage, even if the script is interrupted or crashes.
    
    except (Exception, RuntimeError) as e:
        # try/catch when populating these variables so that Broken Hill won't crash if processing the f-string fails
        e_string = None
        traceback_string = None
        try:
            e_string = f"{e}"
        except (Exception, RuntimeError) as e:
            e_string = "[Unable to convert exception to a string]"
        try:
            traceback_string = f"{traceback.format_exc()}"
        except (Exception, RuntimeError) as e:
            traceback_string = "[Unable to convert traceback to a string]"
        logger.critical(f"Broken Hill encountered an unhandled exception during the GCG attack: {e_string}. The exception details will be displayed below this message for troubleshooting purposes.\n{traceback_string}\n{bug_guidance_message}")
        abnormal_termination = True

    finished_successfully = True

    if not user_aborted and not abnormal_termination:
        logger.info(f"Main loop complete")
    
    try:
        attack_state.persistable.performance_data.collect_torch_stats(attack_state, location_description = f"after main loop completion")
    except (Exception, RuntimeError) as e:
        e_string = None
        traceback_string = None
        finished_successfully = False        
        try:
            e_string = f"{e}"
        except (Exception, RuntimeError) as e:
            e_string = "[Unable to convert exception to a string]"
        try:
            traceback_string = f"{traceback.format_exc()}"
        except (Exception, RuntimeError) as e:
            traceback_string = "[Unable to convert traceback to a string]"
        logger.critical(f"Broken Hill encountered an exception when trying to collect the final set of performance statistics: {e_string}. The exception details will be displayed below this message for troubleshooting purposes.\n{traceback_string}\n{bug_guidance_message}")

    # if attack_state.persistable.attack_params.write_output_every_iteration is True, this step was just performed and can be skipped
    if not attack_state.persistable.attack_params.write_output_every_iteration:
        try:
            attack_state.persistable.performance_data.write_performance_data(attack_state)
        except (Exception, RuntimeError) as e:
            e_string = None
            traceback_string = None
            finished_successfully = False        
            try:
                e_string = f"{e}"
            except (Exception, RuntimeError) as e:
                e_string = "[Unable to convert exception to a string]"
            try:
                traceback_string = f"{traceback.format_exc()}"
            except (Exception, RuntimeError) as e:
                traceback_string = "[Unable to convert traceback to a string]"
            logger.critical(f"Broken Hill encountered an exception when trying to write performance data output files: {e_string}. The exception details will be displayed below this message for troubleshooting purposes.\n{traceback_string}\n{bug_guidance_message}")

    try:
        if attack_state.persistable.attack_params.torch_cuda_memory_history_file is not None and attack_state.persistable.attack_params.using_cuda():
            logger.info(f"Writing PyTorch CUDA profile data to '{attack_state.persistable.attack_params.torch_cuda_memory_history_file}'.")
            try:
                torch.cuda.memory._dump_snapshot(attack_state.persistable.attack_params.torch_cuda_memory_history_file)
                logger.info(f"Wrote PyTorch CUDA profile data to '{attack_state.persistable.attack_params.torch_cuda_memory_history_file}'.")
            except Exception as e:
                logger.error(f"Couldn't write PyTorch CUDA profile data to '{attack_state.persistable.attack_params.torch_cuda_memory_history_file}': {e}\n{bug_guidance_message}")
    except (Exception, RuntimeError) as e:
        e_string = None
        traceback_string = None
        finished_successfully = False
        try:
            e_string = f"{e}"
        except (Exception, RuntimeError) as e:
            e_string = "[Unable to convert exception to a string]"
        try:
            traceback_string = f"{traceback.format_exc()}"
        except (Exception, RuntimeError) as e:
            traceback_string = "[Unable to convert traceback to a string]"
        logger.critical(f"Broken Hill encountered an exception when trying to write the CUDA profile data to persistent storage: {e_string}. The exception details will be displayed below this message for troubleshooting purposes.\n{traceback_string}\n{bug_guidance_message}")

    end_dt = get_now()
    end_ts = get_time_string(end_dt)
    total_elapsed_string = get_elapsed_time_string(start_dt, end_dt)
    logger.info(f"Completed {attack_state.persistable.main_loop_iteration_number} iterations at {end_ts} - elapsed time {total_elapsed_string} - successful attack count: {attack_state.persistable.successful_attack_count}")
    attack_state.persistable.overall_result_data.end_date_time = end_ts
    attack_state.persistable.overall_result_data.elapsed_time_string = total_elapsed_string
    # collect the stats now so that they're in the files that are written
    try:
        logger.info(f"Processing resource-utilization data.")
        attack_state.persistable.performance_data.populate_statistics()
        logger.info(f"Processing performance data.")
        attack_state.persistable.performance_data.populate_performance_statistics(attack_state)
    except (Exception, RuntimeError) as e:
        e_string = None
        traceback_string = None
        finished_successfully = False
        try:
            e_string = f"{e}"
        except (Exception, RuntimeError) as e:
            e_string = "[Unable to convert exception to a string]"
        try:
            traceback_string = f"{traceback.format_exc()}"
        except (Exception, RuntimeError) as e:
            traceback_string = "[Unable to convert traceback to a string]"
        logger.critical(f"Broken Hill encountered an unhandled exception when trying to process performance data: {e_string}. The exception details will be displayed below this message for troubleshooting purposes.\n{traceback_string}\n{bug_guidance_message}")

    try:
        if attack_state.persistable.attack_params.json_output_file is not None:
            logger.info(f"Writing final version of result data to '{attack_state.persistable.attack_params.json_output_file}'.")
        attack_state.write_output_files()
    except (Exception, RuntimeError) as e:
        e_string = None
        traceback_string = None
        finished_successfully = False
        try:
            e_string = f"{e}"
        except (Exception, RuntimeError) as e:
            e_string = "[Unable to convert exception to a string]"
        try:
            traceback_string = f"{traceback.format_exc()}"
        except (Exception, RuntimeError) as e:
            traceback_string = "[Unable to convert traceback to a string]"
        logger.critical(f"Broken Hill encountered an unhandled exception when trying to write result data to persistent storage: {e_string}. The exception details will be displayed below this message for troubleshooting purposes.\n{traceback_string}\n{bug_guidance_message}")
    
    try:
        if attack_state.persistable.attack_params.save_state:
            logger.info(f"Writing final version of state data to '{attack_state.persistable.attack_params.state_file}'.")
        attack_state.write_persistent_state()
    except (Exception, RuntimeError) as e:
        e_string = None
        traceback_string = None
        finished_successfully = False
        try:
            e_string = f"{e}"
        except (Exception, RuntimeError) as e:
            e_string = "[Unable to convert exception to a string]"
        try:
            traceback_string = f"{traceback.format_exc()}"
        except (Exception, RuntimeError) as e:
            traceback_string = "[Unable to convert traceback to a string]"
        logger.critical(f"Broken Hill encountered an unhandled exception when trying to write the attack state to persistent storage: {e_string}. The exception details will be displayed below this message for troubleshooting purposes.\n{traceback_string}\n{bug_guidance_message}")
    
    if attack_state.persistable.attack_params.log_file_path is not None:
            logger.info(f"This attack has been logged to '{attack_state.persistable.attack_params.log_file_path}'.")
    try:
        attack_state.persistable.performance_data.output_statistics(using_cuda = attack_state.persistable.attack_params.using_cuda(), use_ansi = attack_state.persistable.attack_params.console_ansi_format, verbose = attack_state.persistable.attack_params.verbose_statistics)
    except (Exception, RuntimeError) as e:
        e_string = None
        traceback_string = None
        finished_successfully = False
        try:
            e_string = f"{e}"
        except (Exception, RuntimeError) as e:
            e_string = "[Unable to convert exception to a string]"
        try:
            traceback_string = f"{traceback.format_exc()}"
        except (Exception, RuntimeError) as e:
            traceback_string = "[Unable to convert traceback to a string]"
        logger.critical(f"Broken Hill encountered an unhandled exception when trying to generate a performance report for the attack: {e_string}. The exception details will be displayed below this message for troubleshooting purposes.\n{traceback_string}\n{bug_guidance_message}")
    
    completed_all_iterations = True
    if user_aborted:
        finished_successfully = False
        completed_all_iterations = False
    if abnormal_termination:
        finished_successfully = False
        completed_all_iterations = False
    if attack_state.persistable.main_loop_iteration_number < attack_state.persistable.attack_params.max_iterations:
        finished_successfully = False
        completed_all_iterations = False
    
    slm = attack_state.get_state_loading_message(completed_all_iterations)
    logger.info(slm)
    if not finished_successfully:
        sys.exit(1)
    sys.exit(0)

if __name__=='__main__':
    print(get_logo())
    short_description = get_short_script_description()
    print(f"{script_name} version {script_version}, {script_date}\n{short_description}")
    
    attack_params = AttackParams()
    attack_params.original_command_line_array = copy.deepcopy(sys.argv)
    attack_params.original_command_line = command_array_to_string(attack_params.original_command_line_array)

    # any argument processing that needs to affect the attack_params object goes here, such as loading parameters from a file.
    # first, check to see if state is being loaded from a file, because that should take precedence over anything else
    file_contents = get_file_content_from_sys_argv(sys.argv, "--load-state")
    for i in range(0, len(file_contents)):
        try:
            existing_attack_state = PersistableAttackState.from_json(file_contents[i])            
            existing_attack_params = existing_attack_state.attack_params
            attack_params = AttackParams.apply_dict(attack_params, existing_attack_params.to_dict())
        except Exception as e:
            print(f"Couldn't load options from an existing state file: {e}")
            print(traceback.format_exc())
            sys.exit(1)
    file_contents = get_file_content_from_sys_argv(sys.argv, "--load-options-from-state")
    for i in range(0, len(file_contents)):
        try:
            existing_attack_state = PersistableAttackState.from_json(file_contents[i])
            existing_attack_params = existing_attack_state.attack_params
            attack_params = AttackParams.apply_dict(attack_params, existing_attack_params.to_dict())
        except Exception as e:
            print(f"Couldn't load options from an existing state file: {e}")
            print(traceback.format_exc())
            sys.exit(1)
    file_contents = get_file_content_from_sys_argv(sys.argv, "--load-options")
    for i in range(0, len(file_contents)):
        try:
            existing_attack_params = AttackParams.from_json(file_contents[i])
            attack_params = AttackParams.apply_dict(attack_params, existing_attack_params.to_dict())
        except Exception as e:
            print(f"Couldn't load options from a saved options file: {e}")
            sys.exit(1)
        
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
    
    parser.add_argument("--model", type = str, 
        help="Path to the base directory for the large language model you want to attack, e.g. /home/blincoln/LLMs/StabilityAI/stablelm-2-1_6b-chat")
        
    parser.add_argument("--tokenizer", type = str, 
        help="(optional) Path to the base directory for the LLM tokenizer you want to use with the model instead of any tokenizer that may be included with the model itself. Intended for use with models such as Mamba that do not include their own tokenizer.")
        
    parser.add_argument("--peft-adapter", type = str, 
        help="(optional) Path to the base directory for a PEFT pre-trained model/adapter that is based on the model specified with --model. Used to load models such as Guanaco.")
        
    template_name_list = ", ".join(attack_params.get_known_template_names())
    
    parser.add_argument("--model-data-type", type = str, default="default", choices = [ "default", "torch", "auto", "float16", "bfloat16", "float32", "float64", "complex64", "complex128" ],
        help=f"Specify the type to load the model's data as. 'default' will use the Broken Hill default, which is 'float16' for CUDA devices, and 'bfloat16' for CPU devices. 'torch' will use the PyTorch default, which is 'float32' as of this writing. 'auto' is an experimental option that will cause PyTorch to attempt to load the data in its native format. Default: 'default'. Recommended value for CUDA devices: 'default', 'float16', or (if you have an enormous amount of CUDA device memory) 'float32'. Recommended value for CPU devices: 'default' or 'float32'. 'float64', 'complex64', and 'complex128' are untested at this time, and their use is discouraged. Please see the documentation for additional details.")
    
    parser.add_argument("--template", type = str, 
        help=f"An optional model type name, for selecting the correct chat template. Use --list-templates to view available options. If this option is not specified, the fschat library will attempt to load the correct template based on the base model directory contents.")

    parser.add_argument("--list-templates", type = str2bool, nargs='?',
        const=True,
        help="Output a list of all template names for the version of the fschat library you have installed (to use with --template), then exit.")
        
    parser.add_argument("--do-not-override-fschat-templates", type = str2bool, nargs='?',
        const=True,
        help="If the fschat library already includes a template with a name that matches one of Broken Hill's custom conversation templates, use the fschat version instead.")
        
    parser.add_argument("--system-prompt", type = str, 
        help=f"Specify a custom value to use instead of the default system prompt for the conversation template.")

    parser.add_argument("--system-prompt-from-file", type = str, 
        help=f"Specify a custom value to use instead of the default system prompt for the conversation template by reading it from a file. The entire file will be used as a single string system prompt.")

    parser.add_argument("--clear-existing-conversation", type = str2bool, nargs='?',
        const=True, default=attack_params.clear_existing_template_conversation,
        help="Removes any existing non-system messages from the conversation template.")
        
    parser.add_argument("--template-messages-from-file", type = str, 
        help=f"Load custom messages into the conversation template by reading them from a JSON file. If --clear-existing-conversation is not specified, the messages will be appended to any existing messages in the list for the template. The format of the file should be '[[<role ID or name>, <message>],[<role ID or name>, <message>][...][<role ID or name>, <message>]]', e.g. [[0, \"Do you have any albums by Cindergarden or Mari Kattman?\"], [1, \"We have albums by both artists in stock!\"]] or [[\"Human\", \"Do you have any albums by The Birthday Massacre or Night Club?\"], [\"Assistant\", \"We have albums by both artists in stock!\"]]")

    parser.add_argument("--base-prompt", type = str, 
        help=f"The prompt Broken Hill should try to jailbreak, e.g. 'Write a convincing anonymous death threat.'")
        
    parser.add_argument("--target-output", type = str, 
        help=f"The target output Broken Hill should measure potential results against, e.g. 'Sure, here's a convincing anonymous death threat'")
    
    parser.add_argument("--auto-target", type = str, 
        help=f"Instead of manually specifying separate --base-prompt and --target-output values, specify a single goal (without a leading verb such as 'write'), and Broken Hill will generate the base prompt and target output values, e.g. --auto-target 'a convincing anonymous death threat'")
    
    parser.add_argument("--device", default=attack_params.model_device, type = str, 
        help="The device to use for the PyTorch operations ('cuda', 'cuda:0', etc.). This is a shortcut equivalent to specifying both --model-device and --gradient-device with the same device name.")
    parser.add_argument("--model-device", type = str, 
        help="Experimental: The device to use for loading the model and performing PyTorch operations other than those related to the gradient ('cuda', 'cuda:0', etc.).")
    parser.add_argument("--gradient-device", type = str, 
        help="Experimental: The device to use for PyTorch gradient operations ('cuda', 'cuda:0', etc.).")
    parser.add_argument("--forward-device", type = str, 
        help="Experimental: The device to use for PyTorch aggregation of logits values during the 'forward' operation ('cuda', 'cuda:0', etc.).")
    
    parser.add_argument("--torch-dp-model", type = str2bool, nargs='?',
        const=True,
        help="Untested: Enables the PyTorch 'DataParallel' feature for the model, which should allow utilizing multiple CUDA devices at once.")

    parser.add_argument("--self-test", type = str2bool, nargs='?',
        const=True,
        help="Performs self-testing only, then exits before performing an actual attack. Can be used to test basic model-handling support on systems without enough device memory to perform the GCG attack.")

    parser.add_argument("--initial-adversarial-string", default=attack_params.initial_adversarial_string, type = str, 
        help="The initial string to iterate on. Leave this as the default to perform the attack described in the original paper. Specify the output of a previous run to continue iterating at that point (more or less). Specify a custom value to experiment. Specify an arbitrary number of space-delimited exclamation points to perform the standard attack, but using a different number of initial tokens.")
    
    parser.add_argument("--initial-adversarial-string-base64", type = str, 
        help="Identical to --initial-adversarial-string, except that the value should be specified in Base64-encoded form. This allows easily specifying an initial adversarial string that includes newlines or other characters that are difficult to represent as arguments on the command line.")

    parser.add_argument("--initial-adversarial-token", type = str, 
        nargs = 2,
        metavar = ('token', 'count'),
        help="Specify the initial adversarial content as a single token repeated n times, e.g. for 24 copies of the token '?', --initial-adversarial-token-id '?' 24")

    parser.add_argument("--initial-adversarial-token-id", type = numeric_string_to_int, 
        nargs = 2,
        metavar = ('token_id', 'count'),
        help="Specify the initial adversarial content as a single token repeated n times, e.g. for 24 copies of the token with ID 71, --initial-adversarial-token-id 71 24")
    
    parser.add_argument("--initial-adversarial-token-ids", type = str, 
        help="Specify the initial adversarial content as a comma-delimited list of integer token IDs instead of a string. e.g. --initial-adversarial-token-ids '1,2,3,4,5'")
    
    parser.add_argument("--random-adversarial-tokens", type=numeric_string_to_int,
        help=f"Generate this many random tokens to use as the initial adversarial value instead of specifying explicit initial data. The list of possible tokens will be filtered using the same criteria defined by any of the candidate-filtering options that are also specified.")
    
    # TKTK: adversarial tokens are target tokens
    # TKTK: adversarial tokens are <character> * token count of target tokens
    # TKTK: adversarial tokens are loss tokens (this will require a little work)
    # TKTK: adversarial tokens are <character> * token count of loss tokens (this will require a little work)
    
    
    parser.add_argument("--reencode-every-iteration", type = str2bool, nargs='?',
        const=True,
        help="Emulate the original attack's behaviour of converting the adversarial content to a string and then back to token IDs at every iteration of the attack, instead of persisting it as token IDs only. Enabling this option will cause the number of tokens to change between iterations. Use the --adversarial-candidate-filter-tokens-min, --adversarial-candidate-filter-tokens-max, and/or --attempt-to-keep-token-count-consistent options if you want to try to control how widely the number of tokens varies.")
        
    parser.add_argument("--topk", type=numeric_string_to_int,
        default=attack_params.topk,
        help=f"The number of results assessed when determining the best possible candidate adversarial data for each iteration.")
        
    parser.add_argument("--max-topk", type=numeric_string_to_int,
        default = attack_params.max_topk,
        help=f"The maximum number to allow --topk to grow to when no candidates are found in a given iteration. Default: {attack_params.max_topk}.")

    parser.add_argument("--random-seed-numpy", type=numeric_string_to_int,
        default=attack_params.numpy_random_seed,
        help=f"Random seed for NumPy")
    parser.add_argument("--random-seed-torch", type=numeric_string_to_int,
        default=attack_params.torch_manual_seed,
        help=f"Random seed for PyTorch")
    parser.add_argument("--random-seed-cuda", type=numeric_string_to_int,
        default=attack_params.torch_cuda_manual_seed_all,
        help=f"Random seed for CUDA")

    parser.add_argument("--ignore-prologue-during-gcg-operations", type = str2bool, nargs='?',
        help="If this option is specified, any system prompt and/or template messages will be ignored when performing the most memory-intensive parts of the GCG attack (but not when testing for jailbreak success). This can allow testing in some configurations that would otherwise exceed available device memory, but may affect the quality of results as well.")

    parser.add_argument("--max-iterations", type=numeric_string_to_int,
        default=attack_params.max_iterations,
        help=f"Maximum number of times to iterate on the adversarial data")

    parser.add_argument("--number-of-tokens-to-update-every-iteration", type=numeric_string_to_int,
        default=attack_params.number_of_tokens_to_update_every_iteration,
        help="The number of tokens to randomly alter in candidate adversarial content during every iteration. If this option is set to 1 (the default), the gradient-sampling algorithm from Zou, Wang, Carlini, Nasr, Kolter, and Fredrikson's code is used. If it is set to any other value, the nanoGCG gradient-sampling algorithm is used instead.")
        
    parser.add_argument("--always-use-nanogcg-sampling-algorithm", type = str2bool, nargs='?',
        const=True, default=attack_params.always_use_nanogcg_sampling_algorithm,
        help="If this option is specified, the nanoGCG gradient-sampling algorithm is used even when --number-of-tokens-to-update-every-iteration is 1.")

    parser.add_argument("--new-adversarial-value-candidate-count", type=numeric_string_to_int,
        help=f"The number of candidate adversarial values to generate at every iteration. If you are running out of memory and this value is greater than 1, try reducing it. Alternatively, if you *aren't* running out of memory, you can try increasing this value for better performance.")
        
    parser.add_argument("--max-new-adversarial-value-candidate-count", type=numeric_string_to_int,
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

    parser.add_argument("--exclude-nonascii-tokens", type = str2bool, nargs='?',
        const=True, default=attack_params.exclude_nonascii_tokens,
        help="Bias the adversarial content generation data to avoid using tokens that are not ASCII text.")

    parser.add_argument("--exclude-nonprintable-tokens", type = str2bool, nargs='?',
        const=True, default=attack_params.exclude_nonprintable_tokens,
        help="Bias the adversarial content generation data to avoid using tokens that are not printable.")

    # TKTK: make this --exclude-basic-special-tokens, add another --exclude-all-special-tokens if --exclude-additional-special-tokens doesn't do that already
    parser.add_argument("--exclude-special-tokens", type = str2bool, nargs='?',
        const=True, default=attack_params.exclude_special_tokens,
        help="Bias the adversarial content generation data to avoid using basic special tokens (begin/end of string, padding, unknown).")

    parser.add_argument("--exclude-additional-special-tokens", type = str2bool, nargs='?',
        const=True, default=attack_params.exclude_additional_special_tokens,
        help="Bias the adversarial content generation data to avoid using additional special tokens defined in the tokenizer configuration.")

    parser.add_argument("--exclude-whitespace-tokens", type = str2bool, nargs='?',
        const=True, default=False,
        help="Bias the adversarial content generation data to avoid using tokens that consist solely of whitespace characters.")
    
    parser.add_argument("--exclude-language-names-except", type = str, 
        help="Bias the adversarial content generation data to avoid using tokens that represent names of human languages except the specified IETF language tag, e.g. --exclude-language-names-except en")
    
    parser.add_argument("--list-language-tags", type = str2bool, nargs='?',
        const=True, default=False,
        help="List supported IETF language tags for use with --exclude-language-names-except, then exit.")
    
    parser.add_argument("--exclude-slur-tokens", type = str2bool, nargs='?',
        const=True, default=False,
        help="Bias the adversarial content generation data to avoid using tokens that are contained in a hardcoded list of slurs.")

    parser.add_argument("--exclude-profanity-tokens", type = str2bool, nargs='?',
        const=True, default=False,
        help="Bias the adversarial content generation data to avoid using tokens that are contained in a hardcoded list of profanity.")

    parser.add_argument("--exclude-other-offensive-tokens", type = str2bool, nargs='?',
        const=True, default=False,
        help="Bias the adversarial content generation data to avoid using tokens that are contained in a hardcoded list of other highly-offensive words.")

    parser.add_argument("--exclude-token", action='append', nargs='*', required=False,
        help=f"Bias the adversarial content generation data to avoid using the specified token (if it exists as a discrete value in the model). May be specified multiple times to exclude multiple tokens.")
        
    parser.add_argument("--excluded-tokens-from-file", type = str, required=False,
        help=f"Equivalent to calling --exclude-token for every line in the specified file.")

    parser.add_argument("--excluded-tokens-from-file-case-insensitive", type = str, required=False,
        help=f"Equivalent to calling --exclude-token for every line in the specified file, except that matching is performed without taking upper-/lower-case characters into account.")

    parser.add_argument("--exclude-newline-tokens", type = str2bool, nargs='?',
        const=True, default=False,
        help="A shortcut equivalent to specifying just about any newline token variations using --exclude-token.")

    #parser.add_argument("--exclude-three-hashtag-tokens", type = str2bool, nargs='?',
    #    const=True, default=False,
    #    help="A shortcut equivalent to specifying most variations on the token '###' using --exclude-token.")

    parser.add_argument("--token-filter-regex", type = str,
        help="If specified, biases the adversarial content generation to exclude tokens that don't match the specified regular expression.")

    parser.add_argument("--adversarial-candidate-filter-regex", type = str, 
        default=attack_params.candidate_filter_regex,
        help="The regular expression used to filter candidate adversarial strings. The default value is very forgiving and simply requires that the string contain at least one occurrence of two consecutive mixed-case alphanumeric characters.")
    
    parser.add_argument("--adversarial-candidate-repetitive-line-limit", type=numeric_string_to_int,
        help=f"If this value is specified, candidate adversarial strings will be filtered out if any one line is repeated more than this many times.")
        
    parser.add_argument("--adversarial-candidate-repetitive-token-limit", type=numeric_string_to_int,
        help=f"If this value is specified, candidate adversarial strings will be filtered out if any one token is repeated more than this many times.")
        
    parser.add_argument("--adversarial-candidate-newline-limit", type=numeric_string_to_int,
        help=f"If this value is specified, candidate adversarial strings will be filtered out if they contain more than this number of newline characters.")
        
    parser.add_argument("--adversarial-candidate-newline-replacement", type = str, 
        help="If this value is specified, it will be used to replace any newline characters in candidate adversarial strings. This can be useful if you want to avoid generating attacks that depend specifically on newline-based content, such as injecting different role names.")

    parser.add_argument("--adversarial-candidate-filter-tokens-min", type=numeric_string_to_int,
        help=f"If this value is specified, *and* --reencode-every-iteration is also specified, candidate adversarial strings will be filtered out if they contain fewer than this number of tokens.")
        
    parser.add_argument("--adversarial-candidate-filter-tokens-max", type=numeric_string_to_int,
        help=f"If this value is specified, *and* --reencode-every-iteration is also specified, candidate adversarial strings will be filtered out if they contain more than this number of tokens.")

    parser.add_argument("--attempt-to-keep-token-count-consistent", type = str2bool, nargs='?',
        const=True, default=attack_params.attempt_to_keep_token_count_consistent,
        help="Enable the check from the original attack code that attempts to keep the number of tokens consistent between each adversarial string. This will cause all candidates to be excluded for some models, such as StableLM 2. If you want to limit the number of tokens (e.g. to prevent the attack from wasting time on single-token strings or to avoid out-of-memory conditions), using  `--adversarial-candidate-filter-tokens-min` and `--adversarial-candidate-filter-tokens-max` in combination with `--add-token-when-no-candidates-returned` or `--delete-token-when-no-candidates-returned` may be more effective.")

    parser.add_argument("--add-token-when-no-candidates-returned", type = str2bool, nargs='?',
        const=True, default=attack_params.add_token_when_no_candidates_returned,
        help="If this option is specified, and the number of tokens in the adversarial content is below any restrictions specified by the operator, then a failure to generate any new/untested adversarial content variations will result in a random token in the content being duplicated, increasing the length of the adversarial content by one token.")

    parser.add_argument("--delete-token-when-no-candidates-returned", type = str2bool, nargs='?',
        const=True, default=attack_params.delete_token_when_no_candidates_returned,
        help="If this option is specified, and the number of tokens in the adversarial content is greater than any minimum specified by the operator, then a failure to generate any new/untested adversarial content variations will result in a random token in the content being deleted, reducing the length of the adversarial content by one token. If both this option and --add-token-when-no-candidates-returned are enabled, and the prequisites for both options apply, then a token will be added.")

    parser.add_argument("--random-seed-comparisons", type=numeric_string_to_int, default = attack_params.random_seed_comparisons,
        help=f"If this value is greater than zero, at each iteration, Broken Hill will test results using the specified number of additional random seed values, to attempt to avoid focusing on fragile results. The sequence of random seeds is hardcoded to help make results deterministic.")
    
    parser.add_argument("--temperature", type=numeric_string_to_float,
        default=None,
        help=f"If --random-seed-comparisons is specified, the 'Temperature' value to pass to all of the randomized instances of the LLM.")

    parser.add_argument("--temperature-range", type=numeric_string_to_float,
        nargs = 2,
        default=None,
        help=f"If --random-seed-comparisons is specified, the low and high end (inclusive) of a range of temperature values to pass to the model. The instance of the LLM used with the first random seed will be assigned the low temperature value. The instance of the LLM used with the last random seed will be assigned the high temperature value. If there are more than two instances of the LLM, the remaining instances will be assigned temperature values evenly distributed between the low and high values.")

    parser.add_argument("--do-sample", type = str2bool, nargs='?',
        const=True,
        help="Enables the 'do_sample' option for the primary (or only) LLM instance, instead of only the additional instances used in --random-seed-comparisons mode. This option is included for development and testing only. Please do not file an issue if it doesn't do what you expect.")
    
    # not currently used - see discussion in attack_classes.py
    #parser.add_argument("--scoring-mode", type = str, default="median", choices=[ "median", "average", "minimum", "maximum" ],
    #    help=f"If --random-seed-comparisons is set to 1 or more, use this statistical function to generate an overall score for the results. Default: median.")

    parser.add_argument("--generic-role-template", type = str, 
        help="The Python formatting string to use if fschat defaults to a generic chat template. e.g --generic-role-template '[{role}]', '<|{role}|>'.")
    
    parser.add_argument("--trust-remote-code", type = str2bool, nargs='?',
        const=True, default=attack_params.load_options_trust_remote_code,
        help="When loading the model, pass 'trust_remote_code=True', which enables execution of arbitrary Python scripts included with the model. You should probably examine those scripts first before deciding if you're comfortable with this option. Currently required for some models, such as Phi-3.")
    parser.add_argument("--ignore-mismatched-sizes", type = str2bool, nargs='?',
        const=True, default=attack_params.load_options_ignore_mismatched_sizes,
        help="When loading the model, pass 'ignore_mismatched_sizes=True', which may allow you to load some models with mismatched size data. It will probably only let Broken Hill get a little further before erroring out, though.")

    parser.add_argument("--jailbreak-detection-rules-file", type = str, 
        help=f"If specified, loads the jailbreak detection rule set from a JSON file instead of using the default rule set.")
    parser.add_argument("--write-jailbreak-detection-rules-file", type = str, 
        help=f"If specified, writes the jailbreak detection rule set to a JSON file and then exits. If --jailbreak-detection-rules-file is not specified, this will cause the default rules to be written to the file. If --jailbreak-detection-rules-file *is* specified, then the custom rules will be normalized and written in the current standard format to the output file.")

    parser.add_argument("--break-on-success", type = str2bool, nargs='?',
        const=True, default=attack_params.break_on_success,
        help="Stop iterating upon the first detection of a potential successful jailbreak.")
        
    parser.add_argument("--verbose-self-test-output", type = str2bool, nargs='?',
        const=True, default=attack_params.verbose_self_test_output,
        help="If self-test operations fail badly enough to output a comparison of strings generated, also output token ID and token information for debugging.")

    parser.add_argument("--ignore-jailbreak-self-tests", type = str2bool, nargs='?',
        const=True, default=attack_params.ignore_jailbreak_self_tests,
        help="Perform the attack even if the jailbreak self-tests indicate that the results will likely not be useful.")

    parser.add_argument("--model-parameter-info", type = str2bool, nargs='?',
        const=True, default=attack_params.verbose_model_parameter_info,
        help="Display detailed information about the model's parameters after loading the model.")
        
    parser.add_argument("--verbose-resource-info", type = str2bool, nargs='?',
        const=True, default=attack_params.verbose_resource_info,
        help="Display system resource utilization/performance information every time it's collected instead of only at key intervals.")

    parser.add_argument("--verbose-stats", type = str2bool, nargs='?',
        const=True, default=attack_params.verbose_statistics,
        help="Display verbose resource utilization/performance statistics when Broken Hill finishes testing, instead of the shorter default list.")

    # TKTK: add an option to score results by number of tokens that appear in LLM output that are also present in the target output, after removing trivial words like "the", "it", etc. Maybe it's not even an option, it's just another scoring element that's performed at every iteration, like loss.

    parser.add_argument("--required-loss-threshold", type=numeric_string_to_float,
        default=attack_params.required_loss_threshold,
        help=f"During the candidate adversarial content generation stage, require that the loss for the best value be lower than the previous loss plus this amount.")

    parser.add_argument("--loss-threshold-max-attempts", type=numeric_string_to_int,
        default=attack_params.loss_threshold_max_attempts,
        help=f"If --required-loss-threshold has been specified, make this many attempts at finding a value that meets the threshold before giving up. If --exit-on-loss-threshold-failure is *not* specified, Broken Hill will use the value with the lowest loss found during the attempt to find a value that met the threshold. If --exit-on-loss-threshold-failure is specified, Broken Hill will exit if it is unable to find a value that meets the requirement.")

    parser.add_argument("--exit-on-loss-threshold-failure", type = str2bool, nargs='?',
        const=True, default=attack_params.exit_on_loss_threshold_failure,
        help=f"During the candidate adversarial content generation stage, require that the loss for the best value be lower than the previous loss plus this amount.")

    parser.add_argument("--rollback-on-loss-increase", type = str2bool, nargs='?',
        const=True, default=attack_params.rollback_on_loss_increase,
        help="If the loss value increases between iterations, roll back to the last 'good' adversarial data. This option is not recommended, and included for experimental purposes only.")
    parser.add_argument("--rollback-on-loss-threshold", type=numeric_string_to_float,
        help=f"Equivalent to --rollback-on-loss-increase, but only if the loss value increases by more than the specified amount for a given iteration. Like --rollback-on-loss-increase, using this option is not recommended.")

    parser.add_argument("--rollback-on-jailbreak-count-decrease", type = str2bool, nargs='?',
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
    
    parser.add_argument("--loss-slice-is-index-shifted-target-slice", type = str2bool, nargs='?',
        const=True, default=False,
        help="This option causes the loss slice to be determined by starting with the target slice, and decreasing the start and end indices by 1, so that the length remains identical to the target slice, but the loss slice sometimes includes at least part of the LLM-role-indicator token. This is the behaviour that the original GCG attack code used, and it is the default mode.")

    # TKTK: option to set the index shifting
    
    # TKTK: option to shift left, but pad the comparison values to the right, like with the LLM role options
        
    # TKTK: loss slice is index-shifted base prompt plus target (set to zero shift for no shift)

    # TKTK: loss slice is index-shifted base prompt plus target with padding
    
    parser.add_argument("--loss-slice-is-llm-role-and-truncated-target-slice", type = str2bool, nargs='?',
        const=True, default=False,
        help="This option causes the loss slice to be determined by starting with the token(s) that indicate the speaking role is switching from user to LLM, and includes as many of the tokens from the target string as will fit without the result exceeding the length of the target slice. This is similar to the original GCG attack code method (--loss-slice-is-index-shifted-target-slice), but should work with any LLM, even those that use multiple tokens to indicate a role change.")

    parser.add_argument("--loss-slice-is-llm-role-and-full-target-slice", type = str2bool, nargs='?',
        const=True, default=False,
        help="This option causes the loss slice to be determined by starting with the token(s) that indicate the speaking role is switching from user to LLM, and includes all of the target string.")

    parser.add_argument("--loss-slice-is-target-slice", type = str2bool, nargs='?',
        const=True, default=False,
        help="This option makes the loss slice identical to the target slice. This will break the GCG attack, so you should only use this option if you want to prove to yourself that shifting those indices really is a fundamental requirement for the GCG attack.")

    parser.add_argument("--loss-slice-index-shift", type=numeric_string_to_int,
        default=attack_params.loss_slice_index_shift,
        help=f"When using --loss-slice-is-index-shifted-target-slice, shift the indices by this amount instead of the default.")

    # parser.add_argument("--mellowmax", type = str2bool, nargs='?',
        # const=True, default=False,
        # help="If this option is specified, the attack will use the mellowmax loss algorithm (borrowed from nanoGCG) instead of the cross-entropy loss from Zou, Wang, Carlini, Nasr, Kolter, and Fredrikson's code.")

    # parser.add_argument("--mellowmax-alpha", type=numeric_string_to_float,
        # default=attack_params.mellowmax_alpha,
        # help=f"If --loss-mellowmax is specified, this setting controls the 'alpha' value (default: 1.0).")

    parser.add_argument("--display-failure-output", type = str2bool, nargs='?',
        const=True, default=attack_params.display_full_failed_output,
        help="Output the full decoded input and output for failed jailbreak attempts (in addition to successful attempts, which are always output).")
    parser.add_argument("--suppress-attention-mask", type = str2bool, nargs='?',
        const=True,
        help="Do not pass an attention mask to the model. Required for some models, such as Mamba, but may invalidate results.")
    parser.add_argument("--low-cpu-mem-usage", type = str2bool, nargs='?',
        const=True, default=attack_params.low_cpu_mem_usage,
        help="When loading the model and tokenizer, pass 'low_cpu_mem_usage=True'. May or may not affect performance and results.")
    parser.add_argument("--use-cache", type = str2bool, nargs='?',
        const=True,
        help="When loading the model and tokenizer, pass 'use_cache = True'. May or may not affect performance and results. This is the default behaviour for Broken Hill starting with version 0.34.")
    parser.add_argument("--no-torch-cache", type = str2bool, nargs='?',
        const=True,
        help="When loading the model and tokenizer, pass 'use_cache = False'. May or may not affect performance and results.")
    parser.add_argument("--preserve-gradient", type = str2bool, nargs='?',
        const=True,
        help="Retain the coordinate gradient between iterations of the attack instead of deleting it. This may result in increased device memory use.")
    parser.add_argument("--display-model-size", type = str2bool, nargs='?',
        const=True, default=attack_params.display_model_size,
        help="Displays size information for the selected model. Warning: will write the raw model data to a temporary file, which may double the load time.")
    parser.add_argument("--force-python-tokenizer", type = str2bool, nargs='?',
        const=True, default=attack_params.force_python_tokenizer,
        help="Use the Python tokenizer even if the model supports a (usually faster) non-Python tokenizer. May allow use of some models that include incomplete non-Python tokenizers.")
    parser.add_argument("--enable-hardcoded-tokenizer-workarounds", type = str2bool, nargs='?',
        help="Enable the undocumented, hardcoded tokenizer workarounds that the original developers introduced for some models.")
    parser.add_argument("--force-qwen-workaround", type = str2bool, nargs='?',
        help="Attempt to load the model using the dtype workaround necessary for some Qwen-1 models (and derived models), even if Broken Hill does not automatically detect that the workaround may be necessary.")
    padding_token_values = get_missing_pad_token_names()
    parser.add_argument("--missing-pad-token-replacement", type = str, 
        default = "default",
        choices = padding_token_values,
        help=f"If the tokenizer is missing a padding token definition, use an alternative special token instead. Must be one of: {padding_token_values}. If 'default' is specified, no alternative value will be explicitly specified, and the behaviour will be determined by the Transformers library and any other third-party code related to the tokenizer.")
    parser.add_argument("--padding-side", type = str,
        choices = [ "default", "none", "left", "right" ], default = "default",
        help=f"If this value is not 'default', configure the tokenizer to always used the specified padding side. If this value is 'default', use the tokenizer's value. Must be one of 'default', 'left' or 'right'.")
    parser.add_argument("--json-output-file", type = str,
        help=f"Write detailed result data in JSON format to the specified file.")
    parser.add_argument("--performance-output-file", type = str,
        help=f"Write detailed performance/resource-utilization data in JSON format to the specified file.")
    parser.add_argument("--torch-cuda-memory-history-file", type = str,
        help=f"Use PyTorch's built-in CUDA profiling feature to generate a pickled blob of data that can be used to visualize CUDA memory use during the entire Broken Hill run. See https://pytorch.org/docs/stable/torch_cuda_memory.html for more details on the file and how to use it.")
    parser.add_argument("--state-directory", type = str,
        help=f"Back up attack state data to a file in the specified directory at every iteration instead of the default location. If this option is not specified, Broken Hill will store its state files in a subdirectory of the current user's home directory named '{attack_params.default_state_directory}'.")
    parser.add_argument("--state-file", type = str,
        help=f"Override Broken Hill's standard behaviour and write state information to a specific file instead of creating a new file for the attack within the state-backup directory. Using this option is strongly discouraged due to the potential for accidentally overwriting useful information. This option can only be used if --state-directory is not.")
    parser.add_argument("--load-state", type = str,
        help=f"Load the attack state from the specified JSON file, to resume a test that exited early, continue with additional iterations beyond the original limit, etc. If this option is specified, a new state file will be created to store the results of the resumed test, unless --overwrite-existing-state is also specified. The new state file will be created in the same directory as the existing state file, unless --state-directory is also specified.")
    parser.add_argument("--overwrite-output", type = str2bool, nargs='?',
        const=True,
        help="Overwrite any existing output files (--json-output-file, etc.) instead of exiting with an error.")
    parser.add_argument("--disable-state-backup", type = str2bool, nargs='?',
        const=True,
        help="Prevents the automatic backup of attack state at each iteration that Broken Hill performs by default. Using this option is not recommended except during development and testing of Broken Hill.")
    parser.add_argument("--overwrite-existing-state", type = str2bool, nargs='?',
        const=True,
        help="When --load-state is specified, continue saving state to the same file instead of creating a new state file. Using this option is not recommended, because it risks losing all of the progress saved in the existing state file.")
    parser.add_argument("--delete-state-on-completion", type = str2bool, nargs='?',
        const=True,
        help="If this option is specified, *and* Broken Hill reaches the maximum configured number of iterations (or --break-on-success is specified and Broken Hill discovers a jailbreak), the automatically-generated state file will be deleted. If this option is not specified, the state file will be retained for use with the --load-state option. If --load-state is specified, but --overwrite-existing-state is not specified, *only* the new state file will be deleted upon successful completion. If --load-state and --overwrite-existing-state are both specified, the state file that was used to resume testing will be deleted on successful completion. Using this option is strongly discouraged due to the possibility of unintentionally deleting data.")
    parser.add_argument("--only-write-files-on-completion", type = str2bool, nargs='?',
        const=True,
        help="If this option is specified, Broken Hill will only write the following output files to persistent storage at the end of an attack, or if the attack is interrupted: JSON-formatted result data (--json-output-file), performance statistics (--performance-output-file), and the attack state. This can significantly boost the performance of longer attacks by avoiding repeated writes of increasingly large files. However, if Broken Hill encounters an unhandled error, *all* of the results may be lost. When this option is not specified, result data and state are written to persistent storage at every iteration, and performance statistics are written every time they're collected. This option is mainly included for use in testing, to avoid unnecessarily reducing the lifetime of the test system's disk array.")
    parser.add_argument("--save-options", type = str,
        help=f"Save all of the current attack parameters (default values + any explicitly-specified command-line options) to the specified file in JSON format, then exit.")
    parser.add_argument("--load-options", type = str,
        help=f"Load all attack parameters from the specified JSON file. Any additional command-line options specified will override the values from the JSON file. This option may be specified multiple times to merge several partial options files together in the order specified.")
    parser.add_argument("--load-options-from-state", type = str,
        help=f"Load all attack parameters from the specified Broken Hill state-backup file. This option may be specified multiple times to merge several partial options files together in the order specified. If this option and --load-options are both specified, then any files specified using --load-options-from-state are processed first, in order, before applying any files specified using --load-options.")
    parser.add_argument("--log", type = str,
        help=f"Write output to the specified log file in addition to the console.")
    parser.add_argument("--log-level", type = str,
        choices = get_log_level_names(),
        help=f"Limit log file entries to severities of the specified level and above.")
    parser.add_argument("--console-level", type = str,
        choices = get_log_level_names(),
        help=f"Limit console output to severities of the specified level and above.")
    parser.add_argument("--third-party-module-level", type = str,
        choices = get_log_level_names(),
        help=f"Set the default logging level for messages generated by third-party modules. The default is 'warning' because PyTorch in particular is very chatty when set to 'info' or below.")
    parser.add_argument("--debugging-tokenizer-calls", type = str2bool, nargs='?',
        const=True,
        help="Enable extra debug log entries that requires making calls to the tokenizer to encode, decode, etc.")
    parser.add_argument("--no-ansi", type = str2bool, nargs='?',
        const=True,
        help="Do not use ANSI formatting codes to colourize console output")


    args = parser.parse_args()
    
    # BEGIN: any arguments related to logging need to be handled here
    if args.log:
        attack_params.log_file_path = os.path.abspath(args.log)
    # if attack_params.log_file_path is not None:
        # verify_output_file_capability(attack_params.log_file_path, attack_params.overwrite_output)
    if args.log_level:
        attack_params.log_file_output_level = log_level_name_to_log_level(args.log_level)
    if args.console_level:
        attack_params.console_output_level = log_level_name_to_log_level(args.console_level)
    if args.third_party_module_level:
        attack_params.third_party_module_output_level = log_level_name_to_log_level(args.third_party_module_level)
    if args.debugging_tokenizer_calls:
        attack_params.generate_debug_logs_requiring_extra_tokenizer_calls = True
    if args.no_ansi:
        attack_params.console_ansi_format = False
    
    log_manager = BrokenHillLogManager(attack_params)
    log_manager.initialize_handlers()
    log_manager.remove_all_existing_handlers()
    log_manager.attach_handlers_to_all_modules()
    log_manager.attach_handlers(__name__)
    logger = logging.getLogger(__name__)
    logger.setLevel(log_manager.get_lowest_log_level())
    logger.info(f"Log handlers are attached")
    # Capture all Python warnings to avoid PyTorch (and similar) warnings from being displayed outside of the log handler
    logging.captureWarnings(True)
    # END: any arguments related to logging need to be handled here
    
    cuda_available = torch.cuda.is_available()
    mps_available = torch.backends.mps.is_available()
    
    if not cuda_available:
        logger.warning(f"This host does not appear to have a PyTorch CUDA back-end available. Broken Hill will default to the PyTorch device '{attack_params.device_fallback}' instead.")
        attack_params.model_device = attack_params.device_fallback
        attack_params.gradient_device = attack_params.device_fallback
        attack_params.forward_device = attack_params.device_fallback

    if args.list_language_tags:
        attack_params.operating_mode = BrokenHillMode.LIST_IETF_TAGS

    if args.list_templates:
        fc_template_list = []
        for fct_name in fschat_conversation.conv_templates.keys():
            fc_template_list.append(fct_name)
        fc_template_list.sort()     
        list_string = "Custom conversation templates developed by Bishop Fox:\n"
        for ct_name in get_custom_conversation_template_names():
            list_string += f"{ct_name}\n"
        list_string += "\nAll templates included with the version of the fschat library (frequently referred to as 'fastchat') in your environment:\n"
        for fctl in fc_template_list:
            list_string += f"{fctl}\n"
        logger.info(list_string)
        sys.exit(0)

    #attack_params.device = args.device
    default_device = attack_params.model_device
    combined_device_params = False
    individual_device_params = False
    if args.device != default_device:
        attack_params.model_device = args.device
        attack_params.gradient_device = args.device
        attack_params.forward_device = args.device
        combined_device_params = True
    if args.model_device:
        attack_params.model_device = args.model_device
        individual_device_params = True
    if args.gradient_device:
        attack_params.gradient_device = args.gradient_device
        individual_device_params = True
    if args.forward_device:
        attack_params.forward_device = args.forward_device
        individual_device_params = True
    if individual_device_params and combined_device_params:
        logger.critical(f"--device can only be specified if none of --model-device, --gradient-device, and --forward-device are specified.")
        sys.exit(1)

    if args.torch_dp_model:
        attack_params.torch_dataparallel_model = True

    if cuda_available:
        if len(attack_params.get_non_cuda_devices()) > 0:
            logger.warning(f"This system appears to have a PyTorch CUDA back-end available, but at least one back-end option has been set to a non-CUDA device instead. This is likely to result in significantly decreased performance versus using the CUDA back-end. If this decision was intentional (e.g. the test system does not have enough CUDA device memory to perform the processing, but does have enough system RAM), you can ignore this message.")

    check_pytorch_devices(attack_params)

    if mps_available:
        if attack_params.using_mps():
            logger.warning(f"This host appears to be an Apple device with support for the Metal ('mps') PyTorch back-end. At the time this version of Broken Hill was developed, the Metal back-end did not support some features that were critical to the attack code, such as nested tensors. This attack is therefore unlikely to succeed, and you should specify --device cpu instead. This message will be removed when Bishop Fox has verified that the Metal back-end supports the necessary features.")

    if args.tokenizer:
        attack_params.tokenizer_path = os.path.abspath(args.tokenizer)
        if not os.path.isdir(attack_params.tokenizer_path):
            logger.critical(f"The specified tokenizer directory ('{attack_params.tokenizer_path}') does not appear to exist.")
            sys.exit(1)
            
    if args.peft_adapter:
        attack_params.peft_adapter_path = os.path.abspath(args.peft_adapter)
        if not os.path.isdir(attack_params.peft_adapter_path):
            logger.critical(f"The specified PEFT adapter directory ('{attack_params.peft_adapter_path}') does not appear to exist.")
            sys.exit(1)
        
    experimental_dtype = False
    
    if args.model_data_type == "default":
        if attack_params.model_device_is_cuda():
            attack_params.model_data_format_handling = ModelDataFormatHandling.FORCE_FLOAT16
        else:
            attack_params.model_data_format_handling = ModelDataFormatHandling.FORCE_BFLOAT16
        
    if args.model_data_type == "torch":        
        attack_params.model_data_format_handling = ModelDataFormatHandling.TORCH_DEFAULT
        
    if args.model_data_type == "auto":        
        attack_params.model_data_format_handling = ModelDataFormatHandling.AUTO
        experimental_dtype = True
        
    if args.model_data_type == "float16":        
        attack_params.model_data_format_handling = ModelDataFormatHandling.FORCE_FLOAT16
    
    if args.model_data_type == "bfloat16":        
        attack_params.model_data_format_handling = ModelDataFormatHandling.FORCE_BFLOAT16

    if args.model_data_type == "float32":        
        attack_params.model_data_format_handling = ModelDataFormatHandling.FORCE_FLOAT32
    
    if args.model_data_type == "float64":        
        attack_params.model_data_format_handling = ModelDataFormatHandling.FORCE_FLOAT64
        experimental_dtype = True
    
    if args.model_data_type == "complex64":        
        attack_params.model_data_format_handling = ModelDataFormatHandling.FORCE_COMPLEX64
        experimental_dtype = True
    
    if args.model_data_type == "complex128":        
        attack_params.model_data_format_handling = ModelDataFormatHandling.FORCE_COMPLEX128
        experimental_dtype = True
    
    if experimental_dtype:
        dtype_warning = f"The operator has specified --model-data-type {args.model_data_type}. This is an experimental mode and may cause Broken Hill to crash, perform poorly, or produce inaccurate results."
        dtype_suggestions = "'bfloat16', or 'float32'"
        if attack_params.model_device_is_cuda():
            dtype_suggestions = "'float16', 'bfloat16', or 'float32'"
        dtype_warning = f"{dtype_warning} Using one of 'default', {dtype_suggestions} is currently recommended instead."
        logger.warning(dtype_warning)
    
    if args.template:
        attack_params.template_name = args.template

    if args.do_not_override_fschat_templates:
        attack_params.override_fschat_templates = False

    if args.clear_existing_conversation:
        attack_params.clear_existing_template_conversation = True
    
    if args.system_prompt:
        attack_params.custom_system_prompt = args.system_prompt
        
    if args.system_prompt_from_file:
        if args.system_prompt:
            logger.critical(f"Only one of --system-prompt-from-file and --system-prompt may be specified.")
            sys.exit(1)
        system_prompt_file = os.path.abspath(args.system_prompt_from_file)
        attack_params.custom_system_prompt = get_file_content(system_prompt_file, failure_is_critical = True)

    if args.template_messages_from_file:
        message_file = os.path.abspath(args.template_messages_from_file)
        message_file_content = get_file_content(message_file, failure_is_critical = True)
        try:
            attack_params.set_conversation_template_messages(json.loads(message_file_content))
        except Exception as e:
            logger.critical(f"Error loading conversation template messages from file '{message_file}', content '{message_file_content}': {e}\n{traceback.format_exc()}")
            sys.exit(1)

    initial_data_method_count = 0
    if args.initial_adversarial_string != attack_params.initial_adversarial_string:
        initial_data_method_count += 1
    attack_params.initial_adversarial_string = args.initial_adversarial_string
    
    if args.initial_adversarial_string_base64:
        initial_data_method_count += 1
        attack_params.initial_adversarial_string = base64.b64decode(args.initial_adversarial_string_base64).decode("utf-8")
    
    initial_adversarial_string_stripped = attack_params.initial_adversarial_string.strip()
    if attack_params.initial_adversarial_string != initial_adversarial_string_stripped:
        logger.warning(f"The initial adversarial string '{attack_params.initial_adversarial_string}' includes leading and/or trailing whitespace characters. This may result in unexpected behaviour during the attack, such as warning messages about the number of top token indices. If you did not intentionally include the leading/trailing whitespace, you should restart the attack using --initial-adversarial-string {DOUBLE_QUOTE}{initial_adversarial_string_stripped}{DOUBLE_QUOTE} instead of --initial-adversarial-string {DOUBLE_QUOTE}{attack_params.initial_adversarial_string}{DOUBLE_QUOTE}.")
    
    if args.initial_adversarial_token:
        initial_data_method_count += 1
        initial_token_string, attack_params.initial_adversarial_token_count = args.initial_adversarial_token
        attack_params.initial_adversarial_token_count = numeric_string_to_int(attack_params.initial_adversarial_token_count)
        attack_params.initial_adversarial_token_string = initial_token_string
        attack_params.initial_adversarial_content_creation_mode = InitialAdversarialContentCreationMode.SINGLE_TOKEN
        if attack_params.initial_adversarial_token_count < 1:
            logger.critical(f"The number of tokens specified as the second parameter for  --initial-adversarial-token must be a positive integer.")
            sys.exit(1)
    
    if args.initial_adversarial_token_id:
        initial_data_method_count += 1
        initial_token_id, attack_params.initial_adversarial_token_count = args.initial_adversarial_token_id
        attack_params.initial_adversarial_token_ids = []
        for i in range(0, attack_params.initial_adversarial_token_count):
            attack_params.initial_adversarial_token_ids.append(initial_token_id)
        attack_params.initial_adversarial_content_creation_mode = InitialAdversarialContentCreationMode.FROM_TOKEN_IDS
        if attack_params.initial_adversarial_token_count < 1:
            logger.critical(f"The number of tokens specified as the second parameter for  --initial-adversarial-token-id must be a positive integer.")
            sys.exit(1)

    if args.initial_adversarial_token_ids:
        initial_data_method_count += 1
        attack_params.initial_adversarial_token_ids = comma_delimited_string_to_integer_array(args.initial_adversarial_token_ids)
        attack_params.initial_adversarial_content_creation_mode = InitialAdversarialContentCreationMode.FROM_TOKEN_IDS
        if len(attack_params.initial_adversarial_token_ids) < 1:
            logger.critical(f"At least one adversarial token ID must be specified when using --initial-adversarial-token-ids.")
            sys.exit(1)

    if args.random_adversarial_tokens:
        initial_data_method_count += 1
        attack_params.initial_adversarial_token_count = args.random_adversarial_tokens
        attack_params.initial_adversarial_content_creation_mode = InitialAdversarialContentCreationMode.RANDOM_TOKEN_IDS
        if attack_params.initial_adversarial_token_count < 1:
            logger.critical(f"The value specified for --random-adversarial-tokens must be a positive integer.")
            sys.exit(1)

    if args.initial_adversarial_token_id:
        initial_data_method_count += 1
        attack_params.initial_adversarial_token_count = args.random_adversarial_tokens
        attack_params.initial_adversarial_content_creation_mode = InitialAdversarialContentCreationMode.RANDOM_TOKEN_IDS
        if attack_params.initial_adversarial_token_count < 1:
            logger.critical(f"The value specified for --random-adversarial-tokens must be a positive integer.")
            sys.exit(1)

    
    if initial_data_method_count > 1:
        logger.critical(f"Only one of the following options may be specified: --initial-adversarial-string, --initial-adversarial-string-base64, --initial-adversarial-token, --initial-adversarial-token-id, --initial-adversarial-token-ids, --random-adversarial-tokens.")
        sys.exit(1)
    
    if args.self_test:
        attack_params.operating_mode = BrokenHillMode.GCG_ATTACK_SELF_TEST
        
    if args.reencode_every_iteration:
        attack_params.reencode_adversarial_content_every_iteration = args.reencode_every_iteration

    if args.auto_target:
        if args.base_prompt or args.target_output:
            logger.critical(f"Cannot specify --auto-target when either --base-prompt or --target-output are also specified")
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
    #       Broken Hill uses the last node as a starting point.
    #       Then if that branch doesn't result in any successes for enough iterations, Broken Hill will "roll back" to the previous "best score" value in the chain.
    #   It's not perfect, because it's possible that some of the nodes will never be branched from again even though they're equally "good" starting points, but it would be fairly easy to implement.
    # 
    # Another option: a true queuing system, with support for handing off data to other instances of Broken Hill.
    #   e.g. there is a queue for each level of success score, and the user can set a threshold where results are only queued if they're within n of whatever scoring mechanisms are in use.
    #       a given instance of Broken Hill will select the next value from the highest-value queue that has entries queued, process it, and if necessary return results to the queue.
    #   that would be really neat, but it's probably going to be awhile before I get around to implementing it.
    #   Seems like it would require a "queue server" that all of the instances would connect to.

    if not isinstance(args.loss_slice_is_llm_role_and_full_target_slice, type(None)) and args.loss_slice_is_llm_role_and_full_target_slice == True:
        attack_params.loss_slice_mode = LossSliceMode.ASSISTANT_ROLE_PLUS_FULL_TARGET_SLICE        
        
    if not isinstance(args.loss_slice_is_llm_role_and_truncated_target_slice, type(None)) and args.loss_slice_is_llm_role_and_truncated_target_slice == True:
        attack_params.loss_slice_mode = LossSliceMode.ASSISTANT_ROLE_PLUS_TRUNCATED_TARGET_SLICE

    if not isinstance(args.loss_slice_is_index_shifted_target_slice, type(None)) and args.loss_slice_is_index_shifted_target_slice == True:
        attack_params.loss_slice_mode = LossSliceMode.INDEX_SHIFTED_TARGET_SLICE
        #logger.warning(f"--loss-slice-is-index-shifted-target-slice was specified. This will work as expected with some LLMs, but likely fail to generate useful results for LLMs that have multi-token role indicators, such as Gemma and Llama.")

    if not isinstance(args.loss_slice_is_target_slice, type(None)) and args.loss_slice_is_target_slice == True:
        attack_params.loss_slice_mode = LossSliceMode.SAME_AS_TARGET_SLICE
        logger.warning(f"--loss-slice-is-target-slice was specified. This will prevent the GCG attack from working correctly. Expect poor results.")

    attack_params.loss_slice_index_shift = args.loss_slice_index_shift

    # if not isinstance(args.mellowmax, type(None)) and args.mellowmax == True:
        # attack_params.loss_algorithm = LossAlgorithm.MELLOWMAX  

    # if not isinstance(args.mellowmax_alpha, type(None)):
        # attack_params.mellowmax_alpha = args.mellowmax_alpha

    attack_params.topk = args.topk
    
    if not isinstance(args.max_topk, type(None)):
        attack_params.max_topk = args.max_topk

    if args.temperature:
        attack_params.model_temperature_range_begin = args.temperature
        attack_params.model_temperature_range_end = args.temperature

    if args.temperature_range:
        attack_params.model_temperature_range_begin, attack_params.model_temperature_range_end = args.temperature_range

    if args.do_sample:
        attack_params.always_do_sample = True

    attack_params.numpy_random_seed = args.random_seed_numpy

    attack_params.torch_manual_seed = args.random_seed_torch

    attack_params.torch_cuda_manual_seed_all = args.random_seed_cuda

    if args.ignore_prologue_during_gcg_operations:
        attack_params.ignore_prologue_during_gcg_operations = True
        logger.warning(f"Ignoring system prompt (if any) and template messages (if any) when performing GCG-related calculations. This may affect the quality of the results for this test.")

    attack_params.max_iterations = args.max_iterations
    
    attack_params.number_of_tokens_to_update_every_iteration = args.number_of_tokens_to_update_every_iteration

    if not isinstance(args.always_use_nanogcg_sampling_algorithm, type(None)) and args.always_use_nanogcg_sampling_algorithm == True:
        attack_params.always_use_nanogcg_sampling_algorithm = True
        if attack_params.number_of_tokens_to_update_every_iteration == 1:
            logger.warning("Using the nanoGCG gradient-sampling algorithm even though only one token will be updated during each iteration.")

    set_new_adversarial_value_candidate_count = False
    set_max_new_adversarial_value_candidate_count = False

    if args.new_adversarial_value_candidate_count:
        attack_params.new_adversarial_value_candidate_count = args.new_adversarial_value_candidate_count
        set_new_adversarial_value_candidate_count = True
    
    if args.max_new_adversarial_value_candidate_count:
        attack_params.max_new_adversarial_value_candidate_count = args.max_new_adversarial_value_candidate_count
        set_max_new_adversarial_value_candidate_count = True
    
    if attack_params.max_new_adversarial_value_candidate_count < attack_params.new_adversarial_value_candidate_count:
        if set_max_new_adversarial_value_candidate_count:
            value_source_string = "the default new adversarial value candidate count"
            if set_new_adversarial_value_candidate_count:
                value_source_string = "the value specified for --new-adversarial-value-candidate-count"    
            logger.warning(f"The value specified for --max-new-adversarial-value-candidate-count ({attack_params.max_new_adversarial_value_candidate_count}) was less than {value_source_string} ({attack_params.new_adversarial_value_candidate_count}). Both values will be set to {attack_params.max_new_adversarial_value_candidate_count}.")
            attack_params.new_adversarial_value_candidate_count = attack_params.max_new_adversarial_value_candidate_count
        else:
            logger.warning(f"The value specified for --new-adversarial-value-candidate-count ({attack_params.new_adversarial_value_candidate_count}) was greater than the default limit for new adversarial value candidate count ({attack_params.max_new_adversarial_value_candidate_count}). Both values will be set to {attack_params.new_adversarial_value_candidate_count}.")
            attack_params.max_new_adversarial_value_candidate_count = attack_params.new_adversarial_value_candidate_count

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
            logger.critical("--adversarial-candidate-filter-tokens-min must be a positive integer.")
            sys.exit(1)
        attack_params.candidate_filter_tokens_min = args.adversarial_candidate_filter_tokens_min
    
    if not isinstance(args.adversarial_candidate_filter_tokens_max, type(None)):
        if args.adversarial_candidate_filter_tokens_max < 1:
            logger.critical("--adversarial-candidate-filter-tokens-max must be a positive integer.")
            sys.exit(1)
        attack_params.candidate_filter_tokens_max= args.adversarial_candidate_filter_tokens_max
    
    attack_params.attempt_to_keep_token_count_consistent = args.attempt_to_keep_token_count_consistent
    
    attack_params.add_token_when_no_candidates_returned = args.add_token_when_no_candidates_returned
    
    attack_params.delete_token_when_no_candidates_returned = args.delete_token_when_no_candidates_returned
    
    if args.random_seed_comparisons < 0 or args.random_seed_comparisons > 16381:
        logger.critical("--args-random-seed-comparisons must specify a value between 0 and 16381.")
        sys.exit(1)
    attack_params.random_seed_comparisons = args.random_seed_comparisons
    if attack_params.model_temperature_range_end < attack_params.model_temperature_range_begin:
        logger.critical("The values specified for --temperature-range must be floating point numbers, and the second number must be greater than the first.")
        sys.exit(1)
    if attack_params.random_seed_comparisons > 0 and attack_params.model_temperature_range_end <= 1.0:
        logger.critical("--args-random-seed-comparisons can only be used if --temperature is set to a floating-point value greater than 1.0 or --temperature-range is used to set a range that ends at a value greater than 1.0, because otherwise the seed values will be ignored.")
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
            logger.critical("--adversarial-candidate-repetitive-line-limit must be a positive integer.")
            sys.exit(1)
        attack_params.candidate_filter_repetitive_lines = args.adversarial_candidate_repetitive_line_limit
        
    if not isinstance(args.adversarial_candidate_repetitive_token_limit, type(None)):
        if args.adversarial_candidate_repetitive_token_limit < 1:
            logger.critical("--adversarial-candidate-repetitive-token-limit must be a positive integer.")
            sys.exit(1)
        attack_params.candidate_filter_repetitive_tokens = args.adversarial_candidate_repetitive_token_limit
    
    if not isinstance(args.adversarial_candidate_newline_limit, type(None)):
        if args.adversarial_candidate_newline_limit < 0:
            logger.critical("--adversarial-candidate-newline-limit must be an integer greater than or equal to 0.")
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
            logger.critical(f"Error loading jailbreak detection rules from file '{rules_file}', content '{rules_file_content}': {e}\n{traceback.format_exc()}")
            sys.exit(1)
    else:
        attack_params.jailbreak_detection_rule_set = LLMJailbreakDetectorRuleSet.get_default_rule_set()
    
    if args.write_jailbreak_detection_rules_file:
        rules_output_file = os.path.abspath(args.write_jailbreak_detection_rules_file)
        verify_output_file_capability(rules_output_file, attack_params.overwrite_output)
        try:
            rules_data = attack_params.jailbreak_detection_rule_set.to_dict()
            json_rules_data = json.dumps(rules_data, indent=4)
            safely_write_text_output_file(rules_output_file, json_rules_data)
            logger.info(f"Wrote jailbreak detection rules to file '{rules_output_file}'.")
            sys.exit(0)
        except Exception as e:
            logger.critical(f"Error writing jailbreak detection rules to file '{rules_output_file}': {e}\n{traceback.format_exc()}")
            sys.exit(1)
    
    attack_params.verbose_self_test_output = args.verbose_self_test_output
    
    attack_params.ignore_jailbreak_self_tests = args.ignore_jailbreak_self_tests
    
    if args.model_parameter_info:
        attack_params.verbose_model_parameter_info = True
    
    attack_params.verbose_resource_info = args.verbose_resource_info
    
    attack_params.verbose_statistics = args.verbose_stats

    attack_params.break_on_success = args.break_on_success
    
    attack_params.rollback_on_loss_increase = args.rollback_on_loss_increase
    
    if not isinstance(args.required_loss_threshold, type(None)):
        attack_params.required_loss_threshold = args.required_loss_threshold

    if not isinstance(args.loss_threshold_max_attempts, type(None)):
        attack_params.loss_threshold_max_attempts = args.loss_threshold_max_attempts

    if not isinstance(args.required_loss_threshold, type(None)) and isinstance(args.loss_threshold_max_attempts, type(None)):
        logger.warning(f"--required-loss-threshold was specified without also specifying a maximum number of attempts using --loss-threshold-max-attempts. This will cause Broken Hill to potentially loop forever if it cannot find a value that meets the specified loss threshold.")

    if not isinstance(args.loss_threshold_max_attempts, type(None)) and isinstance(args.required_loss_threshold, type(None)):
        logger.warning(f"--loss-threshold-max-attempts was specified without also specifying --required-loss-threshold. --loss-threshold-max-attempts has no effect unless a threshold is also specified using --required-loss-threshold.")

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
    
    if args.suppress_attention_mask:
        attack_params.use_attention_mask = False
    
    attack_params.low_cpu_mem_usage = args.low_cpu_mem_usage
    
    if args.use_cache:
        attack_params.use_cache = True
        if args.no_torch_cache:
            logger.critical(f"Only one of --use-cache and --no-torch-cache may be specified.")
            sys.exit(1)
    
    if args.no_torch_cache:
        attack_params.use_cache = False
    
    if args.preserve_gradient:
        attack_params.preserve_gradient = True

    attack_params.display_model_size = args.display_model_size
    
    attack_params.force_python_tokenizer = args.force_python_tokenizer

    if args.enable_hardcoded_tokenizer_workarounds:
        attack_params.enable_hardcoded_tokenizer_workarounds = True
        
    if args.force_qwen_workaround:
        attack_params.force_qwen_dtype_workaround_check = True
        
    if args.missing_pad_token_replacement == "default":
        attack_params.missing_pad_token_replacement = None
    else:
        attack_params.missing_pad_token_replacement = args.missing_pad_token_replacement

    if args.padding_side:
        if args.padding_side == "default":
            attack_params.padding_side = "default"
        if args.padding_side == "none":
            attack_params.padding_side = None
        else:
            attack_params.padding_side = args.padding_side

    if args.exclude_whitespace_tokens:
        attack_params.exclude_whitespace_tokens = True

    if args.exclude_language_names_except is not None:
        attack_params.exclude_language_names_except = args.exclude_language_names_except

    if args.exclude_slur_tokens:
        attack_params.exclude_slur_tokens = True

    if args.exclude_profanity_tokens:
        attack_params.exclude_profanity_tokens = True

    if args.exclude_other_offensive_tokens:
        attack_params.exclude_other_offensive_tokens = True
                
    if args.exclude_token:
        for elem in args.exclude_token:
            for et in elem:
                if et.strip() != "":
                    attack_params.individually_specified_not_allowed_token_list = add_value_to_list_if_not_already_present(attack_params.not_allowed_token_list, et)

    if args.excluded_tokens_from_file:
        excluded_token_file = os.path.abspath(args.excluded_tokens_from_file)
        excluded_token_file_content = get_file_content(excluded_token_file, failure_is_critical = True)
        for l in excluded_token_file_content.splitlines():
            attack_params.individually_specified_not_allowed_token_list = add_value_to_list_if_not_already_present(attack_params.not_allowed_token_list, l.strip())

    if args.excluded_tokens_from_file_case_insensitive:
        excluded_token_file = os.path.abspath(args.excluded_tokens_from_file_case_insensitive)
        excluded_token_file_content = get_file_content(excluded_token_file, failure_is_critical = True)
        for l in excluded_token_file_content.splitlines():
            attack_params.individually_specified_not_allowed_token_list_case_insensitive = add_value_to_list_if_not_already_present(attack_params.individually_specified_not_allowed_token_list_case_insensitive, l.strip())

    
    if attack_params.operating_mode in [ BrokenHillMode.GCG_ATTACK, BrokenHillMode.GCG_ATTACK_SELF_TEST ]:
        if (not attack_params.base_prompt) or (not attack_params.target_output):
            logger.critical(f"A base prompt and a target must be specified, either as distinct values, or using the --auto-target option to set both.")
            sys.exit(1)

        if attack_params.model_path is None:
            if args.model is None:
                logger.critical("The --model option is required when performing the selected operation")
                sys.exit(1)
            attack_params.model_path = os.path.abspath(args.model)

        if not os.path.isdir(attack_params.model_path):
            logger.critical(f"The specified model directory ('{attack_params.model_path}') does not appear to exist.")
            sys.exit(1)

    # determine if any arbitrary code execution is possible during model load and handle accordingly
    if attack_params.peft_adapter_path is not None:
        if not os.path.isfile(os.path.join(attack_params.peft_adapter_path, SAFETENSORS_WEIGHTS_FILE_NAME)):
            peft_message = f"The specified PEFT adapter directory ('{attack_params.peft_adapter_path}') does not contain a safe tensors file ('{SAFETENSORS_WEIGHTS_FILE_NAME}'). Because the model weights are only available in Python 'pickle' format, loading the adapter could result in arbitrary code execution on your system. "
            if attack_params.load_options_trust_remote_code:
                peft_message += " The --trust-remote-code option was specified, so the model will be loaded."
                logger.warning(f"{peft_message}")
            else:
                peft_message += " If you trust the specified adapter and understand the implications of potentially running untrusted Python code, add the --trust-remote-code option to load the adapter."
                logger.critical(f"{peft_message}")
                sys.exit(1)

    if args.only_write_files_on_completion:
        attack_params.write_output_every_iteration = False
        logger.warning(f"Output files will only be written to persistent storage at the end of the attack, or if the attack is interrupted. ALL OF YOUR PROGRESS WILL BE LOST if this attack fails in a way that causes Broken Hill to exit immediately, such as loss of power or a severe unhandled exception.")

    # set up state-saving parameters
    # TKTK: one resuming from a state file is implemented, wrap this entire section a few check:
    # If a state file is specified using --load-state set attack_params.state_directory using the basename of that path *unless* --state-directory is also specified, in which case that takes precedence
    # Always create a new value for attack_params.state_file, *unless* --overwrite-existing-state is also specified
    if args.disable_state_backup:
        attack_params.save_state = False
        logger.warning(f"Attack state backup has been disabled. If this attack fails or is interrupted for any reason, it will need to be restarted from the beginning.")
    
    if args.delete_state_on_completion:
        attack_params.delete_state_on_completion = True
        logger.warning(f"The option to delete the attack state backup on successful completion has been enabled. Using this option is strongly discouraged because of the potential to accidentally delete useful data.")
    
    set_default_state_directory = False
    if args.state_directory:
        attack_params.state_directory = os.path.abspath(args.state_directory)
        if args.state_file:
            logger.critical(f"Only one of --state-directory and --state-file may be specified.")
            sys.exit(1)
    else:
        if attack_params.state_directory is None:
            set_default_state_directory = True
        else:
            # In case the state directory value is being loaded from a saved state
            if not os.path.isdir(attack_params.state_directory):
                set_default_state_directory = True
        if set_default_state_directory:
            attack_params.state_directory = os.path.abspath(os.path.join(pathlib.Path.home(), attack_params.default_state_directory))

    if args.load_state:
        attack_params.load_state_from_file = os.path.abspath(args.load_state)
        
        # clear the existing state file name from the loaded state
        attack_params.state_file = None
        
        existing_state_directory = os.path.dirname(attack_params.load_state_from_file)
        different_state_directories = False
        if attack_params.state_directory is not None:
            if existing_state_directory != attack_params.state_directory:
                different_state_directories = True
        if different_state_directories:
            logger.warning(f"The option to load an existing attack state from '{attack_params.load_state_from_file}' was specified. Ordinarily, Broken Hill would create a new state file in the directory '{existing_state_directory}'. However, the operator has explicitly specified the --state-directory option, so the new state file will be created in the directory '{attack_params.state_directory}' instead.")

    if args.overwrite_existing_state:
        attack_params.overwrite_existing_state = True
        if not args.state_file:
            attack_params.state_file = attack_params.load_state_from_file
        logger.warning(f"The option to overwrite the existing attack state backup has been enabled. Using this option is strongly discouraged because of the potential to accidentally delete useful data.")

    if args.state_file:
        attack_params.state_file = os.path.abspath(args.state_file)
    # only perform this check if the operator is not loading the state from a file - otherwise it will be overwritten before being loaded
    if attack_params.load_state_from_file is None and attack_params.state_file is not None:
        overwrite_existing = False
        if attack_params.overwrite_output and attack_params.overwrite_existing_state:
            overwrite_existing = True
        verify_output_file_capability(attack_params.state_file, overwrite_existing, is_state_file = True)

    # verify ability to write to any remaining output files that have been specified

    if args.save_options:
        attack_params.operating_mode = BrokenHillMode.SAVE_OPTIONS
        attack_params.save_options_path = os.path.abspath(args.save_options)
    if attack_params.save_options_path is not None:
        verify_output_file_capability(attack_params.save_options_path, attack_params.overwrite_output)

    if args.json_output_file:
        attack_params.json_output_file = os.path.abspath(args.json_output_file)
    if attack_params.json_output_file is not None:
        verify_output_file_capability(attack_params.json_output_file, attack_params.overwrite_output)
    
    if args.performance_output_file:
        attack_params.performance_stats_output_file = os.path.abspath(args.performance_output_file)
    if attack_params.performance_stats_output_file is not None:
        verify_output_file_capability(attack_params.performance_stats_output_file, attack_params.overwrite_output)

    if args.torch_cuda_memory_history_file:
        attack_params.torch_cuda_memory_history_file = os.path.abspath(args.torch_cuda_memory_history_file)
    if attack_params.torch_cuda_memory_history_file is not None:
        verify_output_file_capability(attack_params.torch_cuda_memory_history_file, attack_params.overwrite_output)
    
    main(attack_params, log_manager)

