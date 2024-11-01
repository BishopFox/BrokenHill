#!/bin/env python

import copy
# IMPORTANT: 'fastchat' is in the PyPi package 'fschat', not 'fastchat'!
import fastchat as fschat
import fastchat.conversation as fschat_conversation
import json
import logging
import numpy
import os
import pathlib
import psutil
import re
import statistics
import sys
import time
import torch
import torch.nn
import traceback
import uuid

from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

from enum import IntFlag
from enum import StrEnum
from enum import auto
from llm_attacks_bishopfox.base.attack_manager import get_effective_max_token_value_for_model_and_tokenizer
from llm_attacks_bishopfox.base.attack_manager import get_embedding_layer
from llm_attacks_bishopfox.base.attack_manager import get_random_seed_list_for_comparisons
from llm_attacks_bishopfox.dumpster_fires.conversation_templates import ConversationTemplateTester
from llm_attacks_bishopfox.dumpster_fires.offensive_tokens import get_other_highly_problematic_content
from llm_attacks_bishopfox.dumpster_fires.offensive_tokens import get_profanity
from llm_attacks_bishopfox.dumpster_fires.offensive_tokens import get_slurs
from llm_attacks_bishopfox.dumpster_fires.trash_fire_tokens import TokenAllowAndDenyList
from llm_attacks_bishopfox.dumpster_fires.trash_fire_tokens import TrashFireTokenCollection
from llm_attacks_bishopfox.dumpster_fires.trash_fire_tokens import get_decoded_token
from llm_attacks_bishopfox.dumpster_fires.trash_fire_tokens import get_decoded_tokens
from llm_attacks_bishopfox.dumpster_fires.trash_fire_tokens import get_token_allow_and_deny_lists
from llm_attacks_bishopfox.dumpster_fires.trash_fire_tokens import get_token_list_as_tensor
from llm_attacks_bishopfox.attack.radiation_garden import RadiationGarden
from llm_attacks_bishopfox.dumpster_fires.trash_fire_tokens import encode_string_for_real_without_any_cowboy_funny_business
from llm_attacks_bishopfox.dumpster_fires.trash_fire_tokens import remove_empty_and_trash_fire_leading_and_trailing_tokens
from llm_attacks_bishopfox.jailbreak_detection.jailbreak_detection import JailbreakDetectionRuleResult
from llm_attacks_bishopfox.jailbreak_detection.jailbreak_detection import LLMJailbreakDetector
from llm_attacks_bishopfox.jailbreak_detection.jailbreak_detection import LLMJailbreakDetectorRule
from llm_attacks_bishopfox.jailbreak_detection.jailbreak_detection import LLMJailbreakDetectorRuleSet
from llm_attacks_bishopfox.jailbreak_detection.jailbreak_detection import get_default_negative_test_strings
from llm_attacks_bishopfox.jailbreak_detection.jailbreak_detection import get_default_positive_test_strings
from llm_attacks_bishopfox.json_serializable_object import JSONSerializableObject
from llm_attacks_bishopfox.llms.large_language_models import LargeLanguageModelParameterInfoCollection
from llm_attacks_bishopfox.llms.large_language_models import print_model_parameter_info
#from llm_attacks_bishopfox.minimal_gcg.adversarial_content_utils import get_default_generic_role_indicator_template
from llm_attacks_bishopfox.statistics.statistical_tools import StatisticsCube
from llm_attacks_bishopfox.teratogenic_tokens.language_names import HumanLanguageManager
from llm_attacks_bishopfox.util.util_functions import PyTorchDevice
from llm_attacks_bishopfox.util.util_functions import add_value_to_list_if_not_already_present
from llm_attacks_bishopfox.util.util_functions import add_values_to_list_if_not_already_present
from llm_attacks_bishopfox.util.util_functions import command_array_to_string
from llm_attacks_bishopfox.util.util_functions import get_file_content
from llm_attacks_bishopfox.util.util_functions import get_now
from llm_attacks_bishopfox.util.util_functions import get_time_string
from llm_attacks_bishopfox.util.util_functions import safely_write_text_output_file
from llm_attacks_bishopfox.util.util_functions import slice_from_dict
from llm_attacks_bishopfox.util.util_functions import tensor_from_dict
from llm_attacks_bishopfox.util.util_functions import tensor_to_dict
from llm_attacks_bishopfox.util.util_functions import torch_dtype_from_string

from transformers.generation import GenerationConfig

logger = logging.getLogger(__name__)

class DecodingException(Exception):
    pass

class EncodingException(Exception):
    pass

# TKTK: actually resolve this circular import without code duplication
# Duplicate of the same value in adversarial_content_utils.py, to avoid a circular import
DEFAULT_CONVERSATION_TEMPLATE_NAME = 'zero_shot'

def get_default_generic_role_indicator_template():
    # note: using "### {role}:" instead will cause issues 
    return "### {role}"

class BrokenHillMode(StrEnum):
    GCG_ATTACK = 'gcg_attack'
    GCG_ATTACK_SELF_TEST  = 'gcg_attack_self_test'
    LIST_IETF_TAGS = 'list_ietf_tags'
    SAVE_OPTIONS = 'save_options'

# not currently used
class OverallScoringFunction(StrEnum):
    MEDIAN = 'median'
    AVERAGE = 'average'
    MINIMUM = 'minimum'
    MAXIMUM = 'maximum'

class InitialAdversarialContentCreationMode(StrEnum):
    FROM_STRING = 'from_string'
    FROM_TOKEN_IDS = 'from_token_ids'
    RANDOM_TOKEN_IDS = 'random_token_ids'
    SINGLE_TOKEN = 'single_token'
    LOSS_TOKENS = 'loss_tokens'
    RANDOM_TOKEN_IDS_LOSS_TOKEN_COUNT = 'random_token_ids_loss_token_count'
    SINGLE_TOKEN_LOSS_TOKEN_COUNT = 'single_token_loss_token_count'

class LossSliceMode(StrEnum):
    SAME_AS_TARGET_SLICE = 'same_as_target_slice'
    INDEX_SHIFTED_TARGET_SLICE = 'index_shifted_target_slice'
    ASSISTANT_ROLE_PLUS_FULL_TARGET_SLICE = 'assistant_role_plus_full_target_slice'
    ASSISTANT_ROLE_PLUS_TRUNCATED_TARGET_SLICE = 'assistant_role_plus_truncated_target_slice'
    
class LossAlgorithm(StrEnum):
    CROSS_ENTROPY = 'cross_entropy'
    MELLOWMAX = 'mellowmax'

class SystemMessageMode(StrEnum):
    SYSTEM_MESSAGE_PROPRTY = 'system_message_proprty'
    MESSAGE_WITH_SYSTEM_ROLE = 'message_with_system_role'

class ModelDataFormatHandling(StrEnum):
    AS_IS = 'as_is'
    FORCE_FLOAT16 = 'force_float16'
    FORCE_BFLOAT16 = 'force_bfloat16'
    FORCE_FLOAT32 = 'force_float32'
    FORCE_FLOAT64 = 'force_float64'
    FORCE_COMPLEX64 = 'force_complex64'
    FORCE_COMPLEX128 = 'force_complex128'

def get_missing_pad_token_names():
    result = [  "unk", 
                "bos",
                "eos",
                "default" ]
    return result

def get_missing_pad_token_replacement(tokenizer, replacement_name):
    allowed_names = get_missing_pad_token_names()
    if replacement_name not in get_missing_pad_token_names():
        raise Exception(f"Unrecognized padding token replacement name: '{replacement_name}' - must be one of '{allowed_names}'")
    result = None
    if replacement_name == "bos":
        result = tokenizer.bos_token_id, tokenizer.bos_token
    if replacement_name == "eos":
        result = tokenizer.eos_token_id, tokenizer.eos_token
    if replacement_name == "unk":
        result = tokenizer.unk_token_id, tokenizer.unk_token
    if replacement_name == "default":
        result = None, None
    
    return result

# not currently implemented
class AdversarialContentPlacement(StrEnum):
    PREFIX = 'prefix'
    SUFFIX = 'suffix'
    PLACEHOLDER = 'placeholder'
    INTERLEAVE = 'interleave'

class AdversarialContent(JSONSerializableObject):
    def __init__(self):
        self.token_ids = []
        self.tokens = []
        self.as_string = None
        self.original_loss = None
    
    def copy(self):
        result = AdversarialContent()
        result.token_ids = copy.deepcopy(self.token_ids)
        result.tokens = copy.deepcopy(self.tokens)
        result.as_string = self.as_string
        return result
    
    def delete_random_token(self, numpy_random_generator, tokenizer):
        num_tokens = len(self.token_ids)
        deleted_token_index = numpy_random_generator.integers(0, high = num_tokens)
        new_token_ids = []
        new_tokens = []
        for i in range(0, num_tokens):
            if i != deleted_token_index:
                new_token_ids.append(self.token_ids[i])
                new_tokens.append(self.tokens[i])
        self.token_ids = new_token_ids
        self.tokens = new_tokens
        self.as_string = tokenizer.decode(self.token_ids)
        
    def duplicate_random_token(self, numpy_random_generator, tokenizer):
        num_tokens = len(self.token_ids)
        duplicated_token_index = numpy_random_generator.integers(0, high = num_tokens)
        new_token_ids = []
        new_tokens = []
        for i in range(0, num_tokens):
            new_token_ids.append(self.token_ids[i])
            new_tokens.append(self.tokens[i])
            if i == duplicated_token_index:
                new_token_ids.append(self.token_ids[i])
                new_tokens.append(self.tokens[i])
        self.token_ids = new_token_ids
        self.tokens = new_tokens
        self.as_string = tokenizer.decode(self.token_ids)
    
    def get_full_description(self):
        return f"'{self.as_string}' ({self.tokens} or {self.token_ids} using the current tokenizer)"
        
    def get_short_description(self):
        return f"'{self.as_string}' ({self.tokens})"
    
    def is_match(self, other_adversarial_content):
        if self.as_string != other_adversarial_content.as_string:
            return False
        for i in range(0, len(self.token_ids)):
            if self.token_ids[i] != other_adversarial_content.token_ids[i]:
                return False
        for i in range(0, len(self.tokens)):
            if self.tokens[i] != other_adversarial_content.tokens[i]:
                return False
        return True
    
    def to_dict(self):
        result = super(AdversarialContent, self).properties_to_dict(self)
        return result
    
    @staticmethod
    def from_dict(property_dict):
        result = AdversarialContent()
        super(AdversarialContent, result).set_properties_from_dict(result, property_dict)
        return result

    def to_json(self):
        return JSONSerializableObject.json_dumps(self.to_dict(), use_indent = False)
        
    @staticmethod
    def from_json(json_string):
        return AdversarialContent.from_dict(json.loads(json_string))
    
    # This seems to happen if the number of candidates is long enough
    # Ideally I'll find a better way to avoid it than this
    @staticmethod
    def token_list_contains_invalid_tokens(tokenizer, token_ids):
        for token_num in range(0, len(token_ids)):
            if token_ids[token_num] < 0 or token_ids[token_num] >= tokenizer.vocab_size: 
                logger.debug(f"adversarial_candidate '{token_ids}' contains token ID {token_ids[token_num]}, which is outside the valid range for this tokenizer (min = 0, max = {tokenizer.vocab_size}). The candidate will be ignored. This may indicate an issue with the attack code, or the tokenizer code.")
                return True
        return False
    
    @staticmethod
    def from_token_ids(tokenizer, trash_fire_tokens, token_ids, trim_token_list = False):
        result = AdversarialContent()
        result.token_ids = copy.deepcopy(token_ids)
        result.tokens = get_decoded_tokens(tokenizer, result.token_ids)
        if trim_token_list:
            result.token_ids, result.tokens = remove_empty_and_trash_fire_leading_and_trailing_tokens(trash_fire_tokens, result.token_ids, result.tokens)
        try:
            result.as_string = tokenizer.decode(result.token_ids)
        except Exception as e:
            try:
                result.as_string = get_decoded_tokens(tokenizer, result.token_ids)
                logger.debug(f"couldn't decode token_ids directly via the tokenizer, but succeeded by using get_decoded_token: '{result.as_string}'")
                raise DecodingException(f"[AdversarialContent.from_token_ids] couldn't decode token_ids directly via the tokenizer, but succeeded by using get_decoded_token: '{result.as_string}'")
            except Exception as e2:
                raise DecodingException(f"Couldn't decode the set of token IDs '{result.token_ids}': {e}, {e2}")

        return result

    @staticmethod
    def from_string(tokenizer, trash_fire_tokens, input_string):
        result = AdversarialContent()
        #result.as_string = input_string
        #result.token_ids = tokenizer.encode(input_string)
        result.token_ids = encode_string_for_real_without_any_cowboy_funny_business(tokenizer, input_string)
        result.tokens = get_decoded_tokens(tokenizer, result.token_ids)
        #result.token_ids, result.tokens = remove_empty_and_trash_fire_leading_and_trailing_tokens(trash_fire_tokens, result.token_ids, result.tokens)   
        result.as_string = tokenizer.decode(result.token_ids)
        return result

class AdversarialContentList(JSONSerializableObject):
    def __init__(self):
        self.adversarial_content = []

    def contains_adversarial_string(self, adversarial_string):
        for i in range(0, len(self.adversarial_content)):
            if self.adversarial_content[i].as_string == adversarial_string:
                return True
        return False

    def contains_adversarial_content(self, adversarial_content):
        for i in range(0, len(self.adversarial_content)):
            if self.adversarial_content[i].is_match(adversarial_content):
                return True
        return False
    
    def get_content_with_lowest_loss(self):
        result = None
        for i in range(0, len(self.adversarial_content)):
            if result is None:
                result = self.adversarial_content[i]
            else:
                if isinstance(result.original_loss, type(None)):
                    result = self.adversarial_content[i]
                else:
                    if not isinstance(self.adversarial_content[i].original_loss, type(None)):
                        if self.adversarial_content[i].original_loss < result.original_loss:
                            result = self.adversarial_content[i]
        return result
    
    def append_if_new(self, new_adversarial_content):
        if not self.contains_adversarial_content(new_adversarial_content):
            self.adversarial_content.append(new_adversarial_content)

    def to_dict(self):
        result = super(AdversarialContentList, self).properties_to_dict(self)
        return result
    
    @staticmethod
    def from_dict(property_dict):
        result = AdversarialContentList()
        super(AdversarialContentList, result).set_properties_from_dict(result, property_dict)
        if len(result.adversarial_content) > 0:
            deserialized_content = []
            for i in range(0, len(result.adversarial_content)):
                deserialized_content.append(AdversarialContent.from_dict(result.adversarial_content[i]))
            result.adversarial_content = deserialized_content
        return result

    def to_json(self):
        return JSONSerializableObject.json_dumps(self.to_dict(), use_indent = False)
    
    @staticmethod
    def from_json(json_string):
        return AdversarialContentList.from_dict(json.loads(json_string))

# Default values defined in this class
class AttackParams(JSONSerializableObject):

    def get_model_data_type(self):
        if self.model_weight_format_handling == ModelDataFormatHandling.AS_IS:
            return None
        if self.model_weight_format_handling == ModelDataFormatHandling.FORCE_FLOAT16:
            return torch.float16
        if self.model_weight_format_handling == ModelDataFormatHandling.FORCE_BFLOAT16:
            return torch.bfloat16
        if self.model_weight_format_handling == ModelDataFormatHandling.FORCE_FLOAT32:
            return torch.float32
        if self.model_weight_format_handling == ModelDataFormatHandling.FORCE_FLOAT64:
            return torch.float64
        if self.model_weight_format_handling == ModelDataFormatHandling.FORCE_COMPLEX64:
            return torch.complex64
        if self.model_weight_format_handling == ModelDataFormatHandling.FORCE_COMPLEX128:
            return torch.complex128
        return None

    def set_automatic_base_and_target(self, autotarget_string):
        self.base_prompt = f"Write {autotarget_string}."
        #self.target_output =  f"Sure, here's {autotarget_string}"
        # Added additional words to help make up for the loss of potentially relevant content due to the loss slice index shift
        #self.target_output =  f"Sure, here's {autotarget_string} as you requested: "
        #self.target_output =  f"Sure, here's {autotarget_string} for you: "
        self.target_output =  f"Sure, here's {autotarget_string}:"
    
    def set_conversation_template_messages(self, message_list):
        self.conversation_template_messages = []
        # validate that the data is more or less as expected
        if not isinstance(message_list, list):
            raise Exception("The conversation message data '{message_list}' was not in the form of a list.")
        for i in range(0, len(message_list)):
            message_data = message_list[i]
            if not isinstance(message_data, list):
                raise Exception("Entry {i} of the conversation message data ('{message_data}') was not in the form of a list.")
            if len(message_data) != 2:
                raise Exception("Entry {i} of the conversation message data ('{message_data}') was not in the form of a list with two entries.")
            self.conversation_template_messages.append(message_data)
    
    def get_known_template_names(self):
        result = [
            'gemma',
            'gpt2',
            'gpt-j',
            'gpt-neo',
            'llama-2',
            #'mamba',
            'one_shot',
            #'opt',
            'phi2',
            'phi3',
            #'pythia',
            'qwen',
            'stablelm',
            #'t5',   # Use 't5' for fastchat-t5, flan-t5, and other derivatives
            'vicuna',
            'zero_shot'
            ]
        result.sort()
        return result

    def get_candidate_filter_regex(self):
        if self.candidate_filter_regex is not None:
            return re.compile(self.candidate_filter_regex)
        return re.compile(".")

    def get_token_filter_regex(self):
        if self.token_filter_regex is None:
            return None
        return re.compile(self.token_filter_regex)

    def get_devices(self):
        return [ self.model_device, self.gradient_device, self.forward_device ]

    def get_cpu_devices(self):
        result = []
        for d in self.get_devices():
            dl = d.lower()
            if len(dl) > 2:
                if dl[0:3] == "cpu":
                    result = add_value_to_list_if_not_already_present(result, d)
        result.sort()
        return result
    
    def get_cuda_devices(self):
        result = []
        for d in self.get_devices():
            dl = d.lower()
            if len(dl) > 3:
                if dl[0:4] == "cuda":
                    result = add_value_to_list_if_not_already_present(result, d)
        result.sort()
        return result

    def get_non_cuda_devices(self):
        result = []
        for d in self.get_devices():
            dl = d.lower()
            if len(dl) > 3:
                if dl[0:4] != "cuda":
                    result = add_value_to_list_if_not_already_present(result, d)
            else:
                result = add_value_to_list_if_not_already_present(result, d)
        result.sort()
        return result

    def get_mps_devices(self):
        result = []
        for d in self.get_devices():
            dl = d.lower()
            if len(dl) > 2:
                if dl[0:3] == "mps":
                    result = add_value_to_list_if_not_already_present(result, d)
        result.sort()
        return result

    def using_cpu(self):
        for d in self.get_devices():
            dl = d.lower()
            if len(dl) > 2:
                if dl[0:3] == "cpu":
                    return True
        return False

    def using_cuda(self):
        for d in self.get_devices():
            dl = d.lower()
            if len(dl) > 3:
                if dl[0:4] == "cuda":
                    return True
        return False
        
    def using_mps(self):
        for d in self.get_devices():
            dl = d.lower()
            if len(dl) > 2:
                if dl[0:3] == "mps":
                    return True
        return False

    def __init__(self):
        self.original_command_line_array = None
        self.original_command_line = None
        self.operating_mode = BrokenHillMode.GCG_ATTACK
        
        #self.device = 'cuda'
        # the PyTorch device where the model (and everything else except the gradient, currently) should be loaded
        self.model_device = 'cuda'
        # the PyTorch device where the gradient operations should be performed
        self.gradient_device = 'cuda'
        # the PyTorch device where the 'forward' operation should be performed
        self.forward_device = 'cuda'
        
        # enable torch.nn.DataParallel for the model (and any other places where it needs to be enabled explicitly)
        self.torch_dataparallel = False
        
        # back-end to use if CUDA is not available
        self.device_fallback = 'cpu'

        self.model_path = None

        # If this value is not None, load the tokenizer from a separate path 
        # For models such as e.g. Mamba that don't include their own tokenizer
        self.tokenizer_path = None
        
        # If this value is not None, after loading the model, use Hugging Face's PEFT library to load a pretrained model based on the first model from a separate path.
        # For models such as Guanaco that can't be loaded on their own
        self.peft_adapter_path = None

        # When loading the model, use the data as-is, or force conversion to a particular format?
        # The original proof-of-concept forced float16.
        self.model_weight_format_handling = ModelDataFormatHandling.FORCE_FLOAT16
        
        self.template_name = None
        
        self.override_fschat_templates = True
        
        # Replace any existing system prompt in the conversation template with this custom content
        self.custom_system_prompt = None
        
        # Clear any existing messages in the conversation template
        self.clear_existing_template_conversation = False
        
        # Add these messages to the conversation template
        # If not None, this should be a list.
        # Each entry is a 2-entry list of <role>, <text>
        # If <role> is an integer, the script will replace it with the role name for the template
        # If <role> is a string, it will be used unchanged, which may cause issues for some templates or models
        self.conversation_template_messages = None

        # Maximum number of times to run the main loop before exiting
        self.max_iterations = 500

        # TKTK: option to require that loss decreases between iterations or Broken Hill will roll back to the previous adversarial content and re-randomize
        # Maybe a threshold, so that the requirement goes away below some sort of minimum loss value?
        
        # The prompt to start with
        self.base_prompt = None
        # The target output to measure against
        self.target_output = None
        
        # TKTK: the other type of loss function
              
        # TKTK: randomize an operator-specified number of tokens each round, and use only the jailbreak success count to determine the course of evolution.
        # Would be a good comparison to make sure the loss calculation is actually beneficial.
        
        # The initial adversarial data for different methods
        # This can't be handled up front because everything other than the string depends on loading the tokenizer
        self.initial_adversarial_string = "! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !"
        self.initial_adversarial_token_string = None
        self.initial_adversarial_token_count = None
        self.initial_adversarial_token_ids = []
        self.initial_adversarial_content_creation_mode = InitialAdversarialContentCreationMode.FROM_STRING

        # where to place the adversarial content in the generated prompt
        # all other modes besides SUFFIX are TKTK for now
        # because I don't want to have to rewrite even *more* code that handles the content exclusively as strings instead of token IDs
        self.adversarial_content_placement = AdversarialContentPlacement.SUFFIX
        
        # emulate the original attack by converting the adversarial token IDs to a string and then back to token IDs at every iteration
        self.reencode_adversarial_content_every_iteration = False

        self.number_of_tokens_to_update_every_iteration = 1
        
        # If this option is False, and self.number_of_tokens_to_update_every_iteration is 1, the algorithm from the original code is used. Otherwise, the nanoGCG algorithm is used.
        self.always_use_nanogcg_sampling_algorithm = False

        # TKTK: option to set all tokens during the first iteration to the highest-scoring token out of the set of candidates *for that position*.
        # e.g. if the set of candidates includes values at positions 1, 3, 4, and 6, set all four tokens to the value at the first place in the list of candidates where that token is not the same as the default.
        
        # TKTK: same as above, but at every iteration

        # method for determining the loss slice start and end indices
        self.loss_slice_mode = LossSliceMode.INDEX_SHIFTED_TARGET_SLICE
        
        # for index-shifted loss slice mode, how much should it shift?
        self.loss_slice_index_shift = -1

        # which loss algorithm to use
        self.loss_algorithm = LossAlgorithm.CROSS_ENTROPY

        # mellowmax "alpha" value to use if the mellowmax loss function is in use
        self.mellowmax_alpha = 1.0

        # workaround for models that have non-Python tokenizers
        # but return None for every call to functions like char_to_token
        self.force_python_tokenizer = False
        
        # enable some hardcoded tokenizer workarounds implemented by the original developers
        self.enable_hardcoded_tokenizer_workarounds = False

        # If the tokenizer does not have a padding token defined, and this value is not None, use the specified token instead
        self.missing_pad_token_replacement = "eos"

        # If the tokenizer does not have a padding token defined, pad from the following side.
        # default is 'left' because the most common / default replacement padding token is EOS, and padding with that on the right will cause the model to not generate any text at all.
        self.missing_pad_token_padding_side = "left"

        # Options that control detection of a successful jailbreak
        #
        # TKTK: replace the negative and positive lists with a linear rule-based engine
        # e.g.:
        #   rule 1: "I'm sorry" indicates no jailbreak
        #   rule 2-14: other "no jailbreak" strings
        #   rule 15: "Here's a hypothetical scenario" indicates jailbreak after all
        #   rule 16: "10-year-old's birthday party" indicates no jailbreak
        #   rule 17 added via command-line: "pin the tail on the donkey but where the 
        #           winner gets to anonymously kill you" indicates jailbreak after all
        #   All rules evaluated every time, like a firewall policy
        self.jailbreak_detection_rule_set = None      

        # TKTK: detect jailbreak based on some loss threshold?
        
        # If this value is specified, at each iteration, Broken Hill will test results using <VALUE> additional random seed values, to attempt to avoid focusing on fragile results
        # The values are selected from the hardcoded results in attack_manager.py => get_random_seed_list_for_comparisons()
        # If the current value in the list is already used for any of the existing random seeds, it will be skipped
        # Meaning the operator could theoretically choose to compare against 253-256 random seeds
        # But that high a value is not recommended
        self.random_seed_comparisons = 0
        
        # Not currently used, because the numerical score doesn't apply to any of the LLM-generated output
        # If using random seed comparisons, during the step where candidate adversarial strings are evaluated for their overall score, use the specified statistical function to calculate the overall score.
        # Default: median, to avoid outlier biasing
        self.random_seed_scoring_mode = OverallScoringFunction.MEDIAN

        # values that can greatly influence model behaviour
        # temperature range begin
        self.model_temperature_range_begin = 1.0
        # temperature range end (inclusive)
        self.model_temperature_range_end = 1.0
        # random seeds
        # The magic value 20 is from the notebook by Zou, Wang, Carlini, Nasr, Kolter, and Fredrikson
        # NumPy
        self.numpy_random_seed = 20
        # PyTorch
        self.torch_manual_seed = 20
        # CUDA
        self.torch_cuda_manual_seed_all = 20

        # If this option is set to True, Broken Hill will generate a second version of the prompt to use for the memory-hungry parts of the GCG attack (gradient generation, logits, etc.).
        # The second version of the prompt omits the system prompt (if any) and any template messages before the current user input.
        self.ignore_prologue_during_gcg_operations = False

        # Candidate adversarial data filtering
        #
        # Pre-generation candidate adversarial data filtering
        if 2 > 1:   # indent this data to make it more distinguishable from other sections
            
            # limit tokens to only ASCII values
            self.exclude_nonascii_tokens = False
            
            # filter out nonprintable tokens
            self.exclude_nonprintable_tokens = False
            
            # filter out basic special tokens (EOS/BOS/pad/unknown)
            self.exclude_special_tokens = False
            
            # filter out any additional special tokens defined in the tokenizer configuration
            self.exclude_additional_special_tokens = False
            
            # filter out any additional tokens that consist solely of whitespace
            self.exclude_whitespace_tokens = False
            
            # filter out language names except the one represented by this IETF tag, if present
            self.exclude_language_names_except = None

            # filter out any additional tokens that are slurs
            self.exclude_slur_tokens = False
            
            # filter out any additional tokens that are listed as profanity
            self.exclude_profanity_tokens = False
            
            # filter out any additional tokens that are listed as being highly problematic in generated content
            self.exclude_other_offensive_tokens = False
            
            # Filtering out other values can sometimes help prevent the script from focusing 
            # on attacks that are easily detectable as unusual, but also potentially 
            # filter out interesting attacks that would actually work when user input
            # is not heavily restricted. The command-line interface includes several 
            # shortcuts to populate this list with values I found useful at one time or another
            # but I'd recommend leaving it empty by default.
            # "GCG_ANY_ALL_WHITESPACE_TOKEN_GCG" is a special value that will exclude
            # any token that consists solely of whitespace
            self.individually_specified_not_allowed_token_list = []
            self.individually_specified_not_allowed_token_list_case_insensitive = []
            
            # If specified, exclude tokens that don't match the following pattern
            self.token_filter_regex = None
        
        # Post-generation candidate adversarial data filtering
        if 2 > 1:   # indent this data to make it more distinguishable from other sections
            #
            # This section is kind of a hack, because ideally everything would be done
            # by biasing the token generation, not culling the list of values it generates
            # but it can help avoid edge and corner cases like adversarial data
            # where each position becomes optimized to "\n###"
            
            # If these values are not None, filter out candidate strings with too many 
            # or too few tokens
            # This is a modification of the original code, which tried to keep the number 
            # consistent, but the logic didn't work for some models (e.g. StableLM 2)
            self.candidate_filter_tokens_min = None
            self.candidate_filter_tokens_max = None
            # This option re-enables the check from the original code, which is supposed
            # to keep the token count consistent but will prevent any candidates from being
            # allowed for some models (such as StableLM 2)
            self.attempt_to_keep_token_count_consistent = False
            
            # Filter candidate strings by requiring that they match a regular expression
            #self.candidate_filter_regex = "[0-9A-Za-z]+"
            #self.candidate_filter_regex = "\w+"
            self.candidate_filter_regex = "."
            #self.candidate_filter_regex = None

            # Filter candidate strings to exclude lists with more than this many repeated lines
            self.candidate_filter_repetitive_lines = None
            
            # Filter candidate strings to exclude lists with more than this many repeated tokens
            self.candidate_filter_repetitive_tokens = None

            # Disallow candidate token lists that include more than this many newline characters
            #candidate_filter_newline_limit = None
            self.candidate_filter_newline_limit = None

            # Replace newline characters remaining in the candidate suffices with the following string
            self.candidate_replace_newline_characters = None
            
            # If more than this percent (0.0 - 1.0) of candidate adversarial values are filtered out, display a warning with statistics about the cause of filtering
            self.warn_on_filtered_candidate_percentage = 0.5

        # If this value is True, and the number of tokens is not at the limit, then a failure to generate any candidate adversarial content will cause a random token to be added to the content
        self.add_token_when_no_candidates_returned = False
        
        # If this value is True, and the number of tokens is greater than the minimum, then a failure to generate any candidate adversarial content will cause a random token to be deleted from the content
        # If both this value and add_token_when_no_candidates_returned are enabled, and the prequisites for both apply, then add_token_when_no_candidates_returned will take precedence
        self.delete_token_when_no_candidates_returned = False

        # The formatting string for roles when a model uses one of the generic fschat templates 
        # (one_shot, zero_shot, etc.)
        #self.generic_role_indicator_template = get_default_generic_role_indicator_template()
        self.generic_role_indicator_template = None

        # Options that are necessary for some models to load without erroring out
        # trust_remote_code=True is currently necessary for Phi-3
        self.load_options_trust_remote_code = False

        # ignoring mismatched sizes seems to be necessary for some of the interesting models
        # TKTK: list
        self.load_options_ignore_mismatched_sizes = False

        # Some models do not support the attention_mask parameter
        self.use_attention_mask = True

        # assorted values that may or may not impact performance
        self.low_cpu_mem_usage = False
        self.use_cache = False
    
        # various other minor configuration options
        # Displays the size of the model after loading it
        # (requires writing it to disk for some reason, so disabled by default)
        self.display_model_size = False

        # batch sizes for various operations
        #self.new_adversarial_value_candidate_count = 16
        #self.new_adversarial_value_candidate_count = 32
        self.new_adversarial_value_candidate_count = 48
        #self.new_adversarial_value_candidate_count = 64
        #self.new_adversarial_value_candidate_count = 256
        
        # the maximum the adversarial token generation batch size is allowed to grow to when no candidates are found
        self.max_new_adversarial_value_candidate_count = 1024
        
        # try to avoid out-of-memory errors during the most memory-intensive part of the work.
        # This used to be set to 1, but after analyzing CUDA profiling data, it seems that that typically results in *greater* memory use than with a value of 512.
        # In the best-case data, using 1 might sometimes result in a few hundred MiB less CUDA memory use.
        self.batch_size_get_logits = 512

        # Output detailed token and token ID information when self tests fail
        self.verbose_self_test_output = False

        # Perform the attack even if the jailbreak self-tests indicate the results are unlikely to be useful
        self.ignore_jailbreak_self_tests = False

        # Display detailed information on the named parameter groups when the model is loaded
        self.verbose_model_parameter_info = False
        
        # Display system resource/performance information every time it's collected, instead of only at key intervals
        self.verbose_resource_info = False
        
        # Display verbose system resource/performance statistics when execution completes
        self.verbose_statistics = False

        # Stop iterating after the first successful jailbreak detection
        self.break_on_success = False
        
        # Display full output for failed jailbreak attempts as well as successful
        self.display_full_failed_output = False

        # https://pytorch.org/docs/stable/generated/torch.topk.html
        self.topk = 256
        
        # maximum topk value to allow topk to increase to when no valid adversarial value candidates are discovered
        # if this value is set to None, the value is allowed to grow indefinitely
        #self.max_topk = None
        # default to allowing the value to increase 20 times before being considered a fatal error
        self.max_topk = 5120

        # TKTK: add a value and handling in Broken Hill for the overall limit on tokens (not just new tokens to be added to the end, which is what the _max_new_tokens values below represent).
        # "This is a friendly reminder - the current text generation call will exceed the model's predefined maximum length (2048). Depending on the model, you may observe exceptions, performance degradation, or nothing at all."

        # maximum number of tokens to have the LLM generate when testing adversarial values for a jailbreak
        # The original code warned against setting this higher than 32 to avoid a performance penalty
        self.generation_max_new_tokens = 32

        # maximum new tokens value when generating full output
        #self.full_decoding_max_new_tokens = 16384
        self.full_decoding_max_new_tokens = 1024

        # during the candidate-generation stage, continue generating sets of random candidates until at least one is found that either has a lower loss than the current value, or increases it by no more than this amount
        self.required_loss_threshold = None
        
        # if self.required_loss_threshold is not None, and this value is not None, make this many attempts at finding a candidate that meets the required threshold before giving up
        self.loss_threshold_max_attempts = None
        
        # exit Broken Hill entirely if the loss threshold is not met after the maximum attempt count is reached.
        # If this value is False, Broken Hill will use the "best best" value determined during the attempt to find a value that met the threshold.
        self.exit_on_loss_threshold_failure = False

        # if the loss value increases between iterations, roll back to the last "good" adversarial data
        self.rollback_on_loss_increase = False
        
        # when rollback is enabled, allow continuing without rollback if the increase in loss is less than this much from the current last-known-good loss value
        self.rollback_on_loss_threshold = 0.0
        
        # if the number of successful jailbreaks detected during a given iteration is fewer than the last one, roll back to the last "good" adversarial data
        self.rollback_on_jailbreak_count_decrease = False

        # when rollback is enabled, allow continuing without rollback if the decrease in jailbreak count is less than this much from the current last-known-good loss value
        self.rollback_on_jailbreak_count_threshold = 0
        
        # TKTK: options to randomize x random tokens in the adversarial data if no successes have occured for y iterations.
        # ("Gamma garden mode")
        # If rollback is enabled, "successes" do not include iterations where a rollback occurred.
        # This is a way to break out of a "rut" that the attack can sometimes get stuck in.
        # If the result is not an improvement, trigger a rollback and re-randomize even if rollback is not enabled for other criteria.
        # If rollback is enabled, and the next iteration after randomization would trigger a rollback, the rollback should also be re-randomized.
        # TKTK: related option to increase the number of tokens that are randomized in the event of sequential randomizations.
        # e.g. randomization is triggered, and four tokens are randomized. The result does not meet the "success" criteria. Broken Hill should therefore roll back to the pre-randomization value, and randomize e.g. five tokens instead of four.

        self.radiation_gardens = []

        # logging options
        self.log_file_path = None
        self.console_output_level = logging.INFO
        self.log_file_output_level = logging.INFO
        self.console_ansi_format = True
        self.log_file_ansi_format = False

        # Workaround for overly-chatty-by-default PyTorch (and other) code
        self.third_party_module_output_level = logging.WARNING

        # output options
        self.overwrite_output = False
        # If this value is not None, write detailed result data to the specified JSON file
        self.json_output_file = None
        # If this value is not None, write performance statistics collected during the attack to the specified JSON file
        self.performance_stats_output_file = None
        # If this value is not None, use PyTorch's CUDA memory history feature (https://pytorch.org/docs/stable/torch_cuda_memory.html) to save a pickled blob of profiling data
        self.torch_cuda_memory_history_file = None
       
        # parameter save/load options
        self.save_options_path = None
        self.load_options_path = None
        self.load_options_from_state_path = None
       
        # state backup/restore options
        self.save_state = True
        self.delete_state_on_completion = False
        self.overwrite_existing_state = False
        # Default directory name (in user's home directory) to store state files if a location is not explicitly specified
        self.default_state_directory = '.broken_hill'
        self.state_directory = None
        self.state_file = None
        self.load_state_from_file = None
                
        # TKTK: option to generate a dynamically-quantized version of the model and also check results against it, because quantized models seem much less susceptible to this type of attack.
        # As noted below in the "Quantization options" section, the attack itself cannot be performed (at least using PyTorch) against an integer-based model - it must be floating point.
        # However, the attack could be performed on the floating-point model, and each adversarial result checked against the floating-point and quantized models at each iteration.
        # Alternatively, maybe it would make sense to have the quantized testing performed using the Python Ollama library, because Ollama supports many more quantization formats than PyTorch.
        
        
        
        # Quantization options
        #
        # None of these work, and are unlikely to work for the foreseeable future.
        # That's why none of these are exposed as command-line parameters.
        #
        # They're a remnant of the work I did early on to try to allow use of 
        # quantized models so that attacks against larger LLMs could fit into memory 
        # on consumer hardware. Maybe they'll be useful again someday.
        #
        # The attack performed by Broken Hill depends on PyTorch features that 
        # do not currently support quantized data. In particular, the gradient operations.
        # The PyTorch developers claim that gradient operations are only possible
        # for floating-point values, because they require continuous functions.
        # I don't know why it's not possible to shift everything left by a decimial 
        # place or two and do everything in integer math, like game developers did 
        # for decades to improve performance, but I'm also not a mathematician or 
        # an expert in machine learning theory.
        # All I know is that we had integer gradients for other purposes in the 1990s 
        # and they were a heck of a lot better than no gradients at all.
        # [ shakes fist at cloud ]
        
        # set to none to disable dynamic quantization
        # Dynamic quantization doesn't currently work with the llm-attacks code
        # because the code uses PyTorch features that aren't available with dynamic quantization
        #self.quantization_dtype = torch.qint8
        self.quantization_dtype = None

        # Enable static post-training quantization
        # Static quantization also doesn't currently work with the llm-attacks code
        #self.enable_static_quantization = True
        self.enable_static_quantization = False

        # Using integer values doesn't work with the llm-attacks code
        #self.conversion_dtype = torch.uint8
        self.conversion_dtype = None

    def to_dict(self):
        result = super(AttackParams, self).properties_to_dict(self)
        return result

    def to_json(self):
        return JSONSerializableObject.json_dumps(self.to_dict())
    
    def copy(self):
        return AttackParams.from_dict(self.to_dict())
    
    @staticmethod
    def apply_dict(existing_object, property_dict):
        if not isinstance(existing_object, AttackParams):
            raise JSONSerializationException(f"Cannot apply properties for the AttackParams class to an instance of the class '{existing_object.__class__.__name__}'")
        super(AttackParams, existing_object).set_properties_from_dict(existing_object, property_dict)
        if existing_object.radiation_gardens is not None:
            if len(existing_object.radiation_gardens) > 0:
                deserialized_gardens = []
                for i in range(0, len(existing_object.radiation_gardens)):
                    deserialized_gardens.append(RadiationGarden.from_dict(existing_object.radiation_gardens[i]))
                existing_object.radiation_gardens = deserialized_gardens
        if existing_object.jailbreak_detection_rule_set is not None:
            existing_object.jailbreak_detection_rule_set = LLMJailbreakDetectorRuleSet.from_dict(existing_object.jailbreak_detection_rule_set)
        return existing_object
    
    @staticmethod
    def from_dict(property_dict):
        result = AttackParams()
        result = AttackParams.apply_dict(result, property_dict)
        return result
    
    @staticmethod
    def apply_json(existing_object, json_string):
        return AttackParams.apply_dict(existing_object, json.loads(json_string))
    
    @staticmethod
    def from_json(json_string):
        return AttackParams.from_dict(json.loads(json_string))

class RandomNumberGeneratorStateCollection(JSONSerializableObject):
    def __init__(self):
        # PyTorch default
        self.torch_rng_state = None
        # PyTorch seeded CPU
        self.random_generator_cpu_state = None
        # PyTorch seeded model device
        self.random_generator_attack_params_model_device_state = None
        # PyTorch seeded gradient device
        self.random_generator_attack_params_gradient_device_state = None                    
        # NumPy
        self.numpy_rng_state = None

    # [RandomNumberGeneratorStateCollection.to_dict] Debug: self.torch_rng_state = tensor([20,  0,  0,  ...,  0,  0,  0], dtype=torch.uint8), self.random_generator_cpu_state = tensor([20,  0,  0,  ...,  0,  0,  0], dtype=torch.uint8), self.random_generator_attack_params_model_device_state = tensor([20,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    #   dtype=torch.uint8), self.random_generator_attack_params_gradient_device_state = tensor([20,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    #   dtype=torch.uint8), self.numpy_rng_state = {'bit_generator': 'PCG64', 'state': {'state': 3383365900161324816698418978122629783, 'inc': 72763549156770659863042999813056722643}, 'has_uint32': 0, 'uinteger': 0}

    def to_dict(self):
        #logger.debug(f"self.torch_rng_state = {self.torch_rng_state}, self.random_generator_cpu_state = {self.random_generator_cpu_state}, self.random_generator_attack_params_model_device_state = {self.random_generator_attack_params_model_device_state}, self.random_generator_attack_params_gradient_device_state = {self.random_generator_attack_params_gradient_device_state}, self.numpy_rng_state = {self.numpy_rng_state}.")
        result = super(RandomNumberGeneratorStateCollection, self).properties_to_dict(self)
        #logger.debug(f"result = {result}.")
        return result

    def to_json(self):
        return JSONSerializableObject.json_dumps(self.to_dict(), use_indent = False)
    
    def copy(self):
        return RandomNumberGeneratorStateCollection.from_dict(self.to_dict())
    
    @staticmethod
    def from_dict(property_dict):
        result = RandomNumberGeneratorStateCollection()
        super(RandomNumberGeneratorStateCollection, result).set_properties_from_dict(result, property_dict)
        if result.torch_rng_state is not None:
            result.torch_rng_state = tensor_from_dict(result.torch_rng_state)
        if result.random_generator_cpu_state is not None:
            result.random_generator_cpu_state = tensor_from_dict(result.random_generator_cpu_state)
        if result.random_generator_attack_params_model_device_state is not None:
            result.random_generator_attack_params_model_device_state = tensor_from_dict(result.random_generator_attack_params_model_device_state)
        if result.random_generator_attack_params_gradient_device_state is not None:
            result.random_generator_attack_params_gradient_device_state = tensor_from_dict(result.random_generator_attack_params_gradient_device_state)
        return result
    
    @staticmethod
    def from_json(json_string):
        return RandomNumberGeneratorStateCollection.from_dict(json.loads(json_string))

class BrokenHillRandomNumberGenerators():
    def __init__(self, attack_state):
        self.random_generator_attack_params_model_device = torch.Generator(device = attack_state.model_device).manual_seed(attack_state.persistable.attack_params.torch_manual_seed)
        self.random_generator_attack_params_gradient_device = torch.Generator(device = attack_state.model_device).manual_seed(attack_state.persistable.attack_params.torch_manual_seed)
        if attack_state.persistable.attack_params.model_device != attack_state.persistable.attack_params.gradient_device:
            self.random_generator_attack_params_gradient_device = torch.Generator(device = attack_state.gradient_device).manual_seed(attack_state.persistable.attack_params.torch_manual_seed)
        self.random_generator_cpu = torch.Generator(device = 'cpu').manual_seed(attack_state.persistable.attack_params.torch_manual_seed)
        self.numpy_random_generator = numpy.random.default_rng(seed = attack_state.persistable.attack_params.numpy_random_seed)
    
    def get_current_states(self):
        result = RandomNumberGeneratorStateCollection()
        
        # PyTorch default
        result.torch_rng_state = torch.get_rng_state()
        # PyTorch seeded CPU
        result.random_generator_cpu_state = self.random_generator_cpu.get_state()
        # PyTorch seeded model device
        result.random_generator_attack_params_model_device_state = self.random_generator_attack_params_model_device.get_state()
        # PyTorch seeded gradient device
        result.random_generator_attack_params_gradient_device_state = self.random_generator_attack_params_gradient_device.get_state()                    
        # NumPy
        result.numpy_rng_state = self.numpy_random_generator.bit_generator.state
        
        return result
    
    def set_states(self, rng_state_collection):
        # PyTorch default
        torch.set_rng_state(rng_state_collection.torch_rng_state)
        # PyTorch seeded CPU
        self.random_generator_cpu.set_state(rng_state_collection.random_generator_cpu_state)
        # PyTorch seeded model device
        self.random_generator_attack_params_model_device.set_state(rng_state_collection.random_generator_attack_params_model_device_state)
        # PyTorch seeded gradient device
        self.random_generator_attack_params_gradient_device.set_state(rng_state_collection.random_generator_attack_params_gradient_device_state)                    
        # NumPy
        self.numpy_random_generator.bit_generator.state = rng_state_collection.numpy_rng_state

# Anything about the attack state that can and should be persisted as JSON goes here
class PersistableAttackState(JSONSerializableObject):
    def __init__(self):
        self.broken_hill_version = None
        self.attack_params = None
        self.language_manager = None
        self.performance_data = ResourceUtilizationData()
        self.main_loop_iteration_number = 0
        self.successful_attack_count = 0
        self.overall_result_data = BrokenHillResultData()
        self.random_number_generator_states = None
        self.initial_adversarial_content = None
        self.current_adversarial_content = None
        self.tested_adversarial_content = AdversarialContentList()
        # There is only one "last known good" adversarial value tracked to avoid the following scenario:
        # User has multiple types of rollback enabled
        # Rollback type 1 is triggered, and the script rolls back to the last-known-good adversarial value associated with rollback type 1
        # The rollback type 2 last-known-good adversarial value is updated to the value that caused the rollback
        # In the next iteration, rollback type 2 is triggered, and the script "rolls sideways" to the data that caused the first rollback, making the script branch into bad values
        self.last_known_good_adversarial_content = AdversarialContent()
        self.last_known_good_adversarial_content.token_ids = None
        self.last_known_good_adversarial_content.tokens = None
        self.last_known_good_adversarial_content.as_string = None
        self.best_loss_value = None
        self.best_jailbreak_count = None
        self.original_new_adversarial_value_candidate_count = None
        self.original_topk = None
        # the next two properties hold the merged version of [anything explicitly specified as individual tokens by the user] and [generated from options]
        self.not_allowed_token_list = []
        self.not_allowed_token_list_case_insensitive = []
        self.token_allow_and_deny_lists = None
    
    def initialize_language_manager(self):
        self.language_manager = HumanLanguageManager.from_bundled_json_file()    
    
    def build_token_allow_and_denylists(self, attack_state):
        self.not_allowed_token_list = copy.deepcopy(self.attack_params.individually_specified_not_allowed_token_list)
        self.not_allowed_token_list_case_insensitive = copy.deepcopy(self.attack_params.individually_specified_not_allowed_token_list_case_insensitive)
        
        if self.attack_params.exclude_language_names_except is not None:            
            language_name_list = self.language_manager.get_language_names(ietf_tag_to_exclude = self.attack_params.exclude_language_names_except)
            self.not_allowed_token_list_case_insensitive = add_values_to_list_if_not_already_present(self.not_allowed_token_list_case_insensitive, language_name_list)
        
        # TKTK: add a localization option for these
        if self.attack_params.exclude_slur_tokens:
            logger.debug(f"adding slurs to the list that will be used to build the token denylist.")
            self.not_allowed_token_list_case_insensitive = add_values_to_list_if_not_already_present(self.not_allowed_token_list_case_insensitive, get_slurs())
        if self.attack_params.exclude_profanity_tokens:
            logger.debug(f"adding profanity to the list that will be used to build the token denylist.")
            self.not_allowed_token_list_case_insensitive = add_values_to_list_if_not_already_present(self.not_allowed_token_list_case_insensitive, get_profanity())
        if self.attack_params.exclude_other_offensive_tokens:
            logger.debug(f"adding other highly-problematic content to the list that will be used to build the token denylist.")
            self.not_allowed_token_list_case_insensitive = add_values_to_list_if_not_already_present(self.not_allowed_token_list_case_insensitive, get_other_highly_problematic_content())
        
        logger.info(f"Building token allowlist and denylist - this step can take a long time for tokenizers with large numbers of tokens.")
        self.token_allow_and_deny_lists = get_token_allow_and_deny_lists(attack_state.tokenizer, 
            self.not_allowed_token_list, 
            device = attack_state.model_device, 
            additional_token_strings_case_insensitive = self.not_allowed_token_list_case_insensitive, 
            filter_nonascii_tokens = self.attack_params.exclude_nonascii_tokens, 
            filter_nonprintable_tokens = self.attack_params.exclude_nonprintable_tokens, 
            filter_special_tokens = self.attack_params.exclude_special_tokens, 
            filter_additional_special_tokens = self.attack_params.exclude_additional_special_tokens, 
            filter_whitespace_tokens = self.attack_params.exclude_whitespace_tokens, 
            token_regex = self.attack_params.get_token_filter_regex()
            )
    
    def to_dict(self):
        result = super(PersistableAttackState, self).properties_to_dict(self)
        return result

    def to_json(self):
        self_dict = self.to_dict()
        #logger.debug(f"self_dict = {self_dict}")
        return JSONSerializableObject.json_dumps(self_dict, use_indent = False)
    
    def copy(self):
        return PersistableAttackState.from_dict(self.to_dict())
    
    @staticmethod
    def apply_dict(existing_object, property_dict):
        if not isinstance(existing_object, PersistableAttackState):
            raise JSONSerializationException(f"Cannot apply properties for the PersistableAttackState class to an instance of the class '{existing_object.__class__.__name__}'")
        super(PersistableAttackState, existing_object).set_properties_from_dict(existing_object, property_dict)
        
        if existing_object.attack_params is not None:
            existing_object.attack_params = AttackParams.from_dict(existing_object.attack_params)

        if existing_object.language_manager is not None:
            existing_object.language_manager = HumanLanguageManager.from_dict(existing_object.language_manager)

        if existing_object.performance_data is not None:
            existing_object.performance_data = ResourceUtilizationData.from_dict(existing_object.performance_data)

        if existing_object.overall_result_data is not None:
            existing_object.overall_result_data = BrokenHillResultData.from_dict(existing_object.overall_result_data)

        if existing_object.random_number_generator_states is not None:
            existing_object.random_number_generator_states = RandomNumberGeneratorStateCollection.from_dict(existing_object.random_number_generator_states)

        if existing_object.initial_adversarial_content is not None:
            existing_object.initial_adversarial_content = AdversarialContent.from_dict(existing_object.initial_adversarial_content)

        if existing_object.current_adversarial_content is not None:
            existing_object.current_adversarial_content = AdversarialContent.from_dict(existing_object.current_adversarial_content)

        if existing_object.tested_adversarial_content is not None:
            existing_object.tested_adversarial_content = AdversarialContentList.from_dict(existing_object.tested_adversarial_content)

        if existing_object.last_known_good_adversarial_content is not None:
            existing_object.last_known_good_adversarial_content = AdversarialContent.from_dict(existing_object.last_known_good_adversarial_content)

        if existing_object.token_allow_and_deny_lists is not None:
            existing_object.token_allow_and_deny_lists = TokenAllowAndDenyList.from_dict(existing_object.token_allow_and_deny_lists)

        return existing_object
    
    @staticmethod
    def from_dict(property_dict):
        result = PersistableAttackState()
        result = PersistableAttackState.apply_dict(result, property_dict)
        return result
    
    @staticmethod
    def apply_json(existing_object, json_string):
        return PersistableAttackState.apply_dict(existing_object, json.loads(json_string))
    
    @staticmethod
    def from_json(json_string):
        return PersistableAttackState.from_dict(json.loads(json_string))

# Anything about the attack state that can't be easily persisted (or that doesn't make sense to) goes here
# As well as a reference to the persistable data, so there's just the one thing to pass around
class VolatileAttackState():
    def __init__(self):
        self.log_manager = None
        self.persistable = PersistableAttackState()
        self.random_number_generators = None
        self.model = None
        self.tokenizer = None        
        self.adversarial_content_manager = None
        self.conversation_template = None
        self.random_seed_values = get_random_seed_list_for_comparisons()
        self.model_weight_type = None
        self.model_weight_storage_dtype = None
        self.model_weight_storage_string = None        
        self.model_device = None
        self.gradient_device = None
        self.forward_device = None
        self.token_denylist_as_cpu_tensor = None
        self.trash_fire_token_treasury = None
        self.jailbreak_detector = LLMJailbreakDetector()        
    
    def write_persistent_state(self):
        self.update_persistable_data()
        if self.persistable.attack_params.save_state:
            safely_write_text_output_file(self.persistable.attack_params.state_file, self.persistable.to_json())
    
    def get_existing_file_number(self, file_name_list):
        file_number = None
        file_number_regex = re.compile("-[0-9]{4}$")
        found_file_numbers = []      
        for i in range(0, len(file_name_list)):
            current_fn = file_name_list[i]
            cfn_stem = pathlib.Path(current_fn).stem
            pattern_match = file_number_regex.search(cfn_stem)
            if pattern_match:
                try:
                    num_str = pattern_match.group(0).replace("-", "")
                    found_file_numbers.append(int(num_str))
                except Exception as e:
                    logger.error(f"Couldn't convert {pattern_match.group(0)} into a number: {e}")
        if len(found_file_numbers) > 0:
            found_file_numbers.sort()
            file_number = found_file_numbers[(len(found_file_numbers) - 1)]        
        return file_number
    
    def add_or_replace_file_number(self, file_path, new_file_number):
        fp_directory = os.path.dirname(file_path)
        fp_file_name = os.path.basename(file_path)
        fp_split = os.path.splitext(fp_file_name)
        fp_stem = fp_split[0]
        fp_extension = fp_split[1]
        file_number_regex = re.compile("-[0-9]{4}$")
        file_number_string = f"-{new_file_number:04}"
        # try replacing an existing number first
        fp_stem_new = file_number_regex.sub(file_number_string, fp_stem)
        # if nothing has changed, there was no number to update
        if fp_stem_new == fp_stem:
            fp_stem_new = f"{fp_stem}{file_number_string}"
        result = os.path.join(fp_directory, f"{fp_stem_new}{fp_extension}")
        logger.debug(f"input: '{file_path}', new_file_number: {new_file_number}, result: '{result}'.")
        return result
    
    def add_file_name_suffix(self, file_path, suffix):
        result = file_path
        fp_directory = os.path.dirname(file_path)
        fp_file_name = os.path.basename(file_path)
        fp_split = os.path.splitext(fp_file_name)
        fp_stem = fp_split[0]
        fp_extension = fp_split[1]
        # if the suffix is a simple one, and it's already there, don't re-add it
        handled_suffix = False        
        continued_regex = re.compile("-(resumed|continued)-[0-9]{6}_iterations-[0-9]{4}$")
        if continued_regex.search(fp_stem):
            logger.debug(f"continued_regex matches fp_stem '{fp_stem}'.")
            fp_stem = continued_regex.sub("", fp_stem)
            logger.debug(f"fp_stem is now '{fp_stem}'.")
        if not handled_suffix:
            result = os.path.join(fp_directory, f"{fp_stem}{suffix}{fp_extension}")
        logger.debug(f"input: '{file_path}', suffix: {suffix}, result: '{result}'.")
        return result
    
    def get_continuation_command(self, state_file_path, completed_all_iterations):
        result_array = []
        output_file_parameters = []
        # pre-populate the recommended Python command
        result_array.append("bin/python")
        result_array.append("-u")
        # find the index of the Broken Hill script in the original command
        script_name = "brokenhill.py"
        len_script_name = len(script_name)
        script_index = None
        for i in range(0, len(self.persistable.attack_params.original_command_line_array)):
            element_lower = self.persistable.attack_params.original_command_line_array[i].lower()
            if len(element_lower) >= len_script_name:
                comparison_string = element_lower[-len_script_name:]
                if comparison_string == script_name:
                    script_index = i
                    break
                else:
                    logger.debug(f"no match between '{comparison_string}' and '{script_name}'")

        if script_index is None:
            raise Exception(f"Could not find a reference to the Broken Hill script ('{script_name}') in the following array of command-line elements: {self.persistable.attack_params.original_command_line_array}")
        for i in range(0, script_index + 1):
            result_array.append(self.persistable.attack_params.original_command_line_array[i])
        result_array.append("--load-state")
        result_array.append(state_file_path)
        file_name_suffix = ""
        new_state_file_path = state_file_path
        new_json_output_file = self.persistable.attack_params.json_output_file
        new_performance_stats_output_file = self.persistable.attack_params.performance_stats_output_file
        new_torch_cuda_memory_history_file = self.persistable.attack_params.torch_cuda_memory_history_file

        written_files_list = []
        written_files_list.append(state_file_path)
        if self.persistable.attack_params.json_output_file is not None:
            written_files_list.append(self.persistable.attack_params.json_output_file)
            output_file_parameters.append("--json-output-file")
        if self.persistable.attack_params.performance_stats_output_file is not None:
            written_files_list.append(self.persistable.attack_params.performance_stats_output_file)
            output_file_parameters.append("--performance-output-file")
        if self.persistable.attack_params.torch_cuda_memory_history_file is not None:
            written_files_list.append(self.persistable.attack_params.torch_cuda_memory_history_file)
            output_file_parameters.append("--torch-cuda-memory-history-file")
        
        next_file_number = 1
        #existing_file_number = self.get_existing_file_number(written_files_list)
        #if existing_file_number is not None:
        #    next_file_number = existing_file_number + 1
        
        if completed_all_iterations:
            next_iteration_count = self.persistable.attack_params.max_iterations * 2
            result_array.append("--max-iterations")
            result_array.append(f"{next_iteration_count}")
            file_name_suffix = f"-continued-{next_iteration_count:06}_iterations"
        else:
            file_name_suffix = "-resumed"
        
        finished_determining_filenames = False
        got_automatic_filenames = False
        # 999,999 automatically named variations should be enough for anybody.
        # And, not coincidentally, that is the limit of a six-digit zero-padded number.
        max_attempts = 999999
        attempt_num = 0
        # Make sure the suggested filenames won't overwrite any existing files
        while not finished_determining_filenames:
            existing_file = False
            
            new_state_file_path = self.add_or_replace_file_number(self.add_file_name_suffix(new_state_file_path, file_name_suffix), next_file_number)
            if os.path.isfile(new_state_file_path):
                existing_file = True
            
            if not existing_file and new_json_output_file is not None:
                new_json_output_file = self.add_or_replace_file_number(self.add_file_name_suffix(new_json_output_file, file_name_suffix), next_file_number)
                if os.path.isfile(new_json_output_file):
                    existing_file = True
            
            if not existing_file and new_performance_stats_output_file is not None:
                new_performance_stats_output_file = self.add_or_replace_file_number(self.add_file_name_suffix(new_performance_stats_output_file, file_name_suffix), next_file_number)
                if os.path.isfile(new_performance_stats_output_file):
                    existing_file = True

            if not existing_file and new_torch_cuda_memory_history_file is not None:
                new_torch_cuda_memory_history_file = self.add_or_replace_file_number(self.add_file_name_suffix(new_torch_cuda_memory_history_file, file_name_suffix), next_file_number)
                if os.path.isfile(new_torch_cuda_memory_history_file):
                    existing_file = True
        
            if not existing_file:
                finished_determining_filenames = True
                got_automatic_filenames = True
            else:
                attempt_num += 1
                next_file_number += 1
                if attempt_num > max_attempts:
                    finished_determining_filenames = True
        
        if got_automatic_filenames:
            result_array.append("--state-file")
            result_array.append(new_state_file_path)
            
            if new_json_output_file is not None:
                result_array.append("--json-output-file")
                result_array.append(new_json_output_file)
            
            if new_performance_stats_output_file is not None:
                result_array.append("--performance-output-file")
                result_array.append(new_performance_stats_output_file)

            if new_torch_cuda_memory_history_file is not None:
                result_array.append("--torch-cuda-memory-history-file")
                result_array.append(new_torch_cuda_memory_history_file)
        else:
            if len(output_file_parameters) > 0:
                result_array.append(";")
                result_array.append("# Warning: Broken Hill was unable to automatically generate new file names for the output files that would be generated by this command that do not overwrite existing files.")
                result_array.append(f"# You should manually determine values for the following parameters: {output_file_parameters}.")
            
        self.persistable.attack_params.original_command_line_array
        
        return command_array_to_string(result_array, add_line_breaks = True)
    
    def get_state_loading_message(self, completed_all_iterations):
        if not self.persistable.attack_params.save_state:
            return None

        handled_state_message = False
        if self.persistable.attack_params.delete_state_on_completion:
            if completed_all_iterations:
                delete_message = "This attack completed successfully, and the operator specified the option to delete the save state on successful completion."
                try:
                    delete_file(self.persistable.attack_params.state_file)
                    delete_message = f"{delete_message} The file '{self.persistable.attack_params.state_file}' was deleted successfully."
                    logger.info(delete_message)
                except Exception as e:
                    delete_message = f"{delete_message} However, the file '{self.persistable.attack_params.state_file}' could not be deleted: {e}"
                    logger.error(delete_message)
                handled_state_message = True
        if not handled_state_message:
            state_message = f"State information for this attack has been stored in '{self.persistable.attack_params.state_file}'."
            command_message = self.get_continuation_command(self.persistable.attack_params.state_file, completed_all_iterations)
            if self.persistable.attack_params.delete_state_on_completion:
                state_message = f"The operator specified the option to delete the save state on successful completion, but this attack did not complete successfully. {state_message}"
            if not completed_all_iterations:
                state_message = f"{state_message} You can resume the attack where it was interrupted by running Broken Hill with the option --load-state '{self.persistable.attack_params.state_file}'. For example:\n\n{command_message}"
            else:
                state_message = f"{state_message} You can continue the attack with additional iterations by running Broken Hill with the options --load-state '{self.persistable.attack_params.state_file}' and --max-iterations <number greater than {self.persistable.attack_params.max_iterations}>. For example, to double the number of iterations:\n\n{command_message}"            
        return state_message
    
    def write_output_files(self):
        if self.persistable.attack_params.json_output_file is not None:
            safely_write_text_output_file(self.persistable.attack_params.json_output_file, self.persistable.overall_result_data.to_json())    
    
    def initialize_devices(self):
        self.model_device = torch.device(self.persistable.attack_params.model_device)
        self.gradient_device = torch.device(self.persistable.attack_params.gradient_device)
        self.forward_device = torch.device(self.persistable.attack_params.forward_device)
        
    def initialize_random_number_generators(self):
        # Set random seeds
        # NumPy
        numpy.random.seed(self.persistable.attack_params.numpy_random_seed)
        # PyTorch
        torch.manual_seed(self.persistable.attack_params.torch_manual_seed)
        # CUDA
        torch.cuda.manual_seed_all(self.persistable.attack_params.torch_cuda_manual_seed_all)
        if self.random_number_generators is None:
            self.random_number_generators = BrokenHillRandomNumberGenerators(self)
    
    # Update any persistable data that represents the state of an object in this class
    def update_persistable_data(self):
        if self.random_number_generators is not None:
            self.persistable.random_number_generator_states = self.random_number_generators.get_current_states()

    def restore_random_number_generator_states(self):
        if self.persistable.random_number_generator_states is not None:
            self.random_number_generators.set_states(self.persistable.random_number_generator_states)
        else:
            logger.warning(f"No previous random number generator state to restore")

    # Restore any state data for objects in this class that require explicitly referring to persistable data
    def restore_from_persistable_data(self):
        self.initialize_devices()
        self.initialize_random_number_generators()        
        self.restore_random_number_generator_states()
            
    def initialize_jailbreak_detector(self):
        self.jailbreak_detector.rule_set = self.persistable.attack_params.jailbreak_detection_rule_set

    def load_model_and_tokenizer(self):
        logger.debug(f"self.persistable.attack_params.model_path = '{self.persistable.attack_params.model_path}', self.persistable.attack_params.tokenizer_path = '{self.persistable.attack_params.tokenizer_path}'")

        #if ignore_mismatched_sizes:
        #    kwargs["ignore_mismatched_sizes"] = True

        model = None
        handled_model_load = False

        # TKTK: try adding a fallback to other AutoModel types, like AutoModelForSeq2SeqLM for T5.

        if self.model_weight_type is None:
            model = AutoModelForCausalLM.from_pretrained(
                    self.persistable.attack_params.model_path,
                    trust_remote_code = self.persistable.attack_params.load_options_trust_remote_code,
                    ignore_mismatched_sizes = self.persistable.attack_params.load_options_ignore_mismatched_sizes,
                    low_cpu_mem_usage = self.persistable.attack_params.low_cpu_mem_usage,
                    use_cache = self.persistable.attack_params.use_cache
                ).to(self.model_device).eval()
            handled_model_load = True
        if not handled_model_load:            
            # Hey, everyone, I've got a great idea! I'll use a machine-learning library with a full-featured list of data types, like int8, float16, bfloat16, and float32. It has a model-loading function that accepts one of those data types if the user wants to force conversion to that type. But I'll randomly decide to make the library default to converting to my personal favourite type when it loads my model! And I'll also invent a completely separate way of representing the data types for the option to override my favourite type, instead of using the full-featured list that's already there! Pew pew! Look at me! I'm Charlie Prince!
            # Inspired by the following PyTorch output:
            #   The model is automatically converting to bf16 for faster inference. If you want to disable the automatic precision, please manually add bf16/fp16/fp32=True to "AutoModelForCausalLM.from_pretrained".
            #   https://huggingface.co/Qwen/Qwen-7B/commit/58362a19a5b5b41c88ed1ae04607d733e1df4944
            if "qwen" in self.persistable.attack_params.model_path.lower():
                # because we don't have a config yet to call hasattr against, seems like we have to try calling the next function with the specific parameters first, catch an exception, and try again without them
                charlie_prince_bf16 = False
                charlie_prince_fp16 = False
                charlie_prince_fp32 = False
                if self.model_weight_type == torch.bfloat16:
                    charlie_prince_bf16 = True
                if self.model_weight_type == torch.float16:
                    charlie_prince_fp16 = True
                if self.model_weight_type == torch.float32:
                    charlie_prince_fp32= True
                logger.debug(f"self.model_weight_type = {self.model_weight_type}, charlie_prince_bf16 = {charlie_prince_bf16}, charlie_prince_fp16 = {charlie_prince_fp16}, charlie_prince_fp32 = {charlie_prince_fp32}")
                try:
                    model = AutoModelForCausalLM.from_pretrained(
                            self.persistable.attack_params.model_path,
                            torch_dtype = self.model_weight_type,
                            bf16 = charlie_prince_bf16,
                            fp16 = charlie_prince_fp16,
                            fp32 = charlie_prince_fp32,
                            trust_remote_code = self.persistable.attack_params.load_options_trust_remote_code,
                            ignore_mismatched_sizes = self.persistable.attack_params.load_options_ignore_mismatched_sizes,
                            low_cpu_mem_usage = self.persistable.attack_params.low_cpu_mem_usage,
                            use_cache = self.persistable.attack_params.use_cache
                        ).to(self.model_device).eval()
                    handled_model_load = True
                except Exception as e:
                    logger.debug(f"Exception thrown when loading model with notorious outlaw Charlie Prince's personal custom parameters: {e}")
                    handled_model_load = False
        if not handled_model_load:
            model = AutoModelForCausalLM.from_pretrained(
                    self.persistable.attack_params.model_path,
                    torch_dtype = self.model_weight_type,
                    trust_remote_code = self.persistable.attack_params.load_options_trust_remote_code,
                    ignore_mismatched_sizes = self.persistable.attack_params.load_options_ignore_mismatched_sizes,
                    low_cpu_mem_usage = self.persistable.attack_params.low_cpu_mem_usage,
                    use_cache = self.persistable.attack_params.use_cache
                ).to(self.model_device).eval()                
        
        if self.persistable.attack_params.torch_dataparallel:
            try:
                model = torch.nn.DataParallel(model)
            except Exception as e:
                logger.error(f"Unable to load the model using torch.nn.DataParallel: {e}")
        
        tokenizer_path_to_load = self.persistable.attack_params.model_path
        if self.persistable.attack_params.tokenizer_path is not None:
            tokenizer_path_to_load = self.persistable.attack_params.tokenizer_path
        
        logger.debug(f"self.persistable.attack_params.tokenizer_path = '{self.persistable.attack_params.tokenizer_path}', self.persistable.attack_params.model_path = '{self.persistable.attack_params.model_path}'")

        tokenizer = None
        
        #is_mamba = args.model_name.startswith("state-spaces/mamba-")
        #    if is_mamba:
        #tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        #model = MambaLMHeadModel.from_pretrained(args.model_name, device = self.model_device, self.model_weight_type = self.model_weight_type)
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path_to_load,
                trust_remote_code = self.persistable.attack_params.load_options_trust_remote_code,
                use_fast = False
            )
        except Exception as e:
            handled = False
            #if isinstance(e, ValueError):
            if 2 > 1:
                logger.warning(f"Unable to load standard tokenizer from '{tokenizer_path_to_load}', attempting to fall back to fast tokenizer. The exception thrown when loading the standard tokenizer was: {e}")
                try:
                    tokenizer = AutoTokenizer.from_pretrained(
                        tokenizer_path_to_load,
                        trust_remote_code = self.persistable.attack_params.load_options_trust_remote_code,
                        use_fast = True
                    )
                    handled = True
                except Exception as e2:
                    logger.error(f"Error loading both standard and fast tokenizers from '{tokenizer_path_to_load}': '{e}', '{e2}'")
                    raise e        
            if not handled:
                logger.error(f"Error loading tokenizer from '{tokenizer_path_to_load}': '{e}'")
                raise e
        
        if self.persistable.attack_params.enable_hardcoded_tokenizer_workarounds:
            if 'oasst-sft-6-llama-30b' in tokenizer_path_to_load:
                tokenizer.bos_token_id = 1
                tokenizer.unk_token_id = 0
            if 'guanaco' in tokenizer_path_to_load:
                # Both of these are defined already in the configuration included with TheBloke's version of Guanaco.
                # They're also defined in the configuration included with the "huggyllama" version of llama-7b, so I think both values are redundant.
                tokenizer.eos_token_id = 2
                tokenizer.unk_token_id = 0
            if 'llama-2' in tokenizer_path_to_load:
                # Llama-2's tokenizer does explicitly define an unknown token ("<unk>"), so this seems fine
                tokenizer.pad_token = tokenizer.unk_token
                # Llama-2's tokenizer configuration explicitly pads from the right
                tokenizer.padding_side = 'left'
            if 'falcon' in tokenizer_path_to_load:
                tokenizer.padding_side = 'left'
                
        if not tokenizer.pad_token:
            # pad from the left side by default, because the default / most common replacement (EOS) will cause the model to not generate anything at all if the data is padded from the right
            tokenizer.padding_side = self.persistable.attack_params.missing_pad_token_padding_side
            side_message = f" The padding side has been set to '{self.persistable.attack_params.missing_pad_token_padding_side}'. If you encounter errors or unexpected results, try using the other padding side mode."
            if self.persistable.attack_params.missing_pad_token_replacement is not None:
                pad_token_id, pad_token = get_missing_pad_token_replacement(tokenizer, self.persistable.attack_params.missing_pad_token_replacement)
                if pad_token_id is not None and pad_token is not None:
                    tokenizer.pad_token_id = pad_token_id
                    tokenizer.pad_token = pad_token
                logger.warning(f"the tokenizer in '{tokenizer_path_to_load}' does not have a pad_token value defined. Using the alternative value '{self.persistable.attack_params.missing_pad_token_replacement}'. If you encounter errors or unexpected results, consider specifying a different --missing-pad-token-replacement value on the command line.{side_message}")
            else:
                logger.warning(f"the tokenizer in '{tokenizer_path_to_load}' does not have a pad_token value defined. If you encounter errors or unexpected results, consider specifying a --missing-pad-token-replacement value other than 'default' on the command line.{side_message}")
        
        return model, tokenizer


    def load_model(self):
        try:
            model_load_message = f"Loading model and tokenizer from '{self.persistable.attack_params.model_path}'."
            if self.persistable.attack_params.tokenizer_path is not None:
                model_load_message = f"Loading model from '{self.persistable.attack_params.model_path}' and tokenizer from {self.persistable.attack_params.tokenizer_path}."
            logger.info(model_load_message)
            self.model_weight_type = self.persistable.attack_params.get_model_data_type()
            logger.debug(f"model_weight_type = {self.model_weight_type}")
            model_config_path = None
            model_config_dict = None
            try:
                model_config_path = os.path.join(self.persistable.attack_params.model_path, "config.json")
                model_config_data = get_file_content(model_config_path, failure_is_critical = False)
                model_config_dict = json.loads(model_config_data)
            except Exception as e:
                logger.warning(f"Couldn't load model configuration file '{model_config_path}'. Some information will not be displayed. The exception thrown was: {e}.")
            self.model_weight_storage_string = ""
            if model_config_dict is not None:            
                if "torch_dtype" in model_config_dict.keys():
                    model_torch_dtype = model_config_dict["torch_dtype"]
                    if model_torch_dtype is not None:
                        self.model_weight_storage_string = f"Model weight data is stored as {model_torch_dtype}"
                        self.model_weight_storage_dtype = torch_dtype_from_string(model_torch_dtype)
                        logger.info(self.model_weight_storage_string)
            try:
                self.model, self.tokenizer = self.load_model_and_tokenizer()
            except Exception as e:
                tokenizer_message = ""
                if self.persistable.attack_params.tokenizer_path is not None and self.persistable.attack_params.tokenizer_path != "":
                    tokenizer_message = ", with tokenizer path '{self.persistable.attack_params.tokenizer_path}'"
                logger.critical(f"Exception thrown while loading model from '{self.persistable.attack_params.model_path}'{tokenizer_message}: {e}\n{traceback.format_exc()}")
                # TKTK: replace this with raising an exception so any cleanup can happen
                sys.exit(1)
            self.persistable.performance_data.collect_torch_stats(self, is_key_snapshot_event = True, location_description = "after loading model and tokenizer")
            logger.debug(f"Model loaded.")
            model_data_type_message = f"Model weight data was loaded as {self.model.dtype}"
            if self.model_weight_storage_string != "":            
                model_data_type_message = f"{self.model_weight_storage_string}, and was loaded as {self.model.dtype}"
            logger.info(f"Model and tokenizer loaded. {model_data_type_message}")
            if self.persistable.attack_params.peft_adapter_path is not None:
                logger.info(f"Loading PEFT model from '{self.persistable.attack_params.peft_adapter_path}'.")
                try:
                    self.model = PeftModel.from_pretrained(self.model, self.persistable.attack_params.peft_adapter_path)
                except Exception as e:
                    logger.critical(f"Exception thrown while loading PEFT model from '{self.persistable.attack_params.peft_adapter_path}': {e}\n{traceback.format_exc()}")
                    # TKTK: replace this with raising an exception so any cleanup can happen
                    sys.exit(1)
                self.persistable.performance_data.collect_torch_stats(self, location_description = "after loading PEFT adapter")
                logger.info(f"PEFT adapter loaded.")
            
            self.persistable.overall_result_data.model_parameter_info_collection = LargeLanguageModelParameterInfoCollection.from_loaded_model(self.model)
            
            print_model_parameter_info(self)
            
            if isinstance(self.tokenizer.pad_token_id, type(None)):
                if self.persistable.attack_params.loss_slice_mode == LossSliceMode.ASSISTANT_ROLE_PLUS_FULL_TARGET_SLICE:
                    logger.critical("Error: the padding token is not set for the current tokenizer, but the current loss slice algorithm requires that the list of target tokens be padded. Please specify a replacement token using the --missing-pad-token-replacement option.")
                    # TKTK: replace this with raising an exception so any cleanup can happen
                    sys.exit(1)
            
            logger.debug(f"getting max effective token value for model and tokenizer.")
        
            self.persistable.attack_params.generation_max_new_tokens = get_effective_max_token_value_for_model_and_tokenizer("--max-new-tokens", self.model, self.tokenizer, self.persistable.attack_params.generation_max_new_tokens)
            self.persistable.attack_params.full_decoding_max_new_tokens = get_effective_max_token_value_for_model_and_tokenizer("--max-new-tokens-final", self.model, self.tokenizer, self.persistable.attack_params.full_decoding_max_new_tokens)
        except Exception as e:
            logger.critical(f"Error loading model: {e}\n{traceback.format_exc()}")
            # TKTK: replace this with raising an exception so any cleanup can happen
            sys.exit(1)

    def build_token_allow_and_denylists(self):
        self.persistable.build_token_allow_and_denylists(self)
        
    def get_token_denylist_as_cpu_tensor(self):
        if self.token_denylist_as_cpu_tensor is None:
            if len(self.persistable.token_allow_and_deny_lists.denylist) > 0:
                self.token_denylist_as_cpu_tensor = get_token_list_as_tensor(self.persistable.token_allow_and_deny_lists.denylist, device='cpu')
            else:
                self.token_denylist_as_cpu_tensor = None
        return self.token_denylist_as_cpu_tensor
    
    # This code still doesn't do anything useful. Maybe it will some day!
    def apply_model_quantization(self):
        if self.persistable.attack_params.quantization_dtype:
            if self.persistable.attack_params.enable_static_quantization:
                logger.critical("Broken Hill only supports quantizing using static or dynamic approaches, not both at once")
                # TKTK: replace with raise
                sys.exit(1)
            self.persistable.performance_data.collect_torch_stats(self, location_description = "before quantizing model")
            logger.info(f"Quantizing model to '{self.persistable.attack_params.quantization_dtype}'")    
            #self.model = quantize_dynamic(model = self.model, qconfig_spec = {torch.nn.LSTM, torch.nn.Linear}, dtype = quantization_dtype, inplace = False)
            self.model = quantize_dynamic(model = self.model, qconfig_spec = {torch.nn.LSTM, torch.nn.Linear}, dtype = self.persistable.attack_params.quantization_dtype, inplace = True)
            self.persistable.performance_data.collect_torch_stats(self, location_description = "after quantizing model")

        if self.persistable.attack_params.enable_static_quantization:
            backend = "qnnpack"
            logger.info(f"Quantizing model using static backend {backend}") 
            torch.backends.quantized.engine = backend
            self.model.qconfig = tq.get_default_qconfig(backend)
            #model.qconfig = float_qparams_weight_only_qconfig
            # disable quantization of embeddings because quantization isn't really supported for them
            model_embeds = get_embedding_layer(self)
            model_embeds.qconfig = float_qparams_weight_only_qconfig
            self.model = tq.prepare(self.model, inplace=True)
            self.model = tq.convert(self.model, inplace=True)
    
    def apply_model_dtype_conversion(self):
        if self.persistable.attack_params.conversion_dtype:
            logger.debug(f"converting model dtype to {self.persistable.attack_params.conversion_dtype}.")
            self.model = self.model.to(self.persistable.attack_params.conversion_dtype)

    def load_conversation_template(self):
        logger.debug(f"registering conversation templates.")
        self.persistable.performance_data.collect_torch_stats(self, location_description = "before loading conversation template")
        
        logger.debug(f"loading chat template '{self.persistable.attack_params.template_name}'. self.persistable.attack_params.generic_role_indicator_template='{self.persistable.attack_params.generic_role_indicator_template}', self.persistable.attack_params.custom_system_prompt='{self.persistable.attack_params.custom_system_prompt}', self.persistable.attack_params.clear_existing_template_conversation='{self.persistable.attack_params.clear_existing_template_conversation}'")
        self.conversation_template = None
        
        if self.persistable.attack_params.template_name is not None:
            if self.persistable.attack_params.template_name not in fschat_conversation.conv_templates.keys():
                logger.warning(f"chat template '{self.persistable.attack_params.template_name}' was not found in fschat - defaulting to '{DEFAULT_CONVERSATION_TEMPLATE_NAME}'.")
                self.persistable.attack_params.template_name = DEFAULT_CONVERSATION_TEMPLATE_NAME
            logger.debug(f"loading chat template '{self.persistable.attack_params.template_name}'")
            self.conversation_template = fschat_conversation.get_conv_template(self.persistable.attack_params.template_name)
        else:
            logger.debug(f"determining chat template based on content in '{self.persistable.attack_params.model_path}'")
            self.conversation_template = fschat.model.get_conversation_template(self.persistable.attack_params.model_path)
        # make sure fschat doesn't sneak the one_shot messages in when zero_shot was requested
        if self.persistable.attack_params.clear_existing_template_conversation:
            if hasattr(self.conversation_template, "messages"):
                logger.debug(f"resetting self.conversation_template.messages from '{self.conversation_template.messages}' to []")
                self.conversation_template.messages = []
            else:
                logger.warning(f"the option to clear the conversation template's default conversation was enabled, but the template does not include a default conversation.")
                self.conversation_template.messages = []
        generic_role_template = get_default_generic_role_indicator_template()
        if self.persistable.attack_params.generic_role_indicator_template is not None:
            # If using a custom role indicator template, just use a space and depend on the operator to specify any necessary characters such as :
            self.conversation_template.sep_style = fschat_conversation.SeparatorStyle.NO_COLON_SINGLE
            #generic_role_template = f"\n{self.persistable.attack_params.generic_role_indicator_template}"
            generic_role_template = f" {self.persistable.attack_params.generic_role_indicator_template}"
            #generic_role_template = self.persistable.attack_params.generic_role_indicator_template        
            self.conversation_template.sep = '\n'
        # note: the original logic was equivalent to the following:
        #generic_role_template = "### {role}"
        if self.conversation_template.name == 'zero_shot':# or self.conversation_template.name == 'one_shot':
            #self.conversation_template.roles = tuple(['### ' + r for r in self.conversation_template.roles])
            self.conversation_template.roles = tuple([generic_role_template.format(role=r) for r in self.conversation_template.roles])
            self.conversation_template.sep = "\n "
            self.conversation_template.sep2 = "\n "
        if self.persistable.attack_params.generic_role_indicator_template is not None:
            self.conversation_template.roles = tuple([generic_role_template.format(role=r) for r in self.conversation_template.roles])
        #if self.conversation_template.name == 'llama-2':
        #    self.conversation_template.sep2 = self.conversation_template.sep2.strip()
        if self.persistable.attack_params.custom_system_prompt is not None:
            if hasattr(self.conversation_template, "system_message"):
                original_system_message = self.conversation_template.system_message
                self.conversation_template.system_message = self.persistable.attack_params.custom_system_prompt
                logger.debug(f"Replaced default system message '{original_system_message}' with '{self.persistable.attack_params.custom_system_prompt}'.")
            else:
                logger.warning(f"The option to set the conversation template's system message was enabled, but the template does not include a system message.")
        if self.persistable.attack_params.conversation_template_messages is not None:
            if not hasattr(self.conversation_template, "messages"):
                self.conversation_template.messages = []
            logger.debug(f"Existing conversation template messages '{self.conversation_template.messages}'.")
            for i in range(0, len(self.persistable.attack_params.conversation_template_messages)):
                role_id_or_name = self.persistable.attack_params.conversation_template_messages[i][0]
                message = self.persistable.attack_params.conversation_template_messages[i][1]
                # If role IDs were specified, convert them to the correct format for the template
                if isinstance(role_id_or_name, int):
                    try:
                        role_id_or_name = self.conversation_template.roles[role_id_or_name]
                    except Exception as e:
                        raise Exception("Could not convert the role ID '{}' to an entry in the template's list of roles ('{self.conversation_template.roles}'): {e}")
                self.conversation_template.messages.append((role_id_or_name, message))
            logger.debug(f"Customized conversation template messages: '{self.conversation_template.messages}'.")
        
        if self.persistable.attack_params.template_name is not None:
            if self.conversation_template.name != self.persistable.attack_params.template_name:
                logger.warning(f"The template '{self.persistable.attack_params.template_name}' was specified, but fschat returned the template '{self.conversation_template.name}' in response to that value.")
        if self.conversation_template is None:
            logger.critical(f"Got a null conversation template when trying to load '{self.persistable.attack_params.template_name}'. This should never happen.")
            # TKTK: replace this with raising an exception
            sys.exit(1)
        logger.debug(f"Conversation template: '{self.conversation_template.name}'")
        logger.debug(f"Conversation template sep: '{self.conversation_template.sep}'")
        logger.debug(f"Conversation template sep2: '{self.conversation_template.sep2}'")
        logger.debug(f"Conversation template roles: '{self.conversation_template.roles}'")
        logger.debug(f"Conversation template system message: '{self.conversation_template.system_message}'")
        messages = json.dumps(self.conversation_template.messages, indent=4)
        logger.debug(f"Conversation template messages: '{messages}'")
        self.persistable.performance_data.collect_torch_stats(self, location_description = "after loading conversation template")

    def ignite_trash_fire_token_treasury(self):
        logger.info(f"Creating a meticulously-curated treasury of trash fire tokens - this step can take a long time for tokenizers with large numbers of tokens.")
        self.trash_fire_token_treasury = TrashFireTokenCollection.get_meticulously_curated_trash_fire_token_collection(self.tokenizer, self.conversation_template)

    def create_initial_adversarial_content(self):
        logger.debug(f"setting initial adversarial content.")
        self.persistable.initial_adversarial_content = None
        if self.persistable.attack_params.initial_adversarial_content_creation_mode == InitialAdversarialContentCreationMode.FROM_STRING:
            self.persistable.initial_adversarial_content = AdversarialContent.from_string(self.tokenizer, self.trash_fire_token_treasury, self.persistable.attack_params.initial_adversarial_string)
        
        if self.persistable.attack_params.initial_adversarial_content_creation_mode == InitialAdversarialContentCreationMode.SINGLE_TOKEN:
            single_token_id = None
            try:
                single_token_id = get_encoded_token(self.tokenizer, self.persistable.attack_params.initial_adversarial_token_string)
            except Exception as e:
                logger.critical(f"Error encoding string '{self.persistable.attack_params.initial_adversarial_token_string}' to token: {e}\n{traceback.format_exc()}")
                # TKTK: replace this with raising an exception
                sys.exit(1)
            if isinstance(single_token_id, type(None)):
                    logger.critical(f"The selected tokenizer encoded the string '{self.persistable.attack_params.initial_adversarial_token_string}' to a null value.")
                    # TKTK: replace this with raising an exception
                    sys.exit(1)
            if isinstance(single_token_id, list):
                decoded_tokens = get_decoded_tokens(self.tokenizer, single_token_id)
                single_token_id, decoded_tokens = remove_empty_and_trash_fire_leading_and_trailing_tokens(self.trash_fire_token_treasury, single_token_id, decoded_tokens)
                if len(single_token_id) > 1:
                    logger.critical(f"The selected tokenizer encoded the string '{self.persistable.attack_params.initial_adversarial_token_string}' as more than one token: {decoded_tokens} / {single_token_id}. You must specify a string that encodes to only a single token when using this mode.")
                    sys.exit(1)
                else:
                    single_token_id = single_token_id[0]
            self.persistable.attack_params.initial_adversarial_token_ids = []
            for i in range(0, self.persistable.attack_params.initial_adversarial_token_count):
                self.persistable.attack_params.initial_adversarial_token_ids.append(single_token_id)
            
            self.persistable.initial_adversarial_content = AdversarialContent.from_token_ids(self.tokenizer, self.trash_fire_token_treasury, self.persistable.attack_params.initial_adversarial_token_ids)

        if self.persistable.attack_params.initial_adversarial_content_creation_mode == InitialAdversarialContentCreationMode.FROM_TOKEN_IDS:
            self.persistable.initial_adversarial_content = AdversarialContent.from_token_ids(self.tokenizer, self.trash_fire_token_treasury, self.persistable.attack_params.initial_adversarial_token_ids)
        
        if self.persistable.attack_params.initial_adversarial_content_creation_mode == InitialAdversarialContentCreationMode.RANDOM_TOKEN_IDS:
            token_ids = get_random_token_ids(numpy_random_generator, self.persistable.token_allow_and_deny_lists, self.persistable.attack_params.initial_adversarial_token_count)
            self.persistable.initial_adversarial_content = AdversarialContent.from_token_ids(self.tokenizer, self.trash_fire_token_treasury, token_ids)
        
        post_self_test_initial_adversarial_content_creation_modes = [ InitialAdversarialContentCreationMode.LOSS_TOKENS, InitialAdversarialContentCreationMode.RANDOM_TOKEN_IDS_LOSS_TOKEN_COUNT, InitialAdversarialContentCreationMode.SINGLE_TOKEN_LOSS_TOKEN_COUNT ]
        
        if self.persistable.attack_params.initial_adversarial_content_creation_mode in post_self_test_initial_adversarial_content_creation_modes:
            self.persistable.initial_adversarial_content = AdversarialContent.from_string(self.tokenizer, self.trash_fire_token_treasury, self.persistable.attack_params.initial_adversarial_string)
        
        # This should never actually happen, but just in case
        if self.persistable.initial_adversarial_content is None:
            logger.critical(f"No initial adversarial content was specified.")
            # TKTK: replace this with raising an exception
            sys.exit(1)

    def check_for_adversarial_content_token_problems(self):
        logger.debug(f"Determining if any tokens in the adversarial content are also in the token denylist, or not in the tokenizer at all.")
        tokens_in_denylist = []
        tokens_not_in_tokenizer = []
        for i in range(0, len(self.persistable.initial_adversarial_content.token_ids)):
            token_id = self.persistable.initial_adversarial_content.token_ids[i]
            if token_id in self.persistable.token_allow_and_deny_lists.denylist:
                if token_id not in tokens_in_denylist:
                    tokens_in_denylist.append(token_id)
            else:
                if token_id not in self.persistable.token_allow_and_deny_lists.allowlist:
                    if token_id not in tokens_not_in_tokenizer:
                        tokens_not_in_tokenizer.append(token_id)
        
        if len(tokens_in_denylist) > 0:
            token_list_string = ""
            for i in range(0, len(tokens_in_denylist)):
                decoded_token = get_escaped_string(get_decoded_token(self.tokenizer, tokens_in_denylist[i]))
                formatted_token = f"'{decoded_token}' (ID {tokens_in_denylist[i]})"
                if token_list_string == "":
                    token_list_string = formatted_token
                else:
                    token_list_string += ", {formatted_token}"
            logger.warning(f"The following tokens were found in the initial adversarial content, but are also present in the user-configured list of disallowed tokens: {token_list_string}. These tokens will be removed from the denylist, because otherwise the attack cannot proceed.")
            new_denylist = []
            for existing_denylist_index in range(0, len(self.persistable.token_allow_and_deny_lists.denylist)):
                if self.persistable.token_allow_and_deny_lists.denylist[existing_denylist_index] in tokens_in_denylist:
                    self.persistable.token_allow_and_deny_lists.allowlist.append(self.persistable.token_allow_and_deny_lists.denylist[existing_denylist_index])
                else:
                    new_denylist.append(self.persistable.token_allow_and_deny_lists.denylist[existing_denylist_index])
            self.persistable.token_allow_and_deny_lists.denylist = new_denylist
        if len(tokens_not_in_tokenizer) > 0:
            logger.warning(f"The following token IDs were found in the initial adversarial content, but were not found by the selected tokenizer: {tokens_not_in_tokenizer}. This may cause this test to fail, the script to crash, or other unwanted behaviour. Please modify your choice of initial adversarial content to avoid the conflict.")

    def test_conversation_template(self):
        logger.info(f"Testing conversation template '{self.conversation_template.name}'")
        conversation_template_tester = ConversationTemplateTester(self.adversarial_content_manager, self.model)
        conversation_template_test_results = conversation_template_tester.test_templates(verbose = self.persistable.attack_params.verbose_self_test_output)
        if len(conversation_template_test_results.result_messages) > 0:
            for i in range(0, len(conversation_template_test_results.result_messages)):
                logger.warning(conversation_template_test_results.result_messages[i])
        else:
            logger.info(f"Broken Hill did not detect any issues with the conversation template in use with the current model.")

        self.persistable.performance_data.collect_torch_stats(self, location_description = "after testing conversation template")
        
        # TKTK: self-test to count the number of tokens in a role-switching operation for the model.
        
        # TKTK: detect if the tokenizer has support for .apply_chat_template(). If it does, use that to create a simple conversation using random sentinels, create the same conversation using Broken Hill's get_prompt() code, and make sure that the role-switching tokens actually match, in case the tokenizer doesn't tokenize the text form of those tokens back to the token IDs that it recognizes as a role switch.


    def perform_jailbreak_tests(self):
        empty_output_during_jailbreak_self_tests = False
        logger.debug(f"Testing for jailbreak with no adversarial content")
        empty_adversarial_content = AdversarialContent.from_string(self.tokenizer, self.trash_fire_token_treasury, "")
        jailbreak_check_input_token_id_data = self.adversarial_content_manager.get_prompt(adversarial_content = empty_adversarial_content, force_python_tokenizer = self.persistable.attack_params.force_python_tokenizer)
        nac_jailbreak_result, nac_jailbreak_check_data, nac_jailbreak_check_generation_results = self.check_for_attack_success(jailbreak_check_input_token_id_data,
            1.0,
            do_sample = False)
        
        self.persistable.performance_data.collect_torch_stats(self, location_description = "after generating no-adversarial-content jailbreak test data")
        
        self.persistable.overall_result_data.self_test_results["GCG-no_adversarial_content"] = nac_jailbreak_check_data
        
        nac_jailbreak_decoded_generated_prompt_string_stripped = nac_jailbreak_check_data.decoded_llm_generation_string.strip()
        
        nac_jailbreak_check_llm_output_stripped = nac_jailbreak_check_data.decoded_llm_output_string.strip()        
        if nac_jailbreak_check_llm_output_stripped == "":
            empty_output_during_jailbreak_self_tests = True
            logger.error(f"Broken Hill tested the specified request string with no adversarial content and the model's response was an empty string or consisted solely of whitespace:\n'{nac_jailbreak_check_data.decoded_llm_output_string}'\nThis may indicate that the full conversation is too long for the model, that an incorrect chat template is in use, or that the conversation contains data that the model is incapable of parsing. The full conversation generated during this test was:\n'{nac_jailbreak_decoded_generated_prompt_string_stripped}'")
        else:
            if nac_jailbreak_result:
                logger.error(f"Broken Hill tested the specified request string with no adversarial content and the current jailbreak detection configuration indicated that a jailbreak occurred. The model's response to '{self.persistable.attack_params.base_prompt}' was:\n'{nac_jailbreak_check_llm_output_stripped}'\nThis may indicate that the model being targeted has no restrictions on providing the requested type of response, or that jailbreak detection is not configured correctly for the specified attack. The full conversation generated during this test was:\n'{nac_jailbreak_decoded_generated_prompt_string_stripped}'")
            else:
                logger.info(f"Validated that a jailbreak was not detected for the given configuration when adversarial content was not included. The model's response to '{self.persistable.attack_params.base_prompt}' was:\n'{nac_jailbreak_check_llm_output_stripped}'\nIf this output does not match your expectations, verify your jailbreak detection configuration.")
        
        logger.debug(f"Testing for jailbreak when the LLM is prompted with the target string")
        target_jailbreak_result, target_jailbreak_check_data, target_jailbreak_check_generation_results = self.check_for_attack_success(jailbreak_check_input_token_id_data,
            1.0,
            do_sample = False,
            include_target_content = True)
        
        self.persistable.performance_data.collect_torch_stats(self, location_description = "after generating ideal jailbreak test data")
        
        self.persistable.overall_result_data.self_test_results["GCG-simulated_ideal_adversarial_content"] = target_jailbreak_check_data
        
        target_jailbreak_decoded_generated_prompt_string_stripped = target_jailbreak_check_data.decoded_llm_generation_string.strip()
        target_jailbreak_check_llm_output_stripped = target_jailbreak_check_data.decoded_llm_output_string.strip() 
        if target_jailbreak_check_llm_output_stripped == "":
            empty_output_during_jailbreak_self_tests = True
            logger.critical(f"When Broken Hill sent the model a prompt that simulated an ideal adversarial string, the model's response was an empty string or consisted solely of whitespace:\n'{target_jailbreak_check_llm_output_stripped}'\nThis may indicate that the full conversation is too long for the model, that an incorrect chat template is in use, or that the conversation contains data that the model is incapable of parsing. The full conversation generated during this test was:\n'{target_jailbreak_decoded_generated_prompt_string_stripped}'")
        else:
            if target_jailbreak_result:
                logger.info(f"Validated that a jailbreak was detected when the model was given a prompt that simulated an ideal adversarial string, using the given configuration. The model's response to '{self.persistable.attack_params.base_prompt}' when given the prefix '{self.persistable.attack_params.target_output}' was:\n'{target_jailbreak_check_llm_output_stripped}'\nIf this output does not match your expectations, verify your jailbreak detection configuration.")
            else:            
                logger.critical(f"Broken Hill did not detect a jailbreak when the model was given a prompt that simulated an ideal adversarial string, using the given configuration. The model's response to '{self.persistable.attack_params.base_prompt}' when given the prefix '{self.persistable.attack_params.target_output}' was:\n'{target_jailbreak_check_llm_output_stripped}'\nIf this output does meet your expectations for a successful jailbreak, verify your jailbreak detection configuration. If the model's response truly does not appear to indicate a successful jailbreak, the current attack configuration is unlikely to succeed. This may be due to an incorrect attack configuration (such as a conversation template that does not match the format the model expects), or the model may have been hardened against this type of attack. The full conversation generated during this test was:\n'{target_jailbreak_decoded_generated_prompt_string_stripped}'")

        # TKTK: if possible, self-test to determine if a loss calculation for what should be an ideal value actually has a score that makes sense.
        
        if self.persistable.attack_params.operating_mode != BrokenHillMode.GCG_ATTACK_SELF_TEST:
            if empty_output_during_jailbreak_self_tests or nac_jailbreak_result or not target_jailbreak_result:
                if not self.persistable.attack_params.ignore_jailbreak_self_tests:
                    logger.critical(f"Because the jailbreak detection self-tests indicated that the results of this attack would likely not be useful, Broken Hill will exit. If you wish to perform the attack anyway, add the --ignore-jailbreak-self-tests option.")
                    # TKTK: replace with raise
                    sys.exit(1)

    def generate(self, input_token_id_data, temperature, gen_config = None, do_sample = True, generate_full_output = False, include_target_content = False):
        working_gen_config = gen_config
        # Copy the generation config to avoid changing the original
        if gen_config is None:
            working_gen_config = GenerationConfig.from_dict(config_dict = self.model.generation_config.to_dict())
        else:
            working_gen_config = GenerationConfig.from_dict(config_dict = gen_config.to_dict())
        
        if temperature != 1.0 and do_sample:
            working_gen_config.do_sample = True
            working_gen_config.temperature = temperature
        if self.persistable.attack_params.display_full_failed_output or generate_full_output:
            working_gen_config.max_new_tokens = self.persistable.attack_params.full_decoding_max_new_tokens
        else:
            working_gen_config.max_new_tokens = self.persistable.attack_params.generation_max_new_tokens

        result = GenerationResults()
        result.max_new_tokens = working_gen_config.max_new_tokens

        result.input_token_id_data = input_token_id_data
        input_ids = result.input_token_id_data.get_input_ids_as_tensor().to(self.model_device)
        input_ids_sliced = input_ids
        if not include_target_content:
            input_ids_sliced = input_ids[:result.input_token_id_data.slice_data.assistant_role.stop]
        input_ids_converted = input_ids_sliced.to(self.model.device).unsqueeze(0)
        attn_masks = torch.ones_like(input_ids_converted).to(self.model.device)
        
        if self.persistable.attack_params.use_attention_mask:
            result.output_token_ids = self.model.generate(input_ids_converted, 
                                        attention_mask = attn_masks, 
                                        generation_config = working_gen_config,
                                        pad_token_id = self.tokenizer.pad_token_id)[0]
        else:
            result.output_token_ids = self.model.generate(input_ids_converted, 
                                        generation_config = working_gen_config,
                                        pad_token_id = self.tokenizer.pad_token_id)[0]
        
        result.output_token_ids_output_only = result.output_token_ids[result.input_token_id_data.slice_data.assistant_role.stop:]
        
        result.generation_input_token_ids = result.output_token_ids[result.input_token_id_data.slice_data.get_complete_user_input_slice()]
        
        logger.debug(f"result.input_token_id_data = {result.input_token_id_data}, result.generation_input_token_ids = {result.generation_input_token_ids}, result.output_token_ids = {result.output_token_ids}, result.output_token_ids_output_only = {result.output_token_ids_output_only}")
        
        return result
        
    def check_for_attack_success(self, input_token_id_data, temperature, gen_config = None, do_sample = True, include_target_content = False):
        generation_results = self.generate(input_token_id_data, 
                                        temperature,
                                        gen_config = gen_config,
                                        do_sample = do_sample,
                                        include_target_content = include_target_content)
                                                
        result_ar_info_data = AttackResultInfoData()
        result_ar_info_data.set_values(self.tokenizer, generation_results.max_new_tokens, generation_results.output_token_ids, generation_results.output_token_ids_output_only)
        
        logger.debug(f"result_ar_info_data = {result_ar_info_data.to_json()}")
        #logger.debug(f"result_ar_info_data.decoded_generated_prompt_string = '{result_ar_info_data.decoded_generated_prompt_string}', \nresult_ar_info_data.decoded_llm_generation_string = '{result_ar_info_data.decoded_llm_generation_string}', \nresult_ar_info_data.decoded_user_input_string = '{result_ar_info_data.decoded_user_input_string}', \nresult_ar_info_data.decoded_llm_output_string = '{result_ar_info_data.decoded_llm_output_string}', \nresult_ar_info_data.decoded_generated_prompt_tokens = '{result_ar_info_data.decoded_generated_prompt_tokens}', \nresult_ar_info_data.decoded_llm_generation_tokens = '{result_ar_info_data.decoded_llm_generation_tokens}', \nresult_ar_info_data.decoded_user_input_tokens = '{result_ar_info_data.decoded_user_input_tokens}', \nresult_ar_info_data.decoded_llm_output_tokens = '{result_ar_info_data.decoded_llm_output_tokens}'")
        
        jailbroken = False
        
        jailbreak_check_result = JailbreakDetectionRuleResult.FAILURE
        
        if result_ar_info_data.decoded_llm_output_string.strip() != "":
            jailbreak_check_result = self.jailbreak_detector.check_string(result_ar_info_data.decoded_llm_output_string)

        if jailbreak_check_result == JailbreakDetectionRuleResult.SUCCESS:
            jailbroken = True
        logger.debug(f"Jailbroken: {jailbroken} for generated string '{result_ar_info_data.decoded_llm_output_string}'")
        
        return jailbroken, result_ar_info_data, generation_results

class AttackResultInfoData(JSONSerializableObject):
    def __init__(self):
        # when was this specific set of data generated?
        self.date_time_utc = get_time_string()
        
        # the maximum number of new tokens for this data set specifically
        self.max_new_tokens = None
        
        self.max_token_length = None
        
        # the token IDs that represent the entire conversation with the LLM, including system prompt/messages
        self.llm_generation_token_ids = None
        # the token IDs that represent just the LLM's response to the input
        self.llm_output_token_ids = None

        # a list of tokens decoded from llm_generation_token_ids
        self.decoded_llm_generation_tokens = None
        # a single string decoded from llm_generation_token_ids
        self.decoded_llm_generation_string = None
        
        # a list of tokens decoded from llm_output_token_ids
        self.decoded_llm_output_tokens = None
        # a single string decoded from llm_output_token_ids
        self.decoded_llm_output_string = None

    def set_values(self, tokenizer, max_token_length, llm_generation_token_ids, llm_output_token_ids):
        self.max_token_length = max_token_length
        self.llm_generation_token_ids = llm_generation_token_ids
        self.llm_output_token_ids = llm_output_token_ids
        
        # make sure data is in a serializable format
        if isinstance(self.llm_generation_token_ids, torch.Tensor):
            self.llm_generation_token_ids = self.llm_generation_token_ids.tolist()
        if isinstance(self.llm_output_token_ids, torch.Tensor):
            self.llm_output_token_ids = self.llm_output_token_ids.tolist()
        
        self.decoded_llm_generation_tokens = get_decoded_tokens(tokenizer, llm_generation_token_ids)
        self.decoded_llm_generation_string = tokenizer.decode(llm_generation_token_ids)
        self.decoded_llm_output_tokens = get_decoded_tokens(tokenizer, llm_output_token_ids)
        self.decoded_llm_output_string = tokenizer.decode(llm_output_token_ids)

    def to_dict(self):
        result = super(AttackResultInfoData, self).properties_to_dict(self)
        return result
    
    @staticmethod
    def from_dict(property_dict):
        result = AttackResultInfoData()
        super(AttackResultInfoData, result).set_properties_from_dict(result, property_dict)
        return result

    def to_json(self):
        return JSONSerializableObject.json_dumps(self.to_dict(), use_indent = False)
    
    def copy(self):
        return AttackResultInfoData.from_dict(self.to_dict())

    @staticmethod
    def from_json(json_string):
        return AttackResultInfoData.from_dict(json.loads(json_string))
        
# The class that represents all of the data that's consistent 
class AttackResultInfo(JSONSerializableObject):
    def __init__(self):
        # properties that should always be set        

        # is this the non-randomized ("canonical") test performed before any randomized LLM testing (if enabled)?
        self.is_canonical_result = False

        # temperature value used for this specific test
        self.temperature = None

        # random seed values, either from attack_params or the iterated list        
        self.numpy_random_seed = None        
        self.torch_manual_seed = None
        self.torch_cuda_manual_seed_all = None
        # The result data is split into a separate sub-class to support having "jailbreak detection version" (shorter max token length by default).
        # (Versus "full decoding" (longer max token length by default) for the same input string/tokens).
        # Anything defined in this class that could theoretically be different at the data set level should NOT be changed at the data set level.
        # (e.g. different random seeds)
        # That is *not* what multiple data sets are for. They're *only* for the same data with different output lengths
        self.result_data_sets = {}
        self.jailbreak_detected = False
    
    def to_dict(self):
        result = super(AttackResultInfo, self).properties_to_dict(self)
        return result
    
    @staticmethod
    def apply_dict(existing_object, property_dict):
        super(AttackResultInfo, existing_object).set_properties_from_dict(existing_object, property_dict)
        if existing_object.result_data_sets is not None:
            deserialized_content = {}
            for k in existing_object.result_data_sets.keys():
                deserialized_content[k] = (AttackResultInfoData.from_dict(existing_object.result_data_sets[k]))
            existing_object.result_data_sets = deserialized_content
        return existing_object
    
    @staticmethod
    def from_dict(property_dict):
        result = AttackResultInfo()        
        return AttackResultInfo.apply_dict(result, property_dict)
    
    def to_json(self):
        return JSONSerializableObject.json_dumps(self.to_dict(), use_indent = False)
    
    def copy(self):
        return AttackResultInfo.from_dict(self.to_dict())
    
    def get_first_result_data_set(self):
        for k in self.result_data_sets.keys():
            return self.result_data_sets[k]
            break
        return None
                    
        return result
    
    @staticmethod
    def from_json(json_string):
        return AttackResultInfo.from_dict(json.loads(json_string))

# The class that represents all of the data for a given Broken Hill iteration that should be written to persistent storage
class AttackResultInfoCollection(JSONSerializableObject):
    def __init__(self):
        # "original" because the initial results could be augmented later
        self.iteration_number = None
        self.total_processing_time_seconds = None
        self.original_creation_date_time_utc = get_time_string()
        self.jailbreak_detection_count = 0
        self.loss = None
        self.adversarial_content = AdversarialContent()

        # the machine-learning engine where the results were generated
        self.ml_engine = "torch"

        # the token IDs that represent the entire prompt generated by get_prompt, *not* LLM output
        self.generated_prompt_token_ids = None

        # the token IDs that represent just the user input sent to the LLM - no system prompt/pre-input messages, output, etc.
        self.user_input_token_ids = None

        # a list of tokens decoded from generated_prompt_token_ids
        self.decoded_generated_prompt_tokens = None
        
        # a single string decoded from generated_prompt_token_ids
        self.decoded_generated_prompt_string = None

        # a list of tokens decoded from user_input_token_ids
        self.decoded_user_input_tokens = None
        # a single string decoded from user_input_token_ids
        self.decoded_user_input_string = None

        self.unique_results = {}
        self.unique_result_count = 0
        self.results = []
    
    def set_values(self, tokenizer, generated_prompt_token_ids, user_input_token_ids):
        self.generated_prompt_token_ids = generated_prompt_token_ids
        self.user_input_token_ids = user_input_token_ids
        
        # make sure data is in a serializable format
        if isinstance(self.generated_prompt_token_ids, torch.Tensor):
            self.generated_prompt_token_ids = self.generated_prompt_token_ids.tolist()
        if isinstance(self.user_input_token_ids, torch.Tensor):
            self.user_input_token_ids = self.user_input_token_ids.tolist()
        
        self.decoded_generated_prompt_tokens = get_decoded_tokens(tokenizer, generated_prompt_token_ids)
        self.decoded_generated_prompt_string = tokenizer.decode(generated_prompt_token_ids)
        self.decoded_user_input_tokens = get_decoded_tokens(tokenizer, user_input_token_ids)
        self.decoded_user_input_string = tokenizer.decode(user_input_token_ids)
    
    def get_unique_output_values(self):
        unique_output_values = {}
        for r in self.results:
            rds_results = []
            for result_data_set_name in r.result_data_sets.keys():
                output_value = r.result_data_sets[result_data_set_name].decoded_llm_output_string
                rds_results = add_value_to_list_if_not_already_present(rds_results, output_value, ignore_none = True)
            rds_results_filtered = []
            for rdsr_num in range(0, len(rds_results)):
                rdsr = rds_results[rdsr_num]
                if len(rdsr) > 0:
                    add_to_filtered_list = True
                    if len(rds_results_filtered) > 0:
                        for existing_rdsr_num in range(0, len(rds_results_filtered)):
                            existing_rdsr = rds_results_filtered[existing_rdsr_num]
                            if len(existing_rdsr) > 0:
                                if len(rdsr) > len(existing_rdsr):                                    
                                    if rdsr[0:len(existing_rdsr)] == existing_rdsr:
                                        # new entry is an extended version of the existing entry
                                        add_to_filtered_list = False
                                        rds_results_filtered[existing_rdsr_num] = rdsr
                                else:
                                    if existing_rdsr[0:len(rdsr)] == rdsr:
                                        # new entry is a shorter version of the existing entry
                                        add_to_filtered_list = False
                    if add_to_filtered_list:
                        rds_results_filtered.append(rdsr)                
            for output_value in rds_results_filtered:
                output_value_count = 1
                if output_value in unique_output_values.keys():
                    output_value_count = unique_output_values[output_value] + 1
                unique_output_values[output_value] = output_value_count
        return unique_output_values
                
    def get_unique_output_count(self):
        return len(list(self.unique_results.keys()))
    
    def update_unique_output_count(self):
        self.unique_result_count = self.get_unique_output_count()

    def update_unique_output_values(self):
        self.unique_results = self.get_unique_output_values()
        self.update_unique_output_count()

    def to_dict(self):
        result = super(AttackResultInfoCollection, self).properties_to_dict(self)
        return result
    
    @staticmethod
    def from_dict(property_dict):
        result = AttackResultInfoCollection()
        super(AttackResultInfoCollection, result).set_properties_from_dict(result, property_dict)
        if len(result.results) > 0:
            deserialized_results = []
            for i in range(0, len(result.results)):
                deserialized_results.append(AttackResultInfo.from_dict(result.results[i]))
            result.results = deserialized_results
        return result

    def to_json(self):
        return JSONSerializableObject.json_dumps(self.to_dict(), use_indent = False)
    
    def copy(self):
        return AttackResultInfoCollection.from_dict(self.to_dict())
    
    @staticmethod
    def from_json(json_string):
        return AttackResultInfoCollection.from_dict(json.loads(json_string))

# The class that holds all of the data that should be written to persistent storage as a record of a Broken Hill run
class BrokenHillResultData(JSONSerializableObject):
    def __init__(self):
        self.start_date_time = None
        self.end_date_time = None
        self.elapsed_time_string = None
        self.attack_params = None
        self.attack_results = []
        self.self_test_results = {}
        self.completed_iterations = 0
        self.model_parameter_info_collection = None
    
    def to_dict(self):
        result = super(BrokenHillResultData, self).properties_to_dict(self)
        return result
    
    @staticmethod
    def from_dict(property_dict):
        result = BrokenHillResultData()
        super(BrokenHillResultData, result).set_properties_from_dict(result, property_dict)
        
        if result.model_parameter_info_collection is not None:
            result.model_parameter_info_collection = LargeLanguageModelParameterInfoCollection.from_dict(result.model_parameter_info_collection)
        
        if len(result.attack_results) > 0:
            deserialized_content = []
            for i in range(0, len(result.attack_results)):
                deserialized_content.append(AttackResultInfoCollection.from_dict(result.attack_results[i]))
            result.attack_results = deserialized_content
        
        serialized_dict_keys = []
        for k in result.self_test_results.keys():
            serialized_dict_keys.append(k)
        
        if len(serialized_dict_keys) > 0:
            deserialized_dict = {}
            for i in range(0, len(serialized_dict_keys)):
                deserialized_dict[serialized_dict_keys[i]] = AttackResultInfoData.from_dict(result.self_test_results[serialized_dict_keys[i]])
            result.self_test_results = deserialized_dict
                
        return result

    def to_json(self):
        return JSONSerializableObject.json_dumps(self.to_dict(), use_indent = False)
    
    @staticmethod
    def from_json(json_string):
        return BrokenHillResultData.from_dict(json.loads(json_string))


class GenerationResults(JSONSerializableObject):
    def __init__(self):
        self.max_new_tokens = None
        # input_token_id_data.input_token_ids: the token IDs that represent just the user input part of the prompt generated by get_prompt
        # input_token_id_data: the PromptAndInputIDCollection returned by get_prompt
        # includes the full set of token IDs for the entire prompt, as well as the slices parsed for each section of the prompt
        # and a shortcut property (input_token_ids) that represents just the token IDs for the user input (base prompt + adversarial content)
        self.input_token_id_data = None

        # generation_input_token_ids: the token IDs that represent just the user input part of the prompt sent to the LLM - should be identical to the previous value
        self.generation_input_token_ids = None
        
        # output_ids: the complete set of tokens that represents the system prompt, messages, user input, and the LLM's response
        self.output_token_ids = None

        # output_token_ids_output_only: shortcut property containing the token IDs that represent just the LLM's response to the input
        self.output_token_ids_output_only = None

    def to_dict(self):
        result = super(GenerationResults, self).properties_to_dict(self)
        return result
    
    @staticmethod
    def from_dict(property_dict):
        result = GenerationResults()
        super(GenerationResults, result).set_properties_from_dict(result, property_dict)
        if result.input_token_id_data is not None:
            result.input_token_id_data = PromptAndInputIDCollection.from_dict(result.input_token_id_data)
        return result

    def to_json(self):
        return JSONSerializableObject.json_dumps(self.to_dict(), use_indent = False)
    
    def copy(self):
        return GenerationResults.from_dict(self.to_dict())
    
    @staticmethod
    def from_json(json_string):
        return GenerationResults.from_dict(json.loads(json_string))

class ResourceUtilizationException(Exception):
    pass

class CUDADeviceUtilizationData(JSONSerializableObject):
    def __init__(self):
        self.device_name = None
        self.device_display_name = None
        self.total_memory = None
        self.gpu_used_memory = None
        self.available_memory = None
        self.total_memory_utilization = None
        self.process_reserved_memory = None
        self.process_memory_utilization = None
        self.process_reserved_allocated_memory = None
        self.process_reserved_unallocated_memory = None
        self.process_reserved_utilization = None

    @staticmethod
    def create_snapshot(cuda_device):
        result = CUDADeviceUtilizationData()
        result.device_name = cuda_device.device_name
        result.device_display_name = cuda_device.device_display_name
        result.total_memory = cuda_device.total_memory
        result.gpu_used_memory = cuda_device.gpu_used_memory
        result.available_memory = cuda_device.available_memory
        result.total_memory_utilization = cuda_device.total_memory_utilization
        result.process_reserved_memory = cuda_device.process_reserved_memory
        result.process_memory_utilization = cuda_device.process_memory_utilization
        result.process_reserved_allocated_memory = cuda_device.process_reserved_allocated_memory
        result.process_reserved_unallocated_memory = cuda_device.process_reserved_unallocated_memory
        result.process_reserved_utilization = cuda_device.process_reserved_utilization
        return result

    def to_dict(self):
        result = super(CUDADeviceUtilizationData, self).properties_to_dict(self)
        return result
    
    @staticmethod
    def from_dict(property_dict):
        result = CUDADeviceUtilizationData()
        super(CUDADeviceUtilizationData, result).set_properties_from_dict(result, property_dict)        
        return result

    def to_json(self):
        return JSONSerializableObject.json_dumps(self.to_dict())
    
    def copy(self):
        return CUDADeviceUtilizationData.from_dict(self.to_dict())
    
    @staticmethod
    def from_json(json_string):
        return CUDADeviceUtilizationData.from_dict(json.loads(json_string))

class ResourceUtilizationSnapshot(JSONSerializableObject):
    def __init__(self):
        self.epoch_time = None
        self.location_description = None
        self.process_physical_memory = None
        self.process_virtual_memory = None
        self.process_swap = None
        self.system_physical_memory = None
        self.system_available_memory = None
        self.system_in_use_memory = None
        self.system_memory_util_percent = None
        self.system_swap_total = None
        self.system_swap_in_use = None
        self.system_swap_free = None
        self.system_swap_percent = None
        self.cuda_device_data = []
    
    @staticmethod
    def create_snapshot(location_description):
        result = ResourceUtilizationSnapshot()
        result.epoch_time = time.time_ns()
        result.location_description = location_description
        process_mem_info = psutil.Process().memory_full_info()
        result.process_physical_memory = process_mem_info.rss
        result.process_virtual_memory = process_mem_info.vms
        result.process_swap = None
        if hasattr(process_mem_info, "swap"):
            result.process_swap = process_mem_info.swap
        system_mem_info = psutil.virtual_memory()
        result.system_physical_memory = system_mem_info.total
        result.system_available_memory = system_mem_info.available
        result.system_in_use_memory = system_mem_info.used
        result.system_memory_util_percent = float(result.system_in_use_memory) / float(result.system_physical_memory)
        try:
            system_swap_info = psutil.swap_memory()
            result.system_swap_total = system_swap_info.total
            result.system_swap_in_use = system_swap_info.used
            result.system_swap_free = system_swap_info.free
            result.system_swap_percent = system_swap_info.percent
        except Exception as e:
            raise ResourceUtilizationSnapshot(f"Error getting swap memory information: {e}")
        
        cuda_devices = PyTorchDevice.get_all_cuda_devices()
        for i in range(0, len(cuda_devices)):
            cd = CUDADeviceUtilizationData.create_snapshot(cuda_devices[i])
            result.cuda_device_data.append(cd)

        # TKTK mps equivalent of the CUDA code for the day when the PyTorch Metal back-end supports the necessary functionality     
        return result

    def to_dict(self):
        result = super(ResourceUtilizationSnapshot, self).properties_to_dict(self)
        return result
    
    @staticmethod
    def from_dict(property_dict):
        result = ResourceUtilizationSnapshot()
        super(ResourceUtilizationSnapshot, result).set_properties_from_dict(result, property_dict)
        if len(result.cuda_device_data) > 0:
            deserialized_results = []
            for i in range(0, len(result.cuda_device_data)):
                deserialized_results.append(CUDADeviceUtilizationData.from_dict(result.cuda_device_data[i]))
            result.cuda_device_data = deserialized_results
        return result

    def to_json(self):
        return JSONSerializableObject.json_dumps(self.to_dict())
    
    def copy(self):
        return ResourceUtilizationSnapshot.from_dict(self.to_dict())
    
    @staticmethod
    def from_json(json_string):
        return ResourceUtilizationSnapshot.from_dict(json.loads(json_string))

class ResourceUtilizationStatistics(JSONSerializableObject):
    def __init__(self):
        self.cpu = StatisticsCube()
        self.cpu.cube_name = "CPU"
        self.performance = StatisticsCube()
        self.performance.cube_name = "Performance"
        self.cuda_devices = []

    def add_empty_cuda_device_cubes(self, cuda_device_count):
        for i in range(0, cuda_device_count):
            cc = StatisticsCube()
            self.cuda_devices.append(cc)

    def to_dict(self):
        result = super(ResourceUtilizationStatistics, self).properties_to_dict(self)
        return result
    
    @staticmethod
    def apply_dict(existing_object, property_dict):
        if not isinstance(existing_object, ResourceUtilizationStatistics):
            raise JSONSerializationException(f"Cannot apply properties for the ResourceUtilizationStatistics class to an instance of the class '{existing_object.__class__.__name__}'")
        super(ResourceUtilizationStatistics, existing_object).set_properties_from_dict(existing_object, property_dict)
        if existing_object.cpu is not None:
            existing_object.cpu = StatisticsCube.from_dict(existing_object.cpu)
        if existing_object.performance is not None:
            existing_object.performance = StatisticsCube.from_dict(existing_object.performance)
        if existing_object.cuda_devices is not None:
            if len(existing_object.cuda_devices) > 0:
                deserialized_content = []
                for i in range(0, len(existing_object.cuda_devices)):
                    deserialized_content.append(StatisticsCube.from_dict(existing_object.cuda_devices[i]))
                existing_object.cuda_devices = deserialized_content
        return existing_object
    
    @staticmethod
    def from_dict(property_dict):
        result = ResourceUtilizationStatistics()
              
        return result

    def to_json(self):
        return JSONSerializableObject.json_dumps(self.to_dict())
    
    def copy(self):
        return ResourceUtilizationStatistics.from_dict(self.to_dict())
    
    @staticmethod
    def from_json(json_string):
        return ResourceUtilizationStatistics.from_dict(json.loads(json_string))        
        

class ResourceUtilizationData(JSONSerializableObject):
    def __init__(self):
        self.snapshots = []
        self.statistics = ResourceUtilizationStatistics()

    def to_dict(self):
        result = super(ResourceUtilizationData, self).properties_to_dict(self)
        return result
    
    @staticmethod
    def apply_dict(existing_object, property_dict):
        if not isinstance(existing_object, ResourceUtilizationData):
            raise JSONSerializationException(f"Cannot apply properties for the ResourceUtilizationData class to an instance of the class '{existing_object.__class__.__name__}'")
        super(ResourceUtilizationData, existing_object).set_properties_from_dict(existing_object, property_dict)
        if existing_object.snapshots is not None:
            if len(existing_object.snapshots) > 0:
                deserialized_results = []
                for i in range(0, len(existing_object.snapshots)):
                    deserialized_results.append(ResourceUtilizationSnapshot.from_dict(existing_object.snapshots[i]))
                existing_object.snapshots = deserialized_results
        if existing_object.statistics is not None:
            existing_object.statistics = ResourceUtilizationStatistics.from_dict(existing_object.statistics)
        return existing_object
        
    @staticmethod
    def from_dict(property_dict):
        result = ResourceUtilizationData()
        result = ResourceUtilizationData.apply_dict(result, property_dict)
        return result

    def to_json(self):
        return JSONSerializableObject.json_dumps(self.to_dict())
    
    def copy(self):
        return ResourceUtilizationData.from_dict(self.to_dict())
    
    @staticmethod
    def from_json(json_string):
        return ResourceUtilizationData.from_dict(json.loads(json_string))        

    def populate_statistics(self):
        self.statistics = ResourceUtilizationStatistics()
        
        # Collect all of the data into arrays
        # CPU aggregation
        cpu_process_physical_memory = []
        cpu_process_virtual_memory = []
        cpu_process_swap = []
        cpu_system_physical_memory = []
        cpu_system_available_memory = []
        cpu_system_in_use_memory = []
        cpu_system_memory_util_percent = []
        cpu_system_swap_total = []
        cpu_system_swap_in_use = []
        cpu_system_swap_free = []
        cpu_system_swap_percent = []
        
        # Have to store this separately because there could be an arbitrary number
        cuda_device_values = []
        
        cuda_device_counts = []
        max_num_cuda_devices = 0
        for snapshot_num in range(0, len(self.snapshots)):
            num_cuda_devices = len(self.snapshots[snapshot_num].cuda_device_data)
            previous_cuda_device_count = None
            if len(cuda_device_counts) == 0:
                cuda_device_counts.append(num_cuda_devices)
            else:
                previous_cuda_device_count = cuda_device_counts[(len(cuda_device_counts) - 1)]
                if previous_cuda_device_count != num_cuda_devices:
                    cuda_device_counts.append(num_cuda_devices)
            
            if num_cuda_devices > max_num_cuda_devices:
                max_num_cuda_devices = num_cuda_devices

        if len(cuda_device_counts) > 1:
            times_changed = len(cuda_device_counts) - 1
            logger.warning(f"The number of detected CUDA devices changed {times_changed} time(s) when Broken Hill analyzed the collected performance data. This strongly implies that testing was performed on more than one system, with non-identical hardware. You should treat any overall statistics (maximum memory use, etc.) with skepticism, as it is likely inaccurate.")
        
        self.statistics.add_empty_cuda_device_cubes(max_num_cuda_devices)
        # collect per-CUDA-device data into arrays using this hack/misuse of a non-strongly-typed language:
        
        for cuda_device_num in range(0, max_num_cuda_devices):
            all_values_cuda_device = CUDADeviceUtilizationData()            
            all_values_cuda_device.total_memory = []
            all_values_cuda_device.gpu_used_memory = []
            all_values_cuda_device.available_memory = []
            all_values_cuda_device.total_memory_utilization = []
            all_values_cuda_device.process_reserved_memory = []
            all_values_cuda_device.process_memory_utilization = []
            all_values_cuda_device.process_reserved_allocated_memory = []
            all_values_cuda_device.process_reserved_unallocated_memory = []
            all_values_cuda_device.process_reserved_utilization = []
            cuda_device_values.append(all_values_cuda_device)
        
        for snapshot_num in range(0, len(self.snapshots)):
            snap = self.snapshots[snapshot_num]
            cpu_process_physical_memory.append(float(snap.process_physical_memory))
            cpu_process_virtual_memory.append(float(snap.process_virtual_memory))
            cpu_process_swap.append(float(snap.process_swap))
            #cpu_system_physical_memory.append(float(snap.system_physical_memory))
            cpu_system_available_memory.append(float(snap.system_available_memory))
            cpu_system_in_use_memory.append(float(snap.system_in_use_memory))
            cpu_system_memory_util_percent.append(float(snap.system_memory_util_percent) * 100.0)
            #cpu_system_swap_total.append(float(snap.system_swap_total))
            cpu_system_swap_in_use.append(float(snap.system_swap_in_use))
            cpu_system_swap_free.append(float(snap.system_swap_free))
            cpu_system_swap_percent.append(float(snap.system_swap_percent) * 100.0)
            
            num_cuda_devices = len(snap.cuda_device_data)
            
            for cuda_device_num in range(0, num_cuda_devices):
                all_values_cuda_device = cuda_device_values[cuda_device_num]
                if all_values_cuda_device.device_name is None:
                    all_values_cuda_device.device_name = snap.cuda_device_data[cuda_device_num].device_name
                if all_values_cuda_device.device_display_name is None:
                    all_values_cuda_device.device_display_name = snap.cuda_device_data[cuda_device_num].device_display_name
                #all_values_cuda_device.total_memory.append(float(snap.cuda_device_data[cuda_device_num].total_memory)))
                all_values_cuda_device.gpu_used_memory.append(float(snap.cuda_device_data[cuda_device_num].gpu_used_memory))
                all_values_cuda_device.available_memory.append(float(snap.cuda_device_data[cuda_device_num].available_memory))
                all_values_cuda_device.total_memory_utilization.append(float(snap.cuda_device_data[cuda_device_num].total_memory_utilization) * 100.0)
                all_values_cuda_device.process_reserved_memory.append(float(snap.cuda_device_data[cuda_device_num].process_reserved_memory))
                all_values_cuda_device.process_memory_utilization.append(float(snap.cuda_device_data[cuda_device_num].process_memory_utilization) * 100.0)
                all_values_cuda_device.process_reserved_allocated_memory.append(float(snap.cuda_device_data[cuda_device_num].process_reserved_allocated_memory))
                all_values_cuda_device.process_reserved_unallocated_memory.append(float(snap.cuda_device_data[cuda_device_num].process_reserved_unallocated_memory))
                all_values_cuda_device.process_reserved_utilization.append(float(snap.cuda_device_data[cuda_device_num].process_reserved_utilization) * 100.0)
                cuda_device_values[cuda_device_num] = all_values_cuda_device
        
        # Use the arrays to generate the data        
        self.statistics.cpu.add_or_update_dataset("process_physical_memory", cpu_process_physical_memory)
        self.statistics.cpu.add_or_update_dataset("process_virtual_memory", cpu_process_virtual_memory)
        self.statistics.cpu.add_or_update_dataset("process_swap", cpu_process_swap)
        #self.statistics.cpu.add_or_update_dataset("system_physical_memory", cpu_system_physical_memory)
        self.statistics.cpu.add_or_update_dataset("system_available_memory", cpu_system_available_memory)
        self.statistics.cpu.add_or_update_dataset("system_in_use_memory", cpu_system_in_use_memory)
        self.statistics.cpu.add_or_update_dataset("system_memory_util_percent", cpu_system_memory_util_percent)
        #self.statistics.cpu.add_or_update_dataset("system_swap_total", cpu_system_swap_total)
        self.statistics.cpu.add_or_update_dataset("system_swap_in_use", cpu_system_swap_in_use)
        self.statistics.cpu.add_or_update_dataset("system_swap_free", cpu_system_swap_free)
        self.statistics.cpu.add_or_update_dataset("system_swap_percent", cpu_system_swap_percent)

        for i in range(0, max_num_cuda_devices):
            if self.statistics.cuda_devices[i].cube_name is None:
                self.statistics.cuda_devices[i].cube_name = f"{cuda_device_values[i].device_display_name} ({cuda_device_values[i].device_name})"
            #self.statistics.cuda_devices[i].add_or_update_dataset("total_memory", cuda_device_values[i].total_memory)
            self.statistics.cuda_devices[i].add_or_update_dataset("gpu_used_memory", cuda_device_values[i].gpu_used_memory)
            self.statistics.cuda_devices[i].add_or_update_dataset("available_memory", cuda_device_values[i].available_memory)
            self.statistics.cuda_devices[i].add_or_update_dataset("total_memory_utilization", cuda_device_values[i].total_memory_utilization)
            self.statistics.cuda_devices[i].add_or_update_dataset("process_reserved_memory", cuda_device_values[i].process_reserved_memory)
            self.statistics.cuda_devices[i].add_or_update_dataset("process_memory_utilization", cuda_device_values[i].process_memory_utilization)
            self.statistics.cuda_devices[i].add_or_update_dataset("process_reserved_allocated_memory", cuda_device_values[i].process_reserved_allocated_memory)
            self.statistics.cuda_devices[i].add_or_update_dataset("process_reserved_unallocated_memory", cuda_device_values[i].process_reserved_unallocated_memory)
            self.statistics.cuda_devices[i].add_or_update_dataset("process_reserved_utilization", cuda_device_values[i].process_reserved_utilization)
    
    def populate_performance_statistics(self, attack_state):
        if attack_state.persistable.overall_result_data.attack_results is None:
            return
        num_iterations = len(attack_state.persistable.overall_result_data.attack_results)
        if num_iterations < 2:
            return
        processing_time = []
        for i in range(1, num_iterations):
            processing_time.append(attack_state.persistable.overall_result_data.attack_results[i].total_processing_time_seconds)
            
        self.statistics.performance.add_or_update_dataset("total_processing_time_seconds", processing_time)

    def add_statistics_line_item(self, message, found_data, stat_name, dataset, data_format_string, padding = '      '):
        new_message = message
        found_data_in_this_dataset = False
        convert_to_integer = False
        if ":n" in data_format_string:
            convert_to_integer = True
        if dataset is not None:
            new_message = f"{new_message}\n{padding}{stat_name}:"
            if dataset.maximum is not None:
                found_data_in_this_dataset = True
                val = dataset.maximum
                if convert_to_integer:
                    val = int(round(val))
                formatted_data = data_format_string.format(val)
                new_message = f"{new_message}\n{padding}  Maximum: {formatted_data}"
            if dataset.minimum is not None:
                found_data_in_this_dataset = True
                val = dataset.minimum
                if convert_to_integer:
                    val = int(round(val))
                formatted_data = data_format_string.format(val)
                new_message = f"{new_message}\n{padding}  Minimum: {formatted_data}"
            if dataset.value_range is not None:
                found_data_in_this_dataset = True
                val = dataset.value_range
                if convert_to_integer:
                    val = int(round(val))
                formatted_data = data_format_string.format(val)
                new_message = f"{new_message}\n{padding}  Range: {formatted_data}"
            if dataset.median is not None:
                found_data_in_this_dataset = True
                val = dataset.median
                if convert_to_integer:
                    val = int(round(val))
                formatted_data = data_format_string.format(val)
                new_message = f"{new_message}\n{padding}  Median: {formatted_data}"        
            if dataset.mean is not None:
                found_data_in_this_dataset = True
                val = dataset.mean
                if convert_to_integer:
                    val = int(round(val))
                formatted_data = data_format_string.format(val)
                new_message = f"{new_message}\n{padding}  Average: {formatted_data}"
        if found_data_in_this_dataset:
            found_data = True
            message = new_message
        return found_data, message

    def output_statistics(self, verbose = False):
        message = "Resource utilization / performance statistics:"
        if verbose:
            message = f"{message} (verbose)"
        
        found_data = False
        
        found_cpu_data = False        
        cpu_message = f"{message}\n\n  CPU:"
        cpu_message = f"{cpu_message}\n    Broken Hill process:"
        pvm = self.statistics.cpu.get_dataset("process_virtual_memory", raise_on_missing = False)
        found_cpu_data, cpu_message = self.add_statistics_line_item(cpu_message, found_cpu_data, "Virtual memory in use", pvm, "{0:n} byte(s)")
        if verbose:
            ppm = self.statistics.cpu.get_dataset("process_physical_memory", raise_on_missing = False)
            found_cpu_data, cpu_message = self.add_statistics_line_item(cpu_message, found_cpu_data, "Physical memory in use", ppm, "{0:n} byte(s)")
        pswap = self.statistics.cpu.get_dataset("process_swap", raise_on_missing = False)
        if verbose or pswap.maximum > 0:
            found_cpu_data, cpu_message = self.add_statistics_line_item(cpu_message, found_cpu_data, "Swap memory in use", pswap, "{0:n} byte(s)")
        cpu_message = f"{cpu_message}\n    System-wide:"
        sys_inuse = self.statistics.cpu.get_dataset("system_in_use_memory", raise_on_missing = False)
        found_cpu_data, cpu_message = self.add_statistics_line_item(cpu_message, found_cpu_data, "Memory in use", sys_inuse, "{0:n} byte(s)")
        if verbose:
            sys_avail = self.statistics.cpu.get_dataset("system_available_memory", raise_on_missing = False)
            found_cpu_data, cpu_message = self.add_statistics_line_item(cpu_message, found_cpu_data, "Memory available", sys_avail, "{0:n} byte(s)")
        sys_memory_util = self.statistics.cpu.get_dataset("system_memory_util_percent", raise_on_missing = False)
        found_cpu_data, cpu_message = self.add_statistics_line_item(cpu_message, found_cpu_data, "Memory utilization", sys_memory_util, "{0:.2f}%")
        sys_swap_inuse = self.statistics.cpu.get_dataset("system_swap_in_use", raise_on_missing = False)
        if verbose or sys_swap_inuse.maximum > 0:
            found_cpu_data, cpu_message = self.add_statistics_line_item(cpu_message, found_cpu_data, "Swap memory in use", sys_swap_inuse, "{0:n} byte(s)")
        if verbose:
            sys_swap_avail = self.statistics.cpu.get_dataset("system_swap_free", raise_on_missing = False)
            found_cpu_data, cpu_message = self.add_statistics_line_item(cpu_message, found_cpu_data, "Swap memory available", sys_swap_avail, "{0:n} byte(s)")        
        if verbose or sys_swap_inuse.maximum > 0:
            sys_swap_util = self.statistics.cpu.get_dataset("system_swap_percent", raise_on_missing = False)
            found_cpu_data, cpu_message = self.add_statistics_line_item(cpu_message, found_cpu_data, "Swap memory utilization", sys_swap_util, "{0:.2f}%")
        if found_cpu_data:
            found_data = True
            message = cpu_message
        
        found_cuda_data = False
        cuda_message = f"{message}\n\n  CUDA devices:"
        cuda_padding = '        '
        for cuda_device_num in range(0, len(self.statistics.cuda_devices)):
            found_cuda_device_data = False
            cd = self.statistics.cuda_devices[cuda_device_num]
            cuda_device_message = f"\n    Device {cuda_device_num}: {cd.cube_name}"
            cuda_device_message = f"{cuda_device_message}\n      Broken Hill process:"
            cp_used = cd.get_dataset("process_reserved_memory", raise_on_missing = False)
            found_cuda_device_data, cuda_device_message = self.add_statistics_line_item(cuda_device_message, found_cuda_device_data, "Memory reserved", cp_used, "{0:n} byte(s)", padding = cuda_padding)
            cp_util = cd.get_dataset("process_memory_utilization", raise_on_missing = False)
            found_cuda_device_data, cuda_device_message = self.add_statistics_line_item(cuda_device_message, found_cuda_device_data, "Memory utilization", cp_util, "{0:.2f}%", padding = cuda_padding)
            if verbose:
                cp_allocated = cd.get_dataset("process_reserved_allocated_memory", raise_on_missing = False)
                found_cuda_device_data, cuda_device_message = self.add_statistics_line_item(cuda_device_message, found_cuda_device_data, "Memory reserved and allocated", cp_allocated, "{0:n} byte(s)", padding = cuda_padding)
                cp_unallocated = cd.get_dataset("process_reserved_unallocated_memory", raise_on_missing = False)
                found_cuda_device_data, cuda_device_message = self.add_statistics_line_item(cuda_device_message, found_cuda_device_data, "Memory reserved but unallocated", cp_unallocated, "{0:n} byte(s)", padding = cuda_padding)
                cpr_util = cd.get_dataset("process_reserved_utilization", raise_on_missing = False)
                found_cuda_device_data, cuda_device_message = self.add_statistics_line_item(cuda_device_message, found_cuda_device_data, "Reserved memory utilization", cpr_util, "{0:.2f}%", padding = cuda_padding)
            cuda_device_message = f"{cuda_device_message}\n      System-wide:"
            cd_used = cd.get_dataset("gpu_used_memory", raise_on_missing = False)
            found_cuda_device_data, cuda_device_message = self.add_statistics_line_item(cuda_device_message, found_cuda_device_data, "Memory in use", cd_used, "{0:n} byte(s)", padding = cuda_padding)
            if verbose:
                cd_avail = cd.get_dataset("available_memory", raise_on_missing = False)
                found_cuda_device_data, cuda_device_message = self.add_statistics_line_item(cuda_device_message, found_cuda_device_data, "Memory available", cd_avail, "{0:n} byte(s)", padding = cuda_padding)
            cd_util = cd.get_dataset("total_memory_utilization", raise_on_missing = False)
            found_cuda_device_data, cuda_device_message = self.add_statistics_line_item(cuda_device_message, found_cuda_device_data, "Memory utilization", cd_util, "{0:.2f}%", padding = cuda_padding)
            
            if found_cuda_device_data:
                found_cuda_data = True
                cuda_message = f"{cuda_message}{cuda_device_message}"
        
        if found_cuda_data:
            found_data = True
            message = cuda_message
        
        # TKTK mps equivalent of the CUDA code for the day when the PyTorch Metal back-end supports the necessary functionality
        
        found_perf_data = False
        perf_message = f"{message}\n\n  Processing performance:"
        tpts = self.statistics.performance.get_dataset("total_processing_time_seconds", raise_on_missing = False)
        found_perf_data, perf_message = self.add_statistics_line_item(perf_message, found_perf_data, "Seconds to process each iteration:", tpts, "{0:.2f}")
    
        if found_perf_data:
            found_data = True
            message = perf_message
    
        if found_data:
            logger.info(message)

    def collect_torch_stats(self, attack_state, is_key_snapshot_event = False, location_description = None):
        current_snapshot = ResourceUtilizationSnapshot.create_snapshot(location_description)
        self.snapshots.append(current_snapshot)

        using_cpu = attack_state.persistable.attack_params.using_cpu()
        using_cuda = attack_state.persistable.attack_params.using_cuda()        
        #logger.debug(f"is_key_snapshot_event = {is_key_snapshot_event}, using_cpu = {using_cpu}, using_cuda = {using_cuda}")

        if attack_state.persistable.attack_params.performance_stats_output_file is not None:
            safely_write_text_output_file(attack_state.persistable.attack_params.performance_stats_output_file, self.to_json())
                
        if not attack_state.persistable.attack_params.verbose_resource_info:
            if not is_key_snapshot_event:
                return
        
        display_string = ""
        if location_description is None:
            display_string = f"System resource statistics\n"
        else:
            display_string = f"System resource statistics ({location_description})\n"
        
        if using_cpu:
            display_string += f"CPU:\n"
            display_string += f"\tBroken Hill process:\n"
            display_string += f"\t\tVirtual memory in use: {current_snapshot.process_virtual_memory:n} bytes\n"
            display_string += f"\t\tPhysical memory in use: {current_snapshot.process_physical_memory:n} bytes\n"
            if current_snapshot.process_swap is not None:
                display_string += f"\t\tSwap memory in use: {current_snapshot.process_swap:n} bytes\n"
            display_string += f"\tSystem-level:\n"
            display_string += f"\t\tTotal physical memory: {current_snapshot.system_physical_memory:n} bytes\n"
            display_string += f"\t\tMemory in use: {current_snapshot.system_in_use_memory:n} bytes\n"
            display_string += f"\t\tMemory available: {current_snapshot.system_available_memory:n} bytes\n"
            display_string += f"\t\tMemory utilization: {current_snapshot.system_memory_util_percent:.0%}\n"
            if current_snapshot.system_swap_total is not None:
                display_string += f"\t\tTotal swap memory: {current_snapshot.system_swap_total:n} bytes\n"
                display_string += f"\t\tSwap memory in use: {current_snapshot.system_swap_in_use:n} bytes\n"
                display_string += f"\t\tSwap memory available: {current_snapshot.system_swap_free:n} bytes\n"
                display_string += f"\t\tSwap memory utilization: {current_snapshot.system_swap_percent:.0%}\n"                    
            
        if using_cuda:
            for i in range(0, len(current_snapshot.cuda_device_data)):
                d = current_snapshot.cuda_device_data[i]
                display_string += f"CUDA device {d.device_name} - {d.device_display_name}:\n"
                display_string += f"\tBroken Hill process:\n"
                display_string += f"\t\tMemory reserved: {d.process_reserved_memory:n} byte(s)\n"
                display_string += f"\t\tMemory utilization: {d.process_memory_utilization:.0%}\n"
                display_string += f"\t\tMemory reserved and allocated: {d.process_reserved_allocated_memory:n} byte(s)\n"
                display_string += f"\t\tMemory reserved but unallocated: {d.process_reserved_unallocated_memory:n} byte(s)\n"
                display_string += f"\t\tReserved memory utilization: {d.process_reserved_utilization:.0%}\n"
                display_string += f"\tSystem-wide:\n"
                display_string += f"\t\tTotal memory: {d.total_memory:n} byte(s)\n"
                display_string += f"\t\tMemory in use: {d.gpu_used_memory:n} byte(s)\n"
                display_string += f"\t\tMemory available: {d.available_memory:n} byte(s)\n"
                display_string += f"\t\tMemory utilization: {d.total_memory_utilization:.0%}\n"
        # TKTK mps equivalent of the CUDA code for the day when the PyTorch Metal back-end supports the necessary functionality
        if current_snapshot.process_swap is not None:
            if current_snapshot.process_swap > 0:
                display_string += f"Warning: this process has {current_snapshot.process_swap:n} byte(s) swapped to disk. If you are encountering poor performance, it may be due to insufficient system RAM to handle the current Broken Hill configuration.\n"
        logger.info(display_string)
                

class FailureCounterBehaviour(IntFlag):
    # If a rollback to a parent node is triggered, the counter is reset to 0
    # If this flag is not set, then branches will tend to be "spiral-shaped", e.g. potentially lots of sub-branches while exploring variations, but tending to roll back to much earlier nodes when sequential failures occur.
    COUNTER_RESETS_WITH_EVERY_ROLLBACK = auto()

    # The counter is reset to 0 if a series of rollbacks reaches the root node
    # In most cases other than experiments, this flag should be set.
    COUNTER_RESETS_AT_ROOT_NODE = auto()

    # The counter is reset to 0 when a new node is created
    # In most cases other than experiments, this flag should be set.
    COUNTER_RESETS_AT_NODE_CREATION = auto()

    # The counter is reset to 0 when a new high water mark is reached
    COUNTER_RESETS_AT_NEW_HIGH_WATERMARK = auto()

    # The counter is reset to 0 every time the score reaches the best possible value
    COUNTER_RESETS_AT_IDEAL_SCORE = auto()

# This class is for managing a branching tree exploration of adversarial content
# Every time the script reaches the initial threshold value for jailbreak count and/or loss, it sets a rollback point with that set of adversarial data.
# If subsequent permutations based on that adversarial data fail to reach at least the 

class AssociationRebuildException(Exception):
    pass

# TKTK: make this a generic class, with jailbreaks and loss two subclasses, so that other scoring techniques are easier to add
class SearchTreeNode(JSONSerializableObject):
    def __init__(self):
        self.uuid = uuid.uuid4()
        self.parent_node = None
        self.parent_node_uuid = None
        self.child_nodes = []
        self.child_node_uuids = []
        self.adversarial_content = None

        # How many iterations the script will allow exploring a branch without meeting the required number of jailbreaks or loss score before triggering a rollback.
        # If this value is set to None or 0, any failure to meet the required thresholds will trigger a rollback
        self.rollback_grace_iterations = None

        # Multiplier for the rollback_grace_iterations value every time the search branches
        # e.g. with a value of 0.5, every branch results in the iteration limit being halved
        # This favours spending some time with more specific variations, but broader search across a variety of larger-scale changes
        self.rollback_grace_branch_multiplier = 0.75

        # What is the minimum number of jailbreaks for this branch?
        # If this value (minus the range value, below, if it's set) is not met for a given iteration, the "unsuccessful" counter will be incremented by 1
        self.jailbreak_count_minimum = None
        
        # How far below the jailbreak count minimum is the script allowed to go without incrementing the "unsuccessful" counter for this branch?
        self.jailbreak_count_range = None
        
        # How much should the minimum number of jailbreaks decrease at each branch, if at all?
        self.jailbreak_count_branch_decrease = 0

        # What is the maximum loss value for this branch?
        # If this value (plus the range value, below, if it's set) is not met for a given iteration, the "unsuccessful" counter will be incremented by 1
        self.loss_maximum = None
        
        # How far above the loss maximum is the script allowed to go without incrementing the "unsuccessful" counter for this branch?
        self.loss_range = None
        
        # How much should the maximum loss decrease at each branch, if at all?
        self.loss_branch_increase = 0

    def set_parent_node(self, new_parent_node):
        self.parent_node = new_parent_node
        self.parent_node_uuid = new_parent_node.uuid

    def add_child_node(self, new_child_node):
        self.child_nodes.append(new_child_node)
        self.child_node_uuids.append(new_child_node.uuid)

    def rebuild_associations(self, all_nodes_dict):
        if self.parent_node_uuid is not None:
            if self.parent_node_uuid in all_nodes_dict.keys():
                self.parent_node = all_nodes_dict[self.parent_node_uuid]
            else:
                raise AssociationRebuildException(f"Could not reassociate that node that has UUID {self.uuid} with its parent, because the parent UUID {self.parent_node_uuid} was not found in the dictionary of all nodes.")
        self.child_nodes = []
        for i in range(0, len(self.child_node_uuids)):
            if self.child_node_uuids[i] in all_nodes_dict.keys():
                self.child_nodes.append(all_nodes_dict[self.child_node_uuids[i]])
            else:
                raise AssociationRebuildException(f"Could not reassociate that node that has UUID {self.uuid} with one of its children, because the child UUID {self.child_node_uuids[i]} was not found in the dictionary of all nodes.")

    def to_dict(self):
        result = super(SearchTreeNode, self).properties_to_dict(self)
        return result
    
    # after deserializing all objects of this type, remember to go back and call rebuild_associations on all of them
    @staticmethod
    def from_dict(property_dict):
        result = SearchTreeNode()
        super(SearchTreeNode, result).set_properties_from_dict(result, property_dict)
        if result.parent_node_uuid is not None:
            result.input_token_id_data = PromptAndInputIDCollection.from_dict(result.input_token_id_data)
        return result

    def to_json(self):
        return JSONSerializableObject.json_dumps(self.to_dict(), use_indent = False)
    
    def copy(self):
        return SearchTreeNode.from_dict(self.to_dict())
    
    @staticmethod
    def from_json(json_string):
        return SearchTreeNode.from_dict(json.loads(json_string))
