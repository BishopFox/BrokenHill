#!/bin/env python

import copy
import json
import numpy
import re
import torch
import uuid

from enum import IntFlag
from enum import StrEnum
from enum import auto
from llm_attacks_bishopfox.dumpster_fires.trash_fire_tokens import get_decoded_token
from llm_attacks_bishopfox.dumpster_fires.trash_fire_tokens import get_decoded_tokens
from llm_attacks_bishopfox.attack.radiation_garden import RadiationGarden
from llm_attacks_bishopfox.dumpster_fires.trash_fire_tokens import encode_string_for_real_without_any_cowboy_funny_business
from llm_attacks_bishopfox.dumpster_fires.trash_fire_tokens import remove_empty_and_trash_fire_leading_and_trailing_tokens
#from llm_attacks_bishopfox.jailbreak_detection import LLMJailbreakDetectorRuleSet
from llm_attacks_bishopfox.jailbreak_detection.jailbreak_detection import get_default_negative_test_strings
from llm_attacks_bishopfox.jailbreak_detection.jailbreak_detection import get_default_positive_test_strings
from llm_attacks_bishopfox.json_serializable_object import JSONSerializableObject
from llm_attacks_bishopfox.llms.large_language_models import LargeLanguageModelParameterInfoCollection
from llm_attacks_bishopfox.util.util_functions import add_value_to_list_if_not_already_present
from llm_attacks_bishopfox.util.util_functions import get_now
from llm_attacks_bishopfox.util.util_functions import get_time_string
from llm_attacks_bishopfox.util.util_functions import slice_from_dict

class DecodingException(Exception):
    pass

class EncodingException(Exception):
    pass

# for debugging
class FakeException(Exception):
    pass

class PyTorchDevice():
    def __init__(self):
        self.type_name = None
        # The device's index within its type of device, e.g. 0 for cuda:0
        self.device_number = None
        self.device_name = None
        self.device_display_name = None
        self.total_memory = None
        self.gpu_total_memory = None
        self.gpu_free_memory = None
        self.gpu_used_memory = None
        self.process_reserved_memory = None
        self.available_memory = None
        self.process_reserved_allocated_memory = None
        self.process_reserved_unallocated_memory = None
        self.total_memory_utilization = None
        self.process_reserved_utilization = None
        self.process_reserved_utilization = None
    
    @staticmethod
    def from_cuda_device_number(device_number):
        result = PyTorchDevice()
        result.device_number = device_number
        result.device_name = f"cuda:{device_number}"
        gpu_wide_memory_info = torch.cuda.mem_get_info(device=device_number)
        result.gpu_free_memory = gpu_wide_memory_info[0]
        result.gpu_total_memory = gpu_wide_memory_info[1]
        result.gpu_used_memory = result.gpu_total_memory - result.gpu_free_memory
        
        device_props = torch.cuda.get_device_properties(device_number)
        result.device_display_name = device_props.name
        result.total_memory = device_props.total_memory        
        result.process_reserved_memory = torch.cuda.memory_reserved(device_number)
        result.process_reserved_allocated_memory = torch.cuda.memory_allocated(device_number)
        result.process_reserved_unallocated_memory = result.process_reserved_memory - result.process_reserved_allocated_memory
        #result.available_memory = result.total_memory - result.process_reserved_memory
        result.available_memory = result.gpu_free_memory
        result.total_memory_utilization = float(result.gpu_used_memory) / float(result.total_memory)
        result.process_memory_utilization = float(result.process_reserved_memory) / float(result.total_memory)
        if result.process_reserved_memory > 0:
            result.process_reserved_utilization = float(result.process_reserved_allocated_memory) / float(result.process_reserved_memory)
        else:
            result.process_reserved_utilization = 0.0        
        if result.total_memory != result.gpu_total_memory:
            print(f"[PyTorchDevice.from_cuda_device_number] warning: the amount of total memory available reported by torch.cuda.mem_get_info ({result.gpu_total_memory}) was not equal to the total reported by torch.cuda.get_device_properties ({result.total_memory}). This may cause some statistics to be incorrect.")
        return result

class BrokenHillMode(StrEnum):
    GCG_ATTACK = 'gcg_attack'
    GCG_ATTACK_SELF_TEST  = 'gcg_attack_self_test'
    LIST_IETF_TAGS = 'list_ietf_tags'

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
                #print(f"[AdversarialContent.token_list_contains_invalid_tokens] Warning: adversarial_candidate '{token_ids}' contains token ID {token_ids[token_num]}, which is outside the valid range for this tokenizer (min = 0, max = {tokenizer.vocab_size}). The candidate will be ignored. This may indicate an issue with the attack code, or the tokenizer code.")
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
                #print(f"[AdversarialContent.from_token_ids] Debug: couldn't decode token_ids directly via the tokenizer, but succeeded by using get_decoded_token: '{result.as_string}'")
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

    def __init__(self):
        self.operating_mode = BrokenHillMode.GCG_ATTACK
        
        #self.device = 'cuda'
        # the PyTorch device where the model (and everything else except the gradient, currently) should be loaded
        self.model_device = 'cuda'
        # the PyTorch device where the gradient operations should be performed
        self.gradient_device = 'cuda'
        
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
        
        self.override_fschat_templates = False
        
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

        # TKTK: option to require that loss decreases between iterations or the tool will roll back to the previous adversarial content and re-randomize
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
        self.missing_pad_token_replacement = None

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
        self.jailbreak_detection_rule_set = []      

        # TKTK: detect jailbreak based on some loss threshold?
        
        # If this value is specified, at each iteration, the tool will test results using <VALUE> additional random seed values, to attempt to avoid focusing on fragile results
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
            
            # If specified, exclude tokens that don't match the following pattern
            self.token_filter_regex = None
            
            # Filtering out other values can sometimes help prevent the script from focusing 
            # on attacks that are easily detectable as unusual, but also potentially 
            # filter out interesting attacks that would actually work when user input
            # is not heavily restricted. The command-line interface includes several 
            # shortcuts to populate this list with values I found useful at one time or another
            # but I'd recommend leaving it empty by default.
            # "GCG_ANY_ALL_WHITESPACE_TOKEN_GCG" is a special value that will exclude
            # any token that consists solely of whitespace
            self.not_allowed_token_list = []
            
            self.not_allowed_token_list_case_insensitive = []

        
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

        # The formatting string for roles when a model uses one of the generic fastchat templates 
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
        
        # try to avoid out-of-memory errors during the most memory-intensive part of the work
        self.batch_size_get_logits = 1

        # Output detailed token and token ID information when self tests fail
        self.verbose_self_test_output = False

        # Perform the attack even if the jailbreak self-tests indicate the results are unlikely to be useful
        self.ignore_jailbreak_self_tests = False

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
        
        # exit the tool entirely if the loss threshold is not met after the maximum attempt count is reached.
        # If this value is False, the tool will use the "best best" value determined during the attempt to find a value that met the threshold.
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
        # e.g. randomization is triggered, and four tokens are randomized. The result does not meet the "success" criteria. The tool should therefore roll back to the pre-randomization value, and randomize e.g. five tokens instead of four.

        self.radiation_gardens = []

        # output options
        self.overwrite_output = False
        self.json_output_file = None
        
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
        # The attack performed by this tool depends on PyTorch features that 
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
    def from_dict(property_dict):
        result = AttackParams()
        super(AttackParams, result).set_properties_from_dict(result, property_dict)
        if len(result.radiation_gardens) > 0:
            deserialized_gardens = []
            for i in range(0, len(result.radiation_gardens)):
                deserialized_gardens.append(RadiationGarden.from_dict(result.radiation_gardens[i]))
            result.radiation_gardens = deserialized_gardens
        if len(result.jailbreak_detection_rule_set) > 0:
            deserialized_jailbreak_rule_set = []
            for i in range(0, len(result.jailbreak_detection_rule_set)):
                deserialized_jailbreak_rule_set.append(LLMJailbreakDetectorRule.from_dict(result.jailbreak_detection_rule_set[i]))
            result.jailbreak_detection_rule_set = deserialized_jailbreak_rule_set
        return result
    
    @staticmethod
    def from_json(json_string):
        return AttackParams.from_dict(json.loads(json_string))

class AttackState(JSONSerializableObject):
    def __init__(self):
        self.attack_params = None
        self.iteration_count = 0
        
    def to_dict(self):
        result = super(AttackState, self).properties_to_dict(self)
        return result

    def to_json(self):
        return JSONSerializableObject.json_dumps(self.to_dict(), use_indent = False)
    
    def copy(self):
        return AttackState.from_dict(self.to_dict())
    
    @staticmethod
    def from_dict(property_dict):
        result = AttackState()
        super(AttackState, result).set_properties_from_dict(result, property_dict)
        if result.attack_params is not None:
            result.attack_params = AttackParams.from_dict(result.attack_params)
        return result
    
    @staticmethod
    def from_json(json_string):
        return AttackState.from_dict(json.loads(json_string))

class AttackResultInfoData(JSONSerializableObject):
    def __init__(self):
        # when was this specific set of data generated?
        self.date_time_utc = get_time_string()
        
        # the maximum number of new tokens for this data set specifically
        self.max_new_tokens = None
        
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

    # def set_values(self, tokenizer, max_token_length, generated_prompt_token_ids, llm_generation_token_ids, user_input_token_ids, llm_output_token_ids):
        # self.max_token_length = max_token_length
        # self.generated_prompt_token_ids = generated_prompt_token_ids
        # self.llm_generation_token_ids = llm_generation_token_ids
        # self.user_input_token_ids = user_input_token_ids
        # self.llm_output_token_ids = llm_output_token_ids
        
        # # make sure data is in a serializable format
        # if isinstance(self.generated_prompt_token_ids, torch.Tensor):
            # self.generated_prompt_token_ids = self.generated_prompt_token_ids.tolist()
        # if isinstance(self.llm_generation_token_ids, torch.Tensor):
            # self.llm_generation_token_ids = self.llm_generation_token_ids.tolist()
        # if isinstance(self.user_input_token_ids, torch.Tensor):
            # self.user_input_token_ids = self.user_input_token_ids.tolist()
        # if isinstance(self.llm_output_token_ids, torch.Tensor):
            # self.llm_output_token_ids = self.llm_output_token_ids.tolist()
        
        # self.decoded_generated_prompt_tokens = get_decoded_tokens(tokenizer, generated_prompt_token_ids)
        # self.decoded_generated_prompt_string = tokenizer.decode(generated_prompt_token_ids)
        # self.decoded_llm_generation_tokens = get_decoded_tokens(tokenizer, llm_generation_token_ids)
        # self.decoded_llm_generation_string = tokenizer.decode(llm_generation_token_ids)
        # self.decoded_user_input_tokens = get_decoded_tokens(tokenizer, user_input_token_ids)
        # self.decoded_user_input_string = tokenizer.decode(user_input_token_ids)
        # self.decoded_llm_output_tokens = get_decoded_tokens(tokenizer, llm_output_token_ids)
        # self.decoded_llm_output_string = tokenizer.decode(llm_output_token_ids)

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
    def from_dict(property_dict):
        result = AttackResultInfo()
        super(AttackResultInfo, result).set_properties_from_dict(result, property_dict)
        result_data_set_keys = []
        for rds_key in result.result_data_sets.keys():
            result_data_set_keys.append(rds_key)
        if len(result_data_set_keys) > 0:
            deserialized_result_data_sets = {}
            for i in range(0, len(result_data_set_keys)):
                current_key = result_data_set_keys[i]
                deserialized_result_data_sets[current_key] = AttackResultInfoData.from_dict(result.result_data_sets[current_key])
            result.result.result_data_sets = deserialized_result_data_sets
        return result
    
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

        # The full prompt (including adversarial content) in string form used for this iteration
        # use self.decoded_user_input_string instead
        #self.complete_user_input = None
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
                #if output_value is not None:
                    # output_value_count = 1
                    # if output_value in unique_output_values.keys():
                        # output_value_count = unique_output_values[output_value] + 1
                    # unique_output_values[output_value] = output_value_count
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
        
        result.model_parameter_info_collection is not None:
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