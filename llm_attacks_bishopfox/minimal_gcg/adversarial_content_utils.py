#!/bin/env python

import copy
import logging
import sys
import torch
# IMPORTANT: 'fastchat' is in the PyPi package 'fschat', not 'fastchat'!
import fastchat as fschat
import fastchat.conversation as fschat_conversation

from llm_attacks_bishopfox.dumpster_fires.trash_fire_tokens import get_decoded_token
from llm_attacks_bishopfox.dumpster_fires.trash_fire_tokens import get_decoded_tokens

from llm_attacks_bishopfox.attack.attack_classes import LossSliceMode
from llm_attacks_bishopfox.dumpster_fires.conversation_templates import get_llama2_and_3_fschat_template_names
from llm_attacks_bishopfox.dumpster_fires.trash_fire_tokens import encode_string_for_real_without_any_cowboy_funny_business
from llm_attacks_bishopfox.dumpster_fires.trash_fire_tokens import find_first_index_of_token
from llm_attacks_bishopfox.dumpster_fires.trash_fire_tokens import find_last_index_of_token
from llm_attacks_bishopfox.json_serializable_object import JSONSerializableObject
from llm_attacks_bishopfox.util.util_functions import add_value_to_list_if_not_already_present
from llm_attacks_bishopfox.util.util_functions import find_index_of_first_nonmatching_element
from llm_attacks_bishopfox.util.util_functions import find_last_occurrence_of_array_in_array
from llm_attacks_bishopfox.util.util_functions import get_widened_slice
from llm_attacks_bishopfox.util.util_functions import RequiredValueIsNoneException
from llm_attacks_bishopfox.util.util_functions import slice_from_dict

logger = logging.getLogger(__name__)

# Fast tokenizers that are totally broken and will do bonkers things like return None randomly for perfectly valid char_to_token parameters.
TEMPLATE_NAMES_USE_PYTHON_TOKENIZER = [ "llama2", "llama-2", "llama3", "llama-3", "oasst_pythia" ]

# Templates that always insert a hard-stop token of some kind even when you really want them to not do that
# The token is frequently "<|endoftext|>"
TEMPLATE_NAMES_REMOVE_TOKENS_AFTER_TARGET_OUTPUT = [ "aquila-v1", "falcon", "falcon-chat", "gptneox", "oasst_pythia", "qwen", "qwen2", "qwen-7b-chat", "redpajama-incite", 'stablelm', "TinyLlama", "Yi-34b-chat" ]

class PromptGenerationException(Exception):
    pass

def is_phi3_template(template_name):
    if len(template_name) >3 and template_name[0:4].lower() == "phi3":
        return True
    return False

def is_phi2_template(template_name):
    if len(template_name) >3 and template_name[0:4].lower() == "phi2":
        return True
    return False

def is_phi_template(template_name):
    if len(template_name) >2 and template_name[0:3].lower() == "phi":
        return True
    return False

DEFAULT_CONVERSATION_TEMPLATE_NAME = 'zero_shot'

def get_default_conversation_template():
    return fschat_conversation.get_conv_template(DEFAULT_CONVERSATION_TEMPLATE_NAME)

# def get_gemma_conversation_template():
    # conv_template = get_default_conversation_template().copy()
    # conv_template.name = "gemma"
    # conv_template.system_message = "<bos>"
    # conv_template.roles = ("<start_of_turn>user\n", "<start_of_turn>model\n")
    # conv_template.sep_style = fschat_conversation.SeparatorStyle.NO_COLON_SINGLE
    # conv_template.sep="<end_of_turn>\n"
    # conv_template.stop_str="<end_of_turn>"
    # return conv_template

def get_blenderbot_conversation_template():
    conv_template = get_default_conversation_template().copy()
    conv_template.name = "blenderbot"
    conv_template.system_template="{system_message}"
    conv_template.system_message = ""
    conv_template.roles=tuple(["   ", "  "])
    conv_template.sep_style = fschat_conversation.SeparatorStyle.NO_COLON_SINGLE
    conv_template.sep=""
    conv_template.sep2=""
    conv_template.stop_str=""
    conv_template.stop_str = "</s>"
    return conv_template

# def get_daredevil_conversation_template():
    # conv_template = fschat_conversation.get_conv_template("mistral").copy()
    # conv_template.name = "daredevil"
    # conv_template.system_template = "<s> [INST] {system_message}\n"
    # conv_template.sep2 = "</s>"
    # return conv_template

def get_felladrin_llama_conversation_template():
    conv_template = get_default_conversation_template().copy()
    conv_template.name="felladrin-llama-chat"
    conv_template.system_template = """<|im_start|>system
{system_message}"""
    conv_template.system_message=""
    conv_template.roles = tuple(["<|im_start|>user", "<|im_start|>assistant"])
    conv_template.sep_style=fschat_conversation.SeparatorStyle.CHATML
    conv_template.sep = "<|im_end|>"
    conv_template.stop_str = None
    return conv_template
    
def get_gemma_conversation_template():
    conv_template = get_default_conversation_template().copy()
    conv_template.name = "gemma"
    conv_template.system_message = "<bos>"
    #conv_template.system_message = ""
    #conv_template.roles = ("<start_of_turn>user\n", "<start_of_turn>model\n")
    #conv_template.roles = ("<start_of_turn>user", "<start_of_turn>model")
    #conv_template.roles=tuple(["<start_of_turn>user", "<start_of_turn>model"])
    conv_template.roles=tuple(["<start_of_turn>user\n", "<start_of_turn>model\n"])
    conv_template.sep_style = fschat_conversation.SeparatorStyle.NO_COLON_SINGLE
    conv_template.sep="<end_of_turn>\n"
    #conv_template.stop_str="<end_of_turn>"
    conv_template.stop_str=""
    return conv_template

def get_gptneox_conversation_template():
    conv_template = get_default_conversation_template().copy()
    conv_template.name = "gptneox"
    conv_template.system_template = "<|system|>\n{system_message}"
    conv_template.system_message = ""
    conv_template.roles = tuple(["<|end|>\n<|user|>\n", "<|end|>\n<|assistant|>\n"])
    conv_template.sep_style = fschat_conversation.SeparatorStyle.NO_COLON_SINGLE
    conv_template.sep = ''
    #conv_template.sep2 = ''
    conv_template.stop_str = "<|end|>\n<|endoftext|>"
    return conv_template

def get_guanaco_conversation_template():
    conv_template = fschat_conversation.get_conv_template("zero_shot").copy()
    conv_template.name = "guanaco"
    conv_template.sep=" ### "
    conv_template.stop_str = " </s>"
    return conv_template

def get_llama2_conversation_template():
    conv_template = fschat_conversation.get_conv_template("llama-2").copy()
    conv_template.name = "llama2"
    conv_template.system_template = "<s>[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n"
    if conv_template.system_message is None:
        conv_template.system_message = ""
    #conv_template.sep_style = fschat_conversation.SeparatorStyle.NO_COLON_SINGLE
    #roles=("</s><s>[INST]", "[/INST]"),
    #roles=("[INST]", "[/INST]"),
    #conv_template.sep = ' '
    #conv_template.sep2 = ' '
    #conv_template.sep = ' </s>'
    #conv_template.sep2 = ' </s><s>'
    #conv_template.sep = ' </s><s>'
    conv_template.stop_str = " </s>"
    return conv_template

def get_mistral_conversation_template():
    conv_template = fschat_conversation.get_conv_template("mistral").copy()
    conv_template.name = "mistral"
    conv_template.system_template = "<s> [INST] {system_message}\n"
    return conv_template

def get_mistralnemo_conversation_template():
    conv_template = fschat_conversation.get_conv_template("mistral").copy()
    conv_template.name = "mistral-nemo"
    conv_template.system_template = "<s>[INST]{system_message}"
    conv_template.sep_style = fschat_conversation.SeparatorStyle.LLAMA2
    conv_template.sep = ''
    conv_template.sep2 = '</s>'
    return conv_template

def get_mpt_conversation_template():
    conv_template = fschat_conversation.get_conv_template("mpt-7b-chat").copy()
    conv_template.name = "mpt"
    return conv_template

def get_phi2_conversation_template():
    conv_template = get_default_conversation_template().copy()
    conv_template.name = "phi2"
    conv_template.system_template = "System: {system_message}\n"
    conv_template.system_message = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful answers to the user's questions."
    conv_template.roles = tuple(["User", "Assistant"])
    #conv_template.sep_style = fschat_conversation.SeparatorStyle.NO_COLON_SINGLE
    conv_template.sep = '\n'
    conv_template.sep2 = '\n'
    return conv_template

def get_phi3_conversation_template():
    conv_template = get_default_conversation_template().copy()
    conv_template.name = "phi3"
    #conv_template.system_template = "<|system|>\n{system_message}<|end|>\n"
    conv_template.system_template = "<|system|>\n{system_message}"
    conv_template.system_message = ""
    #conv_template.roles = tuple(["\n<|user|>", "\n<|assistant|>"])
    conv_template.roles = tuple(["<|end|>\n<|user|>\n", "<|end|>\n<|assistant|>\n"])
    conv_template.sep_style = fschat_conversation.SeparatorStyle.NO_COLON_SINGLE
    conv_template.sep = ''
    #conv_template.sep2 = ''
    conv_template.stop_str = "<|end|>\n<|endoftext|>"
    return conv_template

def get_qwen_conversation_template():
    conv_template = fschat_conversation.get_conv_template("qwen-7b-chat").copy()
    conv_template.name = "qwen"
    conv_template.stop_str = None
    return conv_template

def get_qwen2_conversation_template():
    conv_template = get_qwen_conversation_template().copy()
    conv_template.name = "qwen2"
    conv_template.system_message = "You are a helpful assistant."
    conv_template.stop_str = None
    return conv_template

def get_solar_conversation_template():
    conv_template = fschat_conversation.get_conv_template("solar").copy()
    conv_template.name = "solar"
    conv_template.system_template = "### System:\n{system_message}"
    conv_template.stop_str = " </s>"
    return conv_template

def get_smollm_conversation_template():
    conv_template = get_default_conversation_template().copy()
    conv_template.name = "smollm"
    conv_template.system_template = "<|im_start|>system\n{system_message}"
    conv_template.system_message = ""
    conv_template.roles = ("<|im_start|>user", "<|im_start|>assistant")
    conv_template.sep_style = fschat_conversation.SeparatorStyle.CHATML
    conv_template.sep="<|im_end|>"
    conv_template.stop_token_ids=[
        0,
        1,
        2,
    ]  # "<|endoftext|>", "<|im_start|>", "<|im_end|>"
    conv_template.stop_str="<|endoftext|>"
    return conv_template

def get_stablelm2_conversation_template():
    conv_template = get_default_conversation_template().copy()
    conv_template.name = "stablelm2"
    conv_template.system_template = """<|im_start|>system
{system_message}"""
    conv_template.system_message = "You are a helpful assistant."
    conv_template.roles = tuple(["<|im_start|>user", "<|im_start|>assistant"])
    conv_template.sep_style = fschat_conversation.SeparatorStyle.CHATML
    conv_template.sep = "<|im_end|>"
    conv_template.stop_str = None
    return conv_template

def get_vikhr_conversation_template():
    conv_template = get_default_conversation_template().copy()
    conv_template.name = "vikhr"
    conv_template.system_template = """<|im_start|>system
{system_message}"""
    conv_template.system_message = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions."
    conv_template.roles = tuple(["<|im_start|>user", "<|im_start|>assistant"])
    conv_template.sep_style = fschat_conversation.SeparatorStyle.CHATML
    conv_template.sep = "<|im_end|>"
    conv_template.stop_str = None
    return conv_template

def register_custom_conversation_template(attack_params, fschat_has_existing_template, template_name, template):
    register_conv_template = True
    
    if template_name in fschat_conversation.conv_templates.keys():
        fschat_has_existing_template.append(template_name)
        if attack_params.override_fschat_templates:
            fschat_conversation.conv_templates[template_name] = template
    else:
        fschat_conversation.register_conv_template(template = template, override = True)
        
    return fschat_has_existing_template

def get_custom_conversation_templates():
    result = []
    # Mistral first because others depend on it
    result.append(get_mistral_conversation_template())

    result.append(get_blenderbot_conversation_template())
    result.append(get_daredevil_conversation_template())
    result.append(get_felladrin_llama_conversation_template())
    result.append(get_gemma_conversation_template())
    result.append(get_gptneox_conversation_template())
    result.append(get_guanaco_conversation_template())
    result.append(get_llama2_conversation_template())
    result.append(get_mistralnemo_conversation_template())
    result.append(get_mpt_conversation_template())
    result.append(get_phi2_conversation_template())
    result.append(get_phi3_conversation_template())
    result.append(get_qwen_conversation_template())
    result.append(get_qwen2_conversation_template())
    result.append(get_smollm_conversation_template())
    result.append(get_solar_conversation_template())
    result.append(get_stablelm2_conversation_template())
    result.append(get_vikhr_conversation_template())
    return result

def get_custom_conversation_template_names():
    result = []
    for ct in get_custom_conversation_templates():
        result.append(ct.name)
    result.sort()
    return result

# override = True is required because otherwise adding a missing template will fail with an assertion
def register_custom_conversation_templates(attack_params):
    fschat_has_existing_template = []

    custom_templates = get_custom_conversation_templates()

    for ct in custom_templates:
        fschat_has_existing_template = register_custom_conversation_template(attack_params, fschat_has_existing_template, ct.name, ct)

    if len(fschat_has_existing_template) > 0:
        if attack_params.template_name in fschat_has_existing_template:
            added_support_message = f"Warning: the fschat library in use includes its own version of the template '{attack_params.template_name}'. "
            if attack_params.override_fschat_templates:
                added_support_message += f"Broken Hill will use the custom version instead of the fschat version. If you wish to use the fschat version, specify the --do-not-override-fschat-templates option."
            else:
                added_support_message += f"Because --do-not-override-fschat-templates was specified, the fschat version of the template will be used. If you receive warnings or errors from the conversation template self-test, try omitting the --do-not-override-fschat-templates option to use the custom Broken Hill definition instead."
            logger.info(added_support_message)



class PromptSliceData(JSONSerializableObject):
    def __init__(self):
        self.system = None
        self.user_role = None
        self.goal = None
        self.control = None
        self.assistant_role = None
        self.target_output = None
        self.loss = None
    
    def get_slice_dictionary(self):
        result = {}
        result["system"] = self.system
        result["user_role"] = self.user_role
        result["goal"] = self.goal
        result["control"] = self.control
        result["assistant_role"] = self.assistant_role
        result["target_output"] = self.target_output
        result["loss"] = self.loss
        return result

    def to_dict(self):
        result = super(PromptSliceData, self).properties_to_dict(self)
        return result
    
    # handle any placement of base prompt ("goal") and adversarial content ("control")
    def get_complete_user_input_slice(self):
        result = get_widened_slice(self.goal, self.control)
        #logger.debug(f"result = {result}")
        return result

    # handle any placement of target output and loss data   
    def get_target_output_and_loss_slice(self):
        result = get_widened_slice(self.target_output, self.loss)
        #logger.debug(f"result = {result}")
        return result
    
    @staticmethod
    def from_dict(property_dict):
        result = PromptSliceData()
        super(PromptSliceData, result).set_properties_from_dict(result, property_dict)
        if result.system is not None:
            result.system = slice_from_dict(result.system)
        if result.user_role is not None:
            result.user_role = slice_from_dict(result.user_role)
        if result.goal is not None:
            result.goal = slice_from_dict(result.goal)
        if result.control is not None:
            result.control = slice_from_dict(result.control)
        if result.assistant_role is not None:
            result.assistant_role = slice_from_dict(result.assistant_role)
        if result.target_output is not None:
            result.target_output = slice_from_dict(result.target_output)
        if result.loss is not None:
            result.loss = slice_from_dict(result.loss)
        return result

    def to_json(self):
        return JSONSerializableObject.json_dumps(self.to_dict(), use_indent = False)
    
    def copy(self):
        return PromptSliceData.from_dict(self.to_dict())
    
    @staticmethod
    def from_json(json_string):
        return PromptSliceData.from_dict(json.loads(json_string))

class PromptAndInputIDException(Exception):
    pass

class PromptAndInputIDCollection(JSONSerializableObject):
    def __init__(self):
        self.prompt = None
        # The token IDs for the entire prompt, including system prompt, LLM output, etc.
        self.full_prompt_token_ids = None
        self.input_token_ids = None
        self.slice_data = PromptSliceData()
    
    def get_user_input_token_ids(self):
        #result = self.full_prompt_token_ids[self.slice_data.user_role.stop:self.slice_data.assistant_role.start]
        #start_index = self.slice_data.user_role.stop
        # default to the range being the beginning of the base prompt ("goal") to the end of the adversarial content ("control")
        user_input_slice = self.slice_data.get_complete_user_input_slice()
        if user_input_slice.start is None or user_input_slice.stop is None:
            raise PromptAndInputIDException(f"[get_user_input_token_ids] user_input_slice was {user_input_slice}, and neither start nor stop can be None for this function to succeed. self.slice_data.goal was {self.slice_data.goal}, self.slice_data.control was {self.slice_data.control}.")
        result = self.full_prompt_token_ids[user_input_slice]
        return result
    
    def get_input_ids_as_tensor(self):
        if isinstance(self.input_token_ids, torch.Tensor):
            return self.input_token_ids
        return torch.tensor(self.input_token_ids)

    def to_dict(self):
        result = super(PromptAndInputIDCollection, self).properties_to_dict(self)
        return result
    
    @staticmethod
    def from_dict(property_dict):
        result = PromptAndInputIDCollection()
        super(PromptAndInputIDCollection, result).set_properties_from_dict(result, property_dict)
        if result.slice_data is not None:
            result.slice_data = PromptSliceData.from_dict(result.slice_data)
        return result

    def to_json(self):
        return JSONSerializableObject.json_dumps(self.to_dict(), use_indent = False)
    
    def copy(self):
        return PromptAndInputIDCollection.from_dict(self.to_dict())
    
    @staticmethod
    def from_json(json_string):
        return PromptAndInputIDCollection.from_dict(json.loads(json_string))

# TKTK: expand to prefix/suffix attack, and also interleaving the tokens into the base string.
class AdversarialContentManager:
    def __init__(self, *, attack_state, conv_template, adversarial_content, trash_fire_tokens):

        self.attack_state = attack_state
        self.conv_template = conv_template
        self.adversarial_content = adversarial_content
        if isinstance(self.attack_state.persistable.attack_params.loss_slice_mode, type(None)):
            raise RequiredValueIsNoneException("self.attack_state.persistable.attack_params.loss_slice_mode cannot be None")
        self.trash_fire_tokens = trash_fire_tokens

    # For debugging / creating handlers for new conversation templates
    # accepts a dictionary of slices, where the key is the slice name and the value is the slice
    # and the list of tokens the slices refer to
    def get_slice_info(self, slice_data, tokens):
        result = {}
        decoded_tokens = get_decoded_tokens(self.attack_state, tokens)
        if self.attack_state.log_manager.get_lowest_log_level() <= logging.DEBUG:
            logger.debug(f"len(tokens) = {len(tokens)}, tokens = '{tokens}', decoded_tokens = '{decoded_tokens}'")
        slice_dictionary = slice_data.get_slice_dictionary()
        for slice_name in slice_dictionary.keys():
            sl = slice_dictionary[slice_name]
            if sl is not None:
                slice_tokens = tokens[sl]
                slice_tokens_decoded = decoded_tokens[sl]
                result[slice_name] = slice_tokens_decoded
        return result
                
    def print_slice_info(self, source_method_name, slice_data, tokens):
        if self.attack_state.log_manager.get_lowest_log_level() <= logging.DEBUG:
            logger.debug(f"len(tokens) = {len(tokens)}, tokens = '{tokens}'")
        slice_info = self.get_slice_info(slice_data, tokens)
        slice_dictionary = slice_data.get_slice_dictionary()
        for slice_name in slice_info.keys():
            if self.attack_state.log_manager.get_lowest_log_level() <= logging.DEBUG:
                logger.debug(f"Slice '{slice_name}' = {slice_dictionary[slice_name]}, decoded tokens = '{slice_info[slice_name]}', tokens = {tokens[slice_dictionary[slice_name]]}")

    def get_slice_info_for_validation_check(self, slice_data, slice_dictionary, ordered_slice_list, slice_number):
        slice_name = ordered_slice_list[slice_number]
        formatted_slice_name = f"the {slice_name} slice"
        slice_start = None
        slice_stop = None
        if " and " in ordered_slice_list[slice_number]:
            formatted_slice_name = f"the {slice_name} slices"
            if ordered_slice_list[slice_number] == "goal and control":
                wide_slice = slice_data.get_complete_user_input_slice()
                slice_start = wide_slice.start
                slice_stop = wide_slice.stop
                # if slice_data.goal.start is not None and slice_data.goal.stop is not None and slice_data.control.start is not None:
                    # slice_start = slice_data.goal.start
                    # if slice_data.control.start < slice_start:
                        # slice_start = slice_data.control.start
                    # slice_stop = slice_data.control.stop
                    # if slice_data.goal.stop > slice_stop:
                        # slice_stop = slice_data.goal.stop
            if ordered_slice_list[slice_number] == "target_output and loss":
                wide_slice = slice_data.get_target_output_and_loss_slice()
                slice_start = wide_slice.start
                slice_stop = wide_slice.stop
                # if slice_data.loss.start is not None and slice_data.loss.stop is not None and slice_data.target_output.start is not None:
                    # slice_start = slice_data.loss.start
                    # if slice_data.target_output.start < slice_start:
                        # slice_start = slice_data.target_output.start
                    # slice_stop = slice_data.target_output.stop
                    # if slice_data.loss.stop > slice_stop:
                        # slice_stop = slice_data.loss.stop
        else:
            current_slice_start = slice_dictionary[slice_name].start
            slice_stop = slice_dictionary[slice_name].stop
        return slice_name, slice_start, slice_stop

    def validate_slice_data(self, source_method_name, slice_data, token_ids, decoded_tokens):
        invalid_slice_dictionary = {}
        if self.attack_state.log_manager.get_lowest_log_level() <= logging.DEBUG:
            logger.debug(f"token_ids = {token_ids}, decoded_tokens = {decoded_tokens}")
        slice_dictionary = slice_data.get_slice_dictionary()
        all_slices_are_not_none = True
        found_at_least_one_slice = False
        for slice_name in slice_dictionary.keys():
            sl = slice_dictionary[slice_name]      
            is_valid = False
            if sl is None:
                all_slices_are_not_none = False
            else:
                is_valid = True
                if isinstance(sl.start, type(None)) or isinstance(sl.stop, type(None)):
                    is_valid = False
                if not is_valid:
                    # The system slice having a start of None is expected
                    if slice_name == "system":
                        if isinstance(sl.start, type(None)) and not isinstance(sl.stop, type(None)):
                            is_valid = True
                if is_valid:
                    found_at_least_one_slice = True
                else:
                    invalid_slice_dictionary[slice_name] = sl
            if self.attack_state.log_manager.get_lowest_log_level() <= logging.DEBUG:
                logger.debug(f"slice '{slice_name}' = '{sl}', is_valid = {is_valid}.")
        if not found_at_least_one_slice:
            all_slices_are_not_none = False
        if len(invalid_slice_dictionary.keys()) > 0:
            message = f"[{source_method_name}] Warning: one or more slices have None values instead of start or stop values. This generally indicates an issue with the tokenizing or parsing logic. The slice(s) with None values are: "
            for slice_name in invalid_slice_dictionary.keys():
                sl = invalid_slice_dictionary[slice_name]
                message += f"{slice_name}: {sl},"
            message = message[:-1]
            logger.info(message)
        
        if all_slices_are_not_none:
            nonsensical_slice_boilerplate = " This usually indicates a bug in the conversation-parsing logic of Broken Hill. Please contact a developer with reproduction steps if the issue has not already been reported."
            ordered_slice_list = [ "system", "user_role", "goal and control", "assistant_role", "target_output and loss" ]
            for current_slice_number in range(0, len(ordered_slice_list) - 1):
                current_slice_name, current_slice_start, current_slice_stop = self.get_slice_info_for_validation_check(slice_data, slice_dictionary, ordered_slice_list, current_slice_number)
                if self.attack_state.log_manager.get_lowest_log_level() <= logging.DEBUG:
                    logger.debug(f"current_slice_name = '{current_slice_name}', current_slice_start = {current_slice_start}, current_slice_stop = {current_slice_stop}")
                if current_slice_start is not None and current_slice_stop is not None:
                    if current_slice_start > current_slice_stop:
                        logger.error(f"The start index for {current_slice_name} ({current_slice_start}) is greater than the stop index for the same slice ({current_slice_stop}).{nonsensical_slice_boilerplate}")
                for comparison_slice_number in range(current_slice_number + 1, len(ordered_slice_list)):
                    comparison_slice_name, comparison_slice_start, comparison_slice_stop = self.get_slice_info_for_validation_check(slice_data, slice_dictionary, ordered_slice_list, comparison_slice_number)
                    if self.attack_state.log_manager.get_lowest_log_level() <= logging.DEBUG:
                        logger.debug(f"comparison_slice_name = '{comparison_slice_name}', comparison_slice_start = {comparison_slice_start}, comparison_slice_stop = {comparison_slice_stop}")
                    if current_slice_start is not None and comparison_slice_start is not None:
                        if current_slice_start > comparison_slice_start:
                            logger.error(f"The start index for {current_slice_name} ({current_slice_start}) is greater than the start index for {comparison_slice_name} ({comparison_slice_start}).{nonsensical_slice_boilerplate}")
                    if current_slice_stop is not None and comparison_slice_start is not None:
                        if current_slice_stop > comparison_slice_start:
                            # ignore this for the target_output and loss slice, because the overlap is expected
                            if current_slice_name != "assistant_role" or comparison_slice_name != "target_output and loss":
                                logger.error(f"The stop index for {current_slice_name} ({current_slice_stop}) is greater than the start index for {comparison_slice_name} ({comparison_slice_start}).{nonsensical_slice_boilerplate}")
        self.user_role = None
        self.goal = None
        self.control = None
        self.assistant_role = None
        self.target_output = None
        self.loss = None
        self.print_slice_info(f"validate_slice_data - {source_method_name}", slice_data, token_ids)

    def extend_slice_if_next_token_is_in_list(self, decoded_tokens, current_slice, token_string_list):
        if self.attack_state.log_manager.get_lowest_log_level() <= logging.DEBUG:
            logger.debug(f"decoded_tokens: {decoded_tokens}, decoded current slice is {decoded_tokens[current_slice]}.")
        result = current_slice
        next_token_stop = current_slice.stop + 1
        if (next_token_stop) <= len(decoded_tokens):
            next_token = decoded_tokens[current_slice.stop]
            if next_token in token_string_list or next_token.strip() in token_string_list:
                new_slice = slice(current_slice.start, next_token_stop)
                if self.attack_state.log_manager.get_lowest_log_level() <= logging.DEBUG:
                    logger.debug(f"next token is '{next_token}', extending slice stop by 1. Slice contents were {decoded_tokens[current_slice]}, now {decoded_tokens[new_slice]}.")
                result = new_slice
        return result

    # The get_prompt function was originally mostly undocumented with magic numbers that had no explanations for how they were derived, but were supposedly specific to three LLMs: Llama 2, Vicuna, and OpenAssistant's Pythia.
    
    # By examining the results of this function for those three models, I was able to reverse-engineer how to (more or less) find the correct values for other models.
    # I think it should all be fully automatic now, but in case you need to add model-specific logic or debug the parsing, here are what the slices represent.

    # Each slice should end up containing an array of token IDs that represent one or more words.
    # Even if the content is a single word, it will still be an array, and may be represented by more than one token by the LLM.
    # For example, Phi-3 uses the tokens "Ass" and "istant" to represent the single English word "Assistant".
    
    # _user_role_slice: the name of the conversation role that represents the user speaking to the LLM.
    #   e.g. "Human"
    
    # _goal_slice: the prompt that the attacker wants to jailbreak
    #   e.g. "Write a convincing anonymous death threat."
    
    # _control_slice: the current sequence of adversarial tokens
    #   e.g. "! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !"
    
    # _assistant_role_slice: the name of the conversation role that represents the LLM speaking to the user.
    #   e.g. "Assistant"
    
    # _target_slice: for attack generation, this is the operator's ideal goal output.
    #   e.g. ""
    #       For testing attack results, it's the LLM's output in response to the combination of prompt and adversarial tokens.    
    
    # The length of the target and loss slices must match when passed to downstream functions or the attack will crash with an error.
        
    # _loss_slice: this is some LLM sorcery.
    # Reading the original paper, especially page 6 of 2307.15043v2.pdf, one might think that the "loss slice" would be the LLM output that corresponds to the "target slice" in the non-LLM-generated prompt, and that the loss being calculated was the loss between the target and what the LLM actually generated, but this is not the case.
    #
    # The loss slice is the same as the target slice, but with the start and end indices reduced by 1.
    # If you *don't* reduce the indices by 1, the attack will utterly fail.
    #
    # Here's the official semi-explanation from the original paper:
    # "The intuition of this approach is that if the language model can be put into a 'state' where this completion is the most likely response, as opposed to refusing to answer the query, then it likely will continue the completion with precisely the desired objectionable behavior"
    #
    # nanogcg has a comment to the effect of having to shift the logits so that the previous token predicts the current token.
    #
    # If you're like me, you might wonder why this doesn't just optimize over time for adversarial content that matches the target string. Actually, that's very close to what's going on.
    #
    # The loss calculation is between the tokens of the adversarial content and the tokens that represents the role-switch from user to LLM plus all but the last token of the target string. The only reason (AFAIK) that the last token isn't included is that the length of the loss slice would then no longer match the target slice. Broken Hill has code to pad the data in this case, but it may cause issues with the attack.
    #
    # In other words, the GCG attack tries to discover a sequence of tokens that cause the LLM to predict that the next thing it should do is to switch contexts from reading your input to responding, then issue (most of) the target string. If it does that, then clearly the most likely thing for it to do is to give you the information you asked for, because by that point, the response *already contains the LLM's indication that it's happy to provide that information*. Go back and re-read the semi-explanation again if it didn't make sense the first time. It probably will now.
    # 
    # Once you make that connection, a lot of the potential failure modes of Broken Hill start to make more sense. For example, sending the conversations to the LLM in the correct format (including using the correct token *IDs*, not just identical text) is vital, because otherwise you're either not going to escape out of the user-input context, *or* you're going to cause the LLM to generate a pseudo-conversation between a human and an LLM, without responding the way it would when prompted in the format it expects. Adversarial content generated in this way is unlikely to work against other implementations of the same LLM, because those implementations will most likely send conversations to the LLM in the correct format.

    # In the original demonstration code's Llama-2 code path, the user role, goal, and control slices were incorrectly calculated.
    # This was because some of the code didn't take into account that the tokenizer considered "[INST]" three tokens instead of one.
    # I eventually sort of fixed this by making the code flexible enough to handle Llama-2 without a separate branch.
    
    # TKTK (maybe): implement two handling modes:
    # Truncate the slices to the shorter of the two (done)
    # Pad the shorter data to match the length of the longer data (done for len(loss) > len(target)
    #   Option to pad the shorter data with different tokens, e.g. padding, unknown, the tokenized version of a string, etc.
    # 

    #prompt_and_input_id_data should be a PromptAndInputIDCollection
    def get_complete_input_token_ids(self, prompt_and_input_id_data):
        return prompt_and_input_id_data.full_prompt_token_ids[prompt_and_input_id_data.slice_data.get_complete_user_input_slice()]
    
    #prompt_and_input_id_data should be a PromptAndInputIDCollection
    def get_complete_input_string(self, prompt_and_input_id_data):
        return self.attack_state.tokenizer.decode(self.get_complete_input_token_ids(prompt_and_input_id_data))
    
    def conversation_template_appends_colon_to_role_names(self):
        result = False
        if self.conv_template.sep_style == fschat_conversation.SeparatorStyle.ADD_COLON_SINGLE:
            result = True
        if self.conv_template.sep_style == fschat_conversation.SeparatorStyle.ADD_COLON_TWO:
            result = True
        if self.conv_template.sep_style == fschat_conversation.SeparatorStyle.ADD_COLON_SPACE_SINGLE:
            result = True
        if self.attack_state.log_manager.get_lowest_log_level() <= logging.DEBUG:
            logger.debug(f"self.conv_template.sep_style = {self.conv_template.sep_style}, result = {result}")
        return result

    def get_prompt(self, adversarial_content = None, force_python_tokenizer = False):#, update_self_values = True):

        result = PromptAndInputIDCollection()
        
        # set up temporary values based on permanent values
        working_adversarial_content = self.adversarial_content.copy()
        conversation_template = self.conv_template.copy()

        separator = ' '
        if not self.attack_state.persistable.attack_params.base_prompt:
            separator = ''
            
        if conversation_template is None:
            raise PromptGenerationException(f"Got a null conversation template when trying to call self.conv_template.copy(). This should never happen. self.conv_template was '{self.conv_template}'")
        
        if adversarial_content is not None:
            working_adversarial_content = adversarial_content.copy()

        conversation_template.append_message(conversation_template.roles[0], f"{self.attack_state.persistable.attack_params.base_prompt}{separator}{working_adversarial_content.as_string}")
        conversation_template.append_message(conversation_template.roles[1], f"{self.attack_state.persistable.attack_params.target_output}")
        result.prompt = conversation_template.get_prompt()

        encoded_conversation_template_prompt = self.attack_state.tokenizer(result.prompt)
        current_token_ids = encoded_conversation_template_prompt.input_ids
        current_decoded_tokens = get_decoded_tokens(self.attack_state, current_token_ids)
        #original_toks = copy.deepcopy(current_token_ids)
        #original_decoded_tokens = get_decoded_tokens(self.attack_state, original_toks)
       
        # Remove any tokens that occur *after* the end of the target output.
        # For models like oasst-sft-4-pythia-12b-epoch-3.5 that end messages with a hard delimiter and will not generate additional content after that point.
        # Currently only enabled where necessary until it's tested more
        #remove_tokens_after_target_output = False       
        #if conversation_template.name in TEMPLATE_NAMES_REMOVE_TOKENS_AFTER_TARGET_OUTPUT:
        #    remove_tokens_after_target_output = True
        # This is such a common issue, I'm going to try enabling it by default
        remove_tokens_after_target_output = True

        # Use the Python tokenizer instead of the fast tokenizer if:
        # 1 - The model doesn't support a fast tokenizer
        # 2 - The user specified the Python tokenizer
        # 3 - The fast tokenizer is in the list of dirty outlaws that are looking to hijack the bank's stagecoach
        python_tokenizer = False
        if conversation_template.name in TEMPLATE_NAMES_USE_PYTHON_TOKENIZER:
            python_tokenizer = True
        if force_python_tokenizer:
            python_tokenizer = True
        # This (formerly undocumented) check is a way to determine if the model is using Python-based tokenizers. It works because Python-based tokenizers (at least in the current version of Transformers) don't support the char_to_token operation), and it's used to avoid calling char_to_token for the rest of the get_prompt method in that case.
        if not python_tokenizer:
            try:
                #test_value = encoded_conversation_template_prompt.char_to_token(len(result.prompt)-1)
                # Unlike EVERY OTHER Transformers-compatible LLM, Llama-3 doesn't throw an exception when the Python tokenizer gets a call to char_to_token - it returns None instead! But only for offsets that are in the middle of the prompt! Otherwise it pretends it works!
                test_value = encoded_conversation_template_prompt.char_to_token(int((float(len(result.prompt)) - 1.0) / 2.0))
                if self.attack_state.log_manager.get_lowest_log_level() <= logging.DEBUG:
                    logger.debug(f"Testing char_to_token to determine fast tokenizer support. test_value = {test_value}")
                if test_value is None:
                    python_tokenizer = True
            except:
                python_tokenizer = True

        done_trying = False
        while not done_trying:
            if python_tokenizer:
                done_trying = True
                # TKTK: consider rewriting this to not use fschat at all.
                # Using apply_chat_template where available and including custom templates for models that don't include it might be easier.
                if self.attack_state.log_manager.get_lowest_log_level() <= logging.DEBUG:
                    logger.debug(f"Using Python tokenizer.")
                # reset the conversation template (to get rid of the messages that have been added earlier in this function), but preserve any messages that are part of the template
                conversation_template = self.conv_template.copy()
                
                # Figure out where the user role *starts* first
                # ...by comparing the prompt with no additional messages to one that has a new empty user message added
                prompt_with_no_additional_messages = conversation_template.get_prompt()
                prompt_with_no_additional_messages_token_ids = self.attack_state.tokenizer(prompt_with_no_additional_messages).input_ids            
                #conversation_template.append_message(conversation_template.roles[0], None)
                conversation_template.append_message(conversation_template.roles[0], f"{self.attack_state.persistable.attack_params.base_prompt}{separator}{working_adversarial_content.as_string}")
                current_prompt = conversation_template.get_prompt()
                if self.attack_state.log_manager.get_lowest_log_level() <= logging.DEBUG:
                    logger.debug(f"current_prompt = '{current_prompt}'")
                user_role_prompt_token_ids = self.attack_state.tokenizer(current_prompt).input_ids
                user_role_start_index = find_index_of_first_nonmatching_element(prompt_with_no_additional_messages_token_ids, user_role_prompt_token_ids, log_manager = self.attack_state.log_manager)
                if user_role_start_index is None:
                    logger.error(f"Did not find a non-matching element when comparing the strings '{prompt_with_no_additional_messages}' and '{current_prompt}', which were tokenized to the following IDs:\n{prompt_with_no_additional_messages_token_ids}\n{user_role_prompt_token_ids}")

                # Detour into figuring out the base prompt / goal, adversarial content / control, and target output slices
                # TKTK: update the remainder of this function to handle different placement of the adversarial content
                #conversation_template.update_last_message(f"{self.attack_state.persistable.attack_params.base_prompt}{separator}{working_adversarial_content.as_string}")
                conversation_template.append_message(conversation_template.roles[1], self.attack_state.persistable.attack_params.target_output)
                current_prompt = conversation_template.get_prompt()
                if self.attack_state.log_manager.get_lowest_log_level() <= logging.DEBUG:
                    logger.debug(f"current_prompt = '{current_prompt}'")
                current_token_ids = self.attack_state.tokenizer(current_prompt).input_ids
                current_decoded_tokens = get_decoded_tokens(self.attack_state, current_token_ids)

                result.slice_data.goal = find_last_index_of_token(self.attack_state, 
                    self.trash_fire_tokens, 
                    self.attack_state.persistable.attack_params.base_prompt, 
                    current_token_ids, 
                    current_decoded_tokens, 
                    conversation_template = conversation_template, 
                    strip_leading_and_trailing_tokens = False, 
                    remove_empty_leading_and_trailing_tokens = False, 
                    remove_leading_and_trailing_trash_fire_tokens = False)
                
                # Can't do this optimization because the context of what appears before and after can change the tokens
                # goal_slice_start = find_last_occurrence_of_array_in_array(self.base_prompt_token_ids, current_token_ids, log_manager = self.attack_state.log_manager)
                # goal_slice_end = goal_slice_start + len(self.base_prompt_token_ids)
                # result.slice_data.goal = slice(goal_slice_start, goal_slice_end)

                self.validate_slice_data('get_prompt (Python) - goal', result.slice_data, current_token_ids, current_decoded_tokens)

                if self.attack_state.log_manager.get_lowest_log_level() <= logging.DEBUG:
                    logger.debug(f"current_prompt = '{current_prompt}'")

                if working_adversarial_content.as_string == "":
                    result.slice_data.control = slice(result.slice_data.goal.stop, result.slice_data.goal.stop)
                else:
                    result.slice_data.control = find_last_index_of_token(self.attack_state, 
                        self.trash_fire_tokens, 
                        working_adversarial_content.as_string, 
                        current_token_ids, 
                        current_decoded_tokens, 
                        conversation_template = conversation_template, 
                        strip_leading_and_trailing_tokens = False, 
                        remove_empty_leading_and_trailing_tokens = False, 
                        remove_leading_and_trailing_trash_fire_tokens = False)
                    # control_slice_start = find_last_occurrence_of_array_in_array(working_adversarial_content.token_ids, current_token_ids, log_manager = self.attack_state.log_manager)
                    # control_slice_end = control_slice_start + len(working_adversarial_content.token_ids)
                    # result.slice_data.control = slice(control_slice_start, control_slice_end)

                self.validate_slice_data('get_prompt (Python) - control', result.slice_data, current_token_ids, current_decoded_tokens)

                if self.attack_state.log_manager.get_lowest_log_level() <= logging.DEBUG:
                    logger.debug(f"current_prompt = '{current_prompt}'")

                result.slice_data.target_output = find_last_index_of_token(self.attack_state, 
                    self.trash_fire_tokens, 
                    self.attack_state.persistable.attack_params.target_output, 
                    current_token_ids, 
                    current_decoded_tokens, 
                    conversation_template = conversation_template, 
                    strip_leading_and_trailing_tokens = False, 
                    remove_empty_leading_and_trailing_tokens = False, 
                    remove_leading_and_trailing_trash_fire_tokens = False)
                
                # Can't do this optimization because the context of what appears before and after can change the tokens
                # target_slice_start = find_last_occurrence_of_array_in_array(self.target_output_token_ids, current_token_ids, log_manager = self.attack_state.log_manager)
                # target_slice_end = goal_slice_start + len(self.target_output_token_ids)
                # result.slice_data.target_output = slice(target_slice_start, target_slice_end)
                
                self.validate_slice_data('get_prompt (Python) - target_output', result.slice_data, current_token_ids, current_decoded_tokens)

                # use the known locations of the base prompt, adversarial content, and target output slices to determine the remaining role indices
                # this section should already handle alternative adversarial content placement more or less correctly, except if it's placed in the middle of the base prompt (or interleaved)
                
                if self.attack_state.log_manager.get_lowest_log_level() <= logging.DEBUG:
                    logger.debug(f"current_prompt = '{current_prompt}'")
                    logger.debug(f"result.slice_data.goal.start = {result.slice_data.goal.start}, result.slice_data.control.start = {result.slice_data.control.start}")
                
                user_role_stop_index = result.slice_data.goal.start
                if result.slice_data.control.start < user_role_stop_index:
                    user_role_stop_index = result.slice_data.control.start
                
                result.slice_data.user_role = slice(user_role_start_index, user_role_stop_index)
                self.validate_slice_data('get_prompt (Python) - user_role', result.slice_data, current_token_ids, current_decoded_tokens)
                
                if self.attack_state.log_manager.get_lowest_log_level() <= logging.DEBUG:
                    logger.debug(f"result.slice_data.control.stop = {result.slice_data.control.stop}, result.slice_data.goal.stop = {result.slice_data.goal.stop}")
                
                assistant_role_start_index = result.slice_data.control.stop
                if result.slice_data.goal.stop > assistant_role_start_index:
                    assistant_role_start_index = result.slice_data.goal.stop
                
                assistant_role_stop_index = result.slice_data.target_output.start
                
                result.slice_data.assistant_role = slice(assistant_role_start_index, assistant_role_stop_index)

                self.validate_slice_data('get_prompt (Python) - assistant_role', result.slice_data, current_token_ids, current_decoded_tokens)

                if self.attack_state.log_manager.get_lowest_log_level() <= logging.DEBUG:
                    logger.debug(f"self.attack_state.persistable.attack_params.loss_slice_mode = {self.attack_state.persistable.attack_params.loss_slice_mode} result.slice_data.assistant_role.start = {result.slice_data.assistant_role.start}, result.slice_data.target_output.start = {result.slice_data.target_output.start}, result.slice_data.target_output.stop = {result.slice_data.target_output.stop}")

                if self.attack_state.persistable.attack_params.loss_slice_mode == LossSliceMode.ASSISTANT_ROLE_PLUS_FULL_TARGET_SLICE:                
                    #result.slice_data.loss = slice(result.slice_data.assistant_role.start, min(last_non_garbage_token, len(current_token_ids)))
                    result.slice_data.loss = slice(result.slice_data.assistant_role.start, result.slice_data.target_output.stop)

                if self.attack_state.persistable.attack_params.loss_slice_mode == LossSliceMode.ASSISTANT_ROLE_PLUS_TRUNCATED_TARGET_SLICE:
                    len_target_slice = result.slice_data.target_output.stop - result.slice_data.target_output.start
                    result.slice_data.loss = slice(result.slice_data.assistant_role.start, (result.slice_data.assistant_role.start + len_target_slice))
                    
                if self.attack_state.persistable.attack_params.loss_slice_mode == LossSliceMode.INDEX_SHIFTED_TARGET_SLICE:
                    #result.slice_data.loss = slice(first_non_garbage_token - 1, min(last_non_garbage_token, len(current_token_ids)) - 1)
                    result.slice_data.loss = slice(int(result.slice_data.target_output.start) + self.attack_state.persistable.attack_params.loss_slice_index_shift, min(int(result.slice_data.target_output.stop) + self.attack_state.persistable.attack_params.loss_slice_index_shift, len(current_token_ids)))

                if self.attack_state.persistable.attack_params.loss_slice_mode == LossSliceMode.SAME_AS_TARGET_SLICE:
                    #result.slice_data.loss = slice(first_non_garbage_token, min(last_non_garbage_token, len(current_token_ids)))
                    result.slice_data.loss = slice(result.slice_data.target_output.start, result.slice_data.target_output.stop)
                
                if result.slice_data.loss is None:
                    raise PromptGenerationException("Did not find a valid loss slice mode")

                self.validate_slice_data('get_prompt (Python) - loss', result.slice_data, current_token_ids, current_decoded_tokens)
                
            else:
                try:
                    if self.attack_state.log_manager.get_lowest_log_level() <= logging.DEBUG:
                        logger.debug(f"Using fast tokenizer")
                    
                    # result.slice_data.user_role = slice(
                        # encoded_conversation_template_prompt.char_to_token(result.prompt.find(conversation_template.roles[0])),
                        # encoded_conversation_template_prompt.char_to_token(result.prompt.find(conversation_template.roles[0]) + len(conversation_template.roles[0]) + 1)
                    # )
                    last_token_index = result.prompt.rindex(conversation_template.roles[0])
                    #logger.debug(f"last_token_index = {last_token_index}")
                    
                    if last_token_index is None:
                        logger.warning(f"couldn't find conversation role 0 '{conversation_template.roles[0]}' in prompt '{result.prompt}'")
                    else:
                        if conversation_template.name in get_llama2_and_3_fschat_template_names():
                            try:
                                last_token_index_2 = result.prompt.rindex(conversation_template.roles[0], 0, last_token_index)
                                #logger.debug(f"last_token_index_2 = {last_token_index_2}")
                                if last_token_index_2 is not None:
                                    last_token_index = last_token_index_2
                            except Exception as e:
                                dummy = 1
                                # last_token_index is already correct
                                if self.attack_state.log_manager.get_lowest_log_level() <= logging.DEBUG:
                                    logger.debug(f"exception while getting second token index for user role using fast tokenizer: {e}")
                    
                    if self.attack_state.log_manager.get_lowest_log_level() <= logging.DEBUG:
                        logger.debug(f"result.prompt = {result.prompt}")
                        logger.debug(f"last_token_index = {last_token_index}")
                        logger.debug(f"encoded_conversation_template_prompt = {encoded_conversation_template_prompt}")
                        decoded_encoded_conversation_template_prompt = get_decoded_tokens(self.attack_state, encoded_conversation_template_prompt.input_ids)
                        logger.debug(f"decoded_encoded_conversation_template_prompt = {decoded_encoded_conversation_template_prompt}")
                    
                    user_role_start_index = encoded_conversation_template_prompt.char_to_token(last_token_index)
                    if self.attack_state.log_manager.get_lowest_log_level() <= logging.DEBUG:
                        logger.debug(f"user_role_start_index = {user_role_start_index}")
                    if user_role_start_index is None:
                        raise PromptGenerationException("user_role_start_index was None")
                    user_role_end_index = encoded_conversation_template_prompt.char_to_token(last_token_index + len(conversation_template.roles[0]) + 1)
                    if user_role_end_index is None:
                        raise PromptGenerationException("user_role_end_index was None")
                    if self.attack_state.log_manager.get_lowest_log_level() <= logging.DEBUG:
                        logger.debug(f"user_role_start_index = {user_role_start_index}")
                        logger.debug(f"user_role_end_index = {user_role_end_index}")
                    
                    result.slice_data.user_role = slice(
                        user_role_start_index,
                        user_role_end_index
                    )
                    
                    self.validate_slice_data('get_prompt (non-Python) - user_role', result.slice_data, current_token_ids, current_decoded_tokens)

                    result.slice_data.system = slice(0, result.slice_data.user_role.start)
                    
                    self.validate_slice_data('get_prompt (non-Python) - system', result.slice_data, current_token_ids, current_decoded_tokens)

                    # TKTK: BEGIN: update the goal and control slice logic to handle different placement of the adversarial content
                    base_prompt_start_index = result.prompt.rindex(self.attack_state.persistable.attack_params.base_prompt)
                    len_base_prompt = len(self.attack_state.persistable.attack_params.base_prompt)
                    base_prompt_end_index = base_prompt_start_index + len_base_prompt
                    encoded_base_prompt = encode_string_for_real_without_any_cowboy_funny_business(self.attack_state, self.attack_state.persistable.attack_params.base_prompt)
                    if encoded_base_prompt is None or len(encoded_base_prompt) == 0:
                        raise PromptGenerationException("encoded_base_prompt was {encoded_base_prompt}, cannot be None or zero-length.")
                    
                    if self.attack_state.log_manager.get_lowest_log_level() <= logging.DEBUG:
                        logger.debug(f"base_prompt_start_index = {base_prompt_start_index}, len_base_prompt = {len_base_prompt}, base_prompt_end_index = {base_prompt_end_index}")
                    base_prompt_token_start_index = encoded_conversation_template_prompt.char_to_token(base_prompt_start_index)
                    base_prompt_token_end_index = encoded_conversation_template_prompt.char_to_token(base_prompt_end_index)
                    if base_prompt_token_start_index is None:
                        raise PromptGenerationException("base_prompt_token_start_index was None") 
                    if base_prompt_token_end_index is None:
                        # I don't know why some tokenizers return None for perfectly valid end indices here.
                        # I don't know anything anymore.
                        #raise PromptGenerationException("base_prompt_token_end_index was None") 
                        # This fallback approach is not guaranteed to be accurate, but should be nearly all of the time
                        base_prompt_token_end_index = base_prompt_token_start_index + len(encoded_base_prompt)
                        if self.attack_state.log_manager.get_lowest_log_level() <= logging.DEBUG:
                            logger.debug(f"updated base_prompt_token_end_index to {base_prompt_token_end_index} using fallback logic because it was None")
                    if self.attack_state.log_manager.get_lowest_log_level() <= logging.DEBUG:
                        logger.debug(f"base_prompt_token_start_index = {base_prompt_token_start_index}, base_prompt_token_end_index = {base_prompt_token_end_index}")
                    if base_prompt_token_end_index is None:
                        
                    result.slice_data.goal = slice(              
                        base_prompt_token_start_index,
                        base_prompt_token_end_index
                    )
                    self.validate_slice_data('get_prompt (non-Python) - goal', result.slice_data, current_token_ids, current_decoded_tokens)
                    
                    #If the adversarial content is an empty string, make the slice an empty slice right after the goal slice
                    if working_adversarial_content.as_string == "":
                        result.slice_data.control = slice(result.slice_data.goal.stop, result.slice_data.goal.stop)
                    else:
                        working_adversarial_content_start_index = result.prompt.rindex(working_adversarial_content.as_string)
                        len_working_adversarial_content = len(working_adversarial_content.as_string)
                        working_adversarial_content_end_index = working_adversarial_content_start_index + len_working_adversarial_content
                        
                        if self.attack_state.log_manager.get_lowest_log_level() <= logging.DEBUG:
                            logger.debug(f"working_adversarial_content_start_index = {working_adversarial_content_start_index}, len_working_adversarial_content = {len_working_adversarial_content}, working_adversarial_content_end_index = {working_adversarial_content_end_index}")
                        working_adversarial_content_token_start_index = encoded_conversation_template_prompt.char_to_token(working_adversarial_content_start_index)
                        if working_adversarial_content_token_start_index is None:
                            raise PromptGenerationException("working_adversarial_content_token_start_index was None")
                        working_adversarial_content_token_end_index = encoded_conversation_template_prompt.char_to_token(working_adversarial_content_end_index)
                        if self.attack_state.log_manager.get_lowest_log_level() <= logging.DEBUG:
                            logger.debug(f"working_adversarial_content_token_start_index = {working_adversarial_content_token_start_index}, working_adversarial_content_token_end_index = {working_adversarial_content_token_end_index}")
                        if working_adversarial_content_token_end_index is None:
                            working_adversarial_content_token_end_index = working_adversarial_content_token_start_index + len(working_adversarial_content.token_ids)
                            if self.attack_state.log_manager.get_lowest_log_level() <= logging.DEBUG:
                                logger.debug(f"Updated working_adversarial_content_token_end_index to {working_adversarial_content_token_end_index} using fallback logic because it was None")
                        result.slice_data.control = slice(
                            working_adversarial_content_token_start_index,
                            working_adversarial_content_token_end_index
                        )
                    self.validate_slice_data('get_prompt (non-Python) - control', result.slice_data, current_token_ids, current_decoded_tokens)
                    # TKTK: END: update the goal and control slice logic to handle different placement of the adversarial content
                    
                    if self.attack_state.log_manager.get_lowest_log_level() <= logging.DEBUG:
                        logger.debug(f"finding conversation_template.roles[1] = '{conversation_template.roles[1]}' with length {len(conversation_template.roles[1])}.")
                    
                    last_token_index = result.prompt.rindex(conversation_template.roles[1])
                    
                    result.slice_data.assistant_role = slice(
                        encoded_conversation_template_prompt.char_to_token(last_token_index),
                        encoded_conversation_template_prompt.char_to_token(last_token_index + len(conversation_template.roles[1]) + 1)
                    )
                    
                    self.validate_slice_data('get_prompt (non-Python) - assistant_role', result.slice_data, current_token_ids, current_decoded_tokens)

                    self.print_slice_info("get_prompt (non-Python)", result.slice_data, current_token_ids)
                    if self.attack_state.log_manager.get_lowest_log_level() <= logging.DEBUG:
                        logger.debug(f"result.prompt = '{result.prompt}', self.attack_state.persistable.attack_params.target_output = '{self.attack_state.persistable.attack_params.target_output}'")
                    prompt_find_self_target = result.prompt.rindex(self.attack_state.persistable.attack_params.target_output)
                    if self.attack_state.log_manager.get_lowest_log_level() <= logging.DEBUG:
                        logger.debug(f"prompt_find_self_target = '{prompt_find_self_target}'")
                    prompt_find_self_target_c2t = encoded_conversation_template_prompt.char_to_token(prompt_find_self_target)
                    if prompt_find_self_target_c2t is None:
                        if self.attack_state.log_manager.get_lowest_log_level() <= logging.DEBUG:
                            logger.debug(f"got None for encoded_conversation_template_prompt.char_to_token(prompt_find_self_target). prompt_find_self_target = '{prompt_find_self_target}' using '{self.attack_state.persistable.attack_params.target_output}' in '{result.prompt}'. Using value {result.slice_data.assistant.stop} instead of None. This may indicate an error with the parsing logic.")
                        prompt_find_self_target_c2t = result.slice_data.assistant.stop
                    prompt_combined_c2t = None
                    add_length = len(self.attack_state.persistable.attack_params.target_output) + 1
                    while prompt_combined_c2 is None:
                        prompt_combined_c2t = encoded_conversation_template_prompt.char_to_token(prompt_find_self_target + (add_length))
                        add_length -= 1
                        if add_length < 0:
                            prompt_combined_c2t = prompt_find_self_target_c2t
                            logger.debug(f"Gave up trying to find prompt_combined_c2t and set it to prompt_find_self_target_c2t ({prompt_find_self_target_c2t}).")
                            break
                    # Subtract one more than the first valid value so that the length of the slice is correct
                    if self.attack_state.log_manager.get_lowest_log_level() <= logging.DEBUG:
                        logger.debug(f"prompt_find_self_target_c2t = '{prompt_find_self_target_c2t}', prompt_combined_c2t = '{prompt_combined_c2t}'")
                    result.slice_data.target_output = slice(
                        prompt_find_self_target_c2t,
                        prompt_combined_c2t + 1
                    )
                    self.validate_slice_data('get_prompt (non-Python) - target_output', result.slice_data, current_token_ids, current_decoded_tokens)

                    if self.attack_state.log_manager.get_lowest_log_level() <= logging.DEBUG:
                        logger.debug(f"self.attack_state.persistable.attack_params.loss_slice_mode = {self.attack_state.persistable.attack_params.loss_slice_mode} result.slice_data.assistant_role.start = {result.slice_data.assistant_role.start}, result.slice_data.target_output.start = {result.slice_data.target_output.start}, result.slice_data.target_output.stop = {result.slice_data.target_output.stop}")
                    if self.attack_state.persistable.attack_params.loss_slice_mode == LossSliceMode.ASSISTANT_ROLE_PLUS_FULL_TARGET_SLICE:
                        result.slice_data.loss = slice(result.slice_data.assistant_role.start, min(last_non_garbage_token, len(current_token_ids)))

                    if self.attack_state.persistable.attack_params.loss_slice_mode == LossSliceMode.ASSISTANT_ROLE_PLUS_TRUNCATED_TARGET_SLICE:
                        len_target_slice = result.slice_data.target_output.stop - result.slice_data.target_output.start
                        result.slice_data.loss = slice(result.slice_data.assistant_role.start, (result.slice_data.assistant_role.start + len_target_slice))
                        
                    if self.attack_state.persistable.attack_params.loss_slice_mode == LossSliceMode.INDEX_SHIFTED_TARGET_SLICE:
                        result.slice_data.loss = slice(
                            prompt_find_self_target_c2t + self.attack_state.persistable.attack_params.loss_slice_index_shift,
                            (prompt_combined_c2t + 1) + self.attack_state.persistable.attack_params.loss_slice_index_shift
                        )

                    if self.attack_state.persistable.attack_params.loss_slice_mode == LossSliceMode.SAME_AS_TARGET_SLICE :
                        result.slice_data.loss = slice(
                            prompt_find_self_target_c2t,
                            prompt_combined_c2t + 1
                        )
                    
                    if result.slice_data.loss is None:
                        raise PromptGenerationException("Did not find a valid loss slice mode")

                    self.validate_slice_data('get_prompt (non-Python) - loss', result.slice_data, current_token_ids, current_decoded_tokens)
                # There are so many garbagey fast tokenizers, this is the only way I can think of to handle them: try the fast tokenizer first and then fail over to the Python tokenizer.
                # I'd just use the Python tokenizer, but some models don't support it.
                except Exception as e:
                    python_tokenizer = True
                    logger.error(f"Exception thrown while using the fast tokenizer: {e}. Attempting to falling back to the Python tokenizer.")

        # handle buggy conversation templates that always insert a conversation role header even when no conversation messages are specified, and even when there is no system prompt.
        # having an inaccurate slice for the user role *shouldn't* affect anything, but it's possible some other code could depend on it.
        user_input_slice = result.slice_data.get_complete_user_input_slice()
        if user_input_slice is None:
            raise PromptGenerationException("[get_prompt] result.slice_data.get_complete_user_input_slice() returned None.")
        if user_input_slice.start is None or user_input_slice.stop is None:
            raise PromptGenerationException(f"[get_prompt] result.slice_data.get_complete_user_input_slice() returned {user_input_slice}, and neither start nor stop can be None.")        
        user_role_length = 0
        if result.slice_data.user_role is not None:
            if result.slice_data.user_role.start is not None and result.slice_data.user_role.stop is not None:
                user_role_start_index = result.slice_data.user_role.start
                user_role_stop_index = result.slice_data.user_role.stop
                user_role_length =  user_role_stop_index - user_role_start_index
        if user_role_length == 0:
            result.slice_data.user_role = slice(0, user_input_slice.start)

        if self.attack_state.log_manager.get_lowest_log_level() <= logging.DEBUG:
            logger.debug(f"conversation_template (after modifications) = '{conversation_template}'")
            final_decoded_toks = get_decoded_tokens(self.attack_state, current_token_ids)
            logger.debug(f"current_token_ids (after parsing) = '{current_token_ids}', final_decoded_toks = '{final_decoded_toks}'")
        
        if remove_tokens_after_target_output:
            current_token_ids = current_token_ids[:result.slice_data.target_output.stop]
        
        result.full_prompt_token_ids = current_token_ids
        result.input_token_ids = current_token_ids[:result.slice_data.target_output.stop]

        self.print_slice_info("get_prompt (non-Python)", result.slice_data, current_token_ids)

        #conversation_template.messages = []

        return result
