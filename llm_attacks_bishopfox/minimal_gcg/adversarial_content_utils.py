#!/bin/env python

import copy
import sys
import torch
# IMPORTANT: 'fastchat' is in the PyPi package 'fschat', not 'fastchat'!
import fastchat

from llm_attacks_bishopfox.dumpster_fires.trash_fire_tokens import get_decoded_token
from llm_attacks_bishopfox.dumpster_fires.trash_fire_tokens import get_decoded_tokens
from llm_attacks_bishopfox.dumpster_fires.trash_fire_tokens import get_encoded_token 
from llm_attacks_bishopfox.dumpster_fires.trash_fire_tokens import get_encoded_tokens 

from llm_attacks_bishopfox.attack.attack_classes import AdversarialContent
from llm_attacks_bishopfox.attack.attack_classes import AdversarialContentPlacement
from llm_attacks_bishopfox.attack.attack_classes import LossSliceMode
from llm_attacks_bishopfox.dumpster_fires.conversation_templates import get_llama2_and_3_fschat_template_names
from llm_attacks_bishopfox.dumpster_fires.trash_fire_tokens import encode_string_for_real_without_any_cowboy_funny_business
from llm_attacks_bishopfox.dumpster_fires.trash_fire_tokens import find_first_non_garbage_token
from llm_attacks_bishopfox.dumpster_fires.trash_fire_tokens import find_first_index_of_token
from llm_attacks_bishopfox.dumpster_fires.trash_fire_tokens import find_last_index_of_token
from llm_attacks_bishopfox.dumpster_fires.trash_fire_tokens import find_last_non_garbage_token
from llm_attacks_bishopfox.dumpster_fires.trash_fire_tokens import is_disastrous_dumpster_fire_token
from llm_attacks_bishopfox.json_serializable_object import JSONSerializableObject
from llm_attacks_bishopfox.util.util_functions import add_value_to_list_if_not_already_present
from llm_attacks_bishopfox.util.util_functions import find_first_occurrence_of_array_in_array
from llm_attacks_bishopfox.util.util_functions import find_index_of_first_nonmatching_element
from llm_attacks_bishopfox.util.util_functions import find_last_occurrence_of_array_in_array
from llm_attacks_bishopfox.util.util_functions import RequiredValueIsNoneException
from llm_attacks_bishopfox.util.util_functions import slice_from_dict

class PromptGenerationException(Exception):
    pass

def get_default_generic_role_indicator_template():
    # note: using "### {role}:" instead will cause issues 
    return "### {role}"

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
    return fastchat.conversation.get_conv_template(DEFAULT_CONVERSATION_TEMPLATE_NAME)

# def get_gemma_conversation_template():
    # conv_template = get_default_conversation_template().copy()
    # conv_template.name="gemma"
    # conv_template.system_message="<bos>"
    # conv_template.roles=("<start_of_turn>user\n", "<start_of_turn>model\n")
    # conv_template.sep_style=fastchat.conversation.SeparatorStyle.NO_COLON_SINGLE
    # conv_template.sep="<end_of_turn>\n"
    # conv_template.stop_str="<end_of_turn>"
    # return conv_template

def get_blenderbot_conversation_template():
    conv_template = get_default_conversation_template().copy()
    conv_template.name = "blenderbot"
    conv_template.system_template="{system_message}"
    conv_template.system_message=""
    conv_template.roles=tuple(["   ", "  "])
    conv_template.sep_style=fastchat.conversation.SeparatorStyle.NO_COLON_SINGLE
    conv_template.sep=""
    conv_template.sep2=""
    conv_template.stop_str=""
    conv_template.stop_str = "</s>"
    return conv_template

def get_gemma_conversation_template():
    conv_template = get_default_conversation_template().copy()
    conv_template.name="gemma"
    conv_template.system_message="<bos>"
    #conv_template.system_message=""
    #conv_template.roles=("<start_of_turn>user\n", "<start_of_turn>model\n")
    #conv_template.roles=("<start_of_turn>user", "<start_of_turn>model")
    #conv_template.roles=tuple(["<start_of_turn>user", "<start_of_turn>model"])
    conv_template.roles=tuple(["<start_of_turn>user\n", "<start_of_turn>model\n"])
    conv_template.sep_style=fastchat.conversation.SeparatorStyle.NO_COLON_SINGLE
    conv_template.sep="<end_of_turn>\n"
    #conv_template.stop_str="<end_of_turn>"
    conv_template.stop_str=""
    return conv_template

def get_guanaco_conversation_template():
    conv_template = fastchat.conversation.get_conv_template("zero_shot").copy()
    conv_template.name = "guanaco"
    sep=" ### "
    stop_str=""
    conv_template.stop_str = " </s>"
    return conv_template

def get_llama2_conversation_template():
    conv_template = fastchat.conversation.get_conv_template("llama-2").copy()
    conv_template.name = "llama2"
    conv_template.system_template = "<s>[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n"
    if conv_template.system_message is None:
        conv_template.system_message = ""
    #conv_template.sep_style=fastchat.conversation.SeparatorStyle.NO_COLON_SINGLE
    #roles=("</s><s>[INST]", "[/INST]"),
    #roles=("[INST]", "[/INST]"),
    #conv_template.sep = ' '
    #conv_template.sep2 = ' '
    #conv_template.sep = ' </s>'
    #conv_template.sep2 = ' </s><s>'
    #conv_template.sep = ' </s><s>'
    conv_template.stop_str = " </s>"
    return conv_template

def get_mpt_conversation_template():
    conv_template = fastchat.conversation.get_conv_template("mpt-7b-chat").copy()
    conv_template.name="mpt"
    return conv_template

def get_phi2_conversation_template():
    conv_template = get_default_conversation_template().copy()
    conv_template.name="phi2"
    conv_template.system_template = "System: {system_message}\n"
    conv_template.system_message="A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful answers to the user's questions."
    conv_template.roles = tuple(["User", "Assistant"])
    #conv_template.sep_style=fastchat.conversation.SeparatorStyle.NO_COLON_SINGLE
    conv_template.sep = '\n'
    conv_template.sep2 = '\n'
    return conv_template

def get_phi3_conversation_template():
    conv_template = get_default_conversation_template().copy()
    conv_template.name="phi3"
    #conv_template.system_template = "<|system|>\n{system_message}<|end|>\n"
    conv_template.system_template = "<|system|>\n{system_message}"
    conv_template.system_message=""
    #conv_template.roles = tuple(["\n<|user|>", "\n<|assistant|>"])
    conv_template.roles = tuple(["<|end|>\n<|user|>\n", "<|end|>\n<|assistant|>\n"])
    conv_template.sep_style=fastchat.conversation.SeparatorStyle.NO_COLON_SINGLE
    conv_template.sep = ''
    #conv_template.sep2 = ''
    conv_template.stop_str = "<|end|>\n<|endoftext|>"
    return conv_template

def get_qwen_conversation_template():
    conv_template = fastchat.conversation.get_conv_template("qwen-7b-chat").copy()
    conv_template.name="qwen"
    return conv_template

def get_qwen2_conversation_template():
    conv_template = get_qwen_conversation_template().copy()
    conv_template.name="qwen2"
    conv_template.system_message="You are a helpful assistant."
    return conv_template

def get_smollm_conversation_template():
    conv_template = get_default_conversation_template().copy()
    conv_template.name="smollm"
    conv_template.system_template="<|im_start|>system\n{system_message}"
    conv_template.system_message=""
    conv_template.roles=("<|im_start|>user", "<|im_start|>assistant")
    conv_template.sep_style=fastchat.conversation.SeparatorStyle.CHATML
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
    conv_template.name="stablelm2"
    conv_template.system_template = """<|im_start|>system
{system_message}"""
    conv_template.system_message="You are a helpful assistant."
    conv_template.roles = tuple(["<|im_start|>user", "<|im_start|>assistant"])
    conv_template.sep_style=fastchat.conversation.SeparatorStyle.CHATML
    conv_template.sep = "<|im_end|>"
    return conv_template

def register_missing_conversation_template(attack_params, fschat_added_support, template_name, template):
    register_conv_template = True
    
    if template_name in fastchat.conversation.conv_templates.keys():
        fschat_added_support.append(template_name)
        if not attack_params.override_fschat_templates:
            register_conv_template = False
    if register_conv_template:
        fastchat.conversation.register_conv_template(template = template, override = True)
        
    return fschat_added_support

# override = True is required because otherwise adding a missing template will fail with an assertion
def register_missing_conversation_templates(attack_params):
    fschat_added_support = []

    register_missing_conversation_template(attack_params, fschat_added_support, "blenderbot", get_blenderbot_conversation_template())
    
    register_missing_conversation_template(attack_params, fschat_added_support, "gemma", get_gemma_conversation_template())
    
    register_missing_conversation_template(attack_params, fschat_added_support, "guanaco", get_guanaco_conversation_template())
    
    register_missing_conversation_template(attack_params, fschat_added_support, "llama2", get_llama2_conversation_template())
    
    register_missing_conversation_template(attack_params, fschat_added_support, "mpt", get_mpt_conversation_template())
    
    register_missing_conversation_template(attack_params, fschat_added_support, "phi2", get_phi2_conversation_template())
    register_missing_conversation_template(attack_params, fschat_added_support, "phi3", get_phi3_conversation_template())
    
    register_missing_conversation_template(attack_params, fschat_added_support, "qwen", get_qwen_conversation_template())
    register_missing_conversation_template(attack_params, fschat_added_support, "qwen2", get_qwen2_conversation_template())
    
    register_missing_conversation_template(attack_params, fschat_added_support, "smollm", get_smollm_conversation_template())
    
    register_missing_conversation_template(attack_params, fschat_added_support, "stablelm2", get_stablelm2_conversation_template())

    # if "gemma" in fastchat.conversation.conv_templates.keys():
        # fschat_added_support.append("gemma")
    # else:
        # fastchat.conversation.register_conv_template(template = get_gemma_conversation_template(), override = True)

    # if "phi2" in fastchat.conversation.conv_templates.keys():
        # fschat_added_support.append("phi2")
    # else:
        # fastchat.conversation.register_conv_template(template = get_phi2_conversation_template(), override = True)

    # if "phi3" in fastchat.conversation.conv_templates.keys():
        # fschat_added_support.append("phi3")
    # else:
        # fastchat.conversation.register_conv_template(template = get_phi3_conversation_template(), override = True)

    # # For some reason, fschat calls their "qwen" template "qwen-7b-chat" specifically, so this code adds a shortcut
    # if "qwen" in fastchat.conversation.conv_templates.keys():
        # fschat_added_support.append("qwen")
    # else:        
        # fastchat.conversation.register_conv_template(template = get_qwen_conversation_template(), override = True)

    # # Qwen2 *seems* to use the same chat template format at Qwen
    # if "qwen2" in fastchat.conversation.conv_templates.keys():
        # fschat_added_support.append("qwen2")
    # else:        
        # fastchat.conversation.register_conv_template(template = get_qwen2_conversation_template(), override = True)
        
    # if "stablelm2" in fastchat.conversation.conv_templates.keys():
        # fschat_added_support.append("stablelm2")
    # else:
        # fastchat.conversation.register_conv_template(template = get_stablelm2_conversation_template(), override = True)

    if len(fschat_added_support) > 0:
        if attack_params.template_name in fschat_added_support:
            #added_support_message = f"[register_missing_conversation_templates] Warning: the fschat (fastchat) library appears to have added support for the following model(s) that previously required custom definitions: {fschat_added_support}. "
            added_support_message = f"Warning: the fschat library appears to have added support for the template '{attack_params.template_name}'. The corresponding model previously required a custom definition included in Broken Hill. "
            if attack_params.override_fschat_templates:
                added_support_message += f"Because the --override-fschat-templates option was specified, the custom Broken Hill definition will be used instead."
            else:
                added_support_message += f"The fschat template will be used. If you receive warnings or errors from the conversation template self-test, try specifying the --override-fschat-templates option to use the custom Broken Hill definition instead."
            print(added_support_message)

def load_conversation_template(model_path, template_name = None, generic_role_indicator_template = None, system_prompt = None, clear_existing_template_conversation = False, conversation_template_messages=None):
    #print(f"[load_conversation_template] Debug: loading chat template '{template_name}'. generic_role_indicator_template='{generic_role_indicator_template}', system_prompt='{system_prompt}', clear_existing_template_conversation='{clear_existing_template_conversation}'")
    conv_template = None
    
    if template_name is not None:
        if template_name not in fastchat.conversation.conv_templates.keys():
            print(f"[load_conversation_template] Warning: chat template '{template_name}' was not found in fastchat - defaulting to '{DEFAULT_CONVERSATION_TEMPLATE_NAME}'.")
            template_name = DEFAULT_CONVERSATION_TEMPLATE_NAME
        #print(f"[load_conversation_template] Debug: loading chat template '{template_name}'")
        conv_template = fastchat.conversation.get_conv_template(template_name)
    else:
        #print(f"[load_conversation_template] Debug: determining chat template based on content in '{model_path}'")
        conv_template = fastchat.model.get_conversation_template(model_path)
    # make sure fastchat doesn't sneak the one_shot messages in when zero_shot was requested
    if clear_existing_template_conversation:
        if hasattr(conv_template, "messages"):
            #print(f"[load_conversation_template] Debug: resetting conv_template.messages from '{conv_template.messages}' to []")
            conv_template.messages = []
        else:
            print("[load_conversation_template] Warning: the option to clear the conversation template's default conversation was enabled, but the template does not include a default conversation.")
            conv_template.messages = []
    generic_role_template = get_default_generic_role_indicator_template()
    if generic_role_indicator_template is not None:
        # If using a custom role indicator template, just use a space and depend on the operator to specify any necessary characters such as :
        conv_template.sep_style = fastchat.conversation.SeparatorStyle.NO_COLON_SINGLE
        #generic_role_template = f"\n{generic_role_indicator_template}"
        generic_role_template = f" {generic_role_indicator_template}"
        #generic_role_template = generic_role_indicator_template        
        conv_template.sep = '\n'
    # note: the original logic was equivalent to the following:
    #generic_role_template = "### {role}"
    if conv_template.name == 'zero_shot':# or conv_template.name == 'one_shot':
        #conv_template.roles = tuple(['### ' + r for r in conv_template.roles])
        conv_template.roles = tuple([generic_role_template.format(role=r) for r in conv_template.roles])
        conv_template.sep = "\n "
        conv_template.sep2 = "\n "
    if generic_role_indicator_template is not None:
        conv_template.roles = tuple([generic_role_template.format(role=r) for r in conv_template.roles])
    #if conv_template.name == 'llama-2':
    #    conv_template.sep2 = conv_template.sep2.strip()
    if system_prompt is not None:
        if hasattr(conv_template, "system_message"):
            original_system_message = conv_template.system_message
            conv_template.system_message = system_prompt
            #print(f"[load_conversation_template] Debug: replaced default system message '{original_system_message}' with '{system_prompt}'.")
        else:
            print("[load_conversation_template] Warning: the option to set the conversation template's system message was enabled, but the template does not include a system message.")
    if conversation_template_messages is not None:
        if not hasattr(conv_template, "messages"):
            conv_template.messages = []
        #print(f"[load_conversation_template] Debug: existing conversation template messages '{conv_template.messages}'.")
        for i in range(0, len(conversation_template_messages)):
            role_id_or_name = conversation_template_messages[i][0]
            message = conversation_template_messages[i][1]
            # If role IDs were specified, convert them to the correct format for the template
            if isinstance(role_id_or_name, int):
                try:
                    role_id_or_name = conv_template.roles[role_id_or_name]
                except Exception as e:
                    raise Exception("Could not convert the role ID '{}' to an entry in the template's list of roles ('{conv_template.roles}'): {e}")
            conv_template.messages.append((role_id_or_name, message))
        #print(f"[load_conversation_template] Debug: customized conversation template messages: '{conv_template.messages}'.")
    
    return conv_template

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
        return JSONSerializableObject.json_dumps(self.to_dict())
    
    def copy(self):
        return PromptSliceData.from_dict(self.to_dict())
    
    @staticmethod
    def from_json(json_string):
        return PromptSliceData.from_dict(json.loads(json_string))

class PromptAndInputIDCollection(JSONSerializableObject):
    def __init__(self):
        self.prompt = None
        self.full_prompt_token_ids = None
        self.input_token_ids = None
        self.slice_data = PromptSliceData()
    
    def get_input_ids_as_tensor(self):
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
        return JSONSerializableObject.json_dumps(self.to_dict())
    
    def copy(self):
        return PromptAndInputIDCollection.from_dict(self.to_dict())
    
    @staticmethod
    def from_json(json_string):
        return PromptAndInputIDCollection.from_dict(json.loads(json_string))


# TKTK: expand to prefix/suffix attack, and also interleaving the tokens into the base string.
class AdversarialContentManager:
    def __init__(self, *, attack_params, tokenizer, conv_template, adversarial_content, trash_fire_tokens):

        self.attack_params = attack_params
        self.tokenizer = tokenizer
        self.conv_template = conv_template
        self.adversarial_content = adversarial_content
        if isinstance(self.attack_params.loss_slice_mode, type(None)):
            raise RequiredValueIsNoneException("self.attack_params.loss_slice_mode cannot be None")
        self.trash_fire_tokens = trash_fire_tokens

    # For debugging / creating handlers for new conversation templates
    # accepts a dictionary of slices, where the key is the slice name and the value is the slice
    # and the list of tokens the slices refer to
    def get_slice_info(self, slice_data, tokens):
        result = {}
        decoded_tokens = get_decoded_tokens(self.tokenizer, tokens)
        #print(f"[get_slice_info] Debug: len(tokens) = {len(tokens)}, tokens = '{tokens}', decoded_tokens = '{decoded_tokens}'")
        slice_dictionary = slice_data.get_slice_dictionary()
        for slice_name in slice_dictionary.keys():
            sl = slice_dictionary[slice_name]
            if sl is not None:
                slice_tokens = tokens[sl]
                slice_tokens_decoded = decoded_tokens[sl]
                result[slice_name] = slice_tokens_decoded
        return result
                
    def print_slice_info(self, source_method_name, slice_data, tokens):
        #print(f"[print_slice_info] Debug: len(tokens) = {len(tokens)}, tokens = '{tokens}'")
        slice_info = self.get_slice_info(slice_data, tokens)
        slice_dictionary = slice_data.get_slice_dictionary()
        for slice_name in slice_info.keys():
            #print(f"[{source_method_name}] Debug: slice '{slice_name}' decoded tokens = '{slice_info[slice_name]}'")
            print(f"[{source_method_name}] Debug: slice '{slice_name}' decoded tokens = '{slice_info[slice_name]}', tokens = {tokens[slice_dictionary[slice_name]]}")

    def validate_slice_data(self, source_method_name, slice_data, token_ids, decoded_tokens):
        invalid_slice_dictionary = {}
        #print(f"[validate_slice_data - {source_method_name}] Debug: token_ids = {token_ids}, decoded_tokens = {decoded_tokens}")
        slice_dictionary = slice_data.get_slice_dictionary()
        for slice_name in slice_dictionary.keys():
            sl = slice_dictionary[slice_name]
            if sl is not None:
                is_valid = True
                if isinstance(sl.start, type(None)) or isinstance(sl.stop, type(None)):
                    is_valid = False
                if not is_valid:
                    # The system slice having a start of None is expected
                    if slice_name == "system":
                        if isinstance(sl.start, type(None)) and not isinstance(sl.stop, type(None)):
                            is_valid = True
                if not is_valid:
                    invalid_slice_dictionary[slice_name] = sl
                #print(f"[validate_slice_data - {source_method_name}] Debug: slice '{slice_name}' = '{sl}', is_valid = {is_valid}.")
        if len(invalid_slice_dictionary.keys()) > 0:
            message = f"[{source_method_name}] Warning: one or more slices have None values instead of start or stop values. This generally indicates an issue with the tokenizing or parsing logic. The slice(s) with None values are: "
            for slice_name in invalid_slice_dictionary.keys():
                sl = invalid_slice_dictionary[slice_name]
                message += f"{slice_name}: {sl},"
            message = message[:-1]
            print(message)
        #self.print_slice_info(f"validate_slice_data - {source_method_name}", slice_data, token_ids)

    def extend_slice_if_next_token_is_in_list(self, decoded_tokens, current_slice, token_string_list):
        #print(f"[extend_slice_if_next_token_is_in_list] Debug: decoded_tokens: {decoded_tokens}, decoded current slice is {decoded_tokens[current_slice]}.")
        result = current_slice
        next_token_stop = current_slice.stop + 1
        if (next_token_stop) <= len(decoded_tokens):
            next_token = decoded_tokens[current_slice.stop]
            if next_token in token_string_list or next_token.strip() in token_string_list:
                new_slice = slice(current_slice.start, next_token_stop)
                #print(f"[extend_slice_if_next_token_is_in_list] Debug: next token is '{next_token}', extending slice stop by 1. Slice contents were {decoded_tokens[current_slice]}, now {decoded_tokens[new_slice]}.")
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
    # Once you make that connection, a lot of the potential failure modes of this tool start to make more sense. For example, sending the conversations to the LLM in the correct format (including using the correct token *IDs*, not just identical text) is vital, because otherwise you're either not going to escape out of the user-input context, *or* you're going to cause the LLM to generate a pseudo-conversation between a human and an LLM, without responding the way it would when prompted in the format it expects. Adversarial content generated in this way is unlikely to work against other implementations of the same LLM, because those implementations will most likely send conversations to the LLM in the correct format.

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
        start_index = prompt_and_input_id_data.slice_data.goal.start
        if prompt_and_input_id_data.slice_data.control.start < start_index:
            start_index = prompt_and_input_id_data.slice_data.control.start
        end_index = prompt_and_input_id_data.slice_data.control.stop
        if prompt_and_input_id_data.slice_data.goal.stop > end_index:
            end_index = prompt_and_input_id_data.slice_data.goal.stop
        # increment by one to include the last token
        #end_index += 1
        return prompt_and_input_id_data.full_prompt_token_ids[start_index:end_index]
    
    #prompt_and_input_id_data should be a PromptAndInputIDCollection
    def get_complete_input_string(self, prompt_and_input_id_data):
        return self.tokenizer.decode(self.get_complete_input_token_ids(prompt_and_input_id_data))
    
    def conversation_template_appends_colon_to_role_names(self):
        result = False
        if self.conv_template.sep_style == fastchat.conversation.SeparatorStyle.ADD_COLON_SINGLE:
            result = True
        if self.conv_template.sep_style == fastchat.conversation.SeparatorStyle.ADD_COLON_TWO:
            result = True
        if self.conv_template.sep_style == fastchat.conversation.SeparatorStyle.ADD_COLON_SPACE_SINGLE:
            result = True
        #print(f"[conversation_template_appends_colon_to_role_names] Debug: self.conv_template.sep_style = {self.conv_template.sep_style}, result = {result}")
        return result

    # def get_prompt(self, adversarial_content = None, force_python_tokenizer = False):#, update_self_values = True):

        # result = PromptAndInputIDCollection()
        
        # # set up temporary values based on permanent values
        # working_adversarial_content = self.adversarial_content.copy()
        # conversation_template = self.conv_template.copy()
        
        # if conversation_template is None:
            # print(f"[get_prompt] Error: got a null conversation template when trying to call self.conv_template.copy(). This should never happen. self.conv_template was '{self.conv_template}'")
            # sys.exit(1)
        
        # if adversarial_content is not None:
            # working_adversarial_content = adversarial_content.copy()

        # # Get the list of dumpster fire tokens once to make the many references to it run faster
        # #hardcoded_trash_fire_tokens = TrashFireTokenCollection.get_meticulously_curated_trash_fire_token_collection(self.tokenizer, conversation_template)

        # conversation_template.append_message(conversation_template.roles[0], f"{self.attack_params.base_prompt} {working_adversarial_content.as_string}")
        # conversation_template.append_message(conversation_template.roles[1], f"{self.attack_params.target_output}")
        # result.prompt = conversation_template.get_prompt()

        # encoded_conversation_template_prompt = self.tokenizer(result.prompt)
        # toks = encoded_conversation_template_prompt.input_ids
        # #original_toks = copy.deepcopy(toks)
        # #original_decoded_tokens = get_decoded_tokens(self.tokenizer, original_toks)
       
        # python_tokenizer = False
        # # Using the non-Python tokenizer is totally broken right now, because it assumes that e.g. the first occurence of a role name is the correct location to use, even for chat templates with messages
        # #python_tokenizer = True
        # if conversation_template.name == 'oasst_pythia':
            # python_tokenizer = True
        # if force_python_tokenizer:
            # python_tokenizer = True
        # #if "pythia" in conversation_template.name:
        # #    python_tokenizer = True
        # # This (formerly undocumented) check is a way to determine if the model is using Python-based tokenizers. It works because Python-based tokenizers (at least in the current version of Transformers) don't support the char_to_token operation), and it's used to avoid calling char_to_token for the rest of the get_prompt method in that case.
        # if not python_tokenizer:
            # try:
                # encoded_conversation_template_prompt.char_to_token(len(result.prompt)-1)
            # except:
                # python_tokenizer = True

        # if python_tokenizer:
            # # TKTK: consider rewriting this to not use fschat at all.
            # # Using apply_chat_template where available and including custom templates for models that don't include it might be easier.
            # print(f"[get_prompt] Info: using Python tokenizer.")
            # # reset the conversation template (to get rid of the messages that have been added earlier in this function), but preserve any messages that are part of the template
            # conversation_template = self.conv_template.copy()
            
            # role_extension_token_list = []
            # if self.conversation_template_appends_colon_to_role_names():
                # role_extension_token_list = add_value_to_list_if_not_already_present(role_extension_token_list, ":")
            # if conversation_template.name == "llama-3":
                # role_extension_token_list = add_value_to_list_if_not_already_present(role_extension_token_list, "<|end_header_id|>")
            
            # conversation_template.append_message(conversation_template.roles[0], None)
            # current_prompt = conversation_template.get_prompt()
            # print(f"[get_prompt] Debug: current_prompt = '{current_prompt}'")
            # toks = self.tokenizer(current_prompt).input_ids
            # decoded_toks = get_decoded_tokens(self.tokenizer, toks)
            # # find the token that indicates the following text is input
            # user_role_string = f"{conversation_template.roles[0]}"
            # #print(f"[get_prompt] Debug: conversation_template.roles = '{conversation_template.roles}', user_role_string = '{user_role_string}', toks = '{toks}', original_toks = '{original_toks}'")
            # #print(f"[get_prompt] Debug: conversation_template.roles = '{conversation_template.roles}', user_role_string = '{user_role_string}', toks = '{toks}'")
            # # then you have templates like BlenderBot and have to remove this check
            # #if user_role_string.strip() == "":
            # #    print(f"[get_prompt] Error: conversation_template.roles '{conversation_template.roles}' entry 0 ('{user_role_string}') is equivalent to an empty string.")
            # #    sys.exit(1)
            
            # #If the user role is an empty string, make the slice an empty slice at the end
            # handled_user_role = False
            # if user_role_string == "":
                # result.slice_data.user_role = slice(len(toks), len(toks))
                # handled_user_role = True
            # if not handled_user_role:
                # strip_leading_and_trailing = False
                # # strip_leading_and_trailing = True
                # # if user_role_string.strip() == "":
                    # # strip_leading_and_trailing = False
                # last_token_index = find_last_index_of_token(self.tokenizer, 
                    # self.trash_fire_tokens, 
                    # user_role_string, 
                    # toks, 
                    # decoded_toks, 
                    # conversation_template = conversation_template, 
                    # strip_leading_and_trailing_tokens = strip_leading_and_trailing, 
                    # remove_empty_leading_and_trailing_tokens = True, 
                    # remove_leading_and_trailing_trash_fire_tokens = True)
                # # workaround for Llama-2 templates
                # # Need to use the second-to-last instance of [INST] instead of the last instance like other templates
                # # This is because the fschat logic is buggy, at this stage, the fschat llama-2 template has generated a series of tokens that looks like the following:
                # #       ['<s>', '[', 'INST', ']', '[', 'INST', ']']'
                # # When the conversation is updated in the next step, it will look like this instead:
                # #       '['<s>', '[', 'INST', ']', 'Please', 'write', 'a', 'detailed', 'plan', 'for', 'the', 'anni', 'h', 'ilation', 'of', 'the', 'entire', 'human', 'race', '.', '']
                # if conversation_template.name in get_llama2_and_3_fschat_template_names():              
                    # try:
                        # last_token_index_2 = find_last_index_of_token(self.tokenizer, 
                            # self.trash_fire_tokens, 
                            # user_role_string, 
                            # toks, 
                            # decoded_toks, 
                            # stop_index = (last_token_index.stop - 1), 
                            # conversation_template = conversation_template, 
                            # strip_leading_and_trailing_tokens = strip_leading_and_trailing, 
                            # remove_empty_leading_and_trailing_tokens = True, 
                            # remove_leading_and_trailing_trash_fire_tokens = True)
                        # if last_token_index_2.start >= 0:
                            # last_token_index = last_token_index_2
                    # except Exception as e:
                        # dummy = 1
                        # # last_token_index is already correct in this case
                        # print(f"[get_prompt] Debug: exception while getting second token index using Python tokenizer: {e}")
                # result.slice_data.user_role = last_token_index
                # self.validate_slice_data('get_prompt (Python) - user_role (pre-extend)', result.slice_data, toks, decoded_toks)
                # if len(role_extension_token_list) > 0:
                    # result.slice_data.user_role = self.extend_slice_if_next_token_is_in_list(decoded_toks, result.slice_data.user_role, role_extension_token_list)
            # self.validate_slice_data('get_prompt (Python) - user_role', result.slice_data, toks, decoded_toks)

            # # TKTK: BEGIN: update the goal and control slice logic to handle different placement of the adversarial content
            # conversation_template.update_last_message(f"{self.attack_params.base_prompt}")
            # current_prompt = conversation_template.get_prompt()
            # print(f"[get_prompt] Debug: current_prompt = '{current_prompt}'")
            # toks = self.tokenizer(current_prompt).input_ids
            # decoded_toks = get_decoded_tokens(self.tokenizer, toks)
            # first_non_garbage_token = find_first_non_garbage_token(conversation_template, toks, decoded_toks, self.trash_fire_tokens, start_index = result.slice_data.user_role.stop)
            # last_non_garbage_token = find_last_non_garbage_token(conversation_template, toks, decoded_toks, self.trash_fire_tokens, start_index = first_non_garbage_token) + 1
            # result.slice_data.goal = slice(first_non_garbage_token, min(last_non_garbage_token, len(toks)))
            # self.validate_slice_data('get_prompt (Python) - goal', result.slice_data, toks, decoded_toks)

            # separator = ' '
            # if not self.attack_params.base_prompt:
                # separator = ''
            # #If the adversarial content is an empty string, make the slice an empty slice right after the goal slice
            # if working_adversarial_content.as_string == "":
                # result.slice_data.control = slice(result.slice_data.goal.stop, result.slice_data.goal.stop)
            # else:
                # conversation_template.update_last_message(f"{self.attack_params.base_prompt}{separator}{working_adversarial_content.as_string}")
                # toks = self.tokenizer(conversation_template.get_prompt()).input_ids
                # decoded_toks = get_decoded_tokens(self.tokenizer, toks)
                # first_non_garbage_token = find_first_non_garbage_token(conversation_template, toks, decoded_toks, self.trash_fire_tokens, start_index = result.slice_data.goal.stop)
                # last_non_garbage_token = find_last_non_garbage_token(conversation_template, toks, decoded_toks, self.trash_fire_tokens, start_index = first_non_garbage_token) + 1
                # result.slice_data.control = slice(first_non_garbage_token, min(last_non_garbage_token, len(toks)))
            # self.validate_slice_data('get_prompt (Python) - control', result.slice_data, toks, decoded_toks)

            # # TKTK: END: update the goal and control slice logic to handle different placement of the adversarial content

            # # find the token that marks a transition to output
            # #print(f"[get_prompt] Debug: appending conversation_template.roles[1] = '{conversation_template.roles[1]}'")
            # conversation_template.append_message(conversation_template.roles[1], None)
            # current_prompt = conversation_template.get_prompt()
            # print(f"[get_prompt] Debug: current_prompt = '{current_prompt}'")
            # toks = self.tokenizer(current_prompt).input_ids
            # decoded_toks = get_decoded_tokens(self.tokenizer, toks)
            # #print(f"[get_prompt] Debug: conversation with assistant role tokens = '{decoded_toks}'")
            # handled_assistant_role = False
            # assistant_role_string = conversation_template.roles[1]
            # if assistant_role_string == "":
                # result.slice_data.assistant_role = slice(len(toks), len(toks))
                # handled_assistant_role = True
            # if not handled_assistant_role:
                # if assistant_role_string.strip() == "":
                    # last_token_index = find_last_index_of_token(self.tokenizer, 
                        # self.trash_fire_tokens, 
                        # assistant_role_string, 
                        # toks, 
                        # decoded_toks, 
                        # conversation_template = conversation_template, 
                        # strip_leading_and_trailing_tokens = False, 
                        # remove_empty_leading_and_trailing_tokens = True, 
                        # remove_leading_and_trailing_trash_fire_tokens = True)
                    # result.slice_data.assistant_role = last_token_index
                # else:
                    # first_non_garbage_token = find_first_non_garbage_token(conversation_template, toks, decoded_toks, self.trash_fire_tokens, start_index = result.slice_data.control.stop)
                    # last_non_garbage_token = find_last_non_garbage_token(conversation_template, toks, decoded_toks, self.trash_fire_tokens, start_index = first_non_garbage_token) + 1
                    # result.slice_data.assistant_role = slice(first_non_garbage_token, min(last_non_garbage_token, len(toks)))
                # if len(role_extension_token_list) > 0:
                    # result.slice_data.assistant_role = self.extend_slice_if_next_token_is_in_list(decoded_toks, result.slice_data.assistant_role, role_extension_token_list)
            # self.validate_slice_data('get_prompt (Python) - assistant_role', result.slice_data, toks, decoded_toks)

            # conversation_template.update_last_message(f"{self.attack_params.target_output}")
            # current_prompt = conversation_template.get_prompt()
            # print(f"[get_prompt] Debug: current_prompt = '{current_prompt}'")
            # toks = self.tokenizer(current_prompt).input_ids
            # decoded_toks = get_decoded_tokens(self.tokenizer, toks)
            # first_non_garbage_token = find_first_non_garbage_token(conversation_template, toks, decoded_toks, self.trash_fire_tokens, start_index = result.slice_data.assistant_role.stop)
            # last_non_garbage_token = find_last_non_garbage_token(conversation_template, toks, decoded_toks, self.trash_fire_tokens, start_index = first_non_garbage_token) + 1
            # result.slice_data.target_output = slice(first_non_garbage_token, min(last_non_garbage_token, len(toks)))
            # self.validate_slice_data('get_prompt (Python) - target_output', result.slice_data, toks, decoded_toks)
            
            # #print(f"[get_prompt] Debug: getting loss slice using mode {self.attack_params.loss_slice_mode}")
            # if self.attack_params.loss_slice_mode == LossSliceMode.ASSISTANT_ROLE_PLUS_FULL_TARGET_SLICE:                
                # result.slice_data.loss = slice(result.slice_data.assistant_role.start, min(last_non_garbage_token, len(toks)))

            # if self.attack_params.loss_slice_mode == LossSliceMode.ASSISTANT_ROLE_PLUS_TRUNCATED_TARGET_SLICE:
                # len_target_slice = result.slice_data.target_output.stop - result.slice_data.target_output.start
                # result.slice_data.loss = slice(result.slice_data.assistant_role.start, (result.slice_data.assistant_role.start + len_target_slice))
                
            # if self.attack_params.loss_slice_mode == LossSliceMode.INDEX_SHIFTED_TARGET_SLICE:
                # #result.slice_data.loss = slice(first_non_garbage_token - 1, min(last_non_garbage_token, len(toks)) - 1)
                # result.slice_data.loss = slice(int(result.slice_data.target_output.start) + self.attack_params.loss_slice_index_shift, min(int(result.slice_data.target_output.stop) + self.attack_params.loss_slice_index_shift, len(toks)))

            # if self.attack_params.loss_slice_mode == LossSliceMode.SAME_AS_TARGET_SLICE:
                # result.slice_data.loss = slice(first_non_garbage_token, min(last_non_garbage_token, len(toks)))
            
            # if result.slice_data.loss is None:
                # raise PromptGenerationException("Did not find a valid loss slice mode")

            # self.validate_slice_data('get_prompt (Python) - loss', result.slice_data, toks, decoded_toks)
            
        # else:
            # print(f"[get_prompt] Info: not using Python tokenizer")
            # sys_template = None
            # if hasattr(conversation_template, "system"):
                # sys_template = conversation_template.system
            # if sys_template is None and hasattr(conversation_template, "system_template"):
                # sys_template = conversation_template.system_template
            # if sys_template is None:
                # print(f"[get_prompt] Warning: unable to find system template in conversation template for this model - using role 0 template instead")
                # sys_template = conversation_template.roles[0]
            # result.slice_data.system = slice(
                # None, 
                # encoded_conversation_template_prompt.char_to_token(len(sys_template))
            # )
            # self.validate_slice_data('get_prompt (non-Python) - system', result.slice_data, toks, decoded_toks)

            # # result.slice_data.user_role = slice(
                # # encoded_conversation_template_prompt.char_to_token(result.prompt.find(conversation_template.roles[0])),
                # # encoded_conversation_template_prompt.char_to_token(result.prompt.find(conversation_template.roles[0]) + len(conversation_template.roles[0]) + 1)
            # # )
            # last_token_index = result.prompt.rindex(conversation_template.roles[0])
            # if conversation_template.name in get_llama2_and_3_fschat_template_names():
                # try:
                    # last_token_index_2 = result.prompt.rindex(conversation_template.roles[0], 0, last_token_index)
                    # last_token_index = last_token_index_2
                # except Exception as e:
                    # dummy = 1
                    # # last_token_index is already correct
                    # print(f"[get_prompt] Debug: exception while getting second token index for user role using fast tokenizer: {e}")
            
            # result.slice_data.user_role = slice(
                # encoded_conversation_template_prompt.char_to_token(last_token_index),
                # encoded_conversation_template_prompt.char_to_token(last_token_index + len(conversation_template.roles[0]) + 1)
            # )
            
            # self.validate_slice_data('get_prompt (non-Python) - user_role', result.slice_data, toks, decoded_toks)

            # # TKTK: BEGIN: update the goal and control slice logic to handle different placement of the adversarial content
            # result.slice_data.goal = slice(
                # encoded_conversation_template_prompt.char_to_token(result.prompt.find(self.attack_params.base_prompt)),
                # encoded_conversation_template_prompt.char_to_token(result.prompt.find(self.attack_params.base_prompt) + len(self.attack_params.base_prompt))
            # )
            # self.validate_slice_data('get_prompt (non-Python) - goal', result.slice_data, toks, decoded_toks)
            
            # #If the adversarial content is an empty string, make the slice an empty slice right after the goal slice
            # if working_adversarial_content.as_string == "":
                # result.slice_data.control = slice(result.slice_data.goal.stop, result.slice_data.goal.stop)
            # else:
                # result.slice_data.control = slice(
                    # encoded_conversation_template_prompt.char_to_token(result.prompt.find(working_adversarial_content.as_string)),
                    # encoded_conversation_template_prompt.char_to_token(result.prompt.find(working_adversarial_content.as_string) + len(working_adversarial_content.as_string))
                # )
            # self.validate_slice_data('get_prompt (non-Python) - control', result.slice_data, toks, decoded_toks)
            # # TKTK: END: update the goal and control slice logic to handle different placement of the adversarial content
            
            # #print(f"[get_prompt] Debug: finding conversation_template.roles[1] = '{conversation_template.roles[1]}' with length {len(conversation_template.roles[1])}.")
            # # result.slice_data.assistant_role = slice(
                # # encoded_conversation_template_prompt.char_to_token(result.prompt.find(conversation_template.roles[1])),
                # # encoded_conversation_template_prompt.char_to_token(result.prompt.find(conversation_template.roles[1]) + len(conversation_template.roles[1]) + 1)
            # # )
            
            # last_token_index = result.prompt.rindex(conversation_template.roles[1])
            # # if conversation_template.name in get_llama2_and_3_fschat_template_names():
                # # try:
                    # # last_token_index_2 = result.prompt.rindex(conversation_template.roles[1], 0, last_token_index)
                    # # last_token_index = last_token_index_2
                # # except Exception as e:
                    # # dummy = 1
                    # # # last_token_index is already correct
                    # # print(f"[get_prompt] Debug: exception while getting second token index for assistant role using fast tokenizer: {e}")
            
            # result.slice_data.assistant_role = slice(
                # encoded_conversation_template_prompt.char_to_token(last_token_index),
                # encoded_conversation_template_prompt.char_to_token(last_token_index + len(conversation_template.roles[1]) + 1)
            # )
            
            # self.validate_slice_data('get_prompt (non-Python) - assistant_role', result.slice_data, toks, decoded_toks)

            # #self.print_slice_info("get_prompt", result.slice_data, toks)
            # #print(f"[get_prompt] Debug: result.prompt = '{result.prompt}', self.attack_params.target_output = '{self.attack_params.target_output}'")
            # prompt_find_self_target = result.prompt.find(self.attack_params.target_output)
            # #print(f"[get_prompt] Debug: prompt_find_self_target = '{prompt_find_self_target}'")
            # prompt_find_self_target_c2t = encoded_conversation_template_prompt.char_to_token(prompt_find_self_target)
            # if isinstance(prompt_find_self_target_c2t, type(None)):
                # print(f"[get_prompt] Warning: got None for encoded_conversation_template_prompt.char_to_token(prompt_find_self_target). prompt_find_self_target = '{prompt_find_self_target}' using '{self.attack_params.target_output}' in '{result.prompt}'. Using value {result.slice_data.assistant.stop} instead of None. This may indicate an error with the parsing logic.")
                # prompt_find_self_target_c2t = result.slice_data.assistant.stop
            # prompt_combined_c2t = None
            # add_length = len(self.attack_params.target_output) + 1
            # while isinstance(prompt_combined_c2t, type(None)):
                # prompt_combined_c2t = encoded_conversation_template_prompt.char_to_token(prompt_find_self_target + (add_length))
                # add_length -= 1
                # if add_length < 0:
                    # prompt_combined_c2t = prompt_find_self_target_c2t
                    # break
            # # Subtract one more than the first valid value so that the length of the slice is correct
            # #print(f"[get_prompt] Debug: prompt_find_self_target_c2t = '{prompt_find_self_target_c2t}', prompt_combined_c2t = '{prompt_combined_c2t}'")
            # result.slice_data.target_output = slice(
                # prompt_find_self_target_c2t,
                # prompt_combined_c2t + 1
            # )
            # self.validate_slice_data('get_prompt (non-Python) - target_output', result.slice_data, toks, decoded_toks)
            
            # #print(f"[get_prompt] Debug: getting loss slice using mode {self.attack_params.loss_slice_mode}")
            # if self.attack_params.loss_slice_mode == LossSliceMode.ASSISTANT_ROLE_PLUS_FULL_TARGET_SLICE:
                # result.slice_data.loss = slice(result.slice_data.assistant_role.start, min(last_non_garbage_token, len(toks)))

            # if self.attack_params.loss_slice_mode == LossSliceMode.ASSISTANT_ROLE_PLUS_TRUNCATED_TARGET_SLICE:
                # len_target_slice = result.slice_data.target_output.stop - result.slice_data.target_output.start
                # result.slice_data.loss = slice(result.slice_data.assistant_role.start, (result.slice_data.assistant_role.start + len_target_slice))
                
            # if self.attack_params.loss_slice_mode == LossSliceMode.INDEX_SHIFTED_TARGET_SLICE:
                # result.slice_data.loss = slice(
                    # prompt_find_self_target_c2t + self.attack_params.loss_slice_index_shift,
                    # (prompt_combined_c2t + 1) + self.attack_params.loss_slice_index_shift
                # )

            # if self.attack_params.loss_slice_mode == LossSliceMode.SAME_AS_TARGET_SLICE :
                # result.slice_data.loss = slice(
                    # prompt_find_self_target_c2t,
                    # prompt_combined_c2t + 1
                # )
            
            # if result.slice_data.loss is None:
                # raise PromptGenerationException("Did not find a valid loss slice mode")

            # self.validate_slice_data('get_prompt (non-Python) - loss', result.slice_data, toks, decoded_toks)

        # #print(f"[get_prompt] Debug: conversation_template (after modifications) = '{conversation_template}'")
        # #final_decoded_toks = get_decoded_tokens(self.tokenizer, toks)
        # #print(f"[get_prompt] Debug: toks (after parsing) = '{toks}', final_decoded_toks = '{final_decoded_toks}'")
        
        # result.full_prompt_token_ids = toks
        # result.input_token_ids = toks[:result.slice_data.target_output.stop]

        # self.print_slice_info("get_prompt", result.slice_data, toks)

        # #conversation_template.messages = []

        # return result

    def get_prompt(self, adversarial_content = None, force_python_tokenizer = False):#, update_self_values = True):

        result = PromptAndInputIDCollection()
        
        # set up temporary values based on permanent values
        working_adversarial_content = self.adversarial_content.copy()
        conversation_template = self.conv_template.copy()

        separator = ' '
        if not self.attack_params.base_prompt:
            separator = ''
            
        if conversation_template is None:
            print(f"[get_prompt] Error: got a null conversation template when trying to call self.conv_template.copy(). This should never happen. self.conv_template was '{self.conv_template}'")
            sys.exit(1)
        
        if adversarial_content is not None:
            working_adversarial_content = adversarial_content.copy()

        # Get the list of dumpster fire tokens once to make the many references to it run faster
        #hardcoded_trash_fire_tokens = TrashFireTokenCollection.get_meticulously_curated_trash_fire_token_collection(self.tokenizer, conversation_template)

        conversation_template.append_message(conversation_template.roles[0], f"{self.attack_params.base_prompt}{separator}{working_adversarial_content.as_string}")
        conversation_template.append_message(conversation_template.roles[1], f"{self.attack_params.target_output}")
        result.prompt = conversation_template.get_prompt()

        encoded_conversation_template_prompt = self.tokenizer(result.prompt)
        current_token_ids = encoded_conversation_template_prompt.input_ids
        current_decoded_tokens = get_decoded_tokens(self.tokenizer, current_token_ids)
        #original_toks = copy.deepcopy(current_token_ids)
        #original_decoded_tokens = get_decoded_tokens(self.tokenizer, original_toks)
       
        python_tokenizer = False
        # Using the non-Python tokenizer is totally broken right now, because it assumes that e.g. the first occurence of a role name is the correct location to use, even for chat templates with messages
        #python_tokenizer = True
        if conversation_template.name == 'oasst_pythia':
            python_tokenizer = True
        # I give up - the Llama-3 tokenizer is just too dangerous to try calling as if it were a fast tokenizer
        if conversation_template.name == 'llama-3':
            python_tokenizer = True
        if force_python_tokenizer:
            python_tokenizer = True
        #if "pythia" in conversation_template.name:
        #    python_tokenizer = True
        # This (formerly undocumented) check is a way to determine if the model is using Python-based tokenizers. It works because Python-based tokenizers (at least in the current version of Transformers) don't support the char_to_token operation), and it's used to avoid calling char_to_token for the rest of the get_prompt method in that case.
        if not python_tokenizer:
            try:
                #test_value = encoded_conversation_template_prompt.char_to_token(len(result.prompt)-1)
                # Unlike EVERY OTHER Transformers-compatible LLM, Llama-3 doesn't throw an exception when the Python tokenizer gets a call to char_to_token - it returns None instead! But only for offsets that are in the middle of the prompt! Otherwise it pretends it works!
                test_value = encoded_conversation_template_prompt.char_to_token(int((float(len(result.prompt)) - 1.0) / 2.0))                
                #print(f"[get_prompt] Debug: test_value = {test_value}")
                if test_value is None:
                    python_tokenizer = True
            except:
                python_tokenizer = True

        if python_tokenizer:
            # TKTK: consider rewriting this to not use fschat at all.
            # Using apply_chat_template where available and including custom templates for models that don't include it might be easier.
            #print(f"[get_prompt] Info: using Python tokenizer.")
            # reset the conversation template (to get rid of the messages that have been added earlier in this function), but preserve any messages that are part of the template
            conversation_template = self.conv_template.copy()
            
            # Figure out where the user role *starts* first
            # ...by comparing the prompt with no additional messages to one that has a new empty user message added
            prompt_with_no_additional_messages = conversation_template.get_prompt()
            prompt_with_no_additional_messages_token_ids = self.tokenizer(prompt_with_no_additional_messages).input_ids            
            #conversation_template.append_message(conversation_template.roles[0], None)
            conversation_template.append_message(conversation_template.roles[0], f"{self.attack_params.base_prompt}{separator}{working_adversarial_content.as_string}")
            current_prompt = conversation_template.get_prompt()
            #print(f"[get_prompt] Debug: current_prompt = '{current_prompt}'")
            user_role_prompt_token_ids = self.tokenizer(current_prompt).input_ids
            user_role_start_index = find_index_of_first_nonmatching_element(prompt_with_no_additional_messages_token_ids, user_role_prompt_token_ids)
            if user_role_start_index is None:
                print(f"[get_prompt] Error: did not find a non-matching element when comparing the strings '{prompt_with_no_additional_messages}' and '{current_prompt}', which were tokenized to the following IDs:\n{prompt_with_no_additional_messages_token_ids}\n{user_role_prompt_token_ids}")

            # Detour into figuring out the base prompt / goal, adversarial content / control, and target output slices
            # TKTK: update the remainder of this function to handle different placement of the adversarial content
            #conversation_template.update_last_message(f"{self.attack_params.base_prompt}{separator}{working_adversarial_content.as_string}")
            conversation_template.append_message(conversation_template.roles[1], self.attack_params.target_output)
            current_prompt = conversation_template.get_prompt()
            #print(f"[get_prompt] Debug: current_prompt = '{current_prompt}'")
            current_token_ids = self.tokenizer(current_prompt).input_ids
            current_decoded_tokens = get_decoded_tokens(self.tokenizer, current_token_ids)

            result.slice_data.goal = find_last_index_of_token(self.tokenizer, 
                self.trash_fire_tokens, 
                self.attack_params.base_prompt, 
                current_token_ids, 
                current_decoded_tokens, 
                conversation_template = conversation_template, 
                strip_leading_and_trailing_tokens = False, 
                remove_empty_leading_and_trailing_tokens = False, 
                remove_leading_and_trailing_trash_fire_tokens = False)
            
            # Can't do this optimization because the context of what appears before and after can change the tokens
            # goal_slice_start = find_last_occurrence_of_array_in_array(self.base_prompt_token_ids, current_token_ids)
            # goal_slice_end = goal_slice_start + len(self.base_prompt_token_ids)
            # result.slice_data.goal = slice(goal_slice_start, goal_slice_end)

            self.validate_slice_data('get_prompt (Python) - goal', result.slice_data, current_token_ids, current_decoded_tokens)

            if working_adversarial_content.as_string == "":
                result.slice_data.control = slice(result.slice_data.goal.stop, result.slice_data.goal.stop)
            else:
                result.slice_data.control = find_last_index_of_token(self.tokenizer, 
                    self.trash_fire_tokens, 
                    working_adversarial_content.as_string, 
                    current_token_ids, 
                    current_decoded_tokens, 
                    conversation_template = conversation_template, 
                    strip_leading_and_trailing_tokens = False, 
                    remove_empty_leading_and_trailing_tokens = False, 
                    remove_leading_and_trailing_trash_fire_tokens = False)
                # control_slice_start = find_last_occurrence_of_array_in_array(working_adversarial_content.token_ids, current_token_ids)
                # control_slice_end = control_slice_start + len(working_adversarial_content.token_ids)
                # result.slice_data.control = slice(control_slice_start, control_slice_end)

            self.validate_slice_data('get_prompt (Python) - control', result.slice_data, current_token_ids, current_decoded_tokens)

            result.slice_data.target_output = find_last_index_of_token(self.tokenizer, 
                self.trash_fire_tokens, 
                self.attack_params.target_output, 
                current_token_ids, 
                current_decoded_tokens, 
                conversation_template = conversation_template, 
                strip_leading_and_trailing_tokens = False, 
                remove_empty_leading_and_trailing_tokens = False, 
                remove_leading_and_trailing_trash_fire_tokens = False)
            
            # Can't do this optimization because the context of what appears before and after can change the tokens
            # target_slice_start = find_last_occurrence_of_array_in_array(self.target_output_token_ids, current_token_ids)
            # target_slice_end = goal_slice_start + len(self.target_output_token_ids)
            # result.slice_data.target_output = slice(target_slice_start, target_slice_end)
            
            self.validate_slice_data('get_prompt (Python) - target_output', result.slice_data, current_token_ids, current_decoded_tokens)

            # use the known locations of the base prompt, adversarial content, and target output slices to determine the remaining role indices
            # this section should already handle alternative adversarial content placement more or less correctly, except if it's placed in the middle of the base prompt (or interleaved)
            user_role_stop_index = result.slice_data.goal.start
            if result.slice_data.control.start < user_role_stop_index:
                user_role_stop_index = result.slice_data.control.start
            
            result.slice_data.user_role = slice(user_role_start_index, user_role_stop_index)
            self.validate_slice_data('get_prompt (Python) - user_role', result.slice_data, current_token_ids, current_decoded_tokens)
            
            assistant_role_start_index = result.slice_data.control.stop
            if result.slice_data.goal.stop > assistant_role_start_index:
                assistant_role_start_index = result.slice_data.goal.stop
            
            assistant_role_stop_index = result.slice_data.target_output.start
            
            result.slice_data.assistant_role = slice(assistant_role_start_index, assistant_role_stop_index)

            self.validate_slice_data('get_prompt (Python) - assistant_role', result.slice_data, current_token_ids, current_decoded_tokens)

            #print(f"[get_prompt] Debug: getting loss slice using mode {self.attack_params.loss_slice_mode}")
            if self.attack_params.loss_slice_mode == LossSliceMode.ASSISTANT_ROLE_PLUS_FULL_TARGET_SLICE:                
                result.slice_data.loss = slice(result.slice_data.assistant_role.start, min(last_non_garbage_token, len(current_token_ids)))

            if self.attack_params.loss_slice_mode == LossSliceMode.ASSISTANT_ROLE_PLUS_TRUNCATED_TARGET_SLICE:
                len_target_slice = result.slice_data.target_output.stop - result.slice_data.target_output.start
                result.slice_data.loss = slice(result.slice_data.assistant_role.start, (result.slice_data.assistant_role.start + len_target_slice))
                
            if self.attack_params.loss_slice_mode == LossSliceMode.INDEX_SHIFTED_TARGET_SLICE:
                #result.slice_data.loss = slice(first_non_garbage_token - 1, min(last_non_garbage_token, len(current_token_ids)) - 1)
                result.slice_data.loss = slice(int(result.slice_data.target_output.start) + self.attack_params.loss_slice_index_shift, min(int(result.slice_data.target_output.stop) + self.attack_params.loss_slice_index_shift, len(current_token_ids)))

            if self.attack_params.loss_slice_mode == LossSliceMode.SAME_AS_TARGET_SLICE:
                result.slice_data.loss = slice(first_non_garbage_token, min(last_non_garbage_token, len(current_token_ids)))
            
            if result.slice_data.loss is None:
                raise PromptGenerationException("Did not find a valid loss slice mode")

            self.validate_slice_data('get_prompt (Python) - loss', result.slice_data, current_token_ids, current_decoded_tokens)
            
        else:
            #print(f"[get_prompt] Info: using fast tokenizer")
            
            # result.slice_data.user_role = slice(
                # encoded_conversation_template_prompt.char_to_token(result.prompt.find(conversation_template.roles[0])),
                # encoded_conversation_template_prompt.char_to_token(result.prompt.find(conversation_template.roles[0]) + len(conversation_template.roles[0]) + 1)
            # )
            last_token_index = result.prompt.rindex(conversation_template.roles[0])
            #print(f"[get_prompt] Debug: last_token_index = {last_token_index}")
            
            if last_token_index is None:
                print(f"[get_prompt] Warning: couldn't find conversation role 0 '{conversation_template.roles[0]}' in prompt '{result.prompt}'")
            else:
                if conversation_template.name in get_llama2_and_3_fschat_template_names():
                    try:
                        last_token_index_2 = result.prompt.rindex(conversation_template.roles[0], 0, last_token_index)
                        #print(f"[get_prompt] Debug: last_token_index_2 = {last_token_index_2}")
                        if last_token_index_2 is not None:
                            last_token_index = last_token_index_2
                    except Exception as e:
                        dummy = 1
                        # last_token_index is already correct
                        #print(f"[get_prompt] Debug: exception while getting second token index for user role using fast tokenizer: {e}")
            
            #print(f"[get_prompt] Debug: result.prompt = {result.prompt}")
            #print(f"[get_prompt] Debug: last_token_index = {last_token_index}")
            #print(f"[get_prompt] Debug: encoded_conversation_template_prompt = {encoded_conversation_template_prompt}")
            #decoded_encoded_conversation_template_prompt = get_decoded_tokens(self.tokenizer, encoded_conversation_template_prompt.input_ids)
            #print(f"[get_prompt] Debug: decoded_encoded_conversation_template_prompt = {decoded_encoded_conversation_template_prompt}")
            
            user_role_start_index = encoded_conversation_template_prompt.char_to_token(last_token_index)
            #print(f"[get_prompt] Debug: user_role_start_index = {user_role_start_index}")
            user_role_end_index = encoded_conversation_template_prompt.char_to_token(last_token_index + len(conversation_template.roles[0]) + 1)
            #print(f"[get_prompt] Debug: user_role_start_index = {user_role_start_index}")
            #print(f"[get_prompt] Debug: user_role_end_index = {user_role_end_index}")
            
            result.slice_data.user_role = slice(
                user_role_start_index,
                user_role_end_index
            )
            
            self.validate_slice_data('get_prompt (non-Python) - user_role', result.slice_data, current_token_ids, current_decoded_tokens)

            result.slice_data.system = slice(0, result.slice_data.user_role.start)
            
            self.validate_slice_data('get_prompt (non-Python) - system', result.slice_data, current_token_ids, current_decoded_tokens)

            # TKTK: BEGIN: update the goal and control slice logic to handle different placement of the adversarial content
            result.slice_data.goal = slice(
                encoded_conversation_template_prompt.char_to_token(result.prompt.find(self.attack_params.base_prompt)),
                encoded_conversation_template_prompt.char_to_token(result.prompt.find(self.attack_params.base_prompt) + len(self.attack_params.base_prompt))
            )
            self.validate_slice_data('get_prompt (non-Python) - goal', result.slice_data, current_token_ids, current_decoded_tokens)
            
            #If the adversarial content is an empty string, make the slice an empty slice right after the goal slice
            if working_adversarial_content.as_string == "":
                result.slice_data.control = slice(result.slice_data.goal.stop, result.slice_data.goal.stop)
            else:
                result.slice_data.control = slice(
                    encoded_conversation_template_prompt.char_to_token(result.prompt.find(working_adversarial_content.as_string)),
                    encoded_conversation_template_prompt.char_to_token(result.prompt.find(working_adversarial_content.as_string) + len(working_adversarial_content.as_string))
                )
            self.validate_slice_data('get_prompt (non-Python) - control', result.slice_data, current_token_ids, current_decoded_tokens)
            # TKTK: END: update the goal and control slice logic to handle different placement of the adversarial content
            
            #print(f"[get_prompt] Debug: finding conversation_template.roles[1] = '{conversation_template.roles[1]}' with length {len(conversation_template.roles[1])}.")
            # result.slice_data.assistant_role = slice(
                # encoded_conversation_template_prompt.char_to_token(result.prompt.find(conversation_template.roles[1])),
                # encoded_conversation_template_prompt.char_to_token(result.prompt.find(conversation_template.roles[1]) + len(conversation_template.roles[1]) + 1)
            # )
            
            last_token_index = result.prompt.rindex(conversation_template.roles[1])
            # if conversation_template.name in get_llama2_and_3_fschat_template_names():
                # try:
                    # last_token_index_2 = result.prompt.rindex(conversation_template.roles[1], 0, last_token_index)
                    # last_token_index = last_token_index_2
                # except Exception as e:
                    # dummy = 1
                    # # last_token_index is already correct
                    # print(f"[get_prompt] Debug: exception while getting second token index for assistant role using fast tokenizer: {e}")
            
            result.slice_data.assistant_role = slice(
                encoded_conversation_template_prompt.char_to_token(last_token_index),
                encoded_conversation_template_prompt.char_to_token(last_token_index + len(conversation_template.roles[1]) + 1)
            )
            
            self.validate_slice_data('get_prompt (non-Python) - assistant_role', result.slice_data, current_token_ids, current_decoded_tokens)

            #self.print_slice_info("get_prompt", result.slice_data, current_token_ids)
            #print(f"[get_prompt] Debug: result.prompt = '{result.prompt}', self.attack_params.target_output = '{self.attack_params.target_output}'")
            prompt_find_self_target = result.prompt.find(self.attack_params.target_output)
            #print(f"[get_prompt] Debug: prompt_find_self_target = '{prompt_find_self_target}'")
            prompt_find_self_target_c2t = encoded_conversation_template_prompt.char_to_token(prompt_find_self_target)
            if isinstance(prompt_find_self_target_c2t, type(None)):
                print(f"[get_prompt] Warning: got None for encoded_conversation_template_prompt.char_to_token(prompt_find_self_target). prompt_find_self_target = '{prompt_find_self_target}' using '{self.attack_params.target_output}' in '{result.prompt}'. Using value {result.slice_data.assistant.stop} instead of None. This may indicate an error with the parsing logic.")
                prompt_find_self_target_c2t = result.slice_data.assistant.stop
            prompt_combined_c2t = None
            add_length = len(self.attack_params.target_output) + 1
            while isinstance(prompt_combined_c2t, type(None)):
                prompt_combined_c2t = encoded_conversation_template_prompt.char_to_token(prompt_find_self_target + (add_length))
                add_length -= 1
                if add_length < 0:
                    prompt_combined_c2t = prompt_find_self_target_c2t
                    break
            # Subtract one more than the first valid value so that the length of the slice is correct
            #print(f"[get_prompt] Debug: prompt_find_self_target_c2t = '{prompt_find_self_target_c2t}', prompt_combined_c2t = '{prompt_combined_c2t}'")
            result.slice_data.target_output = slice(
                prompt_find_self_target_c2t,
                prompt_combined_c2t + 1
            )
            self.validate_slice_data('get_prompt (non-Python) - target_output', result.slice_data, current_token_ids, current_decoded_tokens)
            
            #print(f"[get_prompt] Debug: getting loss slice using mode {self.attack_params.loss_slice_mode}")
            if self.attack_params.loss_slice_mode == LossSliceMode.ASSISTANT_ROLE_PLUS_FULL_TARGET_SLICE:
                result.slice_data.loss = slice(result.slice_data.assistant_role.start, min(last_non_garbage_token, len(current_token_ids)))

            if self.attack_params.loss_slice_mode == LossSliceMode.ASSISTANT_ROLE_PLUS_TRUNCATED_TARGET_SLICE:
                len_target_slice = result.slice_data.target_output.stop - result.slice_data.target_output.start
                result.slice_data.loss = slice(result.slice_data.assistant_role.start, (result.slice_data.assistant_role.start + len_target_slice))
                
            if self.attack_params.loss_slice_mode == LossSliceMode.INDEX_SHIFTED_TARGET_SLICE:
                result.slice_data.loss = slice(
                    prompt_find_self_target_c2t + self.attack_params.loss_slice_index_shift,
                    (prompt_combined_c2t + 1) + self.attack_params.loss_slice_index_shift
                )

            if self.attack_params.loss_slice_mode == LossSliceMode.SAME_AS_TARGET_SLICE :
                result.slice_data.loss = slice(
                    prompt_find_self_target_c2t,
                    prompt_combined_c2t + 1
                )
            
            if result.slice_data.loss is None:
                raise PromptGenerationException("Did not find a valid loss slice mode")

            self.validate_slice_data('get_prompt (non-Python) - loss', result.slice_data, current_token_ids, current_decoded_tokens)

        #print(f"[get_prompt] Debug: conversation_template (after modifications) = '{conversation_template}'")
        #final_decoded_toks = get_decoded_tokens(self.tokenizer, current_token_ids)
        #print(f"[get_prompt] Debug: current_token_ids (after parsing) = '{current_token_ids}', final_decoded_toks = '{final_decoded_toks}'")
        
        result.full_prompt_token_ids = current_token_ids
        result.input_token_ids = current_token_ids[:result.slice_data.target_output.stop]

        self.print_slice_info("get_prompt", result.slice_data, current_token_ids)

        #conversation_template.messages = []

        return result
