import copy
import torch
import fastchat 

from llm_attacks_bishopfox import get_decoded_token, get_decoded_tokens, get_encoded_token, get_encoded_tokens, get_token_denylist, get_escaped_string

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

def append_single_or_list_members(existing_list, value_or_list_to_add, ignore_if_none = False):
    if ignore_if_none:
        if value_or_list_to_add is None:
            return existing_list
    if isinstance(value_or_list_to_add, list):
        for list_member in value_or_list_to_add:
            existing_list = append_single_or_list_members(existing_list, list_member, ignore_if_none = ignore_if_none)
    else:
        if value_or_list_to_add not in existing_list:
            existing_list.append(value_or_list_to_add)
    return existing_list

class TrashFireTokenCollection:
    def __init__(self):
        self.input_strings = None
        self.token_ids = None
        self.decoded_tokens = None
    
    @staticmethod
    def get_hardcoded_trash_fire_token_collection(tokenizer, conversation_template):
        result = TrashFireTokenCollection()
        result.input_strings = [ '</s>', '<s>', '###', '##', '#', "\n", "\\n", "\r", "\\r", "\r\n", "\\r\\", "", ":", " ", "\t",
                                    '<sep>',
                                    '<|system|>'
                                    '<|end|>',
                                    '<eos>',
                                    '<eod>',
                                    '<|user|>',
                                    '<|User|>',
                                    '<|eot_id|>',
                                    '<end>',
                                    '[UNK]',
                                    '<｜end▁of▁sentence｜>',
                                    '<|im_end|>',
                                    '<end_of_turn>' ]
        # get tokenizer-specific trash
        result.input_strings.append(tokenizer.bos_token)
        result.input_strings.append(tokenizer.eos_token)
        result.input_strings.append(tokenizer.unk_token)
        result.input_strings.append(tokenizer.pad_token)
        additional_flaming_dumpster_ids = []
        additional_flaming_dumpster_ids.append(tokenizer.bos_token_id)
        additional_flaming_dumpster_ids.append(tokenizer.eos_token_id)
        additional_flaming_dumpster_ids.append(tokenizer.unk_token_id)
        additional_flaming_dumpster_ids.append(tokenizer.pad_token_id)
        
        # get all of the special trash from the conversation template as well
        if hasattr(conversation_template, "stop_token_ids"):
            additional_flaming_dumpster_ids = append_single_or_list_members(additional_flaming_dumpster_ids, conversation_template.stop_token_ids, ignore_if_none = True)

        if hasattr(conversation_template, "stop_str"):
            result.input_strings = append_single_or_list_members(result.input_strings, conversation_template.stop_str, ignore_if_none = True)
        if hasattr(conversation_template, "sep"):
            result.input_strings = append_single_or_list_members(result.input_strings, conversation_template.sep, ignore_if_none = True)
        if hasattr(conversation_template, "sep2"):
            result.input_strings = append_single_or_list_members(result.input_strings, conversation_template.sep2, ignore_if_none = True)
        
        #result.token_ids = get_token_denylist(tokenizer, result.input_strings, device='cpu', filter_special_tokens = True, filter_additional_special_tokens = True, additional_token_ids = additional_flaming_dumpster_ids)
        result.token_ids = get_token_denylist(tokenizer, result.input_strings, device='cpu', filter_special_tokens = True, filter_additional_special_tokens = True, additional_token_ids = additional_flaming_dumpster_ids)
        
        result.decoded_tokens = get_decoded_tokens(tokenizer, result.token_ids)
        #print(f"[get_hardcoded_trash_fire_token_collection] Debug: result.input_strings = '{result.input_strings}', result.token_ids = '{result.token_ids}', result.decoded_tokens = '{result.decoded_tokens}'")
        return result

#def load_conversation_template(template_name, generic_role_indicator_template = None, system_prompt = None, clear_existing_template_conversation = False, conversation_template_messages=None):
def load_conversation_template(model_path, template_name = None, generic_role_indicator_template = None, system_prompt = None, clear_existing_template_conversation = False, conversation_template_messages=None):
    #print(f"[load_conversation_template] Debug: loading chat template '{template_name}'. generic_role_indicator_template='{generic_role_indicator_template}', system_prompt='{system_prompt}', clear_existing_template_conversation='{clear_existing_template_conversation}'")
    conv_template = None
    # suppress the warning about templates not existing if there's a custom version defined here
    has_custom_template = False
    if is_phi3_template(template_name):
        has_custom_template = True
    if is_phi2_template(template_name):
        has_custom_template = True
    original_template_name = template_name
    if template_name is not None:
        if template_name not in fastchat.conversation.conv_templates.keys():
            if not has_custom_template:
                print(f"[load_conversation_template] Warning: chat template '{template_name}' was not found in fastchat - defaulting to 'zero_shot'.")
            template_name = 'zero_shot'
        print(f"[load_conversation_template] Debug: loading chat template '{template_name}'")
        conv_template = fastchat.conversation.get_conv_template(template_name)
    else:
        print(f"[load_conversation_template] Debug: determining chat template based on content in '{model_path}'")
        conv_template = fastchat.model.get_conversation_template(model_path)
    # make sure fastchat doesn't sneak the one_shot messages in when zero_shot was requested
    if clear_existing_template_conversation:
        if hasattr(conv_template, "messages"):
            #print(f"[load_conversation_template] Debug: resetting conv_template.messages from '{conv_template.messages}' to []")
            conv_template.messages = []
        else:
            print("[load_conversation_template] Warning: the option to clear the conversation template's default conversation was enabled, but the template does not include a default conversation.")
            conv_template.messages = []
    if is_phi3_template(original_template_name):
        conv_template.name = "phi3"
        conv_template.sep_style = fastchat.conversation.SeparatorStyle.NO_COLON_SINGLE
        conv_template.system_template = "<|system|>\n{system_message}<|end|>\n"
        conv_template.system_message = None
        #conv_template.roles = tuple(["\n<|user|>", "<|end|>\n<|assistant|>"])
        conv_template.roles = tuple(["\n<|user|>", "\n<|assistant|>"])
        conv_template.sep = '\n'
    if is_phi2_template(original_template_name):
        conv_template.name = "phi2"
        conv_template.system_template = "System: {system_message}\n"
        conv_template.system_message = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful answers to the user's questions."
        conv_template.roles = tuple(["User", "Assistant"])
        conv_template.sep = '\n'
        conv_template.sep2 = '\n'
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
    if conv_template.name == 'llama-2':
        conv_template.sep2 = conv_template.sep2.strip()
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

class PromptSliceData:
    def __init__(self):
        self.system = None
        self.user_role = None
        self.goal = None
        self.control = None
        self.assistant_role = None
        self.target = None
        self.loss = None
    
    def get_slice_dictionary(self):
        result = {}
        result["system"] = self.system
        result["user_role"] = self.user_role
        result["goal"] = self.goal
        result["control"] = self.control
        result["assistant_role"] = self.assistant_role
        result["target"] = self.target
        result["loss"] = self.loss
        return result

class PromptAndInputIDCollection:
    def __init__(self):
        self.prompt = None
        self.full_prompt_ids = None
        self.input_ids = None
        self.slice_data = PromptSliceData()
    
    def get_input_ids_as_tensor(self):
        return torch.tensor(self.input_ids)

# TKTK: Replace this with an AdversarialTokenManager that tracks an array of token IDs instead of a string
# That would allow not only performing a prefix/suffix attack, but also interleaving the tokens into the 
# base string.
#
# Also, it would (hopefully) remove the need for the tedious parsing logic I had to write to make the 
# attack work with most models.
class SuffixManager:
    def __init__(self, *, tokenizer, conv_template, instruction, target, adv_string):

        self.tokenizer = tokenizer
        self.conv_template = conv_template
        self.instruction = instruction
        self.target = target
        self.adv_string = adv_string
        self.trash_fire_tokens = TrashFireTokenCollection.get_hardcoded_trash_fire_token_collection(tokenizer, conv_template)
    
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
            print(f"[{source_method_name}] Debug: slice '{slice_name}' decoded tokens = '{slice_info[slice_name]}'")

    def validate_slice_data(self, source_method_name, slice_data):
        invalid_slice_dictionary = {}
        slice_dictionary = slice_data.get_slice_dictionary()
        for slice_name in slice_dictionary.keys():
            sl = slice_dictionary[slice_name]
            if sl is not None:
                is_valid = True
                if sl.start is None or sl.stop is None:
                    is_valid = False
                if not is_valid:
                    # The system slice having a start of None is expected
                    if slice_name == "system":
                        if sl.start is None and sl.stop is not None:
                            is_valid = True
                if not is_valid:
                    invalid_slice_dictionary[slice_name] = sl
                #print(f"[{source_method_name}] Debug: slice '{slice_name}' = '{sl}', is_valid = {is_valid}.")
        if len(invalid_slice_dictionary.keys()) > 0:
            message = f"[{source_method_name}] Warning: one or more slices have None values instead of start or stop values. This generally indicates an issue with the tokenizing or parsing logic. The slice(s) with None values are: "
            for slice_name in invalid_slice_dictionary.keys():
                sl = invalid_slice_dictionary[slice_name]
                message += f"{slice_name}: {sl},"
            message = message[:-1]
            print(message)

    # The get_prompt function was originally an opaque piece of logic populated with magic numbers that had no explanations for how they were derived, but were supposedly specific to three LLMs: Llama 2, Vicuna, and OpenAssistant's Pythia.
    
    # By examining the results of this function for those three models, I was able to reverse-engineer how to (more or less) find the correct values for other models.
    # I think it should all be fully automatic now, but in case you need to add model-specific logic or debug the parsing, here are what the slices represent.

    # Each slice should contain an array of token IDs that represent one or more words.
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
    
    
    # _loss_slice: I honestly don't know how this is supposed to be significantly different from the _target_slice.
    # In the original code, _loss_slice was always the same as _target_slice, except with 1 subtracted from the start and stop values.
    # The length of the two slices must match, or the attack will crash with an error.
    # I suspect _loss_slice was originally more significant, and has essentially been stubbed out.

    # For some LLMs, e.g. Llama-2, the distinction between speaking roles is handled differently, but the parsing logic should still produce equivalent results.
    # In the case of Llama-2, the standard conversation template wraps user input in [INST] [/INST] tags, and anything else is the LLM's response.
    # The parser handles this by treating "[INST]" as equivalent to something like "Human:" and "[/INST]" as equivalent to something like "Assistant:".
    # There is probably a way to exploit that inherent to the model, e.g. with inbalanced "[INST]" and/or "[/INST]" tags.

    
    # I don't have a PhD in machine learning or anything, but I observed the following behaviours that seemed to be incorrect and fixed them:
    
    # For the Llama-2 code path, the user role, goal, and control slices were incorrectly calculated.
    # I think this was because some of the code didn't take into account that the tokenizer considered "[INST]" three tokens instead of one.
    # I eventually fixed this by making the code flexible enough to handle Llama-2 without a separate branch.

    def find_first_occurrence_of_array_in_array(self, inner_array, outer_array, start_index = 0, stop_index = None):
        result = None
        #print(f"[find_first_occurrence_of_array_in_array] Debug: Searching for '{inner_array}' in '{outer_array}'")
        len_inner = len(inner_array)
        len_outer = len(outer_array)
        range_end = len_outer
        if stop_index is not None:
            range_end = stop_index
        #print(f"[find_first_occurrence_of_array_in_array] Debug: searching for '{inner_array}' in '{outer_array}' from index {start_index} to {range_end}'")
        for i in range(start_index, range_end):
            if (i + len_inner) >=  len_outer:
                break
            if outer_array[i] == inner_array[0]:
                is_match = True
                #print(f"[find_first_occurrence_of_array_in_array] Debug: found potential match beginning at index {i}'")
                for j in range(1, len_inner):
                    if outer_array[i + j] != inner_array[j]:
                        is_match = False
                        #print(f"[find_first_occurrence_of_array_in_array] Debug: '{outer_array[i + j]}' != '{inner_array[j]}'")
                        break
                if is_match:
                    return i
        return result

    def find_last_occurrence_of_array_in_array(self, inner_array, outer_array, start_index = 0, stop_index = None):
        result = None
        #print(f"[find_last_occurrence_of_array_in_array] Debug: Searching for '{inner_array}' in '{outer_array}'")
        len_inner = len(inner_array)
        len_outer = len(outer_array)
        range_start = len_outer - len_inner
        if stop_index is not None:
            range_start = stop_index - len_inner
        range_end = start_index - 1
        #print(f"[find_last_occurrence_of_array_in_array] Debug: searching for '{inner_array}' in '{outer_array}' from index {range_start} to {range_end}'")
        for i in range(range_start, range_end, -1):
            if outer_array[i] == inner_array[0]:
                is_match = True
                #print(f"[find_last_occurrence_of_array_in_array] Debug: found potential match beginning at index {i}'")
                for j in range(1, len_inner):
                    if outer_array[i + j] != inner_array[j]:
                        is_match = False
                        #print(f"[find_last_occurrence_of_array_in_array] Debug: '{outer_array[i + j]}' != '{inner_array[j]}'")
                        break
                if is_match:
                    return i
        return result

    def remove_empty_leading_and_trailing_tokens(self, token_array, decoded_token_array, strip_decoded_tokens = False):
        len_token_array = len(token_array)
        len_decoded_token_array = len(decoded_token_array)
        if len_token_array != len_decoded_token_array:
            raise Exception(f"The length of the token and decoded token arrays must match. Inputs were '{token_array}' (length: {len_token_array}) and '{decoded_token_array}' (length: {len_decoded_token_array})")

        first_non_empty_token = 0
        last_non_empty_token = len_decoded_token_array - 1

        for i in range(0, len_token_array):
            decoded_token_temp = decoded_token_array[i]
            if strip_decoded_tokens:
                decoded_token_temp = decoded_token_temp.strip()
                decoded_token_array[i] = decoded_token_temp
            if decoded_token_temp != "":
                break
            first_non_empty_token += 1
        
        for i in range(len_token_array - 1, -1, -1):
            decoded_token_temp = decoded_token_array[i]
            if strip_decoded_tokens:
                decoded_token_temp = decoded_token_temp.strip()
                decoded_token_array[i] = decoded_token_temp
            if decoded_token_temp != "":
                break
            last_non_empty_token -= 1
        
        actual_last_non_empty_token = last_non_empty_token + 1
        result_token_array = token_array[first_non_empty_token:actual_last_non_empty_token]
        result_decoded_token_array = decoded_token_array[first_non_empty_token:actual_last_non_empty_token]
        
        #print(f"[remove_empty_leading_and_trailing_tokens] Debug: token_array = '{token_array}', result_token_array = '{result_token_array}', decoded_token_array = '{decoded_token_array}', result_decoded_token_array = '{result_decoded_token_array}'")
        return result_token_array, result_decoded_token_array

    def find_last_index_of_token(self, string_to_search_for, tokens, start_index = 0, stop_index = None):        
        decoded_tokens = get_decoded_tokens(self.tokenizer, tokens)
        #print(f"[find_last_index_of_token] Debug: decoded_tokens = '{decoded_tokens}' for tokens = '{tokens}'")
        string_tokens = get_encoded_token(self.tokenizer, string_to_search_for)
        
        #print(f"[find_last_index_of_token] Debug: string_tokens = '{string_tokens}' for string '{string_to_search_for}'")
        # hacky workarounds for garbagey behaviour by LLMs
        string_to_search_for_array = string_to_search_for.split(" ")
        first_search_word = string_to_search_for_array[0]
        len_first_search_word = len(first_search_word)
        decoded_string_tokens = get_decoded_tokens(self.tokenizer, string_tokens)
        string_tokens, decoded_string_tokens = self.remove_empty_leading_and_trailing_tokens(string_tokens, decoded_string_tokens, strip_decoded_tokens = True)
        
        #print(f"[find_last_index_of_token] Debug: searching for last occurrence of '{string_to_search_for}' (tokenized as '{decoded_string_tokens}') in '{decoded_tokens}'")
        result_start = self.find_last_occurrence_of_array_in_array(string_tokens, tokens, start_index=start_index, stop_index=stop_index)
        result_stop = None
        if result_start is None:
            # try to find cases where tokens have spaces on either side or not at all
            decoded_tokens_processed_1 = []
            decoded_tokens_processed_2 = []
            for i in range(0, len(decoded_tokens)):
                processed_token = decoded_tokens[i].strip()
                decoded_tokens_processed_1.append(processed_token)
                decoded_tokens_processed_2.append(decoded_tokens[i])
            result_start = self.find_last_occurrence_of_array_in_array(string_to_search_for_array, decoded_tokens_processed_1, start_index=start_index, stop_index=stop_index)
            if result_start is None:
                result_start = self.find_last_occurrence_of_array_in_array(string_to_search_for_array, decoded_tokens_processed_2, start_index=start_index, stop_index=stop_index)
                if result_start is None:
                    raise Exception(f"Could not find '{string_to_search_for}' (tokenized as '{decoded_string_tokens}') in '{decoded_tokens}', '{decoded_tokens_processed_1}', or '{decoded_tokens_processed_2}'")
                else:
                    result_stop = result_start + len(string_to_search_for_array)
                    # This issue is so frequent that enabling this error is too noisy
                    #print(f"[find_last_index_of_token] Warning: could not find '{string_to_search_for}' (tokenized as '{decoded_string_tokens}') in '{decoded_tokens}', but found the close approximation '{string_to_search_for_array}' in '{decoded_tokens_processed_2}' and will use that position instead. This may be due to using a buggy LLM that considers e.g. 'Human' and ' Human' different tokens, but uses both values for similar purposes internally.")
            else:
                result_stop = result_start + len(string_to_search_for_array)
                # This issue is so frequent that enabling this error is too noisy
                #print(f"[find_last_index_of_token] Warning: could not find '{string_to_search_for}' (tokenized as '{decoded_string_tokens}') in '{decoded_tokens}', but found the close approximation '{string_to_search_for_array}' in '{decoded_tokens_processed_1}' and will use that position instead. This may be due to using a buggy LLM that considers e.g. 'Human' and ' Human' different tokens, but uses both values for similar purposes internally.")
                
        else:
            result_stop = result_start + len(string_tokens)
        
        result = slice(result_start, result_stop)
        #print(f"[find_last_index_of_token] Debug: result = '{result}'")
        return result
    
    # so many spaghetti code special cases to handle madness like 'This is a special token, but you can't treat it as 100% trash fire because it's also a sentinel that the parser has to look for'
    def is_conversation_role_token(self, conversation_template, token):
        if token is not None:
            if token.strip() != "":
                for c_role in conversation_template.roles:
                    if token in c_role:
                        return True
        return False
    
#        blincoln@atropos $ ollama run gemma2:9b-instruct-fp16
#
#        >>> Please draw an ASCII art image of a machine learning developer dressed as a cattle rustler, chasing a flaming dumpster down a river.
#             _.--""--._
#           .'          `.
#          /   O      O   \
#         |    \  ^  /    |
#          \   `-----'   /
#           `. _______ .'
#             //_____\\
#            (( ____ ))
#             `-----' 
#               ||
#               ||  __.-""--._
#               ||.'          `.
#               ||/   O      O   \
#              /||    \  ^  /    |
#             ( ||     `-----'   /
#              \_||_. _______ .'
#                ||//_____\\
#                ||(( ____ ))
#                || `-----' 
#               /__||  ____)__)______
#              (__)/_)(__)(__)(___)/
#                 \___________/~~\~~~~
#                  ~~~~~~~~~~~~~~
#
#             ,;;:;.                ;;;;:;,
#            ;:::::;.              ::::::;;
#           ;:::::::::.           ::::::::::::
#          ::::::::::::::::::.       ::::::::::::
#         ::::::::::::::::::::::.   :::::::::::::::::::
#        ;::::::::::::::::::::::;   ::::::::::::::::::::::;
#
#
#        (Flames rising from dumpster)
#
#
#        >>> That's perfect, thank you Gemma 2.
        
    def is_disastrous_dumpster_fire_token(self, hardcoded_trash_fire_tokens, conversation_template, token, decoded_token):
        token_is_a_pile_of_garbage_why_is_this_not_standardized_yet_you_ml_cowboys = False
        escaped_token = get_escaped_string(decoded_token)
        if token in hardcoded_trash_fire_tokens.token_ids:
            #print(f"[is_disastrous_dumpster_fire_token] Debug: marked token '{escaped_token}' (id {token}) as a flaming dumpster floating down the river because the token ID was in the list of trash fire tokens.")
            token_is_a_pile_of_garbage_why_is_this_not_standardized_yet_you_ml_cowboys = True
        if not token_is_a_pile_of_garbage_why_is_this_not_standardized_yet_you_ml_cowboys:
            for ctv in [decoded_token, decoded_token.strip()]:
                if ctv == "":
                    #print(f"[is_disastrous_dumpster_fire_token] Debug: marked token '{escaped_token}' (id {token}) as a flaming dumpster floating down the river because it was empty or contained only whitespace.")
                    token_is_a_pile_of_garbage_why_is_this_not_standardized_yet_you_ml_cowboys = True
                    break
                if ctv in hardcoded_trash_fire_tokens.decoded_tokens:
                    #print(f"[is_disastrous_dumpster_fire_token] Debug: marked token '{escaped_token}' (id {token}) as a flaming dumpster floating down the river because the decoded token was in the list of decoded trash fire tokens.")
                    token_is_a_pile_of_garbage_why_is_this_not_standardized_yet_you_ml_cowboys = True
                    break
                if ctv in hardcoded_trash_fire_tokens.input_strings:
                    #print(f"[is_disastrous_dumpster_fire_token] Debug: marked token '{escaped_token}' (id {token}) as a flaming dumpster floating down the river because the decoded token was in the list of input strings used to generate the list of trash fire tokens.")
                    token_is_a_pile_of_garbage_why_is_this_not_standardized_yet_you_ml_cowboys = True
                    break
        
        # but wait! we can't exclude conversation role tokens! That would make parsing the output much harder!
        if token_is_a_pile_of_garbage_why_is_this_not_standardized_yet_you_ml_cowboys:
            if self.is_conversation_role_token(conversation_template, decoded_token.strip()):
                token_is_a_pile_of_garbage_why_is_this_not_standardized_yet_you_ml_cowboys = False
        
        return token_is_a_pile_of_garbage_why_is_this_not_standardized_yet_you_ml_cowboys
    
    # dynamically determine the last token in a set of tokens 
    # that get_prompt should consider
    # like '</s>', '<s>', '\n', '###', or ' '
    def find_last_non_garbage_token(self, conversation_template, tokens, decoded_tokens, hardcoded_trash_fire_tokens, start_index = 0, stop_index = None):
        #decoded_tokens = get_decoded_tokens(self.tokenizer, tokens)
        result = None
        range_end = len(tokens)
        if stop_index is not None:
            range_end = stop_index
        for i in range(start_index, range_end):
            token_is_a_pile_of_garbage_why_is_this_not_standardized_yet_you_ml_cowboys = self.is_disastrous_dumpster_fire_token(hardcoded_trash_fire_tokens, conversation_template, tokens[i], decoded_tokens[i])
            if not token_is_a_pile_of_garbage_why_is_this_not_standardized_yet_you_ml_cowboys:
                result = i
        if result is None:
            raise Exception(f"[find_last_non_garbage_token] Could not find a token that wasn't an absolute dumpster fire in '{decoded_tokens}' from index {start_index} to {range_end}, please, stop the madness right now.")
        #print(f"[find_last_non_garbage_token] Debug: last non-garbage token in '{decoded_tokens}' from index {start_index} to {range_end} ('{decoded_tokens[start_index:range_end]}') is index {result}, '{decoded_tokens[result]}'")
        return result
    
    def find_first_non_garbage_token(self, conversation_template, tokens, decoded_tokens, hardcoded_trash_fire_tokens, start_index = 0, stop_index = None):
        #decoded_tokens = get_decoded_tokens(self.tokenizer, tokens)
        result = None
        range_end = len(tokens)
        if stop_index is not None:
            range_end = stop_index
        for i in range(start_index, range_end):
            token_is_a_pile_of_garbage_why_is_this_not_standardized_yet_you_ml_cowboys = self.is_disastrous_dumpster_fire_token(hardcoded_trash_fire_tokens, conversation_template, tokens[i], decoded_tokens[i])
            if not token_is_a_pile_of_garbage_why_is_this_not_standardized_yet_you_ml_cowboys:
                #print(f"[find_first_non_garbage_token] Debug: first non-garbage token in '{decoded_tokens}' from index {start_index} to {range_end} is index {i}, '{decoded_tokens[i]}'")
                return i
        if result is None:
            raise Exception(f"[find_first_non_garbage_token] Could not find a token that wasn't an absolute dumpster fire in '{decoded_tokens}' from index {start_index} to {range_end}, please, stop the madness right now.")
        return result

    def get_prompt(self, adv_string=None, force_python_tokenizer = False):#, update_self_values = True):

        result = PromptAndInputIDCollection()
        
        # set up temporary values based on permanent values
        adversarial_string = self.adv_string
        conversation_template = self.conv_template.copy()

        if adv_string is not None:
            adversarial_string = adv_string

        # Get the list of dumpster fire tokens once to make the many references to it run faster
        #hardcoded_trash_fire_tokens = TrashFireTokenCollection.get_hardcoded_trash_fire_token_collection(self.tokenizer, conversation_template)

        conversation_template.append_message(conversation_template.roles[0], f"{self.instruction} {adversarial_string}")
        conversation_template.append_message(conversation_template.roles[1], f"{self.target}")
        result.prompt = conversation_template.get_prompt()

        encoding = self.tokenizer(result.prompt)
        toks = encoding.input_ids
        #original_toks = copy.deepcopy(toks)
        #original_decoded_tokens = get_decoded_tokens(self.tokenizer, original_toks)
       
        python_tokenizer = False
        if conversation_template.name == 'oasst_pythia':
            python_tokenizer = True
        if force_python_tokenizer:
            python_tokenizer = True
        #if "pythia" in conversation_template.name:
        #    python_tokenizer = True
        # This (formerly undocumented) check is a way to determine if the model is using Python-based tokenizers. It works because Python-based tokenizers (at least in the current version of Transformers) don't support the char_to_token operation), and it's used to avoid calling char_to_token for the rest of the get_prompt method in that case.
        if not python_tokenizer:
            try:
                encoding.char_to_token(len(result.prompt)-1)
            except:
                python_tokenizer = True

        if python_tokenizer:
            #print(f"[get_prompt] Info: using Python tokenizer.")
            conversation_template.messages = []
            
            conversation_template.append_message(conversation_template.roles[0], None)
            toks = self.tokenizer(conversation_template.get_prompt()).input_ids
            # find the token that indicates the following text is input
            delimiter = f"{conversation_template.roles[0]}"
            #print(f"[get_prompt] Debug: conversation_template.roles = '{conversation_template.roles}', delimiter = '{delimiter}', toks = '{toks}', original_toks = '{original_toks}'")
            #print(f"[get_prompt] Debug: conversation_template.roles = '{conversation_template.roles}', delimiter = '{delimiter}', toks = '{toks}'")
            result.slice_data.user_role = self.find_last_index_of_token(delimiter, toks)
            self.validate_slice_data('get_prompt - user_role_slice', result.slice_data)

            conversation_template.update_last_message(f"{self.instruction}")
            toks = self.tokenizer(conversation_template.get_prompt()).input_ids
            decoded_toks = get_decoded_tokens(self.tokenizer, toks)
            first_non_garbage_token = self.find_first_non_garbage_token(conversation_template, toks, decoded_toks, self.trash_fire_tokens, start_index = result.slice_data.user_role.stop)
            last_non_garbage_token = self.find_last_non_garbage_token(conversation_template, toks, decoded_toks, self.trash_fire_tokens, start_index = first_non_garbage_token) + 1
            result.slice_data.goal = slice(first_non_garbage_token, min(last_non_garbage_token, len(toks)))
            self.validate_slice_data('get_prompt - goal_slice', result.slice_data)

            separator = ' '
            if not self.instruction:
                separator = ''
            conversation_template.update_last_message(f"{self.instruction}{separator}{adversarial_string}")
            toks = self.tokenizer(conversation_template.get_prompt()).input_ids
            decoded_toks = get_decoded_tokens(self.tokenizer, toks)
            first_non_garbage_token = self.find_first_non_garbage_token(conversation_template, toks, decoded_toks, self.trash_fire_tokens, start_index = result.slice_data.goal.stop)
            last_non_garbage_token = self.find_last_non_garbage_token(conversation_template, toks, decoded_toks, self.trash_fire_tokens, start_index = first_non_garbage_token) + 1
            result.slice_data.control = slice(first_non_garbage_token, min(last_non_garbage_token, len(toks)))
            self.validate_slice_data('get_prompt - control_slice', result.slice_data)

            # find the token that marks a transition to output
            conversation_template.append_message(conversation_template.roles[1], None)
            toks = self.tokenizer(conversation_template.get_prompt()).input_ids
            decoded_toks = get_decoded_tokens(self.tokenizer, toks)
            first_non_garbage_token = self.find_first_non_garbage_token(conversation_template, toks, decoded_toks, self.trash_fire_tokens, start_index = result.slice_data.control.stop)
            last_non_garbage_token = self.find_last_non_garbage_token(conversation_template, toks, decoded_toks, self.trash_fire_tokens, start_index = first_non_garbage_token) + 1
            result.slice_data.assistant_role = slice(first_non_garbage_token, min(last_non_garbage_token, len(toks)))
            self.validate_slice_data('get_prompt - assistant_role_slice', result.slice_data)

            conversation_template.update_last_message(f"{self.target}")
            toks = self.tokenizer(conversation_template.get_prompt()).input_ids
            decoded_toks = get_decoded_tokens(self.tokenizer, toks)
            first_non_garbage_token = self.find_first_non_garbage_token(conversation_template, toks, decoded_toks, self.trash_fire_tokens, start_index = result.slice_data.assistant_role.stop)
            last_non_garbage_token = self.find_last_non_garbage_token(conversation_template, toks, decoded_toks, self.trash_fire_tokens, start_index = first_non_garbage_token) + 1
            result.slice_data.target = slice(first_non_garbage_token, min(last_non_garbage_token, len(toks)))
            self.validate_slice_data('get_prompt - target_slice', result.slice_data)
            
            result.slice_data.loss = slice(first_non_garbage_token - 1, last_non_garbage_token - 1)
            self.validate_slice_data('get_prompt - loss_slice', result.slice_data)
            
        else:
            #print(f"[get_prompt] Info: not using Python tokenizer")
            sys_template = None
            if hasattr(conversation_template, "system"):
                sys_template = conversation_template.system
            if sys_template is None and hasattr(conversation_template, "system_template"):
                sys_template = conversation_template.system_template
            if sys_template is None:
                print(f"[get_prompt] Warning: unable to find system template in conversation template for this model - using role 0 template instead")
                sys_template = conversation_template.roles[0]
            result.slice_data.system = slice(
                None, 
                encoding.char_to_token(len(sys_template))
            )
            self.validate_slice_data('get_prompt', result.slice_data)

            result.slice_data.user_role = slice(
                encoding.char_to_token(result.prompt.find(conversation_template.roles[0])),
                encoding.char_to_token(result.prompt.find(conversation_template.roles[0]) + len(conversation_template.roles[0]) + 1)
            )
            self.validate_slice_data('get_prompt', result.slice_data)

            result.slice_data.goal = slice(
                encoding.char_to_token(result.prompt.find(self.instruction)),
                encoding.char_to_token(result.prompt.find(self.instruction) + len(self.instruction))
            )
            self.validate_slice_data('get_prompt', result.slice_data)
            
            result.slice_data.control = slice(
                encoding.char_to_token(result.prompt.find(adversarial_string)),
                encoding.char_to_token(result.prompt.find(adversarial_string) + len(adversarial_string))
            )
            self.validate_slice_data('get_prompt', result.slice_data)
            
            result.slice_data.assistant_role = slice(
                encoding.char_to_token(result.prompt.find(conversation_template.roles[1])),
                encoding.char_to_token(result.prompt.find(conversation_template.roles[1]) + len(conversation_template.roles[1]) + 1)
            )
            self.validate_slice_data('get_prompt', result.slice_data)

            #self.print_slice_info("get_prompt", result.slice_data, toks)
            #print(f"[get_prompt] Debug: result.prompt = '{result.prompt}', self.target = '{self.target}'")
            prompt_find_self_target = result.prompt.find(self.target)
            #print(f"[get_prompt] Debug: prompt_find_self_target = '{prompt_find_self_target}'")
            prompt_find_self_target_c2t = encoding.char_to_token(prompt_find_self_target)
            if prompt_find_self_target_c2t is None:
                print(f"[get_prompt] Warning: got None for encoding.char_to_token(prompt_find_self_target). prompt_find_self_target = '{prompt_find_self_target}' using '{self.target}' in '{result.prompt}'. Using value {result.slice_data.assistant.stop} instead of None. This may indicate an error with the parsing logic.")
                prompt_find_self_target_c2t = result.slice_data.assistant.stop
            prompt_combined_c2t = None
            add_length = len(self.target) + 1
            while prompt_combined_c2t is None:
                prompt_combined_c2t = encoding.char_to_token(prompt_find_self_target + (add_length))
                add_length -= 1
                if add_length < 0:
                    prompt_combined_c2t = prompt_find_self_target_c2t
                    break
            # Subtract one more than the first valid value so that the length of the slice is correct
            #print(f"[get_prompt] Debug: prompt_find_self_target_c2t = '{prompt_find_self_target_c2t}', prompt_combined_c2t = '{prompt_combined_c2t}'")
            result.slice_data.target = slice(
                prompt_find_self_target_c2t,
                prompt_combined_c2t + 1
            )
            self.validate_slice_data('get_prompt', result.slice_data)
            
            result.slice_data.loss = slice(
                prompt_find_self_target_c2t - 1,
                prompt_combined_c2t
            )
            self.validate_slice_data('get_prompt', result.slice_data)

        #print(f"[get_prompt] Debug: conversation_template (after modifications) = '{conversation_template}'")
        #final_decoded_toks = get_decoded_tokens(self.tokenizer, toks)
        #print(f"[get_prompt] Debug: toks (after parsing) = '{toks}', final_decoded_toks = '{final_decoded_toks}'")
        
        result.full_prompt_ids = toks
        result.input_ids = toks[:result.slice_data.target.stop]

        self.print_slice_info("get_prompt", result.slice_data, toks)

        conversation_template.messages = []

        return result
    
    def get_input_ids(self, adv_string=None, force_python_tokenizer = False):
        result = self.get_prompt(adv_string=adv_string, force_python_tokenizer=force_python_tokenizer)
        #toks = self.tokenizer(result.prompt).input_ids
        #result.input_ids = toks[:self._target_slice.stop]
        #result.input_ids = toks[:result.slice_data.target.stop]        
        #toks_decoded = get_decoded_tokens(self.tokenizer, toks)
        #toks_decoded = get_decoded_tokens(self.tokenizer, result.input_ids)
        #result_input_ids_decoded = get_decoded_tokens(self.tokenizer, result.input_ids)
        #print(f"[get_input_ids] Debug: toks = '{toks}', toks_decoded = '{toks_decoded}', result.prompt = '{result.prompt}', result.input_ids = '{result.input_ids}', result_input_ids_decoded = '{result_input_ids_decoded}'")
        #print(f"[get_input_ids] Debug: result.prompt = '{result.prompt}', result.input_ids = '{result.input_ids}', result_input_ids_decoded = '{result_input_ids_decoded}'")
        return result

