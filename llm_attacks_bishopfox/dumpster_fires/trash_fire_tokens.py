#!/bin/env python

from llm_attacks_bishopfox import get_decoded_token
from llm_attacks_bishopfox import get_decoded_tokens
from llm_attacks_bishopfox import get_encoded_token
from llm_attacks_bishopfox import get_encoded_tokens
from llm_attacks_bishopfox import get_escaped_string
from llm_attacks_bishopfox import get_token_allow_and_deny_lists
from llm_attacks_bishopfox.util.util_functions import add_value_to_list_if_not_already_present
from llm_attacks_bishopfox.util.util_functions import add_values_to_list_if_not_already_present

# methods and classes related to wrangling the planet-sized inferno of garbage that is "special" LLM tokens

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
 
def find_first_occurrence_of_array_in_array(inner_array, outer_array, start_index = 0, stop_index = None):
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

def find_last_occurrence_of_array_in_array(inner_array, outer_array, start_index = 0, stop_index = None):
    result = None
    #print(f"[find_last_occurrence_of_array_in_array] Debug: Searching for '{inner_array}' in '{outer_array}' with start_index = {start_index} and stop_index = {stop_index}")
    len_inner = len(inner_array)
    len_outer = len(outer_array)
    if len_inner > len_outer or len_inner == 0 or len_outer == 0:
        print(f"[find_last_occurrence_of_array_in_array] Warning: cannot search for '{inner_array}' in '{outer_array}' with start_index = {start_index} and stop_index = {stop_index} - returning {result}")
        return result
    range_start = len_outer - len_inner
    if stop_index is not None:
        range_start = stop_index - len_inner
    range_end = start_index - 1
    if range_end < -1 or range_end > len_outer or range_start > len_outer or range_start < -1:
        print(f"[find_last_occurrence_of_array_in_array] Error: cannot search for '{inner_array}' in '{outer_array}' from index {range_start} to {range_end}' with start_index = {start_index} and stop_index = {stop_index}")
        sys.exit(1)
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

# so many spaghetti code special cases to handle madness like 'This is a special token, but you can't treat it as 100% trash fire because it's also a sentinel that the parser has to look for'
def is_conversation_role_token(conversation_template, token):
    if token is not None:
        if token.strip() != "":
            for c_role in conversation_template.roles:
                if token in c_role:
                    return True
    return False
        
def is_disastrous_dumpster_fire_token(trash_fire_tokens, conversation_template, token, decoded_token):
    token_is_a_pile_of_garbage_why_is_this_not_standardized_yet_you_ml_cowboys = False
    escaped_token = get_escaped_string(decoded_token)
    if token in trash_fire_tokens.token_ids:
        #print(f"[is_disastrous_dumpster_fire_token] Debug: marked token '{escaped_token}' (id {token}) as a flaming dumpster floating down the river because the token ID was in the list of trash fire tokens.")
        token_is_a_pile_of_garbage_why_is_this_not_standardized_yet_you_ml_cowboys = True
    if not token_is_a_pile_of_garbage_why_is_this_not_standardized_yet_you_ml_cowboys:
        if decoded_token in trash_fire_tokens.input_strings:
            #print(f"[is_disastrous_dumpster_fire_token] Debug: marked token '{escaped_token}' (id {token}) as a flaming dumpster floating down the river because the token was in the list of input strings that generated the list of trash fire tokens.")
            token_is_a_pile_of_garbage_why_is_this_not_standardized_yet_you_ml_cowboys = True
    if not token_is_a_pile_of_garbage_why_is_this_not_standardized_yet_you_ml_cowboys:
        for ctv in [decoded_token, decoded_token.strip()]:
            if ctv == "":
                #print(f"[is_disastrous_dumpster_fire_token] Debug: marked token '{escaped_token}' (id {token}) as a flaming dumpster floating down the river because it was empty or contained only whitespace.")
                token_is_a_pile_of_garbage_why_is_this_not_standardized_yet_you_ml_cowboys = True
                break
            if ctv in trash_fire_tokens.decoded_tokens:
                #print(f"[is_disastrous_dumpster_fire_token] Debug: marked token '{escaped_token}' (id {token}) as a flaming dumpster floating down the river because the decoded token was in the list of decoded trash fire tokens.")
                token_is_a_pile_of_garbage_why_is_this_not_standardized_yet_you_ml_cowboys = True
                break
            if ctv in trash_fire_tokens.input_strings:
                #print(f"[is_disastrous_dumpster_fire_token] Debug: marked token '{escaped_token}' (id {token}) as a flaming dumpster floating down the river because the decoded token was in the list of input strings used to generate the list of trash fire tokens.")
                token_is_a_pile_of_garbage_why_is_this_not_standardized_yet_you_ml_cowboys = True
                break
    
    # but wait! we can't exclude conversation role tokens! That would make parsing the output much harder!
    if conversation_template is not None:
        if token_is_a_pile_of_garbage_why_is_this_not_standardized_yet_you_ml_cowboys:
            if is_conversation_role_token(conversation_template, decoded_token.strip()):
                #print(f"[is_disastrous_dumpster_fire_token] Debug: marked token '{escaped_token}' (id {token}) as not being a flaming dumpster floating down the river because the decoded token was in the list of that are conversation roles for the current conversation template, even though it is still a flaming dumpster floating down the river.")
                token_is_a_pile_of_garbage_why_is_this_not_standardized_yet_you_ml_cowboys = False
    
    #if not token_is_a_pile_of_garbage_why_is_this_not_standardized_yet_you_ml_cowboys:
        #print(f"[is_disastrous_dumpster_fire_token] Debug: token '{escaped_token}' (id {token}) does not appear to be a garbage inferno.")
    
    return token_is_a_pile_of_garbage_why_is_this_not_standardized_yet_you_ml_cowboys

def remove_empty_leading_and_trailing_tokens(trash_fire_tokens, token_array, decoded_token_array, conversation_template = None, strip_decoded_tokens = False):
    #print(f"[remove_empty_leading_and_trailing_tokens] Debug: token_array = {token_array}, decoded_token_array = {decoded_token_array}")
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
        is_skippable_token = False
        #print(f"[remove_empty_leading_and_trailing_tokens - leading] Debug: checking token '{decoded_token_array[i]}', id {token_array[i]}")
        if decoded_token_temp == "":
            #print(f"[remove_empty_leading_and_trailing_tokens] Debug: token '{decoded_token_array[i]}' is whitespace or empty")
            is_skippable_token = True
        if not is_skippable_token:
            if is_disastrous_dumpster_fire_token(trash_fire_tokens, conversation_template, token_array[i], decoded_token_array[i]):
                is_skippable_token = True
        if not is_skippable_token:
            #print(f"[remove_empty_leading_and_trailing_tokens] Debug: token '{decoded_token_array[i]}' is not skippable")
            break
        first_non_empty_token += 1
    
    for i in range(len_token_array - 1, -1, -1):
        decoded_token_temp = decoded_token_array[i]
        if strip_decoded_tokens:
            decoded_token_temp = decoded_token_temp.strip()
            decoded_token_array[i] = decoded_token_temp
        is_skippable_token = False
        #print(f"[remove_empty_leading_and_trailing_tokens - trailing] Debug: checking token '{decoded_token_array[i]}', id {token_array[i]}")
        if decoded_token_temp == "":
            #print(f"[remove_empty_leading_and_trailing_tokens] Debug: token '{decoded_token_array[i]}' is whitespace or empty")
            is_skippable_token = True
        if not is_skippable_token:
            if is_disastrous_dumpster_fire_token(trash_fire_tokens, conversation_template, token_array[i], decoded_token_array[i]):
                is_skippable_token = True
        if not is_skippable_token:
            #print(f"[remove_empty_leading_and_trailing_tokens] Debug: token '{decoded_token_array[i]}' is not skippable")
            break
        last_non_empty_token -= 1
    
    actual_last_non_empty_token = last_non_empty_token + 1
    result_token_array = token_array[first_non_empty_token:actual_last_non_empty_token]
    result_decoded_token_array = decoded_token_array[first_non_empty_token:actual_last_non_empty_token]
    
    #print(f"[remove_empty_leading_and_trailing_tokens] Debug: token_array = '{token_array}', result_token_array = '{result_token_array}', decoded_token_array = '{decoded_token_array}', result_decoded_token_array = '{result_decoded_token_array}'")
    return result_token_array, result_decoded_token_array

def find_last_index_of_token(tokenizer, trash_fire_tokens, string_to_search_for, tokens, start_index = 0, stop_index = None, conversation_template = None):  
    if string_to_search_for == "":
        print(f"[find_last_index_of_token] Eror: cannot search for empty string '{string_to_search_for}' in tokens = '{tokens}'")
        sys.exit(1)
    decoded_tokens = get_decoded_tokens(tokenizer, tokens)
    if len(decoded_tokens) < 1:
        print(f"[find_last_index_of_token] Eror: got zero-length array '{decoded_tokens}' for tokens = '{tokens}'")
        sys.exit(1)
    #print(f"[find_last_index_of_token] Debug: decoded_tokens = '{decoded_tokens}' for tokens = '{tokens}'")
    string_tokens = get_encoded_token(tokenizer, string_to_search_for)
    if len(decoded_tokens) < 1:
        print(f"[find_last_index_of_token] Eror: got zero-length array '{string_tokens}' when re-encoding tokens '{tokens}'")
        sys.exit(1)
    
    #print(f"[find_last_index_of_token] Debug: string_tokens = '{string_tokens}' for string '{string_to_search_for}'")
    # hacky workarounds for garbagey behaviour by LLMs
    string_to_search_for_array = string_to_search_for.split(" ")
    if len(string_to_search_for_array) < 1:
        print(f"[find_last_index_of_token] Eror: got zero-length array '{string_to_search_for_array}' when splitting '{string_to_search_for}' into words")
        sys.exit(1)
    #first_search_word = string_to_search_for_array[0]
    #len_first_search_word = len(first_search_word)
    decoded_string_tokens = get_decoded_tokens(tokenizer, string_tokens)
    if len(decoded_string_tokens) < 1:
        print(f"[find_last_index_of_token] Eror: got zero-length array '{decoded_string_tokens}' when decoding string_tokens '{string_tokens}'")
        sys.exit(1)

    string_tokens, decoded_string_tokens = remove_empty_leading_and_trailing_tokens(trash_fire_tokens, string_tokens, decoded_string_tokens, conversation_template = conversation_template, strip_decoded_tokens = True)
    
    #print(f"[find_last_index_of_token] Debug: searching for last occurrence of '{string_to_search_for}' (tokenized as '{decoded_string_tokens}') in '{decoded_tokens}'")
    result_start = find_last_occurrence_of_array_in_array(string_tokens, tokens, start_index=start_index, stop_index=stop_index)
    result_stop = None
    if result_start is None:
        # try to find cases where tokens have spaces on either side or not at all
        decoded_tokens_processed_1 = []
        decoded_tokens_processed_2 = []
        for i in range(0, len(decoded_tokens)):
            processed_token = decoded_tokens[i].strip()
            decoded_tokens_processed_1.append(processed_token)
            decoded_tokens_processed_2.append(decoded_tokens[i])
        result_start = find_last_occurrence_of_array_in_array(string_to_search_for_array, decoded_tokens_processed_1, start_index=start_index, stop_index=stop_index)
        if result_start is None:
            result_start = find_last_occurrence_of_array_in_array(string_to_search_for_array, decoded_tokens_processed_2, start_index=start_index, stop_index=stop_index)
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


# dynamically determine the last token in a set of tokens 
# that get_prompt should consider
# like '</s>', '<s>', '\n', '###', or ' '
def find_last_non_garbage_token(conversation_template, tokens, decoded_tokens, trash_fire_tokens, start_index = 0, stop_index = None):
    #decoded_tokens = get_decoded_tokens(tokenizer, tokens)
    result = None
    range_end = len(tokens)
    if stop_index is not None:
        range_end = stop_index
    for i in range(start_index, range_end):
        token_is_a_pile_of_garbage_why_is_this_not_standardized_yet_you_ml_cowboys = is_disastrous_dumpster_fire_token(trash_fire_tokens, conversation_template, tokens[i], decoded_tokens[i])
        if not token_is_a_pile_of_garbage_why_is_this_not_standardized_yet_you_ml_cowboys:
            result = i
    if result is None:
        raise Exception(f"[find_last_non_garbage_token] Could not find a token that wasn't an absolute dumpster fire in '{decoded_tokens}' from index {start_index} to {range_end}, please, stop the madness right now.")
    #print(f"[find_last_non_garbage_token] Debug: last non-garbage token in '{decoded_tokens}' from index {start_index} to {range_end} ('{decoded_tokens[start_index:range_end]}') is index {result}, '{decoded_tokens[result]}'")
    return result

def find_first_non_garbage_token(conversation_template, tokens, decoded_tokens, trash_fire_tokens, start_index = 0, stop_index = None):
    #decoded_tokens = get_decoded_tokens(tokenizer, tokens)
    result = None
    range_end = len(tokens)
    if stop_index is not None:
        range_end = stop_index
    for i in range(start_index, range_end):
        token_is_a_pile_of_garbage_why_is_this_not_standardized_yet_you_ml_cowboys = is_disastrous_dumpster_fire_token(trash_fire_tokens, conversation_template, tokens[i], decoded_tokens[i])
        if not token_is_a_pile_of_garbage_why_is_this_not_standardized_yet_you_ml_cowboys:
            #print(f"[find_first_non_garbage_token] Debug: first non-garbage token in '{decoded_tokens}' from index {start_index} to {range_end} is index {i}, '{decoded_tokens[i]}'")
            return i
    if result is None:
        raise Exception(f"[find_first_non_garbage_token] Could not find a token that wasn't an absolute dumpster fire in '{decoded_tokens}' from index {start_index} to {range_end}, please, stop the madness right now.")
    return result

class TrashFireTokenCollection:
    def __init__(self):
        self.input_strings = None
        self.token_ids = None
        self.decoded_tokens = None
    
    @staticmethod
    def append_default_special_string_denylist(string_list):
        special_string_denylist = [ '</s>', '<s>', '###', '##', '#', "\n", "\\n", "\r", "\\r", "\r\n", "\\r\\", "", ":", " ", "\t",
                                    '<sep>',
                                    '<|system|>'
                                    '<|end|>',
                                    '<bos>',
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
        string_list = add_values_to_list_if_not_already_present(string_list, special_string_denylist)
        return string_list
    
    @staticmethod
    def get_meticulously_curated_trash_fire_token_collection(tokenizer, conversation_template):
        result = TrashFireTokenCollection()
        
        result.input_strings = TrashFireTokenCollection.append_default_special_string_denylist([])
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
        
        #result.token_ids = get_token_denylist(tokenizer, result.input_strings, device='cpu', filter_special_tokens = True, filter_additional_special_tokens = True, filter_whitespace_tokens = True, additional_token_ids = additional_flaming_dumpster_ids)
        allow_and_denylists = get_token_allow_and_deny_lists(tokenizer, result.input_strings, device='cpu', filter_special_tokens = True, filter_additional_special_tokens = True, filter_whitespace_tokens = True, additional_token_ids = additional_flaming_dumpster_ids)
        result.token_ids = allow_and_denylists.denylist
        
        result.decoded_tokens = get_decoded_tokens(tokenizer, result.token_ids)
        #print(f"[get_meticulously_curated_trash_fire_token_collection] Debug: result.input_strings = '{result.input_strings}', result.token_ids = '{result.token_ids}', result.decoded_tokens = '{result.decoded_tokens}'")
        return result
