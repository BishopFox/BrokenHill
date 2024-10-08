#!/bin/env python

import sys
import torch

from llm_attacks_bishopfox.util.util_functions import add_value_to_list_if_not_already_present
from llm_attacks_bishopfox.util.util_functions import add_values_to_list_if_not_already_present
from llm_attacks_bishopfox.util.util_functions import append_single_or_list_members
from llm_attacks_bishopfox.util.util_functions import find_first_occurrence_of_array_in_array
from llm_attacks_bishopfox.util.util_functions import find_index_of_first_nonmatching_element
from llm_attacks_bishopfox.util.util_functions import find_last_occurrence_of_array_in_array
from llm_attacks_bishopfox.util.util_functions import get_escaped_string
from llm_attacks_bishopfox.util.util_functions import remove_whitespace_and_nonprintable_characters

from tokenizers import AddedToken

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

class TrashFireTokenException(Exception):
    pass
    # I bet you'd like to pass on the trash fire to someone else!

def get_decoded_token(tokenizer, token):
    result = None
    #print(f"[get_decoded_token] Debug: decoding token '{token}'")
    if isinstance(token, type(None)):
        print(f"[get_decoded_token] Warning: a null token ID was passed to this function. This usually indicates a bug.")
        return None
    token_to_decode = token
    # workaround for models like Gemma that need all tokens to be in the form of a list
    wrap_in_list = False
    if not isinstance(token, list) and not isinstance(token, torch.Tensor):
        wrap_in_list = True
    if wrap_in_list:
        token_to_decode = [ token ]
        #print(f"[get_decoded_token] Debug: converted '{token}' to '{token_to_decode}'")
    #result = tokenizer.decode(token_to_decode, skip_special_tokens=False)
    try:
        #result = tokenizer.decode(token_to_decode, skip_special_tokens=True)
        result = tokenizer.decode(token_to_decode, skip_special_tokens=False)
    except Exception as e:
        print(f"[get_decoded_token] Error decoding token {token_to_decode}: {e}")
        result = None
    #print(f"[get_decoded_token] Debug: decoded token '{token}' to '{result}'")
    return result

def get_decoded_tokens(tokenizer, tokens, recursively_process_arrays = False):
    #print(f"[get_decoded_tokens] Debug: decoding tokens '{tokens}'")
    decoded_tokens = []
    token_list = tokens
    if isinstance(tokens, torch.Tensor):
        token_list = tokens.tolist()
    if isinstance(token_list, list):
        for tn in range(0, len(token_list)):
            if recursively_process_arrays:
                dt = get_decoded_tokens(tokenizer, token_list[tn])
            else:
                dt = get_decoded_token(tokenizer, token_list[tn])
            decoded_tokens.append(dt)
    else:
        dt = get_decoded_token(tokenizer, tokens)
        decoded_tokens.append(dt)
    #print(f"[get_decoded_tokens] Debug: decoded tokens '{tokens}' to '{decoded_tokens}'")
    return decoded_tokens

def get_encoded_token(tokenizer, token, exterminate_all_cowboy_nonsense = False):
    #print(f"[get_encoded_token] Debug: encoding token '{token}'")
    result = None
    try:
        #result = tokenizer.encode(token, skip_special_tokens=True)
        result = None
        if exterminate_all_cowboy_nonsense:
            result = encode_string_for_real_without_any_cowboy_funny_business(tokenizer, token)
        else:
            result = tokenizer.encode(token)
        if isinstance(result, type(None)):
            print(f"[get_encoded_token] Warning: the tokenizer returned None when asked to encode the token '{token}'. This usually indicates a bug.")
    except Exception as e:
        print(f"[get_encoded_token] Error encoding token {token}: {e}")
    return result

def get_encoded_tokens(tokenizer, tokens, exterminate_all_cowboy_nonsense = False):
    encoded_tokens = []
    for tn in range(0, len(tokens)):
        et = get_encoded_token(tokenizer, tokens[tn], exterminate_all_cowboy_nonsense = exterminate_all_cowboy_nonsense)
        encoded_tokens.append(et)
    return encoded_tokens


# Gets the array of tokens that represent JUST A STRING, *without* any blazing trash fire tokens included, FOR REAL
def encode_string_for_real_without_any_cowboy_funny_business(tokenizer, string):
    if string is None:
        return None
    if string == "":
        return []
    string_encoded = tokenizer.encode(string)
    #print(f"[encode_string_for_real_without_any_cowboy_funny_business] string_encoded = {string_encoded}")
    # First, strip any leading dumpster inferno content
    # make a single-character string that definitely does not start with the same character as the input string
    not_the_same_string_at_all = "A"
    if string[0] == "A":
        not_the_same_string_at_all = "B"
    not_the_same_string_at_all_encoded = tokenizer.encode(not_the_same_string_at_all)
    #print(f"[encode_string_for_real_without_any_cowboy_funny_business] not_the_same_string_at_all_encoded = {not_the_same_string_at_all_encoded}")
    start_index = find_index_of_first_nonmatching_element(string_encoded, not_the_same_string_at_all_encoded)
    # Second, check for any ever-burning beacons of waste at the end of the result
    # make a string that is the same as the input, but has more characters at the end
    string_with_chaff = f"{string} 1987"
    string_with_chaff_encoded = tokenizer.encode(string_with_chaff)
    #print(f"[encode_string_for_real_without_any_cowboy_funny_business] string_with_chaff_encoded = {string_with_chaff_encoded}")
    stop_index = find_index_of_first_nonmatching_element(string_encoded, string_with_chaff_encoded)
    result = string_encoded[start_index:stop_index]
    #print(f"[encode_string_for_real_without_any_cowboy_funny_business] input = '{string}', result = {result}")
    return result

# so many spaghetti code special cases to handle madness like 'This is a special token, but you can't treat it as 100% trash fire because it's also a sentinel that the parser has to look for. Also, sometimes it's more than one token! But not always!'
def is_conversation_role_token(conversation_template, token):
    if not isinstance(token, type(None)):
        if token.strip() != "":
            for c_role in conversation_template.roles:
                #print(f"[is_conversation_role_token] Debug: searching for '{token}' in conversation role '{c_role}'")
                if token in c_role:
                    #print(f"[is_conversation_role_token] Debug: '{token}' is in conversation role '{c_role}'")
                    return True
                token_minus_whitespace = remove_whitespace_and_nonprintable_characters(token)
                c_role_minus_whitespace = remove_whitespace_and_nonprintable_characters(c_role)
                if token in c_role:
                    #print(f"[is_conversation_role_token] Debug: '{token_minus_whitespace}' is in conversation role '{c_role_minus_whitespace}'")
                    return True
                if c_role in token:
                    #print(f"[is_conversation_role_token] Debug: conversation role '{c_role_minus_whitespace}' is in token '{token_minus_whitespace}'")
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
                #print(f"[is_disastrous_dumpster_fire_token] Debug: marked token '{escaped_token}' (id {token}) as not being a flaming dumpster floating down the river because the decoded token was in the list of tokens that indicate conversation role changes for the current conversation template, even though it is still a flaming dumpster floating down the river.")
                token_is_a_pile_of_garbage_why_is_this_not_standardized_yet_you_ml_cowboys = False
    
    #if not token_is_a_pile_of_garbage_why_is_this_not_standardized_yet_you_ml_cowboys:
        #print(f"[is_disastrous_dumpster_fire_token] Debug: token '{escaped_token}' (id {token}) does not appear to be a garbage inferno.")
    
    return token_is_a_pile_of_garbage_why_is_this_not_standardized_yet_you_ml_cowboys

def remove_empty_and_trash_fire_leading_and_trailing_tokens(trash_fire_tokens, 
        token_array, 
        decoded_token_array, 
        conversation_template = None, 
        strip_decoded_tokens = False, 
        remove_empty_leading_and_trailing_tokens = True, 
        remove_leading_and_trailing_trash_fire_tokens = True,
        check_whitespace_tokens_for_trash_fire = False):
    #print(f"[remove_empty_and_trash_fire_leading_and_trailing_tokens] Debug: token_array = {token_array}, decoded_token_array = {decoded_token_array}")
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
        #print(f"[remove_empty_and_trash_fire_leading_and_trailing_tokens - leading] Debug: checking token '{decoded_token_array[i]}', id {token_array[i]}")
        if remove_empty_leading_and_trailing_tokens:
            if decoded_token_temp == "":
                #print(f"[remove_empty_and_trash_fire_leading_and_trailing_tokens] Debug: token '{decoded_token_array[i]}' is whitespace or empty")
                is_skippable_token = True
        if not is_skippable_token:
            if remove_leading_and_trailing_trash_fire_tokens:
                if check_whitespace_tokens_for_trash_fire or decoded_token_array[i].strip() != "":
                    if is_disastrous_dumpster_fire_token(trash_fire_tokens, conversation_template, token_array[i], decoded_token_array[i]):
                        #print(f"[remove_empty_and_trash_fire_leading_and_trailing_tokens] Debug: token '{decoded_token_array[i]}' is a blazing trash bin bringing light to the darkness of the running waters as they make their way to the sea.")
                        is_skippable_token = True
        if not is_skippable_token:
            #print(f"[remove_empty_and_trash_fire_leading_and_trailing_tokens] Debug: token '{decoded_token_array[i]}' is not skippable")
            break
        first_non_empty_token += 1
    
    for i in range(len_token_array - 1, -1, -1):
        decoded_token_temp = decoded_token_array[i]
        if strip_decoded_tokens:
            decoded_token_temp = decoded_token_temp.strip()
            decoded_token_array[i] = decoded_token_temp
        is_skippable_token = False
        #print(f"[remove_empty_and_trash_fire_leading_and_trailing_tokens - trailing] Debug: checking token '{decoded_token_array[i]}', id {token_array[i]}")
        if remove_empty_leading_and_trailing_tokens:
            if decoded_token_temp == "":
                #print(f"[remove_empty_and_trash_fire_leading_and_trailing_tokens] Debug: token '{decoded_token_array[i]}' is whitespace or empty")
                is_skippable_token = True
        if not is_skippable_token:
            if remove_leading_and_trailing_trash_fire_tokens:
                if check_whitespace_tokens_for_trash_fire or decoded_token_array[i].strip() != "":
                    if is_disastrous_dumpster_fire_token(trash_fire_tokens, conversation_template, token_array[i], decoded_token_array[i]):
                        #print(f"[remove_empty_and_trash_fire_leading_and_trailing_tokens] Debug: token '{decoded_token_array[i]}' is a blazing trash bin bringing light to the darkness of the running waters as they make their way to the sea.")
                        is_skippable_token = True
        if not is_skippable_token:
            #print(f"[remove_empty_and_trash_fire_leading_and_trailing_tokens] Debug: token '{decoded_token_array[i]}' is not skippable")
            break
        last_non_empty_token -= 1
    
    actual_last_non_empty_token = last_non_empty_token + 1
    result_token_array = token_array[first_non_empty_token:actual_last_non_empty_token]
    result_decoded_token_array = decoded_token_array[first_non_empty_token:actual_last_non_empty_token]
    
    #print(f"[remove_empty_and_trash_fire_leading_and_trailing_tokens] Debug: token_array = '{token_array}', result_token_array = '{result_token_array}', decoded_token_array = '{decoded_token_array}', result_decoded_token_array = '{result_decoded_token_array}'")
    return result_token_array, result_decoded_token_array

# This actually returns a slice that describes the start and end indices in the token array
def find_first_index_of_token(tokenizer, trash_fire_tokens, string_to_search_for, tokens, decoded_tokens, start_index = 0, stop_index = None, conversation_template = None, strip_leading_and_trailing_tokens = False, remove_empty_leading_and_trailing_tokens = True, remove_leading_and_trailing_trash_fire_tokens = True):
    return find_index_of_token(tokenizer, trash_fire_tokens, string_to_search_for, tokens, decoded_tokens, start_index = start_index, stop_index = stop_index, conversation_template = conversation_template, find_last = False, strip_leading_and_trailing_tokens = strip_leading_and_trailing_tokens, remove_empty_leading_and_trailing_tokens = remove_empty_leading_and_trailing_tokens, remove_leading_and_trailing_trash_fire_tokens = remove_leading_and_trailing_trash_fire_tokens)

# This actually returns a slice that describes the start and end indices in the token array
def find_last_index_of_token(tokenizer, trash_fire_tokens, string_to_search_for, tokens, decoded_tokens, start_index = 0, stop_index = None, conversation_template = None, strip_leading_and_trailing_tokens = False, remove_empty_leading_and_trailing_tokens = True, remove_leading_and_trailing_trash_fire_tokens = True):
    return find_index_of_token(tokenizer, trash_fire_tokens, string_to_search_for, tokens, decoded_tokens, start_index = start_index, stop_index = stop_index, conversation_template = conversation_template, find_last = True, strip_leading_and_trailing_tokens = strip_leading_and_trailing_tokens, remove_empty_leading_and_trailing_tokens = remove_empty_leading_and_trailing_tokens, remove_leading_and_trailing_trash_fire_tokens = remove_leading_and_trailing_trash_fire_tokens)

# This actually returns a slice that describes the start and end indices in the token array
def find_index_of_token(tokenizer, trash_fire_tokens, string_to_search_for, tokens, decoded_tokens, start_index = 0, stop_index = None, conversation_template = None, find_last = False, strip_leading_and_trailing_tokens = False, remove_empty_leading_and_trailing_tokens = True, remove_leading_and_trailing_trash_fire_tokens = True):  
    if string_to_search_for == "":
        raise TrashFireTokenException(f"[find_index_of_token] Error: cannot search for empty string '{string_to_search_for}' in tokens = '{tokens}'")
    # decoded_tokens = get_decoded_tokens(tokenizer, tokens)
    # if len(decoded_tokens) < 1:
        # raise TrashFireTokenException(f"[find_index_of_token] Error: got zero-length array '{decoded_tokens}' for tokens = '{tokens}'")
    #print(f"[find_index_of_token] Debug: decoded_tokens = '{decoded_tokens}' for tokens = '{tokens}'")
    #string_tokens = encode_string_for_real_without_any_cowboy_funny_business(tokenizer, string_to_search_for)
    string_token_variations = []
    # did you know!
    # one of the reasons this function is in the trash_fire_tokens.py file instead of somewhere else is because some tokenizers (I'm looking at *you*, Gemma and Llama-3!) will tokenize the strings "this", " this", "this ", etc. to different values.
    string_meta_variations = [ string_to_search_for, string_to_search_for.strip() ]
    string_variations = []
    for smv in string_meta_variations:
        string_variations = add_value_to_list_if_not_already_present(string_variations, smv)
        string_variations = add_value_to_list_if_not_already_present(string_variations, f" {smv}")
        string_variations = add_value_to_list_if_not_already_present(string_variations, f"{smv} ")
        string_variations = add_value_to_list_if_not_already_present(string_variations, f" {smv} ")
    for i in range(0, len(string_variations)):
        sv = string_variations[i]
        sv_encoded = encode_string_for_real_without_any_cowboy_funny_business(tokenizer, sv)
        string_token_variations.append(sv_encoded)
    
    failure_messages = []
    
    for string_token_set_num in range(0, len(string_token_variations)):
        string_tokens = string_token_variations[string_token_set_num]
        if len(string_tokens) < 1:
            raise TrashFireTokenException(f"[find_index_of_token] Error: got zero-length array '{string_tokens}' when re-encoding tokens '{tokens}'")
        
        #print(f"[find_index_of_token] Debug: string_tokens = '{string_tokens}' for string '{string_to_search_for}'")
        # hacky workarounds for garbagey behaviour by LLMs
        #string_to_search_for_array = string_to_search_for.split(" ")
        current_string_to_search_for = string_variations[string_token_set_num]
        string_to_search_for_array = current_string_to_search_for.split(" ")
        if len(string_to_search_for_array) < 1:
            raise TrashFireTokenException(f"[find_index_of_token] Error: got zero-length array '{string_to_search_for_array}' when splitting '{current_string_to_search_for}' into words")
        decoded_string_tokens = get_decoded_tokens(tokenizer, string_tokens)
        if len(decoded_string_tokens) < 1:
            raise TrashFireTokenException(f"[find_index_of_token] Error: got zero-length array '{decoded_string_tokens}' when decoding string_tokens '{string_tokens}'")

        #trimmed_string_tokens = string_tokens
        #trimmed_decoded_string_tokens = decoded_string_tokens

        trimmed_string_tokens, trimmed_decoded_string_tokens = remove_empty_and_trash_fire_leading_and_trailing_tokens(trash_fire_tokens, 
            string_tokens, 
            decoded_string_tokens, 
            conversation_template = conversation_template, 
            strip_decoded_tokens = strip_leading_and_trailing_tokens, 
            remove_empty_leading_and_trailing_tokens = remove_empty_leading_and_trailing_tokens, 
            remove_leading_and_trailing_trash_fire_tokens = remove_leading_and_trailing_trash_fire_tokens)

        if len(trimmed_decoded_string_tokens) < 1:
            raise TrashFireTokenException(f"[find_index_of_token] Error: got zero-length array '{trimmed_decoded_string_tokens}' after removing ignored tokens from '{decoded_string_tokens}'")
        if len(trimmed_string_tokens) < 1:
            raise TrashFireTokenException(f"[find_index_of_token] Error: got zero-length array '{trimmed_string_tokens}' after removing ignored tokens from '{string_tokens}'")

        string_tokens = trimmed_string_tokens
        decoded_string_tokens = trimmed_decoded_string_tokens
        
        #print(f"[find_index_of_token] Debug: searching for '{current_string_to_search_for}' (tokenized as '{decoded_string_tokens}') in '{decoded_tokens}' from index {start_index} to {stop_index}")
        result_start = None
        if find_last:
            result_start = find_last_occurrence_of_array_in_array(string_tokens, tokens, start_index = start_index, stop_index = stop_index)
        else:
            result_start = find_first_occurrence_of_array_in_array(string_tokens, tokens, start_index = start_index, stop_index = stop_index)
        result_stop = None
        is_failure = False
        if isinstance(result_start, type(None)):
            # try to find cases where tokens have spaces on either side or not at all
            decoded_tokens_processed_1 = []
            decoded_tokens_processed_2 = []
            for i in range(0, len(decoded_tokens)):
                processed_token = decoded_tokens[i].strip()
                decoded_tokens_processed_1.append(processed_token)
                decoded_tokens_processed_2.append(decoded_tokens[i])
            # look for the first word as one string as well as individual decoded token IDs
            for search_array in [ string_to_search_for_array, decoded_string_tokens ]:
                for in_array in [ decoded_tokens_processed_1, decoded_tokens_processed_2 ]:
                    if isinstance(result_start, type(None)):
                        if find_last:
                            result_start = find_last_occurrence_of_array_in_array(search_array, in_array, start_index=start_index, stop_index=stop_index)
                        else:
                            result_start = find_first_occurrence_of_array_in_array(search_array, in_array, start_index=start_index, stop_index=stop_index)
                    else:
                        break
                if not isinstance(result_start, type(None)):
                    break
            if isinstance(result_start, type(None)):
                failure_messages.append(f"Could not find '{current_string_to_search_for}' (tokenized as '{decoded_string_tokens}') in '{decoded_tokens}', '{decoded_tokens_processed_1}', or '{decoded_tokens_processed_2}' from indices {start_index} to {stop_index}")
                is_failure = True
            else:
                #result_stop = result_start + len(string_to_search_for_array)
                result_stop = result_start + len(string_tokens)
                # This issue is so frequent that enabling this error is too noisy
                #print(f"[find_index_of_token] Warning: could not find '{current_string_to_search_for}' (tokenized as '{decoded_string_tokens}') in '{decoded_tokens}', but found the close approximation '{string_to_search_for_array}' in '{decoded_tokens_processed_1}' or '{decoded_tokens_processed_2}' and will use that position instead. This may be due to using a buggy LLM that considers e.g. 'Human' and ' Human' different tokens, but uses both values for similar purposes internally.")
                
        else:
            result_stop = result_start + len(string_tokens)
        
        if not is_failure:
            result = slice(result_start, result_stop)
            #print(f"[find_index_of_token] Debug: result = '{result}'")
            return result
    exception_message = ""
    for i in range(0, len(failure_messages)):
        exception_message += f"{failure_messages[i]}\n\n"
    raise Exception(exception_message)

# dynamically determine the last token in a set of tokens 
# that get_prompt should consider
# like '</s>', '<s>', '\n', '###', or ' '
def find_last_non_garbage_token(conversation_template, tokens, decoded_tokens, trash_fire_tokens, start_index = 0, stop_index = None):
    result = None
    range_end = len(tokens)
    if not isinstance(stop_index, type(None)):
        range_end = stop_index
    for i in range(start_index, range_end):
        token_is_a_pile_of_garbage_why_is_this_not_standardized_yet_you_ml_cowboys = is_disastrous_dumpster_fire_token(trash_fire_tokens, conversation_template, tokens[i], decoded_tokens[i])
        if not token_is_a_pile_of_garbage_why_is_this_not_standardized_yet_you_ml_cowboys:
            result = i
    if isinstance(result, type(None)):
        raise Exception(f"[find_last_non_garbage_token] Could not find a token that wasn't an absolute dumpster fire in '{decoded_tokens}' from index {start_index} to {range_end}, please, stop the madness right now.")
    #print(f"[find_last_non_garbage_token] Debug: last non-garbage token in '{decoded_tokens}' from index {start_index} to {range_end} ('{decoded_tokens[start_index:range_end]}') is index {result}, '{decoded_tokens[result]}'")
    return result

def find_first_non_garbage_token(conversation_template, tokens, decoded_tokens, trash_fire_tokens, start_index = 0, stop_index = None):
    result = None
    range_end = len(tokens)
    if not isinstance(stop_index, type(None)):
        range_end = stop_index
    for i in range(start_index, range_end):
        token_is_a_pile_of_garbage_why_is_this_not_standardized_yet_you_ml_cowboys = is_disastrous_dumpster_fire_token(trash_fire_tokens, conversation_template, tokens[i], decoded_tokens[i])
        if not token_is_a_pile_of_garbage_why_is_this_not_standardized_yet_you_ml_cowboys:
            #print(f"[find_first_non_garbage_token] Debug: first non-garbage token in '{decoded_tokens}' from index {start_index} to {range_end} is index {i}, '{decoded_tokens[i]}'")
            return i
    if isinstance(result, type(None)):
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
        # but be careful about trash that might be equivalent to an empty string or whitespace
        potential_trash = []
        potential_trash.append(tokenizer.bos_token)
        potential_trash.append(tokenizer.eos_token)
        potential_trash.append(tokenizer.unk_token)
        potential_trash.append(tokenizer.pad_token)
        additional_flaming_dumpster_ids = []
        additional_flaming_dumpster_ids.append(tokenizer.bos_token_id)
        additional_flaming_dumpster_ids.append(tokenizer.eos_token_id)
        additional_flaming_dumpster_ids.append(tokenizer.unk_token_id)
        additional_flaming_dumpster_ids.append(tokenizer.pad_token_id)
        
        # get all of the special trash from the conversation template as well
        if hasattr(conversation_template, "stop_token_ids"):
            additional_flaming_dumpster_ids = append_single_or_list_members(additional_flaming_dumpster_ids, conversation_template.stop_token_ids, ignore_if_none = True)

        if hasattr(conversation_template, "stop_str"):
            potential_trash = append_single_or_list_members(potential_trash, conversation_template.stop_str, ignore_if_none = True)
        if hasattr(conversation_template, "sep"):
            potential_trash = append_single_or_list_members(result.input_strings, conversation_template.sep, ignore_if_none = True)
        if hasattr(conversation_template, "sep2"):
            potential_trash = append_single_or_list_members(potential_trash, conversation_template.sep2, ignore_if_none = True)
        
        for i in range(0, len(potential_trash)):
            is_really_trash = True
            pt = potential_trash[i]
            if pt is None:
                is_really_trash = False
            else:
                if pt.strip() == "":
                    is_really_trash = False
            if is_really_trash:
                result.input_strings = add_value_to_list_if_not_already_present(result.input_strings, pt)
        
        allow_and_denylists = get_token_allow_and_deny_lists(tokenizer, result.input_strings, device='cpu', filter_special_tokens = True, filter_additional_special_tokens = True, filter_whitespace_tokens = True, additional_token_ids = additional_flaming_dumpster_ids)
        result.token_ids = allow_and_denylists.denylist
        
        result.decoded_tokens = get_decoded_tokens(tokenizer, result.token_ids)
        #print(f"[get_meticulously_curated_trash_fire_token_collection] Debug: result.input_strings = '{result.input_strings}', result.token_ids = '{result.token_ids}', result.decoded_tokens = '{result.decoded_tokens}'")
        return result

def get_nonascii_token_list(tokenizer, tokenizer_vocabulary_decoded):
    def is_ascii(s):
        if isinstance(s, type(None)):
            return False
        return s.isascii()

    result = []
    for i in range(3, tokenizer.vocab_size):
        decoded_token = tokenizer_vocabulary_decoded[i]
        if not isinstance(decoded_token, type(None)):
            if not is_ascii(decoded_token):
                if i not in result:
                    result.append(i)
    
    return result

def get_nonprintable_token_list(tokenizer, tokenizer_vocabulary_decoded):
    def is_printable(s):
        if isinstance(s, type(None)):
            return False
        return s.isprintable()

    result = []
    for i in range(3, tokenizer.vocab_size):
        decoded_token = tokenizer_vocabulary_decoded[i]
        if not isinstance(decoded_token, type(None)):
            if not is_printable(decoded_token):
                if i not in result:
                    result.append(i)
    
    return result

def get_nonmatching_token_list(tokenizer, tokenizer_vocabulary_decoded, filter_regex):
    nonmatching_tokens = []
    for i in range(3, tokenizer.vocab_size):
        dt = tokenizer_vocabulary_decoded[i]
        if not isinstance(dt, type(None)):
            if not filter_regex.search(dt):
                nonmatching_tokens.append(i)
                #print(f"[get_nonmatching_token_list] Debug: excluding '{dt}' because it did not match the specified regular expression.")
                #if "#" in dt:
                #    print(f"[get_nonmatching_token_list] Debug: excluding '{dt}' because it did not match the specified regular expression.")
            #else:
            #    if "#" in dt:
            #        print(f"[get_nonmatching_token_list] Debug: not excluding '{dt}' because it matched the specified regular expression.")
    
    return nonmatching_tokens

def get_token_list_as_tensor(token_list, device='cpu'):    
    return torch.tensor(token_list, device=device)    

class TokenAllowAndDenyList:
    def __init__(self):
        self.allowlist = []
        self.denylist = []

def add_token_ids_from_strings(token_allow_and_denylist, tokenizer, tokenizer_vocabulary_decoded, string_list, case_sensitive = True):
    for i in range(0, len(string_list)):
        current_string = string_list[i]

        #current_string_escaped = get_escaped_string(current_string)
        denied_toks_original = get_encoded_token(tokenizer, current_string)
        #print(f"[get_token_denylist] Debug: got token(s) '{denied_toks_original}' from string '{current_string_escaped}'")
        # If a given string was transformed into more than one token, ignore it
        
        if denied_toks_original is not None:
            if isinstance(denied_toks_original, list):
                if len(denied_toks_original) == 1:
                    #print(f"[get_token_denylist] Debug: converting token '{denied_toks_original}' to a single value")
                    denied_toks_original = denied_toks_original[0]
                else:
                    #print(f"[get_token_denylist] Debug: did not add tokens '{denied_toks_original}' to the denylist because a single string became multiple tokens")
                    denied_toks_original = None
        if denied_toks_original is not None:
            #print(f"[get_token_denylist] Debug: added token {denied_toks_original} to the denylist")
            token_allow_and_denylist.denylist = add_value_to_list_if_not_already_present(token_allow_and_denylist.denylist, denied_toks_original)
            # if denied_toks_original not in token_allow_and_denylist.denylist:
                # #print(f"[get_token_denylist] Debug: added token {denied_toks_original} to the denylist")
                # token_allow_and_denylist.denylist.append(denied_toks_original)
        # also check to see if any tokens are equivalent to the string value when decoded, 
        # even if the encoder didn't return them
        for j in range(0, tokenizer.vocab_size):
            candidate_token = tokenizer_vocabulary_decoded[j]
            #candidate_token_escaped = get_escaped_string(candidate_token)
            #if candidate_token == current_string:
            if not isinstance(candidate_token, type(None)):
                candidate_token_comparison = candidate_token.strip()
                current_string_comparison = current_string.strip()
                if not case_sensitive:
                    candidate_token_comparison = candidate_token_comparison.lower()
                    current_string_comparison = current_string_comparison.lower()
                if candidate_token_comparison == current_string_comparison:
                    token_allow_and_denylist.denylist = add_value_to_list_if_not_already_present(token_allow_and_denylist.denylist, j)
                    # if j not in token_allow_and_denylist.denylist:
                        # #print(f"[get_token_denylist] Debug: added token {j} ('{candidate_token_escaped}') to the denylist because it is equivalent to a string on the denylist ('{current_string_escaped}') even though the tokenizer converts that string to a different token")
                        # token_allow_and_denylist.denylist.append(j)
    return token_allow_and_denylist

def get_token_allow_and_deny_lists(tokenizer, string_list, device='cpu', additional_token_strings_case_sensitive = [], additional_token_strings_case_insensitive = [], additional_token_ids = None, filter_nonascii_tokens = False, filter_nonprintable_tokens = False, filter_special_tokens = False,filter_additional_special_tokens = False, filter_whitespace_tokens = False, token_regex = None):
    #print(f"[get_token_denylist] Debug: building token allowlist and denylist from string list '{string_list}'")
    result = TokenAllowAndDenyList()
    
    tokenizer_vocabulary_decoded = []
    for j in range(0, tokenizer.vocab_size):
        tokenizer_vocabulary_decoded.append(get_decoded_token(tokenizer, j))
    
    if filter_nonascii_tokens:
        result.denylist = add_values_to_list_if_not_already_present(result.denylist, get_nonascii_token_list(tokenizer, tokenizer_vocabulary_decoded))

    if filter_nonprintable_tokens:
        result.denylist = add_values_to_list_if_not_already_present(result.denylist, get_nonprintable_token_list(tokenizer, tokenizer_vocabulary_decoded))
    
    if token_regex is not None:
        denied_toks2 = get_nonmatching_token_list(tokenizer, tokenizer_vocabulary_decoded, token_regex)
        result.denylist = add_values_to_list_if_not_already_present(result.denylist, denied_toks2, ignore_none = True)
    
    if additional_token_ids is not None:
        result.denylist = add_values_to_list_if_not_already_present(result.denylist, additional_token_ids, ignore_none = True)
    
    # add special tokens if requested
    # Add the token ID directly to the list
    # But also decode it and add the decoded version to the input list to catch equivalents
    if filter_special_tokens:
        special_token_ids = [ tokenizer.bos_token_id,
                                    tokenizer.eos_token_id,
                                    tokenizer.pad_token_id,
                                    tokenizer.unk_token_id ]
    # add any additional special tokens defined in the tokenizer configuration
    # as well as their string equivalents
    if filter_additional_special_tokens:
        if hasattr(tokenizer, "added_tokens_decoder"):
            atd = tokenizer.added_tokens_decoder
            if atd is not None:
                if isinstance(atd, dict):
                    for added_token_id in atd.keys():
                        added_token_data = atd[added_token_id]
                        #if isinstance(added_token_data, dict):
                        if isinstance(added_token_data, AddedToken):
                            if hasattr(added_token_data, "special"):
                                if added_token_data.special:
                                    added_token_data_content = dir(added_token_data)
                                    special_token_ids = add_value_to_list_if_not_already_present(special_token_ids, added_token_id)
                                    #if added_token_id not in special_token_ids:
                                    #    special_token_ids.append(added_token_id)
                                    if hasattr(added_token_data, "content"):
                                        added_token_data_content = atd[added_token_id].content
                                        additional_token_strings_case_sensitive = add_value_to_list_if_not_already_present(additional_token_strings_case_sensitive, added_token_data_content)
                                    #print(f"[get_token_denylist] Debug: adding tokenizer special token ID {added_token_id} ('{added_token_data_content}') to the denylist")
                        else:
                            print(f"[get_token_denylist] Warning: the added_tokens_decoder property for the current tokenizer was in the expected format, but items within that property were not. Expected a hashtable/dictionary, got {type(added_token_data)} '{added_token_data}'")
                else:
                    print(f"[get_token_denylist] Warning: the added_tokens_decoder property for the current tokenizer was not in the expected format. Expected a hashtable/dictionary, got {type(atd)} '{atd}'")
            
        for special_token_id in special_token_ids:
            if special_token_id is not None:
                result.denylist = add_value_to_list_if_not_already_present(result.denylist, special_token_id)
                decoded_token = get_decoded_token(tokenizer, special_token_id)
                additional_token_strings_case_sensitive = add_value_to_list_if_not_already_present(additional_token_strings_case_sensitive, decoded_token, ignore_none = True)

    if filter_whitespace_tokens:
        for j in range(0, tokenizer.vocab_size):
            candidate_token = tokenizer_vocabulary_decoded[j]
            #candidate_token_escaped = get_escaped_string(candidate_token)
            if isinstance(candidate_token, type(None)):
                result.denylist = add_value_to_list_if_not_already_present(result.denylist, j)
            else:
                #candidate_token_escaped = get_escaped_string(candidate_token)
                # don't filter out tokens that are already empty strings, because some models (like Phi 3) use them to represent things like word breaks
                if candidate_token == "":
                    dummy = 1
                    #print(f"[get_token_denylist] Debug: did not add token {j} ('{candidate_token_escaped}') to the denylist because it was already an empty string.")
                else:
                    if candidate_token.strip() == "":
                        result.denylist = add_value_to_list_if_not_already_present(result.denylist, j)

    additional_token_strings_case_sensitive = add_values_to_list_if_not_already_present(additional_token_strings_case_sensitive, string_list, ignore_none = True)

    result = add_token_ids_from_strings(result, tokenizer, tokenizer_vocabulary_decoded, additional_token_strings_case_sensitive, case_sensitive = True)

    if len(additional_token_strings_case_insensitive) > 0:
        result = add_token_ids_from_strings(result, tokenizer, tokenizer_vocabulary_decoded, additional_token_strings_case_insensitive, case_sensitive = False)
           
    # finally, build the corresponding allowlist:
    for j in range(0, tokenizer.vocab_size):
        if j not in result.denylist:
            result.allowlist.append(j)
    #result.denylist.sort()
    #result.allowlist.sort()
    return result
