#!/bin/env python

import logging
import re
import sys
import torch

from llm_attacks_bishopfox.json_serializable_object import JSONSerializableObject
from llm_attacks_bishopfox.util.util_functions import add_value_to_list_if_not_already_present
from llm_attacks_bishopfox.util.util_functions import add_values_to_list_if_not_already_present
from llm_attacks_bishopfox.util.util_functions import append_single_or_list_members
from llm_attacks_bishopfox.util.util_functions import find_first_occurrence_of_array_in_array
from llm_attacks_bishopfox.util.util_functions import find_index_of_first_nonmatching_element
from llm_attacks_bishopfox.util.util_functions import find_last_occurrence_of_array_in_array
from llm_attacks_bishopfox.util.util_functions import get_escaped_string
from llm_attacks_bishopfox.util.util_functions import remove_whitespace_and_nonprintable_characters

from tokenizers import AddedToken

logger = logging.getLogger(__name__)

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

def get_decoded_token(attack_state, token):
    result = None
    if attack_state.log_manager.get_lowest_log_level() <= logging.DEBUG:
        logger.debug(f"decoding token '{token}'")
    if isinstance(token, type(None)):
        logger.warning(f"a null token ID was passed to this function. This usually indicates a bug.")
        return None
    token_to_decode = token
    # workaround for models like Gemma that need all tokens to be in the form of a list
    wrap_in_list = False
    if not isinstance(token, list) and not isinstance(token, torch.Tensor):
        wrap_in_list = True
    if wrap_in_list:
        token_to_decode = [ token ]
        if attack_state.log_manager.get_lowest_log_level() <= logging.DEBUG:
            logger.debug(f"converted '{token}' to '{token_to_decode}'")
    #result = attack_state.tokenizer.decode(token_to_decode, skip_special_tokens=False)
    try:
        #result = attack_state.tokenizer.decode(token_to_decode, skip_special_tokens=True)
        result = attack_state.tokenizer.decode(token_to_decode, skip_special_tokens=False)
    except Exception as e:
        logger.error(f"Error decoding token {token_to_decode}: {e}")
        result = None
    if attack_state.log_manager.get_lowest_log_level() <= logging.DEBUG:
        logger.debug(f"decoded token '{token}' to '{result}'")
    return result

def get_decoded_tokens(attack_state, tokens, recursively_process_arrays = False):
    if attack_state.log_manager.get_lowest_log_level() <= logging.DEBUG:
        logger.debug(f"decoding tokens '{tokens}'")
    decoded_tokens = []
    token_list = tokens
    if isinstance(tokens, torch.Tensor):
        token_list = tokens.tolist()
    if isinstance(token_list, list):
        for tn in range(0, len(token_list)):
            if recursively_process_arrays:
                dt = get_decoded_tokens(attack_state, token_list[tn])
            else:
                dt = get_decoded_token(attack_state, token_list[tn])
            decoded_tokens.append(dt)
    else:
        dt = get_decoded_token(attack_state, tokens)
        decoded_tokens.append(dt)
    if attack_state.log_manager.get_lowest_log_level() <= logging.DEBUG:
        logger.debug(f"decoded tokens '{tokens}' to '{decoded_tokens}'")
    return decoded_tokens

def get_encoded_token(attack_state, token, exterminate_all_cowboy_nonsense = False):
    if attack_state.log_manager.get_lowest_log_level() <= logging.DEBUG:
        logger.debug(f"Encoding token '{token}'")
    result = None
    try:
        # If skip_special_tokens=True is enabled here, some(? all?) tokenizers will log a warning message about it.
        # Some will even error out! Unbelievable. Why write a compatibility layer and then glue incompatible options for different models on top of it?
        #result = attack_state.tokenizer.encode(token, skip_special_tokens=True)
        result = attack_state.tokenizer.encode(token)
        result = None
        if exterminate_all_cowboy_nonsense:
            result = encode_string_for_real_without_any_cowboy_funny_business(attack_state, token)
        else:
            result = attack_state.tokenizer.encode(token)
        if isinstance(result, type(None)):
            logger.warning(f"the tokenizer returned None when asked to encode the token '{token}'. This usually indicates a bug.")
    except Exception as e:
        logger.error(f"Error encoding token {token}: {e}")
    return result

def get_encoded_tokens(attack_state, tokens, exterminate_all_cowboy_nonsense = False):
    encoded_tokens = []
    for tn in range(0, len(tokens)):
        et = get_encoded_token(attack_state, tokens[tn], exterminate_all_cowboy_nonsense = exterminate_all_cowboy_nonsense)
        encoded_tokens.append(et)
    return encoded_tokens

# Gets the array of tokens that represent JUST A STRING, *without* any blazing trash fire tokens included, FOR REAL
def encode_string_for_real_without_any_cowboy_funny_business(attack_state, string):
    if string is None:
        return None
    if string == "":
        return []
    string_encoded = attack_state.tokenizer.encode(string)
    if attack_state.log_manager.get_lowest_log_level() <= logging.DEBUG:
        logger.debug("string_encoded = {string_encoded}")
    # First, strip any leading dumpster inferno content
    # make a single-character string that definitely does not start with the same character as the input string
    not_the_same_string_at_all = "A"
    if string[0] == "A":
        not_the_same_string_at_all = "B"
    not_the_same_string_at_all_encoded = attack_state.tokenizer.encode(not_the_same_string_at_all)
    if attack_state.log_manager.get_lowest_log_level() <= logging.DEBUG:
        logger.debug("not_the_same_string_at_all_encoded = {not_the_same_string_at_all_encoded}")
    start_index = find_index_of_first_nonmatching_element(string_encoded, not_the_same_string_at_all_encoded, log_manager = attack_state.log_manager)
    # Second, check for any ever-burning beacons of waste at the end of the result
    # make a string that is the same as the input, but has more characters at the end
    string_with_chaff = f"{string} 1987"
    string_with_chaff_encoded = attack_state.tokenizer.encode(string_with_chaff)
    if attack_state.log_manager.get_lowest_log_level() <= logging.DEBUG:
        logger.debug("string_with_chaff_encoded = {string_with_chaff_encoded}")
    stop_index = find_index_of_first_nonmatching_element(string_encoded, string_with_chaff_encoded, log_manager = attack_state.log_manager)
    result = string_encoded[start_index:stop_index]
    if attack_state.log_manager.get_lowest_log_level() <= logging.DEBUG:
        logger.debug("input = '{string}', result = {result}")
    return result

# # Gets the decoded string version of a set of token IDs, *without* any extra refuse conflagration tokens included, FOR REAL
# def decode_token_ids_do_not_even_think_about_reaching_for_those_schofields_charlie_prince(attack_state, token_ids):
    # if token_ids is None:
        # return None
    # if len(token_ids) == 0:
        # return ""
    # decoded_string = attack_state.tokenizer.decode(generated_prompt_token_ids)
    # if attack_state.log_manager.get_lowest_log_level() <= logging.DEBUG:
        # logger.debug(f"string_encoded = {string_encoded}")
    # # First, strip any leading dumpster inferno content
    # # make a single-character string that definitely does not start with the same character as the input string
    # not_the_same_string_at_all = "A"
    # if string[0] == "A":
        # not_the_same_string_at_all = "B"
    # not_the_same_string_at_all_encoded = attack_state.tokenizer.encode(not_the_same_string_at_all)
    # if attack_state.log_manager.get_lowest_log_level() <= logging.DEBUG:
        # logger.debug(f"not_the_same_string_at_all_encoded = {not_the_same_string_at_all_encoded}")
    # start_index = find_index_of_first_nonmatching_element(string_encoded, not_the_same_string_at_all_encoded, log_manager = self.attack_state.log_manager)
    # # Second, check for any ever-burning beacons of waste at the end of the result
    # # make a string that is the same as the input, but has more characters at the end
    # string_with_chaff = f"{string} 1987"
    # string_with_chaff_encoded = attack_state.tokenizer.encode(string_with_chaff)
    # if attack_state.log_manager.get_lowest_log_level() <= logging.DEBUG:
        # logger.debug(f"String_with_chaff_encoded = {string_with_chaff_encoded}")
    # stop_index = find_index_of_first_nonmatching_element(string_encoded, string_with_chaff_encoded, log_manager = self.attack_state.log_manager)
    # result = string_encoded[start_index:stop_index]
    # if attack_state.log_manager.get_lowest_log_level() <= logging.DEBUG:
        # logger.debug(f"Input = '{string}', result = {result}")
    # return result

# so many spaghetti code special cases to handle madness like 'This is a special token, but you can't treat it as 100% trash fire because it's also a sentinel that the parser has to look for. Also, sometimes it's more than one token! But not always!'
def is_conversation_role_token(attack_state, conversation_template, token):
    if not isinstance(token, type(None)):
        if token.strip() != "":
            for c_role in conversation_template.roles:
                if attack_state.log_manager.get_lowest_log_level() <= logging.DEBUG:
                    logger.debug(f"Searching for '{token}' in conversation role '{c_role}'")
                if token in c_role:
                    if attack_state.log_manager.get_lowest_log_level() <= logging.DEBUG:
                        logger.debug(f"'{token}' is in conversation role '{c_role}'")
                    return True
                token_minus_whitespace = remove_whitespace_and_nonprintable_characters(token)
                c_role_minus_whitespace = remove_whitespace_and_nonprintable_characters(c_role)
                if token in c_role:
                    if attack_state.log_manager.get_lowest_log_level() <= logging.DEBUG:
                        logger.debug(f"'{token_minus_whitespace}' is in conversation role '{c_role_minus_whitespace}'")
                    return True
                if c_role in token:
                    if attack_state.log_manager.get_lowest_log_level() <= logging.DEBUG:
                        logger.debug(f"Conversation role '{c_role_minus_whitespace}' is in token '{token_minus_whitespace}'")
                    return True
    return False
        
def is_disastrous_dumpster_fire_token(attack_state, trash_fire_tokens, conversation_template, token, decoded_token):
    token_is_a_pile_of_garbage_why_is_this_not_standardized_yet_you_ml_cowboys = False
    escaped_token = get_escaped_string(decoded_token)
    if token in trash_fire_tokens.token_ids:
        if attack_state.log_manager.get_lowest_log_level() <= logging.DEBUG:
            logger.debug(f"marked token '{escaped_token}' (id {token}) as a flaming dumpster floating down the river because the token ID was in the list of trash fire tokens.")
        token_is_a_pile_of_garbage_why_is_this_not_standardized_yet_you_ml_cowboys = True
    if not token_is_a_pile_of_garbage_why_is_this_not_standardized_yet_you_ml_cowboys:
        if decoded_token in trash_fire_tokens.input_strings:
            if attack_state.log_manager.get_lowest_log_level() <= logging.DEBUG:
                logger.debug(f"marked token '{escaped_token}' (id {token}) as a flaming dumpster floating down the river because the token was in the list of input strings that generated the list of trash fire tokens.")
            token_is_a_pile_of_garbage_why_is_this_not_standardized_yet_you_ml_cowboys = True
    if not token_is_a_pile_of_garbage_why_is_this_not_standardized_yet_you_ml_cowboys:
        for ctv in [decoded_token, decoded_token.strip()]:
            if ctv == "":
                if attack_state.log_manager.get_lowest_log_level() <= logging.DEBUG:
                    logger.debug(f"marked token '{escaped_token}' (id {token}) as a flaming dumpster floating down the river because it was empty or contained only whitespace.")
                token_is_a_pile_of_garbage_why_is_this_not_standardized_yet_you_ml_cowboys = True
                break
            if ctv in trash_fire_tokens.decoded_tokens:
                if attack_state.log_manager.get_lowest_log_level() <= logging.DEBUG:
                    logger.debug(f"marked token '{escaped_token}' (id {token}) as a flaming dumpster floating down the river because the decoded token was in the list of decoded trash fire tokens.")
                token_is_a_pile_of_garbage_why_is_this_not_standardized_yet_you_ml_cowboys = True
                break
            if ctv in trash_fire_tokens.input_strings:
                if attack_state.log_manager.get_lowest_log_level() <= logging.DEBUG:
                    logger.debug(f"marked token '{escaped_token}' (id {token}) as a flaming dumpster floating down the river because the decoded token was in the list of input strings used to generate the list of trash fire tokens.")
                token_is_a_pile_of_garbage_why_is_this_not_standardized_yet_you_ml_cowboys = True
                break
    
    # but wait! we can't exclude conversation role tokens! That would make parsing the output much harder!
    if conversation_template is not None:
        if token_is_a_pile_of_garbage_why_is_this_not_standardized_yet_you_ml_cowboys:
            if is_conversation_role_token(attack_state, conversation_template, decoded_token.strip()):
                if attack_state.log_manager.get_lowest_log_level() <= logging.DEBUG:
                    logger.debug(f"marked token '{escaped_token}' (id {token}) as not being a flaming dumpster floating down the river because the decoded token was in the list of tokens that indicate conversation role changes for the current conversation template, even though it is still a flaming dumpster floating down the river.")
                token_is_a_pile_of_garbage_why_is_this_not_standardized_yet_you_ml_cowboys = False
    
    if not token_is_a_pile_of_garbage_why_is_this_not_standardized_yet_you_ml_cowboys:
        if attack_state.log_manager.get_lowest_log_level() <= logging.DEBUG:
            logger.debug(f"token '{escaped_token}' (id {token}) does not appear to be a garbage inferno.")
    
    return token_is_a_pile_of_garbage_why_is_this_not_standardized_yet_you_ml_cowboys

def remove_empty_and_trash_fire_leading_and_trailing_tokens(attack_state, trash_fire_tokens, 
        token_array, 
        decoded_token_array, 
        conversation_template = None, 
        strip_decoded_tokens = False, 
        remove_empty_leading_and_trailing_tokens = True, 
        remove_leading_and_trailing_trash_fire_tokens = True,
        check_whitespace_tokens_for_trash_fire = False):
    if attack_state.log_manager.get_lowest_log_level() <= logging.DEBUG:
        logger.debug(f"token_array = {token_array}, decoded_token_array = {decoded_token_array}")
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
        if attack_state.log_manager.get_lowest_log_level() <= logging.DEBUG:
            logger.debug(f"checking token '{decoded_token_array[i]}', id {token_array[i]}")
        if remove_empty_leading_and_trailing_tokens:
            if decoded_token_temp == "":
                if attack_state.log_manager.get_lowest_log_level() <= logging.DEBUG:
                    logger.debug(f"token '{decoded_token_array[i]}' is whitespace or empty")
                is_skippable_token = True
        if not is_skippable_token:
            if remove_leading_and_trailing_trash_fire_tokens:
                if check_whitespace_tokens_for_trash_fire or decoded_token_array[i].strip() != "":
                    if is_disastrous_dumpster_fire_token(attack_state, trash_fire_tokens, conversation_template, token_array[i], decoded_token_array[i]):
                        if attack_state.log_manager.get_lowest_log_level() <= logging.DEBUG:
                            logger.debug(f"token '{decoded_token_array[i]}' is a blazing trash bin bringing light to the darkness of the running waters as they make their way to the sea.")
                        is_skippable_token = True
        if not is_skippable_token:
            if attack_state.log_manager.get_lowest_log_level() <= logging.DEBUG:
                logger.debug(f"token '{decoded_token_array[i]}' is not skippable")
            break
        first_non_empty_token += 1
    
    for i in range(len_token_array - 1, -1, -1):
        decoded_token_temp = decoded_token_array[i]
        if strip_decoded_tokens:
            decoded_token_temp = decoded_token_temp.strip()
            decoded_token_array[i] = decoded_token_temp
        is_skippable_token = False
        if attack_state.log_manager.get_lowest_log_level() <= logging.DEBUG:
            logger.debug(f"checking token '{decoded_token_array[i]}', id {token_array[i]}")
        if remove_empty_leading_and_trailing_tokens:
            if decoded_token_temp == "":
                if attack_state.log_manager.get_lowest_log_level() <= logging.DEBUG:
                    logger.debug(f"token '{decoded_token_array[i]}' is whitespace or empty")
                is_skippable_token = True
        if not is_skippable_token:
            if remove_leading_and_trailing_trash_fire_tokens:
                if check_whitespace_tokens_for_trash_fire or decoded_token_array[i].strip() != "":
                    if is_disastrous_dumpster_fire_token(attack_state, trash_fire_tokens, conversation_template, token_array[i], decoded_token_array[i]):
                        if attack_state.log_manager.get_lowest_log_level() <= logging.DEBUG:
                            logger.debug(f"token '{decoded_token_array[i]}' is a blazing trash bin bringing light to the darkness of the running waters as they make their way to the sea.")
                        is_skippable_token = True
        if not is_skippable_token:
            if attack_state.log_manager.get_lowest_log_level() <= logging.DEBUG:
                logger.debug(f"token '{decoded_token_array[i]}' is not skippable")
            break
        last_non_empty_token -= 1
    
    actual_last_non_empty_token = last_non_empty_token + 1
    result_token_array = token_array[first_non_empty_token:actual_last_non_empty_token]
    result_decoded_token_array = decoded_token_array[first_non_empty_token:actual_last_non_empty_token]
    
    if attack_state.log_manager.get_lowest_log_level() <= logging.DEBUG:
        logger.debug(f"token_array = '{token_array}', result_token_array = '{result_token_array}', decoded_token_array = '{decoded_token_array}', result_decoded_token_array = '{result_decoded_token_array}'")
    return result_token_array, result_decoded_token_array

# Third, walk step-by-step through the two sets of decoded tokens.
# Make a string out of every non-whitespace character in the list to be searched for ("the first list").
# e.g. if the first list is [ "I", "'", "m", " just", " a", " cow", "boy", " living", " in", " a", "cowboy", " day", "!" ],
# the string would be "I'mjustacowboylivinginacowboyday!".
# Start at the beginning of the list to be searched ("the second list"). Proceed forward by one token at each step.
# Begin building a string using the current token, minus any whitespace.
# Continue adding the characters of the following tokens (minus any whitespace) until either a character that doesn't match the first string at the same position is found, *or* the second string reaches the same length as the first string.
# In the second case, use the index of the current starting token as the start index part of the return value.
# Use (the index + 1) of the token that is currently being added to the second string as the stop index of the return value.
# I implore you once again, LLM researchers, please, stop the madness.
# Who is number one?! You are, number two.
def get_slice_for_token_array_within_token_array_avoid_robot_beachball_sentry(attack_state, decoded_token_list_to_search_for, decoded_token_list_to_search_within, search_start_index = None, search_end_index = None, find_last = False):
    search_string = ""    
    regex_whitespace = re.compile(r'\s')
    result_start_index = None
    result_stop_index = None
    step_size = 1
    range_start = 0
    range_end = len(decoded_token_list_to_search_within)
    compared_strings = []
    if find_last:
        step_size = -1
        range_start = len(decoded_token_list_to_search_within) - 1
        range_end = -1
        if search_start_index is not None:
            range_end = search_start_index - 1
        if search_end_index is not None:
            range_start = search_end_index - 1
    else:
        if search_start_index is not None:
            range_start = search_start_index
        if search_end_index is not None:
            range_end = search_end_index
        
    for i in range(0, len(decoded_token_list_to_search_for)):
        search_string += regex_whitespace.sub('', decoded_token_list_to_search_for[i])

    if attack_state.log_manager.get_lowest_log_level() <= logging.DEBUG:
        logger.debug(f"searching for '{search_string}' in {decoded_token_list_to_search_within}, from index {range_start} to index {range_end}, step {step_size}")
    for i in range(range_start, range_end, step_size):
        within_string = ""
        result_start_index = i
        do_continue = False
        for j in range(i, len(decoded_token_list_to_search_within)):
            within_string += regex_whitespace.sub('', decoded_token_list_to_search_within[j])
            result_stop_index = j + 1
            if len(within_string) >= len(search_string):
                if attack_state.log_manager.get_lowest_log_level() <= logging.DEBUG:
                    logger.debug(f"stopping concatenation at '{within_string}' because its length was greater than or equal to the length of '{search_string}'.")
                break
            current_substring = search_string[0:len(within_string)]
            # skip checking the rest of the current subsection if the material collected so far doesn't match
            if current_substring != within_string:
                if attack_state.log_manager.get_lowest_log_level() <= logging.DEBUG:
                    logger.debug(f"ending comparison of i = {i}, j = {j} early because '{current_substring}' != '{within_string}'")
                do_continue = True
                break
        compared_strings.append(within_string)
        if do_continue:
            continue
        # determine if the resulting string matches the first string
        if len(within_string) >= len(search_string):
            current_substring = within_string[0:len(search_string)]
            if current_substring == search_string:
                result = slice(result_start_index, result_stop_index)
                if attack_state.log_manager.get_lowest_log_level() <= logging.DEBUG:
                    logger.debug(f"result = '{result}'")
                return result
            else:
                if attack_state.log_manager.get_lowest_log_level() <= logging.DEBUG:
                    logger.debug(f"Compared '{within_string}' subset, '{current_substring}' != '{search_string}'.")
    
    raise TrashFireTokenException(f"Could not find {decoded_token_list_to_search_for} (represented as '{search_string}') in {decoded_token_list_to_search_within} (represented as a single string with no whitespace). Compared the search string against the following strings: {compared_strings}.")

# This actually returns a slice that describes the start and end indices in the token array
def find_first_index_of_token(attack_state, trash_fire_tokens, string_to_search_for, tokens, decoded_tokens, start_index = 0, stop_index = None, conversation_template = None, strip_leading_and_trailing_tokens = False, remove_empty_leading_and_trailing_tokens = True, remove_leading_and_trailing_trash_fire_tokens = True):
    return find_index_of_token(attack_state, trash_fire_tokens, string_to_search_for, tokens, decoded_tokens, start_index = start_index, stop_index = stop_index, conversation_template = conversation_template, find_last = False, strip_leading_and_trailing_tokens = strip_leading_and_trailing_tokens, remove_empty_leading_and_trailing_tokens = remove_empty_leading_and_trailing_tokens, remove_leading_and_trailing_trash_fire_tokens = remove_leading_and_trailing_trash_fire_tokens)

# This actually returns a slice that describes the start and end indices in the token array
def find_last_index_of_token(attack_state, trash_fire_tokens, string_to_search_for, tokens, decoded_tokens, start_index = 0, stop_index = None, conversation_template = None, strip_leading_and_trailing_tokens = False, remove_empty_leading_and_trailing_tokens = True, remove_leading_and_trailing_trash_fire_tokens = True):
    return find_index_of_token(attack_state, trash_fire_tokens, string_to_search_for, tokens, decoded_tokens, start_index = start_index, stop_index = stop_index, conversation_template = conversation_template, find_last = True, strip_leading_and_trailing_tokens = strip_leading_and_trailing_tokens, remove_empty_leading_and_trailing_tokens = remove_empty_leading_and_trailing_tokens, remove_leading_and_trailing_trash_fire_tokens = remove_leading_and_trailing_trash_fire_tokens)

# This actually returns a slice that describes the start and end indices in the token array
def find_index_of_token(attack_state, trash_fire_tokens, string_to_search_for, tokens, decoded_tokens, start_index = 0, stop_index = None, conversation_template = None, find_last = False, strip_leading_and_trailing_tokens = False, remove_empty_leading_and_trailing_tokens = True, remove_leading_and_trailing_trash_fire_tokens = True):  
    if string_to_search_for == "":
        raise TrashFireTokenException(f"Error: cannot search for empty string '{string_to_search_for}' in tokens = '{tokens}'")
    # decoded_tokens = get_decoded_tokens(attack_state, tokens)
    # if len(decoded_tokens) < 1:
        # raise TrashFireTokenException(f"Error: got zero-length array '{decoded_tokens}' for tokens = '{tokens}'")
    if attack_state.log_manager.get_lowest_log_level() <= logging.DEBUG:
        logger.debug(f"decoded_tokens = '{decoded_tokens}' for tokens = '{tokens}'")
    #string_tokens = encode_string_for_real_without_any_cowboy_funny_business(attack_state, string_to_search_for)
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
        sv_encoded = encode_string_for_real_without_any_cowboy_funny_business(attack_state, sv)
        string_token_variations.append(sv_encoded)
    
    failure_messages = []
    
    # various ways of finding strings encoded to token form
    # because there are a lot of weird variations on this concept depending on the LLM
    for string_token_set_num in range(0, len(string_token_variations)):
        string_tokens = string_token_variations[string_token_set_num]
        if len(string_tokens) < 1:
            raise TrashFireTokenException(f"Error: got zero-length array '{string_tokens}' when re-encoding tokens '{tokens}'")
        
        if attack_state.log_manager.get_lowest_log_level() <= logging.DEBUG:
            logger.debug(f"string_tokens = '{string_tokens}' for string '{string_to_search_for}'")        
        #string_to_search_for_array = string_to_search_for.split(" ")
        current_string_to_search_for = string_variations[string_token_set_num]
        string_to_search_for_array = current_string_to_search_for.split(" ")
        if len(string_to_search_for_array) < 1:
            raise TrashFireTokenException(f"Error: got zero-length array '{string_to_search_for_array}' when splitting '{current_string_to_search_for}' into words")
        decoded_string_tokens = get_decoded_tokens(attack_state, string_tokens)
        if len(decoded_string_tokens) < 1:
            raise TrashFireTokenException(f"Error: got zero-length array '{decoded_string_tokens}' when decoding string_tokens '{string_tokens}'")

        #trimmed_string_tokens = string_tokens
        #trimmed_decoded_string_tokens = decoded_string_tokens

        trimmed_string_tokens, trimmed_decoded_string_tokens = remove_empty_and_trash_fire_leading_and_trailing_tokens(attack_state, trash_fire_tokens, 
            string_tokens, 
            decoded_string_tokens, 
            conversation_template = conversation_template, 
            strip_decoded_tokens = strip_leading_and_trailing_tokens, 
            remove_empty_leading_and_trailing_tokens = remove_empty_leading_and_trailing_tokens, 
            remove_leading_and_trailing_trash_fire_tokens = remove_leading_and_trailing_trash_fire_tokens)

        if len(trimmed_decoded_string_tokens) < 1:
            raise TrashFireTokenException(f"Error: got zero-length array '{trimmed_decoded_string_tokens}' after removing ignored tokens from '{decoded_string_tokens}'")
        if len(trimmed_string_tokens) < 1:
            raise TrashFireTokenException(f"Error: got zero-length array '{trimmed_string_tokens}' after removing ignored tokens from '{string_tokens}'")

        string_tokens = trimmed_string_tokens
        decoded_string_tokens = trimmed_decoded_string_tokens
        
        # First, look for the encoded version of the string in the encoded version of the prompt.
        # One would think this is all one would need to do for this function, but one would be wrong.
        if attack_state.log_manager.get_lowest_log_level() <= logging.DEBUG:
            logger.debug(f"searching for '{current_string_to_search_for}' (tokenized as '{decoded_string_tokens}') in '{decoded_tokens}' from index {start_index} to {stop_index}")
        result_start = None
        if find_last:
            result_start = find_last_occurrence_of_array_in_array(string_tokens, tokens, start_index = start_index, stop_index = stop_index, log_manager = attack_state.log_manager)
        else:
            result_start = find_first_occurrence_of_array_in_array(string_tokens, tokens, start_index = start_index, stop_index = stop_index, log_manager = attack_state.log_manager)
        result_stop = None
        is_failure = False
        if isinstance(result_start, type(None)):
            # Second, look for the the first match for any of the following combinations:
            #   * In the full prompt decoded tokens after being stripped of whitespace:
            #       * The decoded version of the encoded string tokens
            #       * The decoded, whitespace-stripped version of the encoded string tokens
            #   * In the full prompt decoded tokens without being stripped of whitespace:
            #       * The decoded version of the encoded string tokens
            #       * The decoded, whitespace-stripped version of the encoded string tokens
            # try to find cases where tokens have spaces on either side or not at all
            decoded_tokens_processed_1 = []
            decoded_tokens_processed_2 = []
            for i in range(0, len(decoded_tokens)):
                processed_token = decoded_tokens[i].strip()
                decoded_tokens_processed_1.append(processed_token)
                decoded_tokens_processed_2.append(decoded_tokens[i])
            for search_array in [ decoded_string_tokens, string_to_search_for_array ]:
                for in_array in [ decoded_tokens_processed_1, decoded_tokens_processed_2 ]:
                    if isinstance(result_start, type(None)):
                        if find_last:
                            result_start = find_last_occurrence_of_array_in_array(search_array, in_array, start_index=start_index, stop_index=stop_index, log_manager = attack_state.log_manager)
                        else:
                            result_start = find_first_occurrence_of_array_in_array(search_array, in_array, start_index=start_index, stop_index=stop_index, log_manager = attack_state.log_manager)
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
                if attack_state.log_manager.get_lowest_log_level() <= logging.DEBUG:
                    logger.debug(f"could not find '{current_string_to_search_for}' (tokenized as '{decoded_string_tokens}') in '{decoded_tokens}', but found the close approximation '{string_to_search_for_array}' in '{decoded_tokens_processed_1}' or '{decoded_tokens_processed_2}' and will use that position instead. This may be due to using a buggy LLM that considers e.g. 'Human' and ' Human' different tokens, but uses both values for similar purposes internally.")
        else:
            result_stop = result_start + len(string_tokens)
        if isinstance(result_start, type(None)):
            # One might think "surely, no more ways of searching for the tokens are necessary after five previous variations!"
            # But one would still be mistaken. I'm looking at you, Phi-3!
            # The specific case that prompted this is that Phi-3 would encode the word "Wonderful!" as 'Wonder', 'ful', '!' when it occurred at the beginning of a string, but 'W', 'onder', 'ful', '!' if it occurred after the token "<|assistant|>".
            # This is done only when the other two checks fail because it's more expensive.
            try:
                result = get_slice_for_token_array_within_token_array_avoid_robot_beachball_sentry(attack_state, decoded_string_tokens, decoded_tokens, search_start_index = start_index, search_end_index = stop_index, find_last = find_last)
                return result
            except TrashFireTokenException as tfte:
                is_failure = True
                failure_messages.append(f"{tfte}")        
        if not is_failure:
            result = slice(result_start, result_stop)
            if attack_state.log_manager.get_lowest_log_level() <= logging.DEBUG:
                logger.debug(f"result = '{result}'")
            return result
    exception_message = ""
    for i in range(0, len(failure_messages)):
        exception_message += f"{failure_messages[i]}\n\n"
    raise Exception(exception_message)

# # dynamically determine the last token in a set of tokens 
# # that get_prompt should consider
# # like '</s>', '<s>', '\n', '###', or ' '
# def find_last_non_garbage_token(attack_state, conversation_template, tokens, decoded_tokens, trash_fire_tokens, start_index = 0, stop_index = None):
    # result = None
    # range_end = len(tokens)
    # if not isinstance(stop_index, type(None)):
        # range_end = stop_index
    # for i in range(start_index, range_end):
        # token_is_a_pile_of_garbage_why_is_this_not_standardized_yet_you_ml_cowboys = is_disastrous_dumpster_fire_token(attack_state, trash_fire_tokens, conversation_template, tokens[i], decoded_tokens[i])
        # if not token_is_a_pile_of_garbage_why_is_this_not_standardized_yet_you_ml_cowboys:
            # result = i
    # if isinstance(result, type(None)):
        # raise Exception(f"Could not find a token that wasn't an absolute dumpster fire in '{decoded_tokens}' from index {start_index} to {range_end}, please, stop the madness right now.")
    # #logger.debug(f"last non-garbage token in '{decoded_tokens}' from index {start_index} to {range_end} ('{decoded_tokens[start_index:range_end]}') is index {result}, '{decoded_tokens[result]}'")
    # return result

# def find_first_non_garbage_token(conversation_template, tokens, decoded_tokens, trash_fire_tokens, start_index = 0, stop_index = None):
    # result = None
    # range_end = len(tokens)
    # if not isinstance(stop_index, type(None)):
        # range_end = stop_index
    # for i in range(start_index, range_end):
        # token_is_a_pile_of_garbage_why_is_this_not_standardized_yet_you_ml_cowboys = is_disastrous_dumpster_fire_token(attack_state, trash_fire_tokens, conversation_template, tokens[i], decoded_tokens[i])
        # if not token_is_a_pile_of_garbage_why_is_this_not_standardized_yet_you_ml_cowboys:
            # #logger.debug(f"first non-garbage token in '{decoded_tokens}' from index {start_index} to {range_end} is index {i}, '{decoded_tokens[i]}'")
            # return i
    # if isinstance(result, type(None)):
        # raise Exception(f"Could not find a token that wasn't an absolute dumpster fire in '{decoded_tokens}' from index {start_index} to {range_end}, please, stop the madness right now.")
    # return result

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
    def get_meticulously_curated_trash_fire_token_collection(attack_state, conversation_template):
        result = TrashFireTokenCollection()
        
        result.input_strings = TrashFireTokenCollection.append_default_special_string_denylist([])
        # get tokenizer-specific trash
        # but be careful about trash that might be equivalent to an empty string or whitespace
        potential_trash = []
        potential_trash.append(attack_state.tokenizer.bos_token)
        potential_trash.append(attack_state.tokenizer.eos_token)
        potential_trash.append(attack_state.tokenizer.unk_token)
        potential_trash.append(attack_state.tokenizer.pad_token)
        additional_flaming_dumpster_ids = []
        additional_flaming_dumpster_ids.append(attack_state.tokenizer.bos_token_id)
        additional_flaming_dumpster_ids.append(attack_state.tokenizer.eos_token_id)
        additional_flaming_dumpster_ids.append(attack_state.tokenizer.unk_token_id)
        additional_flaming_dumpster_ids.append(attack_state.tokenizer.pad_token_id)
        
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
        
        allow_and_denylists = get_token_allow_and_deny_lists(attack_state, result.input_strings, device = 'cpu', filter_special_tokens = True, filter_additional_special_tokens = True, filter_whitespace_tokens = True, additional_token_ids = additional_flaming_dumpster_ids)
        result.token_ids = allow_and_denylists.denylist
        
        result.decoded_tokens = get_decoded_tokens(attack_state, result.token_ids)
        if attack_state.log_manager.get_lowest_log_level() <= logging.DEBUG:
            logger.debug(f"result.input_strings = '{result.input_strings}', result.token_ids = '{result.token_ids}', result.decoded_tokens = '{result.decoded_tokens}'")
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
                #logger.debug(f"excluding '{dt}' because it did not match the specified regular expression.")
                #if "#" in dt:
                #    logger.debug(f"excluding '{dt}' because it did not match the specified regular expression.")
            #else:
            #    if "#" in dt:
            #        logger.debug(f"not excluding '{dt}' because it matched the specified regular expression.")
    
    return nonmatching_tokens

def get_token_list_as_tensor(token_list, device='cpu'):    
    return torch.tensor(token_list, device=device)    

class TokenAllowAndDenyList(JSONSerializableObject):
    def __init__(self):
        self.allowlist = []
        self.denylist = []

    def to_dict(self):
        result = super(TokenAllowAndDenyList, self).properties_to_dict(self)
        return result
    
    @staticmethod
    def from_dict(property_dict):
        result = TokenAllowAndDenyList()
        super(TokenAllowAndDenyList, result).set_properties_from_dict(result, property_dict)
        return result

    def to_json(self):
        return JSONSerializableObject.json_dumps(self.to_dict(), use_indent = False)
    
    def copy(self):
        return TokenAllowAndDenyList.from_dict(self.to_dict())
    
    @staticmethod
    def from_json(json_string):
        return TokenAllowAndDenyList.from_dict(json.loads(json_string))


def add_token_ids_from_strings(attack_state, token_allow_and_denylist, tokenizer_vocabulary_decoded, string_list, case_sensitive = True):
    for i in range(0, len(string_list)):
        current_string = string_list[i]

        current_string_escaped = get_escaped_string(current_string)
        denied_toks_original = get_encoded_token(attack_state, current_string)
        if attack_state.log_manager.get_lowest_log_level() <= logging.DEBUG:
            logger.debug(f"got token(s) '{denied_toks_original}' from string '{current_string_escaped}'")
        # If a given string was transformed into more than one token, ignore it
        
        if denied_toks_original is not None:
            if isinstance(denied_toks_original, list):
                if len(denied_toks_original) == 1:
                    if attack_state.log_manager.get_lowest_log_level() <= logging.DEBUG:
                        logger.debug(f"converting token '{denied_toks_original}' to a single value")
                    denied_toks_original = denied_toks_original[0]
                else:
                    if attack_state.log_manager.get_lowest_log_level() <= logging.DEBUG:
                        logger.debug(f"did not add tokens '{denied_toks_original}' to the denylist because a single string became multiple tokens")
                    denied_toks_original = None
        if denied_toks_original is not None:
            if attack_state.log_manager.get_lowest_log_level() <= logging.DEBUG:
                logger.debug(f"added token {denied_toks_original} to the denylist")
            token_allow_and_denylist.denylist = add_value_to_list_if_not_already_present(token_allow_and_denylist.denylist, denied_toks_original)
            # if denied_toks_original not in token_allow_and_denylist.denylist:
                # #logger.debug(f"added token {denied_toks_original} to the denylist")
                # token_allow_and_denylist.denylist.append(denied_toks_original)
        # also check to see if any tokens are equivalent to the string value when decoded, 
        # even if the encoder didn't return them
        for j in range(0, attack_state.tokenizer.vocab_size):
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
                        # #logger.debug(f"added token {j} ('{candidate_token_escaped}') to the denylist because it is equivalent to a string on the denylist ('{current_string_escaped}') even though the tokenizer converts that string to a different token")
                        # token_allow_and_denylist.denylist.append(j)
    return token_allow_and_denylist

# TKTK: Maybe refactor the trash fire function that calls this so the method signature can be simplified
def get_token_allow_and_deny_lists(attack_state, string_list, device='cpu', additional_token_strings_case_sensitive = [], additional_token_strings_case_insensitive = [], additional_token_ids = None, filter_nonascii_tokens = False, filter_nonprintable_tokens = False, filter_special_tokens = False,filter_additional_special_tokens = False, filter_whitespace_tokens = False, token_regex = None):
    if attack_state.log_manager.get_lowest_log_level() <= logging.DEBUG:
        logger.debug(f"building token allowlist and denylist from string list '{string_list}'")
    result = TokenAllowAndDenyList()
    
    tokenizer_vocabulary_decoded = []
    for j in range(0, attack_state.tokenizer.vocab_size):
        tokenizer_vocabulary_decoded.append(get_decoded_token(attack_state, j))
    
    if filter_nonascii_tokens:
        result.denylist = add_values_to_list_if_not_already_present(result.denylist, get_nonascii_token_list(attack_state.tokenizer, tokenizer_vocabulary_decoded))

    if filter_nonprintable_tokens:
        result.denylist = add_values_to_list_if_not_already_present(result.denylist, get_nonprintable_token_list(attack_state.tokenizer, tokenizer_vocabulary_decoded))
    
    if token_regex is not None:
        denied_toks2 = get_nonmatching_token_list(attack_state.tokenizer, tokenizer_vocabulary_decoded, token_regex)
        result.denylist = add_values_to_list_if_not_already_present(result.denylist, denied_toks2, ignore_none = True)
    
    if additional_token_ids is not None:
        result.denylist = add_values_to_list_if_not_already_present(result.denylist, additional_token_ids, ignore_none = True)
    
    # add special tokens if requested
    # Add the token ID directly to the list
    # But also decode it and add the decoded version to the input list to catch equivalents
    if filter_special_tokens:
        special_token_ids = [ attack_state.tokenizer.bos_token_id,
                                    attack_state.tokenizer.eos_token_id,
                                    attack_state.tokenizer.pad_token_id,
                                    attack_state.tokenizer.unk_token_id ]
    # add any additional special tokens defined in the tokenizer configuration
    # as well as their string equivalents
    if filter_additional_special_tokens:
        if hasattr(attack_state.tokenizer, "added_tokens_decoder"):
            atd = attack_state.tokenizer.added_tokens_decoder
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
                                    if attack_state.log_manager.get_lowest_log_level() <= logging.DEBUG:
                                        logger.debug(f"adding attack_state.tokenizer special token ID {added_token_id} ('{added_token_data_content}') to the denylist")
                        else:
                            logger.warning(f"the added_tokens_decoder property for the current tokenizer was in the expected format, but items within that property were not. Expected a hashtable/dictionary, got {type(added_token_data)} '{added_token_data}'")
                else:
                    logger.warning(f"the added_tokens_decoder property for the current tokenizer was not in the expected format. Expected a hashtable/dictionary, got {type(atd)} '{atd}'")
            
        for special_token_id in special_token_ids:
            if special_token_id is not None:
                result.denylist = add_value_to_list_if_not_already_present(result.denylist, special_token_id)
                decoded_token = get_decoded_token(attack_state, special_token_id)
                additional_token_strings_case_sensitive = add_value_to_list_if_not_already_present(additional_token_strings_case_sensitive, decoded_token, ignore_none = True)

    if filter_whitespace_tokens:
        for j in range(0, attack_state.tokenizer.vocab_size):
            candidate_token = tokenizer_vocabulary_decoded[j]
            if isinstance(candidate_token, type(None)):
                result.denylist = add_value_to_list_if_not_already_present(result.denylist, j)
            else:
                # don't filter out tokens that are already empty strings, because some models (like Phi 3) use them to represent things like word breaks
                if candidate_token == "":
                    if attack_state.log_manager.get_lowest_log_level() <= logging.DEBUG:
                        candidate_token_escaped = get_escaped_string(candidate_token)
                        logger.debug(f"did not add token {j} ('{candidate_token_escaped}') to the denylist because it was already an empty string.")
                else:
                    if candidate_token.strip() == "":
                        result.denylist = add_value_to_list_if_not_already_present(result.denylist, j)

    additional_token_strings_case_sensitive = add_values_to_list_if_not_already_present(additional_token_strings_case_sensitive, string_list, ignore_none = True)

    result = add_token_ids_from_strings(attack_state, result, tokenizer_vocabulary_decoded, additional_token_strings_case_sensitive, case_sensitive = True)

    if len(additional_token_strings_case_insensitive) > 0:
        result = add_token_ids_from_strings(attack_state, result, tokenizer_vocabulary_decoded, additional_token_strings_case_insensitive, case_sensitive = False)
           
    # finally, build the corresponding allowlist:
    for j in range(0, attack_state.tokenizer.vocab_size):
        if j not in result.denylist:
            result.allowlist.append(j)
    #result.denylist.sort()
    #result.allowlist.sort()
    return result
