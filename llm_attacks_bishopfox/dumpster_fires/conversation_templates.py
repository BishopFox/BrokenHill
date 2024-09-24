#!/bin/env python

import json

# IMPORTANT: 'fastchat' is in the PyPi package 'fschat', not 'fastchat'!
from fastchat.conversation import Conversation
from fastchat.conversation import SeparatorStyle

from llm_attacks_bishopfox.base.attack_manager import get_decoded_token
from llm_attacks_bishopfox.base.attack_manager import get_decoded_tokens
from llm_attacks_bishopfox.base.attack_manager import get_encoded_token
from llm_attacks_bishopfox.base.attack_manager import get_encoded_tokens
from llm_attacks_bishopfox.dumpster_fires.trash_fire_tokens import find_first_occurrence_of_array_in_array
from llm_attacks_bishopfox.dumpster_fires.trash_fire_tokens import find_last_occurrence_of_array_in_array
from llm_attacks_bishopfox.dumpster_fires.trash_fire_tokens import remove_empty_leading_and_trailing_tokens
from llm_attacks_bishopfox.util.util_functions import find_index_of_first_nonmatching_element
from llm_attacks_bishopfox.util.util_functions import remove_whitespace_and_nonprintable_characters

class SeparatorStyleConversionException(Exception):
    pass
    
class ConversationTemplateSerializationException(Exception):
    pass

# Just the fschat Llama-2 template and equivalents
def get_llama2_fschat_template_names():
    result = [ "llama-2" ]
    return result

# templates that have known imperfect output compared to apply_chat_template
def get_llama2_and_3_fschat_template_names():
    result = [ "llama2", "llama-2", "llama-3" ]
    return result

# Templates that produce identical output to apply_chat_template, except for probably-insignificant spurious tokens at the end
def get_stop_string_or_equivalent_is_different_template_names():
    result = [ "llama2", "llama-3", "qwen2 ", "smollm" ]
    return result

def get_apply_chat_template_ignored_template_names():
    result = [ "guanaco" ]
    return result

def fschat_separator_style_to_string(fschat_separator_style):
    result = None
    member_map = SeparatorStyle._member_map_
    for member_name in member_map.keys():
        if member_map[member_name] == fschat_separator_style:
            result = member_name
            break
    if isinstance(result, type(None)):
        raise SeparatorStyleConversionException(f"Could not convert '{fschat_separator_style}' to a string because it was not in the _member_map_ list for the SeparatorStyle enum ({member_map})")
    return result

def fschat_separator_style_from_string(fschat_separator_style_string):
    result = None
    member_map = SeparatorStyle._member_map_
    for member_name in member_map.keys():
        if member_name == fschat_separator_style_string:
            result = member_map[member_name]
            break
    if isinstance(result, type(None)):
        raise SeparatorStyleConversionException(f"Could not convert '{fschat_separator_style_string}' to a SeparatorStyle because it was not in the _member_map_ list for the SeparatorStyle enum ({member_map})")
    return result

def fschat_conversation_template_to_dict(fschat_conversation_template):
    result = {}
    result["name"] = fschat_conversation_template.name
    result["system_template"] = fschat_conversation_template.system_template
    result["system_message_vision"] = fschat_conversation_template.system_message_vision
    result["roles"] = fschat_conversation_template.roles
    result["messages"] = [[x, y] for x, y in fschat_conversation_template.messages]
    result["offset"] = fschat_conversation_template.offset
    result["sep_style"] = fschat_separator_style_to_string(fschat_conversation_template.sep_style)
    result["sep"] = fschat_conversation_template.sep
    result["sep2"] = fschat_conversation_template.sep2
    result["stop_str"] = fschat_conversation_template.stop_str
    result["stop_token_ids"] = fschat_conversation_template.stop_token_ids
    result["max_image_size_mb"] = fschat_conversation_template.max_image_size_mb
    return result
    
def fschat_conversation_template_to_json(fschat_conversation_template):
    result_dict = fschat_conversation_template_to_dict(fschat_conversation_template)    
    return json.dumps(result_dict)
    
def fschat_conversation_template_from_dict(fschat_conversation_template_dict):
    result = fastchat.conversation.Conversation
    try:
        result.name = fschat_conversation_template_dict["name"]
        result.system_template = fschat_conversation_template_dict["system_template"]
        result.system_message_vision = fschat_conversation_template_dict["system_message_vision"]
        result.roles = fschat_conversation_template_dict["roles"]
        for message in fschat_conversation_template_dict["messages"]:
            result.append_message(role = message[0], message = message[1])
        result.offset = fschat_conversation_template_dict["offset"]
        result.sep_style = fschat_separator_style_from_string(fschat_conversation_template_dict["sep_style"])
        result.sep = fschat_conversation_template_dict["name"]
        result.sep2 = fschat_conversation_template_dict["name"]
        result.stop_str = fschat_conversation_template_dict["name"]
        result.stop_token_ids = fschat_conversation_template_dict["name"]
        result.max_image_size_mb = fschat_conversation_template_dict["name"]
    except Exception as e:
        raise ConversationTemplateSerializationException(f"Couldn't create a conversation template from the data {fschat_conversation_template_dict}: {e}")
    return result
    
def fschat_conversation_template_from_json(fschat_conversation_template_json):
    result_dict = fschat_conversation_template_from_dict(fschat_conversation_template_json)    
    return json.loads(result_dict)


class TokenizerCannotApplyChatTemplateException(Exception):
    pass
    
# class TokenizerConfiguration:
    # def __init__(self):                
        # # don't just collect all of the tokenizer configuration properties, or reference the , because some of them have *enormous* lists of tokens
        
        # # the self-declared class name
        # self.tokenizer_class = None
        
        # # special tokens
        # self.bos_token = None
        # self.eos_token = None
        # self.pad_token = None
        # self.unk_token = None

        # # special token use
        # self.add_bos_token = None
        # self.add_eos_token = None

        # # if the auto_map property exists and has an AutoTokenizer list, it will be stored here
        # autotokenizer_names = []

        # # the chat template string, if one exists in the configuration        
        # self.chat_template = None
                
        # # other assorted properties that may exist
        # self.clean_up_tokenization_spaces = None
        # self.do_lower_case = None
        # self.padding_side = None
        # self.remove_space = None
        # self.spaces_between_special_tokens = None        
        # self.use_default_system_prompt = None

class TokenAndTokenIDListComparisonResult:
    def __init__(self):
        self.strings_match_exactly = False
        self.strings_match_without_whitespace_and_nonprintable = False
        self.strings_first_nonmatching_index = None
        self.token_ids_match = False
        self.decoded_tokens_match = False    
        self.token_ids_lengths_match = False
        self.decoded_tokens_lengths_match = False    
        self.token_ids_first_nonmatching_index = None
        self.decoded_tokens_first_nonmatching_index = None
    
    @staticmethod
    def compare_data(first_string, first_token_id_list, first_decoded_token_list, second_string, second_token_id_list, second_decoded_token_list):
        result = TokenAndTokenIDListComparisonResult()
        
        if first_string == second_string:
            result.strings_match_exactly = True
            result.strings_match_without_whitespace_and_nonprintable = True
        else:
            result.strings_match_exactly = False
            result.strings_first_nonmatching_index = find_index_of_first_nonmatching_element(first_string, second_string)
            first_string_stripped = remove_whitespace_and_nonprintable_characters(first_string)
            second_string_stripped = remove_whitespace_and_nonprintable_characters(second_string)
            if first_string_stripped == second_string_stripped:
                result.strings_match_without_whitespace_and_nonprintable = True
            else:
                result.strings_match_without_whitespace_and_nonprintable = False
        
        if len(first_token_id_list) == len(second_token_id_list):
            result.token_ids_lengths_match = True
        else:
            result.token_ids_lengths_match = False

        if len(first_decoded_token_list) == len(second_decoded_token_list):
            result.decoded_tokens_lengths_match = True
        else:
            result.decoded_tokens_lengths_match = False
        
        result.token_ids_first_nonmatching_index = find_index_of_first_nonmatching_element(first_token_id_list, second_token_id_list)
        if isinstance(result.token_ids_first_nonmatching_index, type(None)):
            result.token_ids_match = True
        else:
            result.token_ids_match = False
        
        result.decoded_tokens_first_nonmatching_index = find_index_of_first_nonmatching_element(first_decoded_token_list, second_decoded_token_list)
        if isinstance(result.decoded_tokens_first_nonmatching_index, type(None)):
            result.decoded_tokens_match = True
        else:
            result.decoded_tokens_match = False
        
        return result

class TokenizerConversationTemplateTestResult:
    def __init__(self):
        self.existing_fschat_template = None
        self.generated_fschat_template = None
        self.result_messages = []
        self.tokenizer_supports_apply_chat_template_method = False
        # specific tests
        self.tokenizer_chat_template_supports_messages_with_system_role = False
        self.existing_fschat_template_supports_system_message = False
        self.existing_fschat_template_supports_system_role = False
        self.existing_fschat_template_handles_system_message_and_system_role_identically = False
        self.existing_fschat_template_and_tokenizer_template_system_handling_matches = False
        self.template_comparison_result = None
        
class ConversationTemplateTester:
    def __init__(self, adversarial_content_manager, model):
        self.adversarial_content_manager = adversarial_content_manager
        self.model = model
    
    def test_templates(self, verbose = False):
        result = TokenizerConversationTemplateTestResult()
        
        result.existing_fschat_template = self.adversarial_content_manager.conv_template.copy()

        # The role names used in a chat template are completely arbitrary, but it seems that LLM developers have at least agreed to consistently support "user" and "assistant", even if some of them internally map "assistant" to another name (e.g. Gemma 2 maps it to "model").
        # Not all chat templates support use of the system role - some will throw an exception - so this approach will try both
        tokenizer_chat_template_system_role = "system"
        tokenizer_chat_template_user_role = "user"
        tokenizer_chat_template_assistant_role = "assistant"
        
        # Test system message support
        system_message = "BROKEN_HILL_TEST_SYSTEM_MESSAGE"
        # Create a two-step conversation between the user and the LLM to make validating the end of an LLM message versus the end of the conversation easier
        user_message_1 = "BROKEN_HILL_TEST_USER_MESSAGE_1"
        user_message_2 = "BROKEN_HILL_TEST_USER_MESSAGE_2"
        assistant_message_1 = "BROKEN_HILL_TEST_ASSISTANT_MESSAGE_1"
        assistant_message_2 = "BROKEN_HILL_TEST_ASSISTANT_MESSAGE_2"

        tokenizer_chat_template_messages_with_system = [
                {"role": tokenizer_chat_template_system_role, "content": system_message},
                {"role": tokenizer_chat_template_user_role, "content": user_message_1},
                {"role": tokenizer_chat_template_assistant_role, "content": assistant_message_1},
                {"role": tokenizer_chat_template_user_role, "content": user_message_2},
                {"role": tokenizer_chat_template_assistant_role, "content": assistant_message_2}
            ]
        
        tokenizer_chat_template_messages_without_system = tokenizer_chat_template_messages_with_system[1:]
        
        # Get the "canonical" version of an example conversation from the tokenizer's own chat template, if it supports doing so
        tokenizer_prompt_with_system_string = None
        tokenizer_prompt_with_system_token_ids = None
        tokenizer_prompt_with_system_decoded_tokens = None
        tokenizer_prompt_no_system_string = None
        tokenizer_prompt_no_system_token_ids = None
        tokenizer_prompt_no_system_decoded_tokens = None
        result.got_tokenizer_chat_template = False
        got_tokenizer_chat_template_with_system_as_string = False
        got_tokenizer_chat_template_with_no_system_as_string = False
        
        if not result.existing_fschat_template.name in get_apply_chat_template_ignored_template_names():
        
            apply_template_messages = []
            if result.existing_fschat_template.name in get_llama2_and_3_fschat_template_names():
                llama_template_warning = f"Warning: the fschat template '{result.existing_fschat_template.name}' may (or may not) produce less accurate conversation output than the canonical Llama-2 and Llama-3 conversation templates. "
                if result.existing_fschat_template.name in get_llama2_fschat_template_names():
                    llama_template_warning += f"If additional warnings regarding mismatches between the conversation template at the tokenizer's apply_chat_template output appear below, consider using the custom 'llama2' conversation template included with Broken Hill as a workaround until the underlying issue in fschat is resolved."
                result.result_messages.append(llama_template_warning)
            try:
                tokenizer_prompt_no_system_string = self.adversarial_content_manager.tokenizer.apply_chat_template(tokenizer_chat_template_messages_without_system, tokenize = False)
                got_tokenizer_chat_template_with_no_system_as_string = True
            except Exception as e:
                result.got_tokenizer_chat_template = False
                result.tokenizer_supports_apply_chat_template_method = False
                result.result_messages.append(f"Warning: the tokenizer does not appear to support an apply_chat_template method. Broken Hill will be unable to compare the current conversation template against the template provided by the developers of the model/tokenizer. If you observe unexpected or incorrect results during this test, verify that the chat template you've selected formats data correctly for the model. Testing this aspect of the tokenizer resulted in the exception '{e}'. The output of the apply_chat_template method for a conversation that did not include an initial system message was '{tokenizer_prompt_no_system_string}'")
            if got_tokenizer_chat_template_with_no_system_as_string:
                try:
                    tokenizer_prompt_no_system_token_ids = self.adversarial_content_manager.tokenizer.apply_chat_template(tokenizer_chat_template_messages_without_system, tokenize = True)
                    tokenizer_prompt_no_system_decoded_tokens = get_decoded_tokens(self.adversarial_content_manager.tokenizer, tokenizer_prompt_no_system_token_ids)
                    result.tokenizer_supports_apply_chat_template_method = True
                    result.got_tokenizer_chat_template = True
                except Exception as e:
                    result.got_tokenizer_chat_template = False
                    result.tokenizer_supports_apply_chat_template_method = False
                    result.result_messages.append(f"Error: an exception occurred when examining the output of the tokenizer's apply_chat_template when the conversation did not contain a system message: '{e}'. The string returned by the tokenizer was '{tokenizer_prompt_no_system_string}'")

            if result.tokenizer_supports_apply_chat_template_method:
                try:
                    tokenizer_prompt_with_system_string = self.adversarial_content_manager.tokenizer.apply_chat_template(tokenizer_chat_template_messages_with_system, tokenize = False)
                    got_tokenizer_chat_template_with_system_as_string = True
                except Exception as e:
                    result.tokenizer_chat_template_supports_messages_with_system_role = False
                    result.result_messages.append(f"Warning: the tokenizer's apply_chat_template method does not appear to support messages with the 'system' role. This may imply that the associated model does not support system messages. If your testing depends on setting a specific system message, you should validate that the model appears to incorporate your message. Testing this aspect of the tokenizer resulted in the exception '{e}'. The output of the apply_chat_template method for a conversation that included an initial system message was '{tokenizer_prompt_with_system_string}'")
                if got_tokenizer_chat_template_with_system_as_string:
                    try:
                        tokenizer_prompt_with_system_token_ids = self.adversarial_content_manager.tokenizer.apply_chat_template(tokenizer_chat_template_messages_with_system, tokenize = True)
                        tokenizer_prompt_with_system_decoded_tokens = get_decoded_tokens(self.adversarial_content_manager.tokenizer, tokenizer_prompt_with_system_token_ids)
                        result.tokenizer_chat_template_supports_messages_with_system_role = True
                        result.tokenizer_supports_apply_chat_template_method = True
                        result.got_tokenizer_chat_template = True
                    except Exception as e:
                        result.tokenizer_chat_template_supports_messages_with_system_role = False
                        result.result_messages.append(f"Error: an exception occurred when examining the output of the tokenizer's apply_chat_template when the conversation contained a system message: '{e}'. The string returned by the tokenizer was '{tokenizer_prompt_with_system_string}'")
            
        existing_fschat_template_working_copy_system_message = result.existing_fschat_template.copy()
        existing_fschat_template_working_copy_system_role = result.existing_fschat_template.copy()
        existing_fschat_template_working_copy_no_system = result.existing_fschat_template.copy()

        working_copy_list = [existing_fschat_template_working_copy_system_message, existing_fschat_template_working_copy_system_role, existing_fschat_template_working_copy_no_system]
        
        #existing_fschat_template_working_copy_system_message.set_system_message(system_message = system_message)
        # for templates like Gemma that need a <bos>-style initial string
        if isinstance(existing_fschat_template_working_copy_system_message.system_message, type(None)):
            existing_fschat_template_working_copy_system_message.system_message = system_message
        else:
            existing_fschat_template_working_copy_system_message.system_message += system_message
        existing_fschat_template_working_copy_system_role.append_message(tokenizer_chat_template_system_role, system_message)

        for conversation_template in working_copy_list:
            conversation_template.append_message(self.adversarial_content_manager.conv_template.roles[0], user_message_1)
            conversation_template.append_message(self.adversarial_content_manager.conv_template.roles[1], assistant_message_1)
            conversation_template.append_message(self.adversarial_content_manager.conv_template.roles[0], user_message_2)
            conversation_template.append_message(self.adversarial_content_manager.conv_template.roles[1], assistant_message_2)
        
        #encoded_conversation_template_prompt = self.adversarial_content_manager.tokenizer(conversation_template_prompt)
        #conversation_template_prompt_tokens = encoded_conversation_template_prompt.input_ids
        
        ef_template_prompt_with_system_message_string = existing_fschat_template_working_copy_system_message.get_prompt()
        ef_template_prompt_with_system_role_string = existing_fschat_template_working_copy_system_role.get_prompt()
        ef_template_prompt_with_no_system_string = existing_fschat_template_working_copy_no_system.get_prompt()
        
        # append the stop string, if any
        if not isinstance(existing_fschat_template_working_copy_system_message.stop_str, type(None)):
            ef_template_prompt_with_system_message_string += existing_fschat_template_working_copy_system_message.stop_str
        if not isinstance(existing_fschat_template_working_copy_system_role.stop_str, type(None)):
            ef_template_prompt_with_system_role_string += existing_fschat_template_working_copy_system_role.stop_str
        if not isinstance(existing_fschat_template_working_copy_no_system.stop_str, type(None)):
            ef_template_prompt_with_no_system_string += existing_fschat_template_working_copy_no_system.stop_str
        
        ef_template_prompt_with_system_message_token_ids = self.adversarial_content_manager.tokenizer.encode( ef_template_prompt_with_system_message_string)
        ef_template_prompt_with_system_role_token_ids = self.adversarial_content_manager.tokenizer.encode(ef_template_prompt_with_system_role_string)
        ef_template_prompt_with_no_system_token_ids = self.adversarial_content_manager.tokenizer.encode(ef_template_prompt_with_no_system_string)

        ef_template_prompt_with_system_message_decoded_tokens = get_decoded_tokens(self.adversarial_content_manager.tokenizer, ef_template_prompt_with_system_message_token_ids)
        ef_template_prompt_with_system_role_decoded_tokens = get_decoded_tokens(self.adversarial_content_manager.tokenizer, ef_template_prompt_with_system_role_token_ids)
        ef_template_prompt_with_no_system_decoded_tokens = get_decoded_tokens(self.adversarial_content_manager.tokenizer, ef_template_prompt_with_no_system_token_ids)        

        # do the templates *actually* support system messages, or are they ignored?
        if result.tokenizer_supports_apply_chat_template_method:
            if result.tokenizer_chat_template_supports_messages_with_system_role:
                if tokenizer_prompt_with_system_string == tokenizer_prompt_no_system_string:
                    result.tokenizer_chat_template_supports_messages_with_system_role = False
                    result.result_messages.append(f"Warning: the tokenizer's apply_chat_template does not incorporate system messages into its results. This may imply that the associated model does not support system messages. If your testing depends on setting a specific system message, you should validate that the model appears to incorporate your message.")
                else:
                    result.tokenizer_chat_template_supports_messages_with_system_role = True
        
        if ef_template_prompt_with_system_message_string == ef_template_prompt_with_system_role_string:
            result.existing_fschat_template_handles_system_message_and_system_role_identically = True
        else:
            result.existing_fschat_template_handles_system_message_and_system_role_identically = False
        
        if ef_template_prompt_with_system_message_string == ef_template_prompt_with_no_system_string:
            result.existing_fschat_template_supports_system_message = False
            result.result_messages.append(f"Warning: the conversation template '{result.existing_fschat_template.name}' does not incorporate system messages into its prompts. This may cause inaccurate or unexpected results if your use case depends upon testing with a system message. The template resulted in the following output whether or not a system message was specified: '{ef_template_prompt_with_system_message_string}'")            
        else:
            result.existing_fschat_template_supports_system_message = True

        if ef_template_prompt_with_system_role_string == ef_template_prompt_with_no_system_string:
            result.existing_fschat_template_supports_system_role = False
        else:
            result.existing_fschat_template_supports_system_role = True

        if result.existing_fschat_template_supports_system_role and not result.existing_fschat_template_supports_system_message:
            result.result_messages.append(f"Warning: the conversation template '{result.existing_fschat_template.name}' does not incorporate system messages into its prompts when the system_message property is set, but *does* incorporate them when specified as conversation entries with the 'system' role. Broken Hill does not currently support passing system messages in this way, as this behaviour should be very unusual. This may cause inaccurate or unexpected results if your use case depends upon testing with a system message. Please contact the Broken Hill developers with information about the model and tokenizer that generated this message.")

        # TKTK: once Broken Hill can handle both types of system message handling, update this next few statements to handle whichever one it's using
        ef_template_supports_system = result.existing_fschat_template_supports_system_message
        
        ef_template_prompt_string = ef_template_prompt_with_system_message_string
        ef_template_prompt_token_ids = ef_template_prompt_with_system_message_token_ids
        ef_template_prompt_decoded_tokens = ef_template_prompt_with_system_message_decoded_tokens
        tokenizer_prompt_string = tokenizer_prompt_with_system_string
        tokenizer_prompt_token_ids = tokenizer_prompt_with_system_token_ids
        tokenizer_prompt_with_decoded_tokens = tokenizer_prompt_with_system_decoded_tokens
        test_strings = []
        
        if ef_template_supports_system and result.tokenizer_chat_template_supports_messages_with_system_role:
            test_strings.append(system_message)
        else:
            ef_template_prompt_string = ef_template_prompt_with_no_system_string
            ef_template_prompt_token_ids = ef_template_prompt_with_no_system_token_ids
            ef_template_prompt_decoded_tokens = ef_template_prompt_with_no_system_decoded_tokens
            tokenizer_prompt_string = tokenizer_prompt_no_system_string
            tokenizer_prompt_token_ids = tokenizer_prompt_no_system_token_ids
            tokenizer_prompt_with_decoded_tokens = tokenizer_prompt_no_system_decoded_tokens

        test_strings.append(user_message_1)
        test_strings.append(user_message_2)
        test_strings.append(assistant_message_1)
        test_strings.append(assistant_message_2)
        
        # only include full output if there are serious issues, like the templates don't match
        include_full_output_as_last_message = False
        
        # Test for the presence of each of the test strings somewhere in the output of the generated prompt
        missing_test_strings_fschat = []
        missing_test_strings_tokenizer = []
        for ts in test_strings:
            if ts not in ef_template_prompt_string and ts not in missing_test_strings_fschat:
                missing_test_strings_fschat.append(ts)
            if result.tokenizer_supports_apply_chat_template_method:
                if ts not in tokenizer_prompt_string and ts not in tokenizer_prompt_string:
                    missing_test_strings_tokenizer.append(ts)
        
        if len(missing_test_strings_fschat) > 0:
            result.result_messages.append(f"Error: the following test strings were not found in the example conversation generated using the conversation template '{result.existing_fschat_template.name}': {missing_test_strings_fschat}. This usually indicates a severe issue with the template or your configuration.")
        
        if len(missing_test_strings_tokenizer) > 0:
            result.result_messages.append(f"Warning: the following test strings were not found in the example conversation generated using the tokenizer's apply_chat_template method: {missing_test_strings_tokenizer}. This may affect the results of this test, because it implies that the tokenizer or model do not accept input in the format you've selected.")
        
        # TKTK: validate that the count of instances of each test string is appropriate
        
        # TKTK: split this out into a separate function and compare several different conversations, e.g.:
        #       * Only one message from user, no response
        #       * Only one user/model message/response
        #       * User/model/user, no second response
        #       * User/model/user/model
        #       * If the tokenizer's chat template supports system role messages, test all of the above with and without a system message
        # the remainder of the tests require the tokenizer's chat template output
        if result.tokenizer_supports_apply_chat_template_method:                
            result.template_comparison_result = TokenAndTokenIDListComparisonResult.compare_data(ef_template_prompt_string, 
                ef_template_prompt_token_ids, 
                ef_template_prompt_decoded_tokens, 
                tokenizer_prompt_string, 
                tokenizer_prompt_token_ids, 
                tokenizer_prompt_with_decoded_tokens)
            
            
            # temporary workaround for minor issue with Llama-2 template that can't be easily corrected without fschat code changes
            if result.existing_fschat_template.name in get_stop_string_or_equivalent_is_different_template_names():
                #print("[ConversationTemplateTester.test_templates] Debug: template name '{result.existing_fschat_template.name}' is in the list of template names with known non-identical endings that are currently displayed as minor warnings. Workaround logic may apply.")
                if not result.template_comparison_result.strings_match_exactly:
                    if len(ef_template_prompt_string) > len(tokenizer_prompt_string):
                        #print("[ConversationTemplateTester.test_templates] Debug: len(ef_template_prompt_string) > len(tokenizer_prompt_string)")
                        truncated_ef_template_prompt_string = ef_template_prompt_string[:len(tokenizer_prompt_string)]
                        if truncated_ef_template_prompt_string == tokenizer_prompt_string:                            
                            result.result_messages.append(f"Warning: the conversation template '{result.existing_fschat_template.name}' and the tokenizer did not generate identical output for a test conversation. This is due to minor issues with the conversation template that cannot be easily resolved without updates to the fschat library. These issues should not materially affect Broken Hill's results.")
                            ef_template_prompt_string = truncated_ef_template_prompt_string
                            ef_template_prompt_token_ids = self.adversarial_content_manager.tokenizer.encode(ef_template_prompt_string)
                            ef_template_prompt_decoded_tokens = get_decoded_tokens(self.adversarial_content_manager.tokenizer, ef_template_prompt_token_ids)
                            result.template_comparison_result = TokenAndTokenIDListComparisonResult.compare_data(ef_template_prompt_string, 
                                ef_template_prompt_token_ids, 
                                ef_template_prompt_decoded_tokens, 
                                tokenizer_prompt_string, 
                                tokenizer_prompt_token_ids, 
                                tokenizer_prompt_with_decoded_tokens)
                        #else:
                        #    print("[ConversationTemplateTester.test_templates] Debug: truncated_ef_template_prompt_string != tokenizer_prompt_string")
                    #else:
                    #    print("[ConversationTemplateTester.test_templates] Debug: len(ef_template_prompt_string) <= len(tokenizer_prompt_string)")
            #else:
            #    print("[ConversationTemplateTester.test_templates] Debug: template name '{result.existing_fschat_template.name}' is not in the list of Llama-2 template names.")
            
            # Test whether the generated prompts are character-for-character identical
            if not result.template_comparison_result.strings_match_exactly:
                include_full_output_as_last_message = True
                string_match_message = f"Warning: the conversation template '{result.existing_fschat_template.name}' and the tokenizer did not generate identical output for a test conversation. The first difference was at character {result.template_comparison_result.strings_first_nonmatching_index}. "                
                # If the generated prompts are not character-for-character identical, test whether they're identical when all whitespace and non-printable characters are removed.
                if result.template_comparison_result.strings_match_without_whitespace_and_nonprintable:
                    string_match_message += f"The strings only differed due to whitespace or nonprintable characters, and so the difference may not materially affect the results of this test, but investigating the reason for the discrepancy is recommended for best results."
                else:
                    string_match_message += f"The strings contained different printable characters, so the discrepancy is likely to affect the reliability of results obtained by Broken Hill. You should verify that the template you've selected is intended for use with this model, and possibly customize it."
                result.result_messages.append(string_match_message)        
            
            # TKTK: Use the test strings to parse out the user and assistant role names from both conversations

            # TKTK: Verify that the user and assistant role names match between the conversations
            
            # TKTK: Verify that the user and assistant role names appear the correct number of times in the conversations
            
            # TKTK: If one or more of the role names was missing, check for them minus whitespace/nonprintable in the generated prompts
            
            # TKTK: If either are still not found, check for them minus whitespace/nonprintable in the generated prompts minus whitespace/nonprintable
                
        if include_full_output_as_last_message:
            last_message  = f"The conversation template '{result.existing_fschat_template.name}' generated the test conversation as the following string:\n"
            last_message += f"'{ef_template_prompt_string}'\n"
            if result.tokenizer_supports_apply_chat_template_method:
                last_message += f"The tokenizer's apply_chat_template method generated the test conversation as the following string:\n"
                last_message += f"'{tokenizer_prompt_string}'\n"
            if verbose:
                last_message += f"The conversation template '{result.existing_fschat_template.name}' output was tokenized to the following token IDs:\n"
                last_message += f"{ef_template_prompt_token_ids}\n"
                if result.tokenizer_supports_apply_chat_template_method:
                    last_message += f"The tokenizer's apply_chat_template method output was tokenized to the following token IDs:\n"
                    last_message += f"{tokenizer_prompt_token_ids}\n"
                last_message += f"The conversation template '{result.existing_fschat_template.name}' output token IDs were decoded to the following strings:\n"
                last_message += f"{ef_template_prompt_decoded_tokens}\n"
                if result.tokenizer_supports_apply_chat_template_method:
                    last_message += f"The tokenizer's apply_chat_template method output token IDs were decoded to the following strings:\n"
                    last_message += f"{tokenizer_prompt_with_decoded_tokens}\n"
            result.result_messages.append(last_message)

        # TKTK: If the templates are not identical, generate an fschat template definition that closely matches the one obtained from the tokenizer
        
        # TKTK: If the templates are not identical, try to find any existing fschat templates that match the generated definition in case the operator is unaware of an existing template 

        return result