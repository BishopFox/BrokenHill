#!/bin/env python

import json
import logging
import os

from llm_attacks_bishopfox.json_serializable_object import JSONSerializableObject
from llm_attacks_bishopfox.util.util_functions import add_value_to_list_if_not_already_present
from llm_attacks_bishopfox.util.util_functions import add_values_to_list_if_not_already_present
from llm_attacks_bishopfox.util.util_functions import load_json_from_file

logger = logging.getLogger(__name__)

BUNDLED_LANGUAGE_DEFINITION_FILE_NAME = "human_language_names.json"

class HumanLanguageException(Exception):
    pass

class HumanLanguage(JSONSerializableObject):
    def __init__(self):
        self.english_names = []
        self.native_names = []
        self.ietf_tag = None
        self.use_english_names_in_allow_or_denylists = True
        self.use_native_names_in_allow_or_denylists = True
        self.comment = ""

    def to_dict(self):
        result = super(HumanLanguage, self).properties_to_dict(self)
        return result
    
    @staticmethod
    def from_dict(property_dict):
        result = HumanLanguage()
        super(HumanLanguage, result).set_properties_from_dict(result, property_dict)
        return result
        
    def to_json(self):
        return JSONSerializableObject.json_dumps(self.to_dict())
            
    @staticmethod
    def from_json(json_string):
        return HumanLanguage.from_dict(json.loads(json_string))


class HumanLanguageManager(JSONSerializableObject):
    def __init__(self):
        self.comment = ""
        self.data = []

    def to_dict(self):
        result = super(HumanLanguageManager, self).properties_to_dict(self)
        return result
    
    # return a list of all words that represent language names
    def get_language_names(self, ietf_tag_to_exclude = None):
        result = []
        found_allowed_ietf_tag = False
        for i in range(0, len(self.data)):
            current_language = self.data[i]
            include_language = True
            if ietf_tag_to_exclude is not None:
                if ietf_tag_to_exclude.lower() == current_language.ietf_tag.lower():
                    include_language = False
                    found_allowed_ietf_tag = True
            if include_language:
                language_names = []               
                if current_language.use_english_names_in_allow_or_denylists:
                    language_names = add_values_to_list_if_not_already_present(language_names, current_language.english_names)
                if current_language.use_native_names_in_allow_or_denylists:
                    language_names = add_values_to_list_if_not_already_present(language_names, current_language.native_names)
                for j in range(0, len(language_names)):
                    current_name = language_names[j]
                    include_name = True
                    # ignore names with spaces for now
                    if " " in current_name:
                        include_name = False
                    if include_name:
                        result = add_value_to_list_if_not_already_present(result, current_name)
        if ietf_tag_to_exclude is not None:
            if not found_allowed_ietf_tag:
                raise HumanLanguageException(f"Could not find the IETF language tag '{ietf_tag_to_exclude}' in the list of known values.")
        return result
    
    # return a list of all known language tags
    def get_ietf_tags(self):
        result = {}
        for i in range(0, len(self.data)):
            result[self.data[i].ietf_tag] = self.data[i].english_names[0]
        sorted_keys = []
        for k in result.keys():
            sorted_keys.append(k)
        sorted_keys.sort()
        return sorted_keys, result
    
        
    @staticmethod
    def from_dict(property_dict):
        result = HumanLanguageManager()
        super(HumanLanguageManager, result).set_properties_from_dict(result, property_dict)
        if len(result.data) > 0:
            deserialized_content = []
            for i in range(0, len(result.data)):
                deserialized_content.append(HumanLanguage.from_dict(result.data[i]))
            result.data = deserialized_content
        return result
        
    def to_json(self):
        return JSONSerializableObject.json_dumps(self.to_dict())
            
    @staticmethod
    def from_json(json_string):
        return HumanLanguageManager.from_dict(json.loads(json_string))

    @staticmethod
    def from_bundled_json_file():
        base_path = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(base_path, BUNDLED_LANGUAGE_DEFINITION_FILE_NAME)
        file_content = load_json_from_file(file_path, failure_is_critical = True)
        if file_content is None:
            raise HumanLanguageException(f"Found no content in the file '{file_path}'")
        return HumanLanguageManager.from_dict(file_content)
