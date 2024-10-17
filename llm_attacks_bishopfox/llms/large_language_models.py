#!/bin/env python3

import json
import os

from llm_attacks_bishopfox.json_serializable_object import JSONSerializableObject
from llm_attacks_bishopfox.util.util_functions import add_value_to_list_if_not_already_present
from llm_attacks_bishopfox.util.util_functions import add_values_to_list_if_not_already_present
from llm_attacks_bishopfox.util.util_functions import get_file_content

BUNDLED_LLM_LIST_FILE_NAME = "model_list.json"

class LargeLanguageModelException(Exception):
    pass

class LargeLanguageModelInfo(JSONSerializableObject):
    def __init__(self):
        self.model_name = None
        self.model_release = None
        self.model_family = None
        self.model_repository = None
        self.first_party_release = False
        self.model_path = None
        self.tokenizer_path = None
        self.peft_path = None
        self.template = None
        self.size = None
        self.parameter_count = None
        self.safe_tensors = None
        self.custom_options = None        
        self.comment = None
    
    def to_dict(self):
        result = super(LargeLanguageModelInfo, self).properties_to_dict(self)
        return result
    
    @staticmethod
    def from_dict(property_dict):
        result = LargeLanguageModelInfo()
        super(LargeLanguageModelInfo, result).set_properties_from_dict(result, property_dict)
        return result

    def to_json(self):
        return JSONSerializableObject.json_dumps(self.to_dict(), use_indent = False)
        
    @staticmethod
    def from_json(json_string):
        return LargeLanguageModelInfo.from_dict(json.loads(json_string))
    
    def copy(self):
        result = LargeLanguageModelInfo()
        return LargeLanguageModelInfo.set_properties_from_dict(result, self.to_dict())

class LargeLanguageModelInfoList(JSONSerializableObject):
    def __init__(self):
        self.entries = []
    
    def to_dict(self):
        result = super(LargeLanguageModelInfoList, self).properties_to_dict(self)
        return result
    
    @staticmethod
    def from_dict(property_dict):
        result = LargeLanguageModelInfoList()
        super(LargeLanguageModelInfoList, result).set_properties_from_dict(result, property_dict)
        if len(result.entries) > 0:
            deserialized_content = []
            for i in range(0, len(result.entries)):
                deserialized_content.append(LargeLanguageModelInfo.from_dict(result.entries[i]))
            result.entries = deserialized_content
        return result

    def to_json(self):
        return JSONSerializableObject.json_dumps(self.to_dict(), use_indent = False)
        
    @staticmethod
    def from_json(json_string):
        return LargeLanguageModelInfoList.from_dict(json.loads(json_string))
    
    def copy(self):
        result = LargeLanguageModelInfoList()
        return LargeLanguageModelInfoList.set_properties_from_dict(result, self.to_dict())
    
    @staticmethod
    def from_bundled_json_file():
        base_path = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(base_path, BUNDLED_LLM_LIST_FILE_NAME)
        file_content = get_file_content(file_path, failure_is_critical = True)
        if file_content is None:
            raise LargeLanguageModelException(f"Found no content in the file '{file_path}'")
        return LargeLanguageModelInfoList.from_json(file_content)
