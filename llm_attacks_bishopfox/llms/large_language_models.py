#!/bin/env python3

import json
import os
import torch

from enum import IntFlag
#from enum import StrEnum
from enum import auto
from llm_attacks_bishopfox.json_serializable_object import JSONSerializableObject
from llm_attacks_bishopfox.util.util_functions import add_value_to_list_if_not_already_present
from llm_attacks_bishopfox.util.util_functions import add_values_to_list_if_not_already_present
from llm_attacks_bishopfox.util.util_functions import get_file_content
from llm_attacks_bishopfox.util.util_functions import torch_dtype_from_string

BUNDLED_LLM_LIST_FILE_NAME = "model_list.json"

class LargeLanguageModelException(Exception):
    pass

class BrokenHillModelSupportState(IntFlag):
    # Is the Torch configuration class for the model supported?
    # e.g. GemmaConfig, Phi3Config
    TORCH_CONFIGURATION_CLASS_SUPPORTED = auto()
    # Can the model be run for at least two iterations in Broken Hill without crashing?
    PASSES_SMOKE_TEST = auto()
    # Is there a chat template for the model that works reasonably well?
    HAS_KNOWN_CHAT_TEMPLATE = auto()
    # 

class LargeLanguageModelInfo(JSONSerializableObject):
    def __init__(self):
        self.model_name = None
        self.model_release = None
        self.model_family = None
        self.model_repository = None
        self.direct_developer_or_publisher = False
        self.model_path = None
        self.tokenizer_path = None
        self.peft_path = None
        self.template = None
        self.size = None
        self.data_type = None
        self.parameter_count = None
        self.safe_tensors = None
        self.custom_options = None        
        self.comment = None
    
    def get_parameter_count(self):
        if self.parameter_count is not None:
            return self.parameter_count
        if self.data_type is None or self.size is None:
            return None
        dtype = None
        try:
            dtype = torch_dtype_from_string(self.data_type)
        except Exception as e:
            dtype = None
        if dtype is None:
            return None
        bits_per_parameter = None            
        if dtype == torch.float16:
            bits_per_parameter = 16
        if dtype == torch.float32:
            bits_per_parameter = 32
        if dtype == torch.bfloat16:
            bits_per_parameter = 16
        if dtype == torch.float64:
            bits_per_parameter = 64
        if dtype == torch.complex64:
            bits_per_parameter = 64
        if dtype == torch.complex128:
            bits_per_parameter = 128
        if dtype == torch.bool:
            bits_per_parameter = 1
        if dtype == torch.int8:
            bits_per_parameter = 8
        if dtype == torch.uint8:
            bits_per_parameter = 8
        if dtype == torch.int16:
            bits_per_parameter = 16
        if dtype == torch.int32:
            bits_per_parameter = 32
        if dtype == torch.int64:
            bits_per_parameter = 64
        if bits_per_parameter is None:
            return None
        result = int(float(self.size) / (float(bits_per_parameter) / 8.0))
        return result
    
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
