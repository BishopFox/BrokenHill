#!/bin/env python3

import json
import logging
import os
import torch
import traceback

from enum import IntFlag
#from enum import StrEnum
from enum import auto
from llm_attacks_bishopfox.json_serializable_object import JSONSerializableObject
from llm_attacks_bishopfox.util.util_functions import add_value_to_list_if_not_already_present
from llm_attacks_bishopfox.util.util_functions import load_json_from_file
from llm_attacks_bishopfox.util.util_functions import torch_dtype_from_string
from llm_attacks_bishopfox.util.util_functions import torch_dtype_to_bit_count

logger = logging.getLogger(__name__)

BUNDLED_LLM_LIST_FILE_NAME = "model_list.json"

class LargeLanguageModelException(Exception):
    pass

class BrokenHillModelSupportState(IntFlag):
    # Is the Torch configuration class for the model supported?
    # e.g. GemmaConfig, Phi3Config
    TORCH_CONFIGURATION_CLASS_SUPPORTED = auto()
    # Can the model be run for at least two iterations in Broken Hill without crashing and generating valid output?
    PASSES_SMOKE_TEST = auto()
    # Is there a chat template for the model that works reasonably well?
    HAS_KNOWN_CHAT_TEMPLATE = auto()

def model_support_state_to_list(model_support_state):
    result = []
    if (model_support_state & BrokenHillModelSupportState.TORCH_CONFIGURATION_CLASS_SUPPORTED) == BrokenHillModelSupportState.TORCH_CONFIGURATION_CLASS_SUPPORTED:
        result.append(str(BrokenHillModelSupportState.TORCH_CONFIGURATION_CLASS_SUPPORTED))
    if (model_support_state & BrokenHillModelSupportState.PASSES_SMOKE_TEST) == BrokenHillModelSupportState.PASSES_SMOKE_TEST:
        result.append(str(BrokenHillModelSupportState.PASSES_SMOKE_TEST))
    if (model_support_state & BrokenHillModelSupportState.TORCH_CONFIGURATION_CLASS_SUPPORTED) == BrokenHillModelSupportState.TORCH_CONFIGURATION_CLASS_SUPPORTED:
        result.append(str(BrokenHillModelSupportState.TORCH_CONFIGURATION_CLASS_SUPPORTED))
    return result

def model_support_state_from_list(model_support_state_flag_list):
    result = 0
    if isinstance(model_support_state_flag_list, list):
        if str(BrokenHillModelSupportState.TORCH_CONFIGURATION_CLASS_SUPPORTED) in model_support_state_flag_list:
            result = result | BrokenHillModelSupportState.TORCH_CONFIGURATION_CLASS_SUPPORTED
        if str(BrokenHillModelSupportState.PASSES_SMOKE_TEST) in model_support_state_flag_list:
            result = result | BrokenHillModelSupportState.PASSES_SMOKE_TEST
        if str(BrokenHillModelSupportState.HAS_KNOWN_CHAT_TEMPLATE) in model_support_state_flag_list:
            result = result | BrokenHillModelSupportState.HAS_KNOWN_CHAT_TEMPLATE
    return result

class BrokenHillModelAlignmentInfo(IntFlag):
    # Does the model have one or more alignment/trained restrictions against generating certain types of output?
    MODEL_HAS_ALIGNMENT_RESTRICTIONS = auto()
    # Has bypass of at least one type of alignment/trained restriction been demonstrated using Broken Hill?
    BROKEN_HILL_HAS_DEFEATED_ALIGNMENT = auto()
    # Does the model generally follow instructions in the system prompt or template messages regarding generation of content?
    MODEL_GENERALLY_FOLLOWS_ADDITIONAL_RESTRICTIONS = auto()
    # Has bypass of system prompt/template message instructions been demonstrated using Broken Hill?
    BROKEN_HILL_HAS_DEFEATED_ADDITIONAL_RESTRICTIONS = auto()

def alignment_info_to_list(model_support_state):
    result = []
    if (model_support_state & BrokenHillModelAlignmentInfo.MODEL_HAS_ALIGNMENT_RESTRICTIONS) == BrokenHillModelAlignmentInfo.MODEL_HAS_ALIGNMENT_RESTRICTIONS:
        result.append(str(BrokenHillModelAlignmentInfo.MODEL_HAS_ALIGNMENT_RESTRICTIONS))
    if (model_support_state & BrokenHillModelAlignmentInfo.BROKEN_HILL_HAS_DEFEATED_ALIGNMENT) == BrokenHillModelAlignmentInfo.BROKEN_HILL_HAS_DEFEATED_ALIGNMENT:
        result.append(str(BrokenHillModelAlignmentInfo.BROKEN_HILL_HAS_DEFEATED_ALIGNMENT))
    if (model_support_state & BrokenHillModelAlignmentInfo.MODEL_GENERALLY_FOLLOWS_ADDITIONAL_RESTRICTIONS) == BrokenHillModelAlignmentInfo.MODEL_GENERALLY_FOLLOWS_ADDITIONAL_RESTRICTIONS:
        result.append(str(BrokenHillModelAlignmentInfo.MODEL_GENERALLY_FOLLOWS_ADDITIONAL_RESTRICTIONS))
    if (model_support_state & BrokenHillModelAlignmentInfo.BROKEN_HILL_HAS_DEFEATED_ADDITIONAL_RESTRICTIONS) == BrokenHillModelAlignmentInfo.BROKEN_HILL_HAS_DEFEATED_ADDITIONAL_RESTRICTIONS:
        result.append(str(BrokenHillModelAlignmentInfo.BROKEN_HILL_HAS_DEFEATED_ADDITIONAL_RESTRICTIONS))
    return result

def alignment_info_from_list(alignment_info_flag_list):
    result = 0
    if isinstance(alignment_info_flag_list, list):
        if str(BrokenHillModelAlignmentInfo.MODEL_HAS_ALIGNMENT_RESTRICTIONS) in alignment_info_flag_list:
            result = result | BrokenHillModelAlignmentInfo.MODEL_HAS_ALIGNMENT_RESTRICTIONS
        if str(BrokenHillModelAlignmentInfo.BROKEN_HILL_HAS_DEFEATED_ALIGNMENT) in alignment_info_flag_list:
            result = result | BrokenHillModelAlignmentInfo.BROKEN_HILL_HAS_DEFEATED_ALIGNMENT
        if str(BrokenHillModelAlignmentInfo.MODEL_GENERALLY_FOLLOWS_ADDITIONAL_RESTRICTIONS) in alignment_info_flag_list:
            result = result | BrokenHillModelAlignmentInfo.MODEL_GENERALLY_FOLLOWS_ADDITIONAL_RESTRICTIONS
        if str(BrokenHillModelAlignmentInfo.BROKEN_HILL_HAS_DEFEATED_ADDITIONAL_RESTRICTIONS) in alignment_info_flag_list:
            result = result | BrokenHillModelAlignmentInfo.BROKEN_HILL_HAS_DEFEATED_ADDITIONAL_RESTRICTIONS
    return result

def print_model_parameter_info(attack_state):
    model_calculated_bytes_in_memory = None
    model_as_is_calculated_bytes_in_memory = None
    current_memory_parameter_info_string = ""
    as_is_storage_memory_parameter_info_string = ""
    
    model_calculated_bytes_in_memory = attack_state.persistable.overall_result_data.model_parameter_info_collection.get_parameter_size_in_memory(attack_state.model.dtype)
    current_memory_parameter_info_string = f" Using the current data type '{attack_state.model.dtype}', the model data should occupy {model_calculated_bytes_in_memory} bytes of device memory."
        
    if attack_state.model_weight_storage_dtype is not None:
        model_as_is_calculated_bytes_in_memory = attack_state.persistable.overall_result_data.model_parameter_info_collection.get_parameter_size_in_memory(attack_state.model_weight_storage_dtype)
        as_is_storage_memory_parameter_info_string = f" Using the model's native data type '{attack_state.model_weight_storage_dtype}', the model data should occupy {model_as_is_calculated_bytes_in_memory} bytes of device memory." 
            
    model_parameter_info_string = f"The current model has {attack_state.persistable.overall_result_data.model_parameter_info_collection.total_parameter_count} total parameters in named groups."
    if attack_state.persistable.attack_params.verbose_model_parameter_info:
        model_parameter_info_string = f"{model_parameter_info_string} {attack_state.persistable.overall_result_data.model_parameter_info_collection.trainable_parameter_count} of the parameters are trainable, and {attack_state.persistable.overall_result_data.model_parameter_info_collection.nontrainable_parameter_count} of the parameters are not trainable.{current_memory_parameter_info_string}{as_is_storage_memory_parameter_info_string}"
        trainable_params = attack_state.persistable.overall_result_data.model_parameter_info_collection.get_trainable_parameters()
        nontrainable_params = attack_state.persistable.overall_result_data.model_parameter_info_collection.get_nontrainable_parameters()
        if len(trainable_params) > 0:
            info_substring = f"\nNamed parameter groups (trainable) and their parameter counts:\n"
            for i in range(0, len(trainable_params)):
                param = trainable_params[i]
                info_substring = f"{info_substring}\t{param.module_name}: {param.parameter_count}\n"
            model_parameter_info_string = f"{model_parameter_info_string}{info_substring}"                
        if len(nontrainable_params) > 0:
            info_substring = f"\nNamed parameter groups (non-trainable) and their parameter counts:\n"
            for i in range(0, len(nontrainable_params)):
                param = nontrainable_params[i]
                info_substring = f"{info_substring}\t{param.module_name}: {param.parameter_count}\n"
            model_parameter_info_string = f"{model_parameter_info_string}{info_substring}"
    else:
        model_parameter_info_string = f"{model_parameter_info_string}{current_memory_parameter_info_string}"
    
    logger.info(model_parameter_info_string)

class LargeLanguageModelParameterException(Exception):
    pass

class LargeLanguageModelParameterInfo(JSONSerializableObject):
    def __init__(self):
        self.module_name = None
        self.parameter_count = None
        self.is_trainable = None

    def to_dict(self):
        result = super(LargeLanguageModelParameterInfo, self).properties_to_dict(self)
        return result
    
    @staticmethod
    def from_dict(property_dict):
        result = LargeLanguageModelParameterInfo()
        super(LargeLanguageModelParameterInfo, result).set_properties_from_dict(result, property_dict)
        return result

    def to_json(self):
        return JSONSerializableObject.json_dumps(self.to_dict(), use_indent = False)
        
    @staticmethod
    def from_json(json_string):
        return LargeLanguageModelParameterInfo.from_dict(json.loads(json_string))
    
    def copy(self):
        result = LargeLanguageModelParameterInfo()
        return LargeLanguageModelParameterInfo.set_properties_from_dict(result, self.to_dict())

class LargeLanguageModelParameterInfoCollection(JSONSerializableObject):
    def __init__(self):
        self.parameters = {}
        self.total_parameter_count = None
        self.trainable_parameter_count = None
        self.nontrainable_parameter_count = None

    def get_total_parameter_count(self, only_trainable = False):
        result = 0
        for param_name in self.parameters.keys():
            param = self.parameters[param_name]
            if only_trainable:
                if not param.is_trainable:
                    continue
            result += param.parameter_count
        return result
    
    def set_parameter_counts(self):
        self.total_parameter_count = self.get_total_parameter_count(only_trainable = False)
        self.trainable_parameter_count = self.get_total_parameter_count(only_trainable = True)
        self.nontrainable_parameter_count = self.total_parameter_count - self.trainable_parameter_count

    def get_parameter_size_in_memory(self, dtype):
        result = None
        try:
            result = int(float(self.total_parameter_count) * float(torch_dtype_to_bit_count(dtype)) / 8.0)
        except Exception as e:
            raise LargeLanguageModelParameterException(f"Error calculating model memory use for parameter count {self.total_parameter_count} and data type {dtype}: {e}\n{traceback.format_exc()}\n")
        return result
    
    def get_trainable_parameters(self):
        result = []
        for k in self.parameters.keys():
            if self.parameters[k].is_trainable:
                result.append(self.parameters[k])
        return result
        
    def get_nontrainable_parameters(self):
        result = []
        for k in self.parameters.keys():
            if not self.parameters[k].is_trainable:
                result.append(self.parameters[k])
        return result

    def get_all_parameters(self):
        result = []
        for k in self.parameters.keys():
            result.append(self.parameters[k])
        return result
    
    # BEGIN: based in part on https://stackoverflow.com/questions/49201236/check-the-total-number-of-parameters-in-a-pytorch-model
    @staticmethod
    def get_model_parameter_info(model):
        result = {}
        for name, parameter in model.named_parameters():
            param_info = LargeLanguageModelParameterInfo()
            param_info.module_name = name
            param_info.is_trainable = parameter.requires_grad
            param_info.parameter_count = parameter.numel()
            result[name] = param_info
        return result
    # END: based in part on https://stackoverflow.com/questions/49201236/check-the-total-number-of-parameters-in-a-pytorch-model

    @staticmethod
    def from_loaded_model(model):
        result = LargeLanguageModelParameterInfoCollection()
        result.parameters = LargeLanguageModelParameterInfoCollection.get_model_parameter_info(model)
        result.set_parameter_counts()        
        return result

    def to_dict(self):
        result = super(LargeLanguageModelParameterInfoCollection, self).properties_to_dict(self)
        return result

    @staticmethod
    def apply_dict(existing_object, property_dict):
        super(LargeLanguageModelParameterInfoCollection, existing_object).set_properties_from_dict(existing_object, property_dict)
        if existing_object.parameters is not None:
            deserialized_content = {}
            for k in existing_object.parameters.keys():
                deserialized_content[k] = (LargeLanguageModelParameterInfo.from_dict(existing_object.parameters[k]))
            existing_object.parameters = deserialized_content            
        return existing_object
    
    @staticmethod
    def from_dict(property_dict):
        result = LargeLanguageModelParameterInfoCollection()
        return LargeLanguageModelParameterInfoCollection.apply_dict(result, property_dict)

    def to_json(self):
        return JSONSerializableObject.json_dumps(self.to_dict(), use_indent = False)
        
    @staticmethod
    def from_json(json_string):
        return LargeLanguageModelParameterInfoCollection.from_dict(json.loads(json_string))
    
    def copy(self):
        result = LargeLanguageModelParameterInfoCollection()
        return LargeLanguageModelParameterInfoCollection.set_properties_from_dict(result, self.to_dict())

    
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
        self.parameter_info_collection = None
        self.safe_tensors = None
        self.custom_options = None
        self.support_state = 0
        self.alignment_info = 0
        self.comment = None    
    
    def get_parameter_count(self):
        if self.parameter_count is not None:
            return self.parameter_count
        if self.parameter_info_collection is None:
            return None
        return self.parameter_info_collection.total_parameter_count
    
    def to_dict(self):
        result = super(LargeLanguageModelInfo, self).properties_to_dict(self)
        result["support_state"] = model_support_state_to_list(self.support_state)
        result["alignment_info"] = alignment_info_to_list(self.alignment_info)
        return result
    
    @staticmethod
    def from_dict(property_dict):
        result = LargeLanguageModelInfo()
        super(LargeLanguageModelInfo, result).set_properties_from_dict(result, property_dict)
        if result.parameter_info_collection is not None:
            result.parameter_info_collection = LargeLanguageModelParameterInfoCollection.from_dict(result.parameter_info_collection)
        if result.support_state is not None:
            result.support_state = model_support_state_from_list(result.support_state)
        if result.alignment_info is not None:
            result.alignment_info = alignment_info_from_list(result.alignment_info)
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
        file_content = None
        try:
            file_content = load_json_from_file(file_path, failure_is_critical = True)
        except Exception as e:
            raise LargeLanguageModelException(f"Could not load JSON data from the file '{file_path}': {e}\n{traceback.format_exc()}")
        if file_content is None:
            raise LargeLanguageModelException(f"Found no content in the file '{file_path}'")
        return LargeLanguageModelInfoList.from_dict(file_content)
