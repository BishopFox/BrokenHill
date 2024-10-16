#!/bin/env python3

import json
import os
import re

from llm_attacks_bishopfox.json_serializable_object import JSONSerializableObject
from llm_attacks_bishopfox.llms.large_language_models import LargeLanguageModelInfo
from llm_attacks_bishopfox.util.util_functions import add_value_to_list_if_not_already_present
from llm_attacks_bishopfox.util.util_functions import add_values_to_list_if_not_already_present
from llm_attacks_bishopfox.util.util_functions import get_time_string

class BrokenHillTestException(Exception):
    pass

class BrokenHillTestParams(JSONSerializableObject):
    def __init__(self):
        self.test_name = None
        self.base_llm_path = None
        self.python_path = "bin/python"
        self.python_params = [ '-u' ]
        self.broken_hill_path = "./BrokenHill/brokenhill.py"
        self.device = None
        self.output_file_directory = None
        self.output_file_base_name = None
        self.model_path = None
        self.tokenizer_path = None
        self.peft_path = None
        self.template = None
        self.auto_target = None
        self.base_prompt = None
        self.target_output = None
        self.initial_adversarial_string = None        
        self.max_iterations = 2
        self.max_new_tokens_final = 256
        self.custom_options = [ '--exclude-nonascii-tokens', '--exclude-nonprintable-tokens', '--exclude-special-tokens', '--exclude-additional-special-tokens', '--exclude-newline-tokens' ]
        # default: thirty minutes
        self.process_timeout = 1800
    
    def set_from_model_info(self, model_info):
        self.model_path = model_info.model_path
        self.tokenizer_path = model_info.tokenizer_path
        self.peft_path = model_info.peft_path
        self.template = model_info.template
        self.custom_options = add_values_to_list_if_not_already_present(self.custom_options, model_info.custom_options)
    
    def set_output_file_base_name(self):
        result = f"{self.test_name}-{self.model_path}-{get_time_string()}"
        result = re.sub(r'[\\/:]', '-', result)
        result = re.sub(r'[ ]', '_', result)
        self.output_file_base_name = result
    
    def get_command_array(self):
        result = [ self.python_path ]
        for i in range(0, len(self.python_params)):
            result.append(self.python_params[i])
        result.append(self.broken_hill_path)
        if self.device is not None and self.device.strip() != "":
            result.append('--device')
            result.append(self.device)
        if self.model_path is not None and self.model_path.strip() != "":
            result.append('--model')
            result.append(os.path.join(self.base_llm_path, self.model_path))
        if self.tokenizer_path is not None and self.tokenizer_path.strip() != "":
            result.append('--tokenizer')
            result.append(self.tokenizer_path)
        if self.peft_path is not None and self.peft_path.strip() != "":
            result.append('--peft-adapter')
            result.append(self.peft_path)
        if self.template is not None and self.template.strip() != "":
            result.append('--template')
            result.append(self.template)
        if self.auto_target is not None and self.auto_target.strip() != "":
            result.append('--auto-target')
            result.append(self.auto_target)
        if self.base_prompt is not None and self.base_prompt.strip() != "":
            result.append('--base-prompt')
            result.append(self.base_prompt)
        if self.target_output is not None and self.target_output.strip() != "":
            result.append('--target-output')
            result.append(self.target_output)
        if self.initial_adversarial_string is not None:
            result.append('--initial-adversarial-string')
            result.append(self.initial_adversarial_string)
        if self.max_iterations is not None:
            result.append('--max-iterations')
            result.append(f"{self.max_iterations}")
        if self.max_new_tokens_final is not None:
            result.append('--max-new-tokens-final')
            result.append(f"{self.max_new_tokens_final}")        
        if self.output_file_directory is not None and self.output_file_base_name is not None and self.output_file_directory.strip() != "" and self.output_file_base_name.strip() != "":
            result.append('--json-output-file')
            result.append(os.path.join(self.output_file_directory, f"{self.output_file_base_name}-results.json"))        
        
        for i in range(0, len(self.custom_options)):
            result.append(self.custom_options[i])
        
        return result
    
    
    def to_dict(self):
        result = super(BrokenHillTestParams, self).properties_to_dict(self)
        return result
    
    @staticmethod
    def from_dict(property_dict):
        result = BrokenHillTestParams()
        super(BrokenHillTestParams, result).set_properties_from_dict(result, property_dict)
        return result

    def to_json(self):
        return JSONSerializableObject.json_dumps(self.to_dict(), use_indent = False)
        
    @staticmethod
    def from_json(json_string):
        return BrokenHillTestParams.from_dict(json.loads(json_string))

    def copy(self):
        result = BrokenHillTestParams()
        return BrokenHillTestParams.set_properties_from_dict(result, self.to_dict())