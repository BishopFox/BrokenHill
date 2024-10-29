#!/bin/env python

import copy
import json
import re
import torch

from llm_attacks_bishopfox.util.util_functions import tensor_to_dict

class JSONSerializationException(Exception):
    pass

JSON_DUMP_DEFAULT_INDENT = 4

class JSONSerializableObject:    
    @staticmethod
    def json_dumps(object_to_dump, use_indent = True):
        if use_indent:
            return json.dumps(object_to_dump, indent = JSON_DUMP_DEFAULT_INDENT)
        return json.dumps(object_to_dump)

    def to_dict(self):
        return JSONSerializableObject.properties_to_dict(self)
        
    def to_json(self):
        return JSONSerializableObject.json_dumps(self.to_dict())

    def copy(self):
        result = JSONSerializableObject()
        return JSONSerializableObject.set_properties_from_dict(result, self.to_dict())
    
    # # Example implementations of the corresponding from_dict and from_json methods
    # # These are implemented in subclasses because there's no advantage I can see to making faux-abstract classes in Pythong

    # @staticmethod
    # def from_json(json_string):
        # return LargeLanguageModelParameterInfo.from_dict(json.loads(json_string))

    # Most straightforward example of from_dict:

    # @staticmethod
    # def from_dict(property_dict):
        # result = LargeLanguageModelParameterInfo()
        # super(LargeLanguageModelParameterInfo, result).set_properties_from_dict(result, property_dict)
        # return result

    # from_dict where one of the class's properties is a subclass of JSONSerializableObject:
    
    # @staticmethod
    # def from_dict(property_dict):
        # result = LargeLanguageModelInfo()
        # super(LargeLanguageModelInfo, result).set_properties_from_dict(result, property_dict)
        # if result.parameter_info_collection is not None:
            # result.parameter_info_collection = LargeLanguageModelParameterInfoCollection.from_dict(result.parameter_info_collection)
        # if result.support_state is not None:
            # result.support_state = model_support_state_from_list(result.support_state)
        # if result.alignment_info is not None:
            # result.alignment_info = alignment_info_from_list(result.alignment_info)
        # return result

    # from_dict where one of the class's properties is an array of another subclass of JSONSerializableObject:

    # @staticmethod
    # def from_dict(property_dict):
        # result = LargeLanguageModelParameterInfoCollection()
        # super(LargeLanguageModelParameterInfoCollection, result).set_properties_from_dict(result, property_dict)
        # if result.parameters is not None:
            # if len(result.parameters) > 0:
                # deserialized_content = []
                # for i in range(0, len(result.parameters)):
                    # deserialized_content.append(LargeLanguageModelParameterInfo.from_dict(result.parameters[i]))
                # result.parameters = deserialized_content
        # return result

    # TKTK: split from_dict into apply_dict and from_dict, where apply_dict accepts an existing object and applies anything in the dict, and from_dict just calls apply_dict after creating a new instance of the class.
    # That would make it very easy to do things like allow the user to specify two or more AttackParams files, applying each one on top of the other.
    # It should also make loading even one jailbreak detection file more reliable, because if the file is from an older version, the result would automatically inherit the defaults of the newer version, even if they were set after creation.
    # like this:
        
    # @staticmethod
    # def apply_dict(existing_object, property_dict):
        # if not isinstance(existing_object, AttackParams):
            # raise JSONSerializationException(f"Cannot apply properties for the AttackParams class to an instance of the class '{existing_object.__class__.__name__}'")
        # super(AttackParams, existing_object).set_properties_from_dict(existing_object, property_dict)
        # if len(existing_object.radiation_gardens) > 0:
            # deserialized_gardens = []
            # for i in range(0, len(existing_object.radiation_gardens)):
                # deserialized_gardens.append(RadiationGarden.from_dict(existing_object.radiation_gardens[i]))
            # existing_object.radiation_gardens = deserialized_gardens
        # if len(existing_object.jailbreak_detection_rule_set) > 0:
            # deserialized_jailbreak_rule_set = []
            # for i in range(0, len(existing_object.jailbreak_detection_rule_set)):
                # deserialized_jailbreak_rule_set.append(LLMJailbreakDetectorRule.from_dict(existing_object.jailbreak_detection_rule_set[i]))
            # existing_object.jailbreak_detection_rule_set = deserialized_jailbreak_rule_set
        # return existing_object
    
    # When one of the properties is a dictionary:    
    # @staticmethod
    # def apply_dict(existing_object, property_dict):
        # if not isinstance(existing_object, StatisticsCube):
            # raise JSONSerializationException(f"Cannot apply properties for the StatisticsCube class to an instance of the class '{existing_object.__class__.__name__}'")
        # super(StatisticsCube, existing_object).set_properties_from_dict(existing_object, property_dict)
        # if existing_object.datasets is not None:
            # deserialized_content = {}
            # for k in existing_object.datasets.keys():
                # deserialized_content[k] = (StatisticsDataSet.from_dict(existing_object.datasets[k]))
            # existing_object.datasets = deserialized_content
        # return existing_object

    # @staticmethod
    # def from_dict(property_dict):
        # result = AttackParams()
        # result = AttackParams.apply_dict(result, property_dict)
        # return result
    
    # @staticmethod
    # def apply_json(existing_object, json_string):
        # return AttackParams.apply_dict(existing_object, json.loads(json_string))

    @staticmethod
    def make_value_serializable(value_to_serialize):
        serialized_value = value_to_serialize
        handled = False
        # if the object is a list, return a list of serialized objects
        if not handled and isinstance(value_to_serialize, list):
            serialized_value = []
            for i in range(0, len(value_to_serialize)):
                serialized_value.append(JSONSerializableObject.make_value_serializable(value_to_serialize[i]))
            handled = True
        # if the object is a dictionary, return a dictionary of serialized objects
        if not handled and isinstance(value_to_serialize, dict):
            serialized_value = JSONSerializableObject.make_dictionary_serializable(value_to_serialize)
            handled = True
        # if the object is a subclass of this class, return its dictionary form
        if not handled and isinstance(value_to_serialize, JSONSerializableObject):
            serialized_value = value_to_serialize.to_dict()
            handled = True
        # if the object is a basic type that json.dumps can serialize on its own, let it do that
        if not handled:
            if isinstance(value_to_serialize, int):
                handled = True
            if not handled and isinstance(value_to_serialize, bool):
                handled = True
            if not handled and isinstance(value_to_serialize, float):
                handled = True
            if not handled and isinstance(value_to_serialize, str):
                handled = True
            if not handled and isinstance(value_to_serialize, type(None)):
                handled = True
        # if the object is a tensor, handle that
        if not handled:
            if isinstance(value_to_serialize, torch.Tensor):
                serialized_value = tensor_to_dict(value_to_serialize)
                handled = True
        # if the object still hasn't been handled, but it has a to_dict() method, use that
        # this is useful for third-party code that follows the to_dict() convention, but also enums and whatnot that probably shouldn't be subclasses of this one.
        if not handled:
            if hasattr(value_to_serialize, "to_dict"):
                to_dict_method = hasattr(value_to_serialize, "to_dict")
                if callable(to_dict_method):
                    serialized_value = value_to_serialize.to_dict()
                    handled = True
        # regex objects => string
        if not handled:
            if isinstance(value_to_serialize, re.Pattern):
                serialized_value = str(value_to_serialize)
                handled = True

        # something was passed to this function that it doesn't know how to serialize
        if not handled:
            raise JSONSerializationException(f"Could not serialize object of type '{type(value_to_serialize)}': {value_to_serialize}")

        return serialized_value

    @staticmethod
    def make_dictionary_serializable(object_to_serialize):
        result = {}
        for key, value in object_to_serialize.items():
            include_property = True
            handled = False
            # don't attempt to include methods
            if callable(value):
                include_property = False
                handled = True
            # don't include private properties
            if not handled and include_property and key.startswith('__'):
                include_property = False
                handled = True

            if include_property:
                try:
                    result[key] = JSONSerializableObject.make_value_serializable(value)
                except Exception as e:
                    raise JSONSerializationException(f"Could not serialize property '{key}' of type '{type(value)}' within object of type '{type(object_to_serialize)}': {e}")
        return result

    @staticmethod
    def properties_to_dict(object_to_serialize):
        result = JSONSerializableObject.make_dictionary_serializable(object_to_serialize.__dict__)
        return result

    @staticmethod
    def properties_to_json(object_to_serialize):
        result = json.dumps(object_to_serialize.to_dict(), indent=4)
        return result
    
    @staticmethod
    def set_properties_from_dict(object_to_deserialize, property_dict):
        for key, value in property_dict.items():
            if hasattr(object_to_deserialize, key):
                setattr(object_to_deserialize, key, value)
            else:
                raise JSONSerializationException(f"Could not set property '{key}' for object of type '{type(object_to_deserialize)}' to '{value}', because the type does not contain a property with that name.")
        return object_to_deserialize
    
    @staticmethod
    def set_properties_from_json(object_to_deserialize, json_string):
        return JSONSerializableObject.set_properties_from_dict(object_to_deserialize, json.loads(json_string))