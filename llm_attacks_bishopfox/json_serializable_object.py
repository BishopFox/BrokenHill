#!/bin/env python

import copy
import json
import re

class JSONSerializationException(Exception):
    pass

class JSONSerializableObject:    
    @staticmethod
    def json_dumps(object_to_dump):
        return json.dumps(object_to_dump, indent = 4)

    def to_dict(self):
        return JSONSerializableObject.properties_to_dict(self)
    
    def to_json(self):
        return JSONSerializableObject.json_dumps(self.to_dict())

    def copy(self):
        result = JSONSerializableObject()
        return JSONSerializableObject.set_properties_from_dict(result, self.to_dict())

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
                result[key] = JSONSerializableObject.make_value_serializable(value)
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