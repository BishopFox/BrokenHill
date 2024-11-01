#!/bin/env python

import logging
import statistics

from llm_attacks_bishopfox.json_serializable_object import JSONSerializableObject

logger = logging.getLogger(__name__)

class StatisticsException(Exception):
    pass

class StatisticsDataSet(JSONSerializableObject):
    def __init__(self):
        self.dataset_name = None
        self.mean = None
        self.median = None
        self.mode = None
        self.maximum = None
        self.minimum = None
        self.value_range = None

    def populate_dataset(self, dataset_name, source_data_array):
        self.dataset_name = dataset_name
        if len(source_data_array) > 0:
            self.mean = statistics.mean(source_data_array)
            self.median = statistics.median(source_data_array)
            self.mode = statistics.mode(source_data_array)
            self.maximum = None
            self.minimum = None
            self.value_range = None
            for i in range(0, len(source_data_array)):
                if self.maximum is None or source_data_array[i] > self.maximum:
                    self.maximum = source_data_array[i]
                if self.minimum is None or source_data_array[i] < self.minimum:
                    self.minimum = source_data_array[i]
            if self.maximum is not None and self.minimum is not None:
                self.value_range = self.maximum - self.minimum
        else:
            logger.warning(f"Got an empty source data array for dataset '{dataset_name}'")

    def to_dict(self):
        result = super(StatisticsDataSet, self).properties_to_dict(self)
        return result

    def to_json(self):
        return JSONSerializableObject.json_dumps(self.to_dict(), use_indent = False)
    
    def copy(self):
        return StatisticsDataSet.from_dict(self.to_dict())
    
    @staticmethod
    def apply_dict(existing_object, property_dict):
        if not isinstance(existing_object, StatisticsDataSet):
            raise JSONSerializationException(f"Cannot apply properties for the StatisticsDataSet class to an instance of the class '{existing_object.__class__.__name__}'")
        super(StatisticsDataSet, existing_object).set_properties_from_dict(existing_object, property_dict)
        
        return existing_object
    
    @staticmethod
    def from_dict(property_dict):
        result = StatisticsDataSet()
        result = StatisticsDataSet.apply_dict(result, property_dict)
        return result
    
    @staticmethod
    def apply_json(existing_object, json_string):
        return StatisticsDataSet.apply_dict(existing_object, json.loads(json_string))
    
    @staticmethod
    def from_json(json_string):
        return StatisticsDataSet.from_dict(json.loads(json_string))
    
class StatisticsCube(JSONSerializableObject):
    def __init__(self):
        self.cube_name = None
        self.datasets = {}

    def get_dataset(self, dataset_name, raise_on_missing = True):
        if dataset_name not in self.datasets.keys():
            if raise_on_missing:
                raise StatisticsException(f"The dataset '{dataset_name}' does not exist in this StatisticsCube")
            else:
                return None
        return self.datasets[dataset_name]

    def add_dataset(self, dataset_name, source_data_array, raise_on_duplicate = True):
        if dataset_name in self.datasets.keys():
            if raise_on_duplicate:
                raise StatisticsException(f"The dataset '{dataset_name}' already exists in this StatisticsCube")
        new_dataset = StatisticsDataSet()
        new_dataset.populate_dataset(dataset_name, source_data_array)
        self.datasets[dataset_name] = new_dataset
    
    def add_or_update_dataset(self, dataset_name, source_data_array):
        self.add_dataset(dataset_name, source_data_array, raise_on_duplicate = False)
    
    def delete_dataset(self, dataset_name, raise_on_missing = True):
        if dataset_name not in self.datasets.keys():
            if raise_on_missing:
                raise StatisticsException(f"The dataset '{dataset_name}' does not exist in this StatisticsCube")
        else:
            self.datasets.pop(dataset_name)

    def to_dict(self):
        result = super(StatisticsCube, self).properties_to_dict(self)
        return result

    def to_json(self):
        return JSONSerializableObject.json_dumps(self.to_dict(), use_indent = False)
    
    def copy(self):
        return StatisticsCube.from_dict(self.to_dict())
    
    @staticmethod
    def apply_dict(existing_object, property_dict):
        if not isinstance(existing_object, StatisticsCube):
            raise JSONSerializationException(f"Cannot apply properties for the StatisticsCube class to an instance of the class '{existing_object.__class__.__name__}'")
        super(StatisticsCube, existing_object).set_properties_from_dict(existing_object, property_dict)
        if existing_object.datasets is not None:
            deserialized_content = {}
            for k in existing_object.datasets.keys():
                deserialized_content[k] = (StatisticsDataSet.from_dict(existing_object.datasets[k]))
            existing_object.datasets = deserialized_content
        return existing_object
    
    @staticmethod
    def from_dict(property_dict):
        result = StatisticsCube()
        result = StatisticsCube.apply_dict(result, property_dict)
        return result
    
    @staticmethod
    def apply_json(existing_object, json_string):
        return StatisticsCube.apply_dict(existing_object, json.loads(json_string))
    
    @staticmethod
    def from_json(json_string):
        return StatisticsCube.from_dict(json.loads(json_string))