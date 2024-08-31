#!/bin/env python

from enum import auto
from enum import IntFlag

from llm_attacks_bishopfox.json_serializable_object import JSONSerializableObject

# For gamma garden and neutron garden modes
class RadiationGardenBehaviour(IntFlag):
    # Select from all allowed tokens ("neutron garden") instead of only tokens that have been part of generated adversarial content ("gamma garden")
    NEUTRON_GARDEN = auto()
    # Trigger a randomization event every time a rollback occurs
    RANDOMIZE_ON_ROLLBACK = auto()
    # Trigger a randomization event every time the sequential failure counter mod n is zero
    RANDOMIZE_ON_SEQUENTIAL_FAILURE_COUNT = auto()
    # Trigger a randomization event every time the failure counter mod n is zero, regardless of how many failures have occurred in a row
    RANDOMIZE_ON_FAILURE_COUNT = auto()
    # Trigger a randomization event every n iterations, regardless of success/failure
    # n is configured in RadiationGarden
    RANDOMIZE_ON_ITERATION_COUNT = auto()

# This class is used to allow the attack to incorporate randomization events into the generation process.
# attack_classes.AttackParams has a radiation_gardens[] property that is empty by default, but can contain an arbitrary number of these.
class RadiationGarden(JSONSerializableObject):
    def __init__(self):
        self.radiation_garden_behaviour_flags = RadiationGardenBehaviour(0)

        # When a randomization event occurs, each token in the adversarial content has this percent chance of being randomized - float value from 0.0 to 1.0
        self.radiation_garden_randomization_likelihood = 0.05

        # if self.radiation_garden_behaviour_flags.RANDOMIZE_ON_SEQUENTIAL_FAILURE_COUNT is set, then a randomization event will be triggered every time the failure count % (this number) is zero, regardless of whether or not a rollback is also triggered.
        self.radiation_garden_sequential_failure_count = 20

        # if self.radiation_garden_behaviour_flags.RANDOMIZE_ON_ITERATION_COUNT is set, then a randomization event will be triggered every time the iteration count % (this number) is zero.
        self.radiation_garden_forced_mutation_iteration_count = 100
    
    
    def to_dict(self):
        result = super(RadiationGarden, self).properties_to_dict(self)
        return result
    
    @staticmethod
    def from_dict(property_dict):
        result = RadiationGarden()
        super(RadiationGarden, result).set_properties_from_dict(result, property_dict)
        radiation_garden_behaviour_flags = RadiationGardenBehaviour(0)
        if result.radiation_garden_behaviour_flags is not None:
            radiation_garden_behaviour_flags = RadiationGardenBehaviour(result.radiation_garden_behaviour_flags)
        result.radiation_garden_behaviour_flags = RadiationGardenBehaviour(0)
        return result
    
    # def to_dict(self):
        # result = {}
        # result["radiation_garden_behaviour_flags"] = self.radiation_garden_behaviour_flags
        # result["radiation_garden_randomization_likelihood"] = self.radiation_garden_randomization_likelihood
        # result["radiation_garden_sequential_failure_count"] = self.radiation_garden_sequential_failure_count
        # result["radiation_garden_forced_mutation_iteration_count"] = self.radiation_garden_forced_mutation_iteration_count

        # return result

    def to_json(self):
        return JSONSerializableObject.json_dumps(self.to_dict())
    
    def copy(self):
        return RadiationGarden.from_dict(self.to_dict())
    
    # @staticmethod
    # def from_dict(property_dict):
        # result = RadiationGarden()
        # result.radiation_garden_behaviour_flags = property_dict["radiation_garden_behaviour_flags"]
        # result.radiation_garden_randomization_likelihood = property_dict["radiation_garden_randomization_likelihood"]
        # result.radiation_garden_sequential_failure_count = property_dict["radiation_garden_sequential_failure_count"]
        # result.radiation_garden_forced_mutation_iteration_count = property_dict["radiation_garden_forced_mutation_iteration_count"]

        # return result
    
    @staticmethod
    def from_json(json_string):
        return RadiationGarden.from_dict(json.loads(json_string))