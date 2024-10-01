#!/bin/env python

from enum import auto
from enum import IntFlag

from llm_attacks_bishopfox.json_serializable_object import JSONSerializableObject

class RadiationGardenTriggerType(IntFlag):
    # Trigger based on the current jailbreak count
    JAILBREAK_COUNT = auto()
    # Trigger based on the fraction (jailbreak count) / (number of randomized LLMs)
    JAILBREAK_PERCENTAGE = auto()
    # Trigger based on loss value
    LOSS = auto()

# For gamma garden and neutron garden modes
# class RadiationGardenBehaviour(IntFlag):
    # # Select from all allowed tokens ("neutron garden") instead of only tokens that have been part of generated adversarial content ("gamma garden")
    # NEUTRON_GARDEN = auto()
    # # Trigger a randomization event every time a rollback occurs
    # RANDOMIZE_ON_ROLLBACK = auto()
    # # Trigger a randomization event every time the sequential failure counter mod n is zero
    # RANDOMIZE_ON_SEQUENTIAL_FAILURE_COUNT = auto()
    # # Trigger a randomization event every time the failure counter mod n is zero, regardless of how many failures have occurred in a row
    # RANDOMIZE_ON_FAILURE_COUNT = auto()
    # # Trigger a randomization event every n iterations, regardless of success/failure
    # # n is configured in RadiationGarden
    # RANDOMIZE_ON_ITERATION_COUNT = auto()
class RadiationGardenTriggerCriteria(IntFlag):
    # Trigger an event every time a rollback occurs
    ON_ROLLBACK = auto()
    # Trigger an event every time the sequential condition counter mod n is zero
    ON_SEQUENTIAL_CONDITION_COUNT = auto()
    # Trigger an event every n iterations, regardless of success/failure
    # n is configured in RadiationGarden
    ON_ITERATION_COUNT = auto()
    # Trigger an event when a 
    ON_

# This class is used to allow the attack to incorporate randomization events into the generation process.
# TKTK: equivalent of --add-token-on-jailbreak-plateau <integer>
# TKTK: equivalent of --add-token-on-loss-plateau <integer>
# attack_classes.AttackParams has a radiation_gardens[] property that is empty by default, but can contain an arbitrary number of these.
class RadiationGarden(JSONSerializableObject):
    def __init__(self):
        # When an event occurs, whatever the change is has this percent chance of being randomized - float value from 0.0 to 1.0
        # For changes that operate at the individual token level, this is the likelihood of *each token* being altered
        self.likelihood = 0.0

        # Counter for whatever condition the radiation garden triggers off of
        self.sequential_condition_count = 0

        # if self.radiation_garden_behaviour_flags.ON_SEQUENTIAL_FAILURE_COUNT is set, then an event will be triggered every time the failure count % (this number) is zero, regardless of whether or not a rollback is also triggered.
        self.sequential_condition_count_trigger_value = None

        # if self.radiation_garden_behaviour_flags.ON_ITERATION_COUNT is set, then a randomization event will be triggered every time the iteration count % (this number) is zero.
        self.cyclic_mutation_iteration_count_trigger_value = None
    
    
    def to_dict(self):
        result = super(RadiationGarden, self).properties_to_dict(self)
        return result
    
    @staticmethod
    def from_dict(property_dict):
        result = RadiationGarden()
        super(RadiationGarden, result).set_properties_from_dict(result, property_dict)
        return result

    def to_json(self):
        return JSONSerializableObject.json_dumps(self.to_dict())
    
    def copy(self):
        return RadiationGarden.from_dict(self.to_dict())
        
    @staticmethod
    def from_json(json_string):
        return RadiationGarden.from_dict(json.loads(json_string))

class RadiationGarden(JSONSerializableObject):
    def __init__(self):