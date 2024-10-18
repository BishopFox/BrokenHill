#!/bin/env python

import datetime
import math
import os
import pathlib
import re
import shlex
import shutil
import sys
import tempfile
import torch

import numpy

from enum import StrEnum

class RequiredValueIsNoneException(Exception):
    pass

def get_escaped_string(input_string):
    #print(f"[get_escaped_string] Debug: escaping string '{input_string}'")
    if input_string is None:
        return None
    result = input_string.replace("\n", "\\n")
    replaced_chars = []
    for i in range(0, 32):
        replaced_chars.append(i)
    for i in range(127, 256):
        replaced_chars.append(i)
    for i in range(0, len(replaced_chars)):
        result = result.replace(chr(replaced_chars[i]), f"\\x{i:02x}")
    #print(f"[get_escaped_string] Debug: escaped string '{input_string}' to '{result}'")
    return result

def get_file_content(file_path, failure_is_critical = True):
    result = None
    try:
        with open(file_path) as input_file:
            result = input_file.read()        
    except Exception as e:
        print(f"Couldn't read the file '{file_path}': {e}")
        if failure_is_critical:
            sys.exit(1)
        result = None
    return result

def numeric_string_to_int(s):
    result = -1
    if s[0:2].lower() == "0x":
        try:
            result = int(s[2:], 16)
            return result
        except Exception as e:
            print(f"Tried to parse the value '{s}' as hexadecimal and failed: {e}")
            sys.exit(1)
    else:
        try:
            result = int(s, 10)
            return result
        except Exception as e:
            print(f"Tried to parse the value '{s}' as decimal and failed: {e}")
            sys.exit(1)
    print(f"Unhandled case while parsing the value '{s}'")
    sys.exit(1)

def numeric_string_to_float(s):
    result = None
    try:
        result = float(s)
        return result
    except Exception as e:
        print(f"Tried to parse the value '{s}' as a floating-point number and failed: {e}")
        sys.exit(1)
    return None
 
 
# begin: https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# end: https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse

def comma_delimited_string_to_integer_array(input_string):
    result = []
    input_array = input_string.split(",")
    for i in range(0, len(input_array)):
        current_value = input_array[i]
        if not isinstance(current_value, type(None)):
            current_value = current_value.strip()
            if current_value != "":
                current_integer = numeric_string_to_int(current_value)
                result.append(current_integer)
    return result

def get_now():
    return datetime.datetime.now(tz=datetime.timezone.utc)

def get_time_string(dt = get_now()):
    return dt.replace(microsecond=0).isoformat()

def update_elapsed_time_string(time_string, new_element_name, count):
    result = time_string
    s = f"{count} {new_element_name}"
    if count != 1:
        s += "s"
    if "(" not in result:
        result += f" ({s}"
    else:
        result += f", {s}"
    
    return result

def get_elapsed_time_string(start_time, end_time):
    delta_value = end_time - start_time
    #print(f"{delta_value}, {delta_value.days}, {delta_value.seconds}, {delta_value.microseconds}")
    result = f"{delta_value}"
    num_days = delta_value.days
    if num_days > 0:
        result = update_elapsed_time_string(result, "day", num_days)
        delta_value -= datetime.timedelta(days=num_days)
    num_hours = int(math.floor(delta_value.seconds / 3600))
    if num_hours > 0:
        result = update_elapsed_time_string(result, "hour", num_hours)
        delta_value -= datetime.timedelta(hours=num_hours)
    num_minutes = int(math.floor(delta_value.seconds / 60))
    if num_minutes > 0:
        result = update_elapsed_time_string(result, "minute", num_minutes)
        delta_value -= datetime.timedelta(minutes=num_minutes)
    num_seconds = delta_value.seconds
    if num_seconds > 0:
        result = update_elapsed_time_string(result, "second", num_seconds)
        delta_value -= datetime.timedelta(seconds=num_seconds)
    num_milliseconds = int(math.floor(delta_value.microseconds / 1000))
    if num_milliseconds > 0:
        result = update_elapsed_time_string(result, "millisecond", num_milliseconds)
        delta_value -= datetime.timedelta(milliseconds=num_milliseconds)
    if "(" in result:
        result += ")"
    return result

def get_model_size(mdl):
    tempfile_path = tempfile.mktemp()
    #print(f"Debug: writing model to '{tempfile_path}'")
    torch.save(mdl.state_dict(), tempfile_path)
    model_size = os.path.getsize(tempfile_path)
    #print(f"Debug: model size: {model_size}")
    #result = "%.2f" %(model_size)
    result = model_size
    os.remove(tempfile_path)
    return result

# write content to a temporary file first, then delete any existing output file, then move the temporary file to the output file location
# Prevents overwriting a complete output file with partial output in the event of a crash
def safely_write_text_output_file(file_path, content, file_mode = "w"):
    file_directory_path = os.path.dirname(file_path)
    if not os.path.isdir(file_directory_path):
        err_message = f"The directory specified for the file '{file_path}' ('{file_directory_path}') does not exist."
        raise Exception(err_message)
    # thanks for deprecating mktemp, Python devs!
    temporary_path = None
    try:
        born_2_lose, temporary_path = tempfile.mkstemp(dir = file_directory_path)
        os.close(born_2_lose)
    except Exception as e:
        err_message = f"Couldn't create a temporary file in '{file_directory_path}': {e}"
        raise Exception(err_message)
    temporary_path_object = pathlib.Path(temporary_path)
    file_path_object = pathlib.Path(file_path)
    # if append mode, copy the existing file to the temporary location first
    if file_mode == "a":
        if os.path.isfile(file_path):
            try:
                # of course pathlib doesn't have a copy method
                shutil.copy(file_path, temporary_path)
            except Exception as e:
                err_message = f"Couldn't copy the existing file '{file_path}' to the temporary path '{temporary_path}': {e}"
                raise Exception(err_message)
    successful_write = False
    try:
        with open(temporary_path, file_mode) as output_file:
            output_file.write(content)
        #return file_path
        successful_write = True
    except Exception as e:
        try:
            with open(file_path, f"{file_mode}b") as output_file:
                output_file.write(str.encode(content))
            successful_write = True
        except Exception as e2:
            err_message = f"Couldn't write to the temporary file '{temporary_path}' in either text mode ('{e}') or binary mode('{e2}')."
            raise Exception(err_message)
    successful_delete = False
    if successful_write:
        if os.path.isfile(file_path):
            try:
                file_path_object.unlink() 
                successful_delete = True
            except Exception as e:
                err_message = f"Couldn't delete the existing file '{file_path}' to replace it with the newly-generated file: {e}."
                raise Exception(err_message)
        else:
            successful_delete = True
    successful_replace = False
    if successful_write and successful_delete:
        try:
            temporary_path_object.rename(file_path) 
            successful_replace = True
        except Exception as e:
            err_message = f"Couldn't rename the temporary file '{temporary_path}' to '{file_path}': {e}."
            raise Exception(err_message)
    if successful_write and successful_delete and successful_replace:
        return file_path            
    return None


def add_value_to_list_if_not_already_present(existing_list, new_value, ignore_none = False):
    if ignore_none:
        if isinstance(new_value, type(None)):
            return existing_list
    if new_value not in existing_list:
        existing_list.append(new_value)
    return existing_list
    
def add_values_to_list_if_not_already_present(existing_list, new_value_list, ignore_none = False):
    for i in range(0, len(new_value_list)):
        existing_list = add_value_to_list_if_not_already_present(existing_list, new_value_list[i], ignore_none = ignore_none)
    return existing_list

# regex flags are an integer, so it would work to just serialize this value as a number, but it would be a number that would have no guarantee of representing the same set of flags for a different Python version/platform/etc.
class RegexFlagString(StrEnum):
    re_ASCII = 're.ASCII'
    re_DEBUG = 're.DEBUG'
    re_IGNORECASE = 're.IGNORECASE'
    re_LOCALE = 're.LOCALE'
    re_MULTILINE = 're.MULTILINE'
    re_NOFLAG = 're.NOFLAG'
    re_DOTALL = 're.DOTALL'
    re_UNICODE = 're.UNICODE'
    re_VERBOSE = 're.VERBOSE'

def regex_flags_to_list(regex_flags):
    result = []
    if (regex_flags & re.ASCII) == re.ASCII:
        result.append(str(RegexFlagString.re_ASCII))
    if (regex_flags & re.DEBUG) == re.DEBUG:
        result.append(str(RegexFlagString.re_DEBUG))
    if (regex_flags & re.IGNORECASE) == re.IGNORECASE:
        result.append(str(RegexFlagString.re_IGNORECASE))
    if (regex_flags & re.LOCALE) == re.LOCALE:
        result.append(str(RegexFlagString.re_LOCALE))
    if (regex_flags & re.MULTILINE) == re.MULTILINE:
        result.append(str(RegexFlagString.re_MULTILINE))
    if (regex_flags & re.DOTALL) == re.DOTALL:
        result.append(str(RegexFlagString.re_DOTALL))
    if (regex_flags & re.UNICODE) == re.UNICODE:
        result.append(str(RegexFlagString.re_UNICODE))
    if (regex_flags & re.VERBOSE) == re.VERBOSE:
        result.append(str(RegexFlagString.re_VERBOSE))
    # only include NOFLAG if there are no flags specified, since it's equivalent to 0
    if len(result) == 0:
        result.append(str(RegexFlagString.re_NOFLAG))
    return result

def regex_flags_from_list(regex_flag_list):
    result = re.NOFLAG
    # normalize any data read from elsewhere
    if RegexFlagString.re_ASCII in regex_flag_list:
        result = result | re.ASCII
    if RegexFlagString.re_DEBUG in regex_flag_list:
        result = result | re.DEBUG
    if RegexFlagString.re_IGNORECASE in regex_flag_list:
        result = result | re.IGNORECASE
    if RegexFlagString.re_LOCALE in regex_flag_list:
        result = result | re.LOCALE
    if RegexFlagString.re_MULTILINE in regex_flag_list:
        result = result | re.MULTILINE
    if RegexFlagString.re_DOTALL in regex_flag_list:
        result = result | re.DOTALL
    if RegexFlagString.re_UNICODE in regex_flag_list:
        result = result | re.UNICODE
    if RegexFlagString.re_VERBOSE in regex_flag_list:
        result = result | re.VERBOSE
    if RegexFlagString.re_ASCII in regex_flag_list:
        result = result | re.ASCII
    return result

def get_random_token_id(numpy_random_generator, token_allow_and_deny_lists):
    token_index = numpy_random_generator.integers(0, high = len(token_allow_and_deny_lists.allowlist))
    return token_allow_and_deny_lists.allowlist[token_index]

def get_random_token_ids(numpy_random_generator, token_allow_and_deny_lists, number_of_token_ids):
    result = []
    for random_token_num in range(0, number_of_token_ids):
        new_token_id = get_random_token_id(numpy_random_generator, token_allow_and_deny_lists)
        result.append(new_token_id)
    return result

def slice_from_dict(property_dict):
    start = None
    stop = None
    step = None
    if "start" in property_dict.keys():
        start = property_dict["start"]
    if "stop" in property_dict.keys():
        stop = property_dict["stop"]
    if "step" in property_dict.keys():
        step = property_dict["step"]
    result = slice(start, stop, step)
    return result

def find_index_of_first_nonmatching_element(list1, list2):
    print(f"[find_index_of_first_nonmatching_element] Debug: list1 = {list1}, list2 = {list2}")
    len_list1 = len(list1)
    len_list2 = len(list2)
    last_index = len_list1
    #result = None
    if len_list2 < last_index:
        last_index = len_list2
        #result = last_index
    
    i = 0
    while i < last_index:
        if list1[i] != list2[i]:
            print(f"[find_index_of_first_nonmatching_element] Debug: result = {i}")
            return i
        i += 1
    
    if len_list1 != len_list2:
        print(f"[find_index_of_first_nonmatching_element] Debug: len_list1 != len_list2, result = {i}")
        return i
    
    return None

def remove_whitespace_and_nonprintable_characters(input_string):    
    result = ""
    
    for i in range(0, len(input_string)):
        current_char = input_string[i]
        include_char = True
        if not current_char.isprintable():
            include_char = False
        if include_char:
            if current_char.strip() == "":
                include_char = False
        if include_char:
            result = f"{result}{current_char}"
    return result

def append_single_or_list_members(existing_list, value_or_list_to_add, ignore_if_none = False):
    if ignore_if_none:
        if isinstance(value_or_list_to_add, type(None)):
            return existing_list
    if isinstance(value_or_list_to_add, list):
        for list_member in value_or_list_to_add:
            existing_list = append_single_or_list_members(existing_list, list_member, ignore_if_none = ignore_if_none)
    else:
        if value_or_list_to_add not in existing_list:
            existing_list.append(value_or_list_to_add)
    return existing_list
 
def find_first_occurrence_of_array_in_array(inner_array, outer_array, start_index = 0, stop_index = None):
    result = None
    #print(f"[find_first_occurrence_of_array_in_array] Debug: Searching for '{inner_array}' in '{outer_array}'")
    len_inner = len(inner_array)
    len_outer = len(outer_array)
    range_end = len_outer
    if not isinstance(stop_index, type(None)):
        range_end = stop_index
    #print(f"[find_first_occurrence_of_array_in_array] Debug: searching for '{inner_array}' in '{outer_array}' from index {start_index} to {range_end}'")
    for i in range(start_index, range_end):
        if (i + len_inner) >=  len_outer:
            break
        if outer_array[i] == inner_array[0]:
            is_match = True
            #print(f"[find_first_occurrence_of_array_in_array] Debug: found potential match beginning at index {i}'")
            for j in range(1, len_inner):
                if outer_array[i + j] != inner_array[j]:
                    is_match = False
                    #print(f"[find_first_occurrence_of_array_in_array] Debug: '{outer_array[i + j]}' != '{inner_array[j]}'")
                    break
            if is_match:
                #print(f"[find_first_occurrence_of_array_in_array] Debug: found match at index {i}")
                return i
    #print(f"[find_first_occurrence_of_array_in_array] Debug: no match found")
    return result

def find_last_occurrence_of_array_in_array(inner_array, outer_array, start_index = 0, stop_index = None):
    result = None
    #print(f"[find_last_occurrence_of_array_in_array] Debug: Searching for '{inner_array}' in '{outer_array}' with start_index = {start_index} and stop_index = {stop_index}")
    len_inner = len(inner_array)
    len_outer = len(outer_array)
    if len_inner > len_outer or len_inner == 0 or len_outer == 0:
        print(f"[find_last_occurrence_of_array_in_array] Warning: cannot search for '{inner_array}' in '{outer_array}' with start_index = {start_index} and stop_index = {stop_index} - returning {result}")
        return result
    range_start = len_outer - len_inner
    if not isinstance(stop_index, type(None)):
        range_start = stop_index - len_inner
    range_end = start_index - 1
    if range_end < -1 or range_end > len_outer or range_start > len_outer or range_start < -1:
        raise TrashFireTokenException(f"[find_last_occurrence_of_array_in_array] Error: cannot search for '{inner_array}' in '{outer_array}' from index {range_start} to {range_end}' with start_index = {start_index} and stop_index = {stop_index}")
    #print(f"[find_last_occurrence_of_array_in_array] Debug: searching for '{inner_array}' in '{outer_array}' from index {range_start} to {range_end}'")
    for i in range(range_start, range_end, -1):
        if outer_array[i] == inner_array[0]:
            is_match = True
            #print(f"[find_last_occurrence_of_array_in_array] Debug: found potential match beginning at index {i}'")
            for j in range(1, len_inner):
                if outer_array[i + j] != inner_array[j]:
                    is_match = False
                    #print(f"[find_last_occurrence_of_array_in_array] Debug: '{outer_array[i + j]}' != '{inner_array[j]}'")
                    break
            if is_match:
                #print(f"[find_last_occurrence_of_array_in_array] Debug: found match at index {i}")
                return i
    #print(f"[find_last_occurrence_of_array_in_array] Debug: no match found")
    return result

# Convert a nice array of command elements to a terrible single value which can be used with bash -c.
# Because it's 2024, and Python still makes it very hard to capture stderr + stdout exactly the way that it would appear in a shell, without deadlocking or running out of buffer space, and while letting the developer set a timeout on execution without some kind of hokey second thread
def command_array_to_bash_c_argument(command_array):
    inner_command = None
    for i in range(0, len(command_array)):
        current_element = shlex.quote(command_array[i])
        if inner_command is None:
            inner_command = current_element
        else:
            inner_command = f"{inner_command} {current_element}"
    #result = shlex.quote(inner_command)
    result = inner_command
    print(f"[command_array_to_bash_c_argument] Debug: input = {command_array}, output = {result}")
    return result


# return a slice that begins with the lower of two other slices start values, and ends with the greater of their stop values.
# Not quite a "union" operation, so I've avoided that term.
def get_widened_slice(slice1, slice2):
    slice_start = None
    slice_stop = None
    if slice1 is not None:
        if slice1.start is not None:
            slice_start = slice1.start
        if slice1.stop is not None:
            slice_stop = slice1.stop
    if slice2 is not None:
        if slice2.start is not None:
            set_start = False
            if slice_start is None:
                set_start = True
            else:
                if slice2.start < slice_start:
                    set_start = True
            if set_start:
                slice_start = slice2.start
        if slice2.stop is not None:
            set_stop = False
            if slice_stop is None:
                set_stop = True
            else:
                if slice2.stop > slice_stop:
                    set_stop = True
            if set_stop:
                slice_stop = slice2.stop
    result = slice(slice_start, slice_stop)
    print(f"[get_widened_slice] slice1 = {slice1}, slice2 = {slice2}, result = {result}")
    return result
