#!/bin/env python

import datetime
import math
import os
import pathlib
import shutil
import sys
import tempfile
import torch

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





