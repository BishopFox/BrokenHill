#!/bin/env python

import copy
import datetime
import enum
import logging
import math
import os
import sys
import time

from llm_attacks_bishopfox.util.util_functions import strip_ansi_codes

logger = logging.getLogger(__name__)

class LoggingException(Exception):
    pass

class ANSIFormatter():
    def __init__(self):
        self.ansi_map = ANSIFormatter.get_ansi_font_format_code_map()
    
    @staticmethod
    def get_ansi_esc(code):
        return f'\033[{code}'

    @staticmethod
    def get_ansi_font_format_code_map():
        result = {}

        result["reset"] = "0"
        result["bold"] = "1"
        result["faint"] = "2"
        result["italic"] = "3"
        result["underline"] = "4"
        result["blink_slow"] = "5"
        result["blink_fast"] = "6"
        result["inverse"] = "7"
        result["strikethrough"] = "9"
        result["normal_intensity"] = "22"
        result["italic_off"] = "23"
        result["underline_off"] = "24"
        result["blink_off"] = "25"
        result["inverse_off"] = "27"
        result["strikethrough_off"] = "29"
        
        result["font_default"] = "10"
        result["font_alt_1"] = "11"
        result["font_alt_2"] = "12"
        result["font_alt_3"] = "13"
        result["font_alt_4"] = "14"
        result["font_alt_5"] = "15"
        result["font_alt_6"] = "16"
        result["font_alt_7"] = "17"
        result["font_alt_8"] = "18"
        result["font_alt_9"] = "19"

        # this section is why I didn't just use an enum
        result["fg_black"] = "22;30"
        result["fg_red"] = "22;31"
        result["fg_green"] = "22;32"
        result["fg_brown"] = "22;33"
        result["fg_blue"] = "22;34"
        result["fg_magenta"] = "22;35"
        result["fg_cyan"] = "22;36"
        result["fg_light_grey"] = "22;37"
        result["fg_dark_grey"] = "1;30"
        result["fg_light_red"] = "1;31"
        result["fg_light_green"] = "1;32"
        result["fg_yellow"] = "1;33"
        result["fg_light_blue"] = "1;34"
        result["fg_light_magenta"] = "1;35"
        result["fg_light_cyan"] = "1;36"
        result["fg_white"] = "1;37"
        
        result["bg_black"] = "40"
        result["bg_red"] = "41"
        result["bg_green"] = "42"
        result["bg_brown"] = "43"
        result["bg_blue"] = "44"
        result["bg_magenta"] = "45"
        result["bg_cyan"] = "46"
        result["bg_light_grey"] = "47"
        
        # the following codes are not universally supported
        result["fg_bright_black"] = "90"
        result["fg_bright_red"] = "91"
        result["fg_bright_green"] = "92"
        result["fg_bright_yellow"] = "93"
        result["fg_bright_blue"] = "94"
        result["fg_bright_magenta"] = "95"
        result["fg_bright_cyan"] = "96"
        result["fg_bright_white"] = "97"
        
        result["bg_bright_black"] = "100"
        result["bg_bright_red"] = "101"
        result["bg_bright_green"] = "102"
        result["bg_bright_yellow"] = "103"
        result["bg_bright_blue"] = "104"
        result["bg_bright_magenta"] = "105"
        result["bg_bright_cyan"] = "106"
        result["bg_bright_white"] = "107"
        
        return result    
        
    def get_ansi_format_code(self, code_name):
        if code_name in self.ansi_map:
            return ANSIFormatter.get_ansi_esc(f"{self.ansi_map[code_name]}m")
        return None

class ConsoleLevelFilter(logging.Filter):
    def __init__(self, attack_params):
        self.attack_params = attack_params
    
    def filter(self, record):
        return (record.levelno >= self.attack_params.console_output_level)

class LogFileLevelFilter(logging.Filter):
    def __init__(self, attack_params):
        self.attack_params = attack_params
    
    def filter(self, record):
        return (record.levelno >= self.attack_params.log_file_output_level)

class BrokenHillLogFormatter(logging.Formatter):
    def __init__(self, fmt = None, datefmt = None, style = '{',
                 defaults = None, attack_params = None, use_ansi = False):
        super().__init__(fmt = fmt, datefmt = datefmt, style = style, validate = False, defaults = defaults)
        self.fmt = fmt
        self.datefmt = datefmt
        self.defaults = defaults        
        self.attack_params = attack_params
        self.ansi_formatter = ANSIFormatter()        
        self.use_ansi = use_ansi
        # create these once at the beginning to avoid constant calls to regenerate them
        self.ac_reset = self.ansi_formatter.get_ansi_format_code("reset") + self.ansi_formatter.get_ansi_format_code("bg_black") + self.ansi_formatter.get_ansi_format_code("fg_white")
        # self.ac_debug = self.ansi_formatter.get_ansi_format_code("fg_light_grey")
        # self.ac_info = self.ansi_formatter.get_ansi_format_code("fg_light_green")
        # self.ac_warning = self.ansi_formatter.get_ansi_format_code("fg_yellow")
        # self.ac_error = self.ansi_formatter.get_ansi_format_code("fg_red")
        # self.ac_critical = self.ansi_formatter.get_ansi_format_code("fg_red") + self.ansi_formatter.get_ansi_format_code("blink_slow")
        #self.ac_debug = self.ansi_formatter.get_ansi_format_code("bg_blue") + self.ansi_formatter.get_ansi_format_code("fg_white")
        self.ac_debug = self.ansi_formatter.get_ansi_format_code("bg_magenta") + self.ansi_formatter.get_ansi_format_code("bg_bright_magenta") + self.ansi_formatter.get_ansi_format_code("fg_white")
        self.ac_info = self.ansi_formatter.get_ansi_format_code("bg_green") + self.ansi_formatter.get_ansi_format_code("bg_bright_green") + self.ansi_formatter.get_ansi_format_code("fg_black")
        #self.ac_warning = self.ansi_formatter.get_ansi_format_code("bg_brown") + self.ansi_formatter.get_ansi_format_code("fg_white")
        self.ac_warning = self.ansi_formatter.get_ansi_format_code("bg_brown") + self.ansi_formatter.get_ansi_format_code("bg_bright_yellow") + self.ansi_formatter.get_ansi_format_code("fg_black")
        self.ac_error = self.ansi_formatter.get_ansi_format_code("bg_red") + self.ansi_formatter.get_ansi_format_code("fg_white")
        self.ac_critical = self.ansi_formatter.get_ansi_format_code("blink_slow") + self.ansi_formatter.get_ansi_format_code("bg_red") + self.ansi_formatter.get_ansi_format_code("fg_white")
        self.ac_separators = self.ansi_formatter.get_ansi_format_code("fg_light_grey") + self.ansi_formatter.get_ansi_format_code("faint")
        self.ac_debug_text = self.ansi_formatter.get_ansi_format_code("fg_light_grey")
        self.ac_normal_text = self.ansi_formatter.get_ansi_format_code("fg_white")
        self.ac_critical_text = self.ansi_formatter.get_ansi_format_code("fg_light_red")
        self.ac_timestamp = self.ansi_formatter.get_ansi_format_code("fg_light_grey")
    
    @staticmethod
    def get_short_level_name(levelno):
        if levelno == logging.DEBUG:
            return "D"
        if levelno == logging.INFO:
            return "I"
        if levelno == logging.WARNING:
            return "W"
        if levelno == logging.ERROR:
            return "E"
        if levelno == logging.CRITICAL:
            return "X"
        return "?"
    
    def get_level_ansi_code(self, levelno):
        if levelno == logging.DEBUG:
            return self.ac_debug
        if levelno == logging.INFO:
            return self.ac_info
        if levelno == logging.WARNING:
            return self.ac_warning
        if levelno == logging.ERROR:
            return self.ac_error
        if levelno == logging.CRITICAL:
            return self.ac_critical
        return self.ac_normal_text
    
    def usesTime(self):
        return True
    
# custom formatTime that uses ISO date/time formatting with milliseconds field
# BEGIN: based on https://stackoverflow.com/a/77821614
    def formatTime(self, record, datefmt = None):
        if datefmt is None:
            result = datetime.fromtimestamp(record.created).astimezone().isoformat(timespec='milliseconds')
        else:
            # BEGIN: borrowed from https://github.com/python/cpython/blob/eac41c5ddfadf52fbd84ee898ad56aedd5d90a41/Lib/logging/__init__.py#L648C9-L655C17
            ct = self.converter(record.created)
            result = time.strftime(datefmt, ct)
            # END: borrowed from https://github.com/python/cpython/blob/eac41c5ddfadf52fbd84ee898ad56aedd5d90a41/Lib/logging/__init__.py#L648C9-L655C17
        return result
# END: based on https://stackoverflow.com/a/77821614
    
    def format(self, record):
        #logger.debug(f"record = {record}")
        short_level_name = BrokenHillLogFormatter.get_short_level_name(record.levelno)
        levelname = record.levelname
        record.message = record.getMessage()
        message = record.message        
        record.asctime = self.formatTime(record, self.datefmt)
        asctime = record.asctime

        # separator_left and separator_right, with short names to avoid even messier formatting strings
        sl = "["
        sr = "]"

        if self.use_ansi:
            sl = f"{self.ac_separators}{sl}{self.ac_reset}"
            sr = f"{self.ac_separators}{sr}{self.ac_reset}"
            short_level_name = f"{self.get_level_ansi_code(record.levelno)}{short_level_name}{self.ac_reset}"
            levelname = f"{self.get_level_ansi_code(record.levelno)}{levelname}{self.ac_reset}"
            handled_text_formatting = False
            if record.levelno <= logging.DEBUG:
                message = f"{self.ac_debug_text}{message}{self.ac_reset}"
                handled_text_formatting = True
            if record.levelno >= logging.CRITICAL:
                message = f"{self.ac_critical_text}{message}{self.ac_reset}"
                handled_text_formatting = True
            if not handled_text_formatting:
                message = f"{self.ac_normal_text}{message}{self.ac_reset}"
            asctime = f"{self.ac_timestamp}{asctime}{self.ac_reset}"
        else:
            # strip any ANSI codes from the message
            message = strip_ansi_codes(message)
        
        # Get the basic values the same way as https://github.com/python/cpython/blob/d0abd0b826cfa574d1515c6f8459c9901939388f/Lib/logging/__init__.py#L477
        values = record.__dict__
        if self.defaults is not None:
            values = self.defaults | record.__dict__
        else:
            values = record.__dict__
        
        # add custom[ized] values
        values["levelname"] = levelname
        values["short_level_name"] = short_level_name
        values["sl"] = sl
        values["sr"] = sr
        values["message"] = message
        values["asctime"] = asctime
        
        # format the data the same way as https://github.com/python/cpython/blob/d0abd0b826cfa574d1515c6f8459c9901939388f/Lib/logging/__init__.py#L477
        result = self.fmt.format(**values)
        if record.exc_info:
            # Cache the traceback text to avoid converting it multiple times
            # (it's constant anyway)
            if not record.exc_text:
                record.exc_text = self.formatException(record.exc_info)
        if record.exc_text:
            if result[-1:] != "\n":
                result = result + "\n"
            result = result + record.exc_text
        if record.stack_info:
            if result[-1:] != "\n":
                result = result + "\n"
            result = result + self.formatStack(record.stack_info)
        #logger.debug(f"result = {result}")
        return result

class BrokenHillLogManager:
    def __init__(self, attack_params):
        self.attack_params = attack_params
        self.module_names = []
        self.console_formatter = None
        self.file_formatter = None
        self.console_handler = logging.StreamHandler(sys.stdout)
        self.file_handler = None
    
    def get_console_formatter(self):
        result = BrokenHillLogFormatter(
            "{sl}{asctime}{sr}{sl}{short_level_name}{sr} {message}",
            datefmt = "%Y-%m-%d@%H:%M:%S",
            attack_params = self.attack_params,
            use_ansi = self.attack_params.console_ansi_format
        )
        return result
        
    def get_log_file_formatter(self):
        result = BrokenHillLogFormatter(
            "{sl}{asctime}{sr} {sl}{funcName}{sr} {sl}{pathname}:{lineno}{sr} {sl}{levelname}{sr} {message}",
            #datefmt = "%Y-%m-%d@%H:%M:%S:uuu%z",
            attack_params = self.attack_params,
            use_ansi = self.attack_params.log_file_ansi_format
        )
        return result
    
    def initialize_handlers(self):
        self.console_formatter = self.get_console_formatter()
        self.file_formatter = self.get_log_file_formatter()
        
        if self.attack_params.log_file_path is not None:
            self.file_handler = logging.FileHandler(self.attack_params.log_file_path, mode = "a", encoding = "utf-8")
        
        self.console_handler.setFormatter(self.console_formatter)
        self.console_handler.setLevel(self.attack_params.console_output_level)
        if self.file_handler is not None:
            self.file_handler.setFormatter(self.file_formatter)
            self.console_handler.setLevel(self.attack_params.log_file_output_level)
    
    def get_lowest_log_level(self):
        result = self.attack_params.console_output_level
        if self.attack_params.log_file_output_level < result:
            result = self.attack_params.log_file_output_level
        return result
    
    def is_broken_hill_module(self, module_name):
        result = False
        module_name_lower = module_name.lower()
        if "bishopfox" in module_name_lower:
            result = True
        if "brokenhill" in module_name_lower:
            result = True
        return result
    
    def attach_handlers(self, module_name):
        self.module_names.append(module_name)
        module_logger = logging.getLogger(module_name)
        if self.console_handler is not None:
            module_logger.addHandler(self.console_handler)
        if self.file_handler is not None:
            module_logger.addHandler(self.file_handler)
        
        module_level = self.get_lowest_log_level()
        
        if not self.is_broken_hill_module(module_name):
            module_level = self.attack_params.third_party_module_output_level

        #print(f"[attach_handlers] Debug: setting log level for module '{module_name}' to {module_level}")

        module_logger.setLevel(module_level)

    def attach_handlers_to_all_modules(self):
        top_level_module_names = []
        for module_name in logging.root.manager.loggerDict:
            name_prefix = module_name.split(".")[0]
            if name_prefix not in top_level_module_names:
                top_level_module_names.append(name_prefix)
        
        #for module_name in logging.root.manager.loggerDict:
        for module_name in top_level_module_names:
            self.attach_handlers(module_name)
    
    def remove_all_existing_handlers(self):
        for module_name in logging.root.manager.loggerDict:
            module_logger = logging.getLogger(module_name)
            module_level = self.get_lowest_log_level()
            if not self.is_broken_hill_module(module_name):
                module_level = self.attack_params.third_party_module_output_level
            #print(f"[remove_all_existing_handlers] Debug: setting log level for module '{module_name}' to {module_level}")            
            module_logger.setLevel(module_level)
            
            for handler in module_logger.handlers[:]:
                module_logger.removeHandler(handler)

    def get_all_module_names(self):
        result = []
        for module_name in logging.root.manager.loggerDict:
            result.append(module_name)
        result.sort()
        return result

class ConsoleGridView:
    def __init__(self, max_table_width = None, use_ansi = True):
        # width in characters
        if max_table_width is not None:
            self.max_table_width = max_table_width
        else:
            console_columns, console_rows = os.get_terminal_size(0)
            # leave one character on each side for padding
            self.max_table_width = console_columns - 2            
            
        self.use_ansi = use_ansi
        self.ansi_formatter = ANSIFormatter()
        self.title = None
        self.column_headers = []
        self.column_widths = []
        self.row_headers = []
        self.data = []
        self.total_width = None
        self.column_separator = "|"
        
        # create these once at the beginning to avoid constant calls to regenerate them
        self.ac_reset = ""
        self.ac_title = ""
        self.ac_column_header = ""
        self.ac_row_header = ""
        self.ac_data_cell = ""
        self.ac_empty_space = ""

        if self.use_ansi:
            self.ac_reset = self.ansi_formatter.get_ansi_format_code("reset") + self.ansi_formatter.get_ansi_format_code("bg_black") + self.ansi_formatter.get_ansi_format_code("fg_white")
            self.ac_title = self.ansi_formatter.get_ansi_format_code("bg_blue") + self.ansi_formatter.get_ansi_format_code("fg_white") + self.ansi_formatter.get_ansi_format_code("bold")
            self.ac_column_header = self.ansi_formatter.get_ansi_format_code("bg_light_grey") + self.ansi_formatter.get_ansi_format_code("bg_bright_white") + self.ansi_formatter.get_ansi_format_code("fg_black") + self.ansi_formatter.get_ansi_format_code("bold")
            self.ac_row_header = self.ansi_formatter.get_ansi_format_code("bg_light_grey") + self.ansi_formatter.get_ansi_format_code("bg_bright_white") + self.ansi_formatter.get_ansi_format_code("fg_black") + self.ansi_formatter.get_ansi_format_code("bold")
            self.ac_data_cell = self.ansi_formatter.get_ansi_format_code("bg_black") + self.ansi_formatter.get_ansi_format_code("fg_white")
            self.ac_empty_space = self.ansi_formatter.get_ansi_format_code("bg_black") + self.ansi_formatter.get_ansi_format_code("fg_white")

    def set_title_colour(self, foreground_colour_code_name, background_colour_code_name):
        self.ac_title = self.ansi_formatter.get_ansi_format_code(f"bg_{background_colour_code_name}") + self.ansi_formatter.get_ansi_format_code(f"fg_{foreground_colour_code_name}") + self.ansi_formatter.get_ansi_format_code("bold")

    # data_list should be a two-dimensional list
    # first dimension is rows
    # second dimension is columns
    def set_data(self, data_list):
        if len(data_list) < 1:
            raise LoggingException("Can't process an empty list of data.")
        num_row_headers = len(self.row_headers)
        num_data_rows = len(data_list)
        if num_data_rows != num_row_headers:
            raise LoggingException(f"The number of rows in the specified data ({num_data_rows}) does not equal the number of column headers configured for this ConsoleGridView ({num_row_headers}).\nRow headers: {self.row_headers}\ndata_list: {data_list}")
        num_data_columns = len(data_list[0])
        num_column_headers = len(self.column_headers)
        if num_data_columns != num_column_headers:
            raise LoggingException(f"The number of columns in the specified data ({num_data_columns}) does not equal the number of column headers configured for this ConsoleGridView ({num_column_headers}). ")
        # reset all column widths to zero
        self.column_widths = []
        for column_num in range(0, num_data_columns):
            self.column_widths.append(0)
        
        first_data_column_index = 0
        
        # if there are row headers, those determine the width of the first column.
        # otherwise, the first column is handled just like the others.        
        if num_row_headers > 0:
            max_width = 0
            for row_num in range(0, num_row_headers):
                len_row_header = len(self.row_headers[row_num])
                if len_row_header > max_width:
                    max_width = len_row_header
            
            if max_width > 0:
                # add one more column width value to account for the row headers
                self.column_widths.append(0)
                self.column_widths[0] = max_width
                first_data_column_index = 1

        for row_num in range(0, num_data_rows):
            for column_num in range(0, num_data_columns):
                destination_column_width_index = first_data_column_index + column_num
                column_width_for_this_row = len(data_list[row_num][column_num])
                if column_width_for_this_row > self.column_widths[destination_column_width_index]:
                    self.column_widths[destination_column_width_index] = column_width_for_this_row
        
        total_width = 0
        num_columns = len(self.column_widths)
        for column_num in range(0, num_columns):
            # Add three characters for padding/separator, except for leftmost columns
            width_addition = 3
            #if column_num == 0 or column_num == (num_columns - 1):
            if column_num == 0:
                width_addition = 2
            self.column_widths[column_num] += width_addition
            total_width += self.column_widths[column_num]
        
        if total_width > self.max_table_width:
            raise LoggingException(f"The total width of the column headers for this table ({total_width}) exceeds the maximum width of the table ({self.max_table_width}). This class does not currently handle wrapping row or column headers.")
        
        self.total_width = total_width
        
        self.data = copy.deepcopy(data_list)
    
    def get_padding_to_center(self, column_width, text_to_center):        
        len_text_to_center = len(text_to_center)
        column_padding_left = int(math.floor(float(column_width - len_text_to_center) / 2.0))
        column_padding_right = column_width - (len_text_to_center + column_padding_left)
        print(f"[get_padding_to_center] Debug: column_width = {column_width}, text_to_center = '{text_to_center}', len_text_to_center = {len_text_to_center}, column_padding_left = {column_padding_left}, column_padding_right = {column_padding_right}")
        return column_padding_left, column_padding_right
    
    def get_padding_to_center_by_column(self, first_data_column_index, column_num, text_to_center):
        column_width = self.column_widths[first_data_column_index + column_num]
        return self.get_padding_to_center(column_width, text_to_center)
    
    def render_table(self):
        console_columns, console_rows = os.get_terminal_size(0)
        table_padding_total = console_columns - self.total_width
        if table_padding_total < 0:
            raise LoggingException(f"The width of the current terminal ({console_columns}) is less than the width of the table ({self.total_width}). This class does not currently handle wrapping row or column headers.")
        table_padding_left = int(math.floor(float(table_padding_total) / 2.0))
        num_rows = len(self.data)
        num_data_columns = len(self.data[0])
        if num_rows < 1:
            raise LoggingException("Can't process an empty list of data.")
        
        num_row_headers = len(self.row_headers)
        num_column_headers = len(self.column_headers)
        
        # title_padding_left = int(math.floor(float(self.total_width - len(self.title)) / 2.0))
        # tpl = " " * title_padding_left
        # title_padding_right = self.total_width - title_padding_left
        # tpr = " " * title_padding_right
        title_padding_left, title_padding_right = self.get_padding_to_center(self.total_width, self.title)
        tpl = " " * title_padding_left
        tpr = " " * title_padding_right
        title_row = f"{self.ac_title}{tpl}{self.title}{tpr}{self.ac_reset}\n"
        
        header_row = ""
        # If there are row headers, the upper-left cell is empty space
        
        first_data_column_index = 0
        
        add_separator = False
        if num_row_headers > 0:
            hr_padding = " " * (self.column_widths[0] - 1)
            header_row = f"{self.ac_empty_space}{hr_padding}"
            first_data_column_index = 1
            add_separator = True
        header_row = f"{header_row}{self.ac_reset}{self.ac_column_header}"
        
        # Add the column headers
        
        for column_num in range(0, num_column_headers):
            if add_separator:
                header_row = f"{header_row}{self.column_separator}"
            else:
                add_separator = True
            #column_padding_left = int(math.floor(float(self.column_widths[first_data_column_index + column_num] - len(self.column_headers[column_num]) - 2) / 2.0)) + 1
            #column_padding_right = self.column_widths[first_data_column_index + column_num] - column_padding_left
            column_padding_left, column_padding_right = self.get_padding_to_center_by_column(first_data_column_index, column_num, self.column_headers[column_num])
            cpl = " " * column_padding_left
            cpr = " " * column_padding_right
            header_row = f"{header_row}{cpl}{self.column_headers[column_num]}{cpr}"
        
        header_row = f"{header_row}\n"
        
        data_rows = ""
        
        # Add the data rows
        for row_num in range(0, num_rows):
            current_row = ""
            add_separator = False
            # if there is a row header, add it
            if num_row_headers > 0:
                current_row_header = self.row_headers[row_num]
                crh_left_padding = ((self.column_widths[0] - 2) - len(current_row_header)) + 1
                crhlp = " " * crh_left_padding
                padded_row_header = f"{crhlp}{current_row_header}"
                current_row = f"{current_row}{self.ac_reset}{self.ac_row_header}{padded_row_header}"
                add_separator = True
            current_row = f"{current_row}{self.ac_reset}{self.ac_data_cell}"
            for column_num in range(0, num_data_columns):                
                if add_separator:
                    current_row = f"{current_row}{self.column_separator}"
                else:
                    add_separator = True
                current_cell_text = self.data[row_num][column_num]
                column_padding_left, column_padding_right = self.get_padding_to_center_by_column(first_data_column_index, column_num, current_cell_text)
                cpl = " " * column_padding_left
                cpr = " " * column_padding_right
                current_row = f"{current_row}{cpl}{current_cell_text}{cpr}"            
            data_rows = f"{data_rows}{current_row}\n"
        
        if data_rows != "":
            return f"{title_row}{header_row}{data_rows}"
        return None