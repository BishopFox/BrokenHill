#!/bin/env python

import enum
import logging
import sys
    
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
        self.ac_debug = self.ansi_formatter.get_ansi_format_code("fg_light_grey")
        self.ac_info = self.ansi_formatter.get_ansi_format_code("fg_light_green")
        self.ac_warning = self.ansi_formatter.get_ansi_format_code("fg_yellow")
        self.ac_error = self.ansi_formatter.get_ansi_format_code("fg_red")
        self.ac_critical = self.ansi_formatter.get_ansi_format_code("fg_red") + self.ansi_formatter.get_ansi_format_code("blink_fast")
        self.ac_separators = self.ansi_formatter.get_ansi_format_code("fg_light_grey") + self.ansi_formatter.get_ansi_format_code("faint")
        self.ac_debug_text = self.ansi_formatter.get_ansi_format_code("fg_light_grey")
        self.ac_normal_text = self.ansi_formatter.get_ansi_format_code("fg_white")
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
            return "C"
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
    
    def format(self, record):
        #print(f"[BrokenHillLogFormatter.format] Debug: record = {record}")
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
            if record.levelno <= logging.DEBUG:
                message = f"{self.ac_debug_text}{message}{self.ac_reset}"
            else:
                message = f"{self.ac_normal_text}{message}{self.ac_reset}"
            asctime = f"{self.ac_timestamp}{asctime}{self.ac_reset}"
        
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
        #print(f"[BrokenHillLogFormatter.format] Debug: result = {result}")
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
            "{sl}{asctime}{sr} {sl}{short_level_name}{sr} {message}",
            datefmt = "%Y-%m-%d@%H:%M:%S",
            attack_params = self.attack_params,
            use_ansi = self.attack_params.console_ansi_format
        )
        return result
        
    def get_log_file_formatter(self):
        result = BrokenHillLogFormatter(
            "{sl}{asctime}{sr} {sl}{pathname}:{lineno}{sr} {sl}{funcName}{sr} {sl}{levelname}{sr} {message}",
            datefmt = "%Y-%m-%d@%H:%M:%S:uuu%z",
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
    
    def attach_handlers(self, module_name):
        self.module_names.append(module_name)
        logger = logging.getLogger(module_name)
        if self.console_handler is not None:
            logger.addHandler(self.console_handler)
        if self.file_handler is not None:
            logger.addHandler(self.file_handler)

    def attach_handlers_to_all_modules(self):
        for name in logging.root.manager.loggerDict:
            self.attach_handlers(name)
    
    def remove_all_existing_handlers(self):
        for name in logging.root.manager.loggerDict:
            logger = logging.getLogger(name)
            for handler in logger.handlers[:]:
                logger.removeHandler(handler)

    def get_all_module_names(self):
        result = []
        for name in logging.root.manager.loggerDict:
            result.append(name)
        result.sort()
        return result