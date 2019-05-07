# coding: utf-8

import logging

class set_logger:
    def __init__(self):
        self.format = None
        self.handler_info = None
        self.log = None
        self.file_name = 'all_info.log'

    def initialize_logger(self):
        self.format = logging.Formatter("%(asctime)s -- %(name)s -- %(levelname)s : %(message)s")
        self.handler_info = logging.FileHandler(self.file_name, mode="a", encoding="utf-8")
        self.handler_info.setFormatter(self.format)
        self.handler_info.setLevel(logging.INFO)
        self.log = logging.getLogger(None)
        self.log.setLevel(logging.INFO)
        self.log.addHandler(self.handler_info)
        return (self.log)
