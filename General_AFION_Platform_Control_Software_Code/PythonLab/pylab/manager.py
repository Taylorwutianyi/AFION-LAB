from . import instruments as ins
import numpy as np
import matplotlib.pyplot as plt
import time, json
from datetime import datetime


# manager
class Manager(object):
    def __init__(self, settings):
        if isinstance(settings, str):
            with open(settings, 'r') as f:
                settings = json.load(f)
        elif not isinstance(settings, list):
            raise ValueError('Settings should be filename or list')

        self.devices = {}

        # register equipments
        for instrument_settings in settings:
            self.register_instrument(instrument_settings)

    def register_instrument(self, settings):
        i_name = settings.get('name', None)
        i_type = settings.get('instrument', None)
        i_sett = settings.get('settings', {})

        if i_name is None or i_type is None:
            raise ValueError('Each instrument settings should have name and instrument type (instrument)')

        # register instrument
        self.devices[i_name] = getattr(ins, i_type)(**i_sett)

    def __getattr__(self, attr):
        device = self.devices.get(attr, None)
        if device:
            return device
        else:
            raise ValueError('No instrument')

class Logger(object):
    def __init__(self, stdout=True, logfile=None):
        self.stdout = stdout
        self.logfile = logfile

    def print(self, message, timestamp=False):
        now_time = str(datetime.now()).split('.')[0]
        if timestamp:
            message = message.format(now_time)

        if self.stdout:
            print(message)
            # _ = input(message)
        if self.logfile:
            with open(self.logfile, 'a') as f:
                f.write(message + '\n')


