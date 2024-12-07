from .. import VisaInstrument, CmdNameMap, mapsetmethod, mapgetmethod, rangemethod, add_set_get
from .pump import AdvancedPump
import pyvisa.constants as pv_const
import time


TYPE_TO_NUM = [3, 4, 3, 8, 4, 6, 0, 6]
MIN_V = 5
MAX_V = 5800
MIN_STOP_V = 50
MAX_STOP_V = 2700
MIN_START_V = 50
MAX_START_V = 900
MIN_A = 1
MAX_A = 40

OUTLET_VALVE = 3
SYR_VOL = 5e-5

CMD_RES_MAP = CmdNameMap([
    (0, 'STANDARD'),
    (1, 'HIGH'), ])


@add_set_get
class PumpPSD(VisaInstrument, AdvancedPump):

    def __init__(self, visa, addr, init_valve = OUTLET_VALVE, syringe_volume = SYR_VOL, ports = {}, psdtype = 'psd8', **kwargs):
        self.addr = str(addr)
        self.header = '/' + chr(int(self.addr, 16) + 49)

        # rs232 settings
        self.rs232_settings = {
            'baud_rate': 9600,
            'stop_bits': pv_const.StopBits.one,
            'parity'   : pv_const.Parity.none,
            'data_bits': 8,
            'read_termination': '\x03\r\n',
            'timeout': 5000,
        }
        super().__init__(visa, **self.rs232_settings, **kwargs)

        # initialize
        # self.init_syringe(init_valve, syringe_volume, 'psd8')
        self.ports = ports
        self.psdtype = psdtype

    def init_syringe(self, init_valve, syringe_volume, psdtype):

        self.write('W1')

        self.set_resolution('STANDARD') # standard resolution
        self.set_acceleration(1) # minimum acceleration

        # offset and backlash
        self.write('k0') #back-off steps (initialize)
        if psdtype == 'psd8':
            self.write('K16') #return steps
        else:
            self.write('K0') #return steps

        # init
        self.write('h30001') # enable h commands



        # h settings
        self.write('h30001') # enable h commands
        self.write('h20000') # intialize valve
        self.write('h20001') # enable valve movement
        self.write('h23001') # enable valve shortest movement

        # syringe volume
        self.syringe_volume = syringe_volume
        self.set_valve(init_valve)
        self.psdtype = psdtype

    # ---- override methods ----
    def write(self, command):
        return self.ask(command)

    def _ask(self, command):
        self.manager.write(self.header + command + 'R\n')
        return self.read()

    def busy(self):
        _, status = self._ask('Q')
        return (status & 32) == 0

    def ask(self, command):
        if command:
            data, status = self._ask(command)
            while self.busy():
                time.sleep(0.1)
            return data, status

    def read(self):
        response = self.manager.read_raw()
        return response[3:-3], response[2]



    # ---- syringe commands ----
    def get_max_steps(self):
        resolution = self.get_resolution()
        if self.psdtype == 'psd8':
            if resolution == 'STANDARD':
                return 3000
            elif resolution == 'HIGH':
                return 24000
        elif self.psdtype == 'psd6':
            if resolution == 'STANDARD':
                return 13714
            elif resolution == 'HIGH':
                return 109714

    @mapgetmethod(CMD_RES_MAP)
    def get_resolution(self):
        try:
            response = int(self.ask('?11000')[0])
        except:
            response = 0
        return (response & 1)

    @mapsetmethod(CMD_RES_MAP)
    def set_resolution(self, cmd):
        self.write('N{:d}'.format(cmd))

    def get_position(self):
        return int(self.ask('?')[0])

    @rangemethod(0, 'get_max_steps')
    def set_position(self, position):
        self.write('A{:d}'.format(round(position)))


    # ---- valve commands ----
    def get_valve_numbers(self):
        num = self.get_valve_type()
        return TYPE_TO_NUM[num]

    def get_valve_type(self):
        return int(self.ask('?21000')[0])

    @rangemethod(0, 7, dtype=int)
    def set_valve_type(self, valve_type):
        self.write('h2100{:d}'.format(valve))

    def get_valve(self):
        return int(self.ask('?24000')[0])

    @rangemethod(1, 'get_valve_numbers', int)
    def set_valve(self, valve):
        self.write('h2600{:d}'.format(valve))

    def full_rotation(self):
        valve = int(self.ask('?24000')[0])
        if valve == 1:
            self.write('h2400{:d}'.format(8))
        else:
            self.write('h2400{:d}'.format(valve - 1))
        self.write('h2400{:d}'.format(valve))



    # ---- motor control commands ----
    def get_velocity(self):
        return int(self.ask('?2')[0])

    @rangemethod(MIN_V, MAX_V)
    def set_velocity(self, velocity):
        self.write('V{:d}'.format(velocity))

    def get_start_velocity(self):
        return int(self.ask('?1')[0])

    @rangemethod(MIN_START_V, MAX_START_V)
    def set_start_velocity(self, velocity):
        self.write('v{:d}'.format(velocity))

    def get_stop_velocity(self):
        return int(self.ask('?3')[0])

    @rangemethod(MIN_STOP_V, MAX_STOP_V)
    def set_stop_velocity(self, velocity):
        self.write('c{:d}'.format(velocity))

    @rangemethod(MIN_A, MAX_A, int)
    def set_acceleration(self, acceleration):
        self.write('L{:d}'.format(acceleration))

    # ---- pump commands ----

    # def get_syringe_volume(self):
    #     return self.syringe_volume
    #
    # def set_syringe_volume(self, syringe_volume):
    #     self.syringe_volume = syringe_volume
    #
    # def draw(self, volume, valve = None):
    #     if valve is not None:
    #         self.set_valve(valve)
    #     if volume > 0:
    #         position = self.get_position() + int( (volume / self.syringe_volume) * self.get_max_steps() )
    #         self.set_position(position)
    #
    # def draw_full(self, valve = None):
    #     if valve is not None:
    #         self.set_valve(valve)
    #     self.set_position(self.get_max_steps())
    #
    # def dispense(self, volume, valve = None):
    #     if valve is not None:
    #         self.set_valve(valve)
    #     if volume > 0:
    #         position = self.get_position() - int( (volume / self.syringe_volume) * self.get_max_steps() )
    #         self.set_position(position)
    #
    # def dispense_all(self, valve = None):
    #     if valve is not None:
    #         self.set_valve(valve)
    #     self.set_position(0)
    #
    # def draw_and_dispense(self, draw_valve, dispense_valve, volume):
    #     while volume >= self.syringe_volume:
    #         volume -= self.syringe_volume
    #         self.draw_full(draw_valve)
    #         self.dispense_all(dispense_valve)
    #     self.draw(valve = draw_valve, volume = volume)
    #     self.dispense_all(dispense_valve)
