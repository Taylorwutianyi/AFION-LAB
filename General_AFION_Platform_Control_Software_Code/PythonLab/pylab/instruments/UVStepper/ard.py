import serial
import time
import numpy as np

class ard:
    def __init__(self, port):
        # self.rm = visa.ResourceManager()
        # self.inst = self.rm.open_resource(visaport, baud_rate=9600)

        self.ser = serial.Serial(port, baudrate=9600, timeout=.2)
        while (self.ser.in_waiting == 0):
            pass
        if self.ser.read(size=10) == b'Ready':
            # print('Ard ready')
            self.ser.reset_input_buffer()

    def turnon(self):
        self.ser.write('on'.encode())
        # print (self.ser.read(size=20))

    def turnoff(self):
        self.ser.write('off'.encode())
        # print (self.ser.read(size=20))

    def moveto(self, height):
        move = str(np.around(8 - height, decimals = 1)) 
        print("stepper motor " + move)
        self.ser.write(move.encode())

    def moveback(self, height):
        move = str(np.around(height - 8, decimals = 1))
        print("stepper motor " + move)
        self.ser.write(move.encode())

    def movetransit(self, height1,height2):
        move = str(height1 - height2)
        print("stepper motor " + move)
        self.ser.write(move.encode())
