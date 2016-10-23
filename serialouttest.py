import time
import serial

class serialConnect(object):
    def __init__(self, calibrateFlag = False):
        self.ser = serial.Serial(
        port='/dev/ttyACM0',
        baudrate=9600,
        parity=serial.PARITY_ODD,
        stopbits=serial.STOPBITS_TWO,
        bytesize=serial.SEVENBITS
        )
        self.isopen = self.ser.isOpen()

    def open(self):
        self.ser.open()

    def close(self):
        self.ser.close()

    def sendSerialdata(self,packet):
        self.ser.write(str(packet)+';')
        response = ''
        numLines=0
        time.sleep(0.01)
        # wait for "#" to print output
        # while True:
        #     response += self.ser.read()
        #     if "n" in response:
        #         print(response)
        #         numLines = numLines + 1
        #     if(numLines >= 1):
        #          break 




# print 'Enter your commands below.\r\nInsert "exit" to leave the application.'

# input=1
# while 1 :
#     # get keyboard input
#     input = raw_input(">> ")
#         # Python 3 users
#         # input = input(">> ")
#     if input == 'exit':
#         ser.close()
#         exit()
#     else:
#         # send the character to the device
#         # (note that I happend a \r\n carriage return and line feed to the characters - this is requested by my device)
#         ser.write(input+'\n')
#         out = ''
#         # let's wait one second before reading output (let's give device time to answer)
#         time.sleep(1)
#         while ser.inWaiting() > 0:
#             out += ser.read(1)

#         if out != '':
#             print ">>" + out