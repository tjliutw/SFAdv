##########################################################################################################
# This program define FlowContainer2 class and related API.
# class FlowContainer2(self, sip, dip, sport, dport, proto, direction, length, iat)
#			add_attribute(self, direction, length, iat)
#			write_container(self, wfp)
# pack_attribute(direction, length, iat), return packed_data    # packed_data is 64bits 
# unpack_attribute(packed_data), return (direction, length, iat)
# Date: 2025.03.06
##########################################################################################################
# Format for packed data
# pack the IAT as unsigned long
# The FLOW TIMEOUT value can be assigned arbitrarily by the individual scheme e.g., 600 seconds for both TCP and UDP in
# (Aghaei-Foroushani and Zincir-Heywood, 2015).
#
# Bit 0- 3 : for direction,
#    value > 7  : FWD (same direction)
#    value <= 7 : BWD
# Bit 4-31 : for length,
#    packed length = 16 * origional length
# Bit32-63 : for Inter-arrival time
#    packed IAT = 1000000 * origional IAT

import struct
import binascii
import numpy as np

DIR_FWD = 1
DIR_BWD = 0
AttributeWide = 16		      # a helf byte => one pixel	(a packet data is 64 bits)

def pack_attribute(direction, length, iat):

    length = length * 16
    if(length > 0x0FFFFFFF):
        print('ERROR: Packet length too large')

    if(direction == DIR_FWD):
        length = length | 0xF0000000	# add dir to bit0-3
    iat = int(abs(iat) * 1000000)		# to integer
    s = struct.Struct('!I I')
    value = (length, iat)
    packed_data = s.pack(*value)

    return packed_data
# end of pack_attribute()

def unpack_attribute(packed_data):
    s = struct.Struct('!I I')
    unpacked_data = s.unpack(packed_data)
    d = unpacked_data[0] & 0xF0000000
    d = d >> 28
    if(d > 7):
        direction = DIR_FWD
    else:
        direction = DIR_BWD
    length = unpacked_data[0] & 0x0FFFFFFF		# remove first 4 bit
    length = length / 16
    iat = unpacked_data[1]
    iat = float(iat / 1000000) # to float

    return (direction, length, iat)
# end of unpack_attribute()

def attribute2image_row(attr):   # translate packed data attributes to image row (pixel array)
    value = np.zeros(AttributeWide)
    for i in range(AttributeWide):
        value[i] = float((int(attr[i], 16)) * 16 + 7)	# 將值移到中間
    return value
# end of attribute2image_row()

def image_row2attribute(pixels):   # image row (pixel array) to packed data attribute
    width = len(pixels)
    code = ''
    for i in range(width):
        if(pixels[i] < 0):
            code += '%X' % (int(0))
        elif (pixels[i] > 255):
            code += '%X' % (int(255 / 16))
        else:
            code += '%X' % (int(pixels[i] / 16))
    return code
# end of image_row2attribute()


class FlowContainer2:
    def __init__(self, sip, dip, sport, dport, proto, direction, length, iat):
        self.sip = sip
        self.dip = dip
        self.sport = sport
        self.dport = dport
        self.proto = proto
        self.attributes = [pack_attribute(direction, length, iat)]
    # end of __init__()

    def add_attribute(self, direction, length, iat):
        self.attributes.append(pack_attribute(direction, length, iat))
    # end of add_attribute()

    def write_container(self, wfp):
        wfp.write('FLOW,%s,%s,%d,%d,%d,%d,###sip:dip:sport:dport:proto:n_attributes\n' % (self.sip, self.dip, self.sport, self.dport, self.proto, len(self.attributes)))
        size = len(self.attributes)
        for i in range(size):
            wfp.write('%d,%s\n' % (i, binascii.hexlify(self.attributes[i]).decode('utf-8')))
    # end of write_container()
# end of FlowContainer2
