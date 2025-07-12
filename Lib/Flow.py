##########################################################################################################
# This program defines flow and calculate flow features.
# Date: 2022.01.04
##########################################################################################################

import math

DIR_FWD = 1
DIR_BWD = 0

LABEL = ['sip', 'dip', 'sport', 'dport', 'proto', 'Duration', 'Total_Packets',  'IAT_Sum', 'IAT_Mean', 'IAT_Std', 'IAT_Max', 'IAT_Min', 'Length_First', 'Length_Min', 'Length_Max', 'Length_Sum', 'Length_Mean', 'Length_Std', 'Num_Dir_Change', 'Freq_Dir_Change', 'Fwd_Total_Packets', 'Fwd_Length_First', 'Fwd_Length_Max', 'Fwd_Length_Min', 'Fwd_Length_Sum', 'Fwd_Length_Mean', 'Fwd_Length_Std', 'Fwd_IAT_Sum', 'Fwd_IAT_Mean', 'Fwd_IAT_Std', 'Fwd_IAT_Max', 'Fwd_IAT_Min', 'Bwd_Total_Packets', 'Bwd_Length_First', 'Bwd_Length_Max', 'Bwd_Length_Min', 'Bwd_Length_Sum', 'Bwd_Length_Mean', 'Bwd_Length_Std', 'Bwd_IAT_Sum', 'Bwd_IAT_Mean', 'Bwd_IAT_Std', 'Bwd_IAT_Max', 'Bwd_IAT_Min', 'Label']

def LABEL_string():
    outstr = ''
    for i in range(len(LABEL)-1):
        outstr += (LABEL[i] + ',')
    outstr += LABEL[len(LABEL)-1] + '\n'
    return outstr
# end of LABEL_string()

class Packet:
    def __init__(self, time, sip, dip, proto, length, sport, dport):
        try:
            self.time = float(time)
            self.sip = sip.replace(' ', '')		# skip ' '
            self.dip = dip.replace(' ', '')
            self.sport = int(sport)
            self.dport = int(dport)
            self.proto = int(proto)
            self.length = int(length)
            return
        except ValueError:
            self.length = -1
            return
# end of class Packet

class Flow:
    def __init__(self, time, sip, dip, sport, dport, proto, length, segment_len):		# initialization & add first packet
        self.time = float(time)
        self.sip = sip
        self.dip = dip
        self.sport = sport
        self.dport = dport
        self.proto = proto
        self.starttime = float(time)
        self.endtime = float(time)
        self.Duration = self.endtime - self.starttime		
        self.Total_Packets = 1
        self.segment_len = segment_len
        
        # temp variable for standard devision
        self.Length_sqr = float(length * length)
        self.IAT_sqr = float(0)
        self.Fwd_Length_sqr = float(length * length)
        self.Fwd_IAT_sqr = float(0)
        self.Bwd_Length_sqr = float(0)
        self.Bwd_IAT_sqr = float(0)
        
        # direction feature
        self.last_dir = DIR_FWD
        self.Num_Dir_Change = 0
        self.Freq_Dir_Change = float(0)
        
        # interval features
        self.last_packet_time = float(time)
        self.IAT_Sum = float(0)
        self.IAT_Mean = float(0)
        self.IAT_Std = float(0)
        self.IAT_Max = float(0)
        self.IAT_Min = float(0)
        
        # length features
        self.Length_First = length
        self.Length_Min = length
        self.Length_Max = length
        self.Length_Sum = length
        self.Length_Mean = float(length)
        self.Length_Std = float(0)
        
        # Fwd features
        self.fwd_last_packet_time = float(time)
        self.Fwd_Total_Packets = 1
        self.Fwd_Length_First = length
        self.Fwd_Length_Max = length
        self.Fwd_Length_Min = length
        self.Fwd_Length_Sum = length
        self.Fwd_Length_Mean = float(length)
        self.Fwd_Length_Std = float(0)
        self.Fwd_IAT_Sum = float(0)
        self.Fwd_IAT_Mean = float(0)
        self.Fwd_IAT_Std = float(0)
        self.Fwd_IAT_Max = float(0)
        self.Fwd_IAT_Min = float(0) 
        
        # Bwd features
        self.bwd_last_packet_time = float(-1)
        self.Bwd_Total_Packets = 0
        self.Bwd_Length_First = 0
        self.Bwd_Length_Max = 0
        self.Bwd_Length_Min = 0
        self.Bwd_Length_Sum = 0
        self.Bwd_Length_Mean = float(0)
        self.Bwd_Length_Std = float(0)
        self.Bwd_IAT_Sum = float(0)
        self.Bwd_IAT_Mean = float(0)
        self.Bwd_IAT_Std = float(0)
        self.Bwd_IAT_Max = float(0)
        self.Bwd_IAT_Min = float(0)
        return
    # end of __init__()

    def is_same_flow(self, sip, dip, sport, dport, proto):
        if((self.sip == sip) and (self.dip == dip) and (self.sport == sport) and \
            (self.dport == dport) and (self.proto == proto)):
            result = True
            direction = DIR_FWD		# DIR_FWD: sip -> dip, DIR_BWD: dip -> sip
        elif((self.sip == dip) and (self.dip == sip) and (self.sport == dport) and \
            (self.dport == sport) and (self.proto == proto)):
            result = True
            direction = DIR_BWD
        else:
            result = False
            direction = -2
        return result, direction
    # end of is_same_flow()

    def add_packet(self, time, sip, dip, sport, dport, proto, length, direction):
        # length features
        if self.segment_len == 0:
            pass
        elif(time-self.starttime) > self.segment_len:
            return

        self.endtime = float(time)
        self.Duration = self.endtime - self.starttime		
        self.Total_Packets += 1

        # length features
        if(length > self.Length_Max):
            self.Length_Max = length
        if(length < self.Length_Min):
            self.Length_Min = length
        self.Length_Sum += length
        self.Length_Mean = float(self.Length_Sum / self.Total_Packets)
        self.Length_sqr += float(length * length)
        self.Length_Std = math.sqrt(self.Length_sqr/self.Total_Packets -  self.Length_Mean*self.Length_Mean)
                
        # interval features
        interval = float(time - self.last_packet_time)
        self.last_packet_time = float(time)
        if(self.IAT_Max == 0):	# the second packet
            self.IAT_Max = interval
            self.IAT_Min = interval
        else:
            if(interval > self.IAT_Max):			self.IAT_Max = interval
            if(interval < self.IAT_Min):			self.IAT_Min = interval
        self.IAT_Sum += interval
        self.IAT_Mean = self.IAT_Sum / (self.Total_Packets - 1)
        self.IAT_sqr += float(interval * interval)
        try:
            self.IAT_Std = math.sqrt(self.IAT_sqr/(self.Total_Packets-1) -  self.IAT_Mean*self.IAT_Mean)
        except:
            self.IAT_Std = math.sqrt(-1*(self.IAT_sqr/(self.Total_Packets-1) -  self.IAT_Mean*self.IAT_Mean))

        # direction feature
        if(self.last_dir != direction):
            self.Num_Dir_Change += 1
            self.last_dir = direction
        self.Freq_Dir_Change = float(self.Num_Dir_Change / (self.Total_Packets - 1))

        if(direction == DIR_FWD):
            # Fwd features
            self.Fwd_Total_Packets += 1
            if(length > self.Fwd_Length_Max):		self.Fwd_Length_Max = length
            if(length < self.Fwd_Length_Min):		self.Fwd_Length_Min = length
            self.Fwd_Length_Sum += length
            self.Fwd_Length_Mean = float(self.Fwd_Length_Sum / self.Fwd_Total_Packets)
            self.Fwd_Length_sqr += float(length * length)
            self.Fwd_Length_Std = math.sqrt(self.Fwd_Length_sqr/self.Fwd_Total_Packets -  self.Fwd_Length_Mean*self.Fwd_Length_Mean)
            fwd_interval = float(time - self.fwd_last_packet_time)
            self.fwd_last_packet_time = float(time)
            if(self.Fwd_IAT_Max == 0): 		# the second fwd packet
                self.Fwd_IAT_Max = fwd_interval
                self.Fwd_IAT_Min = fwd_interval
            else:
                if(fwd_interval > self.Fwd_IAT_Max):		self.Fwd_IAT_Max = fwd_interval
                if(fwd_interval < self.Fwd_IAT_Min):		self.Fwd_IAT_Min = fwd_interval
            self.Fwd_IAT_Sum += fwd_interval
            self.Fwd_IAT_Mean = self.Fwd_IAT_Sum / (self.Fwd_Total_Packets - 1)
            self.Fwd_IAT_sqr += float(fwd_interval * fwd_interval)
            try:
                self.Fwd_IAT_Std = math.sqrt(self.Fwd_IAT_sqr/(self.Fwd_Total_Packets-1) -  self.Fwd_IAT_Mean*self.Fwd_IAT_Mean)
            except:
                self.Fwd_IAT_Std = math.sqrt(-1*(self.Fwd_IAT_sqr/(self.Fwd_Total_Packets-1) -  self.Fwd_IAT_Mean*self.Fwd_IAT_Mean))
        else:			# direction = DIR_BWD
            # Bwd features
            if(self.Bwd_Total_Packets == 0):	# first bwd packet
                self.Bwd_Total_Packets = 1
                self.Bwd_Length_First = length
                self.Bwd_Length_Max = length
                self.Bwd_Length_Min = length
                self.Bwd_Length_Sum = length
                self.Bwd_Length_Mean = float(length)
                self.Bwd_Length_sqr = float(length * length)
                self.Bwd_Length_Std = float(0)
                # Bwd_IAT_Sum, Bwd_IAT_Mean, Bwd_IAT_Std, Bwd_IAT_Max, Bwd_IAT_Min are all 0 when Flow is created
                self.bwd_last_packet_time = float(time)
            else:
                self.Bwd_Total_Packets += 1
                if(length > self.Bwd_Length_Max):		self.Bwd_Length_Max = length
                if(length < self.Bwd_Length_Min):		self.Bwd_Length_Min = length
                self.Bwd_Length_Sum += length
                self.Bwd_Length_Mean = float(self.Bwd_Length_Sum / self.Bwd_Total_Packets)
                self.Bwd_Length_sqr += float(length * length)
                self.Bwd_Length_Std = math.sqrt(self.Bwd_Length_sqr/self.Bwd_Total_Packets -  self.Bwd_Length_Mean*self.Bwd_Length_Mean)
                bwd_interval = float(time - self.bwd_last_packet_time)
                self.bwd_last_packet_time = float(time)
                if(self.Bwd_IAT_Max == 0): 		# the second fwd packet
                    self.Bwd_IAT_Max = bwd_interval
                    self.Bwd_IAT_Min = bwd_interval
                else:
                    if(bwd_interval > self.Bwd_IAT_Max):		self.Bwd_IAT_Max = bwd_interval
                    if(bwd_interval < self.Bwd_IAT_Min):		self.Bwd_IAT_Min = bwd_interval
                self.Bwd_IAT_Sum += bwd_interval
                self.Bwd_IAT_Mean = self.Bwd_IAT_Sum / (self.Bwd_Total_Packets - 1)
                self.Bwd_IAT_sqr += float(bwd_interval * bwd_interval)
                try:
                    self.Bwd_IAT_Std = math.sqrt(self.Bwd_IAT_sqr/(self.Bwd_Total_Packets-1) -  self.Bwd_IAT_Mean*self.Bwd_IAT_Mean)
                except:
                    self.Bwd_IAT_Std = math.sqrt(-1*(self.Bwd_IAT_sqr/(self.Bwd_Total_Packets-1) -  self.Bwd_IAT_Mean*self.Bwd_IAT_Mean))
                
        return interval
    # end of add_packet()

    def flow_feature(self): 
        features = self.sip + "," + self.dip + "," + str(self.sport) + "," + str(self.dport) + "," + str(self.proto) + "," \
                + str(self.Duration) + "," +  str(self.Total_Packets) + "," + str(self.IAT_Sum) + "," + str(self.IAT_Mean) + "," + str(self.IAT_Std) + "," + str(self.IAT_Max) + "," + str(self.IAT_Min) + "," + str(self.Length_First) + ","\
                + str(self.Length_Min) + "," + str(self.Length_Max) + "," + str(self.Length_Sum) + "," + str(self.Length_Mean) + "," + str(self.Length_Std) + "," + str(self.Num_Dir_Change) + "," + str(self.Freq_Dir_Change) + ","\
                + str(self.Fwd_Total_Packets) + "," + str(self.Fwd_Length_First) + "," + str(self.Fwd_Length_Max) + "," + str(self.Fwd_Length_Min) + "," + str(self.Fwd_Length_Sum) + "," + str(self.Fwd_Length_Mean) + ","\
                + str(self.Fwd_Length_Std) + "," + str(self.Fwd_IAT_Sum) + "," + str(self.Fwd_IAT_Mean) + "," + str(self.Fwd_IAT_Std) + "," + str(self.Fwd_IAT_Max) + "," + str(self.Fwd_IAT_Min) + ","\
                + str(self.Bwd_Total_Packets) + "," + str(self.Bwd_Length_First) + "," + str(self.Bwd_Length_Max) + "," + str(self.Bwd_Length_Min) + "," + str(self.Bwd_Length_Sum) + "," + str(self.Bwd_Length_Mean) + ","\
                + str(self.Bwd_Length_Std) + "," + str(self.Bwd_IAT_Sum) + "," + str(self.Bwd_IAT_Mean) + "," +  str(self.Bwd_IAT_Std) + "," + str(self.Bwd_IAT_Max) + ","  + str(self.Bwd_IAT_Min) + ","\
                + str('\n')
        return features
    # end of flow_feature()
    
    def feature_list(self):
        L = {'sip' : self.sip, 'dip' :  self.dip, 'sport' : self.sport, 'dport' : self.dport, 'proto' : self.proto,
                'Duration' : self.Duration,   'Total_Packets' : self.Total_Packets, 'IAT_Sum' : self.IAT_Sum, 'IAT_Mean' : self.IAT_Mean, 'IAT_Std' : self.IAT_Std,
                'IAT_Max' : self.IAT_Max,  'IAT_Min' : self.IAT_Min, 'Length_First' : self.Length_First, 'Length_Min' : self.Length_Min, 'Length_Max' : self.Length_Max,
                'Length_Sum' : self.Length_Sum, 'Length_Mean' : self.Length_Mean, 'Length_Std' : self.Length_Std, 'Num_Dir_Change' : self.Num_Dir_Change,
                'Freq_Dir_Change' : self.Freq_Dir_Change, 'Fwd_Total_Packets' : self.Fwd_Total_Packets, 'Fwd_Length_First' : self.Fwd_Length_First,
                'Fwd_Length_Max' : self.Fwd_Length_Max, 'Fwd_Length_Min' : self.Fwd_Length_Min, 'Fwd_Length_Sum' : self.Fwd_Length_Sum,
                'Fwd_Length_Mean' : self.Fwd_Length_Mean,  'Fwd_Length_Std' : self.Fwd_Length_Std, 'Fwd_IAT_Sum' : self.Fwd_IAT_Sum,
                'Fwd_IAT_Mean' : self.Fwd_IAT_Mean, 'Fwd_IAT_Std' : self.Fwd_IAT_Std, 'Fwd_IAT_Max' : self.Fwd_IAT_Max, 'Fwd_IAT_Min' : self.Fwd_IAT_Min,
                'Bwd_Total_Packets' : self.Bwd_Total_Packets, 'Bwd_Length_First' : self.Bwd_Length_First, 'Bwd_Length_Max' : self.Bwd_Length_Max,
                'Bwd_Length_Min' : self.Bwd_Length_Min, 'Bwd_Length_Sum' : self.Bwd_Length_Sum, 'Bwd_Length_Mean' : self.Bwd_Length_Mean,
                'Bwd_Length_Std' : self.Bwd_Length_Std, 'Bwd_IAT_Sum' : self.Bwd_IAT_Sum, 'Bwd_IAT_Mean' : self.Bwd_IAT_Mean,
                'Bwd_IAT_Std' : self.Bwd_IAT_Std, 'Bwd_IAT_Max' : self.Bwd_IAT_Max, 'Bwd_IAT_Min' : self.Bwd_IAT_Min}
        return L
    # end of feature_list()
#end of class Flow

