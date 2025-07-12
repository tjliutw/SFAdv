##########################################################################################################
# The dataset module reads container file (flowContainer2) and builds dataset
# The containerDataset class reads input container file and returns (container, feature_vector) by using load_data() API.
#		containerDataset(self, container_file, max_num_packets)
#		load_data(self, ), return (self.flow_container, self.feature)
# The clusterDataset class is for training GAN, it filters flows based on the HDBSCAN model trained by source class
#		clusterDataset(self, container_files, labels, max_num_packets, max_num_flows = MaxNumFlows, min_cluster_size = MinClusterSize)
# image2feature(image, label), return flag, flow_code, features, attribute_string
# Date: 2025.03.06
##########################################################################################################
import sys
from io import StringIO
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import Flow
import FlowContainer2
import hdbscan
import joblib			# for saving model


# constant for container image
AttributeWide = FlowContainer2.AttributeWide
END_CODE = ''
for i in range(AttributeWide):
    END_CODE += '0'

NumFeatures = 39
MaxNumFlows = 1000000

MinClusterSize = 110
HDB_Metrics='euclidean'

# constant for reading container file
FLOW = 1
ATTRIBUTE = 2

# type of features
ALL = -1

# If pd.append() is used, the performance is bad. Thus, write to temp file is used
TEMPFILE = 'temp.csv'
#NormalizedModelFile = 'NormalizedModel.csv'

LABEL = Flow.LABEL

def GetFeature(data):
    return pd.DataFrame([
        data["Duration"],				data["Total_Packets"],				data["IAT_Sum"], 					data["IAT_Mean"],
        data["IAT_Std"], 				data["IAT_Max"], 					data["IAT_Min"], 					data["Length_First"],
        data["Length_Min"], 			data["Length_Max"], 				data["Length_Sum"],					data["Length_Mean"],
        data["Length_Std"], 			data["Num_Dir_Change"],				data["Freq_Dir_Change"], 			data["Fwd_Total_Packets"],
        data["Fwd_Length_First"], 		data["Fwd_Length_Max"], 			data["Fwd_Length_Min"], 			data["Fwd_Length_Sum"],
        data["Fwd_Length_Mean"], 	    data["Fwd_Length_Std"], 			data["Fwd_IAT_Sum"], 				data["Fwd_IAT_Mean"],
        data["Fwd_IAT_Std"], 			data["Fwd_IAT_Max"], 				data["Fwd_IAT_Min"], 				data["Bwd_Total_Packets"],
        data["Bwd_Length_First"], 		data["Bwd_Length_Max"], 			data["Bwd_Length_Min"], 			data["Bwd_Length_Sum"],
        data["Bwd_Length_Mean"], 	    data["Bwd_Length_Std"], 			data["Bwd_IAT_Sum"], 				data["Bwd_IAT_Mean"],
        data["Bwd_IAT_Std"], 			data["Bwd_IAT_Max"], 				data["Bwd_IAT_Min"]
        ]).T
# end of GetFeature()

# def StandardScaler(data, training=True):
# 	MEAN = 0		# for array index
# 	STD = 1			# for array index
# 	if(training):
# 		mean_std = np.zeros((2, len(LABEL)))
# 		for i in range(5, 44):
# 			mean_std[MEAN][i] = data[LABEL[i]].mean()
# 			mean_std[STD][i] = data[LABEL[i]].std()
# 		mean_std.tofile(NormalizedModelFile, sep=',')
# 	else:
# 		#print('Read saved model: mean_std.csv');
# 		mean_std = np.loadtxt(NormalizedModelFile, dtype=np.float64, delimiter=',')
# 		mean_std = mean_std.reshape((2, len(LABEL)))
#
# 	for i in range(5, 44):	# From Duration to Bwd_IAT_Min
# 		data[LABEL[i]] = (data[LABEL[i]] - mean_std[MEAN][i]) / mean_std[STD][i]
# 	return data
# # end of StandardScaler()

# The containerDataset class reads container file and returns (flow_container, feature)
# max_num_packets: we only condiser first max_num_packets packets in a flow
class containerDataset:
    def __init__(self, container_file, max_num_packets, max_num_flows = MaxNumFlows):
        self.max_num_packets = max_num_packets
        self.max_num_flows = max_num_flows
        self.n_flows = 0
        self.flow_container = []
        self.read_file(container_file)			# will write features to TEMPFILE & build flow_container
        self.feature = pd.read_csv(TEMPFILE)	# read features from TEMPFILE
    # end of __init__():

    def load_data(self, ):
        return (self.flow_container, self.feature)
    # end of load_data()

    def read_file(self, container_file):
        idx = 0
        container = np.zeros((self.max_num_packets, AttributeWide)) # a container is an image (self.max_num_packets x AttributeWide)
        role = FLOW
        sip = ''
        dip = ''
        sport = 0
        dport = 0
        proto = 0
        n_attributes = 0
        first_packet = True
        time = 0
        flow = []
        n_flows = 0

        ifp = open(container_file, 'r')
        ofp = open(TEMPFILE, 'w')

        # write labels
        outstr = ''
        for i in range(len(LABEL)-1):
            outstr += (LABEL[i] + ',')
        outstr += LABEL[len(LABEL)-1] + '\n'
        ofp.write(outstr)

        for line in ifp:
            line = line.replace(' ', '')	# skip ' '

            if(role == FLOW):		# initialize flow info.
                if(n_flows >= self.max_num_flows):
                    print('The system only consider %d flows. The remaining flows are discared' % (n_flows))
                    break			# Here, we conly consider first MaxNumFlows flows
                n_flows += 1
                token = line.split(',')
                if(token[0] != 'FLOW'):
                    print('ERROR: NOT FLOW: %s' % (line))
                    continue
                sip = token[1]
                dip = token[2]
                sport = int(token[3])
                dport = int(token[4])
                proto = int(token[5])
                n_attributes = int(token[6])		# number of attribute
                idx = 0
                container = np.zeros((self.max_num_packets, AttributeWide))
                role = ATTRIBUTE
                first_packet = True
                time = 0
            else:							# process packet attribute
                token = line.split(',')
                # Calculate flow features
                (direction, length, iat) = FlowContainer2.unpack_attribute(bytes.fromhex(token[1]))
                time += iat
                if(first_packet):
                    flow = Flow.Flow(time, sip, dip, sport, dport, proto, length, 0)
                    first_packet = False
                else:
                    flow.add_packet(time, sip, dip, sport, dport, proto, length, direction)
                # end of if

                idx += 1
                if(idx == n_attributes):		# at the end of a flow
                    role = FLOW
                    # write feature to TEMPFILE
                    ofp.write(flow.flow_feature())
                if(idx > self.max_num_packets):
                    # skip attributes
                    continue

                # fill flow attribute (a row of image) into flow_container (a image)
                attribute = FlowContainer2.attribute2image_row(token[1])
                for i in range(AttributeWide):
                    container[idx-1][i] = attribute[i]
                if((idx == n_attributes) or (idx == (self.max_num_packets))): # the last attribute of a flow
                    self.flow_container.append(container)
                    self.n_flows += 1
            # end if role
        # end for line
        self.flow_container = np.array(self.flow_container)
        ifp.close()
        ofp.close()
    # end of load_file()
# end of class containerDataset

# The clusterDatset filters flows based on the HDBSCAN model
# container_files[0] is the source container
class clusterDataset:
    def __init__(self, container_files, labels, max_num_packets, max_num_flows = MaxNumFlows, min_cluster_size = MinClusterSize):
        self.imgHigh = max_num_packets
        self.imgWidth = AttributeWide
        self.max_num_packets = max_num_packets
        self.max_num_flows = max_num_flows
        self.nFiles = len(container_files)

        # read source container
        data = containerDataset(container_files[0], max_num_packets, max_num_flows)
        self.containers, self.features = data.load_data()					# NOTE: self.features is a panda type
        self.Y = np.full((len(self.containers), ), labels[0])

        # read target containers
        for i in range(1, self.nFiles):
            data = containerDataset(container_files[i], max_num_packets, max_num_flows)
            container, feature = data.load_data()								# NOTE: self.features is a panda type
            y = np.full((len(container), ), labels[i])
            self.containers = np.append(self.containers, container, axis=0)
            self.features = pd.concat([self.features, feature])			# NOTE: self.features is a panda type
            self.Y = np.append(self.Y, y, axis=0)
        # end for

        self.containers = (self.containers - 127.5) / 127.5		#scale [0..255] to [-1..1]

        # reset index for self.features (after concat)
        self.features = self.features.reset_index(drop=True)

        # normalization
        #self.features = StandardScaler(self.features, True)

        # cluster feature vectors
        X = GetFeature(self.features)
        self.hdb = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=1, metric=HDB_Metrics, prediction_data=True)
        self.hdb.fit(X)

        self.features['DB_Label'] = self.hdb.labels_
        self.features['Label'] = self.Y
        self.features.to_csv('clustered_feature.csv')

        # skip noise flows
        self.features = self.features[self.features['DB_Label'] >= 0]
        self.containers = self.containers[self.features.index]
        self.Y = self.Y[self.features.index]

        self.clusterNum = self.features['DB_Label'].to_numpy()
        self.nClusters = np.max(self.clusterNum) + 1

        self.clusterLabel = np.zeros(self.nClusters)		# record the labelY of the cluster
        # re-assign self.Y.  self.Y in each cluster has the same value
        for i in range(self.nClusters):
            idx = np.where(self.clusterNum == i)
            y = self.Y[idx]
            c = np.bincount(y)	# count the number of each value
            val = np.argmax(c)
            self.Y[idx] = val
            self.clusterLabel[i] = val
        # end of for

        self.features = GetFeature(self.features)
        self.features = np.array(self.features)
    # end of __init__():

    # The input is an array of image
    # The output is the clusterNum of the images
    def hdb_predict(self, images):
        features = self.images2feature(images)
        #features = StandardScaler(features, False)
        X = GetFeature(features)
        pred_cluster, strengths = hdbscan.approximate_predict(self.hdb, X)
        labels = np.zeros(len(pred_cluster))
        for i in range(len(pred_cluster)):
            if(pred_cluster[i] >= 0):
                labels[i] = self.clusterLabel[pred_cluster[i]]
            else:
                labels[i] = -1

        return pred_cluster, labels, strengths
    # end of hdb_predict()

    def num_clusters(self, ):
        return self.nClusters
    # end of num_classes()

    def image_size(self, ):
        return self.imgHigh, self.imgWidth
    # end of image_size()

    def load_data(self, ):
        return self.containers, self.features, self.Y.copy(), self.clusterNum.copy()
    # end of load_data()

    def images2feature(self, images):
        n_images = len(images)
        features = []

        tmp_features_str = Flow.LABEL_string()
        for i in range(n_images):
            image = images[i]
            image = (image * 127.5 + 127.5)					# scale [-1..1] to [0..255]
            image = image.astype(int).squeeze()				# convert to integer and dimensionality reduction

            high = image.shape[0]
            # process first line
            f_code =  FlowContainer2.image_row2attribute(image[0])
            try:
                (direction, length, iat) = FlowContainer2.unpack_attribute(bytes.fromhex(f_code))
            except ValueError:
                (direction, length, iat) = (0, 0, 0)
                print("ValueError: f_code = %s" % (f_code))
            # end of try
            #attribute_str = '%d, %d, %f' % (direction, length, iat)
            time = float(0)
            flow = Flow.Flow(time, '0', '0', 0, 0, 0, length, 0)
            #process the following lines
            for j in range(1, high):
                code =  FlowContainer2.image_row2attribute(image[j])
                if(code == END_CODE):
                    break
                f_code += code
                (direction, length, iat) = FlowContainer2.unpack_attribute(bytes.fromhex(code))
                #attribute_str += '\n%d, %d, %f' % (direction, length, iat)
                time += iat
                flow.add_packet(time, '0', '0', 0, 0, 0, length, direction)
            # end for j
            #feature = np.formstring(flow.flow_feature(), dtype=float, sep=',') # only in numpy 1.22
            tmp_features_str += flow.flow_feature()
        # end for i
        df = pd.read_csv(StringIO(tmp_features_str))
        return df
    # end of images2feature()
# end of class clusterDataset

# The normalDataset does not filter flows using HDBSCAN
class normalDataset:
    def __init__(self, container_files, labels, max_num_packets, max_num_flows = MaxNumFlows):
        self.nClasses = len(labels)			# the number of classes will be used in the system
        self.imgHigh = max_num_packets
        self.imgWidth = AttributeWide
        self.max_num_packets = max_num_packets
        self.max_num_flows = max_num_flows
        self.nFiles = len(container_files)

        data = containerDataset(container_files[0], max_num_packets, max_num_flows)
        self.containers, self.features = data.load_data()					# NOTE: self.features is a panda type
        self.Y = np.full((len(self.containers), ), labels[0])
        for i in range(1, self.nFiles):
            data = containerDataset(container_files[i], max_num_packets, max_num_flows)
            container, feature = data.load_data()								# NOTE: self.features is a panda type
            y = np.full((len(container), ), labels[i])
            self.containers = np.append(self.containers, container, axis=0)
            self.features = pd.concat([self.features, feature])			# NOTE: self.features is a panda type
            self.Y = np.append(self.Y, y, axis=0)
        # end for

        self.containers = (self.containers - 127.5) / 127.5		#scale [0..255] to [-1..1]

        # reset index for self.features (after append)
        self.features = self.features.reset_index(drop=True)

		# normailization. based on the normalization information build on the clusterDataset
        #self.features = StandardScaler(self.features, False) #################

        self.features = GetFeature(self.features)
        self.features = np.array(self.features)

        randomize = np.arange(len(self.containers))
        np.random.shuffle(randomize)

        self.containers = self.containers[randomize]
        self.features = self.features[randomize]
        self.Y = self.Y[randomize]
    # end of __init__():

    def num_classes(self, ):
        return self.nClasses
    # end of num_classes()

    def image_size(self, ):
        return self.imgHigh, self.imgWidth
    # end of image_size()

    def load_data(self, ):
        return self.containers, self.features, self.Y.copy()
    # end of load_data()
# end of class normalDataset


# the API translate an image to a feature vector
# the flow label is in the protocol field.
# Return: flag (True/False), flow_code, features, attribute_string
def image2feature(image, label):
    image = (image * 127.5 + 127.5)						# scale [-1..1] to [0..255]
    image = image.astype(int).squeeze()				# convert to integer and dimensionality reduction

    high = image.shape[0]

    # process first line
    f_code =  FlowContainer2.image_row2attribute(image[0])
    (direction, length, iat) = FlowContainer2.unpack_attribute(bytes.fromhex(f_code))
    if not (direction == 1 and iat == float(0)):
        return False, '', '', ''			# Incorrect flow
    attribute_str = '%d, %d, %f' % (direction, length, iat)
    time = float(0)
    flow = Flow.Flow(time, '0.0.0.0', '0.0.0.0', 0, 0, label, length, 0)
    #process the following lines
    for i in range(1, high):
        code =  FlowContainer2.image_row2attribute(image[i])
        if(code == END_CODE):
            break
        f_code += code
        (direction, length, iat) = FlowContainer2.unpack_attribute(bytes.fromhex(code))
        attribute_str += '\n%d, %d, %f' % (direction, length, iat)
        time += iat
        flow.add_packet(time, '0.0.0.0', '0.0.0.0', 0, 0, label, length, direction)
    # end for

    return True, f_code, flow.flow_feature(), attribute_str
# end of image2feature()
