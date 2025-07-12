##################################################################
# The program implelents the machine learning classifier
# 2025.04.29
##################################################################
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
sys.path.append('./Lib')		# add library path
import FlowContainer2
import argparse
import datetime
import numpy as np
import random
from tools import check_directory, write_log, save_image, save_image_with_label, gen_pca, gen_kde
from evaluations import PSNR, MSE
from sklearn import tree, svm, naive_bayes, neighbors, ensemble, neural_network
import joblib

def get_args():
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument('--rootDir', type=str, help='Root directory', default='./CIC-IDS2018')  ####
    parser.add_argument('--clfRootDir', type=str, help='Root directory', default='./ClassifierML')
#    parser.add_argument('--batchSize', type=int, help='Batch size', default=64)    
#    parser.add_argument('--epochs', type=int, help='Number of epochs', default=30)
#    parser.add_argument('--lr_clf', type=float, help='Learnging rate for classifier, default=1e-3', default=1e-4)
#    parser.add_argument('--threshold_clf', type=float, help='Threshold for training classifier early stopping, default=1e-6', default=1e-4)
    parser.add_argument('--aeFile', type=str, help='Adversarial example file in numpy', default='')
    parser.add_argument('--num', type=int, help='The number of data', default=0)
    parser.add_argument('--used_ae_num', type=int, help='The number of data', default=6500)
    opt = parser.parse_args()
    
    # other global parameters
    opt.ae_num = 0
    opt.aePath = './AE'
    
    opt.logPath = opt.clfRootDir + '/Log%d' % (opt.ae_num)    
    opt.statFile = opt.clfRootDir + '/asr_stat_ml.csv'
    opt.logFile = opt.logPath + '/clf_log-%s.csv' % (datetime.datetime.now().strftime("%Y-%m-%d"))
    opt.clfPath = opt.clfRootDir + '/Models-%d' % (opt.num)
    opt.classifierName = ['DT', 'RF', 'SVM', 'NB', 'KNN', 'GB', 'MLP']     
    opt.classifierFile = [opt.clfPath + '/DT_test.pkl', opt.clfPath + '/RF_test.pkl', opt.clfPath + '/SVM_test.pkl', opt.clfPath + '/NB_test.pkl', opt.clfPath + '/KNN_test.pkl', opt.clfPath + '/GB_test.pkl', opt.clfPath + '/MLP_test.pkl'] 
    opt.seed = 42  

    ####
    opt.datasetDir = './data/CIC-IDS2018'   
    opt.trainContainersFile = opt.datasetDir + '/train_containers-%d.npy' % (opt.num)
    opt.trainFeaturesFile = opt.datasetDir + '/train_features-%d.npy' % (opt.num)
    opt.trainLabelsFile = opt.datasetDir + '/train_labels-%d.npy' % (opt.num)
    opt.testContainersFile = opt.datasetDir + '/test_containers-%d.npy' % (opt.num)
    opt.testFeaturesFile = opt.datasetDir + '/test_features-%d.npy' % (opt.num)
    opt.testLabelsFile = opt.datasetDir + '/test_labels-%d.npy' % (opt.num)

    opt.aeContainersFile = opt.aePath + '/' + opt.aeFile
    opt.imageH = 32
    opt.imageW = 16            
    opt.imageChannel = 1
    opt.content_size = 12    ####
    opt.style_size = 12      ####
    
    print(opt)
    
    # make directories
    check_directory(opt.rootDir)
    check_directory(opt.clfRootDir)
    check_directory(opt.logPath)
    check_directory(opt.clfPath)
    check_directory(opt.aePath)
 
    # save parameter information to log
    write_log(opt.logFile, "============= %s =============\n" % (datetime.datetime.now().strftime("%Y-%m-%d")), 'a')
    opt_str = vars(opt)
    opt_str = '\n'.join([f'{key}: {value}' for key, value in opt_str.items()])
    opt_str += "\n==================\n"
    write_log(opt.logFile, opt_str, 'a')
    
    return opt
# end of get_args()

class Dataset:      
    def __init__(self, opt):        ####
        self.testX = np.load(opt.testContainersFile).reshape((-1, 1, opt.imageH, opt.imageW))
        self.testF = np.load(opt.testFeaturesFile)
        self.testY = np.load(opt.testLabelsFile)
        
        self.numClasses = int(np.max(self.testY) + 1)
    # end of __init()__
    
    def classes_number(self, ):
        return self.numClasses
    # end of classes_number()
    
    def load_data(self, ):     
        return self.testX, self.testF, self.testY
    # end of load_data()  
# end of class dataset

class AE_Dataset:
    def __init__(self, opt):
        self.opt = opt
        self.aeX = np.load(opt.aeContainersFile).reshape((-1, 1, opt.imageH, opt.imageW))
    # end of __init__()
    
    def load_ae(self, ):
        return self.aeX[:self.opt.used_ae_num]
    # end of load_ae()
# end of class AE_Dataset

class Classifier:
    def __init__(self, opt, dataset, clf, clfId):
        self.opt = opt
        self.dataset = dataset
        self.numClasses = self.dataset.classes_number()
        self.clf = clf
        self.clfId = clfId
    # end of __init__()

    def train_classifier(self, ):
        testX, _, testY = self.dataset.load_data()
        testX = testX.reshape(testX.shape[0], -1)
        self.clf.fit(testX, testY)
                
        # save clf
        joblib.dump(self.clf, self.opt.classifierFile[self.clfId])
        
        # evaluate clf by testing data
        print("Evaluate classifier ....")
        predict = self.clf.predict(testX)
        idx = np.where(predict == testY)
        print("%s classifier accuracy=%f\n" % (self.opt.classifierName[self.clfId], len(idx[0])/len(predict)))
        out_str = "%s classifier accuracy=%f\n" % (self.opt.classifierName[self.clfId], len(idx[0])/len(predict))
        write_log(self.opt.logFile, out_str, 'a')
    # end of train_classifier()
    
    def get_classifier(self, ):
        if not os.path.exists(self.opt.classifierFile[self.clfId]):  # clf file does not exist
            self.train_classifier()
        else:
            # load clf
            print("Loading classifier ....")
            self.clf = joblib.load(self.opt.classifierFile[self.clfId])
        # end if
        
        testX, _, testY = self.dataset.load_data()   ####
        testX = testX.reshape(testX.shape[0], -1)
        # evaluate clf by testing data
        print("Evaluate classifier ....")
        predict = self.clf.predict(testX)
        idx = np.where(predict == testY)
        print("%s classifier accuracy=%f\n" % (self.opt.classifierName[self.clfId], len(idx[0])/len(predict)))
        out_str = "%s classifier accuracy=%f\n" % (self.opt.classifierName[self.clfId], len(idx[0])/len(predict))
        write_log(self.opt.logFile, out_str, 'a')

        return self.clf
    # end of get_classifier()
# end of class Classifier

def main():
    opt = get_args()
    dataset = Dataset(opt)
    class_num = dataset.classes_number()
    classifier = [
        Classifier(opt, dataset, tree.DecisionTreeClassifier(), 0),
        Classifier(opt, dataset, ensemble.RandomForestClassifier(), 1),
        Classifier(opt, dataset, svm.SVC(kernel='rbf'), 2),
        Classifier(opt, dataset, naive_bayes.GaussianNB(), 3),
        Classifier(opt, dataset, neighbors.KNeighborsClassifier(), 4),
        Classifier(opt, dataset, ensemble.GradientBoostingClassifier(), 5),
        Classifier(opt, dataset, neural_network.MLPClassifier(), 6)
    ]
    ae_data = AE_Dataset(opt)
    
    for i in range(len(classifier)):
        if(not os.path.exists(opt.classifierFile[i])):
            print("Training classifier %s .....\n" % (opt.classifierName[i]))
            write_log(opt.logFile, "Training classifier %s .....\n" % (opt.classifierName[i]), 'a')
            classifier[i].train_classifier()
        else:
            print("Loading classifier %s .....\n" % (opt.classifierName[i]))
            write_log(opt.logFile, "Loading classifier %s .....\n" % (opt.classifierName[i]), 'a')
            classifier[i].get_classifier()
    # end for
    
    print("Evaluate attack success rate (ASR) ....")
    aeX = ae_data.load_ae() 
    aeX = aeX.reshape(aeX.shape[0], -1)
    Y = np.full(len(aeX), 0)
    asr = np.zeros(len(classifier))
    for i in range(len(classifier)):
        predict = classifier[i].clf.predict(aeX)
        idx = np.where(predict == Y)
        asr[i] = len(idx[0])/len(predict)
        print("%s classifier ASR=%f\n" % (opt.classifierName[i], len(idx[0])/len(predict)))
        out_str = "%s: %s classifier ASR=%f\n" % (opt.aeFile, opt.classifierName[i], len(idx[0])/len(predict))
        write_log(opt.logFile, out_str, 'a')       
    # end for
    
    ofp = open(opt.statFile, 'a')
    ofp.write("%d, %s, " %(opt.num, opt.aeFile))
    for i in range(len(classifier)):
        ofp.write("%s," % (opt.classifierName[i]))
    for i in range(len(classifier)):
        ofp.write("%.6f," % (asr[i]))
    ofp.write("\n")
    ofp.close()
# end of main

if __name__ == '__main__':
    main()
