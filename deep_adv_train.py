##################################################################
# The program implelents the deep learning classifier by PyTorch
# 2025.04.13
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
import torch
import torchvision
from tqdm import tqdm
import gc
from tools import check_directory, write_log, save_image, save_image_with_label, gen_pca, gen_kde
from models import LeNet5, VGG11, SmallAlexNet, ResNet18
from evaluations import PSNR, MSE

def get_args():
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument('--rootDir', type=str, help='Root directory', default='./CIC-IDS2018')  ####
    parser.add_argument('--clfRootDir', type=str, help='Root directory', default='./ClassifierDL')
    parser.add_argument('--batchSize', type=int, help='Batch size', default=64)    
    parser.add_argument('--epochs', type=int, help='Number of epochs', default=30)
    parser.add_argument('--lr_clf', type=float, help='Learnging rate for classifier, default=1e-3', default=1e-4)
    parser.add_argument('--threshold_clf', type=float, help='Threshold for training classifier early stopping, default=1e-6', default=1e-4)
    parser.add_argument('--test_data_szie', type=int, help='The number of test data for each class', default=6500)
    parser.add_argument('--num', type=int, help='The number of data', default=0)
    parser.add_argument('--train_ratio', type=float, help='The ratio of adversarial training file, default=0.05', default=0.05)
    parser.add_argument('--used_ae_num', type=int, help='The number of data', default=5000)
    parser.add_argument('--train_ae', type=str, help='Adversarial training file in numpy, (st_adv, adv_gan, pgd)', default='st_adv')
    parser.add_argument('--test_ae', type=str, nargs='+', help='Test adversarial example file array in numpy, ex: --test_ae a1.npy a2.npy', default=[])
    
    opt = parser.parse_args()
    
    # other global parameters
    opt.ae_num = 0
    opt.aePath = './AE' 
    
    opt.logPath = opt.clfRootDir + '/Log%d' % (opt.ae_num)    
    opt.statFile = opt.clfRootDir + '/asr_stat_%s_%d_dl.csv' % (opt.train_ae, int(opt.train_ratio * 100))
    opt.logFile = opt.logPath + '/clf_log-%s.csv' % (datetime.datetime.now().strftime("%Y-%m-%d"))
    opt.modelPath = opt.clfRootDir + '/Models-%d' % (opt.num)
    opt.classifierName = ['VGG11', 'ResNet18', 'LeNet5', 'SmallAlexNet']     
    opt.classifierFile = [opt.modelPath + '/VGG11_%s_%d.pth' % (opt.train_ae, int(opt.train_ratio * 100)), 
                        opt.modelPath + '/ResNet18_%s_%d.pth' % (opt.train_ae, int(opt.train_ratio * 100)), 
                        opt.modelPath + '/LeNet5_%s_%d.pth' % (opt.train_ae, int(opt.train_ratio * 100)), 
                        opt.modelPath + '/SmallAlexNet_%s_%d.pth' % (opt.train_ae, int(opt.train_ratio * 100))] 
    opt.seed = 42  

    ####
    opt.datasetDir = './data/CIC-IDS2018'   
    opt.trainContainersFile = opt.datasetDir + '/train_containers-%d.npy' % (opt.num)
    opt.trainFeaturesFile = opt.datasetDir + '/train_features-%d.npy' % (opt.num)
    opt.trainLabelsFile = opt.datasetDir + '/train_labels-%d.npy' % (opt.num)
    opt.testContainersFile = opt.datasetDir + '/test_containers-%d.npy' % (opt.num)
    opt.testFeaturesFile = opt.datasetDir + '/test_features-%d.npy' % (opt.num)
    opt.testLabelsFile = opt.datasetDir + '/test_labels-%d.npy' % (opt.num)

#    opt.aeContainersFile = opt.aePath + '/' + opt.test_ae
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
    check_directory(opt.modelPath)
    check_directory(opt.aePath)
 
    # save parameter information to log
    write_log(opt.logFile, "============= %s =============\n" % (datetime.datetime.now().strftime("%Y-%m-%d")), 'a')
    opt_str = vars(opt)
    opt_str = '\n'.join([f'{key}: {value}' for key, value in opt_str.items()])
    opt_str += "\n==================\n"
    write_log(opt.logFile, opt_str, 'a')
    
    return opt
# end of get_args()

#----- Loss Functions ----
loss_fn_CE = torch.nn.CrossEntropyLoss()
loss_fn_MSE = torch.nn.MSELoss()
loss_fn_l1 = torch.nn.L1Loss()

class Dataset:      
    def __init__(self, opt):        ####
        self.testX = np.load(opt.testContainersFile).reshape((-1, 1, opt.imageH, opt.imageW))
        self.testY = np.load(opt.testLabelsFile)
        
        self.numClasses = int(np.max(self.testY) + 1)
        
        # load and add training AE
        train_ae_num = int(opt.test_data_szie * opt.train_ratio)
        for i in range(1, self.numClasses):     # for each class
            ae_file_name = opt.aePath + '/' + opt.train_ae + '_class%d.npy' % (i)
            aeX = np.load(ae_file_name).reshape((-1, 1, opt.imageH, opt.imageW))
            aeX = aeX[opt.used_ae_num : opt.used_ae_num+train_ae_num]
            aeY = np.full((len(aeX)), i)
            self.testX = np.append(self.testX, aeX, axis=0)
            self.testY = np.append(self.testY, aeY, axis=0)
        # end for
        
        idx = list(range(len(self.testX)))
        np.random.shuffle(idx)
        self.testX = self.testX[idx]
        self.testY = self.testY[idx]
    # end of __init()__
    
    def classes_number(self, ):
        return self.numClasses
    # end of classes_number()
    
    def load_data(self, ):     
        return torch.tensor(self.testX).float(),  torch.tensor(self.testY)
    # end of load_data()  
# end of class dataset

class AE_Dataset:
    def __init__(self, opt, ae_file):
        self.opt = opt
        self.aeX = np.load(ae_file).reshape((-1, 1, opt.imageH, opt.imageW))
    # end of __init__()
    
    def load_ae(self, ):
        return torch.tensor(self.aeX[:self.opt.used_ae_num]).float()
    # end of load_ae()
# end of class AE_Dataset

class Classifier:
    def __init__(self, opt, dataset, model, modelId):
        self.opt = opt
        self.dataset = dataset
        self.numClasses = self.dataset.classes_number()
        self.model = model
        self.modelId = modelId
        self.seed = opt.seed
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.opt.lr_clf, weight_decay=0.01)
    # end of __init__()

    def train_classifier(self, ):
        # set random seed              #####<<<<>>>>
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # train model
        for epoch in range(1, self.opt.epochs+1):
            gc.collect() # collect garbage
            self.model.train()   # set model trainable
            running_loss = 0.0
            
            testX, testY = self.dataset.load_data()   ####
            testY = torch.nn.functional.one_hot(testY, num_classes=self.numClasses)
            test_ds = torch.utils.data.TensorDataset(testX, testY)   
            test_ds = torch.utils.data.DataLoader(test_ds, batch_size = self.opt.batchSize, shuffle=True)
            progress = tqdm(test_ds, bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}")
            for image, label in progress:
                # clear gradient 
                self.optimizer.zero_grad()
                # forward
                predict = self.model(image)
                # calculate loss
                loss = loss_fn_CE(predict, label.float())
                # backward
                loss.backward()
                # update model
                self.optimizer.step()
                # record loss
                running_loss += loss.item()
            # end for 
            
            avg_loss = running_loss / len(progress)
            print(f"Epoch [{epoch}/{self.opt.epochs}], Loss: {avg_loss:.8f}")
            if(avg_loss < self.opt.threshold_clf):
                print(f"{self.opt.classifierName[self.modelId]} classifier training break at epochs: {epoch}, loss: {avg_loss:.8f}")
                out_str = "%s classifier training break at epochs: %d, loss:%.8f\n" % (self.opt.classifierName[self.modelId], epoch, avg_loss)
                write_log(self.opt.logFile, out_str, 'a')
                break
        # end for epoch
        
        # save model
        torch.save(self.model.state_dict(), self.opt.classifierFile[self.modelId])
        
        # evaluate model by testing data
        print("Evaluate classifier ....")
        self.model.eval() # set model evaluable
        with torch.no_grad():
            predict = self.model(testX)
        predict = torch.argmax(predict, dim=1)
        validY = torch.argmax(testY, dim=1)
        idx = np.where(predict.numpy() == validY.numpy())
        print("%s classifier model accuracy=%f\n" % (self.opt.classifierName[self.modelId], len(idx[0])/len(predict)))
        out_str = "%s classifier model accuracy=%f\n" % (self.opt.classifierName[self.modelId], len(idx[0])/len(predict))
        write_log(self.opt.logFile, out_str, 'a')
    # end of train_classifier()
    
    def get_classifier(self, ):
        if not os.path.exists(self.opt.classifierFile[self.modelId]):  # model file does not exist
            self.train_classifier()
        else:
            # load model
            state_dict = torch.load(self.opt.classifierFile[self.modelId])
            self.model.load_state_dict(state_dict)
        # end if
        
        testX, testY = self.dataset.load_data()   ####
        self.model.eval() # set model evaluable
        with torch.no_grad():
            predict = self.model(testX)
        predict = torch.argmax(predict, dim=1)
        idx = np.where(predict.numpy() == testY.numpy())
        print("%s classifier model accuracy=%f\n" % (self.opt.classifierName[self.modelId], len(idx[0])/len(predict)))
        out_str = "%s classifier model accuracy=%f\n" % (self.opt.classifierName[self.modelId], len(idx[0])/len(predict))
        write_log(self.opt.logFile, out_str, 'a')
        
        return self.model
    # end of get_classifier()
# end of class Classifier

def main():
    opt = get_args()
    dataset = Dataset(opt)
    class_num = dataset.classes_number()
    classifier = [
        Classifier(opt, dataset, VGG11(c_in=opt.imageChannel, c_out=class_num), 0),
        Classifier(opt, dataset, ResNet18(c_in=opt.imageChannel, c_out=class_num), 1),
        Classifier(opt, dataset, LeNet5(c_in=opt.imageChannel, c_out=class_num, H=opt.imageH, W=opt.imageW), 2),
        Classifier(opt, dataset, SmallAlexNet(c_in=opt.imageChannel, c_out=class_num, H=opt.imageH, W=opt.imageW), 3),
    ]
    
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
    
    for f_idx in range(len(opt.test_ae)):
        aeContainersFile = opt.aePath + '/' + opt.test_ae[f_idx]
        ae_data = AE_Dataset(opt, aeContainersFile)
        
        print("Evaluate attack success rate (ASR) for %s...." % (opt.test_ae[f_idx]))
        aeX = ae_data.load_ae() 
        Y = np.full(len(aeX), 0)
        asr = np.zeros(len(classifier))
        for i in range(len(classifier)):
            classifier[i].model.eval()  # set model evaluable
            with torch.no_grad():
                predict = classifier[i].model(aeX)
            predict = torch.argmax(predict, dim=1)
            idx = np.where(predict.numpy() == Y)
            asr[i] = len(idx[0])/len(predict)
            print("%s classifier ASR=%f\n" % (opt.classifierName[i], len(idx[0])/len(predict)))
            out_str = "adv train:%s/%f, test ae:%s, %s classifier ASR=%f\n" % (opt.train_ae, opt.train_ratio, opt.test_ae[f_idx], opt.classifierName[i], len(idx[0])/len(predict))      
            write_log(opt.logFile, out_str, 'a')       
        # end for
        
        ofp = open(opt.statFile, 'a')
        ofp.write("%d, train:%s/%f, test:%s, " % (opt.num, opt.train_ae, opt.train_ratio, opt.test_ae[f_idx]))
        for i in range(len(classifier)):
            ofp.write("%s," % (opt.classifierName[i]))
        for i in range(len(classifier)):
            ofp.write("%.6f," % (asr[i]))
        ofp.write("\n")
        ofp.close()
    # end for
# end of main

if __name__ == '__main__':
    main()
