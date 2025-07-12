##################################################################
# The program implelents the Style Fusion AutoEncoder for 
# flow container by PyTorch
# The generator generates fake image and perturbation.
# The discriminator model is LeNet5
# 2025.03.13
##################################################################
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
sys.path.append('./Lib')		# add library path
import FlowContainer2
import argparse
import datetime
import numpy as np
import torch
import torchvision
from tqdm import tqdm
import gc
from tools import check_directory, write_log, save_image, save_image_with_label, gen_pca, gen_kde
from models import LeNet5, Generator, Discriminator
from losses import dCov2
from evaluations import PSNR, MSE

# compute regularization hyperparameter
def get_lambda(lambda0, lambda_schedule, n_total_iter):
    s = lambda_schedule
    if s == 0:
        return lambda0
    else:
        return lambda0 * float(min(n_total_iter, s)) / s
# end of get_lambda()

def get_args():
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument('--rootDir', type=str, help='Root directory', default='./CIC-IDS2018')  ####
    parser.add_argument('--batchSize', type=int, help='Batch size', default=64)
    parser.add_argument('--recon_level', type=int, help='Level wherein computing feature reconstruction error, default=2', default=2)
    parser.add_argument('--num', type=int, help='The number of data', default=0)
    
    parser.add_argument('--epochs_gan', type=int, help='Number of epochs for GAN', default=30)
    parser.add_argument('--lr_dis', type=float, help='Learning rate for discriminator, default=1e-4', default=1e-4)
    parser.add_argument('--lr_dec', type=float, help='Learning rate for decoder, default=1e-4', default=1e-4)
    parser.add_argument('--lr_se', type=float, help='Learning rate for style_encoder, default=1e-4', default=1e-4)
    
    parser.add_argument('--trainClassifier', type=bool, help='Re-train classifier', default=False)
    parser.add_argument('--epochs_clf', type=int, help='Number of epochs for classifier', default=10)
    parser.add_argument('--lr_clf', type=float, help='Learnging rate for classifier, default=1e-3', default=1e-3)
    parser.add_argument('--beta1', type=float, help='beta1 for Adam. default=0.5', default=0.5)
    parser.add_argument('--add_mask', type=bool, help='Mask the generated images', default=True)
    parser.add_argument('--add_noise', type=bool, help='Add noise while generating adversarial images', default=True)
    parser.add_argument('--noise_weight', type=float, help='Weight of the noise', default=0.1)
    
    parser.add_argument('--trainCE', type=bool, help='Re-train content encoder', default=False )
    parser.add_argument('--epochs_ce', type=int, help='Number of epochs for content_encoder', default=30)
    parser.add_argument('--lr_ce', type=float, help='Learning rate for content_encoder, default=1e-4', default=1e-4)
    parser.add_argument('--threshold_ce', type=float, help='Threshold for training content encoder early stopping, default=1e-4', default=1e-4)
    
    parser.add_argument('--genAE', type=int, help='Generate adversarial examples. The number is the epoch number of the model.', default=-1 )
    parser.add_argument('--source', type=int, help='The class number of the source images, default=1', default=1)
    parser.add_argument('--target', type=int, help='The class number of the target images, default=0', default=0)
    parser.add_argument('--gen_num', type=int, help='The number of AEs to generate, default=8000', default=8500)
    parser.add_argument('--split_num', type=int, help='Split ratio number, default=16', default=16)
    opt = parser.parse_args()
    
    # other global parameters
    opt.logPath = opt.rootDir + '/Log%d' % (opt.num)
    opt.logFile = opt.logPath + '/log-%s.csv' % (datetime.datetime.now().strftime("%Y-%m-%d"))
    opt.modelPath = opt.rootDir + '/Models-%d' % (opt.num)
    opt.classifierFile = opt.modelPath + '/LeNet5_clf.pth'
    opt.CE_File = opt.modelPath + '/CE.pth'
    opt.SE_File = opt.modelPath + '/SE-%d.pth'
    opt.decoderFile = opt.modelPath + '/Decoder-%d.pth'
#    opt.sourceImage = opt.logPath + '/source.png'
#    opt.targetImage = opt.logPath + '/target.png'
#    opt.reconsImage = opt.logPath + '/reconstruct.png'
    opt.aePath = opt.rootDir + '/AE-%d' % (opt.num)
    opt.ae_logFile = opt.aePath + '/adv_log-%s.csv' % (datetime.datetime.now().strftime("%Y-%m-%d"))
#    opt.advImage = opt.aePath + '/adv_class%d_epoch%d.png'
    opt.advFile = opt.aePath + '/adv_class%d_epoch%d.npy'
    opt.srcFile = opt.aePath + '/src_class%d_epoch%d.npy'
    opt.tarFile = opt.aePath + '/tar_class%d_epoch%d.npy'
    opt.pcaFile = opt.aePath + '/pca_class%d_epoch%d.png'
    opt.kdeFile = opt.aePath + '/kde_class%d_epoch%d.png'
    ####
    opt.datasetDir = './data/CIC-IDS2018'   
    opt.trainContainersFile = opt.datasetDir + '/train_containers-%d.npy' % (opt.num)
    opt.trainFeaturesFile = opt.datasetDir + '/train_features-%d.npy' % (opt.num)
    opt.trainLabelsFile = opt.datasetDir + '/train_labels-%d.npy' % (opt.num)
    opt.testContainersFile = opt.datasetDir + '/test_containers-%d.npy' % (opt.num)
    opt.testFeaturesFile = opt.datasetDir + '/test_features-%d.npy' % (opt.num)
    opt.testLabelsFile = opt.datasetDir + '/test_labels-%d.npy' % (opt.num)
    opt.imageH = 32
    opt.imageW = 16         ####
    opt.imageChannel = 1
    opt.content_size = 12      ####
    opt.style_size = 12      ####
    opt.weight_loss_recons_image = 1
    opt.weight_loss_recons_content = 1
    opt.weight_loss_recons_feature = 1
    opt.weight_loss_adv = 1 
    opt.adv_perturbRateMax = 0.5  ##### 0.5 -> 1 -> 0.75   ==> 0.5: the quality is the better
    opt.lambda_decorr = 1       # Decorrelation regularization coefficient
    opt.lambda_schedule = 50000 # Progressively increase decorrelation lambda (0 to disable)
    opt.lambda_dis = 1e-2       # Discriminator coefficient
#    opt.img_min = 0.0           # adversarial example pixel values minimum
#    opt.img_max = 1.0           # adversarial example pixel values maximun
#    opt.perturb = 0.01           # perturbtion delta in [-opt.perturb, opt.perturb]
    
#    # test device
#    opt.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#    print("Using device:", opt.device)
    
    print(opt)
    
    # make directories
    check_directory(opt.rootDir)
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
loss_fn_dCov = dCov2()
loss_fn_l1 = torch.nn.L1Loss()
#----- Dataset -----

class Dataset:      
    def __init__(self, opt):        ####
        self.trainX = np.load(opt.trainContainersFile).reshape((-1, 1, opt.imageH, opt.imageW))
        self.trainF = np.load(opt.trainFeaturesFile)
        self.trainY = np.load(opt.trainLabelsFile)
        
        # Total_Packets, IAT_Sum, IAT_Mean, IAT_Std, IAT_Max, IAT_Min, Length_First, Length_Min, Length_Max, Length_Sum, Length_Mean, Length_Std
        self.trainCF = np.array([self.trainF[:,1], self.trainF[:,2], self.trainF[:,3], self.trainF[:,4],
                                 self.trainF[:,5], self.trainF[:,6], self.trainF[:,7], self.trainF[:,8],
                                 self.trainF[:,9], self.trainF[:,10], self.trainF[:,11], self.trainF[:,12]]).T  # content features
        # normalize to [0, 1]
        self.trainCF = (self.trainCF - np.min(self.trainCF)) / (np.max(self.trainCF) - np.min(self.trainCF))
        self.trainCF = self.trainCF.astype(np.float32)
        
#        self.testX = np.load(opt.testContainersFile).reshape((-1, 1, opt.imageH, opt.imageW))
#        self.testF = np.load(opt.testFeaturesFile)
#        self.testY = np.load(opt.testLabelsFile)
        
        self.numClasses = int(np.max(self.trainY) + 1)
    # end of __init()__
    
    def classes_number(self, ):
        return self.numClasses
    # end of classes_number()
    
    def load_training_data(self, ):     ####
        return torch.tensor(self.trainX).float(), torch.tensor(self.trainF).float(), torch.tensor(self.trainCF).float(), torch.tensor(self.trainY)
    # end of load_mnist_data()
    
    def load_sampling_data(self, source, target, num = 512):
        if(num == 0):
            s_idx = np.where(self.trainY == source)[0]
            s_images = self.trainX[s_idx]
            t_idx = np.where(self.trainY == target)[0]
            t_images = self.trainX[t_idx]
            return torch.tensor(s_images).float(), torch.tensor(t_images).float()
        else:
            s_idx = np.where(self.trainY == source)[0]
            np.random.shuffle(s_idx)
            s_images = self.trainX[s_idx[:num]]           
            t_idx = np.where(self.trainY == target)[0]
            np.random.shuffle(t_idx)
            t_images = self.trainX[t_idx[:num]]       
            return torch.tensor(s_images).float(), torch.tensor(t_images).float()
    # end of load_sampling_data()
    
    def gen_test_images(self, num):
        idx = np.where(self.trainY > 0)[0]
        np.random.shuffle(idx)
        x_images = self.trainX[idx[:num]]
        idx = np.where(self.trainY == 0)[0]
        np.random.shuffle(idx)
        t_images = self.trainX[idx[:num]]
        return torch.tensor(x_images).float(), torch.tensor(t_images).float()
    # end of gen_test_images()
# end of class dataset

class Classifier:
    def __init__(self, opt, dataset):
        self.opt = opt
        self.dataset = dataset
        self.numClasses = self.dataset.classes_number()
        self.model = LeNet5(c_in=self.opt.imageChannel, c_out=self.numClasses, H=self.opt.imageH, W=self.opt.imageW)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.opt.lr_clf, betas=(self.opt.beta1, 0.999))
    # end of __init__()

    def train_classifier(self, ):
        # train model
        for epoch in range(1, self.opt.epochs_clf+1):
            gc.collect() # collect garbage
            self.model.train()   # set model trainable
            running_loss = 0.0
            
            trainX, _, _, trainY = self.dataset.load_training_data()   ####
            trainY = torch.nn.functional.one_hot(trainY, num_classes=self.numClasses)
            train_ds = torch.utils.data.TensorDataset(trainX, trainY)   
            train_ds = torch.utils.data.DataLoader(train_ds, batch_size = self.opt.batchSize, shuffle=True)
            progress = tqdm(train_ds, bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}")
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
            print(f"Epoch [{epoch}/{self.opt.epochs_clf}], Loss: {avg_loss:.5f}")
        # end for epoch
        
        # save model
        torch.save(self.model, self.opt.classifierFile)
        
        # evaluate model by training data
        print("Evaluate classifier ....")
        self.model.eval() # set model evaluable
        with torch.no_grad():
            predict = self.model(trainX)
        predict = torch.argmax(predict, dim=1)
        validY = torch.argmax(trainY, dim=1)
        idx = np.where(predict.numpy() == validY.numpy())
        print("Classifier Model Accuracy=%f\n" % (len(idx[0])/len(predict)))
        out_str = "Classifier Model Accuracy=%f\n" % (len(idx[0])/len(predict))
        write_log(self.opt.logFile, out_str, 'a')
    # end of train_classifier()
    
    def get_classifier(self, ):
        if not os.path.exists(self.opt.classifierFile):  # model file does not exist
            self.train_classifier()
        # end if
        # load model
        self.model = torch.load(self.opt.classifierFile)
        
        trainX, _, _, trainY = self.dataset.load_training_data()   ####
        self.model.eval() # set model evaluable
        with torch.no_grad():
            predict = self.model(trainX)
        predict = torch.argmax(predict, dim=1)
        idx = np.where(predict.numpy() == trainY.numpy())
        print("Classifier Model Accuracy=%f\n" % (len(idx[0])/len(predict)))
        out_str = "Classifier Model Accuracy=%f\n" % (len(idx[0])/len(predict))
        write_log(self.opt.logFile, out_str, 'a')
        
        return self.model
    # end of get_classifier()
# end of class Classifier

class StyleAdv:
    def __init__(self, opt, dataset, classifier):
        self.opt = opt
        self.dataset = dataset
        self.generator = Generator(content_size=opt.content_size, style_size=opt.style_size, c_in=opt.imageChannel, nf=32, H=opt.imageH, 
                        W=opt.imageW, add_mask= opt.add_mask, add_noise=opt.add_noise, noise_weight=opt.noise_weight)
        self.discriminator = Discriminator(c_in=opt.imageChannel, nf=32, H=opt.imageH, W=opt.imageW, recon_level=opt.recon_level)
        self.classifier = classifier.get_classifier()
        # optimizers
        self.optim_ce = torch.optim.AdamW(self.generator.ce.parameters(), lr=opt.lr_ce, weight_decay=0.001)
        self.optim_se = torch.optim.AdamW(self.generator.se.parameters(), lr=opt.lr_se, weight_decay=0.001)
        self.optim_decoder = torch.optim.AdamW(self.generator.decoder.parameters(), lr=opt.lr_dec, weight_decay=0.001)
        self.optim_discriminator = torch.optim.AdamW(self.discriminator.parameters(), lr=opt.lr_dis, weight_decay=0.001)
    # end of __init__()
    
    def train(self,):   # train GAN
        if(self.opt.trainCE or not os.path.exists(self.opt.CE_File)):
            self.train_ce()
        else:
            self.generator.ce = torch.load(self.opt.CE_File)
            out_str = "Reload previous CE model at: " + self.opt.CE_File + "\n"
            write_log(self.opt.logFile, out_str, 'a')
        # end if
        
        write_log(self.opt.logFile, "Training GAN\n", 'a')
        for epoch in range(1, self.opt.epochs_gan+1):
            gc.collect() # collect garbage
            self.generator.ce.eval()
            self.generator.se.train()
            self.generator.decoder.train()
            self.discriminator.train()
            
            trainX, _, _, trainY = self.dataset.load_training_data()     
            trainY = torch.nn.functional.one_hot(trainY, num_classes=self.dataset.classes_number())
            train_ds = torch.utils.data.TensorDataset(trainX, trainY)
            train_ds = torch.utils.data.DataLoader(train_ds, batch_size = self.opt.batchSize, shuffle=True)
            progress = tqdm(train_ds, bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}")
            counter = 0
            perturbRateMax = (self.opt.epochs_gan - epoch)/self.opt.epochs_gan * self.opt.adv_perturbRateMax
            for image, label in progress:
                loss_disc = self.train_discriminator(image)
                loss_adv = self.train_adv(image[:len(image)//2], image[len(image)//2:], perturbRateMax)
#                loss_adv = 0
                loss_recons_image, loss_decoder = self.train_decoder(image)
                loss_decorr, loss_se = self.train_se(image, counter)
                counter += 1
            # end for
            
            avg_loss_disc = loss_disc / len(progress)
            avg_recons_img = loss_recons_image / len(progress)
            avg_decoder = loss_decoder / len(progress)
            avg_decorr = loss_decorr / len(progress)
            avg_se = loss_se / len(progress)
            avg_adv = loss_adv / len(progress)
            
            print(f"Epoch [{epoch}/{self.opt.epochs_gan}], Discriminator loss: {avg_loss_disc:.6f}, Recons Image: {avg_recons_img:.6f}")
            out_str = "Epoch [%d/%d], Discriminator loss, %.6f, recons_img, %.6f, decoder loss, %.6f, decorr loss, %.6f, se loss, %.6f, adv, %.6f\n" % (epoch, self.opt.epochs_gan, avg_loss_disc, avg_recons_img, avg_decoder, avg_decorr, avg_se, avg_adv)
            write_log(self.opt.logFile, out_str, 'a')
            
            # save model
            torch.save(self.generator.se, self.opt.SE_File % (epoch))    
            torch.save(self.generator.decoder, self.opt.decoderFile % (epoch))
        # end for epoch
    # end of train
    
    def train_discriminator(self, x_inputs):
        self.discriminator.requires_grad_(True)
        self.generator.se.requires_grad_(False)
        self.generator.decoder.requires_grad_(False)
 
        # clear gradient 
        self.optim_discriminator.zero_grad()
        
        # forward
        _, _, x_recons = self.generator(x_inputs)
        
        dis_output = self.discriminator(x_recons, x_inputs, mode='GAN')
        
        # loss
        dis_output_sampled = dis_output[:len(x_inputs)]
        dis_output_original = dis_output[len(x_inputs):]
        dis_original = -torch.log(dis_output_original + 1e-3)
        dis_sampled = -torch.log(1 - dis_output_sampled + 1e-3)
        loss_discriminator = torch.mean(dis_original) + torch.mean(dis_sampled)

#        # calculate loss
#        label_real = torch.full((x_inputs.size(0),), 1)
#        label_fake = torch.full((x_inputs.size(0),), 0)
#        label = torch.cat((label_fake, label_real), 0).unsqueeze(1).float() 
#        loss_discriminator = loss_fn_CE(label, dis_output)
        
        if(loss_discriminator > 0.1):
            # backward
            loss_discriminator.backward()
            # update model
            self.optim_discriminator.step()
            
        return loss_discriminator
    # end of train_discriminator()
    
    def train_decoder(self, x_inputs):
        self.discriminator.requires_grad_(False)
        self.generator.se.requires_grad_(False)
        self.generator.decoder.requires_grad_(True)
        
        # forward
        x_content, x_style, x_recons = self.generator(x_inputs)
        
        mid_repre = self.discriminator(x_recons, x_inputs, mode='RECON')
        mid_repre_recons = mid_repre[:x_inputs.size(0)]
        mid_repre_original = mid_repre[x_inputs.size(0):]
        
        dis_output = self.discriminator(x_recons, x_inputs, mode='GAN')
        dis_output_recons = dis_output[:x_inputs.size(0)]
        dis_output_original = dis_output[x_inputs.size(0):]
        
        with torch.no_grad():       # generate recons's content
            x_recons_content = self.generator.ce(x_recons)
       
        # Loss3: image reconstruction loss        
        loss_recons_image = loss_fn_MSE(x_inputs, x_recons) * self.opt.weight_loss_recons_image
        # Loss4: image & recons content feature loss
        loss_recons_content = loss_fn_MSE(x_content, x_recons_content) * self.opt.weight_loss_recons_content
        # Loss5: midle representation loss
        loss_mid_repre = loss_fn_MSE(mid_repre_original, mid_repre_recons) * self.opt.weight_loss_recons_feature       
        # Loss6: GAN loss
        dis_original = -torch.log(dis_output_original + 1e-3)
        dis_recons = -torch.log(1 - dis_output_recons + 1e-3)
        loss_discriminator = torch.mean(dis_original) + torch.mean(dis_recons)
        # decoder total loss     
        loss_decoder = loss_recons_image + loss_mid_repre + loss_recons_content - self.opt.lambda_dis * loss_discriminator
        
        # update
        self.optim_decoder.zero_grad()
        loss_decoder.backward()
        self.optim_decoder.step()
        
        return loss_recons_image, loss_decoder
    # end of train_decoder()
    
    def train_se(self, x_inputs, counter):
        self.discriminator.requires_grad_(False)
        self.generator.se.requires_grad_(True)
        self.generator.decoder.requires_grad_(False)
        
        # forward
        x_content, x_style, x_recons = self.generator(x_inputs)

        mid_repre = self.discriminator(x_recons, x_inputs, mode='RECON')
        mid_repre_recons = mid_repre[:x_inputs.size(0)]
        mid_repre_original = mid_repre[x_inputs.size(0):]
        
        # Loss2: decorrelation loss
        loss_decorr = loss_fn_dCov(x_content, x_style)
        # Loss3: image reconstruction loss        
        loss_recons_image = loss_fn_MSE(x_inputs, x_recons) * self.opt.weight_loss_recons_image
        # Loss5: midle representation loss
        loss_mid_repre = loss_fn_MSE(mid_repre_original, mid_repre_recons) * self.opt.weight_loss_recons_feature       
        # se total loss
        loss_se = loss_recons_image + loss_mid_repre + get_lambda(self.opt.lambda_decorr, self.opt.lambda_schedule, counter+1) * loss_decorr

        # update
        self.optim_se.zero_grad()
        loss_se.backward()
        self.optim_se.step()
        
        return loss_decorr, loss_se
    # end of train_se()

    def train_ce(self, ):    # train content encoder
        write_log(self.opt.logFile, "Training content encoder\n", 'a')
        for epoch in range(1, self.opt.epochs_ce+1):
            gc.collect() # collect garbage
            self.generator.ce.train()   # set model trainable
            self.generator.se.eval()
            self.generator.decoder.eval()
            running_loss = 0.0
            
            trainX, _, trainCF, _ = self.dataset.load_training_data()    # trainCF is content feature not label ####
            train_ds = torch.utils.data.TensorDataset(trainX, trainCF)
            train_ds = torch.utils.data.DataLoader(train_ds, batch_size = self.opt.batchSize, shuffle=True)
            progress = tqdm(train_ds, bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}")
            for image, content_feature in progress:
                # clear gradient 
                self.optim_ce.zero_grad()
                # forward
                content = self.generator.ce(image)
                # calculate loss
                loss = loss_fn_MSE(content, content_feature)
                # backward
                loss.backward()
                # update model
                self.optim_ce.step()
                # record loss
                running_loss += loss.item()               
            # end for
            avg_loss = running_loss / len(progress)
            print(f"Epoch [{epoch}/{self.opt.epochs_ce}], Loss: {avg_loss:.5f}")
            out_str = "Epoch [%d/%d], Loss, %.5f\n" % (epoch, self.opt.epochs_ce, avg_loss)
            write_log(self.opt.logFile, out_str, 'a')
            
            if(loss.item() < self.opt.threshold_ce):
                print(f"Content encoder training break at epochs: {epoch}, loss: {loss.item():.8f}")
                out_str = "Content encoder training break at epochs:%d, loss:%.8f\n" % (epoch, loss.item())
                write_log(self.opt.logFile, out_str, 'a')
                break
        # end for epoch
        
        # save model
        torch.save(self.generator.ce, self.opt.CE_File)    
    # end of train_ce()
    
    def train_adv(self, s_inputs, t_inputs, perturbRateMax):
        self.discriminator.requires_grad_(False)
        self.generator.se.requires_grad_(False)
        self.generator.decoder.requires_grad_(True)
        
        # forward
        _, x_adv = self.generator.get_adv(s_inputs, t_inputs)
        
        with torch.no_grad():
            t_label = torch.argmax(self.classifier(t_inputs), dim=1).numpy()
            x_adv_label = torch.argmax(self.classifier(x_adv), dim=1).numpy()
            perturbRate = np.mean(x_adv_label == t_label)
            weight_loss_adv = max(perturbRateMax - perturbRate, 0) * self.opt.weight_loss_adv

        # Loss7: adv loss        
        loss_adv = loss_fn_MSE(t_inputs, x_adv) * weight_loss_adv
        
 
        # update
        if(loss_adv > 0):
            self.optim_decoder.zero_grad()
            loss_adv.backward()
            self.optim_decoder.step()
            
        return loss_adv  
    # end of train_adv()
#
#    def train_adv(self, s_inputs, t_inputs, update_threshold):
#        self.discriminator.requires_grad_(False)
#        self.generator.se.requires_grad_(False)
#        self.generator.decoder.requires_grad_(True)
#        
#        # forward
#        _, x_adv = self.generator.get_adv(s_inputs, t_inputs)
#        
#        # Loss7: adv loss        
#        loss_adv = loss_fn_MSE(t_inputs, x_adv) * self.opt.weight_loss_adv
#        
#        with torch.no_grad():
#            x_adv_label = torch.argmax(self.classifier(x_adv), dim=1).numpy()
#            t_label = torch.argmax(self.classifier(t_inputs), dim=1).numpy()
#            perturbRate = np.mean(x_adv_label == t_label)
#            if(perturbRate > update_threshold):
#                return 0
# 
#        # update
#        self.optim_decoder.zero_grad()
#        loss_adv.backward()
#        self.optim_decoder.step()
#            
#        return loss_adv  
#    # end of train_adv()
# end of class StyleAdv

def image2code(image, imageW):
    image = (image * 127.5 + 127.5).astype('int32') # scale [-1..1] to [0..255]

    END_CODE = ''
    for i in range(imageW):
        END_CODE += '0'
        
    f_code =  FlowContainer2.image_row2attribute(image[0])
    for j in range(1, image.shape[0]):
        code =  FlowContainer2.image_row2attribute(image[j])
        if(code == END_CODE):
            break
        f_code += code
    # end for
    return f_code
# end of image2code()

def generate_ae(opt, dataset, clf, epochId):
    generator = Generator(content_size=opt.content_size, style_size=opt.style_size, c_in=opt.imageChannel, nf=32, 
                H=opt.imageH, W=opt.imageW, add_noise=opt.add_noise, noise_weight=opt.noise_weight)
    # load model
    generator.ce = torch.load(opt.CE_File)
    generator.se = torch.load(opt.SE_File % (epochId))
    generator.decoder = torch.load(opt.decoderFile % (epochId))
    classifier = clf.get_classifier()
    
    split_num = opt.split_num
    gen_num = opt.gen_num
    source = opt.source
    target = opt.target
    adv_num = 0
    adv_list = []
    adv_set = set()
    src_list = []
    tar_list = []
    iterations = 0
    duplicate = 0
    batch_size = 512
    
    while(adv_num < gen_num):
        iterations += 1
        sourceX, targetX = dataset.load_sampling_data(source, target, num=batch_size)
        temp = torch.zeros(batch_size, split_num, opt.imageChannel, opt.imageH, opt.imageW)      # batch_size*split_num images
        with torch.no_grad():
            for j in range(split_num):
                fusion_ratio = ((j+1) / split_num)
                _, x_adv = generator.get_adv(sourceX, targetX, fusion_ratio=fusion_ratio)
                for i in range(batch_size):
                    temp[i][j] = x_adv[i]
                # end for i
            # end for j
            predict = classifier(temp.view(-1, opt.imageChannel, opt.imageH, opt.imageW))
            predict = torch.nn.functional.softmax(predict, 1).argmax(1).view(batch_size, split_num)
            predict = predict.numpy()
        # end with
        for i in range(batch_size):
            idx = np.where(predict[i] == target)[0]
            if(len(idx) > 0 ):
                adv_image = temp[i][idx[0]]
                code = image2code(adv_image.view(opt.imageH, opt.imageW).numpy(), opt.imageW)
                if not adv_set.__contains__(code):
                    adv_set.add(code)
                    adv_list.append(adv_image)
                    src_list.append(sourceX[i])
                    tar_list.append(targetX[i])
                    adv_num += 1
                else:
                    duplicate += 1
        # end for i
#        print(np.asarray(adv_list).shape)
        print("iterations = %d, number of AE=%d, duplicate=%d"%(iterations, adv_num, duplicate))
    # end of while
    adv_list = np.array(adv_list)
    src_list = np.array(src_list)
    tar_list = np.array(tar_list)
    np.save(opt.advFile % (source, epochId), adv_list)
    np.save(opt.srcFile % (source, epochId), src_list)
    np.save(opt.tarFile % (source, epochId), tar_list)
    
    psnr_val = PSNR().evaluate(torch.tensor(src_list).float(), torch.tensor(adv_list).float())
    mse_val = MSE().evaluate(torch.tensor(src_list).float(), torch.tensor(adv_list).float())
    out_str = "Epoch %d, Class %d, Iterations = %d, Number of AEs = %d, PSNR=%f, MSE=%f\n" % (epochId, source, iterations, adv_num, psnr_val, mse_val)
    write_log(opt.ae_logFile, out_str, 'a')
    print(out_str)
    trainX, _, _, _ = dataset.load_training_data()
    gen_pca(opt.pcaFile % (source, epochId), trainX, [src_list, tar_list, adv_list])
    gen_kde(opt.kdeFile % (source, epochId), trainX, [src_list, tar_list, adv_list])
    
#    with torch.no_grad():
#        predict = classifier(torch.tensor(adv_list).float())
#    predict = torch.argmax(predict, dim=1)
#    idx = np.where(predict.numpy() == target)
#    print("ASR=%f\n" % (len(idx[0])/len(predict)))
## end of generate_ae()

#def generate_matrix_ae(opt, dataset, clf, epochId):
#    generator = Generator(content_size=opt.content_size, style_size=opt.style_size, c_in=opt.imageChannel, nf=32, 
#                H=opt.imageH, W=opt.imageW, add_noise=opt.add_noise, noise_weight=opt.noise_weight)
#    # load model
#    generator.ce = torch.load(opt.CE_File)
#    generator.se = torch.load(opt.SE_File % (epochId))
#    generator.decoder = torch.load(opt.decoderFile % (epochId))
#    classifier = clf.get_classifier()
#    
#    split_num = opt.split_num
#    num = split_num + 3
#    
#    out_images = torch.zeros(num*num, opt.imageChannel, opt.imageH, opt.imageW)
#    
#    # get samples
#    sourceX, targetX = dataset.gen_test_images(num)
#    
#    with torch.no_grad():
#        for i in range(num):
#            base = i * num
#            out_images[base] = sourceX[i]
#            out_images[base+num-1] = targetX[i]
#        # end for
#        for j in range(split_num+1):
#            fusion_ratio = (j / split_num)
#            _, x_adv = generator.get_adv(sourceX, targetX, fusion_ratio=fusion_ratio)
#            for i in range(num):
#                base = i * num
#                out_images[base+(j+1)] = x_adv[i]
#            # end for i
#        # end for j
#
#        predict = classifier(out_images)
#        predict = torch.nn.functional.softmax(predict, 1).argmax(1).tolist()
#        org_label = []
#        for i in range(num):
#            base = i * num
#            temp = [predict[base]] * num
#            org_label += temp
#        # end for
#                
#    # end with
#    
#    ####
#    save_image_with_label(out_images, opt.advImage, labels=predict, labels2=org_label, width=num)
### end of generate_matrix_ae()

def main():
    opt = get_args()
    dataset = Dataset(opt)
    classifier = Classifier(opt, dataset)
    if(opt.trainClassifier):
        write_log(opt.logFile, "Training classifier .....\n", 'a')
        classifier.train_classifier()
        exit(0)
    # end if
    
    style_adv = StyleAdv(opt, dataset, classifier)

    if(opt.genAE >= 0):
        write_log(opt.logFile, "Generate adversarial example images .....\n", 'a')
        generate_ae(opt, dataset, classifier, opt.genAE)
    else:
        style_adv.train()
    # end if    
# end of main

if __name__ == '__main__':
    main()
