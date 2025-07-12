##################################################################
# The program implelents the tools
# 2024.11.05
###################################################################

import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from scipy.stats import gaussian_kde

def check_directory(directory):
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
            print("Add directory %s success." % (directory))
        except OSError:
            print("Add directory %s failed." % (directory))
        pass
    else:
        print("Directory %s exist." % (directory))
# end of make_directory()

def check_file_exist(path):
    ret = os.path.exists(path)
    print("%s exist?" % (path), ret)
    
    return ret
# end of check_file_exist()

def write_log(file_name, out_str, action='a'):
    wfp = open(file_name, action)
    wfp.write(out_str)
    wfp.close()
# end of write_log()

def save_image(img, f_name, width=10, height=None):
    img = img.squeeze(0).permute(0,2,3,1) 
    if height is None:
        height = width
    plt.figure(figsize = (width, height))
    for i in range(height * width):
        plt.subplot(height, width, 1+i)
        plt.imshow(img[i], cmap='gray')
        plt.axis('off')
        
    plt.savefig(f_name)
    plt.close()
# end of save_image()

def save_image_with_label(img, f_name, labels=None, labels2=None, width=10, height=None, errorColor="red"):
    img = img.squeeze(0).permute(0,2,3,1) 
    if height is None:
        height = width
    plt.figure(figsize = (width+2, height+2))
    for i in range(height * width):
        plt.subplot(height, width, 1+i)
        plt.imshow(img[i], cmap='gray')
        plt.xticks([])
        plt.yticks([])
        if labels is not None and labels2 is not None:
            plt.xlabel(str(labels[i]), color=errorColor if labels[i]!=labels2[i] else "black")
        elif labels is not None:
            plt.xlabel(str(labels[i]))
        else:
            plt.xlabel(str(labels2[i]))
    # end for
    
    plt.savefig(f_name)
    plt.close()
# end of save_image_with_label()

def gen_pca(f_name, allImages, imageSet, labelSet=['source', 'target', 'adversarial'], useLable="", colors=None):
    if len(imageSet[0]) < 2:
        return
    
    if colors is None:
        colors = ['blue', 'gray', 'red', 'green', 'purple', 'orange', 'cyan', 'magenta']
        
    for i in range(len(imageSet)):
        if not isinstance(imageSet[i], np.ndarray):
            imageSet[i] = np.array(imageSet[i].numpy())
        imageSet[i] = imageSet[i].reshape(len(imageSet[i]), -1)

    pca = PCA(n_components=2)
#    allImages = np.concatenate(imageSet, axis=0)
    allImages = allImages.reshape(len(allImages), -1)
    pca.fit(allImages)
    
    plt.figure(figsize=(10, 8))
    pcaData = pca.transform(allImages)
    for i in range(len(imageSet)):
        pcaData = pca.transform(imageSet[i])
        
        color_idx = i % len(colors)
        plt.scatter(pcaData[:, 0], pcaData[:, 1], c=colors[color_idx], alpha=0.5, label=f'{labelSet[i]}')
    # end for
    plt.legend()
    plt.grid(True)
    plt.title(f'PCA with {len(imageSet[0])} samples, {useLable}', fontsize=16)
    plt.xlabel('X')
    plt.ylabel('Y')
    
    plt.savefig(f_name, dpi=300)
    plt.clf()
    plt.close()
# end of gen_pca()

def gen_kde(f_name, allImages, imageSet, labelSet=['source', 'target', 'adversarial'], useLable="", colors=None):
    if len(imageSet[0]) < 2:
        return
    
    if colors is None:
        colors = ['blue', 'gray', 'red', 'green', 'purple', 'orange', 'cyan', 'magenta']
    
    for i in range(len(imageSet)):
        if not isinstance(imageSet[i], np.ndarray):
            imageSet[i] = np.array(imageSet[i].numpy())
        imageSet[i] = imageSet[i].reshape(len(imageSet[i]), -1)
    
    pca = PCA(n_components=1)
    allImages = np.concatenate(imageSet, axis=0)
#    allImages = allImages.reshape(allImages.shape[0], -1)
    allImages = allImages.reshape(len(allImages), -1)
    pca.fit(allImages)
    
    plt.figure(figsize=(10, 8))
    for i in range(len(imageSet)):
        pcaData = pca.transform(imageSet[i]).flatten()
        
        x_vals = np.linspace(pcaData.min(), pcaData.max(), len(imageSet[i]))
        kde = gaussian_kde(pcaData)
        pdf = kde(x_vals)
        
        color_idx = i % len(colors)
        plt.plot(x_vals, pdf, colors[color_idx], label=f'{labelSet[i]}', linewidth=2)
    
    plt.legend()
    plt.grid(True)
    plt.title(f'KDE with {len(imageSet[0])} samples, {useLable}', fontsize=16)
    plt.xlabel('X')
    plt.ylabel('Density')
    
    plt.savefig(f_name, dpi=300)
    plt.clf()
    plt.close()
# end of gen_kde()