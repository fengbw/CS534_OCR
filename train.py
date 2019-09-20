from __future__ import division
import numpy as np
from sklearn.metrics import confusion_matrix
from scipy.spatial.distance import cdist
from skimage.measure import label, regionprops, moments, moments_central, moments_normalized, moments_hu
from skimage import io, exposure, morphology
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pickle
import cv2 as cv

def extractFeatures():
    imageNames = ['a', 'd', 'f', 'h', 'k', 'm', 'n', 'o', 'p', 'q', 'r', 's', 'u', 'w', 'x', 'z']
    #imageNames = ['a']
    Features = []
    featuresLabels = []
    for name in imageNames:
        # Reading an Image File
        img = io.imread('Assignment1materials/H1-16images/'+ name + '.bmp')
        #print img.shape
        # Visualizing an Image/Matrix
        '''
        io.imshow(img)
        plt.title('Original Image')
        io.show()
        '''
        # Image Histogram
        '''
        hist = exposure.histogram(img)
        plt.bar(hist[1], hist[0])
        plt.title('Histogram')
        plt.show()
        '''
        # Binarization by Thresholding
        ret, binary = cv.threshold(img, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
        #print ret
        th = ret
        img_binary = (img < th).astype(np.double)
        img_dilation = morphology.binary_dilation(img_binary, selem=None)
        img_erosion = morphology.binary_erosion(img_binary, selem=None)
        # Displaying Binary Image
        '''
        io.imshow(img_binary)
        plt.title('Binary Image')
        io.show()
        '''
        # Connected Component Analysis
        img_label = label(img_binary, background=0)
        '''
        io.imshow(img_label)
        plt.title('Labeled Image')
        io.show()
        print np.amax(img_label)
        '''
        # Displaying Component Bounding Boxes
        regions = regionprops(img_label)
        #io.imshow(img_binary)
        #ax = plt.gca()
        thresholdR = 15
        thresholdC = 15
        for props in regions:
            minr, minc, maxr, maxc = props.bbox
            if (maxr - minr) >= thresholdR and (maxc - minc) >= thresholdC:
                # Computing Hu Moments and Removing Small Components
                roi = img_binary[minr:maxr, minc:maxc]
                m = moments(roi)
                cr = m[0, 1] / m[0, 0]
                cc = m[1, 0] / m[0, 0]
                mu = moments_central(roi, cr, cc)
                nu = moments_normalized(mu)
                hu = moments_hu(nu)
                Features.append(hu)
                featuresLabels.append(name)
                #ax.add_patch(Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='red', linewidth=1))
        #plt.title('Bounding Boxes')
        #io.show()
    return Features, featuresLabels

def normalization(Features):
    means = np.zeros(7)
    deviations = np.zeros(7)
    dataForOneFeature = []
    for i in range(7):
        dataForOneFeature = []
        for item in Features:
            dataForOneFeature.append(item[i])
        means[i] = np.mean(dataForOneFeature)
        deviations[i] = np.std(dataForOneFeature)
    
    for i in range(7):
        for item in Features:
            item[i] = (item[i] - means[i]) / deviations[i]
    return means, deviations
    

def main():
    #featuresLabel = ['a', 'd', 'f', 'h', 'k', 'm', 'n', 'o', 'p', 'q', 'r', 's', 'u', 'w', 'x', 'z']
    features, featuresLabel = extractFeatures()
    #print(features[0])
    normalization(features)
    #print(len(featuresLabel))

if __name__ == '__main__':
    main()
