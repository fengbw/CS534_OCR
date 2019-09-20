from __future__ import division
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn import neighbors
from scipy.spatial.distance import cdist
from skimage.measure import label, regionprops, moments, moments_central, moments_normalized, moments_hu
from skimage import io, exposure, morphology
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pickle
import cv2 as cv
import copy

def extractFeatures():
    imageNames = ['a', 'd', 'f', 'h', 'k', 'm', 'n', 'o', 'p', 'q', 'r', 's', 'u', 'w', 'x', 'z']
    #imageNames = ['a']
    Features = []
    featuresLabels = []
    for name in imageNames:
        # Reading an Image File
        img = io.imread(name + '.bmp')
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
        #ret, binary = cv.threshold(img, 0, 255, cv.THRESH_BINARY | cv.THRESH_TRIANGLE)
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
        io.imshow(img_binary)
        ax = plt.gca()
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
                plt.text(maxc, minr, name, bbox=dict(facecolor='white', alpha=0.5))
                ax.add_patch(Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='red', linewidth=1))
        plt.title('Bounding Boxes')
        #plt.savefig('report/' + name + '.png')
        io.show()
    '''
    D = cdist(Features, Features)
    print D
    io.imshow(D)
    plt.title('Distance Matrix')
    io.show()
    '''
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

def test():
    trainFeatures, trainLebels = extractFeatures()
    trainMeans, trainDeviations = normalization(trainFeatures)
    testNames = ['test1', 'test2']
    #testNames = ['test1']
    testFeatures = []
    testLabels = []
    testTruth = []
    correct = 0
    D_copy = np.array(0)
    #textPosition = []
    for i in range(len(testNames)):
        classes, locations = readPkl(testNames[i])
        img = io.imread(testNames[i] + '.bmp')
        #testTruth = ['a']*7+['d']*7+['m']*7+['n']*7+['o']*7+['p']*7+['q']*7+['r']*7+['u']*7+['w']*7
        ret, binary = cv.threshold(img, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
        #ret, binary = cv.threshold(img, 0, 255, cv.THRESH_BINARY | cv.THRESH_TRIANGLE)
        #print ret
        th = ret
        img_binary = (img < th).astype(np.double)
        img_dilation = morphology.binary_dilation(img_binary, selem=None)
        img_erosion = morphology.binary_erosion(img_binary, selem=None)
        img_label = label(img_binary, background=0)
        regions = regionprops(img_label)
        io.imshow(img_binary)
        ax = plt.gca()
        thresholdR = 15
        thresholdC = 15
        for props in regions:
            minr, minc, maxr, maxc = props.bbox
            # Computing Hu Moments and Removing Small Components
            if (maxr - minr) >= thresholdR and (maxc - minc) >= thresholdC:
                #textPosition.append((maxc, minr))
                roi = img_binary[minr:maxr, minc:maxc]
                m = moments(roi)
                cr = m[0, 1] / m[0, 0]
                cc = m[1, 0] / m[0, 0]
                mu = moments_central(roi, cr, cc)
                nu = moments_normalized(mu)
                hu = moments_hu(nu)
                testFeatures.append(hu)
                
                for i in range(7):
                    testFeatures[-1][i] = (testFeatures[-1][i] - trainMeans[i]) / trainDeviations[i]
                D = cdist(testFeatures, trainFeatures)
                #D_copy = copy.deepcopy(D)
                D_index = np.argsort(D, axis=1)
                testLabels.append(trainLebels[D_index[-1][0]])
                
                indexFix = locationFix(locations, minr, minc, maxr, maxc)
                if indexFix is not None:
                    if testLabels[-1] == classes[indexFix]:
                        correct += 1
                
                plt.text(maxc, minr, testLabels[-1], bbox=dict(facecolor='white', alpha=0.5))
                ax.add_patch(Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='red', linewidth=1))
        plt.title('Bounding Boxes')
        io.show()
    print correct, len(testLabels)
    correctRate = correct / len(testLabels)
    print correctRate
       
def readPkl(filename):
    pkl_file = open(filename + '_gt.pkl', 'rb')
    mydict = pickle.load(pkl_file)
    pkl_file.close()
    classes = mydict['classes']
    locations = mydict['locations']
    return classes, locations
    
def locationFix(locations, minr, minc, maxr, maxc):
    index = 0
    for item in locations:
        if item[0] >= minc and item[0] <= maxc and item[1] >= minr and item[1] <= maxr:
            return index
        index += 1
    return
       
def testKNN():
    trainFeatures, trainLebels = extractFeatures()
    knn = neighbors.KNeighborsClassifier()
    knn.fit(trainFeatures, trainLebels)
    #score = knn.score(trainFeatures, trainLebels)
    testNames = ['test1', 'test2']
    #testNames = ['test2']
    testFeatures = []
    testLabels = []
    testTruth = []
    correct = 0
    #textPosition = []
    for i in range(len(testNames)):
        classes, locations = readPkl(testNames[i])
        img = io.imread(testNames[i] + '.bmp')
        #testTruth = ['a']*7+['d']*7+['m']*7+['n']*7+['o']*7+['p']*7+['q']*7+['r']*7+['u']*7+['w']*7
        ret, binary = cv.threshold(img, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
        #ret, binary = cv.threshold(img, 0, 255, cv.THRESH_BINARY | cv.THRESH_TRIANGLE)
        th = ret
        img_binary = (img < th).astype(np.double)
        img_dilation = morphology.binary_dilation(img_binary, selem=None)
        img_erosion = morphology.binary_erosion(img_binary, selem=None)
        img_label = label(img_binary, background=0)
        regions = regionprops(img_label)
        io.imshow(img_binary)
        ax = plt.gca()
        thresholdR = 15
        thresholdC = 15
        for props in regions:
            minr, minc, maxr, maxc = props.bbox
            # Computing Hu Moments and Removing Small Components
            if (maxr - minr) >= thresholdR and (maxc - minc) >= thresholdC:
                #textPosition.append((maxc, minr))
                roi = img_binary[minr:maxr, minc:maxc]
                m = moments(roi)
                cr = m[0, 1] / m[0, 0]
                cc = m[1, 0] / m[0, 0]
                mu = moments_central(roi, cr, cc)
                nu = moments_normalized(mu)
                hu = moments_hu(nu)
                testFeatures.append(hu)
                
                testLabels.append(knn.predict([testFeatures[-1]]))
                
                indexFix = locationFix(locations, minr, minc, maxr, maxc)
                if indexFix is not None:
                    if testLabels[-1] == classes[indexFix]:
                        correct += 1
                
                plt.text(maxc, minr, testLabels[-1][0], bbox=dict(facecolor='white', alpha=0.5))
                ax.add_patch(Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='red', linewidth=1))
        plt.title('Bounding Boxes')
        io.show()
    print correct, len(testLabels)
    correctRate = correct / len(testLabels)
    print correctRate

def main():
    test()
    #testKNN()
    
if __name__ == '__main__':
    main()














