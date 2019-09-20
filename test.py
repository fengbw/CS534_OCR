from __future__ import division
import numpy as np
from sklearn.metrics import confusion_matrix
from scipy.spatial.distance import cdist
from skimage.measure import label, regionprops, moments, moments_central, moments_normalized, moments_hu
from skimage import io, exposure, morphology
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pickle
import train
import cv2 as cv

#testNames = ['test1', 'test2']
testNames = ['test1']
testFeatures = []
testTruth = []
textPosition = []
for i in range(len(testNames)):
    img = io.imread('Assignment1materials/H1-16images/' + testNames[i] + '.bmp')
    #testTruth = ['a']*7+['d']*7+['m']*7+['n']*7+['o']*7+['p']*7+['q']*7+['r']*7+['u']*7+['w']*7
    ret, binary = cv.threshold(img, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
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
            textPosition.append((maxc, minr))
            roi = img_binary[minr:maxr, minc:maxc]
            m = moments(roi)
            cr = m[0, 1] / m[0, 0]
            cc = m[1, 0] / m[0, 0]
            mu = moments_central(roi, cr, cc)
            nu = moments_normalized(mu)
            hu = moments_hu(nu)
            testFeatures.append(hu)
            ax.add_patch(Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='red', linewidth=1))
    plt.title('Bounding Boxes')
    #io.show()

print textPosition
trainFeatures, trainLebels = train.extractFeatures()
trainMeans, trainDeviations = train.normalization(trainFeatures)
for i in range(7):
    for item in testFeatures:
        item[i] = (item[i] - trainMeans[i]) / trainDeviations[i]

D = cdist(testFeatures, trainFeatures)
D_index = np.argsort(D, axis=1)
testLabels = []
correct = 0
for i in range(len(testFeatures)):
    testLabels.append(trainLebels[D_index[i][0]])
print testLabels
print len(testLabels)

x = []
y = []
labelindex = []
index = 0
for item in textPosition:
    x.append(item[0])
    y.append(item[1])
    labelindex.append(index)
    index += 1
    
for x, y, i in zip(x, y, labelindex):
    plt.text(x, y, testLabels[i], bbox=dict(facecolor='red', alpha=0.5))
io.show()
'''
for i in range(len(testLabels)):
    if testLabels[i] == testTruth[i]:
        correct += 1
print correct
rate = correct / len(testLabels)
print rate
confM = confusion_matrix(testTruth, testLabels)
'''

