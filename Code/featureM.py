
# coding: utf-8

# In[1]:

get_ipython().magic(u'pylab inline')


# In[2]:

import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from skimage import io
from skimage import transform, filters, color
import operator


# In[3]:

def drawMatches(img1, kp1, img2, kp2, matches):
    """
    My own implementation of cv2.drawMatches as OpenCV 2.4.9
    does not have this function available but it's supported in
    OpenCV 3.0.0

    This function takes in two images with their associated 
    keypoints, as well as a list of DMatch data structure (matches) 
    that contains which keypoints matched in which images.

    An image will be produced where a montage is shown with
    the first image followed by the second image beside it.

    Keypoints are delineated with circles, while lines are connected
    between matching keypoints.

    img1,img2 - Grayscale images
    kp1,kp2 - Detected list of keypoints through any of the OpenCV keypoint 
              detection algorithms
    matches - A list of matches of corresponding keypoints through any
              OpenCV keypoint matching algorithm
    """

    # Create a new output image that concatenates the two images together
    # (a.k.a) a montage
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]

    out = np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')

    # Place the first image to the left
    out[:rows1,:cols1] = np.dstack([img1, img1, img1])

    # Place the next image to the right of it
    out[:rows2,cols1:] = np.dstack([img2, img2, img2])

    # For each pair of points we have between both images
    # draw circles, then connect a line between them
    for mat in matches:

        # Get the matching keypoints for each of the images
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx

        # x - columns
        # y - rows
        (x1,y1) = kp1[img1_idx].pt
        (x2,y2) = kp2[img2_idx].pt

        # Draw a small circle at both co-ordinates
        # radius 4
        # colour blue
        # thickness = 1
        cv2.circle(out, (int(x1),int(y1)), 4, (255, 0, 0), 1)   
        cv2.circle(out, (int(x2)+cols1,int(y2)), 4, (255, 0, 0), 1)

        # Draw a line in between the two points
        # thickness = 1
        # colour blue
        cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (255, 0, 0), 1)


    # Show the image
    #cv2.imshow('Matched Features', out)
    #cv2.waitKey(0)
    #cv2.destroyWindow('Matched Features')

    # Also return the image if you'd like a copy
    return out


# In[4]:

def smoothImage(image):
    #kernel = np.ones((5, 5), np.float32) / 25
    #return cv2.filter2D(image, -1, kernel)
    return cv2.GaussianBlur(image,(5,5),0)


# In[ ]:




# In[5]:

def addSetToList(files, name):
    files.extend(map(lambda x: '/data/img/' + name + '/' + x, [f for f in os.listdir(os.getcwd() + '/data/img/' + name) if os.path.isfile(os.path.join(os.getcwd() + '/data/img/' + name, f))]))


# In[6]:

def similarityScore(matches):
    score = 0
    for match in matches:
        score += (1 / (match.distance ** 2))
    return score


# In[24]:

def prepareImage(image):
    image =  cv2.resize(image, (128, 128), interpolation = cv2.INTER_CUBIC)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image
    


# In[8]:

orb = cv2.ORB()
sift = cv2.SIFT()
surf = cv2.SURF(400)


# In[9]:

def findMatches(descriptor1, descriptor2, featureMethod):
    if(featureMethod == 'orb'):
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        return bf.knnMatch(descriptor1, descriptor2, k=2)
    if(featureMethod == 'sift'):
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)   # or pass empty dictionary

        flann = cv2.FlannBasedMatcher(index_params,search_params)

        return flann.knnMatch(descriptor1,descriptor2,k=2)
    elif(featureMethod == 'surf'):
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)   # or pass empty dictionary

        flann = cv2.FlannBasedMatcher(index_params,search_params)

        return flann.knnMatch(descriptor1,descriptor2,k=2)


# In[10]:

def getKeypointsAndDescriptor(img, featureMethod):
    if(featureMethod == 'orb'):
        return orb.detectAndCompute(img, None)
    elif(featureMethod == 'sift'):
        return sift.detectAndCompute(img, None)
    elif(featureMethod == 'surf'):
        return surf.detectAndCompute(img, None)


# In[176]:

def computeKeypoints(cardList, featureMethod):
    cardData = {}
    allDescriptors = None
    cardIndex = []
    
    
    print "Started computing keypoints for ", featureMethod

    for fileName, i  in zip(cardList, range(0, len(cardList))):
        img = cv2.imread(os.getcwd() + fileName)
        img = img[36:174, 18:206]
        img = prepareImage(img)
        img = smoothImage(img)

        keypoints, descriptor = getKeypointsAndDescriptor(img, featureMethod)

        
        if allDescriptors is None:
            allDescriptors = np.array(descriptor)
        else:
            allDescriptors = np.concatenate([allDescriptors, descriptor])
            
        for i in range(0, descriptor.shape[0]):
            cardIndex.append(fileName)
        # cardData[0] = keypoints
        # cardData[1] = descriptor
        # cardData[2] = similarityScore
        # cardDate[3] = softMaxScore
        cardData[fileName] = [keypoints, descriptor, 0, 0]

    print "Finished computing keypoints for", featureMethod
    
    return cardData, allDescriptors, cardIndex


# In[ ]:




# In[ ]

# In[172]:

def getCardScores(descriptor, allDescriptors, cardData, cardIndex, method):
    matches =  findGoodMatches(descriptor, allDescriptors, method)
    
    for match in matches:
        cardData[cardIndex[match.trainIdx]][2] += (1 / (match.distance ** 2))
    


# In[13]:

def resetScores(cardData):
    for name in cardData:
        cardData[name][2] = 0


# In[14]:

def findGoodMatches(descriptor1, descriptor2, featureMethod):
    matches = findMatches(descriptor1, descriptor2, featureMethod)
    
    goodMatches = []

    for i in range(0, len(matches)):
        # We only want matches between two descriptors
        if(len(matches[i]) == 2):
            m = matches[i][0]
            n = matches[i][1]
            #D.Lowe's ratio test 
            if m.distance < 1 * n.distance:
                goodMatches.append(m)
    return goodMatches


# In[15]:

def compareAndScoreDescriptors(descriptor1, descriptor2, featureMethod):
    goodMatches = findGoodMatches(descriptor1, descriptor2, featureMethod)
    
    return similarityScore(goodMatches)


# In[16]:

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)


# In[261]:

def findBestMatch(cardImg, cardData, allDescriptors, cardIndex, featureMethod, realName):
    cardImg = prepareImage(cardImg)
    
    keypoints, descriptor = getKeypointsAndDescriptor(cardImg, featureMethod)

    namedScores = {}
    
    scores = np.zeros((len(cardData)))
    i = 0
    
    getCardScores(descriptor, allDescriptors, cardData, cardIndex, featureMethod)
    """
    for cardName in cardData:
        score = compareAndScoreDescriptors(descriptor, cardData[cardName][1], featureMethod)
        cardData[cardName][2] = score
        namedScores[cardName] = score
        scores[i] = score
        i = i + 1
    """
    for card  in cardData:
        factor = 0
        if featureMethod == 'orb':
            factor = 1500
        elif featureMethod == 'surf':
            factor = 0.1
        elif featureMethod == 'sift':
            factor = 1000
        scores[i] = cardData[card][2] * factor
        i = i + 1
    
    softMaxScores = softmax(scores)
    
    #print scores
    #print softMaxScores
    

    
    softMaxValue = np.max(softMaxScores)
    print 'Softmax', softMaxValue
    
    
    sortedNames = sorted(cardData, key=lambda x: cardData[x][2])
    #sortedNames = sorted(namedScores, key=namedScores.get)

    bestMatchCardName = sortedNames[len(sortedNames) - 1]
    bestMatch = cardData[bestMatchCardName]
    #print  "[", featureMethod , "] Best Match: ", os.path.basename(bestMatchCardName) , " Score: ",  bestMatch[2]
    
    
    cardData[bestMatchCardName][3] = softMaxValue
    
    if os.path.basename(bestMatchCardName.split(".")[0]) != realName and realName != 'liveVid':


        bestMatchImg = cv2.imread(os.getcwd() + bestMatchCardName)
        bestMatchImg = bestMatchImg[36:174, 18:206]
        bestMatchImg = prepareImage(bestMatchImg)
        bestMatchImg = smoothImage(bestMatchImg)

        matches = findGoodMatches(descriptor, cardData[bestMatchCardName][1], featureMethod)

        plt.fig = plt.figure(figsize= (20,20))
        io.imshow(drawMatches(cardImg, keypoints, bestMatchImg, cardData[bestMatchCardName][0], matches))
    if realName == 'liveVid':
        return bestMatchCardName, softMaxValue
    return  os.path.basename(bestMatchCardName.split(".")[0])


# In[254]:

def findBestMatchOrb(cardImg):
    cardData = cardDataOrb
    descriptors = desciptorsOrb
    cardIndex = indexOrb
    
    testCardImg = cardImg[0:70, 0:85]
    resetScores(cardData)
        
    bestMatchName = findBestMatch(testCardImg, cardData, descriptors, cardIndex, 'orb', 'liveVid')
    
    return bestMatchName


# In[239]:

def runTest(featureMethod, testSet):
    cardData = {}
    descriptors = []
    cardIndex = {}
    
    if(featureMethod == 'orb'):
        cardData = cardDataOrb
        descriptors = desciptorsOrb
        cardIndex = indexOrb
    elif(featureMethod == 'sift'):
        cardData = cardDataSift
        descriptors = descriptorsSift
        cardIndex = indexSift
    elif(featureMethod == 'surf'):
        cardData = cardDatSurf
        descriptors = desciptorsSurf
        cardIndex = indexSurf
        
    correctMatches = []
    missClassified = []
    lowConfidence = []

    for testCard in testSet:
        testCardImg = cv2.imread(os.getcwd() + testCard)
        testCardImg = testCardImg[0:70, 0:85]

        #print "Finding match for: ", os.path.basename(testCard)
        resetScores(cardData)
        
        bestMatchName = findBestMatch(testCardImg, cardData, descriptors, cardIndex, featureMethod, os.path.basename(testCard.split('.')[0].split("[")[0]))

        softMaxValue = cardData['/data/img/KLD/' + bestMatchName  + '.png'][3]
        
        if softMaxValue < 0.05:
            print 'Too low confidence'
            lowConfidence.append((bestMatchName, os.path.basename(testCard.split('.')[0].split("[")[0]), cardData['/data/img/KLD/' + bestMatchName  + '.png'][3]))
        else:
            if(bestMatchName == os.path.basename(testCard.split('.')[0].split("[")[0])):
                correctMatches.append((bestMatchName, os.path.basename(testCard.split('.')[0].split("[")[0]), cardData['/data/img/KLD/' + bestMatchName  + '.png'][3]))
            else:
                missClassified.append((bestMatchName, os.path.basename(testCard.split('.')[0].split("[")[0]), cardData['/data/img/KLD/' + bestMatchName  + '.png'][3]))


        #print "_____"

    print "Classified ", len(correctMatches), " of ", len(testSet), " correctly"
    print "Classified ", len(missClassified), " of ", len(testSet), " falsy"
    print "Classified ", len(lowConfidence), " of ", len(testSet), "  with to low confidence"

    return correctMatches, missClassified, lowConfidence


# In[249]:

def showPrettyResult(result):
    sizeTestSet = len(result[0]) + len(result[1]) + len(result[2])
    print 'Correctly classified ', len(result[0]), 'of ', sizeTestSet
    print 'Falsly classified ', len(result[1]), 'of ', sizeTestSet
    print 'To low confidence for ', len(result[2]), 'of ', sizeTestSet
    
    
    lowConfCorrect = 0
    lowConfFalse = 0
    for sample in result[2]:
        if(sample[0] == sample[1]):
            lowConfCorrect += 1
        else:
            lowConfFalse += 1
    print 'Of those low confidence cases ', lowConfCorrect, ' would have been correctly classified'
    print 'Of those low confidence cases ', lowConfFalse, ' would have been falsely classified'


# In[19]:

def addFolderToSet(testSet, folderName):
    testSet.extend(map(lambda x: '/data/' + folderName + x, [f for f in os.listdir(os.getcwd() + '/data/' + folderName) if os.path.isfile(os.path.join(os.getcwd() + '/data/' + folderName, f))]))


# In[20]:

# A list of the shorthand codes for all sets currently in the Standard format.

#SETS_IN_STANDARD = ["BFZ", "OGW", "SOI", "EMN", "KLD"]
SETS_IN_STANDARD = ["KLD"]

cardList = []

for setCode in SETS_IN_STANDARD:
    addSetToList(cardList, setCode)


# In[225]:

#cardDataOrb['/data/img/KLD/Harsh Scrutiny.png']


# In[196]:

cardDataOrb, desciptorsOrb, indexOrb = computeKeypoints(cardList, 'orb')
"""
cardDataSift, descriptorsSift, indexSift = computeKeypoints(cardList, 'sift')
cardDatSurf, desciptorsSurf, indexSurf = computeKeypoints(cardList, 'surf')


# In[22]:

testSetNoDistortion = []

addFolderToSet(testSetNoDistortion, 'testSet/')

testSetRotation = []

addFolderToSet(testSetRotation, 'testSet/DISTORT_ROTATION/')

testSetGamma = []

addFolderToSet(testSetGamma, 'testSet/DISTORT_GAMMA/')

testSetGammaRotation = []

addFolderToSet(testSetGammaRotation, 'testSet/DISTORT_GAMMADISTORT_ROTATION/')




# In[233]:

len(indexOrb)
desciptorsOrb.shape


# In[241]:

orbResultStandard = runTest('orb', testSetNoDistortion)
#orbRotation, missClassifiedOrbRotation = runTest('orb', testSetRotation)
#orbGamma, missClassifiedOrbGamma = runTest('orb', testSetGamma)
#orbGammaRotation, missClassifiedOrbGammaRotation = runTest('orb', testSetGammaRotation)


# In[251]:

showPrettyResult(orbResultStandard)


# In[244]:

start_time = time.time()

surfResult = runTest('surf', testSetNoDistortion)
#surfRotation, missClassifiedSurfRotation = runTest('surf', testSetRotation)
#surfGamma, missClassifiedSurfGamma = runTest('surf', testSetGamma)
#surfGammaRotation, missClassifiedSurfGammaRotation = runTest('surf', testSetGammaRotation)

#surfTime = time.time()  - start_time


# In[250]:

showPrettyResult(surfResult)


# In[122]:

start_time = time.time()

#siftStandard, missClassifiedSiftStandard = runTest('sift', testSetNoDistortion)
siftRotation, missClassifiedSiftRotation = runTest('sift', testSetRotation)
siftGamma, missClassifiedSiftGamma = runTest('sift', testSetGamma)
siftGammaRotation, missClassifiedSiftGammaRotation = runTest('sift', testSetGammaRotation)

#siftTime = time.time() - start_time


# In[125]:

print 'ORB'
print 'Standard:', orbStandard
print 'Rotation:', orbRotation
print 'Gamma:', orbGamma
print 'GammRotation:', orbGammaRotation
print 'Time:', orbTime
print '_____'


# In[126]:

print 'SURF'
print 'Standard:', surfStandard
print 'Rotation:', surfRotation
print 'Gamma:', surfGamma
print 'GammRotation:', surfGammaRotation
print 'Time:', surfTime
print '_____'


# In[127]:

print 'SIFT'
print 'Standard:', siftStandard
print 'Rotation:', siftRotation
print 'Gamma:', siftGamma
print 'GammRotation:', siftGammaRotation
print 'Time:', siftTime
print '_____'


# In[35]:

def showMissmatch(missClasified):
    for missMatch in missClasified:
        plt.fig = plt.figure(figsize= (20,20))
        io.imshow('data/testSet/' + missMatch[0])
        plt.fig = plt.figure(figsize= (20,20))
        io.imshow('data/img/KLD/' + missMatch[1] + ".png")


# In[ ]:


testSet = []

testSet.extend(map(lambda x: '/data/testSet/' + x, [f for f in os.listdir(os.getcwd() + '/data/testSet') if os.path.isfile(os.path.join(os.getcwd() + '/data/testSet/', f))]))

correctMatches = 0

for testCard in testSet:
    testCardImg = cv2.imread(os.getcwd() + testCard)
    testCardImg = testCardImg[0:70, 0:85]
    
    print "Finding match for: ", testCard.split("/")[3].split(".")[0].split("[")[0]

    bestMatchName = findBestMatch(testCardImg)
    
    if(bestMatchName == testCard.split("/")[3].split(".")[0].split("[")[0]):
        correctMatches = correctMatches + 1
    
    print "_____"

print "Classified ", correctMatches, " of ", len(testSet), " correctly"    


# In[137]:

resultOrb = np.array([[293, 292, 295, 291, 296, 294, 291, 292, 293, 293], [295, 297, 298, 295, 295, 294, 295, 294, 293, 296], [292, 295, 295, 293, 293, 291, 293, 289, 292, 292]])
resultSurf = np.array([[292, 293, 291, 290, 290, 293, 293, 296, 293, 294], [297, 295, 296, 296, 297, 296, 297, 296, 298], [293, 295, 295, 292, 295, 293, 292, 296, 295, 294]])
resultSift = np.array([[284, 285, 288, 289, 288, 280, 290, 285, 286, 286], [300, 300, 299, 299, 298, 299, 298, 299, 299, 299], [283, 288, 287, 289, 283, 284, 280, 282, 288, 281]])

print 'ORB avg R:', np.average(resultOrb[0])
print 'ORB avg G:', np.average(resultOrb[1])
print 'ORB avg RG:', np.average(resultOrb[2])


print 'SURF avg R:', np.average(resultSurf[0])
print 'SURF avg G:', np.average(resultSurf[1])
print 'SURF avg RG:', np.average(resultSurf[2])


print 'SIFT avg R:', np.average(resultSift[0])
print 'SIFT avg G:', np.average(resultSift[1])
print 'SIFT avg RG:', np.average(resultSift[2])



# In[262]:

imgD = cv2.imread(os.getcwd() + '/data/test/img2.png')
findBestMatchOrb(imgD)


# In[60]:

scores = [6.0, 3.0, 0.0]
print(softmax(scores))


# In[ ]:

"""

