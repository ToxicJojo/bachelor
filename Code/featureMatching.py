
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


# In[7]:

def prepareImage(image):
    image =  cv2.resize(image, (128, 128), interpolation = cv2.INTER_CUBIC)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image
    


# In[ ]:




# In[8]:

# A list of the shorthand codes for all sets currently in the Standard format.

#SETS_IN_STANDARD = ["BFZ", "OGW", "SOI", "EMN", "KLD"]
SETS_IN_STANDARD = ["KLD"]

cardList = []

for setCode in SETS_IN_STANDARD:
    addSetToList(cardList, setCode)


# In[9]:

orb = cv2.ORB()
# Precompute the keypoints for all the images

cardData = {}

print "Started computing keypoints"

for fileName, i  in zip(cardList, range(0, len(cardList))):
    img = cv2.imread(os.getcwd() + fileName)
    img = img[36:174, 18:206]
    img = prepareImage(img)
    img = smoothImage(img)
    
    keypoints, descriptor = orb.detectAndCompute(img, None)

    
    # cardData[0] = keypoints
    # cardData[1] = descriptor
    # cardData[2] = similarityScore
    cardData[fileName] = [keypoints, descriptor, 0]

print "Finished computing keypoints"



# In[18]:

def findGoodMatches(descriptor1, descriptor2):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    
    
    matches = bf.knnMatch(descriptor1, descriptor2, k=2)
    
    goodMatches = []
    
    
    for i in range(0, len(matches)):
        # We only want matches between two descriptors
        if(len(matches[i]) == 2):
            m = matches[i][0]
            n = matches[i][1]
            if m.distance < 0.75 * n.distance:
                goodMatches.append(m)
    return goodMatches


# In[11]:

def compareAndScoreDescriptors(descriptor1, descriptor2):
    goodMatches = findGoodMatches(descriptor1, descriptor2)
    
    return similarityScore(goodMatches)


# In[12]:

def findBestMatch(cardImg):

    cardImg = prepareImage(cardImg)

    keypoints, descriptor = orb.detectAndCompute(cardImg, None)

    namedScores = {}

    for cardName in cardData:
        score = compareAndScoreDescriptors(descriptor, cardData[cardName][1])
        cardData[cardName][2] = score
        namedScores[cardName] = score


    sortedNames = sorted(namedScores, key=namedScores.get)

    bestMatchCardName = sortedNames[len(sortedNames) - 1]
    bestMatch = cardData[bestMatchCardName]
    print  "Best Match: ", bestMatchCardName.split('/')[4], " Score: ",  bestMatch[2]


    bestMatchImg = cv2.imread(os.getcwd() + bestMatchCardName)
    #bestMatchImg = bestMatchImg[36:174, 18:206]
    #bestMatchImg = prepareImage(bestMatchImg)
    #bestMatchImg = smoothImage(bestMatchImg)

    #matches = findGoodMatches(descriptor, cardData[bestMatchCardName][1])

    #plt.fig = plt.figure(figsize= (20,20))
    #io.imshow(drawMatches(cardImg, keypoints, bestMatchImg, cardData[bestMatchCardName][0], matches))
    #return bestMatchCardName.split('/')[4].split(".")[0]
    return bestMatchImg


# In[19]:

"""
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
"""

# In[14]:




# In[22]:

71.0 /73.0


# In[16]:



# In[ ]:



