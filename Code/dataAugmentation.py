
# coding: utf-8

# In[2]:

get_ipython().magic(u'pylab inline')


# In[3]:

import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from skimage import io


# In[4]:

def showImg(img):
    # Shows the given image in the notebook.
    # cv2 uses BGR while skimage expects the image in RGB so we need to convert the colorspace
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    io.imshow(img)


# In[5]:

def distortRotation(img, degree):
    """
    Rotates an image.
    
    img - The image that should be rotated
    degree - The amount of rotation in degrees
    
    returns - The rotated image.
    """
    rows,cols, color = img.shape
    
    # Create a 2D rotationsmatrix
    M = cv2.getRotationMatrix2D((cols / 2,rows / 2), degree, 1)
    
    return cv2.warpAffine(img, M, (cols, rows))


# In[6]:

def distortGamma(img, gamma=1.0):
    """
    Changes the gamma value of an image.
    
    img - 
    gamma - The new gamma value
    
    returns - The image with a changed gamma value
    """
    invGamma = 1.0 / gamma
    
    table = np.array([((i / 255.0) ** invGamma) * 255
      for i in np.arange(0, 256)]).astype("uint8")

    return cv2.LUT(img, table)


# In[7]:

distortions = {}


# In[8]:

ROTATION_FACTOR_MIN = 5
ROTATION_FACTOR_MAX = 20

DISTORT_ROTATION = 'DISTORT_ROTATION'

def randomRotation(img):
    """
    Applies a random rotation to an image.
    The random rotation is between 5 to 20 degree.
    
    img - The image that will be rotated
    
    returns - The rotated image.
    """
    rotation = np.random.randint(ROTATION_FACTOR_MIN, ROTATION_FACTOR_MAX)
    if(np.random.random() > .5):
        rotation = rotation * -1

    return distortRotation(img, rotation)

distortions[DISTORT_ROTATION] = randomRotation


# In[9]:

GAMMA_FACTOR_MIN = 0.8
GAMMA_FACTOR_MAX = 1.8

DISTORT_GAMMA = 'DISTORT_GAMMA'

def randomGamma(img):
    """
    Applies a random gamma change to an image.
    The random gamma change is between 0.8 and 1.8
    
    img - The image
    
    
    returns - The gamma corrected image.
    """
    gamma = np.random.uniform(GAMMA_FACTOR_MIN, GAMMA_FACTOR_MAX)
    return distortGamma(img, gamma)

distortions[DISTORT_GAMMA] = randomGamma


# In[10]:

def applyDistortion(img, distortion):
    distortionFunction = distortions[distortion]
    
    return distortionFunction(img)


# In[11]:

def getImagePath(imgPath):
    imgPath.split('/')


# In[12]:

def distortFiles(files, distortionList):
    for cardFile in files:
        img = cv2.imread(os.getcwd() + cardFile)
        
        distortionString = ''
        
        for distortion in distortionList:
            img = applyDistortion(img, distortion)
            distortionString = distortionString + distortion
            
        fileDir = os.getcwd() + os.path.dirname(cardFile) + '/' + distortionString + '/'
        if not os.path.exists(fileDir):
            os.makedirs(fileDir)
        
        #print 'Writing to ' + fileDir + os.path.basename(cardFile)
        cv2.imwrite(fileDir + os.path.basename(cardFile), img) 


# In[13]:




# In[14]:

def loadFileNames(files, path):
    # Adds the names of all files in data/$path to the array files
    files.extend(map(lambda x: '/data/' + path + '/' + x, [f for f in os.listdir(os.getcwd() + '/data/' + path) if os.path.isfile(os.path.join(os.getcwd() + '/data/' + path, f))]))


# In[15]:

cardFiles = []

loadFileNames(cardFiles, 'testSet')


# In[29]:

def createNewTestSet():
    distortFiles(cardFiles, [DISTORT_GAMMA])
    distortFiles(cardFiles, [DISTORT_ROTATION])
    distortFiles(cardFiles, [DISTORT_GAMMA, DISTORT_ROTATION])


# In[20]:




# In[23]:




# In[ ]:



