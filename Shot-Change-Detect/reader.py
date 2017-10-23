import os
import cv2
import sys

def readVideo(videoFN):
    if type(videoFN)!=str:
        raise ValueError('Please pass a string!')
    res = []
    vid = cv2.VideoCapture(videoFN)
    while vid.isOpened():
        s, frame = vid.read()
        if not s: break
        res.append(frame)
    vid.release()
    return res

def readfrom(directory):
    if type(directory)!=str:
        raise ValueError('Please pass a string!')
    """
    Read video frames rom directory,
    output raw frame data by a list
    """
    frameList = os.listdir(directory)
    frameList.sort() ## sort by # of frame
    res = []
    for line in frameList:
        img = cv2.imread(directory+'/'+line, cv2.IMREAD_COLOR) ## read a color img
        if not img is None:
            res.append(img)
    return res
