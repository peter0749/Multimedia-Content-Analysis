import os
import cv2
import sys

class readVideo:
    def __init__(self, videoFN, lim=None, scale=None):
        if not scale is None and type(scale)!=float: raise ValueError('please pass float')
        self.videoFN = videoFN
        self.lim = lim
        self.frame_n = 0
        self.scale=scale
        test = cv2.VideoCapture(videoFN)
        if test.isOpened(): ## test
            s, f = test.read()
            if not s: raise Exception('Invalid video format!')
            if not scale is None:
                f = cv2.resize(f, (0,0), fx=scale, fy=scale) ## subsampling
            self.shape=f.shape
        test.release()
        del test
        self.vid = cv2.VideoCapture(self.videoFN)
    def __enter__(self):
        return self
    def __exit__(self):
        self.vid.release()
    def __iter__(self):
        return self
    def next(self):
        if self.vid.isOpened() and (self.lim is None or self.frame_n<self.lim):
            s, f = self.vid.read()
            if not s:
                raise StopIteration
            self.frame_n += 1
            if not self.scale is None:
                f = cv2.resize(f, (0,0), fx=self.scale, fy=self.scale) ## subsampling
            return f
        raise StopIteration

class readDir:
    def __init__(self, directory, lim=None, scale=None):
        if not scale is None and type(scale)!=float: raise ValueError('please pass float')
        self.scale = scale
        self.directory=directory
        self.lim=lim
        self.frame_n = 0
        self.frameList = os.listdir(directory)
        self.frameList.sort()
        test = cv2.imread(self.directory+'/'+self.frameList[0], cv2.IMREAD_COLOR)
        if not scale is None:
            test = cv2.resize(test, (0,0), fx=scale, fy=scale) ## subsampling
        self.shape = test.shape
        del test
    def __iter__(self):
        return self
    def next(self):
        img = None
        while img is None:
            if ((not self.lim is None) and self.frame_n>=self.lim) or self.frame_n>=len(self.frameList):
                raise StopIteration
            img = cv2.imread(self.directory+'/'+self.frameList[self.frame_n], cv2.IMREAD_COLOR)
            if not self.scale is None:
                img = cv2.resize(img, (0,0), fx=self.scale, fy=self.scale) ## subsampling
            self.frame_n += 1
        return img

