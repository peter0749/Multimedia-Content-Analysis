import numpy as np
import cv2
import reader

class BaseDetector(object):
    """
    Base Class of Scene Change Detector
    """
    def __init__(self, directory, threshold, min_length, img_type):
        if img_type=='video':
            self._frames = reader.readVideo(directory) ## a list of frames
        else:
            self._frames = reader.readfrom(directory) ## a list of frames
        self._frame_shape = self._frames[0].shape
        self._threshold = threshold
        self._min_length = min_length
    def _pre_process(self):
        pass
    def _post_process(self):
        pass
    def run(self):
        pass

class ContentBased(BaseDetector):
    def __init__(self, directory, threshold, min_length, bin_shape=(8,4,4), img_type='video'):
        super(ContentBased, self).__init__(directory, threshold, min_length, img_type)
        self.__hsvDiff = []
        self.__bin_shape = bin_shape
    def _pre_process(self):
        """
        Use the method from PPT p.5
        """
        lastHist = np.zeros(self.__bin_shape) ## h, s, v, 1D
        pixn = self._frame_shape[0]*self._frame_shape[1]
        for v in self._frames:
            hsv = np.array(cv2.split(cv2.cvtColor(v, cv2.COLOR_BGR2HSV))).transpose((1,2,0)).reshape((pixn,3))
            ## rgb -> hsv
            currHist, _ = np.histogramdd(hsv, bins = self.__bin_shape) ## color histogram
            currHist /= float(pixn) ## normalize
            histInst = np.sum(np.minimum(currHist, lastHist))
            self.__hsvDiff.append(histInst)
            lastHist = currHist
        self.__hsvDiff[0] = np.inf ## first frame -> no shot change
    def _post_process(self):
        pass
    def run(self):
        self._pre_process()
        sol = []
        for i, v in enumerate(self.__hsvDiff):
            if v<self._threshold and (len(sol)==0 or i-sol[-1]>=self._min_length):
                sol.append(i)
        return sol
    def test(self):
        pass
