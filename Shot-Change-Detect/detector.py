import os
import numpy as np
import cv2
import reader
from tqdm import tqdm

class BaseDetector(object):
    """
    Base Class of Scene Change Detector
    """
    def __init__(self, directory, img_type, scale):
        self.shot = None
        self._Diff = None
        self._scale=scale
        self._img_type=img_type
        self._path=directory
        if self._img_type=='video':
            self._videoReader = reader.readVideo
        else:
            self._videoReader = reader.readDir
        t = [ _ for _ in self._videoReader(directory,1,self._scale)]
        self._frame_shape = t[0].shape
        del t
        self._shot = None
    def get_score(self):
        return self._Diff
    def get_frame_num(self):
        return len(self._Diff)
    def _pre_process(self):
        pass
    def _post_process(self):
        pass
    def run(self, threshold, min_length):
        pass
    def get_keyframe(self, threshold, output):
        pass

def proxy_method(instance, method, *args):
    return getattr(instance, method)(*args)

class EdgeBased(BaseDetector):
    def __init__(self, directory, img_type='video', scale=None, kernel_size=(5,5), canny_th1=100, canny_th2=200):
        super(EdgeBased, self).__init__(directory, img_type, scale)
        self._canny_th1 = canny_th1
        self._canny_th2 = canny_th2
        self._kernel = np.ones(kernel_size,np.uint8)
        self._Edge = []
        self._Diff = [0.] ## first frame is not a shot change
        self._pre_process()
    def _get_ECR(self, t0, t1):
        diff = t1-t0
        enterER = np.sum(diff>0) / (float(np.sum(t1>0))+1e-8)
        exitER  = np.sum(diff<0) / (float(np.sum(t0>0))+1e-8)
        return max(enterER, exitER)
    def pre(self, v):
        edged = cv2.dilate(cv2.Canny(v,self._canny_th1,self._canny_th2), self._kernel, iterations=1)
        return edged
    def _pre_process(self):
        self._Edge = []
        self._Diff = [0.]
        for v in tqdm(self._videoReader(self._path, scale=self._scale)):
            self._Edge.append(self.pre(v))
            if len(self._Edge) >= 2:
                self._Diff.append(self._get_ECR(self._Edge[-2], self._Edge[-1]))
    def run(self, threshold=0.8, min_length=12):
        self.shot = [] ## a list
        for i, v in enumerate(self._Diff):
            if v>=threshold and (len(self.shot)==0 or i-self.shot[-1]>=min_length):
                self.shot.append(i)
        return self.shot
    
    def select(self, leftf, rightf, cut_n, threshold):
        temp = dict()
        mid = (leftf+rightf)//2
        temp[mid] = cut_n
        keyf = set([mid]) ## first key frame
        ## set() is a hashset, O(1) avg.
        for i in range(leftf, rightf):
            if i==mid: continue
            mind = np.inf
            for v in keyf:
                mind = min(mind, self._get_ECR(self._Edge[i], self._Edge[v]))
            if mind>=threshold:
                temp[i] = cut_n
                keyf.add(i)
        return temp
    
    def get_keyframe(self, threshold, output):
        if self.shot is None: raise Exception('Call run() first!')
        if type(output) != str: raise ValueError('Please pass a string!')
        if not os.path.exists(output):
            os.makedirs(output)
        
        keyframe_index = dict()
        leftf = 0 ## [leftf, rightf)
        for cut_n, cut in enumerate(self.shot):
            rightf = cut
            temp = self.select(leftf, rightf, cut_n, threshold)
            keyframe_index.update(temp)
            leftf = rightf
        for i,f in tqdm(enumerate(self._videoReader(self._path))):
            if i in keyframe_index:
                cv2.imwrite(output+'/'+'keyframe_%d_%d.jpg'%(keyframe_index[i], i), f)

class ContentBased(BaseDetector):
    def __init__(self, directory, bin_shape=(8,4,4), img_type='video', scale=None):
        super(ContentBased, self).__init__(directory, img_type, scale)
        self._Diff = [0.0]
        self._hsvHist = []
        self._bin_shape = bin_shape
        self._pre_process()
    def _dist(self, x, y):
        return np.clip(1. - np.sum(np.minimum(x, y)), 0, 1)
    def conv(self, v, pixn):
        hsv = cv2.cvtColor(v, cv2.COLOR_BGR2HSV).reshape(pixn,3)
        currHist, _ = np.histogramdd(hsv, bins = self._bin_shape) ## color histogram
        currHist /= float(pixn) ## normalize
        return currHist
    def _pre_process(self):
        """
        Use the method from PPT p.5
        """
        self._Diff = [0.0]
        self._hsvHist = []
        pixn = int(self._frame_shape[0]*self._frame_shape[1])
        
        for v in tqdm(self._videoReader(self._path,scale=self._scale)):
            self._hsvHist.append(self.conv(v, pixn))
            if len(self._hsvHist) >= 2:
                self._Diff.append(self._dist(self._hsvHist[-2], self._hsvHist[-1]))
    def _post_process(self):
        pass
    def run(self, threshold=0.8, min_length=12):
        self.shot = [] ## a list
        for i, v in enumerate(self._Diff):
            if v>=threshold and (len(self.shot)==0 or i-self.shot[-1]>=min_length):
                self.shot.append(i)
        return self.shot
    
    def select(self, leftf, rightf, cut_n, threshold):
        temp = dict()
        mid = (leftf+rightf)//2
        temp[mid] = cut_n
        keyf = set([mid]) ## first key frame
        ## set() is a hashset, O(1) avg.
        for i in range(leftf, rightf):
            if i==mid: continue
            mind = np.inf
            for v in keyf:
                mind = min(mind, self._dist(self._hsvHist[i],self._hsvHist[v]))
            if mind>=threshold:
                temp[i] = cut_n
                keyf.add(i)
        return temp
    
    def get_keyframe(self, threshold, output):
        if self.shot is None: raise Exception('Call run() first!')
        if type(output) != str: raise ValueError('Please pass a string!')
        if not os.path.exists(output):
            os.makedirs(output)
        
        keyframe_dict = dict()
        leftf = 0 ## [leftf, rightf)
        for cut_n, cut in enumerate(self.shot):
            rightf = cut
            temp = self.select(leftf, rightf, cut_n, threshold)
            keyframe_dict.update(temp)
            leftf = rightf
        
        for i,f in tqdm(enumerate(self._videoReader(self._path))):
            if i in keyframe_dict:
                cv2.imwrite(output+'/'+'keyframe_%d_%d.jpg'%(keyframe_dict[i], i), f)

class HSV1(ContentBased):
    def __init__(self, directory, bin_shape=(8,4,4), img_type='video', scale=None):
        super(HSV2, self).__init__(directory, bin_shape, img_type, scale)
    def _dist(self, x, y):
        return np.mean(np.abs(x-y))

class HSV2(ContentBased):
    def __init__(self, directory, bin_shape=(8,4,4), img_type='video', scale=None):
        super(HSV2, self).__init__(directory, bin_shape, img_type, scale)
    def _dist(self, x, y):
        return np.mean(np.square(x-y))

class RGBBased(ContentBased):
    def __init__(self, directory, bin_shape=(4,8,7), img_type='video', scale=None):
        super(RGBBased, self).__init__(directory, bin_shape, img_type, scale)
    def conv(self, v, pixn):
        currHist, _ = np.histogramdd(v.reshape(pixn,3), bins = self._bin_shape) ## color histogram
        currHist /= float(pixn) ## normalize
        return currHist
    def _pre_process(self): ## override
        """
        Use the method from PPT p.5
        """
        pixn = int(self._frame_shape[0]*self._frame_shape[1])
        self._Diff = [0.0]
        self._hsvHist = []
        
        for v in tqdm(self._videoReader(self._path,scale=self._scale)):
            self._hsvHist.append(self.conv(v, pixn))
            if len(self._hsvHist) >= 2:
                self._Diff.append(self._dist(self._hsvHist[-2], self._hsvHist[-1]))

class RGB1(RGBBased):
    def __init__(self, directory, bin_shape=(4,8,7), img_type='video', scale=None):
        super(RGB1, self).__init__(directory, bin_shape, img_type, scale)
    def _dist(self, x, y):
        return np.mean(np.abs(x-y))
class RGB2(RGBBased):
    def __init__(self, directory, bin_shape=(4,8,7), img_type='video', scale=None):
        super(RGB2, self).__init__(directory, bin_shape, img_type, scale)
    def _dist(self, x, y):
        return np.mean(np.square(x-y))
