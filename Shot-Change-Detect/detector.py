import numpy as np
import cv2
import reader

class BaseDetector(object):
    """
    Base Class of Scene Change Detector
    """
    def __init__(self, directory, img_type, scale):
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
    def _pre_process(self):
        pass
    def _post_process(self):
        pass
    def run(self, threshold, min_length):
        pass
    def get_keyframe(self, threshold, output):
        pass

class ContentBased(BaseDetector):
    def __init__(self, directory, bin_shape=(8,4,4), img_type='video', scale=None):
        super(ContentBased, self).__init__(directory, img_type, scale)
        self.__hsvDiff = []
        self.__hsvHist = []
        self.__bin_shape = bin_shape
        self._pre_process()
    def _dist(self, x, y):
        return np.sum(np.minimum(x, y))
    def _pre_process(self):
        def conv(v, pixn, tid, results):
            if type(results)!=dict: raise ValueError('please pass dict')
            hsv = cv2.cvtColor(v, cv2.COLOR_BGR2HSV).reshape(pixn,3)
            currHist, _ = np.histogramdd(hsv, bins = self.__bin_shape) ## color histogram
            currHist /= float(pixn) ## normalize
            results[tid] = currHist
        def get_dist(tid, lastHist, currHist, results):
            if type(results)!=dict: raise ValueError('please pass dict')
            results[tid] = self._dist(currHist, lastHist)

        import threading
        thread_queue = []
        thread_results = dict()
        """
        Use the method from PPT p.5
        """
        pixn = self._frame_shape[0]*self._frame_shape[1]
        for tid, v in enumerate(self._videoReader(self._path,scale=self._scale)):
            thread_queue.append(threading.Thread(target=conv, args=(v, pixn, tid, thread_results), name='thread-%d'%tid))
            thread_queue[-1].start()
            ## rgb -> hsv
        # sync
        for tid, th in enumerate(thread_queue):
            th.join()
            self.__hsvHist.append(thread_results[tid])
        thread_queue = []
        thread_results = dict()
        lastHist = np.zeros(self.__bin_shape) ## h, s, v, 1D
        for tid, currHist in enumerate(self.__hsvHist):
            thread_queue.append(threading.Thread(target=get_dist, args=(tid, lastHist, currHist, thread_results), name='thread-%d'%tid))
            thread_queue[-1].start()
            lastHist = currHist
        for tid, th in enumerate(thread_queue):
            th.join()
            self.__hsvDiff.append(thread_results[tid])
        thread_queue = []
        thread_results = []
        self.__hsvDiff[0] = np.inf ## first frame -> no shot change
    def _post_process(self):
        pass
    def run(self, threshold=0.8, min_length=12):
        self.shot = [] ## a list
        for i, v in enumerate(self.__hsvDiff):
            if v<threshold and (len(self.shot)==0 or i-self.shot[-1]>=min_length):
                self.shot.append(i)
        return self.shot
    def get_keyframe(self, threshold, output):
        import threading
        if self.shot is None: raise Exception('Call run() first!')
        if type(output) != str: raise ValueError('Please pass a string!')
        temp = dict()
        def select(left, right, cut_n):
            mid = (leftf+rightf)//2
            temp[mid] = cut_n
            keyf = set([mid]) ## first key frame
            ## set() is a hashset, O(1) avg.
            for i in range(leftf, rightf):
                if i==mid: continue
                maxd = 0
                for v in keyf:
                    maxd = max(maxd, self._dist(self.__hsvHist[i],self.__hsvHist[v]))
                if maxd<threshold:
                    temp[i] = cut_n
                    keyf.add(i)

        thread_queue = []
        leftf = 0 ## [leftf, rightf)
        for cut_n, cut in enumerate(self.shot):
            rightf = cut
            thread_queue.append(threading.Thread(target=select, args=(leftf, rightf, cut_n), name='thread-%d'%(cut_n)))
            thread_queue[-1].start()
            leftf = rightf
        for th in thread_queue:
            th.join()
        del thread_queue
        for i,f in enumerate(self._videoReader(self._path)):
            if i in temp:
                cv2.imwrite(output+'/'+'keyframe_%d_%d.png'%(temp[i], i), f)
