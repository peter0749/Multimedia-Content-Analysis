import sys
import numpy as np

def gt_parser(path):
    if type(path)!=str: raise ValueError('not a string')
    gt = []
    with open(path, 'r') as fp:
        for line in fp:
            cuts = line.split('~')
            for v in cuts:
                try:
                    gt.append(int(v))
                except:
                    continue
    return gt

class benchmark(object):
    def __init__(self, method, truth, delta, ub, min_length):
        self.method = method
        self.truth = set(truth)
        self.delta = delta
        self.min_length = min_length
        self.upper_bound = ub
        self.ROCx = []
        self.ROCy = []
        self.PRx = []
        self.PRy = []
    def compute_acc(self, threshold):
        positive = set(self.method.run(threshold=threshold, min_length=self.min_length))
        TP = len(positive & self.truth)
        FN = len(self.truth - positive)
        FP = len(positive - self.truth)
        TN = self.method.get_frame_num() - (TP+FN+FP)
        recall = float(TP)/float(TP+FN) ## a.k.a. true positive rate
        precision= float(TP)/float(TP+FP)
        false_positive_rate = float(FP)/(FP+TN)
        return precision, recall, false_positive_rate
    def run(self):
        sys.stderr.write(" t  | p  | r  | f\n")
        sys.stderr.write("-"*16 + "\n")
        r = np.arange(0.0, self.upper_bound, self.delta)
        for d in r:
            try:
                p,r,f = self.compute_acc(d)
            except:
                continue
            sys.stderr.write("%.2f %.2f %.2f %.2f\n" % (d,p,r,f))
            self.ROCx.append(f)
            self.ROCy.append(r)
            self.PRx.append(r)
            self.PRy.append(p)
        import matplotlib as pl
        pl.use('Agg')
        import matplotlib.pyplot as plt
        plt.ylim(0, 1)
        plt.xlim(0, 1)
        plt.plot(self.ROCx, self.ROCy)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC curve')
        plt.savefig('ROC_curve.png')
        plt.cla()
        plt.clf()
        plt.close()
        plt.ylim(0, 1)
        plt.xlim(0, 1)
        plt.plot(self.PRx, self.PRy)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('PR curve')
        plt.savefig('PR_curve.png')
        plt.cla()
        plt.clf()
        plt.close()

