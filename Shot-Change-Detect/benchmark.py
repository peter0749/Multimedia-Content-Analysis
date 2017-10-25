import sys
import numpy as np

def gt_parser(path, ac_range=False):
    if type(path)!=str: raise ValueError('not a string')
    with open(path, 'r') as fp:
        gt = []
        for line in fp:
            cuts = line.split('~')
            try:
                if len(cuts)>1:
                    if ac_range:
                        ds = range(int(cuts[0]), int(cuts[1])+1)
                    else:
                        ds = [int(cuts[0]), int(cuts[1])]
                    gt.extend(ds)
                else:
                    d = int(cuts[0])
                    gt.append(d)
            except:
                continue
        return gt

class benchmark(object):
    def __init__(self, method, truth, min_length):
        self.method = method
        self.total = self.method.get_frame_num()
        self.truth = np.zeros(self.total)
        for i in truth: self.truth[i]=1
        self.min_length = min_length
        self.score = None
        self.ap = None
        self.recall = None
        self.precision = None
        self.threshold = None
    def run(self, plot=True):
        from sklearn.metrics import precision_recall_curve, average_precision_score
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set_style('whitegrid')
        self.score = self.method.get_score()
        self.precision, self.recall, self.threshold = precision_recall_curve(self.truth, self.score)
        self.ap = average_precision_score(self.truth, self.score)
        if plot:
            fig, ax = plt.subplots(dpi=100)
            ax.step(self.recall, self.precision, color='r', alpha=0.8)
            ax.fill_between(self.recall, self.precision, alpha=0.15, color='g')
            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')
            ax.set_title('Average Precision: %.3f'%self.ap)
            plt.show()
