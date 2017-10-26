import sys
import numpy as np

def gt_parser(path, ac_point=False):
    if type(path)!=str: raise ValueError('not a string')
    with open(path, 'r') as fp:
        gt = []
        for line in fp:
            cuts = line.split('~')
            try:
                if len(cuts)>1:
                    if ac_point:
                        ds = [int(cuts[0]), int(cuts[1])]
                    else:
                        ds = range(int(cuts[0]), int(cuts[1])+1)
                    gt.extend(ds)
                else:
                    d = int(cuts[0])
                    gt.append(d)
            except:
                continue
        return gt

class benchmark_plot_all(object):
    def __init__(self, methods, truth, min_length, frame_num):
        if type(methods)!=dict:
            raise ValueError('must be dict')
        self.methods = methods
        self.total = frame_num
        self.truth = np.zeros(self.total)
        for i in truth: self.truth[i]=1
        self.min_length = min_length
        self.scores = None
    def run(self):
        from sklearn.metrics import precision_recall_curve, average_precision_score
        import matplotlib.pyplot as plt
        import seaborn as sns
        import string
        import colorsys
        sns.set_style('darkgrid')
        sns.set_context('paper')
        self.scores = [ (self.methods[m].get_score(), m) for m in self.methods ]
        fig, ax = plt.subplots(dpi=120)
        colors = []
        for i in xrange(len(self.scores)):
            r,g,b = colorsys.hsv_to_rgb(i/float(len(self.scores)),1,1)
            colors.append((r,g,b))
        for c, result in enumerate(self.scores):
            color = colors[c]
            score, name = result
            precision, recall, threshold = precision_recall_curve(self.truth, score)
            ax.xlim((0,1))
            ax.ylim((0,1))
            ax.step(recall, precision, color=color)

            ax.text(0.85, 0.95-c*0.05, str(name), fontsize=10, color=color)
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        plt.show()

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
        print('Average Precision: %.2f'%self.ap)
        if plot:
            fig, ax = plt.subplots(dpi=100)
            ax.xlim((0,1))
            ax.ylim((0,1))
            ax.step(self.recall, self.precision, color='r', alpha=0.8)
            ax.fill_between(self.recall, self.precision, alpha=0.15, color='g')
            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')
            ax.set_title('Average Precision: %.3f'%self.ap)
            plt.show()
