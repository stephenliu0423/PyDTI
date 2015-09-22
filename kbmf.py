
'''
[1] M. Gonen, "Predicting drug-target interactions from chemical and genomical kernels using Bayesian matrix factorization, Bioinformatics, 2012"
'''

import os
import numpy as np
from pymatbridge import Matlab
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.metrics import auc


class KBMF:

    def __init__(self, num_factors=10):
        self.num_factors = num_factors

    def fix_model(self, W, intMat, drugMat, targetMat, seed=None):
        R = W*intMat
        drugMat = (drugMat+drugMat.T)/2
        targetMat = (targetMat+targetMat.T)/2
        mlab = Matlab()
        mlab.start()
        # print os.getcwd()
        # self.predictR = mlab.run_func(os.sep.join([os.getcwd(), "kbmf2k", "kbmf.m"]), {'Kx': drugMat, 'Kz': targetMat, 'Y': R, 'R': self.num_factors})['result']
        self.predictR = mlab.run_func(os.path.realpath(os.sep.join(['../kbmf2k', "kbmf.m"])), {'Kx': drugMat, 'Kz': targetMat, 'Y': R, 'R': self.num_factors})['result']
        # print os.path.realpath(os.sep.join(['../kbmf2k', "kbmf.m"]))
        mlab.stop()

    def predict_scores(self, test_data, N):
        inx = np.array(test_data)
        return self.predictR[inx[:, 0], inx[:, 1]]

    def evaluation(self, test_data, test_label):
        scores = self.predictR[test_data[:, 0], test_data[:, 1]]
        prec, rec, thr = precision_recall_curve(test_label, scores)
        aupr_val = auc(rec, prec)
        fpr, tpr, thr = roc_curve(test_label, scores)
        auc_val = auc(fpr, tpr)
        return aupr_val, auc_val

    def __str__(self):
        return "Model: KBMF, num_factors:%s" % (self.num_factors)
