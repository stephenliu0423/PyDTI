'''
[1] Xia, Zheng, et al. "Semi-supervised drug-protein interaction prediction from heterogeneous biological spaces." BMC systems biology 4.Suppl 2 (2010): S6.

Default parameters in [1]:
    gamma_d1 = gamma_p1 = 1
    gamma_d2 = gamma_p2 = 0.01
    beta_d = beta_p = 0.3
Default parameters in this implementation:
    gamma_d = 0.01, gamma_d=gamma_d2/gamma_d1
    gamma_t = 0.01, gamma_t=gamma_p2/gamma_p1
    beta_d = 0.3
    beta_t = 0.3
'''
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.metrics import auc


class NetLapRLS:

    def __init__(self, gamma_d=0.01, gamma_t=0.01, beta_d=0.3, beta_t=0.3):
        self.gamma_d = float(gamma_d)
        self.gamma_t = float(gamma_t)
        self.beta_d = float(beta_d)
        self.beta_t = float(beta_t)

    def fix_model(self,  W, intMat, drugMat, targetMat, seed=None):
        R = W*intMat
        m, n = R.shape
        drugMat = (drugMat+drugMat.T)/2
        targetMat = (targetMat+targetMat.T)/2
        Wd = (drugMat+self.gamma_d*np.dot(R, R.T))/(1.0+self.gamma_d)
        Wt = (targetMat+self.gamma_t*np.dot(R.T, R))/(1.0+self.gamma_t)
        Wd = Wd-np.diag(np.diag(Wd))
        Wt = Wt-np.diag(np.diag(Wt))

        D = np.diag(np.sqrt(1.0/np.sum(Wd, axis=1)))
        Ld = np.eye(m) - np.dot(np.dot(D, Wd), D)
        D = np.diag(np.sqrt(1.0/np.sum(Wt, axis=1)))
        Lt = np.eye(n) - np.dot(np.dot(D, Wt), D)

        X = np.linalg.inv(Wd+self.beta_d*np.dot(Ld, Wd))
        Fd = np.dot(np.dot(Wd, X), R)
        X = np.linalg.inv(Wt+self.beta_t*np.dot(Lt, Wt))
        Ft = np.dot(np.dot(Wt, X), R.T)
        self.predictR = 0.5*(Fd+Ft.T)

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
        return "Model: NetLapRLS, gamma_d:%s, gamma_t:%s, beta_d:%s, beta_t:%s" % (self.gamma_d, self.gamma_t, self.beta_d, self.beta_t)
