
'''
[1] X. Zheng, H. Ding, H. Mamitsuka, and S. Zhu, "Collaborative matrix factorization with multiple similarities for predicting drug-target interaction", KDD, 2013.

'''
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.metrics import auc


class CMF:

    def __init__(self, K=10, lambda_l=0.01, lambda_d=0.01, lambda_t=0.01, max_iter=100):
        self.K = K
        self.lambda_l = lambda_l
        self.lambda_d = lambda_d
        self.lambda_t = lambda_t
        self.max_iter = max_iter

    def fix_model(self, W, intMat, drugMat, targetMat, seed):
        self.num_drugs, self.num_targets = intMat.shape
        self.drugMat, self.targetMat = drugMat, targetMat
        x, y = np.where(W > 0)
        self.train_drugs = set(x.tolist())
        self.train_targets = set(y.tolist())
        if seed is None:
            self.U = np.sqrt(1/float(self.K))*np.random.normal(size=(self.num_drugs, self.K))
            self.V = np.sqrt(1/float(self.K))*np.random.normal(size=(self.num_targets, self.K))
        else:
            prng = np.random.RandomState(seed)
            self.U = np.sqrt(1/float(self.K))*prng.normal(size=(self.num_drugs, self.K))
            self.V = np.sqrt(1/float(self.K))*prng.normal(size=(self.num_targets, self.K))
        self.ones = np.identity(self.K)
        last_loss = self.compute_loss(W, intMat, drugMat, targetMat)
        WR = W*intMat
        for t in xrange(self.max_iter):
            self.U = self.als_update(self.U, self.V, W, WR, drugMat, self.lambda_l, self.lambda_d)
            self.V = self.als_update(self.V, self.U, W.T, WR.T, targetMat, self.lambda_l, self.lambda_t)
            curr_loss = self.compute_loss(W, intMat, drugMat, targetMat)
            delta_loss = (curr_loss-last_loss)/last_loss
            # print "Epoach:%s, Curr_loss:%s, Delta_loss:%s" % (t+1, curr_loss, delta_loss)
            if abs(delta_loss) < 1e-6:
                break
            last_loss = curr_loss

    def als_update(self, U, V, W, R, S, lambda_l, lambda_d):
        X = R.dot(V) + 2*lambda_d*S.dot(U)
        Y = 2*lambda_d*np.dot(U.T, U)
        Z = lambda_d*(np.diag(S)-np.sum(np.square(U), axis=1))
        U0 = np.zeros(U.shape)
        D = np.dot(V.T, V)
        m, n = W.shape
        for i in xrange(m):
            # A = np.dot(V.T, np.diag(W[i, :]))
            # B = A.dot(V) + Y + (lambda_l+Z[i])*self.ones
            ii = np.where(W[i, :] > 0)[0]
            if ii.size == 0:
                B = Y + (lambda_l+Z[i])*self.ones
            elif ii.size == n:
                B = D + Y + (lambda_l+Z[i])*self.ones
            else:
                A = np.dot(V[ii, :].T, V[ii, :])
                B = A + Y + (lambda_l+Z[i])*self.ones
            U0[i, :] = X[i, :].dot(np.linalg.inv(B))
        return U0

    def compute_loss(self, W, intMat, drugMat, targetMat):
        loss = np.linalg.norm(W * (intMat - np.dot(self.U, self.V.T)), "fro")**(2)
        loss += self.lambda_l*(np.linalg.norm(self.U, "fro")**(2)+np.linalg.norm(self.V, "fro")**(2))
        loss += self.lambda_d*np.linalg.norm(drugMat-self.U.dot(self.U.T), "fro")**(2)+self.lambda_t*np.linalg.norm(targetMat-self.V.dot(self.V.T), "fro")**(2)
        return 0.5*loss

    def evaluation(self, test_data, test_label):
        ii, jj = test_data[:, 0], test_data[:, 1]
        scores = np.sum(self.U[ii, :]*self.V[jj, :], axis=1)
        prec, rec, thr = precision_recall_curve(test_label, scores)
        aupr_val = auc(rec, prec)
        fpr, tpr, thr = roc_curve(test_label, scores)
        auc_val = auc(fpr, tpr)
        return aupr_val, auc_val

    def predict_scores(self, test_data, N):
        inx = np.array(test_data)
        return np.sum(self.U[inx[:, 0], :]*self.V[inx[:, 1], :], axis=1)

    def __str__(self):
        return "Model: CMF, K:%s, lambda_l:%s, lambda_d:%s, lambda_t:%s, max_iter:%s" % (self.K, self.lambda_l, self.lambda_d, self.lambda_t, self.max_iter)
