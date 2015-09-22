'''
[1] Yong Liu, Min Wu, Chunyan Miao, Peilin Zhao, Xiao-Li Li, "Neighborhood Regularized Logistic Matrix Factorization for Drug-target Interaction Prediction", under review.
'''
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.metrics import auc


class NRLMF:

    def __init__(self, cfix=5, K1=5, K2=5, num_factors=10, theta=1.0, lambda_d=0.625, lambda_t=0.625, alpha=0.1, beta=0.1, max_iter=100):
        self.cfix = int(cfix)  # importance level for positive observations
        self.K1 = int(K1)
        self.K2 = int(K2)
        self.num_factors = int(num_factors)
        self.theta = float(theta)
        self.lambda_d = float(lambda_d)
        self.lambda_t = float(lambda_t)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.max_iter = int(max_iter)

    def AGD_optimization(self, seed=None):
        if seed is None:
            self.U = np.sqrt(1/float(self.num_factors))*np.random.normal(size=(self.num_drugs, self.num_factors))
            self.V = np.sqrt(1/float(self.num_factors))*np.random.normal(size=(self.num_targets, self.num_factors))
        else:
            prng = np.random.RandomState(seed)
            self.U = np.sqrt(1/float(self.num_factors))*prng.normal(size=(self.num_drugs, self.num_factors))
            self.V = np.sqrt(1/float(self.num_factors))*prng.normal(size=(self.num_targets, self.num_factors))
        dg_sum = np.zeros((self.num_drugs, self.U.shape[1]))
        tg_sum = np.zeros((self.num_targets, self.V.shape[1]))
        last_log = self.log_likelihood()
        for t in range(self.max_iter):
            dg = self.deriv(True)
            dg_sum += np.square(dg)
            vec_step_size = self.theta / np.sqrt(dg_sum)
            self.U += vec_step_size * dg
            tg = self.deriv(False)
            tg_sum += np.square(tg)
            vec_step_size = self.theta / np.sqrt(tg_sum)
            self.V += vec_step_size * tg
            curr_log = self.log_likelihood()
            delta_log = (curr_log-last_log)/abs(last_log)
            if abs(delta_log) < 1e-5:
                break
            last_log = curr_log

    def deriv(self, drug):
        if drug:
            vec_deriv = np.dot(self.intMat, self.V)
        else:
            vec_deriv = np.dot(self.intMat.T, self.U)
        A = np.dot(self.U, self.V.T)
        A = np.exp(A)
        A /= (A + self.ones)
        A = self.intMat1 * A
        if drug:
            vec_deriv -= np.dot(A, self.V)
            vec_deriv -= self.lambda_d*self.U+self.alpha*np.dot(self.DL, self.U)
        else:
            vec_deriv -= np.dot(A.T, self.U)
            vec_deriv -= self.lambda_t*self.V+self.beta*np.dot(self.TL, self.V)
        return vec_deriv

    def log_likelihood(self):
        loglik = 0
        A = np.dot(self.U, self.V.T)
        B = A * self.intMat
        loglik += np.sum(B)
        A = np.exp(A)
        A += self.ones
        A = np.log(A)
        A = self.intMat1 * A
        loglik -= np.sum(A)
        loglik -= 0.5 * self.lambda_d * np.sum(np.square(self.U))+0.5 * self.lambda_t * np.sum(np.square(self.V))
        loglik -= 0.5 * self.alpha * np.sum(np.diag((np.dot(self.U.T, self.DL)).dot(self.U)))
        loglik -= 0.5 * self.beta * np.sum(np.diag((np.dot(self.V.T, self.TL)).dot(self.V)))
        return loglik

    def construct_neighborhood(self, drugMat, targetMat):
        self.dsMat = drugMat - np.diag(np.diag(drugMat))
        self.tsMat = targetMat - np.diag(np.diag(targetMat))
        if self.K1 > 0:
            S1 = self.get_nearest_neighbors(self.dsMat, self.K1)
            self.DL = self.laplacian_matrix(S1)
            S2 = self.get_nearest_neighbors(self.tsMat, self.K1)
            self.TL = self.laplacian_matrix(S2)
        else:
            self.DL = self.laplacian_matrix(self.dsMat)
            self.TL = self.laplacian_matrix(self.tsMat)

    def laplacian_matrix(self, S):
        x = np.sum(S, axis=0)
        y = np.sum(S, axis=1)
        L = 0.5*(np.diag(x+y) - (S+S.T))  # neighborhood regularization matrix
        return L

    def get_nearest_neighbors(self, S, size=5):
        m, n = S.shape
        X = np.zeros((m, n))
        for i in xrange(m):
            ii = np.argsort(S[i, :])[::-1][:min(size, n)]
            X[i, ii] = S[i, ii]
        return X

    def fix_model(self, W, intMat, drugMat, targetMat, seed=None):
        self.num_drugs, self.num_targets = intMat.shape
        self.ones = np.ones((self.num_drugs, self.num_targets))
        self.intMat = self.cfix*intMat*W
        self.intMat1 = (self.cfix-1)*intMat*W + self.ones
        x, y = np.where(self.intMat > 0)
        self.train_drugs, self.train_targets = set(x.tolist()), set(y.tolist())
        self.construct_neighborhood(drugMat, targetMat)
        self.AGD_optimization(seed)

    def predict_scores(self, test_data, N):
        dinx = np.array(list(self.train_drugs))
        DS = self.dsMat[:, dinx]
        tinx = np.array(list(self.train_targets))
        TS = self.tsMat[:, tinx]
        scores = []
        for d, t in test_data:
            if d in self.train_drugs:
                if t in self.train_targets:
                    val = np.sum(self.U[d, :]*self.V[t, :])
                else:
                    jj = np.argsort(TS[t, :])[::-1][:N]
                    val = np.sum(self.U[d, :]*np.dot(TS[t, jj], self.V[tinx[jj], :]))/np.sum(TS[t, jj])
            else:
                if t in self.train_targets:
                    ii = np.argsort(DS[d, :])[::-1][:N]
                    val = np.sum(np.dot(DS[d, ii], self.U[dinx[ii], :])*self.V[t, :])/np.sum(DS[d, ii])
                else:
                    ii = np.argsort(DS[d, :])[::-1][:N]
                    jj = np.argsort(TS[t, :])[::-1][:N]
                    v1 = DS[d, ii].dot(self.U[dinx[ii], :])/np.sum(DS[d, ii])
                    v2 = TS[t, jj].dot(self.V[tinx[jj], :])/np.sum(TS[t, jj])
                    val = np.sum(v1*v2)
            scores.append(np.exp(val)/(1+np.exp(val)))
        return np.array(scores)

    def evaluation(self, test_data, test_label):
        dinx = np.array(list(self.train_drugs))
        DS = self.dsMat[:, dinx]
        tinx = np.array(list(self.train_targets))
        TS = self.tsMat[:, tinx]
        scores = []
        if self.K2 > 0:
            for d, t in test_data:
                if d in self.train_drugs:
                    if t in self.train_targets:
                        val = np.sum(self.U[d, :]*self.V[t, :])
                    else:
                        jj = np.argsort(TS[t, :])[::-1][:self.K2]
                        val = np.sum(self.U[d, :]*np.dot(TS[t, jj], self.V[tinx[jj], :]))/np.sum(TS[t, jj])
                else:
                    if t in self.train_targets:
                        ii = np.argsort(DS[d, :])[::-1][:self.K2]
                        val = np.sum(np.dot(DS[d, ii], self.U[dinx[ii], :])*self.V[t, :])/np.sum(DS[d, ii])
                    else:
                        ii = np.argsort(DS[d, :])[::-1][:self.K2]
                        jj = np.argsort(TS[t, :])[::-1][:self.K2]
                        v1 = DS[d, ii].dot(self.U[dinx[ii], :])/np.sum(DS[d, ii])
                        v2 = TS[t, jj].dot(self.V[tinx[jj], :])/np.sum(TS[t, jj])
                        val = np.sum(v1*v2)
                scores.append(np.exp(val)/(1+np.exp(val)))
        elif self.K2 == 0:
            for d, t in test_data:
                val = np.sum(self.U[d, :]*self.V[t, :])
                scores.append(np.exp(val)/(1+np.exp(val)))
        prec, rec, thr = precision_recall_curve(test_label, np.array(scores))
        aupr_val = auc(rec, prec)
        fpr, tpr, thr = roc_curve(test_label, np.array(scores))
        auc_val = auc(fpr, tpr)
        return aupr_val, auc_val

    def __str__(self):
        return "Model: NRLMF, c:%s, K1:%s, K2:%s, r:%s, lambda_d:%s, lambda_t:%s, alpha:%s, beta:%s, theta:%s, max_iter:%s" % (self.cfix, self.K1, self.K2, self.num_factors, self.lambda_d, self.lambda_t, self.alpha, self.beta, self.theta, self.max_iter)
