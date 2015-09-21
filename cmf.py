
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.metrics import auc


class CMF:

    def __init__(self, num_factors=10, lmbda=0.01, alpha=0.01, beta=0.01, max_iter=100):
        self.num_factors = num_factors
        self.lmbda = lmbda
        self.alpha = alpha
        self.beta = beta
        self.max_iter = max_iter

    def fix_model(self, W, intMat, drugMat, targetMat, seed):
        self.num_drugs, self.num_targets = intMat.shape
        self.drugMat, self.targetMat = drugMat, targetMat
        x, y = np.where(W > 0)
        self.train_drugs = set(x.tolist())
        self.train_targets = set(y.tolist())
        if seed is None:
            self.U = np.sqrt(1/float(self.num_factors))*np.random.normal(size=(self.num_drugs, self.num_factors))
            self.V = np.sqrt(1/float(self.num_factors))*np.random.normal(size=(self.num_targets, self.num_factors))
        else:
            prng = np.random.RandomState(seed)
            self.U = np.sqrt(1/float(self.num_factors))*prng.normal(size=(self.num_drugs, self.num_factors))
            self.V = np.sqrt(1/float(self.num_factors))*prng.normal(size=(self.num_targets, self.num_factors))
        self.ones = np.identity(self.num_factors)
        last_loss = self.compute_loss(W, intMat, drugMat, targetMat)
        WR = W*intMat
        for t in xrange(self.max_iter):
            self.U = self.als_update(self.U, self.V, W, WR, drugMat, self.lmbda, self.alpha)
            self.V = self.als_update(self.V, self.U, W.T, WR.T, targetMat, self.lmbda, self.beta)
            curr_loss = self.compute_loss(W, intMat, drugMat, targetMat)
            delta_loss = (curr_loss-last_loss)/last_loss
            # print "Epoach:%s, Curr_loss:%s, Delta_loss:%s" % (t+1, curr_loss, delta_loss)
            if abs(delta_loss) < 1e-6:
                break
            last_loss = curr_loss

    def als_update(self, U, V, W, R, S, lmbda, alpha):
        X = R.dot(V) + 2*alpha*S.dot(U)
        Y = 2*alpha*np.dot(U.T, U)
        Z = alpha*(np.diag(S)-np.sum(np.square(U), axis=1))
        U0 = np.zeros(U.shape)
        D = np.dot(V.T, V)
        m, n = W.shape
        for i in xrange(m):
            # A = np.dot(V.T, np.diag(W[i, :]))
            # B = A.dot(V) + Y + (lmbda+Z[i])*self.ones
            ii = np.where(W[i, :] > 0)[0]
            if ii.size == 0:
                B = Y + (lmbda+Z[i])*self.ones
            elif ii.size == n:
                B = D + Y + (lmbda+Z[i])*self.ones
            else:
                A = np.dot(V[ii, :].T, V[ii, :])
                B = A + Y + (lmbda+Z[i])*self.ones
            U0[i, :] = X[i, :].dot(np.linalg.inv(B))
        return U0

    def compute_loss(self, W, intMat, drugMat, targetMat):
        loss = np.linalg.norm(W * (intMat - np.dot(self.U, self.V.T)), "fro")**(2)
        loss += self.lmbda*(np.linalg.norm(self.U, "fro")**(2)+np.linalg.norm(self.V, "fro")**(2))
        loss += self.alpha*np.linalg.norm(drugMat-self.U.dot(self.U.T), "fro")**(2)+self.beta*np.linalg.norm(targetMat-self.V.dot(self.V.T), "fro")**(2)
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

    def evaluation_one(self, test_data, test_label, N=5):
        M = self.U.dot(self.V.transpose())
        S = self.drugMat - np.diag(np.diag(self.drugMat))
        dinx = np.array(list(self.train_drugs))
        DS = S[:, dinx]
        S = self.targetMat - np.diag(np.diag(self.targetMat))
        tinx = np.array(list(self.train_targets))
        TS = S[:, tinx]
        scores = []
        for d, t in test_data:
            ii = np.argsort(DS[d, :])[::-1][:N]
            jj = np.argsort(TS[t, :])[::-1][:N]
            if d in self.train_drugs:
                if t in self.train_targets:
                    scores.append(M[d, t])
                else:
                    scores.append(np.sum(TS[t, jj]*M[d, tinx[jj]])/np.sum(TS[t, jj]))
            else:
                if t in self.train_targets:
                    scores.append(np.sum(DS[d, ii]*M[dinx[ii], t])/np.sum(DS[d, ii]))
                else:
                    v1 = DS[d, ii].dot(self.U[dinx[ii], :])/np.sum(DS[d, ii])
                    v2 = TS[t, jj].dot(self.V[tinx[jj], :])/np.sum(TS[t, jj])
                    scores.append(np.sum(v1*v2))
        prec, rec, thr = precision_recall_curve(test_label, np.array(scores))
        aupr_val = auc(rec, prec)
        fpr, tpr, thr = roc_curve(test_label, np.array(scores))
        auc_val = auc(fpr, tpr)
        return aupr_val, auc_val

    def __str__(self):
        return "Model: CMF, num_factors:%s, lmbda:%s, alpha:%s, beta:%s, max_iter:%s" % (self.num_factors, self.lmbda, self.alpha, self.beta, self.max_iter)

if __name__ == "__main__":
    from functions import *
    cv_setting = 3
    dataset = "e"
    intMat, drugMat, targetMat = load_data_from_file(dataset, "../dataset/")
    if cv_setting == 1:  # CV setting S1
        X, D, T, cv = intMat, drugMat, targetMat, 1
    if cv_setting == 2:  # CV setting S2
        X, D, T, cv = intMat, drugMat, targetMat, 0
    if cv_setting == 3:  # CV setting S3
        X, D, T, cv = intMat.T, targetMat, drugMat, 0

    optimal_para, max_aupr, results = '', 0, []
    seeds = [7771, 8367, 22, 1812, 4659]
    # seeds = np.random.choice(10000, 5, replace=False)
    cv_data = cross_validation(X, seeds, cv)

    for x in np.arange(-2, -1):
        for y in np.arange(-3, -2):
            for z in np.arange(-3, -2):
                import time
                tic = time.clock()
                x, y, z = 0, -3, -3
                model = CMF(num_factors=50, lmbda=2**(x), alpha=2**(y), beta=2**(z), max_iter=30)
                cmd = str(model)
                print "dataset:"+dataset+"\n"+cmd
                aupr_vec, auc_vec = train(model, cv_data, X, D, T)
                aupr_avg, aupr_st = mean_confidence_interval(aupr_vec)
                auc_avg, auc_st = mean_confidence_interval(auc_vec)
                print "AUPR: %s, AUC:%s, AUPRst:%s, AUCst:%s, Time:%s" % (aupr_avg, auc_avg, aupr_st, auc_st, time.clock() - tic)
                if aupr_avg > max_aupr:
                    max_aupr = aupr_avg
                    optimal_para = cmd
                    results = [aupr_avg, auc_avg]
                print time.clock() - tic
                write_metric_vector_to_file(aupr_vec, "../output/cmf_aupr_"+str(cv_setting)+"_"+dataset+".txt")
                write_metric_vector_to_file(auc_vec, "../output/cmf_auc_"+str(cv_setting)+"_"+dataset+".txt")

    # print "Optimal Parameters:\n%s" % optimal_para
    # print "Optimal AUPR: %s, AUC: %s" % (results[0], results[1])
    # with open("../output/cmf_results.txt", "a+") as outf:
    #     outf.write("Dataset:"+dataset+"\n"+optimal_para+"\n"+"AUPR:"+str(results[0])+" AUC:"+str(results[1])+"\n")
