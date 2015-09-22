
from functions import *
import scipy.stats as st


for cv in ["1", "2", "3"]:
    print "CVS:"+cv
    for dataset in ["nr", "gpcr", "ic", "e"]:
        nrlmf_auc = load_metric_vector("../output/logmf_auc_"+cv+"_"+dataset+".txt")
        nrlmf_aupr = load_metric_vector("../output/logmf_aupr_"+cv+"_"+dataset+".txt")
        for cp in ["netlaprls", "blm", "wnnrls", "kbmf", "cmf"]:
            cp_auc = load_metric_vector("../output/"+cp+"_auc_"+cv+"_"+dataset+".txt")
            cp_aupr = load_metric_vector("../output/"+cp+"_aupr_"+cv+"_"+dataset+".txt")
            x1, y1 = st.ttest_ind(nrlmf_auc, cp_auc)
            x2, y2 = st.ttest_ind(nrlmf_aupr, cp_aupr)
            print dataset, cp, x1, y1, x2, y2
        print ""
