from scipy.stats import chi2_contingency
import pandas as pd
import numpy as np

def compute_contigency_chi2(df):
    results = chi2_contingency(df)
    p_val = results[1]
    dof = results[2]
    expected_vals = results[3]
    chi2 = results[0]
    cra_v = np.sqrt(chi2/sum((sum(df.values))/min(df.shape[0]-1, df.shape[1]-1)))


    return p_val,dof, pd.DataFrame(expected_vals,index=df.index,columns=df.columns)

    #         active  canceled
# no_reqs    1189     10971
# reqs      25996     31912
# (10791/(10791+1189))/(31912/(25996+31912))

def compute_risk_ratio(df):
    """

    :param df:  Nx2 table with
                    result1 result2
            cat1
            cat2
            ...
            catN
    :return:
    """
    if df.shape == (2,2):
        rr = _2by2_risk(df)
        return rr

    else:
        if df.shape[1] !=2:
            raise NotImplementedError("Only N x 2 tables currently supported")
        else:
            agg_rr = {}
            n_rows = df.shape[0]
            for i in range(n_rows):
                for k in range(i+1,n_rows):
                    sub_df = df.ix[[i,k],:]
                    sub_rr = _2by2_risk(sub_df)
                    agg_rr.update(sub_rr)
            return agg_rr



def _2by2_risk(df):
    row_sums = df.sum(axis=1)
    risk_ratio = (df.ix[0,1]/row_sums[0])/(df.ix[1,1]/row_sums[1])
    risk_ratio = float_trim(risk_ratio,2)
    a,b,c,d = df.ix[0,0],df.ix[0,1],df.ix[1,0],df.ix[1,1]
    rr_se = np.sqrt(((1/a)+(1/c)) - ((1/(a+b)) + (1/(c+d))))
    prefix = "{}/{}_{}/{}".format(df.index[0],df.index[1],df.columns[1],df.columns[0])
    rr = {"{}_risk".format(prefix): risk_ratio}
    ci = _conf_interval(risk_ratio, rr_se,prefix)
    rr.update(ci)
    return rr


def _conf_interval(ratio, std_error,prefix):
    """
    Calculate 95% confidence interval for odds ratio and relative risk.
    """

    _lci = np.log(ratio) - 1.96*std_error
    _uci = np.log(ratio) + 1.96*std_error

    lci = round(np.exp(_lci), 2)
    uci = round(np.exp(_uci), 2)

    return {"{}_lci".format(prefix): lci, "{}_uci".format(prefix): uci}







def chi2_stats(df):
    chi_stats = compute_contigency_chi2(df)
    test_stats = {"p-value": float_trim(chi_stats[0],3),
                  "DoF": int(chi_stats[1])}
    rr = compute_risk_ratio(df)
    test_stats.update(rr)
    ts = pd.Series(test_stats)
    ts.name = "Cohort Stats"
    return ts.to_frame()

def float_trim(value,digits=2):
    return float("{0:.{1}f}".format(value,digits))