import os
import glob

import pandas as pd
import arrow





class DataTransformer:
    def __init__(self):
        self.data_path = None
        self.data = None
        self._load_data()



    def transform(self):
        self._drop_id()
        self._drop_future()
        print("handling missing data...")
        self._handle_missing()
        print("transforming features...")
        self._transform_earliest_credit_line()
        self._transform_interest_rate()
        self._drop_features()
        self._map_ordinal_features()
        self._code_categorical_features()

    def _load_data(self):
        dirname, fname = os.path.split(os.path.abspath(__file__))
        path_parts = dirname.split("/")[:-1]
        path_parts.append("data")
        self.data_path = "/".join(path_parts)
        data_files = glob.glob("{0}/*.csv".format(self.data_path))
        for csv in data_files:
            if "LoanStats" in csv:
                self.loan_data = pd.read_csv(csv, skiprows=1)
                self.original_data = self.loan_data.copy()
            else:
                self.income_data = pd.read_csv(csv)
            print("loaded %s" % csv)


    def _drop_id(self):
        # drop uninformative LC ids
        self.loan_data.drop(["id", "member_id"], axis=1, inplace=True)

    def _drop_future(self):
        #features that are post-loan issuance
        post_loan_features = ['acc_now_delinq',
                              'acc_open_past_24mths',
                              'all_util',
                              'avg_cur_bal',
                              'bc_open_to_buy',
                              'bc_util',
                              'chargeoff_within_12_mths',
                              'collection_recovery_fee',
                              'collections_12_mths_ex_med',
                              'delinq_2yrs',
                              'delinq_amnt',
                              'funded_amnt',
                              'funded_amnt_inv',
                              'il_util',
                              'initial_list_status',
                              'inq_fi',
                              'inq_last_12m',
                              'inq_last_6mths',
                              'last_credit_pull_d',
                              'last_pymnt_amnt',
                              'last_pymnt_d',
                              'max_bal_bc',
                              'mo_sin_rcnt_rev_tl_op',
                              'mo_sin_rcnt_tl',
                              'mths_since_last_delinq',
                              'mths_since_last_major_derog',
                              'mths_since_last_record',
                              'mths_since_rcnt_il',
                              'mths_since_recent_bc',
                              'mths_since_recent_bc_dlq',
                              'mths_since_recent_inq',
                              'mths_since_recent_revol_delinq',
                              'next_pymnt_d',
                              'num_accts_ever_120_pd',
                              'num_actv_bc_tl',
                              'num_actv_rev_tl',
                              'num_rev_tl_bal_gt_0',
                              'num_tl_120dpd_2m',
                              'num_tl_30dpd',
                              'num_tl_90g_dpd_24m',
                              'num_tl_op_past_12m',
                              'open_acc_6m',
                              'open_il_12m',
                              'open_il_24m',
                              'open_il_6m',
                              'open_rv_12m',
                              'open_rv_24m',
                              'out_prncp',
                              'out_prncp_inv',
                              'pct_tl_nvr_dlq',
                              'percent_bc_gt_75',
                              'policy_code',
                              'pymnt_plan',
                              'recoveries',
                              'tax_liens',
                              'tot_coll_amt',
                              'tot_cur_bal',
                              'tot_hi_cred_lim',
                              'total_acc',
                              'total_bal_ex_mort',
                              'total_bal_il',
                              'total_bc_limit',
                              'total_cu_tl',
                              'total_il_high_credit_limit',
                              'total_pymnt',
                              'total_pymnt_inv',
                              'total_rec_int',
                              'total_rec_late_fee',
                              'total_rec_prncp',
                              'total_rev_hi_lim',
                              'url',
                              ]
        self.loan_data.drop(post_loan_features, axis=1, inplace=True)

    def _handle_missing(self):
        # drop rows with missing loan_amnt
        self.loan_data.drop(self.loan_data[self.loan_data.loan_amnt.isnull()].index, inplace=True)
        # cast percentage strings to float
        self.loan_data.revol_util = self.loan_data.revol_util.astype(str).apply(lambda x: float(x.strip("%")))
        # impute missing revolving utilization with median
        median_revol_util = self.loan_data.revol_util.median()
        self.loan_data.revol_util.fillna(median_revol_util, inplace=True)
        # fill missing number of revolving accounts with zero
        self.loan_data.num_rev_accts.fillna(0, inplace=True)
        self.loan_data.mo_sin_old_il_acct.fillna(0,inplace=True)

    def _drop_features(self):
        # Description is 99% missing
        self._drop_feature("desc")
        # title is mostly redundant with purpose and is user generated
        self._drop_feature("title")
        # zipcode has high cardinality and a long tail
        self._drop_feature("zip_code")
        # job titles are highly varied, more effort needed to extract signal
        self._drop_feature("emp_title")
        joint_applications = self.loan_data.query("application_type=='JOINT'").index
        # joint applications represent a very small minority, too few to learn from, drop for now
        self.loan_data.drop(joint_applications, inplace=True)
        self.loan_data.drop(["annual_inc_joint", "dti_joint", "verification_status_joint", "application_type"], axis=1, inplace=True)
        #for more granularity use sub-grade instead of grade
        self.loan_data.drop("grade", axis=1,inplace=True)

    def _drop_feature(self, column):
        self.loan_data.drop(column, axis=1, inplace=True)

    def _transform_interest_rate(self):
        self.loan_data.int_rate = self.loan_data.int_rate.astype(str).apply(lambda x: float(x.strip("%")))

    def _transform_earliest_credit_line(self):
        self.loan_data.earliest_cr_line = self.loan_data.earliest_cr_line.apply(self._months_credit)

    def _transform_issue_date(self):
        self.loan_data.issue_d = self.loan_data.issue_d.apply(lambda x: arrow.get(x).month)


    def _months_credit(self,date_str):
        #using now introduces error but is 100x faster than taking the diff between issue_d and earlist_cr_line
        delta = arrow.now() - arrow.get(date_str)
        delta_months = delta.days/30
        return delta_months

    def _map_ordinal_features(self):
        feature_mapping = {
            "emp_length": {
                "10+ years": 10,
                "9 years": 9,
                "8 years": 8,
                "7 years": 7,
                "6 years": 6,
                "5 years": 5,
                "4 years": 4,
                "3 years": 3,
                "2 years": 2,
                "1 year": 1,
                "< 1 year": 0,
                "n/a": 0

            },
            "sub_grade": {
                'A1': 1,
                'A2': 2,
                'A3': 3,
                'A4': 4,
                'A5': 5,
                'B1': 6,
                'B2': 7,
                'B3': 8,
                'B4': 9,
                'B5': 10,
                'C1': 11,
                'C2': 12,
                'C3': 13,
                'C4': 14,
                'C5': 15,
                'D1': 16,
                'D2': 17,
                'D3': 18,
                'D4': 19,
                'D5': 20,
                'E1': 21,
                'E2': 22,
                'E3': 23,
                'E4': 24,
                'E5': 25,
                'F1': 26,
                'F2': 27,
                'F3': 28,
                'F4': 29,
                'F5': 30,
                'G1': 31,
                'G2': 32,
                'G3': 33,
                'G4': 34,
                'G5': 35

            }
        }
        self.loan_data = self.loan_data.replace(feature_mapping)

    def _code_categorical_features(self):

        self.loan_data = pd.get_dummies(self.loan_data,
                                        columns=["verification_status","issue_d","term","home_ownership","purpose","addr_state"])


