import pandas as pd
import numpy as np
from sklearn.ensemble.forest import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score,confusion_matrix
from lending_club.loan_prediction.data_transformer import DataTransformer
import pdb


class LoanPredictor(DataTransformer):
    def __init__(self):
        super(LoanPredictor, self).__init__()
        self.model_data = None
        self.features = None
        self.labels = None
        self.sampled_data = None
        self.transform()
        self.label_data()

    def label_data(self):
        mature_loans = self.loan_data.query("loan_status=='Fully Paid' | loan_status=='Charged Off'")
        status_dict = {
            "loan_status":
                {
                    "Fully Paid": 1,
                    "Charged Off": 0
                }
        }
        mature_loans = mature_loans.replace(status_dict)
        self.model_data = mature_loans

    def split_labels(self, data_df):
        features = data_df.drop("loan_status", axis=1)
        labels = data_df.loan_status
        return features, labels

    def undersample(self):
        # random undersampling of majority class
        paid = self.model_data.query("loan_status==1")
        charged_off = self.model_data.query("loan_status==0")
        ratio = len(charged_off) / len(paid)
        paid_undersample = paid.sample(frac=ratio)
        self.sampled_data = pd.concat([paid_undersample, charged_off])

    def _prep_for_modeling(self, undersample=True):
        self.label_data()
        if undersample:
            self.undersample()
            data = self.sampled_data
        else:
            data = self.model_data

        self.features, self.labels = self.split_labels(data)

    def tune_gbc_hyperparameters(self):
        self._prep_for_modeling()
        clf = XGBClassifier(n_estimators=3000)
        params = {
            "learning_rate": (0.2, 0.1, 0.05, 0.02),
            "max_depth": (4, 6, 8, 10),
            "min_child_weight": (2, 3, 4, 6)
        }
        clf = GridSearchCV(clf, param_grid=params, cv=3, scoring='roc_auc', n_jobs=4, verbose=10)
        clf.fit(self.features, self.labels)
        return clf

    def test_train_undersampled_split(self, frac=0.5):
        self.undersample()
        train_undersampled, test_undersampled = train_test_split(self.sampled_data, train_size=0.3,
                                                                 stratify=self.sampled_data.loan_status)
        # do not test on undersampled data, remove train from full data
        test = self.model_data.drop(train_undersampled.index)
        return train_undersampled, test

    def train_test(self):
        train, test = self.test_train_undersampled_split()
        X_train, y_train = self.split_labels(train)
        X_test, y_test = self.split_labels(test)
        self.clf = XGBClassifier(n_estimators=3000, learning_rate=0.1, max_depth=6, min_child_weight=6)
        self.clf.fit(X_train, y_train)
        y_pred = self.clf.predict(X_test)
        prec_score = precision_score(y_test, y_pred)
        rec_score = recall_score(y_test, y_pred)
        cm = confusion_matrix(y_test,y_pred)
        return cm, {"precision": prec_score, "recall": rec_score}


    def feature_importances(self):
        importances = self.clf.feature_importances_
        importances = 100*(importances / importances.max())
        feature_names = self.model_data.drop("loan_status", axis=1).columns
        feat_importances = [k for k in zip(feature_names, importances)]
        feat_importance_df = pd.DataFrame.from_records(feat_importances, columns=["feature", "importance"],
                                                       index="feature")
        feat_importance_df = feat_importance_df.sort_values("importance",ascending=False)
        return feat_importance_df
