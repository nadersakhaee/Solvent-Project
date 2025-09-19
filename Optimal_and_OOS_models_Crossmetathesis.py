import numpy as np
import sys
import h5py
import time
import pandas as pd
import math
from dataclasses import dataclass, field
from pylab import *
from sklearn.feature_selection import RFE
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_regression
from sklearn import preprocessing
import os 
import re
from sklearn.decomposition import TruncatedSVD, PCA
from scipy.stats.mstats import pearsonr
from sklearn.pipeline import make_pipeline
from sklearn.metrics import make_scorer
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC, SVR
from sklearn import svm,  linear_model, datasets
from numpy import arange
from pandas import read_csv 
from sklearn.linear_model import ARDRegression, BayesianRidge, ElasticNetCV, HuberRegressor, LarsCV, LassoCV, LassoLarsCV, LassoLarsIC, LogisticRegressionCV, OrthogonalMatchingPursuitCV, RidgeCV, SGDClassifier
from sklearn.model_selection import KFold,  train_test_split, cross_val_score
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, RandomForestClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.decomposition import PCA, FastICA, SparseCoder, KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import seaborn as sns
from scipy.sparse import csr_matrix
from sklearn.metrics import confusion_matrix, accuracy_score, mean_absolute_error, r2_score, mean_squared_error 
from sklearn.feature_selection import f_regression, mutual_info_regression, chi2, SelectKBest, GenericUnivariateSelect
from sklearn.feature_selection import SelectFwe, SelectPercentile, SelectFdr, SelectFpr, RFE
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier
from collections import namedtuple
from itertools import product, combinations
from sklearn.linear_model import LinearRegression
from openpyxl.workbook import Workbook
from datetime import datetime
import csv
from joblib import Memory
import h5py



####################################################################################################################################################################
####################################################################################################################################################################

U_Solv_New_121 = [
    "tAmylalcohol", "benzene", "benzylethylether", "triethylamine", "3-4-lutidine", "hexafluorobenzene", "dimethylformamide", "diethyleneglycol", 
    "methylformamide", "hexafluoroisopropanol", "ethanol", "tetramethylurea", "isoamylalcohol", "1-3-Dimethyl-2-imidazolidinone", "1-propanol", 
    "dmso", "2-methoxyethanol", "decalin", "ethyltertbutylether", "methyltetrahydropyran", "gamma-valerolactone", "cumene", "cyclopentanol", 
    "gamma-butyrolactone", "carbontetrachloride", "ethylbenzene", "orthoxylene", "pyridine", "acetonitrile", "cyclohexanol", "toluene", 
    "aceticacid", "trifluorotoluene", "isopropylacetate", "glycerol", "dioxane", "orthodichlorobenzene", "NNdimethylpropyleneurea", "lacticacid", 
    "benzylalcohol", "methyl-2-pyrrolidinone", "tetrahydropyran", "orthodifluorobenzene", "ethylacetate", "2-5-dimethylisosorbide", "formicacid", 
    "isooctane", "3-methyl-2-oxazolidinone", "fluorobenzene", "ethylenecarbonate", "anisole", "dibutylether", "dimethoxybenzene", 
    "methyltertbutylether", "dcm", "trichloroethanol", "hexamethylphosphoramide", "ethyleneglycol", "aniline", "diethyleneglycolmonoethylether", 
    "nitromethane", "cyclopentylmethylether", "methylacetate", "mesitylene", "diisopropylether", "limonene", "trifluoromethoxybenzene", 
    "1-octanol", "pCymene", "dimethylcarbonate", "water", "chloroform", "carbondisulfide", "formamide", "cyrene", "cyclohexane", "1-butanol", 
    "heptane", "sulfolane", "dichloroethylene", "trifluoroethanol", "diethylether", "nitrobenzene", "phenol", "hexachloroacetone", "tButanol", 
    "methanol", "dioxolane", "chlorobenzene", "2-6-lutidine", "trifluoroaceticacid", "butylacetate", "methyltetrahydrofuran", "secbutanol", 
    "isopropanol", "acetone", "propionitrile", "cyclobutanol", "dichloroethane", "cyclohexanone", "propyleneglycol", "aceticanhydride", 
    "tetrahydrofuran", "dimethoxyethane", "ditertbutylether", "ethyllactate", "methylcyclohexane", "propylenecarbonate", "diglyme", 
    "hexane", "dimethylacetamide", "paraxylene", "benzonitrile", "tamylmethylether", "methylethylketone", "pentane", 
    "propylacetate", "tbutylbenzene", "diethyleneglycoldiethylether", "ethylmethylcarbonate", "diethylcarbonate"

]

####################################################################################################################################################################
####################################################################################################################################################################
Start_Time = datetime.now()
# combo_camps =  [ 
    # ('Featurs_Reic','120_Trs_Reic_03np', 'Reic'),
    # ('Featurs_Morgan','120_Trs_Morg_03np', 'Morg'),
    # ('Featurs_Fuku','120_Trs_Fuku_03np', 'Fuku'),
    # ('Featurs_FRM','120_Trs_FRM_03np', 'FRM'),
    # ('Featurs_FR','120_Trs_FR_03np', 'FR'),
    # ('Featurs_FM','120_Trs_FM_03np', 'FM'),
    # ('Featurs_RM','120_Trs_RM_03np', 'RM')        
#  ]

# with open(f"Trsry/Featurs_FRM.csv") as f:
with open(f"nTrsry/Featurs_FRM.csv") as f:
    for line in f:
        row_features_tags = line.strip().split(',')

def r_2_real(x,y):
    fun_ = pearsonr(x,y)
    return fun_[0]**2

def create_dict_data_tag( path ):
    dict_, data_, tags_data_ = {}, [], []
    for solv in U_Solv_New_121:
        with open(path) as f:
            for line in f:
                row = line.strip().split(',')
                if solv == row[0].strip():
                    dict_[row[0].strip()] = [float(x) for x in row[1:]]
                    data_.append(dict_[row[0].strip()])
                    tags_data_.append(row[0].strip())
                else:
                    pass
            np_data_ = np.array(data_, dtype='f' )
    return dict_, data_, tags_data_, np_data_

def create_smp_outofsmp_dictionaries(original_dict, keys_list):
    smp_dict, tags_smp, data_smp, out_of_smp_dict, tags_out_of_smp, data_out_of_smp  = {}, [], [], {}, [], []
    for key, val in original_dict.items():
        if key not in keys_list:
            out_of_smp_dict[key] = original_dict[key]
        else:
            pass
    for key in keys_list:
        if key in original_dict:
            smp_dict[key] = original_dict[key]
        else:
            pass
    for key, val in smp_dict.items():
        tags_smp.append(key)
        data_smp.append(val)
    for key, val in out_of_smp_dict.items():
        tags_out_of_smp.append(key)
        data_out_of_smp.append(val)
    np_data_smp = np.array(data_smp, dtype='f' )
    np_data_out_of_smp = np.array(data_out_of_smp, dtype='f' )
    return smp_dict, tags_smp, data_smp, np_data_smp, out_of_smp_dict, tags_out_of_smp, data_out_of_smp, np_data_out_of_smp

####################################################################################################################################################################
####################################################################################################################################################################
dict_all_x, all_x, tags_all_x, np_all_x = create_dict_data_tag('nTrsry/121_FRM_04np.csv')   ###tytyyty

dict_smp_y, smp_y, tags_smp_y, np_smp_y = create_dict_data_tag('nTrsry/Y_Srv_CM_36.csv')
# dict_smp_y, smp_y, tags_smp_y, np_smp_y = create_dict_data_tag('nTrsry/Y_OOS_CM_121.csv')


dict_smp_x, tags_smp_x, smp_x, np_smp_x, dict_outsmp_x, tags_outsmp_x, outsmp_x, np_outsmp_x = create_smp_outofsmp_dictionaries(dict_all_x, tags_smp_y)
####################################################################################################################################################################
###########################################        These parts do prints just for checking for code cleaning purposes  #############################################
####################################################################################################################################################################
# for i, (key, val) in enumerate(dict_smp_x.items()):
#     print(f"{i}-----------------the key is {key}-----------------dat={val[15]}")   
# for k , tag in enumerate(tags_smp_x):
#     print(f"{k}-----------------the key is {tag}")
# for j , dat in enumerate(smp_x):
#     print(f"{j}-------------------------------xxxxxxxxxxxxxxxxxxxxxxx-----------------------dat={dat[15]}")
# print(f"\n####################################################################################################################################################\n")
# for i, (key, val) in enumerate(dict_outsmp_x.items()):
#     print(f"{i}-----------------the key is {key}-----------------dat={val[15]}")   
# for k , tag in enumerate(tags_outsmp_x):
#     print(f"{k}-----------------the key is {tag}")
# for j , dat in enumerate(outsmp_x):
#     print(f"{j}-------------------------------xxxxxxxxxxxxxxxxxxxxxxx-----------------------dat={dat[15]}")
# for i, (key, val) in enumerate(dict_all_x.items()):
#     print(f"{i}-----------------the key is {key}-----------------dat={val[20]}")   
# for k , tag in enumerate(tags_all_x):
#     print(f"{k}-----------------the key is {tag}")
# for j , dat in enumerate(all_x):
#     print(f"{j}-------------------------------xxxxxxxxxxxxxxxxxxxxxxx-----------------------dat={dat[20]}")
# print(f"\n####################################################################################################################################################\n")
# for i, (key, val) in enumerate(dict_smp_y.items()):
#     print(f"{i}-----------------the key is {key}-----------------dat={val}")   
# for k , tag in enumerate(tags_smp_y):
#     print(f"{k}-----------------the key is {tag}")
# for j , dat in enumerate(smp_y):
#     print(f"{j}-------------------------------xxxxxxxxxxxxxxxxxxxxxxx-----------------------dat={dat}")
# print(f"\n####################################################################################################################################################\n")
####################################################################################################################################################################
####################################################################################################################################################################
####################################################################################################################################################################
####################################################################################################################################################################
####################################################################################################################################################################
####################################################################################################################################################################
# models_code = ['3', '8', '9', '10', '11']
# models_code = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11']
# models_code = [ '3', '4', '8', '9', '10', '11']
# models_code = [ '1','2','3','7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18']
# models_code = [ '3','7', '9', '13', '14', '18']
models_code = [ '8','9','11','12','14','18']
# models_code = [ '11', '12', '18']
# models_code = [ '14', '18']
# models_code = [ '11', '15', '17']
# models_code = ['12', '8']
# models_code = ['1', '7', '11']
###################################### These are the models to be screened   Namedtuple makes tuples much more legible #####################################
# ppww1 = len(models_code)


model = namedtuple("model", ["code", "name", "model"])
models = [
    model('1', 'SVR2', SVR(kernel='linear')),
    model('2', 'lin-Lasso', linear_model.Lasso(tol=0.00001, max_iter=100000)),
    model('3', 'lars-Lasso', LassoLarsCV(cv=7, n_jobs=-1, max_iter=10000)),
    model('4', 'GrBoost', GradientBoostingRegressor(n_estimators=1000, learning_rate=0.0015, max_depth=2, loss='absolute_error')),
    model('5', 'Rand-Forest', RandomForestRegressor(n_estimators=1000, random_state=0)),
    model('6', 'Decis-Tree', DecisionTreeRegressor(max_depth=3)),
    model('7', 'Elastic-Net', ElasticNetCV(cv=7, max_iter=100000, tol=0.00001)),
    model('8', "linRdige", linear_model.Ridge(alpha=0.8)),
    model('9', "RidgeCV", RidgeCV(cv=7)),
    model('10', 'Bay-Ridge', BayesianRidge()),
    model('11', "ARD", ARDRegression()),
    model('12', "Huber", HuberRegressor(max_iter=1000)),
    model('13', 'LarsCV', LarsCV(cv=7)),
    model('14', "LassoCV", LassoCV(cv=7)),
    model('15', 'LassoLarsIC', LassoLarsIC(criterion='aic', max_iter=100 )),
    model('16', 'LogisticRegressionCV', LogisticRegressionCV(cv=7)),
    model('17', 'ModifiedHuber', SGDClassifier(loss="modified_huber")),
    model('18', "OrthogMP", OrthogonalMatchingPursuitCV(cv=7))
]


####################################################################################################################################################################
####################################################################################################################################################################
# c5_rests = [
#     # ([0,11,17,28,20,30,12,31,8], "Small_Train_CM", ),   #### old
#     ([0,1,3,5,12,13,16,18,19,22,23,24,25,35], "Small_Train_CM", ),

#     ([5,7,8,10,11,14,15,16,17,19,20,21,23,25,26,29,32,33 ], "Half_Train_CM_", ),  #### old
#     ([ 8,4,31,3,16,18,29,32,23,20,9,28,0,1,7,24,15,17], "Half_Train_CM", ),

#     # ([0,1,2,5,7,10,11,12,13,14,15,17,18,19,20,21,23,24,26,27,28,29,30,31,32,33,34 ], "Big_Train_CM", ),   #### old
#     ([0,1,2,3,4,6,7,8,9,10,11,14,15,16,17,18,20,21,23,26,27,28,29,30,31,32,33,34,35 ], "Big_Train_CM", ),

# ]

c5_rests = [
    # ([ 0,3,8,9,10,19,28,33,36,43,50,51,54,55,60,61,67,70,72,81,82,89,106,115,117,120 ], "all_Solv_CM", ),
    # ([ 51,54,28,36,70,106,117,9,0,10,3,82,33,19,72,43,81,67,8,46,109,89,20,45,77,105,99,63 ], "all_Solv_CM", ),
    # ([ 117, 19, 82, 3, 8, 9, 67, 70, 106, 46, 92, 110, 78, 60, 34, 6, 0, 105, 89, 63, 45, 36, 28 ], "Union_Gold", ),
    ([ 2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,18,19,20,21,22,23,26,27,28,29,30,31,32,33,34 ], "all", ),


]
########################################################################
# c5_rests = [
#     ([ 1,2,3,7,8,10,13,15 ], "Small_Train_Hk", ),
#     ([ 1,2,3,5,6,8,9,10,11,14,15,16,20 ], "Half_Train_Hk", ),
#     ([ 1,2,3,5,6,8,9,10,11,12,13,14,15,16,17,20 ], "Big_Train_Hk", ),
# ]

# c5_rests = [
#     ([ 0,15,105,73,71,63,60,36,56,6,19,92,35,89,61,18,26,117,45,79,28,112,107,37,74,58,78,12,93,52,34,29,110 ], "all_Solv_Hk", ),
# ]
########################################################################
# c5_rests = [
#     ([7,12,15,16,17,18,19,21,23], "Small_Train_CHactiv", ),
#     ([1,2,3,7,10,11,14,16,17,18,19,21,23,24,25], "Half_Train_CHactiv", ),
#     ([1,2,3,4,6,7,8,9,10,11,14,15,16,17,18,19,21,22,23,24,25,27], "Big_Train_CHactiv", ),
# ]

# c5_rests = [
#     ([24,8,9,78,70,55,19,48,34,22,6,82,120,1,59,94,110,46,117,16,67,106,66,95,60,31,3,103,28,13,7 ], "all_Solv_CHactiv", ),
# ]


# tup[1] = f"{len(combo_picked)}_opt_FRM_desc_Preds"   ###tytyyty

####################################################################################################################################################################
####################################################################################################################################################################
############################################################# OpenPlotter then Model Plot domino     ###############################################################
####################################################################################################################################################################

min_max_scalerM = preprocessing.MinMaxScaler()
# scl_x_outsmp    = min_max_scalerM.fit_transform(np_outsmp_x)
tot_lis_y_pred_outsmp, tot_lis_y_pred_train, tot_lis_y_pred_test, tot_lis_y_obsv_train, tot_lis_y_obsv_test,   = [], [], [], [], []
tot_lis_smp_train_id, tot_lis_smp_test_id, lis_trn_id, lis_tst_id, pw_cod = [], [], [], [], []

for pw1, tup in enumerate(c5_rests):

    ##The 'loss' parameter of GradientBoostingRegressor must be a str among {'quantile', 'huber', 'absolute_error', 'squared_error'}   
    nnn = len(tags_smp_x)
    results_1 = { "Best_X": [], "r2_train": [], "delta":[],  "Q2":[], "MAE_train":[], "RMSE_train":[], "r2_test": [], "MAE_test":[], "RMSE_test":[], "Bias":[]}
    # results_2 = {"Coeffs":[] }
    results_3 = {"Rank":[] }
    results_4 = {"avg_y_pred_outsmp":[] }
    results_5 = {"tot_Y_pred_outsmp":[] }
    Best_x_lis, r2_train_lis, delta_lis, Q2_lis, MAE_train_lis, RMSE_train_lis, r2_test_lis, MAE_test_lis, RMSE_test_lis, Bias_lis, Coeffs_lis, rank_lis  = [], [], [], [], [], [], [], [], [], [], [], []

    with rc_context(fname="Plots/plot_settings.yml"):
        train_indices = np.array(tup[0][:])
        test_indices = np.array([i for i in range(nnn) if i not in train_indices])            
        # Annotate each point with its index
        train_id_tag = []
        for train_id in train_indices:
            train_id_tag.append(str(train_id))
            # print(train_id)
            # Split data based on indices
            x_train, x_test = np_smp_x[train_indices], np_smp_x[test_indices]
            y_train, y_test = np.array(np_smp_y[train_indices]).ravel(), np.array(np_smp_y[test_indices]).ravel()
            # y_train, y_test = np_smp_y[train_indices], np_smp_y[test_indices]
            # Yr = np.array(Yr1).ravel()

            # # apply Dimensionality Reduction
            # sPca = TruncatedSVD(n_components= 38*j)
            # sPca.fit(sXr, Yr)
            # scl_x_train1= sPca.transform(sXr)
            # scl_x_test1= sPca.transform(sXt)
            scl_x_train     = min_max_scalerM.fit_transform(x_train)
            scl_x_test      = min_max_scalerM.transform(x_test) 
            w3 = 0
    ##############################################################################################
        for w2, m in enumerate(models):
            scorer = make_scorer(r_2_real)
            if m.name not in ["linRdige"]:
                print(f"Oh Shit---- its {m.name}")
                pass
            else:
                plt.figure()  # Create a new figure for each plot
                w3 += 1
                m.model.fit(scl_x_train, y_train)
                y_pred_train = m.model.predict(scl_x_train)
                y_pred_test = m.model.predict(scl_x_test)
                select = RFE(m.model, n_features_to_select=3, step=1)
                selector = select.fit(scl_x_train, y_train)
                ranks = selector.ranking_
                # Coefficients (contribution) of each feature
                # coefficients = m.model.coef_          
                # coefficients = m.model.feature_importances_
                intercept = np.average(m.model.intercept_)
                lis_trn_id.append(train_indices)
                lis_tst_id.append(test_indices)
                pw_cod.append(pw1)              
                q2 = cross_val_score(m.model, scl_x_train, y_train, cv=2, scoring=scorer).mean()
                r2_r = r_2_real(y_train, y_pred_train)
                r2_t = r_2_real(y_test, y_pred_test)
                r2_rs = r2_score(y_train, y_pred_train)
                r2_ts = r2_score(y_test, y_pred_test)
                mae_r = mean_absolute_error(y_train, y_pred_train)
                mae_t = mean_absolute_error(y_test, y_pred_test)
                rmse_r = mean_squared_error(y_train, y_pred_train, squared=False)
                rmse_t = mean_squared_error(y_test, y_pred_test, squared=False)
                Best_x_lis.append(m.name)
                r2_train_lis.append(r2_r)
                delta_lis.append(r2_r-r2_t)
                Q2_lis.append(q2)
                MAE_train_lis.append(mae_r)
                RMSE_train_lis.append(rmse_r)
                r2_test_lis.append(r2_t)
                MAE_test_lis.append(mae_t)
                RMSE_test_lis.append(rmse_t)
                Bias_lis.append(intercept)
                # Coeffs_lis.append(coefficients)
                rank_lis.append(ranks)
                # Coeffs_lis.append(coefficients)
                # y_pred_outsmp = m.model.predict(scl_x_outsmp)
                # tot_lis_y_pred_outsmp.append(y_pred_outsmp)
                tot_lis_y_pred_train.append(y_pred_train)
                tot_lis_y_pred_test.append(y_pred_test)
                tot_lis_y_obsv_train.append(y_train)
                tot_lis_y_obsv_test.append(y_test)
                tot_lis_smp_train_id.append(train_indices)
                tot_lis_smp_test_id.append(test_indices)                     
##############################################################################################
                for z, train_str_tag in enumerate(train_id_tag):
                    plt.annotate(train_str_tag, (y_train[z], y_pred_train[z]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=3)
                test_id_tag = []
                for test_id in test_indices:
                    test_id_tag.append(str(test_id))
                    # print(test_id)
                for z, test_str_tag in enumerate(test_id_tag):
                    plt.annotate(test_str_tag, (y_test[z], y_pred_test[z]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=3)                      
                # Actual plotting 
                plt.scatter(y_train, y_pred_train, color='blue', alpha=0.3, marker='o', s=15, label=f"Train: $R^2={r2_r:.3f}, Delta={(float(r2_r) - float(r2_t)):.3f}$\n $Q^2={q2:.3f}$, mae={mae_r:.3f}, rmse={rmse_r:.3f}")
                plt.scatter(y_test, y_pred_test, color='red', marker='<', s=15, label=f"Test: $R^2 ={r2_t:.3f}$\n mae={mae_t:.3f},  rmse={rmse_t:.3f}, Bias={intercept:.1f}")
                #Actual Y out of Sample gathering
                plt.title(f" Solvent_{tup[1]}, Sourav_data_ ")
                # plt.xlabel("Observed yields for Cross Metathesis (Y_Observed Data)  ")
                # plt.ylabel("Predicted yields for Cross Metathesis (Y_Pred's Data) ")
                plt.xlabel("Observed yields for Cross Metathesis   (Sourav's Data)  ")
                plt.ylabel("Predicted yields for Cross Metathesis ")
                plt.ylim(25, 95)
                plt.xlim(25, 95)  
                ################################################################################################################
                # plt.xlabel("Observed yields for Heck (Y_Observed Data)  ")
                # plt.ylabel("Predicted yields for Heck (Y_Pred's Data) ")     
                # plt.xlabel("Observed yields for Heck (Y_Trust's Data)  ")
                # plt.ylabel("Predicted yields for Heck (Y_Pred's Data) ")    
                # plt.ylim(0, 100)
                # plt.xlim(0, 100)
                ################################################################################################################
                # plt.xlabel("Observed yields for CH-activation (Y_Observed Data)  ")
                # plt.ylabel("Predicted yields for CH-activation (Y_Pred's Data) ") 
                # plt.xlabel("Observed yields for CH-activation (Y_Trust's Data)  ")
                # plt.ylabel("Predicted yields for CH-activation (Y_Pred's Data) ")                     
                # plt.ylim(-20, 85)
                # plt.xlim(-20, 85)  
                ################################################################################################################
                plt.legend()                 
                plt.savefig(f"Plots/{tup[1]}/depo/PltId_{pw1}_{m.name}.svg", dpi=900)
                plt.savefig(f"Plots/{tup[1]}/PltId_{pw1}_{m.name}.png", dpi=900)
                plt.close()                  
                # tot_np_y_pred_outsmp = np.array(tot_lis_y_pred_outsmp, dtype='f' )
                # tot_np_y_obsv_train = np.array(tot_lis_y_obsv_train, dtype='f' )
                # tot_np_y_obsv_test = np.array(tot_lis_y_obsv_test, dtype='f' )    
                # tot_np_y_pred_train = np.array(tot_lis_y_pred_train, dtype='f' )
                # tot_np_y_pred_test = np.array(tot_lis_y_pred_test, dtype='f' )
                ################################################################################################################
                tot_lis_smp_id = tot_lis_smp_train_id[0].tolist() + tot_lis_smp_test_id[0].tolist()
                tot_lis_y_obsv = tot_lis_y_obsv_train[0].tolist() + tot_lis_y_obsv_test[0].tolist()
                tot_lis_y_pred = tot_lis_y_pred_train[0].tolist() + tot_lis_y_pred_test[0].tolist()
                # tot_y_obsv_lis = tot_lis_y_obsv_train + tot_lis_y_obsv_test
                # tot_y_obsv = np.concatenate([tot_np_y_obsv_train, tot_np_y_obsv_test])
                # tot_y_pred = np.concatenate([tot_np_y_pred_train, tot_np_y_pred_test])
                tot_np_y_smp = np.array( [tot_lis_smp_id, tot_lis_y_obsv, tot_lis_y_pred ], dtype='f')
                # print(np.shape(tot_np_y_pred_outsmp))
                tot_np_rank = np.array(rank_lis, dtype='f' )
                # avg_y_pred_outsmp = np.sum(tot_np_y_pred_outsmp, axis = 1)/np.shape(tot_np_y_pred_outsmp)[1]
                # all_predictions_np.append(tot_np_y_pred_outsmp)
                # print(len(Best_x_lis) )
                # print(Best_x_lis)
                ##################################################################################################################
                ##################################################################################################################
                with open(f"Plots/{tup[1]}/depo/TrnCombTrsry.csv", mode="w", newline="") as file:
                # with open(f"Plots/{GenX}/depo/TrnCombTrsry.csv", mode="w", newline="") as file:
                    writer = csv.writer(file)
                    # Writing rows one at a time
                    writer.writerow([" ************  Train_solvents ************  "])
                    for j , trainIDs in enumerate(lis_trn_id):
                        writer.writerow([pw_cod[j]])
                        writer.writerow([tags_smp_x[i] for i in trainIDs])
                    writer.writerow([" ************  Train_idices ************  "])
                    for j , trainIDs in enumerate(lis_trn_id):
                        writer.writerow([pw_cod[j]])
                        writer.writerow([ids for ids in trainIDs])
                    writer.writerow([" ***********************************************  "])
                    writer.writerow([" ***********************************************  "])
                    writer.writerow(["                                                  "])
                    writer.writerow(["                                                  "])
                # # Writing to CSV file
                # Trn_Data = [lists of lists]
                # with open("Plots/{tup[1]}/depo/TrnCombTrsry.csv", mode="w", newline="") as file:
                #     writer = csv.writer(file)
                #     for row in Trn_Data:
                #         writer.writerow(row)
                with open(f"Plots/{tup[1]}/depo/TstCombTrsry.csv", mode="w", newline="") as file:
                    writer = csv.writer(file)
                    # Writing rows one at a time
                    writer.writerow([" ************  Test_solvents ************  "])
                    for j , testIDs in enumerate(lis_tst_id):
                        writer.writerow([pw_cod[j]])
                        writer.writerow([tags_smp_x[i] for i in testIDs])
                    writer.writerow([" ************  Test_idices ************  "])
                    for j , testIDs in enumerate(lis_tst_id):
                        writer.writerow([pw_cod[j]])
                        writer.writerow([ids for ids in testIDs])
                    writer.writerow([" ***********************************************  "])
                    writer.writerow([" ***********************************************  "])
                    writer.writerow(["                                                  "])
                    writer.writerow(["                                                  "])

                ##################################################################################################################
                ##################################################################################################################

                for w in range(len(Best_x_lis)):
                    results_1["Best_X"].append(Best_x_lis[w])
                    results_1["r2_train"].append(r2_train_lis[w])
                    results_1["delta"].append(delta_lis[w])
                    results_1["Q2"].append(Q2_lis[w])
                    results_1["MAE_train"].append(MAE_train_lis[w])
                    results_1["RMSE_train"].append(RMSE_train_lis[w])
                    results_1["r2_test"].append(r2_test_lis[w])
                    results_1["MAE_test"].append(MAE_test_lis[w])
                    results_1["RMSE_test"].append(RMSE_test_lis[w])
                    results_1["Bias"].append(Bias_lis[w])
                df = pd.DataFrame.from_dict(results_1)
                df.to_excel(f"Plots/{tup[1]}/depo/Metrics_{tup[1]}.xlsx")
                # for w in range(len(Best_x_lis)):
                #     results_2["Coeffs"].append(Coeffs_lis[w])
                # df = pd.DataFrame.from_dict(results_2)
                # df.to_excel(f"Plots/Model_Feature_Imporatnce.xlsx")
                #########################################################
                # results_3["Rank"].append(row_features_tags)
                # for w in range(len(Best_x_lis)):
                #     results_3["Rank"].append(rank_lis[w])
                # df = pd.DataFrame.from_dict(results_3)
                # df.to_excel(f"Plots/{tup[1]}/depo/Feature_Rankings_{tup[1]}.xlsx")
                #########################################################
                # print(row_features_tags)
                # print(len(row_features_tags))
                df = pd.DataFrame(np.transpose(tot_np_rank), columns=['rank'])
                df['Tag'] = row_features_tags
                df_sorted = df.sort_values(by='rank', ascending=True)
                df_sorted.to_excel(f"Plots/{tup[1]}/depo/tot_np_rank__{tup[1]}.xlsx", index=False)
                #########################################################
                # df = pd.DataFrame(np.transpose(tot_np_y_pred_outsmp ))
                # df['Tag'] = [f"{tag}, index {k}" for k , tag in enumerate(tags_outsmp_x)]
                # df.to_excel(f"Plots/{tup[1]}/depo/tot_outsmp_y__{tup[1]}.xlsx")
                # df['Avg'] = pd.DataFrame(np.average(np.transpose(tot_np_y_pred_outsmp ), axis=1 ))
                # df_sorted = df.sort_values(by='Avg', ascending=False)
                # df_sorted.to_excel(f"Plots/{tup[1]}/depo/tot_outsmp_y__{tup[1]}.xlsx", index=False)
                #########################################################
                # test_indices = combo_picked[0][0]
                # train_indices = [i for i in range(nnn) if i not in test_indices]

                train_indices = np.array(tup[0][:])
                test_indices = np.array([i for i in range(nnn) if i not in train_indices])    

                # tags_smp_x_train = {train_ind: tags_smp_x[train_ind] for train_ind in train_indices}
                tags_smp_x_train = [tags_smp_x[j] for j in train_indices]
                tags_smp_x_test = [tags_smp_x[j] for j in test_indices ]
                # tags_smp_x_test = {test_ind: tags_smp_x[test_ind] for test_ind in test_indices}
                tags_smp_x_reord = tags_smp_x_train + tags_smp_x_test
                # print (tags_smp_x_train)
                #########################################################
                df = pd.DataFrame(np.transpose(tot_np_y_smp), columns=['id', 'observed', 'predicted'])
                df['Tag'] = [f"{tag}" for k , tag in enumerate(tags_smp_x_reord)]
                df = df[['id', 'Tag', 'observed', 'predicted']]
                df.to_excel(f"Plots/{tup[1]}/depo/tot_smp_y{tup[1]}.xlsx", index=False)
                #########################################################
                # df = pd.DataFrame(np.transpose(tot_np_y_pred_test ))
                # df['Tag'] = tags_smp_x_test
                # df['Tag'] = [f"{tag}, index {k}" for k , tag in enumerate(tags_smp_x_test)]
                # df.to_excel(f"Plots/{tup[1]}/depo/tot_smp_y_test_{tup[1]}.xlsx")
                #########################################################
                #########################################################
                #########################################################
                #########################################################
                # print(f"Oh Such a relief all {tup[1]} plots are done")
                # print(f"\n####################################################################################################################################################\n")
                # print(f"\n####################################################################################################################################################\n")

                #########################################################################################################################################################################################
                #########################################################################################################################################################################################
                # print (tags_outsmp_x)
                # Once the loop is done, concatenate all numpy arrays along the 0th axis (rows stay the same)
                # final_np_array = np.concatenate(all_predictions_np, axis=0)
                # # print(np.shape(final_np_array))
                # final_df = pd.DataFrame(np.transpose(final_np_array))
                # final_df['Tag'] = tags_outsmp_x  # Assuming the same tags are used throughout
                # final_df['Tag'] = [f"{tag}, index {k}" for k , tag in enumerate(tags_outsmp_x)]
                # final_df.to_excel("Plots/Gen/final_tot_outsmp_y.xlsx")
                # final_df['Avg'] = pd.DataFrame(np.average(np.transpose(final_np_array), axis=1))
                # final_sorted_df = final_df.sort_values(by='Avg', ascending=False)
                # final_sorted_df.to_excel("Plots/Gen/final_tot_outsmp_y_sorted.xlsx", index=True)
                #########################################################################################################################################################################################
                #########################################################################################################################################################################################

End_Time = datetime.now()
print(f"It exactly took {End_Time - Start_Time} seconds")
