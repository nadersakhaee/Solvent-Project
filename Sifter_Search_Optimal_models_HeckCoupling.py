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



# Features_tags = []
# with open('Trsry/ReichTableFnp.csv') as f:
# with open('Trsry/Featurs_FRM.csv') as f:
with open('nTrsry/Featurs_FRM.csv') as f:
    for line in f:
        row_features_tags = line.strip().split(',')
#         x_dict[row_features_tags[0]] = [float(x) for x in row_features_tags[1:]]
#         all_x.append(x_dict[row_features_tags[0]])
#         tags_all_x.append(row_features_tags[0])
# print(len(row_features_tags))
# print(row_features_tags)
# x_tot = np.array(all_x, dtype='f' )



####################################################################################################################################################################
####################################################################################################################################################################
Start_Time = datetime.now()
# print(Start_Time)
####################################################################################################################################################################
####################################################################################################################################################################

U_Solv_lis = [
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
    "hexane", "dimethylacetamide", "paraxylene", "benzonitrile", "acetophenone", "tamylmethylether", "methylethylketone", "pentane", 
    "propylacetate", "tbutylbenzene", "diethyleneglycoldiethylether", "ethylmethylcarbonate", "diethylcarbonate"

]



####################################################################################################################################################################
####################################################################################################################################################################

def r_2_real(x,y):
    fun_ = pearsonr(x,y)
    return fun_[0]**2

def create_dict_data_tag( path ):
    dict_, data_, tags_data_ = {}, [], []
    for solv in U_Solv_lis:
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

# dict_all_x, all_x, tags_all_x, np_all_x = create_dict_data_tag('Trsry/120_Trs_FRM_03np.csv')
dict_all_x, all_x, tags_all_x, np_all_x = create_dict_data_tag('nTrsry/121_FRM_04np.csv')

# dict_smp_y, smp_y, tags_smp_y, np_smp_y = create_dict_data_tag('Trsry/Y_OOS_CM_121.csv')
dict_smp_y, smp_y, tags_smp_y, np_smp_y = create_dict_data_tag('nTrsry/Y_OOS_Hk_121.csv')

dict_smp_x, tags_smp_x, smp_x, np_smp_x, dict_outsmp_x, tags_outsmp_x, outsmp_x, np_outsmp_x = create_smp_outofsmp_dictionaries(dict_all_x, tags_smp_y)

####################################################################################################################################################################
###########################################        These parts do prints just for checking for code cleaning purposes  #############################################
####################################################################################################################################################################
# for i, (key, val) in enumerate(dict_smp_x.items()):
#     print(f"{i}-----------------the key is {key}-----------------dat={val[15]}")   
# for k , tag in enumerate(tags_smp_x):
#     print(f"{k}-----------------the key is {tag}")
# for j , dat in enumerate(smp_x):
#     print(f"{j}-----------------xxxxxxxxxxxxxxxxxxxxxxx-----------------dat={dat[15]}")
# print(f"\n####################################################################################################################################################\n")
# for i, (key, val) in enumerate(dict_outsmp_x.items()):
#     print(f"{i}-----------------the key is {key}-----------------dat={val[15]}")   
# for k , tag in enumerate(tags_outsmp_x):
#     print(f"{k}-----------------the key is {tag}")
# for j , dat in enumerate(outsmp_x):
#     print(f"{j}-----------------xxxxxxxxxxxxxxxxxxxxxxx-----------------dat={dat[15]}")
# for i, (key, val) in enumerate(dict_all_x.items()):
#     print(f"{i}-----------------the key is {key}-----------------dat={val[20]}")   
# for k , tag in enumerate(tags_all_x):
#     print(f"{k}-----------------the key is {tag}")
# for j , dat in enumerate(all_x):
#     print(f"{j}-----------------xxxxxxxxxxxxxxxxxxxxxxx-----------------dat={dat[20]}")
# print(f"\n####################################################################################################################################################\n")
# for i, (key, val) in enumerate(dict_smp_y.items()):
#     print(f"{i}-----------------the key is {key}-----------------dat={val}")   
# for k , tag in enumerate(tags_smp_y):
#     print(f"{k}-----------------the key is {tag}")
# for j , dat in enumerate(smp_y):
#     print(f"{j}-----------------xxxxxxxxxxxxxxxxxxxxxxx-----------------dat={dat}")
# print(f"\n####################################################################################################################################################\n")
####################################################################################################################################################################
####################################################################################################################################################################
####################################################################################################################################################################
####################################################################################################################################################################
####################################################################################################################################################################
####################################################################################################################################################################




# @@@@@@@@@@@@@@@@@@@@@@@@@@@@
# models_code = ['3', '8', '9', '10', '11']
# models_code = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11']
# models_code = [ '3', '4', '8', '9', '10', '11']
# models_code = [ '1','2','3','7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18']
# models_code = [ '3','7', '9', '13', '14', '18']
models_code = [ '8','9', '11', '12', '14', '18']
# models_code = [ '11', '12', '18']
ppww1 = len(models_code)
# models_code = [ '14', '18']
# models_code = [ '11', '15', '17']
# models_code = ['4', '8']
# models_code = ['1', '7', '11']

###################################### These are the models to be screened   Namedtuple makes tuples much more legible #####################################

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
##The 'loss' parameter of GradientBoostingRegressor must be a str among {'quantile', 'huber', 'absolute_error', 'squared_error'}
### for gradient boost regressor  "5"      feature_importance = model.feature_importances_
### for Random Forest  regressor   "6"     feature_importance = model.feature_importances_   
### for Decision Tree  regressor    "7"    feature_importance = model.feature_importances_ 


results_1 = { "Best_X": [], "r2_train": [], "delta":[],  "Q2":[], "MAE_train":[], "RMSE_train":[], "r2_test": [], "MAE_test":[], "RMSE_test":[], "Bias":[]}
# results_2 = {"Coeffs":[] }
results_3 = {"Rank":[] }
Best_x_lis, r2_train_lis, delta_lis, Q2_lis, MAE_train_lis, RMSE_train_lis, r2_test_lis, MAE_test_lis, RMSE_test_lis, Bias_lis, Coeffs_lis, rank_lis  = [], [], [], [], [], [], [], [], [], [], [], []
####################################################################################################################################################################################################
####################################             These are some of the most predicitive test split indices picked after seeing all Models       ####################################################
####################################                                         OpenPlotter then Model Plot domino                                 ####################################################
####################################################################################################################################################################################################
#####################     This should be renewed before each run        ############################################################################################################################

#######################################################



########################################################################################
#######################                c_1_47_47_58_n1                ###########################
########################################################################################
tscn, iscn, bscn, oscn =    '5',       '6' ,       '8',        '5'
tcn, icn, bcn, ocn     =      5,        6 ,          8,          5
tlim, ilim, blim, olim =   2002,    13983816,       1081575,    237336
tsn, isn, bsn, osn     =    125,      62150,       16899,      2373
tsna, isna, bsna, osna =    31,      4143,       2112,      237
tinia, iinia, binia, oinia =    125,      2921062,       794273,      137649
tfina, ifina, bfina, ofina =    250,      2983212,       811172,      140022
########################################################################################
########################################################################################


GenX = "b_1_47_47_58_n40"
#######################################################

####################################################################################################################################################################################################


h5f = h5py.File('nTrsry/HDFs/np_cmb_14_'+tscn+'.h5','r')
np_cmb_14_t = h5f['np_cmb_14_'+tscn][:]
h5f.close()

h5f = h5py.File('nTrsry/HDFs/np_cmb_49_'+iscn+'.h5','r')
np_cmb_49_i = h5f['np_cmb_49_'+iscn][:]
h5f.close()

h5f = h5py.File('nTrsry/HDFs/np_cmb_25_'+bscn+'.h5','r')
np_cmb_25_b = h5f['np_cmb_25_'+bscn][:]
h5f.close()

h5f = h5py.File('nTrsry/HDFs/np_cmb_33_'+oscn+'.h5','r')
np_cmb_33_o = h5f['np_cmb_33_'+oscn][:]
h5f.close()


print('uploaded all HDF5s')


# h5f = h5py.File('nTrsry/HDFs/np_cmb_33_4.h5','r')
# np_cmb_33_4 = h5f['np_cmb_33_4'][:]
# # print(f"shape of cmb_33_4 uploaded array is{np.shape(np_cmb_33_4)}\n\n")
# h5f.close()

# with h5py.File('nTrsry/np_cmb_33_4.h5','r') as h5f:
#     print(list(h5f.keys()))    #### This goes to how you save it 
###################################################################################################################################################################################################
####################################################################################################################################################################################################



# for tup in c5_rests:
    #define the indice list first  

index_combinations = []
# for t in range(0,tlim, tsna):
#     for i in range(0,ilim, isna):
#         for b in range(0,blim, bsna):
#             for o in range(0,olim, osna):
for t in range(tinia,tfina, tsna):
    for i in range(iinia,ifina, isna):
        for b in range(binia,bfina, bsna):
            for o in range(oinia,ofina, osna):                
                # print(f"{t},{i},{b},{o}")
                # print(f"{np_cmb_14_5[t]},{np_cmb_49_6[i]},{np_cmb_25_8[b]},{np_cmb_33_4[o]}")
                index_combinations.append( [[int(t),int(i),int(b),int(o)] ,[np.concatenate((np_cmb_14_t[t], np_cmb_49_i[i], np_cmb_25_b[b], np_cmb_33_o[o])).astype(int).tolist()]])                
                # print(type(np.concatenate((np_cmb_14_5[t], np_cmb_49_6[i], np_cmb_25_8[b], np_cmb_33_4[o])).astype(int).tolist()))
print('did all loops successfully ')

NameYourPlotFucker = "Swift"
min_max_scalerM = preprocessing.MinMaxScaler()
# scl_x_outsmp    = min_max_scalerM.fit_transform(np_outsmp_x)
tot_lis_y_pred_outsmp, awesome_id_tags, tot_test_MAEs, nnn = [], [[],[]], [], len(tags_smp_x)
picked_lis_y_pred_train, picked_lis_y_pred_test, picked_lis_y_train, picked_lis_y_test, lis_trn_id, lis_tst_id, tibo_cods, tibo_itrs, pwcs = [], [], [], [], [], [], [], [], []
with rc_context(fname="Plots/plot_settings.yml"):
    for pw1, ind_lis in enumerate(index_combinations):
        ### We use this one for normal sifitng with Small Successful Train lookups
        # test_indices = np.array(ind_lis[1][:])
        # train_indices = np.array([i for i in range(nnn) if i not in test_indices[:]])
        # ###    We used this one for big Train Series initial Evals 
        train_indices = np.array(ind_lis[1][0][:])
        # train_indices = "_".join(str(x) for x in ind_lis[0][:])
        # print(train_indices)
        # print(f"shape of array is{np.shape(train_indices)}\n\n")    #####@@@@@@@@@@@@@@@@@
        test_indices = np.array([i for i in range(nnn) if i not in train_indices[:]])
        # Annotate each point with its index
        train_id_tag = []
        for train_id in train_indices:
            train_id_tag.append(str(train_id))
            # print(train_id)  # Split data based on indices
            x_train, x_test = np_smp_x[train_indices], np_smp_x[test_indices]
            y_train, y_test = np.array(np_smp_y[train_indices]).ravel(), np.array(np_smp_y[test_indices]).ravel()
            min_max_scalerM = preprocessing.MinMaxScaler()
            scl_x_train  = min_max_scalerM.fit_transform(x_train)
            scl_x_test   = min_max_scalerM.transform(x_test) 
    ##############################################################################################
        for w2, m in enumerate(models):
            w3 = 0
            scorer = make_scorer(r_2_real)
            # if m.name not in ["linRdige","RidgeCV","ARD","Huber","LassoCV","OrthogMP",]:
            if m.name not in ["linRdige","Huber"]:
                print(f"Oh Shit Code's still runnig @---------------------------- {GenX} ")
                pass
            else:
                plt.figure()  # Create a new figure for each plot
                w3 += 1                                                                                                                                                                                                                           
                m.model.fit(scl_x_train, y_train)
                y_pred_train = m.model.predict(scl_x_train)
                y_pred_test = m.model.predict(scl_x_test)
                mae_t = mean_absolute_error(y_test, y_pred_test)
                r2_t = r_2_real(y_test, y_pred_test)
                rmse_t = mean_squared_error(y_test, y_pred_test, squared=False)
                q2 = cross_val_score(m.model, scl_x_train, y_train, cv=2, scoring=scorer).mean()
                r2_r = r_2_real(y_train, y_pred_train) 
                intercept = np.average(m.model.intercept_)
                # if mae_t < 12 :        # @@@@@@@@@@@@@@@@@@@@@    !!!!!!!     SWITCH HERE        !!!!!!!!!     @@@@@@@@@@@@@@@@
                if mae_t < 9 and r2_t > 0.639 and  intercept < 60:        # @@@@@@@@@@@@@@@@@@@@@    !!!!!!!     SWITCH HERE        !!!!!!!!!     @@@@@@@@@@@@@@@@@
                # if mae_t < 5 and r2_t > 0.73 and intercept < 40:         # @@@@@@@@@@@@@@@@@@@@@    !!!!!!!     SWITCH HERE        !!!!!!!!!     @@@@@@@@@@@@@@@@@
                # if mae_t < 5 and r2_t >= 0.76 and intercept < 40:         # @@@@@@@@@@@@@@@@@@@@@    !!!!!!!     SWITCH HERE        !!!!!!!!!     @@@@@@@@@@@@@@@@@
                # if mae_t < 12 and r2_t > 0.56 and q2 > 0.37:         # @@@@@@@@@@@    !!!!!!!     SWITCH HERE        !!!!!!!!!     @@@@@@@@@
                    tibo_itrs.append("_".join(str(y) for y in [int(ind_lis[0][0] // tsna), int(ind_lis[0][1] // isna),int(ind_lis[0][2] // bsna), int(ind_lis[0][3] // osna) ]))
                    tibo_cods.append("_".join(str(x) for x in ind_lis[0][:]))
                    pwcs.append(pw1)
                    lis_trn_id.append(train_indices)
                    lis_tst_id.append(test_indices)
                    tot_test_MAEs.append(mae_t)
                    select = RFE(m.model, n_features_to_select=3, step=1)
                    selector = select.fit(scl_x_train, y_train)
                    ranks = selector.ranking_
                    # Coefficients (contribution) of each feature
                    # coefficients = m.model.coef_          
                    # coefficients = m.model.feature_importances_
                    r2_rs = r2_score(y_train, y_pred_train)
                    mae_r = mean_absolute_error(y_train, y_pred_train)
                    rmse_r = mean_squared_error(y_train, y_pred_train, squared=False)
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
                    picked_lis_y_pred_train.append(y_pred_train)
                    picked_lis_y_pred_test.append(y_pred_test)
                    picked_lis_y_train.append(y_train)
                    picked_lis_y_test.append(y_test)
                    for z, train_str_tag in enumerate(train_id_tag):
                        plt.annotate(train_str_tag, (y_train[z], y_pred_train[z]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=3)
                    test_id_tag = []
                    for test_id in test_indices:
                        test_id_tag.append(str(test_id))
                        # print(test_id)
                    for z, test_str_tag in enumerate(test_id_tag):
                        plt.annotate(test_str_tag, (y_test[z], y_pred_test[z]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=3)
                    awesome_id_tags[0].append(train_id_tag)
                    awesome_id_tags[1].append(test_id_tag)
                    # plt.annotate(train_indices, (y_train[:], y_pred_train[:]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=3)
                    # plt.annotate(test_indices, (y_test[:], y_pred_test[:]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=3)                        
                    # Actual plotting 
                    plt.scatter(y_train, y_pred_train, color='blue', alpha=0.5, marker='o', s=15, label=f"Train: $R^2={r2_r:.3f}, Delta={(float(r2_r) - float(r2_t)):.3f}$\n $Q^2={q2:.3f}$, mae={mae_r:.3f}, rmse={rmse_r:.3f}")
                    plt.scatter(y_test, y_pred_test, color='red', marker='<', s=15, label=f"Test: $R^2 ={r2_t:.3f}$\n mae={mae_t:.3f},  rmse={rmse_t:.3f}, Bias={intercept:.1f}")
                    #Actual Y out of Sample gathering
                    # This Part is singled Out Plotter 
                    plt.title(f" Plt_{pw1}_{m.name}_Y_Exp_data_ ")
                    plt.xlabel("Observed yields for Heck (Y_Exp's Data)  ")
                    plt.ylabel("Predicted yields for Heck (Y_Exp's Data) ")
                    # plt.ylim(0, 100)
                    # plt.xlim(0, 100)                    
                    plt.ylim(-10, 110)
                    plt.xlim(-10, 110)
                    plt.legend() 
                    plt.savefig(f"Plots/{GenX}/depo/PltId_{pw1}_{m.name}.svg", dpi=900)
                    plt.savefig(f"Plots/{GenX}/PltId_{pw1}_{m.name}.png", dpi=900)
                    plt.close()
                else: 
                    pass
####################################################################################################################################################################
####################################################################################################################################################################

with open(f"Plots/{GenX}/depo/TrnCombTrsry.csv", mode="w", newline="") as file:
    writer = csv.writer(file)
    # Writing rows one at a time
    writer.writerow([" ************  Train_solvents ************  "])
    for j , trainIDs in enumerate(lis_trn_id):
        writer.writerow([tibo_cods[j]])
        writer.writerow([tibo_itrs[j]])
        writer.writerow([pwcs[j]])
        writer.writerow([tags_smp_x[i] for i in trainIDs])
    writer.writerow([" ************  Train_idices ************  "])
    for j , trainIDs in enumerate(lis_trn_id):
        writer.writerow([tibo_cods[j]])
        writer.writerow([tibo_itrs[j]])
        writer.writerow([pwcs[j]])
        writer.writerow([ids for ids in trainIDs])
    writer.writerow([" ***********************************************  "])
    writer.writerow([" ***********************************************  "])
    writer.writerow(["                                                  "])
    writer.writerow(["                                                  "])
with open(f"Plots/{GenX}/depo/TstCombTrsry.csv", mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow([" ************  Test_solvents ************  "])
    for j , testIDs in enumerate(lis_tst_id):
        writer.writerow([tibo_cods[j]])
        writer.writerow([tibo_itrs[j]])
        writer.writerow([pwcs[j]])
        writer.writerow([tags_smp_x[i] for i in testIDs])
    writer.writerow([" ************  Test_idices ************  "])
    for j , testIDs in enumerate(lis_tst_id):
        writer.writerow([tibo_cods[j]])
        writer.writerow([tibo_itrs[j]])
        writer.writerow([pwcs[j]])
        writer.writerow([ids for ids in testIDs])
    writer.writerow([" ***********************************************  "])
    writer.writerow([" ***********************************************  "])
    writer.writerow(["                                                  "])
    writer.writerow(["                                                  "])



np_tot_test_MAEs = np.array(tot_test_MAEs, dtype='f' )
if len (np_tot_test_MAEs) !=0:
    print(np.mean(np_tot_test_MAEs))
# #########################################################
# #########################################################
# print(len(index_combinations)*scapeNumber)


End_Time = datetime.now()
# print(f"Oh Such a relief all {NameYourPlotFucker} plots are done")
print(f"It exactly took {End_Time - Start_Time} seconds")
# print(len(rests))










