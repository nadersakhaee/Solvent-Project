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

# dict_smp_y, smp_y, tags_smp_y, np_smp_y = create_dict_data_tag('Trsry/Y_Srv_CM_36.csv')
dict_smp_y, smp_y, tags_smp_y, np_smp_y = create_dict_data_tag('nTrsry/Y_OOS_CM_121.csv')


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
####################################################################################################################################################################
#################################### These are some of the most predicitive test split indices picked after seeing all Models 

####################################################################################################################################################################
####################################################################################################################################################################
def N_comb(n, k):
    return math.factorial(n) // (math.factorial(k) * math.factorial(n - k))
###################      @@@@@@@@@@@@@@@@@@@@@    !!!!!!!     SWITCH HERE        !!!!!!!!!     @@@@@@@@@@@@@@@@@@@@@@@@@
scapeNumber = 91
def generate_combinations(pool_a, na, lis_highs:list, step=scapeNumber):
    all_combinations = []
    combo_gen = combinations(pool_a, na)
    count = 0
    for combo in combo_gen:
        if count % step == 0:
            ex_combo= combo + tuple(lis_highs)
            all_combinations.append(ex_combo)
        count += 1
    return all_combinations
##################################################################################################
##################################################################################################
def generate_similar_combinations(pool_s, ns, targets):
    sim_combs = []
    for trg in targets:
        sim_combs.append(find_similar_combinations(pool_s, ns, trg))
    return sim_combs
###################################################################################################################################################################
#                 @@@@@@@@@@@@@@@@@@@@@    !!!!!!!     SWITCH HERE        !!!!!!!!!     @@@@@@@@@@@@@@@@@@@@@@@@@
ini_rests = [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
          41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
          80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114,
          115, 116, 117, 118, 119, 120]   

c5_rests = [
    # ([51,54,28,36,70,106,117,9,0,10,3], "wu_11"),     ###  len_11
    # ([51,54,28,36,70,106,117,9,0,10,3,82,33,19,72,43,81,67,8], "wu_19"),       ###  len_19
    # ([51,54,28,36,70,106,117,9,0,10,3,82,33,19,72,43,81,67,8,50,55,115,60,61,89,120], "wu_e"),        ###  len_26_elbow   
    # ##############        Specialized on twenty golden refinements ##################################
    # ([51,54,28,36,70,106,117,9,0,10,3,82,33,19,72,43,81,67,8,46], "wu_20"),       ###  len_19
    # ##############        Specialized on twenty golden refinements ##################################
    # ([51,54,28,36,70,106,117,9,0,10,3,82,33,19,72,43,81,67,8,46,109], "wua_109"),       ###  
    # ([51,54,28,36,70,106,117,9,0,10,3,82,33,19,72,43,81,67,8,46,104], "wua_104"),       ###  
    # ([51,54,28,36,70,106,117,9,0,10,3,82,33,19,72,43,81,67,8,46,63], "wua_63"),       ###  
    # ([51,54,28,36,70,106,117,9,0,10,3,82,33,19,72,43,81,67,8,46,89], "wua_89"),       ###  
    # ([51,54,28,36,70,106,117,9,0,10,3,82,33,19,72,43,81,67,8,46,48], "wua_48"),       ###  
    # ##############        Specialized on twenty golden refinements ##################################
    # ([51,54,28,36,70,106,117,9,0,10,3,82,33,19,72,43,81,67,8,46,109,89], "wub"),       ###  
    # ##############        Specialized on twenty golden refinements ##################################
    # ([51,54,28,36,70,106,117,9,0,10,3,82,33,19,72,43,81,67,8,46,109,89,20 ], "wub_20"),       ###  
    # ([51,54,28,36,70,106,117,9,0,10,3,82,33,19,72,43,81,67,8,46,109,89,104 ], "wub_104"),       ###  
    # ([51,54,28,36,70,106,117,9,0,10,3,82,33,19,72,43,81,67,8,46,109,89,45 ], "wub_45"),       ###  
    # ([51,54,28,36,70,106,117,9,0,10,3,82,33,19,72,43,81,67,8,46,109,89,52 ], "wub_52"),       ###  
    # ([51,54,28,36,70,106,117,9,0,10,3,82,33,19,72,43,81,67,8,46,109,89,61 ], "wub_61"),       ###  
    # ([51,54,28,36,70,106,117,9,0,10,3,82,33,19,72,43,81,67,8,46,109,89,37 ], "wub_37"),       ###
    # ##############        Specialized on twenty golden refinements ##################################
    # ([51,54,28,36,70,106,117,9,0,10,3,82,33,19,72,43,81,67,8,46,109,89,20,45 ], "wuc_45"),       ###  
    # ([51,54,28,36,70,106,117,9,0,10,3,82,33,19,72,43,81,67,8,46,109,89,20,77 ], "wuc_77"),       ###  
    # ([51,54,28,36,70,106,117,9,0,10,3,82,33,19,72,43,81,67,8,46,109,89,20,61 ], "wuc_61"),       ###  
    # ([51,54,28,36,70,106,117,9,0,10,3,82,33,19,72,43,81,67,8,46,109,89,20,49 ], "wuc_49"),       ###  
    # ([51,54,28,36,70,106,117,9,0,10,3,82,33,19,72,43,81,67,8,46,109,89,20,37 ], "wuc_37"),       ###  
    # ([51,54,28,36,70,106,117,9,0,10,3,82,33,19,72,43,81,67,8,46,109,89,20,62 ], "wuc_62"),       ###  
    # ([51,54,28,36,70,106,117,9,0,10,3,82,33,19,72,43,81,67,8,46,109,89,20,120 ], "wuc_120"),       ###  
    # ([51,54,28,36,70,106,117,9,0,10,3,82,33,19,72,43,81,67,8,46,109,89,20,52 ], "wuc_52"),       ###  
    # ([51,54,28,36,70,106,117,9,0,10,3,82,33,19,72,43,81,67,8,46,109,89,20,104 ], "wuc_104"),       ###  
    # ([51,54,28,36,70,106,117,9,0,10,3,82,33,19,72,43,81,67,8,46,109,89,20,71 ], "wuc_71"),       ###  
    # ##############        Specialized on twenty golden refinements ##################################
    ([51,54,28,36,70,106,117,9,0,10,3,82,33,19,72,43,81,67,8,46,109,89,20,45,77 ], "wud_77"),       ###  
    ([51,54,28,36,70,106,117,9,0,10,3,82,33,19,72,43,81,67,8,46,109,89,20,45,62 ], "wud_62"),       ###  
    ([51,54,28,36,70,106,117,9,0,10,3,82,33,19,72,43,81,67,8,46,109,89,20,45,104 ], "wud_104"),       ###  
    ([51,54,28,36,70,106,117,9,0,10,3,82,33,19,72,43,81,67,8,46,109,89,20,45,68 ], "wud_68"),       ###  
    ([51,54,28,36,70,106,117,9,0,10,3,82,33,19,72,43,81,67,8,46,109,89,20,45,18 ], "wud_18"),       ###  
    ([51,54,28,36,70,106,117,9,0,10,3,82,33,19,72,43,81,67,8,46,109,89,20,45,61 ], "wud_61"),       ###  
    ([51,54,28,36,70,106,117,9,0,10,3,82,33,19,72,43,81,67,8,46,109,89,20,45,120 ], "wud_120"),       ###  
    ([51,54,28,36,70,106,117,9,0,10,3,82,33,19,72,43,81,67,8,46,109,89,20,45,52 ], "wud_52"),       ###  
    ([51,54,28,36,70,106,117,9,0,10,3,82,33,19,72,43,81,67,8,46,109,89,20,45,49 ], "wud_49"),       ###  
    ([51,54,28,36,70,106,117,9,0,10,3,82,33,19,72,43,81,67,8,46,109,89,20,45,35 ], "wud_35"),       ###  
    ([51,54,28,36,70,106,117,9,0,10,3,82,33,19,72,43,81,67,8,46,109,89,20,45,87 ], "wud_87"),       ### 


]



for tup in c5_rests:
    chosen = len(ini_rests)-(len(tup[0])+3)
    rests = list(set(ini_rests)-set(tup[0]))
    # # rests = list((set(tmp_rests) - set(b_rests)) - set(c1_rests))     ### len effective = 110 


    #  len rest = 121 - 1 since there are two hps in Train omitted from rests. The number of test is 5 as indicated and there are 2 mandatory test as in  [tup[2]]. 
    # The number of train is len(rests)-number of test so 121 - 111 equals (7) this will be added to mandatory trains omitted from rest so Train is 7 + 1 hps equals 8. 
    ###      121 - c5rest  - after rest number  = 121 - 3 - 114 = 3  so three would be added to training.

    # index_combinations = generate_combinations(rests, 1113, [] )
    # index_combinations = generate_combinations(rests, 15 , tup[2] )
    index_combinations = generate_combinations(rests, chosen, [])
    midpoint, midpoint_1, midpoint_2  = len(index_combinations) // 2, len(index_combinations) // 3, 2*(len(index_combinations) // 3)
    index_combinationsA, index_combinationsB = index_combinations[:midpoint + 1], index_combinations[midpoint:]
    index_combinationsT1, index_combinationsT2 , index_combinationsT3 = index_combinations[:midpoint_1 + 1], index_combinations[midpoint_1:midpoint_2 + 1], index_combinations[midpoint_2:]
    ####################################################################################################################################################################
    ####################################################################################################################################################################
    ############################################################# OpenPlotter then Model Plot domino     ###############################################################
    ####################################################################################################################################################################
    ###    @@@@@@@@@@@@@@@@@@@@@    !!!!!!!     SWITCH HERE        !!!!!!!!!     @@@@@@@@@@@@@@@@@@@@@@@@@

    NameYourPlotFucker = "Swift"
    min_max_scalerM = preprocessing.MinMaxScaler()
    # scl_x_outsmp    = min_max_scalerM.fit_transform(np_outsmp_x)
    tot_lis_y_pred_outsmp, awesome_id_tags, tot_test_MAEs, nnn = [], [[],[]], [], len(tags_smp_x)
    picked_lis_y_pred_train, picked_lis_y_pred_test, picked_lis_y_train, picked_lis_y_test, lis_trn_id, lis_tst_id, pw_cod = [], [], [], [], [], [], []
    with rc_context(fname="Plots/plot_settings.yml"):
        for pw1, indices in enumerate(index_combinations):
        # for pw1, indices in enumerate(index_combinationsA):
        # for pw1, indices in enumerate(index_combinationsB):
        # for pw1, indices in enumerate(index_combinationsT1):
        # for pw1, indices in enumerate(index_combinationsT2):
        # for pw1, indices in enumerate(index_combinationsT3):            
            ### We use this one for normal sifitng with Small Successful Train lookups
            test_indices = np.array(indices[:])
            train_indices = np.array([i for i in range(nnn) if i not in test_indices[:]])
            # ###    We used this one for big Train Series initial Evals 
            # train_indices = np.array(indices[:])
            # test_indices = np.array([i for i in range(nnn) if i not in train_indices[:]])

            # Annotate each point with its index
            train_id_tag = []
            for train_id in train_indices:
                train_id_tag.append(str(train_id))
                # print(train_id)  # Split data based on indices
                x_train, x_test = np_smp_x[train_indices], np_smp_x[test_indices]
                y_train, y_test = np.array(np_smp_y[train_indices]).ravel(), np.array(np_smp_y[test_indices]).ravel()
                min_max_scalerM = preprocessing.MinMaxScaler()
                scl_x_train     = min_max_scalerM.fit_transform(x_train)
                scl_x_test      = min_max_scalerM.transform(x_test) 
        ##############################################################################################
            for w2, m in enumerate(models):
                w3 = 0
                scorer = make_scorer(r_2_real)
                # if m.name not in ["linRdige","RidgeCV","ARD","Huber","LassoCV","OrthogMP",]:
                if m.name not in ["linRdige","Huber"]:
                    print(f"Oh Shit, Processing -----------------------------------  {tup[1]}")
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
                    if mae_t < 9 and r2_t > 0.826 and  intercept < 45:        # @@@@@@@@@@@@@@@@@@@@@    !!!!!!!     SWITCH HERE        !!!!!!!!!     @@@@@@@@@@@@@@@@@
                    # if mae_t < 5 and r2_t > 0.73 and intercept < 40:         # @@@@@@@@@@@@@@@@@@@@@    !!!!!!!     SWITCH HERE        !!!!!!!!!     @@@@@@@@@@@@@@@@@
                    # if mae_t < 6 and r2_t > 0.71 and rmse_t < 8 and q2 > 0.175:         # @@@@@@@@@@@    !!!!!!!     SWITCH HERE        !!!!!!!!!     @@@@@@@@@
                    # if mae_t < 12 and r2_t > 0.56 and q2 > 0.37:         # @@@@@@@@@@@    !!!!!!!     SWITCH HERE        !!!!!!!!!     @@@@@@@@@
                        pw_cod.append(pw1)
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
                        plt.title(f" Plt_[{pw1}]_{m.name}_Y_Exp_data_ ")
                        plt.xlabel("Observed yields for Cross Metathesis (Y_Trust's Data)")
                        plt.ylabel("Predicted yields for Cross Metathesis (Y_Pred's Data)")
                        plt.ylim(25, 95)
                        plt.xlim(25, 95)                    
                        # plt.ylim(-10, 110)
                        # plt.xlim(-10, 110)
                        plt.legend() 
                        plt.savefig(f"Plots/{tup[1]}/depo/PltId_{pw1}_{m.name}.svg", dpi=900)
                        plt.savefig(f"Plots/{tup[1]}/PltId_{pw1}_{m.name}.png", dpi=900)
                        plt.close()
                    else: 
                        pass
    ####################################################################################################################################################################
    ####################################################################################################################################################################

    with open(f"Plots/{tup[1]}/depo/TrnCombTrsry.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        # Writing rows one at a time
        writer.writerow([" ************  Train_solvents ************  "])
        for j , trainIDs in enumerate(lis_trn_id):
            writer.writerow([pw_cod[j]])
            writer.writerow([tags_smp_x[i] for i in list(set(trainIDs)-set(tup[0]))])
        writer.writerow([" ************  Train_idices ************  "])
        for j , trainIDs in enumerate(lis_trn_id):
            writer.writerow([pw_cod[j]])
            writer.writerow([ids for ids in list(set(trainIDs)-set(tup[0]))])
        writer.writerow([" ***********************************************  "])
        writer.writerow([" ***********************************************  "])
        writer.writerow(["                                                  "])
        writer.writerow(["                                                  "])
    with open(f"Plots/{tup[1]}/depo/TstCombTrsry.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
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

    np_tot_test_MAEs = np.array(tot_test_MAEs, dtype='f' )
    if len (np_tot_test_MAEs) !=0:
        print(np.mean(np_tot_test_MAEs))
    # #########################################################
    # #########################################################
    print(len(index_combinations)*scapeNumber)





    End_Time = datetime.now()
    # print(f"Oh Such a relief all {NameYourPlotFucker} plots are done")
    print(f"It exactly took {End_Time - Start_Time} seconds")
    # print(len(rests))

# # ######  First Ten 
# ,0,15,105,73,71,63,60,36,56,6,
# # ###### Second Ten
# ,19,92,35,89,61,18,26,117,45,79,
# # ###### Elbow Twenty Three
# ,19,92,35,89,61,18,26,117,45,79,    74,39,27,37,108,110,22,118,44,113,38,46,106,

