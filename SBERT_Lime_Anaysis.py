event_flagname='<flagname>'

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

n_jobs=25
random_state=802


from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn import model_selection
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn import model_selection
#import _pickle as cPickle
from sklearn.metrics import f1_score
from pprint import pprint
import joblib,traceback
from sentence_transformers import SentenceTransformer
from imblearn.ensemble import BalancedRandomForestClassifier

def evaluate_model(y_test, y_pred):
    # Print the Confusion Matrix and slice it into four pieces

    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_test, y_pred)

    print('Confusion matrix\n\n', cm)
    # visualize confusion matrix with seaborn heatmap

    cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive', 'Actual Negative'], 
                                     index=['Predict Positive', 'Predict Negative'])

#     sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')
    from sklearn.metrics import classification_report

    print(classification_report(y_test, y_pred))



def apply_str1(value):
    try:
#         print(type(value))
#         print(value[1:])
        return value[1:]
    except:
        traceback.print_exc()
        print("except",value)
        return value


def fill_not_available(dataset,fieldname):
#     print(fieldname)
#     print("before operation")
# #     print(dataset[fieldname].value_counts())
#     print("None",dataset[dataset[fieldname]==None].shape)
#     print("np.NaN",dataset[dataset[fieldname]==np.nan].shape)
#     print("empty string",dataset[dataset[fieldname]==""].shape)
#     print('dataset[fieldname]=="|Not available"]',dataset[dataset[fieldname]=="|Not available"].shape)
    dataset[fieldname]=dataset[fieldname].replace(r'^\s*$', "|Not available(M)", regex=True)
#     print(dataset[dataset[fieldname]=="|Not available(M)"].shape)
    dataset.loc[dataset[fieldname]=="",fieldname]="|Not available(M)"
    dataset.loc[dataset[fieldname]==np.nan,fieldname]="|Not available(M)"
    dataset.loc[dataset[fieldname]==None,fieldname]="|Not available(M)"
    dataset.loc[dataset[fieldname]=="nan",fieldname]="|Not available(M)"
#     print("after operation")
#     print(dataset[dataset[fieldname]=="|Not available(M)"].shape)
#     print(dataset[fieldname].value_counts())

    return dataset
def func_split_set(x,fieldname):
    try:

        split_x=set(str(x).split("|"))
        len_processed=len(split_x)
        len_raw=len(str(x).split("|"))
#         if "|" in str(x) and len_processed<len_raw:
#             print("*****\nraw,",x)
#             print('processed:',"|".join(list(split_x)))
            
        return "|".join(list(split_x))
    except:
        print(fieldname, 'in func')
        traceback.print_exc()

def func_split_set2(x):
    try:
#         print(type(x))
#         print(x)
        split_x=set(str(x).split("|"))
        len_processed=len(split_x)
        len_raw=len(str(x).split("|"))
#         if "|" in str(x) and len_processed<len_raw:
#             print("*****\nraw,",x)
#             print('processed:',"|".join(list(split_x)))
            
        return "|".join(list(split_x))
    except:
        print(x, 'func_split_set2 traceback')
        traceback.print_exc()
        return x
def  get_split_data_for_event_flag(df,event_flagname):

    X = df.drop([
           event_flagname+"_01"], axis=1)

    # for indicator in indicator_fieldname_list:
    # X = X.drop(indicator_fieldname_list, axis=1)

    y = df[event_flagname+"_01"]

    class_weight_full_df=get_class_weights(df,event_flagname)
    print("class_weight_full_df",class_weight_full_df)
    
    
    
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = random_state,stratify=y)
    # check the shape of X_train and X_test
    
#     class_weight_train_df=get_class_weights(df,event_flagname)
    
    print(X_train.shape, X_test.shape)
    
    return X_train, X_test, y_train, y_test,class_weight_full_df

def get_class_weights(df,event_flagname):
    print(df.columns)
    print(df.shape)
    print(df[event_flagname+"_01"].value_counts())
    neg, pos = np.bincount(df[event_flagname+"_01"])
    print("neg, pos",neg, pos)
    total = neg + pos
    print('\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(total, pos, 100 * pos / total))
    
    # Scaling by total/2 helps keep the loss to a similar magnitude.
    # The sum of the weights of all examples stays the same.
    weight_for_0 = (1 / neg) * (total / 2.0)
    weight_for_1 = (1 / pos) * (total / 2.0)

    class_weight = {0: weight_for_0, 1: weight_for_1}

    print('Weight for class 0: {:.2f}'.format(weight_for_0))
    print('Weight for class 1: {:.2f}'.format(weight_for_1))
    return class_weight

def  get_split_data_for_event_flag(df,event_flagname):

    X = df.drop([
           event_flagname+"_01"], axis=1)

    # for indicator in indicator_fieldname_list:
    # X = X.drop(indicator_fieldname_list, axis=1)

    y = df[event_flagname+"_01"]

    class_weight_full_df=get_class_weights(df,event_flagname)
    print("class_weight_full_df",class_weight_full_df)
    
    
    
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = random_state,stratify=y)
    # check the shape of X_train and X_test
    
#     class_weight_train_df=get_class_weights(df,event_flagname)
    
    print(X_train.shape, X_test.shape)
    
    return X_train, X_test, y_train, y_test,class_weight_full_df



def get_random_forest_score(train_embedding,test_embedding,X_train,X_test,y_train,y_test, event_flagname,feature,identifier,random_grid):
    print(y_train.shape)
    print(y_test.shape)
    
    rf = Pipeline([
    ("clf", BalancedRandomForestClassifier())
    ])

    rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid
                              ,n_iter = 5
                               , scoring='recall'
                               ,cv = 5
                               , verbose=2, n_jobs=n_jobs,random_state=17,
                              return_train_score=True)

    rf_random.fit(train_embedding,y_train[event_flagname+"_01"])
    
    print("rf_random.best_params_") 
    print(rf_random.best_params_)

    print("rf_random.best_score_")
    print(rf_random.best_score_)
    
    best_rf_model=rf_random.best_estimator_
    joblib.dump(best_rf_model, feature+'_sbert_rf_model_EV_'+event_flagname+'_balancedRFC_'+identifier+'.pkl')
    
    y_train_proba=best_rf_model.predict_proba(train_embedding)
    y_test_proba=best_rf_model.predict_proba(test_embedding)
    
    y_test_yes_prob=[]
    for i in y_test_proba:
        y_test_yes_prob.append(i[1])
    y_train_yes_prob=[]
    for i in y_train_proba:
        y_train_yes_prob.append(i[1])
    X_train[feature+'_sbert_rf_score']=y_train_yes_prob
    X_test[feature+'_sbert_rf_score']=y_test_yes_prob
    
    y_test_pred=best_rf_model.predict(test_embedding)
    evaluate_model(y_test[event_flagname+"_01"], y_test_pred)
    return X_train,X_test

def get_random_grid():
    from imblearn.ensemble import BalancedRandomForestClassifier
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 5)]
    # Number of features to consider at every split
    max_features = [ 'sqrt']
    #max_features = [int(x) for x in np.linspace(start = 5000, stop = 10000, num = 5)]
    # Maximum number of levels in tree
    #     max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth = [int(x) for x in np.linspace(10, 50, num = 5)]
    #     max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]

    random_grid = {'clf__n_estimators': n_estimators,
               'clf__max_features': ['sqrt'],
               'clf__max_depth': max_depth,
               'clf__min_samples_split': min_samples_split,
               'clf__min_samples_leaf': min_samples_leaf,
               'clf__bootstrap': bootstrap
               #,'clf__n_features':n_features 
    #                   ,'vect__ngram_range': [(1,1), (1,2)]
    #                ,'vect__max_df':[0.95]
    #                ,'tfidf__use_idf': (True,False)
               ,"clf__criterion": ["gini", "entropy"]
              }
    # random_grid['vect__ngram_range']= [ ngram_tuple]
    from pprint import pprint

    pprint(random_grid)
    return random_grid
def get_new_feature(row):
    return(str(row['GENERIC_NAME'])+" "+str(row['EVENT_VERBATIM'])+" SEX is "+str(row['PT_SEX'])+" PREGNANCY_STATUS is "+str(row['PT_PREGNANCY_STATUS']))


sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
identifier="25Oct2021_sbert_new_feature_"
file_name="/efsdata/faraaz/pickle_files_193/majorst_feature_importance_all_"+event_flagname+"_30aug2021_eventlevel_v5.pkl"
if event_flagname=='PATIENT_DIED_FLAG':
    file_name="/efsdata/faraaz/pickle_files_193/majorst_feature_importance_all_PATIENT_DIED_FLAG_14sep2021_event_patient_died_v6.pkl"
print(file_name)
df=pd.read_pickle(file_name)
df.shape
df=df.reset_index()

print(df.shape)
print(df.columns)
#         df=df[0:5000]
#         flagname=['EVENTS__EVENT_VERBATIM-'+event_flagname]
feature_list=['PRODUCT_NAME_AS_REPORTED',
'PRODUCT_NAME', 'GENERIC_NAME', 'EVENT_VERBATIM']

categorical_field_list=['REPORT_SOURCE_TYPE','PT_SEX','PT_PREGNANCY_STATUS']

if 'PT_SEX' in categorical_field_list:
#     df.PT_SEX=df.PT_SEX.str[1:]
    df.PT_SEX=df.PT_SEX.apply(func_split_set,fieldname='PT_SEX')
    print(df.PT_SEX.value_counts())
    index = df.index
    df.loc[df.PT_SEX!='female','PT_SEX']='male'    
    print("df.PT_SEX\n",df.PT_SEX.value_counts())


if 'PT_PREGNANCY_STATUS' in categorical_field_list:            
#     df.PT_PREGNANCY_STATUS=df.PT_PREGNANCY_STATUS.str[1:]
    df.PT_PREGNANCY_STATUS=df.PT_PREGNANCY_STATUS.apply(func_split_set,fieldname='PT_PREGNANCY_STATUS')

    print("df.PT_PREGNANCY_STATUS\n",df.PT_PREGNANCY_STATUS.value_counts())
    index = df.index
    df.loc[df.PT_PREGNANCY_STATUS!='Yes','PT_PREGNANCY_STATUS']='No'   

    print("df.PT_PREGNANCY_STATUS\n",df.PT_PREGNANCY_STATUS.value_counts())



df['new_feature']=df.apply(lambda row: get_new_feature(row), axis=1)
print(df.columns)
df[event_flagname+"_01"] = df['EVENTS__EVENT_VERBATIM-'+event_flagname].replace(['No', 'Yes'], [0, 1]).replace(['NO', 'YES'], [0, 1])

df=pd.get_dummies(data=df, columns=categorical_field_list)
print(df.shape)
print("df.columns",df.columns)   

sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
identifier="25Oct2021_sbert_new_feature_"
file_name="/efsdata/faraaz/pickle_files_193/majorst_feature_importance_all_"+event_flagname+"_30aug2021_eventlevel_v5.pkl"
if event_flagname=='PATIENT_DIED_FLAG':
    file_name="/efsdata/faraaz/pickle_files_193/majorst_feature_importance_all_PATIENT_DIED_FLAG_14sep2021_event_patient_died_v6.pkl"
print(file_name)
df=pd.read_pickle(file_name)
df.shape
df=df.reset_index()

print(df.shape)
print(df.columns)
#         df=df[0:5000]
#         flagname=['EVENTS__EVENT_VERBATIM-'+event_flagname]
feature_list=['PRODUCT_NAME_AS_REPORTED',
'PRODUCT_NAME', 'GENERIC_NAME', 'EVENT_VERBATIM']

categorical_field_list=['REPORT_SOURCE_TYPE','PT_SEX','PT_PREGNANCY_STATUS']

if 'PT_SEX' in categorical_field_list:
#     df.PT_SEX=df.PT_SEX.str[1:]
    df.PT_SEX=df.PT_SEX.apply(func_split_set,fieldname='PT_SEX')
    print(df.PT_SEX.value_counts())
    index = df.index
    df.loc[df.PT_SEX!='female','PT_SEX']='male'    
    print("df.PT_SEX\n",df.PT_SEX.value_counts())


if 'PT_PREGNANCY_STATUS' in categorical_field_list:            
#     df.PT_PREGNANCY_STATUS=df.PT_PREGNANCY_STATUS.str[1:]
    df.PT_PREGNANCY_STATUS=df.PT_PREGNANCY_STATUS.apply(func_split_set,fieldname='PT_PREGNANCY_STATUS')

    print("df.PT_PREGNANCY_STATUS\n",df.PT_PREGNANCY_STATUS.value_counts())
    index = df.index
    df.loc[df.PT_PREGNANCY_STATUS!='Yes','PT_PREGNANCY_STATUS']='No'   

    print("df.PT_PREGNANCY_STATUS\n",df.PT_PREGNANCY_STATUS.value_counts())



df['new_feature']=df.apply(lambda row: get_new_feature(row), axis=1)
print(df.columns)
df[event_flagname+"_01"] = df['EVENTS__EVENT_VERBATIM-'+event_flagname].replace(['No', 'Yes'], [0, 1]).replace(['NO', 'YES'], [0, 1])

df=pd.get_dummies(data=df, columns=categorical_field_list)
print(df.shape)
print("df.columns",df.columns)   

# df=df[0:10000]


# df=df[0:10000]


X_train, X_test, y_train, y_test,class_weight_full_df=get_split_data_for_event_flag(df,event_flagname)

print("X_train.shape",X_train.shape)
print("X_test.shape",X_test.shape)
print("y_train.shape",y_train.shape)
print("y_test.shape",y_test.shape)

print("X_train.columns\n",X_train.columns)

X_train=X_train.reset_index()
X_test=X_test.reset_index()
y_train=y_train.reset_index()
y_test=y_test.reset_index()


feature='new_feature'
print(feature)
#     print(X_train[feature].head())
train_embedding=sbert_model.encode(X_train[feature])

test_embedding=sbert_model.encode(X_test[feature])
#     print(type(train_embedding))
print("train_embedding.shape",train_embedding.shape)
#     print(type(test_embedding))
print("test_embedding.shape",test_embedding.shape)
random_grid=get_random_grid()
# X_train,X_test=get_random_forest_score(train_embedding ,test_embedding,X_train,X_test,y_train,y_test,event_flagname,feature,identifier,random_grid)


rf = Pipeline([
("clf", BalancedRandomForestClassifier(n_estimators=1550
                                      ,min_samples_split= 5
                                       ,min_samples_leaf= 1
                                       ,max_features= 'sqrt'
                                       ,max_depth= 40
                                       ,criterion= 'gini'
                                       ,bootstrap= True
                                      ))
])


rf.fit(train_embedding,y_train[event_flagname+"_01"])

y_test_pred=rf.predict(test_embedding)

import lime
import lime.lime_tabular

predict_fn_rf = lambda x: rf.predict_proba(x).astype(float)
explainer = lime.lime_tabular.LimeTabularExplainer(test_embedding,class_names=[0,1],kernel_width=5)

text="<input text>"
exp = explainer.explain_instance(sbert_model.encode(text), rf.predict_proba,num_features=10)

exp.show_in_notebook(show_all=False)
