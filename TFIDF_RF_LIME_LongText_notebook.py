import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from unidecode import unidecode
import re
def clean_text(raw_text):
    try:
        if type(raw_text)==None:
            return 'NAN'
        text=unidecode(raw_text)
        text=str(text).lower() #Normalization
        text=re.sub(r"(@\[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", text) # Removing Unicode Characters
        text=str(text).replace("\n","").replace("\r","").replace("\t","").replace(' \n ',' ')
        text=str(text).strip()

    #         text = re.sub("https?://.*[\t\r\n]*", "", text)
        return text
    except:
        print(raw_text)
        traceback.print_exc()
        return 'NAN'

df=pd.read_pickle("<input_filename>")
print(df.shape)
df=df.dropna()

print(df.shape)
print(df.columns)

random_state=802
df['text']=df['NARRATIVE_TEXT'].apply(clean_text)
print(df[df['text']=='NAN'].shape)

event_flagname='<flagname>'
print(df[event_flagname].value_counts())
no_variants=['no','nO','No','NO']
for no_variant in no_variants:
    df.loc[df[event_flagname] == no_variant, event_flagname] = 0
yes_variants=['yes' ,'yeS' ,'yEs' ,'yES' ,'Yes' ,'YeS' ,'YEs' ,'YES']
for yes_variant in yes_variants:
    df.loc[df[event_flagname] == yes_variant, event_flagname] = 1

df=df.reset_index()        

# X_train, X_test, y_train, y_test = train_test_split(df['text'],
#                            df[event_flagname],stratify=df[event_flagname], random_state=random_state
# )
# y_train = y_train.astype(int)
# y_test = y_test.astype(int)
# print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


train, test = train_test_split(df,stratify=df[event_flagname], random_state=802)
print(train.shape)
print(train.columns)

X_train = train.drop([event_flagname
            ], axis=1)

y_train = train[event_flagname]

X_test = test.drop([event_flagname
            ], axis=1)

y_test = test[event_flagname]

# check the shape of X_train and X_test

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

feature_name='text'

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn import model_selection
import _pickle as cPickle
from sklearn.metrics import f1_score
import joblib

y_train=y_train.astype('int')
y_test=y_test.astype('int')

from sklearn.feature_extraction.text import TfidfVectorizer

print("feature_name",feature_name)
feature=feature_name
random_state=802
n_jobs=25
best_params={}
best_params[feature]={'vect__ngram_range': (1, 2), 'vect__max_features': 1000, 'vect__max_df': 0.75, 'tfidf__use_idf': True, 'tfidf__sublinear_tf': False, 'tfidf__smooth_idf': False, 'tfidf__norm': 'l1', 'clf__n_estimators': 2000, 'clf__min_samples_split': 5, 'clf__min_samples_leaf': 4, 'clf__max_features': 'sqrt', 'clf__max_depth': 70, 'clf__criterion': 'gini', 'clf__bootstrap': False}
                     
vectorizer = TfidfVectorizer(lowercase=True, 
                                      ngram_range=best_params[feature]['vect__ngram_range'],
                                      max_features=best_params[feature]['vect__max_features'],
                                      max_df=best_params[feature]['vect__max_df'],
                                      use_idf=best_params[feature]['tfidf__use_idf'],
                                      sublinear_tf=best_params[feature]['tfidf__sublinear_tf'],
                                      smooth_idf=best_params[feature]['tfidf__smooth_idf'],
                                      norm=best_params[feature]['tfidf__norm'],
                                     )
train_vectors = vectorizer.fit_transform(X_train[feature].astype('U'))
rf_train=RandomForestClassifier(random_state=random_state
                                        ,n_jobs=n_jobs
                                ,n_estimators= best_params[feature]['clf__n_estimators']
                                ,min_samples_split= best_params[feature]['clf__min_samples_split']
                                 ,min_samples_leaf= best_params[feature]['clf__min_samples_leaf']
                                 ,max_features= best_params[feature]['clf__max_features']
                                 ,max_depth= best_params[feature]['clf__max_depth']
                                 ,criterion= best_params[feature]['clf__criterion']
                                 ,bootstrap= best_params[feature]['clf__bootstrap']

                                        )
#     print(y_train)
#     print(train_vectors)
rf_train.fit(train_vectors, y_train)
test_vectors= vectorizer.transform(X_test[feature])
y_test_pred=rf_train.predict(test_vectors)
print(y_test_pred)



from lime import lime_text
from sklearn.pipeline import make_pipeline

c={}

feature_list=['text']
for feature in feature_list:
    c[feature] = make_pipeline(vectorizer, rf_train)
    
from lime.lime_text import LimeTextExplainer
class_names = [0, 1]

X_test[event_flagname]=y_test
X_test['pred_'+event_flagname]=y_test_pred
X_test.columns

X_test[event_flagname].value_counts()


X_test[np.logical_and(X_test[event_flagname]==1,X_test['pred_'+event_flagname]==1)
                      ]['CASE_NUMBER']
                      
import traceback
def execute_lime_explainer(X_test,case_number):
    rf_model_input_dict={}
    explainer={}
    for idex,final_feature_dict in X_test[X_test.CASE_NUMBER==case_number].iterrows():
        for feature in feature_list:
            try:
                print("Feature=>",feature,":",final_feature_dict[feature])
                explainer[feature] = LimeTextExplainer(class_names=class_names)
                if feature =='NARRATIVE_TEXT1':
                    exp = explainer[feature].explain_instance(final_feature_dict[feature], c['NARRATIVE_TEXT'].predict_proba, num_features=50)
                    print('\tProbability("Yes") =', c['NARRATIVE_TEXT'].predict_proba([clean_text(final_feature_dict[feature])])[0,1])
                    rf_model_input_dict[feature+'_tfidf_score']=c['NARRATIVE_TEXT'].predict_proba([clean_text(final_feature_dict[feature])])[0,1]
                #     print('True class: %s' % class_names[case1.actual_value[idx]])
                    print("\tLIME Analysis:",sorted(exp.as_list(), key = lambda x: x[1],reverse=True))
                else:
                    exp = explainer[feature].explain_instance(final_feature_dict[feature], c[feature].predict_proba, num_features=50)
                    print('\tProbability("Yes") =', c[feature].predict_proba([clean_text(final_feature_dict[feature])])[0,1])
                    rf_model_input_dict[feature+'_tfidf_score']=c[feature].predict_proba([clean_text(final_feature_dict[feature])])[0,1]
                #     print('True class: %s' % class_names[case1.actual_value[idx]])
                    print("\tLIME Analysis:",sorted(exp.as_list(), key = lambda x: x[1],reverse=True))
            except:
                traceback.print_exc()

execute_lime_explainer(X_test,'2019-233579')
