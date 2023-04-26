import pandas as pd
import numpy as np

def convert_event_flags(df):
    event_flagnames=[
        'MEDICALLY_SIGNIFICANT_FLAG','DISABILITY_FLAG',
                     'HOSPITALIZATION_FLAG',
           'LIFE_THREATENING_FLAG', 'CONGENITAL_ANOMALY_FLAG',
            'PATIENT_DIED_FLAG'
    ]
    for event_flagname in event_flagnames:
        no_variants=['no','nO','No','NO']
        for no_variant in no_variants:
            df.loc[df['ans_val_'+event_flagname] == no_variant, 'ans_val_'+event_flagname] = 'no'
            df.loc[df['res_val_'+event_flagname] == no_variant, 'res_val_'+event_flagname] = 'no'
        yes_variants=['yes' ,'yeS' ,'yEs' ,'yES' ,'Yes' ,'YeS' ,'YEs' ,'YES']
        for yes_variant in yes_variants:
            df.loc[df['ans_val_'+event_flagname] == yes_variant, 'ans_val_'+event_flagname] = 'yes'
            df.loc[df['res_val_'+event_flagname] == yes_variant, 'res_val_'+event_flagname] = 'yes'

#     for event_flagname in event_flagnames:
#         print(df['ans_val_'+event_flagname].value_counts())
#         print(df['res_val_'+event_flagname].value_counts())
    df=df.reset_index()
    return df
    
  def get_event_serious_flag(row,key):
#     print(row.keys())
    if row[key+"MEDICALLY_SIGNIFICANT_FLAG"]=='yes' or row[key+"PATIENT_DIED_FLAG"]=='yes' or row[key+"CONGENITAL_ANOMALY_FLAG"]=='yes' or  row[key+"LIFE_THREATENING_FLAG"]=='yes' or  row[key+"HOSPITALIZATION_FLAG"]=='yes' or  row[key+"DISABILITY_FLAG"]=='yes':
        return 'yes'
    return 'no'
    
  from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
def evaluate_model(y_test, y_pred):
    # Print the Confusion Matrix and slice it into four pieces

    

    cm = confusion_matrix(y_test, y_pred)

    print('Confusion matrix\n\n', cm)
    # visualize confusion matrix with seaborn heatmap

    cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive', 'Actual Negative'],
                                     index=['Predict Positive', 'Predict Negative'])

#     sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')
    

    print(classification_report(y_test, y_pred))
    
  def get_evaluation_results_MEDICALLY_SIGNIFICANT_FLAG(source_type):
    #US PV Data Capture Form
    df_input=pd.read_csv("all_docs_dev2_2022_11_01_1_caseassembly_event_"+source_type+".csv")
    df_input=convert_event_flags(df_input)
    df_input["ans_val_EVENT_SERIOUS_FLAG"]=df_input.apply(get_event_serious_flag,key='ans_val_',axis=1)
    print(df_input["ans_val_EVENT_SERIOUS_FLAG"].value_counts())
#     df_input["res_val_EVENT_SERIOUS_FLAG"]=df_input.apply(get_event_serious_flag,key='res_val_',axis=1)
    print(df_input["res_val_MEDICALLY_SIGNIFICANT_FLAG"].value_counts(dropna=False))
    df_input.loc[np.logical_and(df_input["res_val_MEDICALLY_SIGNIFICANT_FLAG"]!='yes',df_input["res_val_MEDICALLY_SIGNIFICANT_FLAG"]!='no'), 'res_val_MEDICALLY_SIGNIFICANT_FLAG'] = 'no'
    print(df_input["res_val_MEDICALLY_SIGNIFICANT_FLAG"].value_counts(dropna=False))
    evaluate_model(df_input["ans_val_EVENT_SERIOUS_FLAG"], df_input["res_val_MEDICALLY_SIGNIFICANT_FLAG"])
    
  get_evaluation_results("US PV Data Capture Form")
  
  
  def get_evaluation_results_TP_Events(source_type):
    #US PV Data Capture Form
    df_input=pd.read_csv("all_docs_dev2_2022_11_01_1_caseassembly_event_"+source_type+".csv")
    print(df_input.shape)
    df_input=df_input[df_input["tp_count_EVENT_VERBATIM"]==1]
    print(df_input.shape)
    df_input=convert_event_flags(df_input)
    df_input["ans_val_EVENT_SERIOUS_FLAG"]=df_input.apply(get_event_serious_flag,key='ans_val_',axis=1)
    print(df_input["ans_val_EVENT_SERIOUS_FLAG"].value_counts())
    df_input["res_val_EVENT_SERIOUS_FLAG"]=df_input.apply(get_event_serious_flag,key='res_val_',axis=1)
    print(df_input["res_val_EVENT_SERIOUS_FLAG"].value_counts())
    evaluate_model(df_input["ans_val_EVENT_SERIOUS_FLAG"], df_input["res_val_EVENT_SERIOUS_FLAG"])
  
  def get_evaluation_results_TP_Events_MEDICALLY_SIGNIFICANT_FLAG(source_type):
    #US PV Data Capture Form
    df_input=pd.read_csv("all_docs_dev2_2022_11_01_1_caseassembly_event_"+source_type+".csv")
    print(df_input.shape)
    df_input=df_input[df_input["tp_count_EVENT_VERBATIM"]==1]
    print(df_input.shape)
    df_input=convert_event_flags(df_input)
    df_input["ans_val_EVENT_SERIOUS_FLAG"]=df_input.apply(get_event_serious_flag,key='ans_val_',axis=1)
    print(df_input["ans_val_EVENT_SERIOUS_FLAG"].value_counts())
#     df_input["res_val_EVENT_SERIOUS_FLAG"]=df_input.apply(get_event_serious_flag,key='res_val_',axis=1)
#     print(df_input["res_val_EVENT_SERIOUS_FLAG"].value_counts())
#     evaluate_model(df_input["ans_val_EVENT_SERIOUS_FLAG"], df_input["res_val_EVENT_SERIOUS_FLAG"])
    print(df_input["res_val_MEDICALLY_SIGNIFICANT_FLAG"].value_counts(dropna=False))
    df_input.loc[np.logical_and(df_input["res_val_MEDICALLY_SIGNIFICANT_FLAG"]!='yes',df_input["res_val_MEDICALLY_SIGNIFICANT_FLAG"]!='no'), 'res_val_MEDICALLY_SIGNIFICANT_FLAG'] = 'no'
    print(df_input["res_val_MEDICALLY_SIGNIFICANT_FLAG"].value_counts(dropna=False))
    evaluate_model(df_input["ans_val_EVENT_SERIOUS_FLAG"], df_input["res_val_MEDICALLY_SIGNIFICANT_FLAG"])
    
  get_evaluation_results_TP_Events("US PV Data Capture Form")
  
  def get_evaluation_results_event_flagname_TP_Events(source_type,event_flagname):
    #US PV Data Capture Form
    df_input=pd.read_csv("all_docs_dev2_2022_11_01_1_caseassembly_event_"+source_type+".csv")
    print(df_input.shape)
    df_input=df_input[df_input["tp_count_EVENT_VERBATIM"]==1]
    print(df_input.shape)
    df_input=convert_event_flags(df_input)
    print(df_input["res_val_"+event_flagname].value_counts(dropna=False))
    print("shape",df_input[np.logical_and(df_input["res_val_"+event_flagname]!='yes',df_input["res_val_"+event_flagname]!='no')].shape)
    df_input.loc[np.logical_and(df_input["res_val_"+event_flagname]!='yes',df_input["res_val_"+event_flagname]!='no'), 'res_val_'+event_flagname] = 'no'
    print(df_input["res_val_"+event_flagname].value_counts(dropna=False))
#     print(df_input.columns)
#     df_input["ans_val_EVENT_SERIOUS_FLAG"]=df_input.apply(get_event_serious_flag,key='ans_val_',axis=1)
    print(df_input["ans_val_"+event_flagname].value_counts(dropna=False))
#     df_input["res_val_EVENT_SERIOUS_FLAG"]=df_input.apply(get_event_serious_flag,key='res_val_',axis=1)
    print(df_input["res_val_"+event_flagname].value_counts(dropna=False))
    evaluate_model(df_input["ans_val_"+event_flagname], df_input["res_val_"+event_flagname])
    
  get_evaluation_results_event_flagname_TP_Events("US PV Data Capture Form","HOSPITALIZATION_FLAG")
  
  def get_evaluation_results_event_flagname(source_type,event_flagname):
    #US PV Data Capture Form
    df_input=pd.read_csv("all_docs_dev2_2022_11_01_1_caseassembly_event_"+source_type+".csv")
#     print(df_input.shape)
#     df_input=df_input[df_input["tp_count_EVENT_VERBATIM"]==1]
    print(df_input.shape)
    df_input=convert_event_flags(df_input)
    print(df_input["res_val_"+event_flagname].value_counts(dropna=False))
    print("shape",df_input[np.logical_and(df_input["res_val_"+event_flagname]!='yes',df_input["res_val_"+event_flagname]!='no')].shape)
    df_input.loc[np.logical_and(df_input["res_val_"+event_flagname]!='yes',df_input["res_val_"+event_flagname]!='no'), 'res_val_'+event_flagname] = 'no'
    print(df_input["res_val_"+event_flagname].value_counts(dropna=False))
    
    print(df_input["ans_val_"+event_flagname].value_counts(dropna=False))
    print("shape ans_val",df_input[np.logical_and(df_input["ans_val_"+event_flagname]!='yes',df_input["ans_val_"+event_flagname]!='no')].shape)
    df_input.loc[np.logical_and(df_input["ans_val_"+event_flagname]!='yes',df_input["ans_val_"+event_flagname]!='no'), 'ans_val_'+event_flagname] = 'no'
    print(df_input["ans_val_"+event_flagname].value_counts(dropna=False))
#     print(df_input.columns)
#     df_input["ans_val_EVENT_SERIOUS_FLAG"]=df_input.apply(get_event_serious_flag,key='ans_val_',axis=1)
#     print(df_input["ans_val_"+event_flagname].value_counts(dropna=False))
#     df_input["res_val_EVENT_SERIOUS_FLAG"]=df_input.apply(get_event_serious_flag,key='res_val_',axis=1)
    print(df_input["res_val_"+event_flagname].value_counts(dropna=False))
    evaluate_model(df_input["ans_val_"+event_flagname], df_input["res_val_"+event_flagname])
    
  # "US PV Data Capture Form" "AE Form - Accredo" "CIOMS" "MedwatchFDAForm"
get_evaluation_results_event_flagname("US PV Data Capture Form","HOSPITALIZATION_FLAG")

