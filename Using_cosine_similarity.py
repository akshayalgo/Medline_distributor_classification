import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import argparse
import pickle

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="path to input dataset")
ap.add_argument("-T", "--target", required=True,
                help="Give name of target columnname")
ap.add_argument("-i", "--input", required=True,
                help="Give name of input columnname")
ap.add_argument("-it", "--item", required=False, default = None,
                help="Enter the item description")
ap.add_argument("-r", "--result", required=False, default = None,
                help="Enter the item description")
args = vars(ap.parse_args())

def get_csv_in_dataframe(path):
    return pd.read_csv(path)

def cleaning(data,target,input):
    print("Start cleaning process ......")
    data[input] = data[input].apply(lambda x: x.replace(","," "))
    data = data[[target,input]]
    print("Done cleaning process.......")
    return data

def Process_tfidf(data,input,target):
    print("Appliying tf-idf process ......")
    dct_mapping = df.groupby(target).groups
    final_dict = {}
    for key, value in dct_mapping.items(): 
        final_dict[key] = ",".join((list(df.loc[list(value),:][input])))
    vec = TfidfVectorizer(lowercase=True,ngram_range=(1, 4))
    cat_vec = vec.fit_transform(final_dict.values())
    print("Completed tf-idf process ......")
    tfidf_dict = {"vec":vec,"cat_vec":cat_vec,"final_dct":list(final_dict.keys())}
    with open('vector.pickle', 'wb') as handle:
        pickle.dump(tfidf_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Stored pickle file ......")

def get_pred(x,vect_des,vec,final_dict):
    example = vec.transform([x.replace(","," ")])
    lst = list(cosine_similarity(example,vect_des)[0])
    ind = lst.index(max(lst))
    return final_dict[ind]

def get_vector():
    with open('vector.pickle', 'rb') as handle:
        vector_data = pickle.load(handle)
    return vector_data

def get_pred_on_data(df,input,target):
    vector_data = get_vector()
    vec, vect_des, final_dict = vector_data['vec'], vector_data['cat_vec'], vector_data['final_dct']
    pre = df[input].apply(lambda x: get_pred(x,vect_des,vec,final_dict))
    df['pred_by_model'] = pre
    df.to_csv("result.csv", index=False)

def get_pred_on_itm_description(x,input,target):
    vector_data = get_vector()
    vec, vect_des, final_dict = vector_data['vec'], vector_data['cat_vec'], vector_data['final_dct']
    return get_pred(x,vect_des,vec,final_dict)

df = get_csv_in_dataframe(path = args['dataset'])
clean_df = cleaning(data = df, target = args['target'], input = args['input'])
Process_tfidf(data = clean_df, input = args['input'], target = args['target'])
if args['result'] is None:
    print("Skipped the prediction on our csv data ...")
else:
    print("genrating the result on the csv")
    get_pred_on_data(clean_df,input = args['input'], target = args['target'])
    print("Completed the genrating prediction on csv")

if args['item'] is None:
    print("Completed all process .....")
else:
    print("Predicted calss for item ----> ",get_pred_on_itm_description(x = args['item'], input = args['input'], target = args['target']))
    print("Completed all process .......")