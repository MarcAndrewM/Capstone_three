#!python 3

from os import name
from flask import Flask, request, render_template, url_for
import pandas as pd
import numpy as np

from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.impute import KNNImputer

from sklearn.metrics.pairwise import cosine_similarity
import pickle


app = Flask(__name__)

with open('./data/four_year_df.pkl','rb') as f:
    four_year_df = pickle.load(f)

with open('./data/num_scaled_df.pkl','rb') as f:
    num_scaled_df = pickle.load(f) 

with open('./data/school_df.pkl','rb') as f:
    school_df = pickle.load(f) 

cosine_sim = cosine_similarity(num_scaled_df,num_scaled_df)

def weighted_df(w1,w2,w3,w4,w5,w6):
    weighted_df=num_scaled_df.copy()
    weighted_df['Average Net Price']=(weighted_df['Average Net Price']*w1)
    weighted_df['Percent Admitted']=(weighted_df['Percent Admitted']*w2)
    weighted_df['Percent Women']=(weighted_df['Percent Women']*w3)
    weighted_df['Institution Size Cat']=(weighted_df['Institution Size Cat']*w4)
    weighted_df['SAT Math 25th']=(weighted_df['SAT Math 25th']*w5)
    weighted_df['SAT Verbal 25th']=(weighted_df['SAT Verbal 25th']*w6)
    return weighted_df

    
def get_recommendations(school, cosine_sim=cosine_sim):
    indices = pd.Series(four_year_df.index,index=four_year_df['School']).drop_duplicates()
    # Get the index of the school that matches user input
    idx = indices[school]
    # Get the pairwsie similarity scores of all schools with requested
    sim_scores = list(enumerate(cosine_sim[idx]))
    # Sort the schools based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # Get the scores of the 10 most similar schools
    sim_scores = sim_scores[:6]
    # Get the school indices/list comp
    indices = [i[0] for i in sim_scores]
    # Return the top 5 most similar colleges
    return four_year_df[['School','City','State','Average Net Price','Percent Admitted','Percent Women','Institution Size Cat','SAT Math 25th','SAT Verbal 25th','Admissions Web Address']].iloc[indices].reset_index(drop=True)


@app.route('/')
def welcome():
    return render_template("index.html")

@app.route('/results',methods=['GET','POST'])
def results():
    name = request.form.get('name')
    school_lst = school_df.iloc[:,2].tolist()
    if name not in school_lst:
        return render_template("error.html")
    else:
        df = get_recommendations(name)
        return render_template("results.html",df=df, name=name)

@app.route('/custom',methods=['GET','POST'])
def custom():
    percent_admitted = int(request.form.get("percent_admitted"))
    avg_net_price = int(request.form.get("avg_net_price"))
    percent_women = int(request.form.get("percent_women"))
    inst_size = int(request.form.get("inst_size"))
    sat_math = int(request.form.get("sat_math"))
    sat_verbal = int(request.form.get("sat_verbal"))
    name = request.form.get('name')
    df = weighted_df(avg_net_price,percent_admitted,percent_women,inst_size,sat_math,sat_verbal)
    weighted_cosine_sim = cosine_similarity(df,df)
    new_df = get_recommendations(name, weighted_cosine_sim)
    return render_template("custom.html", percent_admitted=percent_admitted, avg_net_price=avg_net_price, percent_women=percent_women,inst_size=inst_size,sat_math=sat_math,sat_verbal=sat_verbal,name=name, new_df = new_df)

@app.route('/school_list',methods=['GET','POST'])
def school_list():
    df = school_df
    return render_template("school_list.html",df=df)

@app.route('/base')
def base():
    return render_template("base.html")

@app.route('/error',methods=['GET','POST'])
def error():
    return render_template("error.html")

# @app.route('/metrics')
# def metrics():
#     return render_template("metrics.html")

# @app.route('/practice')
# def practice():
#     return render_template("practice.html")

if  __name__ == '__main__':
    app.run(debug=True)