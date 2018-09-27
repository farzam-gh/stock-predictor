#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 23:04:58 2018

@author: farzam
"""
from flask import Flask,render_template,request
from wtforms import Form,validators,TextAreaField
import numpy as np
import pandas as pd
import finance


app=Flask(__name__)
url=('http://www.nasdaq.com/screening/'
     'companies-by-industry.aspx?exchange=NASDAQ&render=download')
stocklist=pd.read_csv("/home/stockpredictor/mysite/companylist.csv")

class StockSelection(Form):
#    url=('http://www.nasdaq.com/screening/'
#     'companies-by-industry.aspx?exchange=NASDAQ&render=download')
    stocklist=pd.read_csv(r"/home/stockpredictor/mysite/companylist.csv")
    lst=stocklist['Symbol'].tolist()
    stock=TextAreaField('',
        [validators.InputRequired('data needed'),
         validators.length(min=1,max=6)])



@app.route('/')
def index():
    form=StockSelection(request.form)
#    user_input=request.form['stock']

    return render_template('home.html',form=form)

@app.route('/prediction', methods=['POST'])
def prediction():
    error=[]
    form=StockSelection(request.form)
    stock=request.form['stock'].upper()
    lst=stocklist['Symbol'].tolist()

    if (request.method=='POST' and stock in lst ):
        ind=lst.index(stock)
        detail=stocklist.iloc[ind,:].values.astype(str)
        finance.stock=stock
        finance.get_data()
        predictions=finance.weekly_predict()
        return render_template('prediction.html',
                              stock=stock,
                              predictions=predictions,
                              detail=detail)
    else:
        error=['Stock name not found','You can find companies list',
               'http://eoddata.com/stocklist/NASDAQ.htm']
        return render_template('home.html',error=error)

@app.route('/about')
def about():
    return render_template("about.html")



if __name__ == '__main__':
    app.run(debug=True)




