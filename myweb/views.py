from django.http import HttpResponse
from django.shortcuts import render
import pickle
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler,StandardScaler  



def index(request):
  return render(request,"index.html",{})
   
def result(request):
   

    
    
    # scaler = joblib.load('minmaxscale.pkl')

    lis = []

    age = int(request.GET['age'])
    work = int(request.GET['workclass'])
    education = int(request.GET['education'])
    occupation = int(request.GET['occupation'])
    sex = int(request.GET['sex'])
    workhour = int(request.GET['WorkHour'])
    country = int(request.GET['country'])


  # create dataframe
    data = {
       'age' : [age],
       'workclass' : [work],
       'education' : [education],
       'occupation' : [occupation],
       'sex' : [sex],
       'workHour' : [workhour],
       'Country' : [country]
    }

    # to DATAFRAME
    x_data = pd.DataFrame(data)
    x_data = np.array(x_data)

   
  # model Test
    mod = joblib.load('XGBMODEL.pkl')

    ans = mod.predict(x_data)
    ans = np.where(ans>0.5,1,0)

    if ans == 0:
      ans = 'Below or Equal to 50,0000Rs'
    elif ans == 1:
      ans = 'Above 50,0000Rs'

    return render(request,"result.html",{'ans':ans,'lis':data})

    # return render(request,"result.html",{'ans':ans,'lis':data})

    