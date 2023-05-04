from flask import Flask,render_template,request
from src.exception import CustomException
from src.logger import logging
from src.pipeline.training_pipeline import mtrain
from src.pipeline.prediction_pipeline import CustomData
from src.pipeline.prediction_pipeline import PredictPipeline
import os
import sys

# def ttrain():
#     logging.info("starting")
#     mtrain()
#     logging.info("ending")

app=Flask(__name__)

@app.route('/')
def show(): 
    return render_template('index.html')

@app.route("/review",methods=['GET','POST'])
def drone():
    if request.method=="POST":
        year1=int(request.form['year'])
        hour1=int(request.form['hour'])
        season1=int(request.form['season'])
        holiday1=int(request.form['holiday'])
        workingday=int(request.form['working'])
        weather1=int(request.form['weather'])
        temp1=float(request.form['temp'])
        atemp1=float(request.form['atemp'])
        humid=int(request.form['humidity'])
        windspeed=float(request.form['wind'])

        results=[year1,hour1,season1,holiday1,workingday,weather1,temp1,atemp1,humid,windspeed]

        data=CustomData(hour=hour1,
                        season=season1,
                        holiday=holiday1,
                        workingday=workingday,
                        weather=weather1,
                        temp=temp1,
                        atemp=atemp1,
                        humidity=humid,
                        windspeed=windspeed
                        )
        

        
        final_data=data.get_data_as_dataframe()
        predict_pipeline=PredictPipeline()
        pred=int(predict_pipeline.predict(final_data))

        return render_template('result.html',final_result=pred)
    



if __name__ == '__main__':
    app.run(debug=True)
