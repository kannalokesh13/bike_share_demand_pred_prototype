import sys
import os
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
import pandas as pd


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            model_path=os.path.join('artifacts','model.pkl')

            preprocessor=load_object(preprocessor_path)
            model=load_object(model_path)

            data_scaled=preprocessor.transform(features)

            pred=model.predict(data_scaled)
            return pred
            

        except Exception as e:
            logging.info("Exception occured in prediction")
            raise CustomException(e,sys)
        
class CustomData:
    def __init__(self,
                 hour:int,
                 season:int,
                 holiday:int,
                 workingday:int,
                 weather:int,
                 temp:float,
                 atemp:float,
                 humidity:int,
                 windspeed:float
                 ):
        
        self.hour=hour
        self.season=season
        self.holiday=holiday
        self.workingday=workingday
        self.weather=weather
        self.temp=temp
        self.atemp = atemp
        self.humidity = humidity
        self.windspeed = windspeed

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'hour':[self.hour],
                'season':[self.season],
                'holiday':[self.holiday],
                'workingday':[self.workingday],
                'weather':[self.weather],
                'temp':[self.temp],
                'atemp':[self.atemp],
                'humidity':[self.humidity],
                'windspeed':[self.windspeed]

            }
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Dataframe Gathered')
            return df
        except Exception as e:
            logging.info('Exception Occured in prediction pipeline')
            raise CustomException(e,sys)
