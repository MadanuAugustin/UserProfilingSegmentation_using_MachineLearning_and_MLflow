
import sys
import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from src.UserProfilingSegmentation import logger, CustomException
from src.UserProfilingSegmentation.pipeline.prediction_pipeline import PredictionPipeline



# initializing the flask app

app = Flask(__name__)


# route to display the home page

@app.route('/predict', methods = ['POST', 'GET'])
def predict_datapoint():

    if request.method == 'GET':
        return render_template('index.html')
    

    else : 
        try:
            Age = request.form.get('Age')
            Gender = request.form.get('Gender')
            IncomeLevel = request.form.get('IncomeLevel')
            TimeSpentOnlinehrsweekday = request.form.get('TimeSpentOnlinehrsweekday')
            TimeSpentOnlinehrsweekend = request.form.get('TimeSpentOnlinehrsweekend')
            LikesandReactions = request.form.get('LikesandReactions')
            ClickThroughRates = request.form.get('ClickThroughRates')

            data = [TimeSpentOnlinehrsweekday, TimeSpentOnlinehrsweekend, LikesandReactions, ClickThroughRates, Age, Gender, IncomeLevel]
            
            logger.info(f'-----------Feteched data successfully from the user--------------')
            

            data = np.array(data).reshape(1, 7)

            data = pd.DataFrame(data)

            print(data)

            obj = PredictionPipeline()

            results = obj.predictDatapoint(data)

            logger.info(f'-----------Below is the final result {results}------------------')

            print(results)

            return render_template('index.html', results = str(results))


        except Exception as e:
            raise CustomException(e, sys)
        



if __name__ == '__main__':
    app.run(host = "0.0.0.0", debug = True) ## http://127.0.0.1:5000