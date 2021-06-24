"""File for utilizing predictive model trained on Neural Network
leveraging Natural Language Processing."""

# Import package for loading the trained model
from tensorflow.keras.models import load_model

# Import package for loading the data transformation pipelines
import joblib

# Import dependencies for the predictive model
import pandas as pd
import numpy as np

## TODO
# Import routing and HTML support


## TODO
# Instantiate the router


## TODO
# Instantiate templates


## NOTE - please be sure to check specific file path matches 
# Load the model and data pipelines
model = load_model("modeling_files/model.sav")
text_pipeline = joblib.load("modeling_files/text_pipeline.pkl")
quant_pipeline = joblib.load("modeling_files/quant_pipeline.pkl")


## NOTE - the parameters for this may change tonight due to changes in the model 
        # training process, but I (Ryan) will help advise on how to change this file based
        # on those changes if necessary.
# Create function for generating predictions
def predict(blurb, backers, goal):
    """This function will take user input gathered from the front end
    generate a prediction on whether or not the kickstarter campaign
    will succeed.

    The entered text is also automatically converted to vectors and
    processed for the model to receive.

    Quantitative features are converted in their own pipeline as well.

    -------------
    Parameters:
        blurb: (String) Pitch from the user that will be processed and 
            vectorized to be run through the predictive model.
        backers: (Integer) Anticipated number of backers for the campaign.
        goal: (Integer) Dollar amount that is being sought for the campaign.

    -------------
    Returns:
        A pass or fail prediction for the campaign
    """

    # Put the arguments into a dataframe
    df = pd.DataFrame(columns=["blurb",
                               "backers",
                               "goal"
                              ],
                      data=[[blurb,
                            backers,
                            goal
                           ]]
                     )

    # Separate the data
    text_data = df["blurb"]
    quant_data = df[["backers", "goal"]]

    # Transform the data
    text_feat_set = text_pipeline.transform(text_data)
    quant_feat_set = quant_pipeline.transform(quant_data)

    # Merge the transformed data
    prediction_data = np.concatenate(
        (text_feat_set.todense(), quant_feat_set),
        axis=1
    )

    # Generate a prediction from the transformed data
    y_pred = model.predict(prediction_data)

    # Establish a result based on the prediction
    if y_pred[0][0] == 0:
        result = "Fail"
    
    elif y_pred[0][0] == 1:
        result = "Succeed"

    return f"Your campaign will likely: {result}!"

## TODO - I recommend taking a look at my airbnb projeect from last month 
        # for insight on this part if you get stuck:
        # https://github.com/Lambdata-Build-Week/DS-airbnb/blob/main/app/ml/ml.py
# Route the inputs of from the HTML form into the predictive model