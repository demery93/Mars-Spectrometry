import os
import logging
from config import config

from preprocessing import process_and_concatenate, pivot_and_impute
from postprocessing import average_predictions, median_predictions
from base_model import generate_predictions

RETRAIN = False
VALIDATE = True

if(VALIDATE):
    logging.basicConfig(format='%(asctime)s || %(levelname)s || %(message)s || %(lineno)d || %(process)d', filename='logs/log_validate.txt', filemode='w', level = logging.INFO)
else:
    logging.basicConfig(format='%(asctime)s || %(levelname)s || %(message)s || %(lineno)d || %(process)d', filename='logs/log_submission.txt', filemode='w', level = logging.INFO)


abunds, temps = process_and_concatenate()
logging.info("Initial files input and stacked")
abunds_pvt, temps_pvt = pivot_and_impute(abunds, temps)
logging.info("Data pivoted and filled")
for timesteps in config.timesteps:
    for nion in config.nions:
        generate_predictions(abunds_pvt, temps_pvt, timesteps, nion, retrain=RETRAIN, validate=VALIDATE, model_type='cnn')
        logging.info(f"Predictions generated for {timesteps} timesteps and {nion} ions")

average_predictions()
median_predictions()