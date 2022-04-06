import os
import logging
from config import config

from preprocessing import process_and_concatenate, pivot_and_impute
from postprocessing import average_predictions, median_predictions
from base_model import generate_predictions


def main(retrain=True, validate=True):
    if(validate):
        logging.basicConfig(format='%(asctime)s || %(levelname)s || %(message)s || %(lineno)d || %(process)d', filename='logs/log_validate.txt', filemode='w', level = logging.INFO)
    else:
        logging.basicConfig(format='%(asctime)s || %(levelname)s || %(message)s || %(lineno)d || %(process)d', filename='logs/log_submission.txt', filemode='w', level = logging.INFO)


    abunds, temps = process_and_concatenate()
    logging.info("Initial files input and stacked")
    abunds_pvt, temps_pvt = pivot_and_impute(abunds, temps)
    logging.info("Data pivoted and filled")
    for kernel_width in config.kernel_width:
        for input_smoothing in config.input_smoothing:
            for timesteps in config.timesteps:
                for nion in config.nions:
                    generate_predictions(abunds_pvt,
                                         temps_pvt,
                                         timesteps,
                                         nion,
                                         retrain=retrain,
                                         validate=validate,
                                         model_type='cnn',
                                         input_smoothing=input_smoothing,
                                         kernel_width=kernel_width)
                    logging.info(f"Predictions generated for {timesteps} timesteps and {nion} ions with {input_smoothing} step smoothing and {kernel_width} kernel size")

    average_predictions()
    median_predictions()

if __name__ == '__main__':
    main(retrain=False, validate=True)