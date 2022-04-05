import pandas as pd
import numpy as np

import os
import gc

from tqdm import tqdm
from config import config
from utils import preprocess_sample

def process_and_concatenate():
    '''
    :return: stacked dataframe of all samples

    This function iterates through the data directories, processes each sample, and stores it in an abundance
    dataframe and a temperature dataframe
    '''
    dirs = [config.train_data_path, config.val_data_path, config.test_data_path]
    abunds, temps = [], []
    for dir in dirs:
        for file in tqdm(os.listdir(dir)):
            sample = pd.read_csv(f"{dir}/{file}")
            sample['sample_id'] = file.split('.')[0]
            abund, temp = preprocess_sample(sample)

            abunds.append(abund)
            temps.append(temp)

            del abund, temp, sample
            gc.collect()

    abunds = pd.concat(abunds, axis=0)
    temps = pd.concat(temps, axis=0)

    return abunds, temps

def process_and_concatenate_with_supplemental():
    '''
    :return: stacked dataframe of all samples, including supplemental data

    This function iterates through the data directories, processes each sample, and stores it in an abundance
    dataframe and a temperature dataframe
    '''
    dirs = [config.train_data_path, config.val_data_path, config.test_data_path]
    supp_samples = get_supplemental_samples()
    abunds, temps = [], []
    for dir in dirs:
        for file in tqdm(os.listdir(dir)):
            sample = pd.read_csv(f"{dir}/{file}")
            sample['sample_id'] = file.split('.')[0]
            abund, temp = preprocess_sample(sample)

            abunds.append(abund)
            temps.append(temp)

            del abund, temp, sample
            gc.collect()

    for file in tqdm(supp_samples):
        sample = pd.read_csv(f"input/{file}")
        sample['sample_id'] = file.split("/")[1].split('.')[0]
        abund, temp = preprocess_sample(sample)

        abunds.append(abund)
        temps.append(temp)

        del abund, temp, sample
        gc.collect()

    abunds = pd.concat(abunds, axis=0)
    temps = pd.concat(temps, axis=0)

    return abunds, temps

def get_supplemental_samples():
    '''
    :return: List of supplemental samples file paths
    '''
    supp = pd.read_csv("input/supplemental_metadata.csv")
    supp = supp[supp.carrier_gas == 'he'].reset_index(drop=True)
    supp = supp[supp.different_pressure == 0].reset_index(drop=True)

    return list(supp.features_path.values)


def pivot_and_impute(abunds, temps):
    '''
    :param abunds: dataframe containing melted form of scaled abundance data
    :param temps: dataframe containing melted form of raw temperature data
    :return: pivoted, filled, and scaled abundance and temperature information

    For each sample, the abundance information is a matrix size nions x ntimesteps.
    The temperature information is averaged across ions at each timestep, making the matrix size 1 x ntimesteps

    NAs: Not all ions will have information at each timestep. I linearly interpolate where I can and then
    forward fill the end of the sequence and backfill the beginning of the sequence
    '''
    abunds = pd.pivot_table(abunds, index=['sample_id', 'm/z'], columns='time', values='abundance_scaled')
    temps = pd.pivot_table(temps, index=['sample_id'], columns='time',values='temp')

    abunds = abunds.interpolate(axis=1)
    temps = temps.interpolate(axis=1)

    abunds = abunds.ffill(axis=1).bfill(axis=1)
    temps = temps.ffill(axis=1).bfill(axis=1)

    abunds.fillna(0, inplace=True)
    temps.fillna(0, inplace=True)

    temps = temps / (temps.max()) # Not exactly sure why this works better than temps / (temps.max().max()) but followed my validation statistics here

    return abunds, temps