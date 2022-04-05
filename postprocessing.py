import pandas as pd
import numpy as np

from config import config

def average_predictions():
    subs = []
    for timestep in config.timesteps:
        for nion in config.nions:
            sub = pd.read_csv(f"output/submission_{timestep}_{nion}.csv")
            cols = [c for c in sub.columns[1:]]
            subs.append(sub[cols].values)

    out = sub.copy()
    out[cols] = np.mean(np.dstack(subs),axis=-1)
    out.to_csv("output/submission_mean.csv", index=False, header=True)

def median_predictions():
    subs = []
    for timestep in config.timesteps:
        for nion in config.nions:
            sub = pd.read_csv(f"output/submission_{timestep}_{nion}.csv")
            cols = [c for c in sub.columns[1:]]
            subs.append(sub[cols].values)

    out = sub.copy()
    out[cols] = np.median(np.dstack(subs),axis=-1)
    out.to_csv("output/submission_median.csv", index=False, header=True)

def postprocess(abunds, temps):
    targets = pd.read_csv(config.val_label_path)
    abunds[abunds.sample_id.isin(targets.sample_id.unique())].to_csv("output/abundance.csv", index=False, header=True)
    temps[temps.sample_id.isin(targets.sample_id.unique())].to_csv("output/temperature.csv", index=False, header=True)

