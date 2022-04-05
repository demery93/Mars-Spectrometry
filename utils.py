import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'

#create dataset of size batch, time, m/h, temp, abundance+
ionlist = [i for i in range(120)] #max number of ions for all models is 120
fill = pd.DataFrame({'m/z':ionlist, 'temp':np.zeros(len(ionlist)), 'timestep':np.ones(len(ionlist))})
fill['m/z'] = fill['m/z'].astype(int)

def preprocess_sample(sample):
    sample['temp'] = sample.temp.ffill().bfill()
    sample['time'] = np.round(sample['time']).astype(int)

    sample['abundance'] = sample.abundance.ffill().bfill()

    sample['m/z'] = np.round(sample['m/z']).astype(int)
    sample = sample[sample["m/z"] < 120] #Max number of ions per model is 120
    sample = sample[sample["m/z"] != 4] #Helium is our carrier gas for all samples used
    sample["abundance_scaled"] = sample.groupby(["m/z"])["abundance"].transform(lambda x: (x - x.min())) / sample["abundance"].max()

    ionfill = fill[~fill['m/z'].isin(sample['m/z'])]
    sample_id = sample['sample_id'].values[0]
    ionfill['sample_id'] = sample_id
    sample = pd.concat([sample, ionfill], axis=0).reset_index(drop=True).fillna(0)

    res1 = sample.groupby(['sample_id','m/z','time'], as_index=False)['abundance_scaled'].mean()
    res2 = sample.groupby(['sample_id', 'time'], as_index=False)['temp'].mean()

    return res1, res2