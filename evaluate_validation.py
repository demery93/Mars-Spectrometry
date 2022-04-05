import pandas as pd
import numpy as np
from sklearn.metrics import log_loss

sub = pd.read_csv("output/submission_mean.csv")
labels = pd.read_csv("input/val_labels.csv")

val = labels[['sample_id']].merge(sub, how='inner')
targetcols = val.columns[1:]

scores = []
for target in targetcols:
    score = log_loss(labels[target], val[target], eps=1e-7)
    scores.append(score)

print(np.mean(scores)) #0.12305385221858704