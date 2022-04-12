import pandas as pd
import numpy as np
from sklearn.metrics import log_loss

def main():
    sub = pd.read_csv("output/submission_mean.csv")
    labels = pd.read_csv("input/val_labels.csv")

    val = labels[['sample_id']].merge(sub, how='inner')
    targetcols = val.columns[1:]

    scores = []
    for target in targetcols:
        score = log_loss(labels[target], val[target], eps=1e-7)
        scores.append(score)

    print(f"Submission Score: {np.mean(scores)}") #Submission Score:  0.12214444100983654

if __name__ == "__main__":
    main()