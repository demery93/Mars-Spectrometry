import pandas as pd
import numpy as np

import tensorflow as tf
import gc

from config import config
from model import cnn, scheduler
from sklearn.metrics import log_loss
import logging

def generate_predictions(df1, df2, timesteps, nions, retrain=True, validate=True, model_type='cnn', input_smoothing=4, kernel_width=3):
    targets = pd.read_csv("input/train_labels.csv")
    val_targets = pd.read_csv("input/val_labels.csv")
    metadata = pd.read_csv("input/metadata.csv")

    abunds = df1[[c for c in range(timesteps)]]
    temps = df2[[c for c in range(timesteps)]]

    abunds = abunds[abunds.index.get_level_values(1) < nions]

    targetcols = targets.columns[1:].tolist()

    if(validate):
        x_train = [abunds.reset_index()[abunds.reset_index().sample_id.isin(metadata[metadata.split.isin(['train'])].sample_id)].set_index(['sample_id','m/z']).values.reshape((-1, timesteps, nions)),
                    temps.reset_index()[temps.reset_index().sample_id.isin(metadata[metadata.split.isin(['train'])].sample_id)].set_index(['sample_id']).values.reshape((-1, timesteps, 1))]
    else:
        x_train = [abunds.reset_index()[abunds.reset_index().sample_id.isin(metadata[metadata.split.isin(['train','val'])].sample_id)].set_index(['sample_id','m/z']).values.reshape((-1, timesteps, nions)),
                    temps.reset_index()[temps.reset_index().sample_id.isin(metadata[metadata.split.isin(['train','val'])].sample_id)].set_index(['sample_id']).values.reshape((-1, timesteps, 1))]

    x_val = [abunds.reset_index()[abunds.reset_index().sample_id.isin(metadata[metadata.split == 'val'].sample_id)].set_index(['sample_id', 'm/z']).values.reshape((-1, timesteps, nions)),
        temps.reset_index()[temps.reset_index().sample_id.isin(metadata[metadata.split == 'val'].sample_id)].set_index(['sample_id']).values.reshape((-1, timesteps, 1))]

    x_test = [abunds.reset_index()[abunds.reset_index().sample_id.isin(metadata[metadata.split == 'test'].sample_id)].set_index(['sample_id', 'm/z']).values.reshape((-1, timesteps, nions)),
        temps.reset_index()[temps.reset_index().sample_id.isin(metadata[metadata.split == 'test'].sample_id)].set_index(['sample_id']).values.reshape((-1, timesteps, 1))]


    y_train = targets[targetcols].values
    y_val = val_targets[targetcols].values
    if(validate==False):
        y_train = np.concatenate([y_train, y_val], axis=0)

    callback = [tf.keras.callbacks.LearningRateScheduler(scheduler)]
    val_pred, test_pred = [],[]
    for i in range(config.n_bags):
        print(f"Running model {i}")

        model = cnn(timesteps, nions, kernel_width=3, input_smoothing=6)

        if(retrain):
            model.fit(x_train, y_train,
                      epochs=20, batch_size=16,
                      validation_data=(x_val, y_val),
                      verbose=0, callbacks=callback)
            model.save_weights(f"trained_models/{model_type}_{timesteps}_{nions}_{input_smoothing}_{kernel_width}_{i}.h5")
        model.load_weights(f"trained_models/{model_type}_{timesteps}_{nions}_{input_smoothing}_{kernel_width}_{i}.h5")

        val_pred.append(model.predict(x_val))
        test_pred.append(model.predict(x_test))
        del model
        gc.collect()

    val_pred = np.mean(np.dstack(val_pred), axis=-1)
    test_pred = np.mean(np.dstack(test_pred), axis=-1)

    scores = []
    for i in range(len(targetcols)):
        score = log_loss(y_val[:,i], val_pred[:,i], eps=1e-7)
        scores.append(score)

    val_score = np.round(np.mean(scores),3)
    logging.info(f"Mean Aggregated Logloss for bagged model with {timesteps} steps and {nions} ions: {val_score}")

    sub = pd.read_csv("input/submission_format.csv")
    pred = np.concatenate([val_pred, test_pred], axis=0)
    pred = pd.DataFrame(pred, columns = targetcols)
    pred['sample_id'] = metadata[metadata.split != 'train'].sample_id.values
    pred = pred[sub.columns]
    pred.to_csv(f"output/submission_{timesteps}_{nions}_{input_smoothing}_{kernel_width}.csv", index=False, header=True)
