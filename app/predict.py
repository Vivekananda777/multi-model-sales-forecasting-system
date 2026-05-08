def predict_future(model, future_df):

    preds = model.predict(future_df)

    return preds.tolist()