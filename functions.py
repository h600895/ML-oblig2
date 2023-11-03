# Predicting with the inserted model

# Train model
def trainModel(model, X, y):
    model.fit(X, y)

def predictWithModel(model, X_test):
    return model.predict(X_test)

def determineScore(y_test, y_pred, measures):

    predsDict = {}
    for key in measures.keys():
        predsDict[key] = measures[key](y_test, y_pred)


    return predsDict

