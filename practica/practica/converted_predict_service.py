# Toma la información de un cliente en formato diccionario, la # instancia DictVectorizer y
# la instancia del modelo entrenado
def predict_single(customer, dv, model):
    x = dv.transform([customer])
    y_pred = model.predict_proba(x)[:,1]
    return (y_pred[0] >= 0.5, y_pred[0])
    