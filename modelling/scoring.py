"""
Scoring function to determine strenght of models tested.

David O'Connell
do363@student.aru.ac.uk
"""

from sklearn.metrics import f1_score, matthews_corrcoef, precision_score, recall_score


def scoring(model, x_test, y_test) -> tuple:
    ps = precision_score(y_test, model.predict(x_test))
    rs = recall_score(y_test, model.predict(x_test))
    fs = f1_score(y_test, model.predict(x_test))
    mcc = matthews_corrcoef(y_test, model.predict(x_test))
    return ps, rs, fs, mcc
