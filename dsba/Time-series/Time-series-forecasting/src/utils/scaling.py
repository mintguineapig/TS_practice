from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler
import numpy as np

def apply_scaling(scaler, trn, val, tst):
    # scaler : standard, minmax, maxabs, robust, minmax square, minmax m1p1
    square = False  # for minmax square
    
    if scaler == 'standard':
        sclr = StandardScaler()
    elif scaler == 'minmax':
        sclr = MinMaxScaler()
    elif scaler == 'minmax square':
        sclr = MinMaxScaler()
        square = True
    elif scaler == 'minmax m1p1':
        sclr = MinMaxScaler(feature_range=(-1, 1))
    elif scaler == 'maxabs':
        sclr = MaxAbsScaler()
    elif scaler == 'robust':
        sclr = RobustScaler()
    else:
        raise ValueError(f"""Unknown scaler '{scaler}'| 
                         Available: ['standard', 'minmax', 'maxabs', 'robust','minmax square', 'minmax m1p1]""")
     
    # scaling
    trn = sclr.fit_transform(trn)
    val = sclr.transform(val)
    tst = sclr.transform(tst)
    
    # final scaling step for 'minmax square'
    if square:
        trn, val, tst = trn**2, val**2, tst**2
    
    return trn, val, tst