import pickle

def load_model(filepath):
    model = pickle.load(open(filepath, 'rb'))
    return model

def load_scaler(scaler_file):
    scaler = pickle.load(open(scaler_file, 'rb'))
    return scaler
