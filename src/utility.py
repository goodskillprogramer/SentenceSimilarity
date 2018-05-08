try:
    import cPickle
except :
    import _pickle as cPickle

def save_model(clf,modelpath): 
    with open(modelpath, 'wb') as f: 
        cPickle.dump(clf, f) 
 
def load_model(modelpath): 
    try: 
        with open(modelpath, 'rb') as f: 
            rf = cPickle.load(f) 
            return rf 
    except Exception as e:        
        return None 
    
def write_to_file(path,txt,mode='ab+'):
    with open(path,mode=mode) as f:
        f.write(txt)