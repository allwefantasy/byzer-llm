from sentence_transformers import SentenceTransformer

def init_model(model_dir,infer_params,sys_conf={}):        
    model = SentenceTransformer(model_dir)     
    return (None,model)


