from sentence_transformers import SentenceTransformer

def init_model(model_dir,infer_params):        
    model = SentenceTransformer(model_dir)     
    return (None,model)


