
import io
import os
import uuid
import tempfile

def process_pdf(binnary_data):
    from langchain.document_loaders import PyPDFLoader
    docs = []
    with tempfile.TemporaryDirectory()  as temp_dir:
        pdf_content = io.BytesIO(binnary_data)            
        temp_pdf = os.path.join(temp_dir.name, f"tmp{str(uuid.uuid4())}")

        with open(temp_pdf, 'wb') as f:
            f.write(pdf_content.read())
        
        loader = PyPDFLoader(temp_pdf)
        docs = loader.load()
    
    content = "\n".join([doc.page_content for doc in docs])
    return content