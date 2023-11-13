from . import code_utils
import json

def should_generate_code_to_response(data_analysis,prompt:str):    
    preview_csv = data_analysis.file_preview.to_csv(index=False)        
    v = data_analysis.llm.chat(None,request=f'''I have a file the path is /home/byzerllm/projects/jupyter-workspace/test.csv, 
The preview of the file is:
```text
{preview_csv}
```
you should try to check the following quesion is whether need to generate python code to answer.

The question is:
                                            
```text
{prompt}
```

if you need to generate python code to answer, please output the following json format:

```json
{{"need_code":true}}
```

otherwise, output the following json format:

```json 
{{"need_code":false}}
```
''')[0].output
    need_code = True
    responses = code_utils.extract_code(v)
    for lang,code in responses:
        if lang == "json":
            try:
                need_code = json.loads(code)["need_code"]
            except Exception as inst:
                pass 
    return need_code        