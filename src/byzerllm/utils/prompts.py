from langchain import PromptTemplate

PROMPT_ANALYZE_TEXT='''
We have the following json format data:

```json
{content}
```

Try to answer quession according the json format data we provided above.
the question is:

{prompt}
'''

PROMPT_SUMARIOZATION ='''                
please try to summarize the following text:

```
{answer_chunk}
{chunk}
```

Finally, please try to match the following requirements:

```
{prompt}
```
''' 

PROMPT_IS_SUMARY = '''
Please check the following question is whether have the same meaning with "summarize this article"

```
{prompt}                               
```                               

If the question have the same meaning with "summarize this article" , please output the following json format:

```json
{{"is_summary":true}}
```

otherwise, output the following json format:

```json 
{{"is_summary":false}}
```
'''

PROMPT_IS_VISUALIZATION = '''
Please check the following question is whether about data visualization:

```text
{prompt}                               
```   

If the question is about data visualization, please output the following json format:

```json
{{"is_visualization":true}}
```

otherwise, output the following json format:

```json 
{{"is_visualization":false}}
```
'''

PROMPT_SHOULD_GENERATE_CODE_TO_RESPONSE = '''I have a file the path is {file_path}, 
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
'''

PROMPT_PREVIEW_FILE='''I have a file where the path is {file_path}, I want to use pandas to read it.The packages all are installed, you can use it directly.
Try to help me to generate python code which should match the following requirements:
1. try to read the file according the suffix of file name in Try block
2. if read success, set variable loaded_successfully to True, otherwise set it to False.
3. if loaded_successfully is True, then assigh the loaded data with head() to file_preview, otherwise assign error message to file_preview
4. make sure the loaded_successfully, file_preview are defined in the global scope
'''

PROMPT_NO_NEED_CODE='''I have a file the path is {file_path}, 
The preview of the file is:

```text
{preview_csv}
```

Please try to answer the following questions:
{prompt}'''

PROMPT_VISUALIZATION = '''When the question require you to do visualization, please use package Plotly or matplotlib to do this.
Try to use base64 to encode the image, assign the base64 string to the variable named image_base64. 
Make sure the image_base64 defined in the global scope. Notice that try to create figure with `plt.figure()` before you plot the image.

Here is the example code how to save the plot to a BytesIO object and encode the image to base64:

```python
# Save the plot to a BytesIO object
buf = io.BytesIO()
plt.savefig(buf, format='png')
buf.seek(0)

# Encode the image to base64
image_base64 = base64.b64encode(buf.read()).decode('utf-8')
buf.close()
```
'''

PROMPT_ANALYSIS_DATA_WITH_VISUALIZATION='''I have a file the path is {file_path}, 
The preview of the file is:
```text
{preview_csv}
```
Use pandas to analyze it. 
Please DO NOT consider the package installation, the packages all are installed, you can use it directly.

{visualization_prompt}

Please try to generate python code to analyze the file and answer the following questions:
'''

def prompt_is_summary(prompt:str):
    prompt_template = PromptTemplate.from_template(PROMPT_IS_SUMARY) 
    return prompt_template.format(prompt=prompt)

def prompt_is_visualization(prompt:str):
    prompt_template = PromptTemplate.from_template(PROMPT_IS_VISUALIZATION) 
    return prompt_template.format(prompt=prompt)

def prompt_should_generate_code_to_response(file_path:str,prompt:str,preview_csv:str):
    prompt_template = PromptTemplate.from_template(PROMPT_SHOULD_GENERATE_CODE_TO_RESPONSE) 
    return prompt_template.format(file_path=file_path,prompt=prompt,preview_csv=preview_csv)

def prompt_preview_file(file_path:str):
    prompt_template = PromptTemplate.from_template(PROMPT_PREVIEW_FILE) 
    return prompt_template.format(file_path=file_path)

def prompt_no_need_code(file_path:str,prompt:str,preview_csv:str):
    prompt_template = PromptTemplate.from_template(PROMPT_NO_NEED_CODE) 
    return prompt_template.format(file_path=file_path,prompt=prompt,preview_csv=preview_csv)

def prompt_analysis_data_with_visualization(file_path:str,visualization_prompt:str,preview_csv:str):
    prompt_template = PromptTemplate.from_template(PROMPT_ANALYSIS_DATA_WITH_VISUALIZATION) 
    return prompt_template.format(file_path=file_path,visualization_prompt=visualization_prompt,preview_csv=preview_csv)

def prompt_sumarization(answer_chunk:str,chunk:str,prompt:str):
    prompt_template = PromptTemplate.from_template(PROMPT_SUMARIOZATION) 
    return prompt_template.format(answer_chunk=answer_chunk,chunk=chunk,prompt=prompt)

def prompt_analyze_text(content:str,prompt:str):
    prompt_template = PromptTemplate.from_template(PROMPT_ANALYZE_TEXT) 
    return prompt_template.format(content=content,prompt=prompt)






