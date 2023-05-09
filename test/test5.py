from typing import List
def raw_sentence(t:str) -> List[str]:
    import re       
    pattern = r'[。；：?！.:;?!]'  # regular expression pattern to match the punctuation marks
    # split the string based on the punctuation marks
    segments = [s for s in re.split(pattern, t) if len(s.strip()) > 0]
    return segments

def not_toolong(t:str) -> List[str]: 
    # split the string by white space 
    # check the length of every segment which is greater than 20, get the count
    # use the count to divide the length of the list
    l = t.split(" ")
    element_num_15 = len([s for s in l if len(s.strip()) < 15])
    
    is_chinese = True

    if element_num_15/len(l) > 0.9:
        is_chinese = False
    
    if(is_chinese and len(t) > 30):
        import re           
        pattern = r'[。；：，?！,.:;?!]'         
        segments = [s for s in re.split(pattern, t) if len(s.strip()) > 0]
        return segments
    elif(not is_chinese and len(l) > 30):
        import re           
        pattern = r'[。；：，?！,.:;?!]'         
        segments = [s for s in re.split(pattern, t) if len(s.strip()) > 0]
        return segments    
    else:
        return [t]  

s1 = '''
  好,请问请你，么说怎么，说怎么说怎么说?
'''

s2 = '''
Hello, and thank you for inviting me to dinner. I am威廉二世界， a foreign teacher at威廉学院， and I am happy to help you with your English learning. I can provide you with a study plan and help you with any confusion you may have. Please feel free to ask me any questions.
'''
segments = []
for s in raw_sentence(s1):
    print(s)
    ss = not_toolong(s)    
    for sss in ss:
        if len(sss.strip()) > 0:
            segments.append(sss)
print("sentences to:",segments)