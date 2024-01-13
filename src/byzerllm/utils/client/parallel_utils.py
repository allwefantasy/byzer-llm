import concurrent.futures

def chat_oai(llm,workers: int=3, **kwargs):
    """
    Invoke llm.chat_oai in multi-threading with specified size
    and return the combined result.
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        # Submit tasks to the executor
        futures = [executor.submit(llm.chat_oai, **kwargs) for _ in range(workers)]

        # Collect results as they are completed
        results = [future.result() for future in concurrent.futures.as_completed(futures)]
    
    return results

def get_single_result(ts):    
    if not hasattr(ts[0][0],"values")  and not hasattr(ts[0][0],"value"):
        for t in ts:
            if t[0].output:
                return t       
        

    if hasattr(ts[0][0],"values"):        
        for t in ts:
            if t[0].values:
                return t 
    
    if hasattr(ts[0][0],"value"):        
        for t in ts:
            if t[0].value:
                return t        
    
    return None