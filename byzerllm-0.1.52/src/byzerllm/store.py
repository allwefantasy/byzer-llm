import asyncio
import ray
import time
import os
import tarfile
import tempfile
import uuid
from .utils import print_flush

async def producer(items,udf_name, queue):    
    for item in items:        
        await queue.put(ray.get(item))
    await queue.put(None)

async def worker(queue,udf_name,tf_path,total_count):
    count = 0
    with open(tf_path, "wb") as tf:
        while True:
            item = await queue.get()
            if item is None:
                # Signal to exit when queue is empty
                break
            tf.write(item["value"]) 
            if count % 1000 == 0:
                print_flush(f"MODEL[{udf_name}] UDFWorker pull model: {float(count)/total_count*100}%")
            count += 1  

async def _transfer_from_ob(udf_name, model_refs,target_dir):
    queue = asyncio.Queue(1000)
    
    tf_path = os.path.join(tempfile.gettempdir(), str(uuid.uuid4()))      

    worker_task = asyncio.create_task(worker(queue,udf_name,tf_path,len(model_refs)))
    producer_task = asyncio.create_task(producer(model_refs, udf_name,queue))    
    
    await asyncio.gather(producer_task,worker_task)
    
    with open(tf_path, "rb") as tf:
        tt = tarfile.open(tf.name, mode="r:")
        tt.extractall(target_dir)
        tt.close()
    os.remove(tf_path)

def block_transfer_from_ob(udf_name, model_refs,target_dir):
    tf_path = os.path.join(tempfile.gettempdir(), str(uuid.uuid4()))  
    count = 0
    total_count = len(model_refs)
    with open(tf_path, "wb") as tf:
        for item_ref in model_refs:
            item = ray.get(item_ref)            
            tf.write(item["value"]) 
            if count % 1000 == 0:
                print_flush(f"MODEL[{udf_name}] UDFWorker pull model: {float(count)/total_count*100}%")
            count += 1  
    with open(tf_path, "rb") as tf:
        tt = tarfile.open(tf.name, mode="r:")
        tt.extractall(target_dir)
        tt.close()
    os.remove(tf_path)        


def transfer_from_ob(udf_name,model_refs,model_dir):
    print_flush(f"[{udf_name}] model_refs:{len(model_refs)} model_dir:{model_dir}")
    time1 = time.time()
    # loop = asyncio.get_event_loop()
    # loop.run_until_complete(_transfer_from_ob(udf_name,model_refs,model_dir))            
    block_transfer_from_ob(udf_name,model_refs,model_dir)
    print_flush(f"[{udf_name}] UDFWorker pull model from object store cost {time.time() - time1} seconds")
