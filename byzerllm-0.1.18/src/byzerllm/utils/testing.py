import pyarrow as pa
import json
from pyjava.udf import UDFBuilder
from pyjava import PythonContext, RayContext
import uuid

def build_model_input(data):
    value = pa.array([json.dumps(data,ensure_ascii=False)])
    names = ["value"]
    batch = pa.record_batch([value], names=names)
    return batch

async def async_call_model(udf_name, model_input,**kwargs):    
    context = PythonContext(
        context_id= str(uuid.uuid4()),     
        iterator=[model_input],
        conf={
            "UDF_CLIENT":udf_name,
            "pythonMode":"ray",
            "directData": "true"
        }
    )
    ray_address = kwargs.get("rayAddress","worker")
    ray_context = RayContext.connect(context,ray_address)
    await UDFBuilder.async_apply(ray_context)
    return context.output_data[0].to_pylist()


def call_model(udf_name,model_input,**kwargs):
    context = PythonContext(
        context_id= str(uuid.uuid4()),     
        iterator=[model_input],
        conf={
            "UDF_CLIENT":udf_name,
            "pythonMode":"ray",
            "directData": "true"
        }
    )
    ray_address = kwargs.get("rayAddress","worker")
    ray_context = RayContext.connect(context,ray_address)
    UDFBuilder.block_apply(ray_context)
    return context.output_data[0].to_pylist()

