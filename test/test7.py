from pyjava.api.mlsql import RayContext,PythonContext
import uuid
import ray
from byzerllm.apps.qa import ByzerLLMQA,ByzerLLMClient,ClientParams

context:PythonContext = context
conf = context.conf
ray_context = RayContext.connect(globals(),conf["rayAddress"])

# ray_context.collect = mock_collect

name = "qa"
db_dir = "/tmp/byzer-qa/{}".format(str(uuid.uuid4()))

RayByzerLLMQA.options(name=name, lifetime="detached").remote(db_dir,ByzerLLMClient(params=ClientParams(owner=conf["owner"])))
qa = ray.get_actor(name)
qa.query.remote()





