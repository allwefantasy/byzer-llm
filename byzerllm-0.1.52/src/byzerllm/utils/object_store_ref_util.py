import ray
from ray._private.client_mode_hook import client_mode_wrap

@client_mode_wrap
def get_locations(blocks):
    core_worker = ray.worker.global_worker.core_worker
    return [
        core_worker.get_owner_address(block)
        for block in blocks
    ]

def get_object_ids(blocks):
    object_ids = [block.binary() for block in blocks]
    return object_ids