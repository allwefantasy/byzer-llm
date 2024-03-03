from ray.util.placement_group import (    
    remove_placement_group,
    PlacementGroup
)
from ray._private.utils import hex_to_binary
from ray._raylet import PlacementGroupID
from ray.util.state import (
        StateApiClient,
        get_log,
        list_logs,
        summarize_actors,
        summarize_objects,
        summarize_tasks,
    )
from ray.util.state.common import (
    DEFAULT_LIMIT,
    DEFAULT_LOG_LIMIT,
    DEFAULT_RPC_TIMEOUT,
    GetApiOptions,
    ListApiOptions,
    PredicateType,
    StateResource,
    StateSchema,
    SupportedFilterType,
    resource_to_schema,
)
from ray.util.state.exception import RayStateApiException
from ray.util.annotations import PublicAPI

def cancel_placement_group(group_id:str):
    remove_placement_group(PlacementGroup(
                PlacementGroupID(hex_to_binary(group_id))
            ))

def get_actor_info(actor):            
    resource = StateResource("actors".replace("-", "_"))
    # Create the State API server and put it into context
    client = StateApiClient(address="auto")
    options = GetApiOptions(timeout=30)
    # If errors occur, exceptions will be thrown. Empty data indicate successful query.
    try:
        state = client.get(
            resource,
            options=options,
            id=actor._ray_actor_id.hex(),
            _explain=True,
        )
        return state
    except RayStateApiException as e:
        raise e
    

    
  
    