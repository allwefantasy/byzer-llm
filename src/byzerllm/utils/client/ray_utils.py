from ray.util.placement_group import (    
    remove_placement_group,
    PlacementGroup
)
from ray._private.utils import hex_to_binary
from ray._raylet import PlacementGroupID

def cancel_placement_group(group_id:str):
    remove_placement_group(PlacementGroup(
                PlacementGroupID(hex_to_binary(group_id))
            ))