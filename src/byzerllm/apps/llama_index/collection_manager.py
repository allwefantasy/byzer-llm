import json
import os
from typing import List, Dict
from os.path import expanduser

from typing import List, Optional
from pydantic import BaseModel

class CollectionItem(BaseModel):
    name: str
    description: Optional[str] = ""    


class CollectionManager:
   def __init__(self,base_dir:str):
        home = expanduser("~")        
        base_dir = base_dir or os.path.join(home, ".auto-coder")
        self.collections_json_file = os.path.join(base_dir, "storage", "data", "collections.json") 
        self.collections_json_file_dir = os.path.join(base_dir, "storage", "data")
        
        if not os.path.exists(self.collections_json_file_dir):
            os.makedirs(self.collections_json_file_dir)

        self.collections: Dict[str, CollectionItem] = {}
        self.load_collections()

   def load_collections(self):
        if os.path.exists(self.collections_json_file):
            with open(self.collections_json_file, "r") as f:
                collections_data = json.load(f)
                for name, item_data in collections_data.items():
                    self.collections[name] = CollectionItem(**item_data)
        if "default" not in self.collections:
            self.collections["default"] = CollectionItem(name="default", description="Default collection")            
                  

   def save_collections(self):
       collections_data = {name: item.model_dump() for name, item in self.collections.items()}
       with open(self.collections_json_file, "w") as f:
           json.dump(collections_data, f, indent=2,ensure_ascii=False)

   def add_collection(self, collection_item: CollectionItem):
       self.collections[collection_item.name] = collection_item
       self.save_collections()

   def update_collection(self, collection_item: CollectionItem):
       if collection_item.name in self.collections:
           self.collections[collection_item.name] = collection_item
           self.save_collections()
       else:
           raise ValueError(f"Collection '{collection_item.name}' not found.")

   def delete_collection(self, name: str):
       if name in self.collections:
           del self.collections[name]
           self.save_collections()
       else:
           raise ValueError(f"Collection '{name}' not found.")

   def get_collection(self, name: str) -> CollectionItem:
       if name in self.collections:
           return self.collections[name]
       else:
           raise ValueError(f"Collection '{name}' not found.")

   def check_collection_exists(self, name: str) -> bool:
       return name in self.collections   

   def get_all_collections(self) -> List[CollectionItem]:
       return list(self.collections.values())