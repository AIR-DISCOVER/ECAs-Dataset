import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional
import json

from langchain.chains import LLMChain
from client.modules.time_weighted_retriever import TimeWeightedVectorStoreRetriever
from langchain.schema import BaseMemory, Document
from langchain.utils import mock_now
from langchain_core.language_models import BaseLanguageModel

logger = logging.getLogger(__name__)


class GenerativeAgentMemory(BaseMemory):
    """Memory for the generative agent."""

    llm: BaseLanguageModel
    """The core language model."""
    """The retriever to fetch related memories."""
    vector_retriever: TimeWeightedVectorStoreRetriever
    """The vector retriever to fetch top-k memories."""
    verbose: bool = False
    reflection_threshold: Optional[float] = None
    """When aggregate_importance exceeds reflection_threshold, stop to reflect."""
    current_plan: List[str] = []
    """The current plan of the agent."""
    # A weight of 0.15 makes this less important than it
    # would be otherwise, relative to salience and time
    importance_weight: float = 0.15
    """How much weight to assign the memory importance."""
    aggregate_importance: float = 0.0  # : :meta private:
    """Track the sum of the 'importance' of recent memories.

    Triggers reflection when it reaches reflection_threshold."""

    max_tokens_limit: int = 1200  # : :meta private:
    # input keys
    queries_key: str = "queries"
    most_recent_memories_token_key: str = "recent_memories_token"
    add_memory_key: str = "add_memory"
    # output keys
    relevant_memories_key: str = "relevant_memories"
    relevant_memories_simple_key: str = "relevant_memories_simple"
    most_recent_memories_key: str = "most_recent_memories"
    now_key: str = "now"
    reflecting: bool = False

    # ***memory output path
    memory_path: str = ""
    embodied_memory_path: str = ""
    

    def surface_memory_dict_to_document(self, key, data, index, memory_type):
        ori_metadata = data['metadata']
        wri_metadata = {}
        
        if memory_type == "embodied_memory":
            page_content = ori_metadata['emotion'] + ori_metadata['behavior'] + ori_metadata['physiological_response']
            wri_metadata['temp_memory'] = data['page_content']
        elif memory_type == "auto_thoughts":
            page_content = ori_metadata['auto_thought']
            wri_metadata['temp_memory'] = data['page_content']
        else:
            page_content = data['page_content']
        
        wri_metadata['node_id'] = key
        wri_metadata['node_type'] = ori_metadata['node_type']
        wri_metadata['stage_key'] = ori_metadata['stage_key']
        wri_metadata['core_belief_id'] = ori_metadata['core_belief_id']
        wri_metadata['intermediate_belief_id'] = ori_metadata['intermediate_belief_id']
        wri_metadata['auto_thought'] = ori_metadata['auto_thought']
        wri_metadata['emotion'] = ori_metadata['emotion']
        wri_metadata['behavior'] = ori_metadata['behavior']
        wri_metadata['physiological_response'] = ori_metadata['physiological_response']
        wri_metadata['created_at'] = datetime.fromisoformat(ori_metadata['created_at'].replace(" ", "T"))
        wri_metadata['buffer_idx'] = index
        return Document(page_content=page_content, metadata=wri_metadata)

    # ***document format
    def dict_to_document(self, key, data, index):
        ori_metadata = data['metadata']
        wri_metadata = {}
        wri_metadata['node_id'] = key
        wri_metadata['node_type'] = ori_metadata['node_type']
        wri_metadata['importance'] = ori_metadata['importance']
        wri_metadata['last_accessed_at'] = datetime.fromisoformat(ori_metadata['last_accessed_at'])
        wri_metadata['created_at'] = datetime.fromisoformat(ori_metadata['created_at'])
        wri_metadata['buffer_idx'] = index
        return Document(page_content=data['page_content'], metadata=wri_metadata)
    def embodied_dict_to_document(self, key, data, index):
        ori_metadata = data['metadata']
        wri_metadata = {}
        wri_metadata['node_id'] = key
        wri_metadata['ori_node'] = ori_metadata['ori_node']
        wri_metadata['node_type'] = ori_metadata['node_type']
        wri_metadata['importance'] = ori_metadata['importance']
        wri_metadata['last_accessed_at'] = datetime.fromisoformat(ori_metadata['last_accessed_at'])
        wri_metadata['created_at'] = datetime.fromisoformat(ori_metadata['created_at'])
        wri_metadata['buffer_idx'] = index
        return Document(page_content=data['page_content'], metadata=wri_metadata)
    
    # init retrieve memory, use vector_retriever
    def add_retrieve_memories(self, retrieve_memory_content: dict, memory_type: str)->str:
        self.vector_retriever.memory_stream.clear()
        if memory_type == "fact":
            documents = [self.dict_to_document(key, data, index) for index, (key, data) in enumerate(reversed(list(retrieve_memory_content.items())))]
        elif memory_type == "embodied":
            documents = [self.embodied_dict_to_document(key, data, index) for index, (key, data) in enumerate(reversed(list(retrieve_memory_content.items())))]
        else:
            documents = [self.surface_memory_dict_to_document(key, data, index, memory_type) for index, (key, data) in enumerate(reversed(list(retrieve_memory_content.items())))]
        for doc in documents:
            self.vector_retriever.memory_stream.extend([doc])
    
    
    def fetch_retrieve_memories(
        self, observation: str, now: Optional[datetime] = None
    ) -> List[Document]:
        """Fetch related memories."""
        if now is not None:
            with mock_now(now):
                return self.vector_retriever.invoke(observation)
        else:
            return self.vector_retriever.invoke(observation)
        

    def clear(self):
        """Clear the memory."""
        logger.info("Clearing memory.")
        self.vector_retriever.memory_stream.clear()

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Load memory-related variables."""
        logger.info("Loading memory variables.")
        return {
            self.most_recent_memories_key: self.vector_retriever.memory_stream
        }

    def memory_variables(self) -> Dict[str, Any]:
        """Return relevant memory variables."""
        logger.info("Fetching memory variables.")
        return {
            self.most_recent_memories_key: self.vector_retriever.memory_stream
        }

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None:
        """Save the context, including new observations or reflections."""
        logger.info("Saving context.")
        new_memory = inputs.get(self.add_memory_key, "")
        if new_memory:
            logger.debug(f"Adding new memory: {new_memory}")
            self.vector_retriever.memory_stream.append(Document(page_content=new_memory))