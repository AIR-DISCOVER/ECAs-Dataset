from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import PromptTemplate
from client.modules.memory import GenerativeAgentMemory
from pydantic import BaseModel, Field

from client.modules.information_collector import InformationCollector
from client.modules.reflector import Reflector
from client.modules.llm_set import GlobalConfig

import sqlite3


class GenerativeAgent(BaseModel):
    """Agent as a character with memory and innate characteristics."""
    folder_name: str
    retrieve_name: str
    en_name: str
    name: str
    """The character's name."""
    age: Optional[int] = None
    """The optional age of the character."""
    traits: str = "N/A"
    """Permanent traits to ascribe to the character."""
    status: str
    base_status: str
    social_summary: str
    diagnose_information: str
    ds_risk: list
    """The traits of the character you wish not to change."""
    memory: GenerativeAgentMemory
    """The memory object that combines relevance, recency, and 'importance'."""
    llm: BaseLanguageModel
    """The underlying language model."""
    verbose: bool = False
    summary: str = ""  #: :meta private:
    """Stateful self-summary generated via reflection on the character's memory."""
    summary_refresh_seconds: int = 3600  #: :meta private:
    """How frequently to re-generate the summary."""
    last_refreshed: datetime = Field(default_factory=datetime.now)  # : :meta private:
    """The last time the character's summary was regenerated."""
    daily_summaries: List[str] = Field(default_factory=list)  # : :meta private:
    """Summary of the events in the plan that the agent took."""

    collector: InformationCollector
    reflector: Reflector
    next_action: str = ""
    client_next_action: str = ""

    class Config:
        """Configuration for this pydantic object."""
        arbitrary_types_allowed = True
    

    def generate_client_response_with_new_memory(
        self, observation: str, consult_turns, counselor_name, chat_history, if_need_constraint, now: Optional[datetime] = None
    ) -> str:
        
        # Retrieve process
        retrieve_memories = []
        retrieve_result = self.reflector.reflect_retrieve_context_kewords(observation, self.status, self.social_summary, chat_history)

        if retrieve_result["result"] == "YES":
            retrieve_keywords = retrieve_result["keywords"]

            # The database to be searched is surface_memory (determine the fields to be searched)
            if len(self.folder_name) > 0:
                self.retrieve_name = self.folder_name
            else:
                self.retrieve_name = self.en_name
            if retrieve_result["memory_type"] == "fact_memory":
                retrieve_memories = self.collector.select_surface_memory_by_fact_memory_keywords(self.retrieve_name, retrieve_keywords)
            elif retrieve_result["memory_type"] == "embodied_memory":
                retrieve_memories = self.collector.select_surface_memory_by_embodied_memory_keywords(self.retrieve_name, retrieve_keywords)
            else:
                retrieve_memories = self.collector.select_surface_memory_by_auto_thought_keywords(self.retrieve_name, retrieve_keywords)

            if len(retrieve_memories) > 0:
                # Vector matching context
                conversation_context_list = self.collector.chat_history()
                conversation_context = ""
                for entry in conversation_context_list:
                    for key, value in entry.items():
                        conversation_context += f"{key}: {value}\n"

                if len(retrieve_memories) > 3:
                    if self.ds_risk[0] == 0 and self.ds_risk[1] == 0:
                        retrieve_memories = [self.memory.surface_memory_dict_to_document(key, data, index, retrieve_result["memory_type"]) for index, (key, data) in enumerate(list(retrieve_memories.items()))]
                        retrieve_memories = retrieve_memories[-3:]
                    else:
                        self.memory.add_retrieve_memories(retrieve_memories, retrieve_result["memory_type"])
                        retrieve_memories = self.memory.fetch_retrieve_memories(conversation_context)  # Document type
                else:
                    retrieve_memories = [self.memory.surface_memory_dict_to_document(key, data, index, retrieve_result["memory_type"]) for index, (key, data) in enumerate(list(retrieve_memories.items()))]
        
        # reflecct current goal
        current_goal = self.reflector.reflect_current_goals(self.status)
        core_beliefs = self.collector.extract_core_belief(self.retrieve_name)
        intermediate_beliefs = []
        for retrieve_memorie in retrieve_memories:
            stage_key = retrieve_memorie.metadata["stage_key"]
            intermediate_belief_id = retrieve_memorie.metadata["intermediate_belief_id"]
            intermediate_belief = self.collector.extract_intermediate_belief(self.retrieve_name, stage_key, intermediate_belief_id)
            intermediate_beliefs.append(intermediate_belief)
        
        response = self.reflector.reflect_response_with_new_memory_and_current_goals(observation, self.status, self.social_summary, consult_turns, counselor_name, retrieve_memories, chat_history, current_goal, core_beliefs, intermediate_beliefs, retrieve_result, self.diagnose_information, self.ds_risk, self.en_name, now=now)
        return response


    ######################################################
    # generate response methods                          #
    ###################################################### 
    def generate_client_dialogue_response_for_evaluation(
        self, observation: str, consult_turns: int, loop_reason: str, counselor_name: str, action:str, now: Optional[datetime] = None
    ) -> Tuple[bool, str]:
        """React to a given observation.""" 
        # connect memory dataset
        self.collector.conn = sqlite3.connect(GlobalConfig.memory_database_path)
        self.collector.cursor = self.collector.conn.cursor()

        if action == "yes":
            self.collector.clean_session_chat_history_offline()

        if action == "single":
            chat_history = []
        else:
            chat_history = self.collector.read_session_chat_history_offline()
        self.client_next_action = action

        # new memory
        response = self.generate_client_response_with_new_memory(observation, consult_turns, counselor_name, chat_history, True, now) 
        
        self.collector.user_message_write_offline(observation)
        self.collector.agent_message_write_offline(response)
        
        # close dataset connection
        self.collector.conn.close()
        return f"{response}"