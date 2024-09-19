import os
import json
import sqlite3

from client.modules.llm_set import GlobalConfig


class InformationCollector:
    def __init__(self):
        self.whether_to_use_DB = False
        # 设定文件夹的相对路径
        self.client_data_directory = os.path.join(GlobalConfig.client_storage_path, GlobalConfig.client_name)
        self.client_consulting_data_directory = os.path.join(GlobalConfig.client_storage_path, GlobalConfig.client_name, "consulting")
        self.client_character_directory = os.path.join(GlobalConfig.client_character_path, GlobalConfig.client_name + '.json')
        # self.counselor_data_directory = os.path.join(GlobalConfig.counselor_storage_path, GlobalConfig.counselor_name)
        # 初始化sql对象
        self.conn = sqlite3.connect(GlobalConfig.memory_database_path)
        self.cursor = self.conn.cursor()


    def surface_memory_list_to_dict(self, rows):
        memories_dict = {}
        for row in rows:
            node_id = row[3]
            memories_dict[node_id] = {
                "page_content": row[2],
                "metadata": {
                    'node_id': node_id,
                    "node_type": row[4],
                    "event_id": row[5],
                    "extended_event_id": row[6],
                    "stage_key": row[7],
                    "core_belief_id": row[8],
                    "intermediate_belief_id": row[9],
                    "auto_thought": row[10],
                    "emotion": row[11],
                    'behavior': row[12],
                    'physiological_response': row[13],
                    'created_at': row[14]
                }
            }
        return memories_dict

    def select_surface_memory_by_fact_memory_keywords(self, client_en_name, retrieve_keywords):
        retrieve_fact_memories = []
        for keyword in retrieve_keywords:
            self.cursor.execute(f'SELECT * FROM surfaceMemory WHERE name = "{client_en_name}" AND page_content LIKE "%{keyword}%"')
            retrieve_fact_memories.extend(self.cursor.fetchall())
        return self.surface_memory_list_to_dict(retrieve_fact_memories)
    
    def select_surface_memory_by_embodied_memory_keywords(self, client_en_name, retrieve_keywords):
        retrieve_embodied_memories = []
        for keyword in retrieve_keywords:
            self.cursor.execute(f'SELECT * FROM surfaceMemory WHERE name = "{client_en_name}" AND (emotion LIKE "%{keyword}%" OR behavior LIKE "%{keyword}%" OR physiological_response LIKE "%{keyword}%")')
            retrieve_embodied_memories.extend(self.cursor.fetchall())
        return self.surface_memory_list_to_dict(retrieve_embodied_memories)
    
    def select_surface_memory_by_auto_thought_keywords(self, client_en_name, retrieve_keywords):
        retrieve_auto_thoughts = []
        for keyword in retrieve_keywords:
            self.cursor.execute(f'SELECT * FROM surfaceMemory WHERE name = "{client_en_name}" AND auto_thought LIKE "%{keyword}%"')
            retrieve_auto_thoughts.extend(self.cursor.fetchall())
        return self.surface_memory_list_to_dict(retrieve_auto_thoughts)
    
    def extract_core_belief(self, name):
        self.cursor.execute('''
        SELECT page_content 
        FROM coreBelief 
        WHERE name = ? AND stage_key = ?
        ''', (name, "recent_events"))
        results = self.cursor.fetchall()
        core_beliefs = [row[0] for row in results]
        return core_beliefs
    
    def extract_intermediate_belief(self, name, stage_key, belief_id):
        self.cursor.execute('''
        SELECT page_content 
        FROM intermediateBelief 
        WHERE name = ? AND stage_key = ? AND belief_id = ?
        ''', (name, stage_key, belief_id))

        intermediate_beliefs_dict = []
        intermediate_belief_dict = {}
        results = self.cursor.fetchall()
        intermediate_beliefs = [row[0] for row in results]
        for intermediate_belief in intermediate_beliefs:
            intermediate_belief = json.loads(intermediate_belief.replace("'", '"'))
            intermediate_belief_dict["态度"] = intermediate_belief['attitude']['self'] + intermediate_belief['attitude']['others'] + intermediate_belief['attitude']['world']
            intermediate_belief_dict["规则"] = intermediate_belief['rules']
            intermediate_belief_dict["积极假设"] = intermediate_belief['positive_assumption']
            intermediate_belief_dict["消极假设"] = intermediate_belief['negative_assumption']
            intermediate_beliefs_dict.append(intermediate_belief_dict)
            intermediate_belief_dict = {}
        
        return intermediate_beliefs_dict
    
    def read_txt_file(self, file_name):
        # 构建文件完整路径
        file_path = os.path.join(self.client_consulting_data_directory, file_name)
        # 打印尝试访问的文件路径（调试用）
        # print(f"Attempting to read from: {file_path}")
        try:
            if not os.path.exists(file_path):
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write('')
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                # print(f"File: {file_name}")
                # print("Content:")
                # print(content)
                return content
        except FileNotFoundError:
            print(f"Error: File '{file_name}' not found in directory '{self.client_consulting_data_directory}'.")
        except Exception as e:
            print(f"An error occurred: {e}")

    def append_txt_file(self, file_name, content):
        # 构建文件完整路径
        file_path = os.path.join(self.client_consulting_data_directory, file_name)
        # 打印尝试访问的文件路径（调试用）
        # print(f"Attempting to append to: {file_path}")
        try:
            with open(file_path, 'a', encoding='utf-8') as file:
                # content加入换行符
                content += "\n"
                file.write(content)
                print(f"Content appended to file '{file_name}'.")
        except Exception as e:
            print(f"An error occurred: {e}")

    def delect_txt_content(self, file_name):
        # 构建文件完整路径
        file_path = os.path.join(self.client_consulting_data_directory, file_name)
        # 打印尝试访问的文件路径（调试用）
        print(f"Attempting to delect content from: {file_path}")
        try:
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write("")
                print(f"Content delected from file: {file_name}")
        except Exception as e:
            print(f"An error occurred: {e}")

    def chat_history(self):
        return self.read_txt_file("chat_history.txt")
    
    def read_session_chat_history_offline(self):
        return self.read_txt_file("session_chat_history.txt")
    
    def clean_session_chat_history_offline(self):
        return self.delect_txt_content("session_chat_history.txt")
    
    def user_message_write_offline(self, content):
        content = "counselor:" + content
        self.append_txt_file("session_chat_history.txt", content)
        
    def agent_message_write_offline(self, content):
        content = "client:" + content
        self.append_txt_file("session_chat_history.txt", content)

    def write_memory_retrieve_history(self, consult_turns, fact_memories, embodied_memories):
        content = f"---------------------{consult_turns}-----------------------\n"
        content = content + str(fact_memories) + '\n' + str(embodied_memories)
        self.append_txt_file("memory_retrieve_history.txt", content)

    def write_response_prompt(self, consult_turns, user_message, prompt):
        content = f"---------------------{consult_turns}-----------------------\n"
        content = content + str(user_message) + '\n' + str(prompt)
        self.append_txt_file("response_prompt.txt", content)


collector = InformationCollector()