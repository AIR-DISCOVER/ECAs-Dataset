from client.modules.generative_agent import GenerativeAgent
from client.modules.memory import GenerativeAgentMemory
from client.modules.llm_set import *
import os
import json

from client.modules.information_collector import InformationCollector
from client.modules.reflector import Reflector

class Client:
    client_name: str

    def __init__(self, client_name):
        self.folder_name = ""
        self.client_name = client_name
        self.persona_summary = ""
        self.social_summary = ""
        self.base_status = ""
        self.client_character_path = os.path.join(GlobalConfig.client_character_path, 'full_portrait'+'.json')
        self.client_character_base_path = os.path.join(GlobalConfig.client_character_path, 'portrait'+'.json')
        self.client_character_content = self.read_file(self.client_character_path)
        self.self_portrait = self.client_character_content["self-portrait"]
        self.basic_information = self.self_portrait["basic_information"]
        self.diagnosis = self.self_portrait["diagnosis"]
        self.character = self.self_portrait["character"]
        self.appearance = self.self_portrait["appearance"]
        self.interests_and_hobbies = self.self_portrait["interests_and_hobbies"]
        self.dreams_and_aspirations = self.self_portrait["dreams_and_aspirations"]
        self.daily_life = self.self_portrait["daily_life"]
        self.past_experiences = self.self_portrait["past_experiences"]
        self.social_portrait = self.client_character_content["social-portrait"]
        self.social_connections = self.social_portrait["social_connections_closeness"]

        self.memory_path = os.path.join(GlobalConfig.client_storage_path, self.basic_information["en_name"], 'memory.json')
        # self.questionnaire_prompt_path = GlobalConfig.questionnaire_prompt_path
        self.sim_setup_path = os.path.join(GlobalConfig.client_storage_path, self.basic_information["en_name"], 'sim_set.json')


        # init memory
        self.memory = GenerativeAgentMemory(
            llm=LLM,
            vector_retriever=vector_retriever(),
            verbose=False,
            reflection_threshold=4,  # we will give this a relatively low number to show how reflection works
            memory_path=self.memory_path,
        )

        # init agent
        depressed_risk, suicide_risk = self.judge_ill_degree(self.diagnosis)
        drisk = self.diagnosis["drisk"]
        srisk = self.diagnosis["srisk"]
        ds_risk = [drisk, srisk]
        self.agent = GenerativeAgent(
            retrieve_name="",
            folder_name=self.folder_name,
            en_name=self.basic_information["en_name"],
            name=self.basic_information["name"],
            age=0,
            traits=",".join(self.character),
            status=self.generate_status(),
            base_status=self.get_base_status(),
            social_summary=self.generate_social_status(),
            diagnose_information = depressed_risk+"。"+suicide_risk+"。"+self.diagnosis["symptoms"]+"。"+self.diagnosis["status"],
            ds_risk=ds_risk,
            llm=LLM,
            memory=self.memory,
            collector=InformationCollector(),
            reflector=Reflector()
        )

        # init oringin memory
        self.add_origin_memories(self.memory_path)

        # init sim setup file
        self.add_sim_setup(self.sim_setup_path)

    def get_base_status(self):
        if os.path.exists(self.client_character_base_path):
            with open(self.client_character_base_path, 'r', encoding='utf-8') as f:
                base_status = json.load(f)
                for key in ["id", "name", "en_name"]:
                    base_status["basic_information"].pop(key, None)
            self.base_status = base_status
            # print(f"{self.basic_information['name']}的baseline信息为：{base_status}")
        else:
            base_status = ""
        return str(base_status)

    def generate_status(self):
        vars = {
            "self_portrait": self.self_portrait,
            "name": self.basic_information["name"],
            }

        prompt_template = PromptTemplate.from_template(
            template = """
            # INPUT
            self_portrait(来访者的个人画像): {self_portrait}

            # OUTPUT
            请根据以上来访者{name}的人物画像提炼重点，总结{name}的人设信息，严格要求输出内容的字数在150字以内。
            """,
            )    
        if "persona_summary" not in self.self_portrait:
            self.persona_summary = chain_with_error_deal(prompt_template, vars, False)
            self.self_portrait["persona_summary"] = self.persona_summary
            with open(self.client_character_path, 'w', encoding='utf-8') as f:
                json.dump(self.client_character_content, f, ensure_ascii=False, indent=4)
        else:
            self.persona_summary = self.self_portrait["persona_summary"]
        # print(f"{self.basic_information['name']}的人设信息为：{self.persona_summary}")
        return self.persona_summary
    
    def generate_social_status(self):
        vars = {
            "social_portrait": self.social_portrait,
            "name": self.basic_information["name"],
            }

        prompt_template = PromptTemplate.from_template(
            template = """
            # INPUT
            social_portrait(来访者的社会画像): {social_portrait}

            # OUTPUT
            请根据以上来访者{name}的社会画像提炼重点，总结{name}的社会关系信息，要求包括所有的社会关系人，并严格要求输出内容的字数在150字以内。
            """,
            )    
        if "social_summary" not in self.self_portrait:
            self.social_summary = chain_with_error_deal(prompt_template, vars, False)
            self.self_portrait["social_summary"] = self.social_summary
            with open(self.client_character_path, 'w', encoding='utf-8') as f:
                json.dump(self.client_character_content, f, ensure_ascii=False, indent=4)
        else:
            self.social_summary = self.self_portrait["social_summary"]
        # print(f"{self.basic_information['name']}的社会关系信息为：{self.social_summary}")
        return self.social_summary
    
    def generate_social_connections_description(self, stage_key):
        vars = {
            "name": self.basic_information["name"],
            "stage_key": stage_key,
            "social_connections_closeness": self.social_connections[stage_key],
            "social_connections_change": self.social_portrait["social_connections_change"],
            }

        prompt_template = PromptTemplate.from_template(
            template = """
            # INPUT
            social_connections_closeness(来访者的社会关系密度): {social_connections_closeness}
            social_connections_change(来访者社会关系的变化过程): {social_connections_change}

            # OUTPUT
            请根据以上来访者{name}在{stage_key}阶段的社会关系密度和社会关系的变化过程提炼重点，其中familiarity的程度值为1-5(完全不熟悉-非常熟悉)，总结{name}的社会关系信息，严格要求输出内容的字数在200字以内。
            """,
            )   
        social_connections_description = chain_with_error_deal(prompt_template, vars, False)
        # print(f"{self.basic_information['name']}的社会关系总结为：{social_connections_description}")
        return social_connections_description

    def read_file(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            file_content = f.read().strip()
            content = json.loads(file_content)
        return content
    
    def judge_ill_degree(self, diagnosis):
        if diagnosis["drisk"] == 0:
            depressed_risk = "无抑郁风险"
        elif diagnosis["drisk"] == 1:
            depressed_risk = "轻度抑郁风险"
        elif diagnosis["drisk"] == 2:
            depressed_risk = "中度抑郁风险"
        else:
            depressed_risk = "重度抑郁风险"

        if diagnosis["srisk"] == 0:
            suicide_risk = "无自杀风险"
        elif diagnosis["drisk"] == 1:
            suicide_risk = "轻度自杀风险"
        elif diagnosis["drisk"] == 2:
            suicide_risk = "中度自杀风险"
        else:
            suicide_risk = "重度自杀风险"
        return depressed_risk, suicide_risk

    def add_origin_memories(self, memory_path):
        if not os.path.exists(os.path.dirname(memory_path)):
            os.makedirs(os.path.dirname(memory_path))

        # 初始化memory
        if not os.path.exists(memory_path):
            with open(memory_path, 'w', encoding='utf-8') as f:
                f.write('')
                # self.add_init_observation()
        else:
            with open(memory_path, 'r', encoding='utf-8') as f:
                file_content = f.read().strip()
                if file_content:
                    memory = json.loads(file_content)
                    # self.agent.memory.add_origin_memories(memory)


    def add_init_observation(self):
        agent_observations = self.client_character_content["observations"]
        # for observation in agent_observations:
        #     self.agent.memory.add_memory(observation)
        #Pre-summary
        # print(self.agent.get_summary(force_refresh=True))

    def add_sim_setup(self, sim_setup_path):
        if not os.path.exists(sim_setup_path):
            with open(sim_setup_path, 'w', encoding='utf-8') as f:
                sim_setup_dict = {
                                    "sim_start_time": "2024年08月31日00:00",
                                    "sim_days": 1,
                                    "before_day_summary": "",
                                    "last_update_time": "",
                                    "if_new_day": True,
                                    "past_stage_num": 0
                                }
                json.dump(sim_setup_dict, f, ensure_ascii=False, indent=4)
