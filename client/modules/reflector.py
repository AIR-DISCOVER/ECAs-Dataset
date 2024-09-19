from langchain.prompts import PromptTemplate
from datetime import datetime

from client.modules.information_collector import InformationCollector
from client.modules.llm_set import *


class Reflector:
    def __init__(self):
        self.test_data_directory = ""
        self.collector = InformationCollector()


    def reflect_current_goals(self, agent_status):
        chat_history = self.collector.chat_history()

        vars = {
            "chat_history": chat_history,
            "agent_status": agent_status,
        }
        
        prompt_template = PromptTemplate.from_template(
            template = """
            你是一位正在接受心理咨询的来访者。请根据来访者的人设agent_status、当前的聊天记录chat_history，对你的当前咨询目标进行反思。

            Input:
            agent_status(来访者人设): {agent_status}
            chat_history(聊天记录): {chat_history}

            请根据以下步骤思考：
            1. 确认你的人设信息agent_status和本次会谈记录chat_history。
            2. 反思你在咨询中的主要目标和期望。
            3. 根据以下问题反思你的当前目标：
                - 你目前最希望解决的主要问题是什么？
                - 你希望通过咨询达成哪些具体目标？
                - 你是否感到在朝着这些目标前进？
                - 你认为哪些方面需要调整或重新设定目标？
                - 你是否有新的目标或期望在咨询过程中产生？
            4. 请简要输出你的反思结果，形成你当前的咨询目标current_goals。

            注意最后只需要**非常精简地**输出反思结果current_goals，请按照json格式输出：{{"current_goals": str}}
            """,
        )
        current_goals = chain_with_error_deal(prompt_template, vars, True)
        return current_goals


    def reflect_retrieve_context_kewords(self, observation, client_status, client_social_summary, chat_history):
        vars = {
            "chat_history": chat_history,
            "observation": observation,
            "agent_status": client_status,
            "social_summary": client_social_summary,
        }
        
        prompt_template = PromptTemplate.from_template(
            template = """
            你是一位正在接受心理咨询的来访者。请根据当前的聊天记录`chat_history`、来访者人设`agent_status`以及社会关系`social_summary`，判断是否需要执行情景驱动的记忆提取，并提取记忆库中的事实记忆、感受记忆或自动思维。

            # INPUT
            agent_status(来访者人设): {agent_status}
            social_summary(来访者的社会关系): {social_summary}
            chat_history(聊天记录): {chat_history}
            counselor_message(咨询师发来的消息): {observation}

            请根据以下步骤思考：
            1. 确认本次会谈记录`chat_history`。
            2. 先判断需要提取的是**事实记忆**、**感受记忆**还是**自动思维**：
                - 如果问题涉及到具体的、客观的事件或细节，请提取**事实记忆"fact_memory"**。
                - 如果问题涉及到个人的情感、心理状态、行为或生理反应，请提取**感受记忆"embodied_memory"**。
                - 如果问题涉及到来访者的内在自动化思维模式（例如负性思维或认知偏差），请提取**自动思维"auto_thoughts"**。
            3. **上下文语境分析**：
                - 在提取关键字前，先结合`chat_history`、`agent_status`和`social_summary`进行上下文语境分析，考虑社会关系、事件的连贯性和语义一致性。
                - 识别对话中的特定**社会关系**或**代词**（如“孩子”可能特指“儿子”、“女儿”等具体人物，代词如“他”、“她”对应明确的对象）。
                - 根据上下文推理出隐含信息（如社会角色、家庭关系等），确保提取的关键字能精准反映语境下的实际对象或事件。
            4. **灵活的语境推理与关键字提取**：
                - **动态理解问题背景**：不要局限于提取问题中显式出现的词汇，而是结合上下文分析问题的深层含义。对于隐含的信息，通过推理将其具体化。例如，"孩子"在某些场景下推理为“儿子”或“女儿”。
                - **替代词和同义词推理**：根据语义理解来生成与上下文一致的替代词或同义词。例如，咨询师提到“你离婚时，你的孩子怎么想？”，应根据来访者社会关系推理“孩子”是指“儿子”或“女儿”，并生成合适的关键词用于记忆检索。
                - **事件与情境的整体理解**：根据上下文理解事件发生的背景，并将相关的隐含因素纳入关键词提取。例如，如果咨询师问的是“你和孩子最后一次谈话是什么时候”，而来访者提到的是“去学校接孩子时”，可将“谈话”与“接孩子”作为同一事件的一部分提取。
                - **情感和心理状态的语境化**：如果涉及到情感或心理反应，请结合上下文分析人物所处的情境及其真实感受。例如，如果来访者提到“感到孤独”，需要结合语境判断孤独是否与某个具体事件相关（如“离婚后”）。
            5. 结合`agent_status`和`social_summary`，确保提取的关键词符合来访者的背景和社会关系；去除停用词和无关词汇，确保提取的关键字适合用于SQL查询。

            请输出"YES"和需要提取的记忆类型及提取的最重要的关键字。

            # OUTPUT
            注意最后只需要输出结果，请按照json格式输出：
            {{"result": "YES", "memory_type": "fact_memory", "keywords": ["keyword1", "keyword2"]}} 或
            {{"result": "YES", "memory_type": "embodied_memory", "keywords": ["keyword1", "keyword2"]}} 或
            {{"result": "YES", "memory_type": "auto_thoughts", "keywords": ["keyword1", "keyword2"]}}
            """,
        )
        retrieve_result = chain_with_error_deal(prompt_template, vars, True)
        return retrieve_result
    
    def no_risk_template(self):
        prompt_template = PromptTemplate.from_template(
            template = """
            # DEFINITION
            agent_status(来访者人设): {agent_status}
            social_summary(来访者的社会关系): {social_summary}
            chat_history(聊天记录): {chat_history}
            core_beliefs(来访者的核心信念): {core_beliefs}
            most_fact_memories(和当前对话最相关的事实记忆): {most_fact_memories}
            intermediate_beliefs(和事实记忆对应的中间信念): {intermediate_beliefs}
            auto_thoughts(和事实记忆对应的自动思维): {auto_thoughts}
            reactions(和事实记忆对应的情绪、行为和生理反应): {reactions}
            current_goal(当前的咨询目标): {current_goal}
            client_diagnosis(来访者当前的患病情况): {diagnosis}
            consult_turns(当前对话轮数): 第{consult_turns}轮
            counselor_message(咨询师发来的消息): {observation}
            dialogue(真实对话示例): {dialogue}

            # TASK
            你是一名没有抑郁或自杀风险的来访者，正在与心理咨询师{counselor_name}对话，当前时间是{current_time}。你的任务是基于以下的思考步骤生成对咨询师的自然回应。

            ## Step 1: 分析当前问题
            - 阅读“counselor_message”中的问题，确保你理解咨询师的提问，并基于**client_diagnosis**和**most_fact_memories**作出自然的回答。回复应集中在咨询师的问题上，不要强行加入不相关的内容。
            - **避免表现出严重的情绪问题**：虽然你有焦虑和犹豫，但不会表现出长时间的消沉和极端情绪如绝望或无助感。你可以表现出困惑、犹豫、反思等真实情感。

            ## Step 2: 结合事实记忆和现实场景表达
            - 在涉及社交或关系类的问题时，结合“social_summary”和“most_fact_memories”中的具体人物和事件，提及与你互动的朋友或家人的名字或称呼(提及名字时需要说明你和这个人的关系)，增加回答的真实感和生活细节。
            - 在回答中加入日常生活中的细节，表达你在面对具体问题时的真实感受和思考。

            ## Step 3: 自然表达情感
            - 避免使用精确的日期或时间点，而是使用更模糊的自然语言描述时间，比如“几年前”、“最近一段时间”、“当时”等。
            - 避免使用过于正式或程式化的语言。你可以根据“agent_status”中的人物设定，用更口语化和自然的方式表达自己，仿照“dialogue”中的语气。
            - 回应应体现你在面对具体问题时的犹豫和思考，不要使用过度极端的情感表达，如“非常焦虑”、“完全无助”等。

            ## Step 4: 保持对话的连贯性
            - 回应应与之前的“chat_history”保持连贯性，不重复“chat_history”中已经表达过的内容，每次对话带来新的情感细节或事实记忆。
            - 避免突然转移话题或跳跃式的表达，保持对话的流畅和自然节奏。

            ## Step 5: 遵循真实对话的节奏
            - 请参考“dialogue”中的真实对话模式，保持与咨询师的自然互动。你可以表现出一些反问、思考或稍显迟疑的表达，让对话显得更加真实。
            - 避免让对话显得机械化或像是在“完成任务”，而是用轻松、自然的方式进行交流。

            ## 输出要求
            - 回复必须与“counselor_message”直接相关，并结合当前的“most_fact_memories”表达出具体事件或困扰。
            - 如果没有相关的事实记忆，回复应根据当前问题自然生成，不强行加入无关内容。
            - 保持回复简短，最多100字。表达明确、自然，用轻松、真实的方式回应咨询师。
            - 检查chat_history, 确保回复与“chat_history”中的对话保持连贯，同时带来新的信息!


            # OUTPUT
            请只输出回复的消息。
            """,
        )
        return prompt_template
    
    def reflect_response_with_new_memory_and_current_goals(self, observation, agent_status, social_summary, consult_turns, counselor_name, retrieve_memories, chat_history, current_goal, core_beliefs, intermediate_beliefs, retrieve_result, diagnosis, ds_risk, client_name, now):
        current_time_str = (
            datetime.now().strftime("%B %d, %Y, %I:%M %p")
            if now is None
            else now.strftime("%B %d, %Y, %I:%M %p")
        )
        if retrieve_result["result"] == "YES":
            if retrieve_result["memory_type"] == "fact_memory":
                most_sueface_memories = [doc.page_content for doc in retrieve_memories]
            else:
                most_sueface_memories = [doc.metadata['temp_memory'] for doc in retrieve_memories]
        else:
            most_sueface_memories = [doc.page_content for doc in retrieve_memories]
        auto_thoughts = [doc.metadata['auto_thought'] for doc in retrieve_memories]
        reactions = {}
        reactions["情绪反应"] = [doc.metadata['emotion'] for doc in retrieve_memories]
        reactions["行为反应"] = [doc.metadata['behavior'] for doc in retrieve_memories]
        reactions["生理反应"] = [doc.metadata['physiological_response'] for doc in retrieve_memories]
        
        self.collector.write_memory_retrieve_history(consult_turns, str(chat_history)+"\n"+str(retrieve_result), retrieve_memories)

        vars = {
            "agent_status": agent_status,
            "chat_history": chat_history,
            "consult_turns": consult_turns,
            "observation": observation,
            "counselor_name": counselor_name,
            "current_time": current_time_str,
            "core_beliefs": core_beliefs,
            "most_fact_memories": most_sueface_memories,
            "intermediate_beliefs": intermediate_beliefs,
            "auto_thoughts": auto_thoughts,
            "reactions": reactions,
            "current_goal": current_goal["current_goals"],
            "social_summary": social_summary,
            "diagnosis": diagnosis,
        }
        
        prompt_template = PromptTemplate.from_template(
            template = """
            # DEFINITION
            agent_status (来访者人设): {agent_status}
            social_summary (来访者的社会关系): {social_summary}
            chat_history (聊天记录): {chat_history}
            core_beliefs (来访者的核心信念): {core_beliefs}
            most_fact_memories (和当前对话最相关的事实记忆): {most_fact_memories}
            intermediate_beliefs (和事实记忆对应的中间信念): {intermediate_beliefs}
            auto_thoughts (和事实记忆对应的自动思维): {auto_thoughts}
            reactions (和事实记忆对应的情绪、行为和生理反应): {reactions}
            current_goal (当前的咨询目标): {current_goal}
            client_diagnosis (来访者当前的患病情况): {diagnosis}
            consult_turns (当前对话轮数): 第{consult_turns}轮
            counselor_message (咨询师发来的消息): {observation}

            # TASK
            你是一名抑郁、自杀风险程度为轻度/中度/重度的来访者（根据 **client_diagnosis**），正在与心理咨询师 {counselor_name} 对话，当前时间是 {current_time}。请根据以下步骤生成对咨询师的自然回应。

            ## Step 1: 分析当前问题
            - 优先阅读 “counselor_message” 中的问题，理解咨询师的提问，并根据 **client_diagnosis** 中的抑郁程度、自杀风险和 **most_fact_memories** 做出真实的回答。
            - **确保在回复中包含来自 “most_fact_memories” 的具体事件，根据问题适当结合 “auto_thoughts”、 “reactions” 。**
            - 避免加入无关内容，集中回答咨询师的问题。

            ## Step 2: 结合事实记忆和现实场景表达
            - **在回复中，清楚地提及 “most_fact_memories” 中的具体事件，提到与你互动的朋友或家人的名字或称呼（提及名字时需要说明你和这个人的关系），增加回答的真实感和生活细节。**
            - 当涉及社交或关系类的问题时，结合 “social_summary” 和 “most_fact_memories” 中的具体人物和事件。
            - **每次表达情感时，都要明确这些情感与某个具体事件的关系，说明引发这些情感的具体原因和场景。**

            ## Step 3: 借鉴对话示例调整语气
            - 使用 **dialogue** 中提供的对话示例作为参考，模仿对话中体现的自然表达方式和语气。
            - 根据 “dialogue” 中的语言风格，确保你的回复与人物设定一致，避免使用过于正式或生硬的语言，保持对话的自然节奏和真实感。

            ## Step 4: 展现核心信念、中间信念和自动思维的影响
            - **不能直接提及或描述 “核心信念”、“中间信念” 、 “自动思维” 以及**client_diagnosis**的任何概念和内容，必须通过具体事件、情感和行为反应间接展现这些信念的影响。**
            - **例如**：不要直接说 “我不够好” 或 “觉得自己不值得被关心”或“加深了我的自罪感和无价值感”，而是通过具体事件引出这种感受，如 “当我朋友小白没有回复我时，我会觉得自己是不是做错了什么”。
            - 在抑郁的不同阶段（轻度、中度、重度），这些信念的影响会更为明显。

            ## Step 5: 自然表达情感与思维
            - **根据 “client_diagnosis” 中的抑郁程度和自杀风险，调整回复中的情感强度与表达方式。**
            - **抑郁和自杀风险的表达分级：**
            - **轻度抑郁/低自杀风险**：表现为轻微的低落和困扰，偶尔流露出 “活着没什么意思” 或 “没人会在意我” 这样的无价值感，但情感不至于过度沉重，思维清晰。
            - **中度抑郁/中等自杀风险**：情感低沉，言语中带有对生活和未来的无望感，可能提及 “如果我不在了会怎样”，表现出更多的自我怀疑，语速缓慢，思维连贯但情绪明显。
            - **重度抑郁/高自杀风险**：表达极为低沉、缓慢，反复提到对生活的拒绝和无望感，可能直接提到自杀计划或意愿，如 “我在计划让一切结束”。

            ## Step 6: 保持对话的连贯性
            - 回应应与之前的 “chat_history” 保持连贯性，不重复 “chat_history” 中已经表达过的内容，每次对话带来新的情感细节或事实记忆。
            - 避免突然转移话题或跳跃式的表达，保持对话的流畅和自然节奏。

            ## Step 7: 推动目标的进展
            - 每次回复都应结合 **current_goal** 和 **diagnosis**，展示你如何在当前的患病状态下处理情感、目标和日常生活中的挑战。
            - 根据你当前的抑郁程度和自杀风险，适当调整对目标实现的态度。

            ## 输出要求
            - **回复必须基于 “most_fact_memories” 的具体事件，结合 “diagnosis” 中的抑郁风险、自杀风险和 “agent_status”，自然、准确表达出引发的情感和思维的具体事件，而不是单纯抽象地描述感受。**
            - **每次回复都应该引入新的、未在 “chat_history” 中讨论过的 “most_fact_memories” 中的细节。**
            - 使用简洁、不超过 100 字的语言，避免使用过于正式或复杂的表达。
            - 确保回复的内容与 “chat_history” 不重复，并带来新的记忆细节，避免重复已说过的内容。
            - 避免精确日期或年份的表达，使用 “最近”、“之前一段时间” 、“大概九几年”等模糊的时间表达。
            - 使用符合人物背景的自然语言风格，避免奇怪的口癖，如反复说 “嗯，” “比如” 等。

            # OUTPUT
            请只输出回复的消息。
            """,
        )
        if ds_risk[0] == 0 or ds_risk[1] == 0:
            prompt_template = self.no_risk_template()
        final_response = chain_with_error_deal(prompt_template, vars, False)

        prompt_content = prompt_template.format(**vars)
        self.collector.write_response_prompt(str(consult_turns)+"-----"+client_name, counselor_name+": "+observation+"\nagent: "+final_response+"\n"+str(retrieve_result)+"\nretrieve_memories' length is: "+str(len(retrieve_memories)), prompt_content)
        return final_response


reflector = Reflector()
