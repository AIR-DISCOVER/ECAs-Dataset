from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_community.docstore.in_memory import InMemoryDocstore
from client.modules.time_weighted_retriever import TimeWeightedVectorStoreRetriever
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

import faiss
import math
import time
import json
import re
from utils.utils import model_name, api_key, api_url, base_path

def relevance_score_fn(score: float) -> float:
    """Return a similarity score on a scale [0, 1]."""
    return 1.0 - score / math.sqrt(2)

def create_vector_retriever(api_key, api_url):
    """Create a normal vector retriever."""
    # Define embedding model
    embeddings_model = OpenAIEmbeddings(api_key=api_key, openai_api_base=api_url, model="text-embedding-3-small")
    # Initialize the vectorstore as empty
    embedding_size = 1536
    index = faiss.IndexFlatL2(embedding_size)
    vectorstore = FAISS(
        embeddings_model,
        index,
        InMemoryDocstore({}),
        {},
        relevance_score_fn=relevance_score_fn,
    )
    return TimeWeightedVectorStoreRetriever(
        vectorstore=vectorstore, decay_rate=0, k=3
    )

def vector_retriever():
    return create_vector_retriever(api_key, api_url)


def clean_JSON(ai_output):
    try:
        data = json.loads(ai_output)
        return data
    except json.JSONDecodeError:
        pass  # 如果解析失败，继续尝试通过正则表达式提取 JSON
    
    # 使用正则表达式查找 JSON 块
    json_match = re.search(r'```json(.*?)```', ai_output, re.DOTALL)
    
    if not json_match:
        return ""  # 如果没有找到 JSON 块，返回 None

    # 提取并清理 JSON 字符串
    cleaned_json_str = json_match.group(1).strip()
    cleaned_json_str = cleaned_json_str.replace("True", "true").replace("False", "false")
    
    if not cleaned_json_str:
        return ""  # 如果清理后的 JSON 字符串为空，返回 None

    try:
        data = json.loads(cleaned_json_str)
        return data
    except json.JSONDecodeError as e:
        print(f"JSONDecodeError: {e.msg} at line {e.lineno} column {e.colno}")
        return ""  # 如果解析失败，返回 None


def chain_with_error_deal(prompt_template: PromptTemplate, vars, if_json):
    request_content = ""
    while True:
        try:
            chain = prompt_template | LLM
            request_content = chain.invoke(vars).content
            if if_json:
                request_content = clean_JSON(request_content)
            if len(request_content) > 0:
                break
        except KeyboardInterrupt:
            break
        except Exception as e:
            print("Exception happen! Error is:", e)
            time.sleep(3.0)
            if len(request_content) > 0:
                break
    return request_content


class GlobalConfig:
    client_name = ""
    counselor_name = ""

    base_path = base_path

    client_storage_path_base = base_path + "/client/embodied_memory/ori_data/"
    client_storage_path = base_path + "/client/embodied_memory/ori_data/"

    memory_database_path = base_path + "/client/embodied_memory/database.db"

    client_character_path_base = base_path + "/client/profiles/"
    client_character_path = base_path + "/client/profiles/"


model_name = model_name
api_key = api_key
api_url = api_url

LLM = ChatOpenAI(openai_api_base=api_url, model_name=model_name, openai_api_key=api_key)