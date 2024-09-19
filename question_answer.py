import json
import os
import argparse
import re

from utils.utils import client_profile_path, client_memory_path
from client.modules.llm_set import GlobalConfig
from client.client import Client


parser = argparse.ArgumentParser(description='Client Agents Answer.')
parser.add_argument('--client_name', '--cn', default="test0", type=str, help="client's folder name.")


def natural_sort_key(s):
    # Extract the numeric part of the string and sort it by the actual numeric size
    return [int(text) if text.isdigit() else text for text in re.split('([0-9]+)', s)]


def process_single_turn(questions, client: Client):    
    responses = []

    for idx, question in enumerate(questions):
        user_message = question
        response = client.agent.generate_client_dialogue_response_for_evaluation(user_message, 1, "", "counselor", "single")

        responses.append({str(idx): {
            "Counselor": user_message,
            "Client": response
        }})
    return responses


def main(args):
    folder_name = args.client_name

    client_base_path = client_profile_path

    client_path = os.path.join(client_base_path, folder_name)

    full_portrait_path = os.path.join(client_path, 'full_portrait.json')
    sim_portrait_path = os.path.join(client_memory_path, folder_name)
    # check full_portrait.json if exist
    if os.path.exists(full_portrait_path) and os.path.exists(sim_portrait_path):
        with open(full_portrait_path, 'r', encoding='utf-8') as f:
            portrait_data = json.load(f)
        
        # extract client_name
        client_name = portrait_data["self-portrait"]["basic_information"]["en_name"]
        
        # set GlobalConfig.client_name for client's folder name
        GlobalConfig.client_name = client_name
        GlobalConfig.client_character_path = os.path.join(GlobalConfig.client_character_path_base, folder_name)
        GlobalConfig.client_storage_path = os.path.join(GlobalConfig.client_storage_path_base, folder_name)
        
        # init Client object
        client = Client(client_name)
        client.folder_name = folder_name
        client.agent.folder_name = folder_name

        # answer questions
        questions = ["你最近对你原本喜欢的活动（如娱乐或工作）失去兴趣了吗？是什么时候开始的？", "你回忆一下过去两周，是否经常感到疲倦，即使没有做什么事情？"]
        responses = process_single_turn(questions, client)
        print(f"The responses of questions are: {responses}")
    else:
        print(f"Error happend! This client's memory is not exist!")
    


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)