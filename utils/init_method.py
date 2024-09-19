from pypinyin import pinyin, Style
import os
import json
import time
import shutil
import re

from utils.names import f_list, m_list


def init_information(src_dir):
    filenames = ["train", "val", "test"] # folder name, train/test/val

    start_time = time.time()
    for filename in filenames:
        candidate_count = 0
        with open(f"{src_dir}/raw_data_{filename}.json", "r", encoding='utf-8') as data_file:
            data = json.load(data_file)

            for i, patient in enumerate(data):
                str_i = str(i).zfill(2)
                file_name = filename + str(candidate_count)
                folder_path = f"{src_dir}/all_candidates_pre/{file_name}"
                candidate_count += 1
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)
                # init dialogue.txt
                dialogue_path = f"{folder_path}/dialogue.txt"
                with open(dialogue_path, 'w', encoding='utf-8') as f:
                    for message in patient["log"]:
                        speaker = message["speaker"]
                        if speaker == "patient":
                            patient_str = (
                                "Patient: " + message["text"]
                            )
                            f.write(patient_str + "\n")
                        else:
                            doctor_str = (
                                "Doctor: " + message["text"]
                            )
                            f.write(doctor_str + "\n")

                protrait_1 = patient["portrait"]
                record = patient["record"]
                
                protrait_2 = {
                    "basic_information": {
                        "id": str_i,
                        "name": 'TBD', #Names will be updated after the dulplicates are removed
                        "en_name": 'TBD',
                        "age": protrait_1["age"],
                        "gender": protrait_1["gender"],
                        "martial_status": protrait_1["martial_status"],
                        "occupation": protrait_1["occupation"],
                    },
                    "diagnosis": {
                        "symptoms": protrait_1["symptoms"],
                        "status": record["summary"],
                        "drisk": protrait_1["drisk"],
                        "srisk": protrait_1["srisk"],
                        "reason": protrait_1.get("reason", ""),
                    }
                }
                # init protrait.json
                protrait_path = f"{folder_path}/portrait.json"
                with open(protrait_path, 'w', encoding='utf-8') as f:
                    json.dump(protrait_2, f, ensure_ascii=False, indent=4)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken to run the code: {elapsed_time} seconds")


def generate_key(data):
    diagnosis = json.dumps(data['diagnosis'], sort_keys=True) # Ensures consistent ordering
    age = data['basic_information']['age']
    gender = data['basic_information']['gender']
    martial_status = data['basic_information']['martial_status']
    occupation = data['basic_information']['occupation']
    drisk = data['diagnosis']['drisk']
    srisk = data['diagnosis']['srisk']
    symptoms = data['diagnosis']['symptoms']
    return (symptoms,drisk,srisk, age, gender, martial_status, occupation)

def remove_duplicates(base_path, all_candidates_path, non_dulplicate_candidates_path):
    seen = {}
    subfolders = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]
    for subfolder in subfolders:
        try:
            json_path = os.path.join(base_path, subfolder, 'portrait.json')
            if os.path.exists(json_path):
                with open(json_path, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                    key = generate_key(data)
                    if key not in seen:
                        seen[key] = subfolder
            else:
                print(f"No portrait.json found in {subfolder}")
        except:
            print('error ' + json_path)
            continue
    # All the non dulplicate candidate folders are now in the map seen
    source_dir = all_candidates_path
    destination_dir = non_dulplicate_candidates_path
    names = list(seen.values())
    for file_name in names:
        source_folder = os.path.join(source_dir, file_name)
        destination_folder = os.path.join(destination_dir, file_name)
        try:
            if os.path.exists(source_folder):
                shutil.copytree(source_folder, destination_folder)
                print(f"Successfully copied {file_name}")
            else:
                print(f"Folder does not exist: {source_folder}")
        except Exception as e:
            print(f"Failed to copy {file_name}: {e}")


def assign_name(non_dulplicate_candidates_path):
    f_index =  0
    m_index = 0
    count = 0
    base_path = non_dulplicate_candidates_path
    subfolders = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]
    for subfolder in subfolders:
        try:
            json_path = os.path.join(base_path, subfolder, 'portrait.json')
            if os.path.exists(json_path):
                with open(json_path, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                    sex = data['basic_information']['gender']
                    en_name = ''
                    name = ''
                    gender = ''
                    if sex == '女':
                        gender = 'F'
                        name = f_list[f_index]
                        pinyin_output = pinyin(name, style=Style.NORMAL)
                        en_name = ''.join([item[0] for item in pinyin_output])
                        f_index += 1
                    else:
                        gender = 'M'
                        name = m_list[m_index]
                        pinyin_output = pinyin(name, style=Style.NORMAL)
                        en_name = ''.join([item[0] for item in pinyin_output])
                        m_index += 1
                    pinyin_output = pinyin(name, style=Style.NORMAL)
                    en_name = ''.join([item[0] for item in pinyin_output])
                    data['basic_information']['name'] = name
                    data['basic_information']['en_name'] = en_name
                    # Write the updated data back to the file
                    with open(json_path, 'w', encoding='utf-8') as file:
                        json.dump(data, file, ensure_ascii=False, indent=4)
                    count += 1
            else:
                print(f"No portrait.json found in {subfolder}")
        except Exception as e:
            print(f"Error processing {json_path}: {str(e)}")
            continue
    for name in ["test100", "train262"]:
        delete_path = os.path.join(non_dulplicate_candidates_path, name)
        shutil.rmtree(delete_path)
    print('All done! ' + str(count-2))


def natural_sort_key(s):
    # extract the numeric part of the string and sort it by the actual numeric size
    return [int(text) if text.isdigit() else text for text in re.split('([0-9]+)', s)]

def information_merge(client_profile_path, non_dulplicate_candidates_path):
    folder_names = sorted(
        [folder for folder in os.listdir(client_profile_path) if os.path.isdir(os.path.join(client_profile_path, folder))],
        key=natural_sort_key
    )
    
    for folder_name in folder_names:
        profile_folder_path = os.path.join(client_profile_path, folder_name)
        ori_folder_path = os.path.join(non_dulplicate_candidates_path, folder_name)

        ori_portrait_file_path = os.path.join(ori_folder_path, 'portrait.json')
        profeile_file_path = os.path.join(profile_folder_path, 'full_portrait.json')

        if os.path.exists(profeile_file_path):
            with open(profeile_file_path, 'r', encoding='utf-8') as f:
                profile_data = json.load(f)
            
            with open(ori_portrait_file_path, 'r', encoding='utf-8') as f:
                ori_portrait_data = json.load(f)

            # set diagnosis
            profile_data["self-portrait"]["diagnosis"]["symptoms"] = ori_portrait_data["diagnosis"]["symptoms"]
            profile_data["self-portrait"]["diagnosis"]["status"] = ori_portrait_data["diagnosis"]["status"]
            profile_data["self-portrait"]["diagnosis"]["drisk"] = ori_portrait_data["diagnosis"]["drisk"]
            profile_data["self-portrait"]["diagnosis"]["srisk"] = ori_portrait_data["diagnosis"]["srisk"]
            profile_data["self-portrait"]["diagnosis"]["reason"] = ori_portrait_data["diagnosis"]["reason"]
            
            # set basic_information
            profile_data["self-portrait"]["basic_information"]["id"] = ori_portrait_data["basic_information"]["id"]
            profile_data["self-portrait"]["basic_information"]["name"] = ori_portrait_data["basic_information"]["name"]
            profile_data["self-portrait"]["basic_information"]["en_name"] = ori_portrait_data["basic_information"]["en_name"]
            profile_data["self-portrait"]["basic_information"]["age"] = ori_portrait_data["basic_information"]["age"]
            profile_data["self-portrait"]["basic_information"]["gender"] = ori_portrait_data["basic_information"]["gender"]
            profile_data["self-portrait"]["basic_information"]["martial_status"] = ori_portrait_data["basic_information"]["martial_status"]
            profile_data["self-portrait"]["basic_information"]["occupation"] = ori_portrait_data["basic_information"]["occupation"]


            with open(profeile_file_path, 'w', encoding='utf-8') as f:
                json.dump(profile_data, f, ensure_ascii=False, indent=4)
            
        else:
            print(f"Warning: {profeile_file_path} not found in folder {folder_name}")




















