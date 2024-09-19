from utils.utils import *
from utils.init_method import init_information, remove_duplicates, assign_name, information_merge


def main():
    # init client base info
    src_dir = d4_raw_data_path
    init_information(src_dir)

    # find out all non duplicate candidates and move them to a new folder
    base_path = all_candidates_path
    remove_duplicates(base_path, all_candidates_path, non_dulplicate_candidates_path)
    
    # assign name for each client agent
    assign_name(non_dulplicate_candidates_path)
    
    # assing base information into client profile
    information_merge(client_profile_path, non_dulplicate_candidates_path)


if __name__ == '__main__':
    main()
