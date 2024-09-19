# Client Initialization and Duplicate Removal

To init clients, remove dulplicates and assign names to candidates, you need three Python scripts: `client_init.py`, `init_method.py` and `names.py` for processing candidate data from D4 dataset.

## Requirements

- Python 3.7+
- Install required library:

  ```
  pip install pypinyin
  ```

## Run scripts
1. `client_init.py`
- Initializes candidate data by generating folders and saving dialogue and profile information from D4 dataset.
- Set `src_dir` and `filename` variables in the script, and run:

  ```
  python client_init.py
  ```

2. `remove_dulplicate.py`
- Removes duplicate candidate entries and moves non-duplicates to a new folder.
- Set `base_path`, `source_dir`, and `destination_dir` variables in the script, and run:

  ```
  python remove_dulplicate.py
  ```
3. `assign_name.py`
- Assign Chinese and English names to the non duplicate candidates.
- Set `base_path` variable in the script, and run:

  ```
  python assign_name.py
  ```