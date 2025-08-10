import os

def get_data_folder():
    # Get the project root directory (parent of src)
    try:
        # Path to the src folder
        src_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        # If running in notebook, assume working dir is project root
        src_dir = os.getcwd()

    # Project root is one level up from src
    project_root = os.path.abspath(os.path.join(src_dir, ".."))

    # Data directory at project root
    data_folder = os.path.join(project_root, "data")
    os.makedirs(data_folder, exist_ok=True)

    
    return data_folder