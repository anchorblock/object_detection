import os
import shutil



def create_directory(directory):
    """
    Creates a new directory if it does not already exist.
    
    Args:
    - directory (str): The path to the directory that needs to be created.
    """
    
    if not os.path.exists(directory):
        os.makedirs(directory) 


def remove_directory(directory):
    """
    Removes a directory and its contents if it exists.
    
    Args:
    - directory (str): The path to the directory that needs to be removed.
    """
    
    if os.path.exists(directory):
        shutil.rmtree(directory)


def generate_filepath_list(folder_path, extension):
    """
    Returns a list of filepaths that have a certain extension in a given directory and its subdirectories.
    
    Args:
    - folder_path (str): The path to the directory that contains the files.
    - extension (str): The file extension to look for.
    
    Returns:
    - filepaths (List[str]): A list of filepaths that have the specified extension in the directory and its subdirectories.
    """
    
    filepaths = []
    for root, _, files in os.walk(folder_path):
        for filename in files:
            if filename.endswith(f".{extension}"):
                filepath = os.path.join(root, filename)
                filepaths.append(filepath)
    return filepaths