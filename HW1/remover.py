import os
import shutil

def remove_all_files(directory_path):
    """
    Removes all files in the specified directory
    
    :param directory_path: Path to the directory whose files should be removed
    """
    try:
        # Check if the directory exists
        if not os.path.exists(directory_path):
            print(f"Directory '{directory_path}' does not exist.")
            return
            
        # List all items in the directory
        for item in os.listdir(directory_path):
            item_path = os.path.join(directory_path, item)
            
            # Check if it's a file and remove it
            if os.path.isfile(item_path):
                os.remove(item_path)
                print(f"Removed file: {item_path}")
            # Optionally: handle subdirectories
            # elif os.path.isdir(item_path):
            #     shutil.rmtree(item_path)
            #     print(f"Removed directory: {item_path}")
    
    except Exception as e:
        print(f"Error occurred: {e}")
    
    print(f"All files in '{directory_path}' have been removed.")

# Example usage:
if __name__ == "__main__":
    remove_all_files("C:/Users/inpir/OneDrive/AI_Capstone/HW1/datasets/0")
    remove_all_files("C:/Users/inpir/OneDrive/AI_Capstone/HW1/datasets/1")
    remove_all_files("C:/Users/inpir/OneDrive/AI_Capstone/HW1/datasets/2")
    remove_all_files("C:/Users/inpir/OneDrive/AI_Capstone/HW1/datasets/3")
    remove_all_files("C:/Users/inpir/OneDrive/AI_Capstone/HW1/datasets/4")
    remove_all_files("C:/Users/inpir/OneDrive/AI_Capstone/HW1/datasets/5")