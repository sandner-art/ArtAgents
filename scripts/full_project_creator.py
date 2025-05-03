# ArtAgent/scripts/full_project_creator.py
import os

# --- Configuration ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Project root is parent of scripts/
OUTPUT_FILE = os.path.join(PROJECT_ROOT, "artagent_fullproject.txt")
FOLDERS_TO_INCLUDE = ["", "agents", "core", "scripts", "tests"] # Root + subfolders
FILE_EXTENSIONS_TO_INCLUDE = ['.py', '.json', '.bat', '.md', '.txt']
FILES_TO_EXCLUDE = ['artagent_fullproject.txt'] # Exclude the output file itself
FOLDERS_TO_EXCLUDE = ['venv', '__pycache__', '.git', '.vscode'] # Folders to skip entirely

def create_full_project_file():
    """Gathers content from specified files and writes to an output file."""
    all_files_content = {}

    print(f"Scanning project root: {PROJECT_ROOT}")
    print(f"Including folders: {FOLDERS_TO_INCLUDE}")
    print(f"Including extensions: {FILE_EXTENSIONS_TO_INCLUDE}")
    print(f"Excluding files: {FILES_TO_EXCLUDE}")
    print(f"Excluding folders: {FOLDERS_TO_EXCLUDE}")


    for folder_rel_path in FOLDERS_TO_INCLUDE:
        folder_abs_path = os.path.join(PROJECT_ROOT, folder_rel_path)
        if not os.path.isdir(folder_abs_path):
            print(f"Warning: Folder not found, skipping: {folder_abs_path}")
            continue

        print(f"Processing folder: {folder_rel_path if folder_rel_path else '.'}")

        # Walk through the directory
        for subdir, dirs, files in os.walk(folder_abs_path, topdown=True):
             # Exclude specified folders
            dirs[:] = [d for d in dirs if d not in FOLDERS_TO_EXCLUDE]
            files[:] = [f for f in files if f not in FILES_TO_EXCLUDE]

            for filename in files:
                # Check file extension
                if any(filename.lower().endswith(ext) for ext in FILE_EXTENSIONS_TO_INCLUDE):
                    file_abs_path = os.path.join(subdir, filename)
                    file_rel_path = os.path.relpath(file_abs_path, PROJECT_ROOT).replace("\\", "/") # Use forward slashes

                    print(f"  Adding: {file_rel_path}")
                    try:
                        with open(file_abs_path, 'r', encoding='utf-8', errors='ignore') as infile:
                            content = infile.read()
                        all_files_content[file_rel_path] = content
                    except Exception as e:
                        print(f"  ERROR reading {file_rel_path}: {e}")
                        all_files_content[file_rel_path] = f"*** ERROR READING FILE: {e} ***"


    print(f"\nWriting combined content to {OUTPUT_FILE}...")
    try:
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as outfile:
            outfile.write("--- ArtAgent Project Dump ---\n\n")
            outfile.write("Structure (Relative Paths from Project Root):\n")
            # Write sorted list of files included
            for file_rel_path in sorted(all_files_content.keys()):
                outfile.write(f"- {file_rel_path}\n")

            outfile.write("\n\n--- File Contents ---\n\n")

            # Write content of each file
            for file_rel_path in sorted(all_files_content.keys()):
                outfile.write(f"--- START OF FILE {file_rel_path} ---\n")
                outfile.write(all_files_content[file_rel_path])
                outfile.write(f"\n--- END OF FILE {file_rel_path} ---\n\n")

        print("Successfully created project dump file.")

    except Exception as e:
        print(f"ERROR writing to output file {OUTPUT_FILE}: {e}")


if __name__ == "__main__":
    create_full_project_file()