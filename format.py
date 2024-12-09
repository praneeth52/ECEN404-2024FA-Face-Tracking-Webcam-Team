import os

def check_label_format(file_path):
    with open(file_path, 'r') as file:
        for line_number, line in enumerate(file, start=1):
            parts = line.strip().split()
            # Check if there are exactly 5 elements in each line
            if len(parts) != 5:
                print(f"Error in {file_path} at line {line_number}: Expected 5 elements, got {len(parts)}")
                return False
            # Check if all parts can be converted to float
            try:
                float_values = [float(value) for value in parts]
            except ValueError:
                print(f"Error in {file_path} at line {line_number}: Non-numeric value found")
                return False
            # Optional: You could add range checks here if needed
    return True

def check_all_labels(label_directory):
    errors_found = False
    for root, dirs, files in os.walk(label_directory):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                if not check_label_format(file_path):
                    errors_found = True
    if not errors_found:
        print("All label files are correctly formatted.")
    else:
        print("Some label files have formatting issues.")

# Replace with the path to your labels directory
label_directory = '/Users/praneethboddu/Downloads/dataset/labels/train'
check_all_labels(label_directory)
