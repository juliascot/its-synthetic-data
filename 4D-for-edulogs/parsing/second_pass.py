# This code produced by me with frame from Claude and ChatGPT

import pandas as pd

def parse_log_files(file_path, output_csv="parsed_data.csv"):
        
    print(f"Processing: {file_path}")

    df = pd.read_csv(file_path)


    # Prepare new columns
    student_ids = []
    attempts = []
    curr_id_counter = -1
    curr_attempt_counter = -1
    curr_id = ""

    # Iterate over each row
    for _, row in df.iterrows():
        student_id = row['name'][:6]
        if student_id == curr_id:
            curr_attempt_counter += 1
        else:
            curr_id = student_id
            curr_id_counter += 1
            curr_attempt_counter = 0
        
        student_ids.append(curr_id_counter)
        attempts.append(curr_attempt_counter)

    
    df['student'] = student_ids
    df['attempt'] = attempts

    df['timestamp'] = df.groupby('student')['timestamp'].transform(lambda x: x - x.min())

    df.pop('name')


    df.to_csv(output_csv, index=False)

    print("Complete!")
    


# Example usage
if __name__ == "__main__":
    # Change this to your folder path
    file_path = "parsing/file_wrangler_first_pass_results.csv"

    # Parse the files and create CSV
    parse_log_files(file_path, "student_commands.csv")