# This code produced by me with frame from Claude and ChatGPT

import pandas as pd

def parse_log_files(file_path, milestone_path, output_csv="parsed_data.csv"):
        
    print(f"Processing: {file_path}")

    df = pd.read_csv(file_path)

    print(f"Processing: {milestone_path}")

    milestones = []
    with open(milestone_path, 'r', encoding='utf-8') as milestone_file:
        for line in milestone_file:
            milestones.append(line.strip().split(" || "))

    print(milestones)


    seen_milestones = {}
    output_rows = []
    
    for _, row in df.iterrows():
        student = row['student']
        command = row['command']

        is_milestone = False
        for i, milestone in enumerate(milestones):
            if command in milestone:
                index = i
                is_milestone = True
                achieved_milestone = str(milestone)
                break

        if not is_milestone: continue 

        if student not in seen_milestones:
            seen_milestones[student] = set()

        if achieved_milestone in seen_milestones[student]:
            continue

        seen_milestones[student].add(achieved_milestone)
        
        output_rows.append({
            'student': student,
            'milestone': index,
            'timestamp': row['timestamp'],
            'attempt': row['attempt']
        })


    milestone_df = pd.DataFrame(output_rows)
    milestone_df.to_csv(output_csv, index=False)

    print("Complete!")
    


# Example usage
if __name__ == "__main__":
    # Change this to your folder path
    file_path = "parsing/file_wrangler_second_pass_results.csv"
    milestone_path = "parsing/milestones.txt"

    # Parse the files and create CSV
    parse_log_files(file_path, milestone_path, "file_wrangler_third_pass_results.csv")