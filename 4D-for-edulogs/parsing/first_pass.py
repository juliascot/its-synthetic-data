# This code produced by Claude

import csv
import re
from pathlib import Path

def parse_log_files(folder_path, output_csv="parsed_data.csv"):
    """
    Parse log files in the format annotator.[name].[number] and extract INPUT| lines
    
    Args:
        folder_path (str): Path to folder containing log files
        output_csv (str): Name of output CSV file
    """
    
    # List to store all parsed data
    parsed_data = []
    
    # Get all files matching the pattern annotator.*.*
    folder = Path(folder_path)
    log_files = list(folder)
    
    print(f"Found {len(log_files)} files to process")
    
    for file_path in log_files:
        print(f"Processing: {file_path.name}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                for line_num, line in enumerate(file, 1):
                    # Strip whitespace and check if line starts with INPUT|
                    line = line.strip()
                    if line.startswith("INPUT|"):
                        
                        # Split the line by | delimiter
                        parts = line.split("|")
                        
                        # We need at least 7 parts to have all required fields
                        # INPUT|name|commands|timestamp|user|path|StartingLine:command|%
                        if len(parts) >= 7:
                            name = parts[1]  # Second part is the name
                            timestamp = parts[3]  # Fourth part is timestamp
                            command_part = parts[6]  # Seventh part contains the command
                            
                            # Extract command after "StartingLine:" or "gateway:"
                            command = extract_command(command_part)
                            
                            if command:  # Only add if we successfully extracted a command
                                parsed_data.append({
                                    'name': name,
                                    'timestamp': timestamp,
                                    'command': command
                                })
                        else:
                            print(f"  Warning: Line {line_num} has insufficient parts: {len(parts)}")
                            
        except Exception as e:
            print(f"  Error processing {file_path.name}: {e}")
    
    # Write to CSV
    if parsed_data:
        with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['name', 'timestamp', 'command']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            writer.writerows(parsed_data)
        
        print(f"\nSuccessfully wrote {len(parsed_data)} records to {output_csv}")
    else:
        print("No data found to write to CSV")

def extract_command(command_part):
    """
    Extract command from the command part of the log line
    Looks for text after "file_wrangler:" and before the next "|"
    
    Args:
        command_part (str): The part of the line containing the command
        
    Returns:
        str: Extracted command or None if not found
    """
    
    # Pattern to match "StartingLine:" or "gateway:" followed by the command
    # The command continues until we hit a "|" or end of string
    pattern = r'(?:file_wrangler:)([^|]*)'
    
    match = re.search(pattern, command_part)
    if match:
        command = match.group(1).strip()  # Get the captured group and strip whitespace
        return command if command else None
    
    return None

# Example usage
if __name__ == "__main__":
    # Change this to your folder path
    folder_path = ""

    # Parse the files and create CSV
    parse_log_files(folder_path, "student_commands.csv")