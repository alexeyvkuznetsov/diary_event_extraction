import csv
import re
import os

def fix_multiline_csv(input_filepath, output_filepath):
    """
    Reads a CSV file with multiline fields that might be incorrectly formatted,
    reassembles the rows, and writes a clean, correctly formatted CSV file.

    Args:
        input_filepath (str): The path to the malformed input CSV file.
        output_filepath (str): The path where the corrected CSV file will be saved.
    """
    print(f"Starting to process '{input_filepath}'...")

    # Regular expression to detect the start of a new logical row (e.g., "1;", "10;")
    # ^     - matches the beginning of the string
    # \d+   - matches one or more digits
    # ;     - matches the semicolon delimiter
    start_of_row_pattern = re.compile(r'^\d+;')

    corrected_rows = []
    current_row_lines = []
    header = []

    try:
        with open(input_filepath, 'r', encoding='utf-8') as infile:
            # Read header separately
            header_line = infile.readline().strip()
            if header_line:
                header = [h.strip() for h in header_line.split(';')]
            else:
                print("Error: Input file appears to be empty or has no header.")
                return

            # Process the rest of the file
            for line in infile:
                # Check if the line looks like the beginning of a new row
                if start_of_row_pattern.match(line):
                    # If the buffer for the previous row is not empty, process it
                    if current_row_lines:
                        # Join all collected lines for the previous row
                        full_row_str = "".join(current_row_lines).strip()
                        # Split only on the first two delimiters to keep the chunk intact
                        parts = full_row_str.split(';', 2)
                        if len(parts) == 3:
                            corrected_rows.append(parts)

                    # Start a new buffer with the current line
                    current_row_lines = [line]
                else:
                    # If it's not a new row, it's a continuation of the current chunk
                    current_row_lines.append(line)

            # Process the very last row remaining in the buffer after the loop
            if current_row_lines:
                full_row_str = "".join(current_row_lines).strip()
                parts = full_row_str.split(';', 2)
                if len(parts) == 3:
                    corrected_rows.append(parts)

    except FileNotFoundError:
        print(f"Error: The file '{input_filepath}' was not found.")
        return
    except Exception as e:
        print(f"An unexpected error occurred while reading the file: {e}")
        return

    # Now, write the cleaned data to the output file using the csv module
    print(f"Found {len(corrected_rows)} logical rows. Writing to '{output_filepath}'...")
    try:
        with open(output_filepath, 'w', newline='', encoding='utf-8') as outfile:
            writer = csv.writer(outfile, delimiter=';', quoting=csv.QUOTE_MINIMAL)

            # Write the header
            if header:
                writer.writerow(header)

            # Write the processed rows
            for row_parts in corrected_rows:
                # Clean up each part: strip leading/trailing whitespace and quotes
                cleaned_row = [part.strip().strip('"') for part in row_parts]
                writer.writerow(cleaned_row)

    except Exception as e:
        print(f"An error occurred while writing the file: {e}")
        return

    print("-" * 20)
    print(f"Successfully fixed the file.")
    print(f"Corrected data saved to: '{os.path.abspath(output_filepath)}'")
    print("-" * 20)


# --- HOW TO USE ---
# 1. Save this script as a Python file (e.g., `fix_csv.py`).
# 2. Place the `kapustin.csv` file in the same directory as the script.
# 3. Run the script from your terminal: python fix_csv.py

if __name__ == "__main__":
    input_filename = 'kapustin.csv'
    output_filename = 'kapustin_corrected.csv'
    fix_multiline_csv(input_filename, output_filename)