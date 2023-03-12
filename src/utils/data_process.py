from .config import acup_config, gener_config
import os


def get_acupoint_data():
    """
    Returns a dictionary of data, where the keys are labels and the values are lists of corresponding .wav files.
    Assumes that each .wav file has a corresponding .txt file with the same name (excluding extension) in the same directory.

    Returns:
    data (dict): A dictionary where the keys are labels and the values are lists of corresponding .wav files.
    """

    # Get a list of all files in the data directory
    all_files = os.listdir(acup_config['data_path'])

    # Filter the list to only include .wav files
    wav_files = [file for file in all_files if file.endswith('.wav')]

    # Create a list of corresponding label file names by replacing the .wav extension with .txt
    label_files = [os.path.splitext(file)[0] + '.txt' for file in wav_files]

    # Create a dictionary to store the data, with labels as keys and lists of corresponding .wav files as values
    data = {}
    for label_file, wav_file in zip(label_files, wav_files):
        with open(os.path.join(acup_config['data_path'], label_file), 'r') as label_file_obj:
            # Read the label from the label file
            label = label_file_obj.read().strip()

            # Add the .wav file to the list of files for this label in the data dictionary
            if label not in data:
                data[label] = [wav_file]
            else:
                data[label].append(wav_file)

    # Return the data dictionary
    return data
        

def get_pres_formual():
    """
    Reads a prescription formula file and returns a list of entries, where each entry is a list of required items and an optional list of items.

    The prescription formula file should have one entry per line, where each line consists of the required items followed by an optional list of items, separated by one or more spaces. Each item is a string, and the required and optional items are separated by a single space.

    Empty lines and lines starting with '#' are ignored.

    Returns:
    entries (list): A list of prescription formula entries, where each entry is a list containing two sublists. The first sublist contains the required items, and the second sublist contains the optional items. Each item is a string.
    """
    required_items = []
    optional_items = []
    entries = []

    # Read the prescription file line by line
    with open(gener_config['pres_file'], "r") as infile:
        for line in infile:
            # Ignore lines starting with '#' or empty lines (unless this is the first line of an entry)
            if line[0] == "#":
                continue
            if line[0] == "\n" and len(required_items) != 0:
                # Add the current entry to the list of entries and start a new entry
                entries.append([required_items, optional_items])
                required_items = []
                optional_items = []
            elif line[0] == "\n":
                continue
            elif line[0] == " ":
                # This line contains optional items - add them to the list
                optional_items.append(line.replace(" ", "").strip().split("、"))
            else:
                # This line contains required items - add them to the list
                required_items = line.strip().split("、")
                if required_items == [""]:
                    print("Warning, line:", line)

    return entries


def generate_missing_word_file(prescription, acup_data):
    """
    Generates a list of missing words by comparing the prescription entries to the available acupoint data.

    Args:
    prescription (list): A list of prescription entries, where each entry is a list of required and optional items.
    acup_data (dict): A dictionary of acupoint data, where the keys are acupoint names and the values are lists of .wav files.
    Writes the list of missing words to a file specified in the gener_config dictionary.

    Each missing word is written on a separate line.

    Returns:
    None
    """
    # Generating a list of missing words
    # There's not a lot of them, I've noted some
    # observations down in Clean.txt
    missing_words = []
    for required_items, optional_items in prescription:
        for item in required_items:
            if item not in acup_data:
                if item not in missing_words:
                    missing_words.append(item)
        for term in optional_items:
            for item in term:
                if item not in acup_data:
                    if item not in missing_words:
                        missing_words.append(item)

    # Write the list of missing words to a file
    with open(gener_config['miss_file'], "w") as out_file:
        out_file.write("\n".join(missing_words + [""]))