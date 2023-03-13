import random
import itertools
import os
from tqdm import tqdm
import numpy as np
import torchaudio
import torch

from src.utils import generate_gap_audio
from .config import gener_config, signal_config, acup_config
from .data_process import get_acupoint_data, get_pres_formual, generate_missing_word_file


def get_all_prescriptions(formulas, acupoint_data):
    """
    This function takes a list of formulas as input and returns a list of all possible prescriptions.
    A prescription is formed by combining the required items from the chosen formula with one of the optional items, 
    if any, from the formula's optional list.

    Args:
    formulas (list): A list of formulas where each formula is represented by a tuple containing the required_items 
                     and optional_item_lists.

    Returns:
    list: A list of all possible prescriptions formed by combining the required and optional items.
    """

    all_prescriptions = []

    for formula in formulas:

        required_items, optional_items = formula

        # Filter required and optional items based on available acupoint data
        required_items = [item for item in required_items if item in acupoint_data]
        optional_items = [[item for item in opt  if item in acupoint_data] for opt in optional_items]

        # Start with the required items
        prescriptions = [required_items]

        # For each optional item list, add an optional item to the required items to form a new prescription
        for optional_item in optional_items:
            new_prescription = required_items + optional_item
            prescriptions.append(new_prescription)

        all_prescriptions.extend(prescriptions)

    return all_prescriptions



def generate_prescription_audio(pres, acupoint_data):
    """
    Generates audio data for each permutation of the prescription.

    Args:
        pres (list): A list of prescription items (acupoint names).
        acupoint_data (dict): A dictionary of acupoint data containing the audio files for each acupoint.
    Returns:
        tuple: A tuple containing the generated audio data as a NumPy array and the frame labels as a NumPy array.
    """
    # Intro/outtro time
    min_fade_time = gener_config["min_fade_time"]  # in ms
    max_fade_time = gener_config["max_fade_time"]  # in ms

    # Gap time
    min_gap_time = gener_config["min_gap_time"]
    max_gap_time = gener_config["max_gap_time"]
    max_gap_time = random.randrange(min_gap_time+1, max_gap_time)


    # Randomly sample the audio in acupoint_data for each item in the prescription
    try:
        audio_files = [random.choice(acupoint_data[item]) for item in pres]
    except:
        print(pres)
        z=input()
    acp_audio = [torchaudio.load(os.path.join(acup_config["data_path"], file))[0][0].numpy() for file in audio_files]

    # add intro audio
    intro_time = random.randrange(min_fade_time, max_fade_time)
    pres_audio = [generate_gap_audio(intro_time)]

    # Generate segmentation frame labels
    # In the segmentation task, each frame is assigned a label based on whether it is in a segment or not.
    # The label is 1 if the frame is in a segment and 0 otherwise.
    total_audio_length = np.concatenate(pres_audio).shape[0]
    frame_num = calculate_num_frames(total_audio_length)
    if frame_num > 0:
        frame_labels = np.zeros(frame_num, dtype=np.int8)
    else:
        frame_labels = np.array([])

    # Add audio for each item in the prescription
    for i, audio_item in enumerate(acp_audio):
        pres_audio.append(audio_item)

        # Add the frame labels for the current item
        total_audio_length = np.concatenate(pres_audio).shape[0]
        frame_num = calculate_num_frames(total_audio_length) - len(frame_labels)
        if frame_num > 0:
            frame_labels = np.concatenate((frame_labels, np.ones(frame_num, dtype=np.int8)))

        # Add a gap between each item
        if i < len(acp_audio) - 1:
            _time = random.randrange(min_gap_time, max_gap_time)
            pres_audio.append(generate_gap_audio(_time))

            # Add the frame labels for the gap
            total_audio_length = np.concatenate(pres_audio).shape[0]
            frame_num = calculate_num_frames(total_audio_length) - len(frame_labels)
            if frame_num > 0:
                frame_labels = np.concatenate((frame_labels, np.zeros(frame_num, dtype=np.int8)))


    # add outro audio
    _time = random.randrange(min_fade_time, max_fade_time)
    pres_audio.append(generate_gap_audio(_time))

    # Add the frame labels for the outro
    total_audio_length = np.concatenate(pres_audio).shape[0]
    frame_num = calculate_num_frames(total_audio_length) - len(frame_labels)
    if frame_num > 0:
        frame_labels = np.concatenate((frame_labels, np.zeros(frame_num, dtype=np.int8)))

    # Concatenate the audio
    pres_audio = np.concatenate(pres_audio)
    return (pres_audio, frame_labels, pres)



def calculate_num_frames(audio_length):
    """
    Calculates the number of frames of audio after signal processing features extraction (usually using fbank).

    Args:
        audio_length_samples (int): The length of the audio signal in samples.

    Returns:
        int: The number of frames after converting the audio signal to fbank.
    """
    # Convert frame length and shift from milliseconds to samples
    frame_length = int(signal_config["sample_rate"] * signal_config["frame_length"] / 1000)
    frame_shift = int(signal_config["sample_rate"] * signal_config["frame_shift"] / 1000)

    # Calculate number of frames
    num_frames = (audio_length - frame_length) // frame_shift

    # Return the number of frames
    return num_frames


if __name__ == "__main__":
    # Get the prescription formulas and acupoint data
    pres_formulas = get_pres_formual()
    acupoint_data = get_acupoint_data()

    generate_missing_word_file(pres_formulas, acupoint_data)


    # Get all possible prescriptions
    all_pres = get_all_prescriptions(pres_formulas, acupoint_data)

    # Maximum number of permutations to generate for each prescription
    max_permut = gener_config["max_permut"]

    # Generate the permutations of all prescriptions
    all_permuts = []
    for pres in all_pres:
        # Generate all permutations of the prescription
        single_item_permut = list(itertools.permutations(pres, len(pres)))
        random.shuffle(single_item_permut)
        if len(single_item_permut) > max_permut:
            single_item_permut = single_item_permut[:max_permut]

        all_permuts.extend(single_item_permut)

    all_gen = []
    # Generate the audio for each permutation
    for pres in tqdm(all_permuts):
        # Repeat each prescription
        for _ in range(gener_config['repeat']):
            all_gen.append(generate_prescription_audio(pres, acupoint_data))

    print(f'Generated {len(all_gen)} audio files.')

    # Save the generated data
    for i in tqdm(range(len(all_gen))):
        audio, seg_labels, labels = all_gen[i]

        # Save the audio
        torchaudio.save(os.path.join(gener_config["data_path"], f'{i}.wav'), 
                        torch.from_numpy(audio).unsqueeze(0), 
                        signal_config["sample_rate"])
        # Save the segmentation labels
        with open(f'{gener_config["data_path"]}/{i}.seg', 'w') as f:
            for label in seg_labels:
                f.write(str(int(label)))

        with open(f'{gener_config["data_path"]}/{i}.txt', 'w') as f:
            f.write('\t'.join(labels)+'\n')