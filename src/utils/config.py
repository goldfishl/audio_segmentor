import os

# acupoint dataset config
acup_config = {
   'root_path' : os.path.join('data', 'acupoint'),
    'split_rate' : [0.7, 0.15, 0.15],
    'split_files' : {},
}

acup_config['data_path'] = os.path.join(acup_config['root_path'], 'data')
acup_config['split_files']['train'] = os.path.join(acup_config['root_path'], 'train.txt')
acup_config['split_files']['valid'] = os.path.join(acup_config['root_path'], 'val.txt')
acup_config['split_files']['test'] = os.path.join(acup_config['root_path'], 'test.txt')
acup_config['label_file'] = os.path.join(acup_config['root_path'], 'label.txt')


# signal process config
signal_config = {
    'sample_rate' : 16000,  # Hz
    'frame_length' : 25,  # ms
    'frame_shift' : 10,  # ms
    'window_type' : 'hanning',  # ‘hamming’|’hanning’|’povey’|’rectangular’|’blackman’
    'compliance' : 'kaldi',  # 'kaldi' | 'torchaudio'
}


# generation dataset config
gener_config = {
    'root_path' : os.path.join('data', 'generation'),
    'split_rate' : [0.7, 0.15, 0.15],
    'split_files' : {},
}

gener_config['data_path'] = os.path.join(gener_config['root_path'], 'data')
gener_config['split_files']['train'] = os.path.join(gener_config['root_path'], 'train.txt')
gener_config['split_files']['valid'] = os.path.join(gener_config['root_path'], 'val.txt')
gener_config['split_files']['test'] = os.path.join(gener_config['root_path'], 'test.txt')

# pres_file is the file that contains all the prescriptions
gener_config['pres_file'] = os.path.join(gener_config['root_path'], 'Cleaned.txt')
# miss_file is the file that contains the acupoints that are not in the prescription
gener_config['miss_file'] = os.path.join(gener_config['root_path'], 'Missing.txt')

# random setting for the gap time between two acupoints audio
# Depending on speaker style, the maximum is randomly 
# sampled from [min_gap_time, max_gap_time]
gener_config['min_gap_time'] = 200 # ms
gener_config['max_gap_time'] = 1000 # ms

# random setting for the intro/outro time of generated audio
gener_config['min_fade_time'] = 0 # ms
gener_config['max_fade_time'] = 200 # ms

# maximum number of permutations for each prescription
gener_config['max_permut'] = 10
# number of times to repeat each prescription
gener_config['repeat'] = 2



os.makedirs(gener_config['root_path'], exist_ok=True)
os.makedirs(gener_config['data_path'], exist_ok=True)