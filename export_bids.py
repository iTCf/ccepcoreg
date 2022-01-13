import mne
import os
import pandas as pd
import shutil
from fx_bids import make_coordsys_ieeg_json, make_electrodes_ieeg_tsv, \
    make_coordsys_eeg_json, make_electrodes_eeg_tsv, export_data_bids, \
    make_events_json, make_mri_json

dir_base = ''
dir_anon = ''
dir_dig = ''
dir_dat = ''
runs_table = pd.read_csv(os.path.join(dir_base, 'runs_table.csv'))
subj_codes = pd.read_csv('')
all_bip_coords = pd.read_csv('')

task = 'ccepcoreg'
event_id = {'stim': 1}

dir_out = os.path.join(dir_base, 'gnode', 'ccepcoreg')
dir_spatial = os.path.join(dir_base, 'spatial')  # todo: check

subjects = runs_table.subj_id.unique()

# create folder structure and populate it
folders_epo = ['eeg', 'ieeg']

for s in subjects:
    subj_code = runs_table.loc[runs_table.subj_id == s, 'subj'].iloc[0]
    subj_runs = runs_table.loc[runs_table.subj_id == s]
    subj_name = subj_codes.loc[subj_codes.code == subj_code, 'subj'].values[0]

    # folders
    _ = [os.makedirs(os.path.join(dir_out, 'derivatives', 'epochs',
                                  s, folder)) for folder in folders_epo]

    os.makedirs(os.path.join(dir_out, s, 'anat'))

    # MRIs
    shutil.copy(os.path.join(dir_anon, f'{s}_T1_anonymi.nii'),
               os.path.join(dir_out, s, 'anat',
                            '%s_T1w.nii' % s))

    # seeg coordsystem
    dir_json_ieeg = os.path.join(dir_out, 'derivatives', 'epochs',
                                 s, 'ieeg')
    make_coordsys_ieeg_json(dir_json_ieeg, s, task)

    # ieeg electrodes
    ch_info = all_bip_coords.loc[all_bip_coords.code == subj_code]
    make_electrodes_ieeg_tsv(dir_json_ieeg, s, ch_info, task)

    # eeg coordsystem
    fname_dig = os.path.join(dir_dig, f'{subj_name}_egi_dig.hpts')

    dir_coordsys_eeg = os.path.join(dir_out, 'derivatives', 'epochs',
                                    s, 'eeg')
    make_coordsys_eeg_json(dir_coordsys_eeg, fname_dig, s, task)

    # eeg electrodes
    make_electrodes_eeg_tsv(dir_coordsys_eeg, fname_dig, s, task)

    # events json
    fname_events_json = os.path.join(dir_out, 'derivatives', 'epochs',
                                     s, 'eeg', '%s_task-%s_events.json' % (s, task))
    # make_events_json(fname_events_json) # FIX, not needed

    # mri json
    make_mri_json(s, dir_out)


for ix, r in runs_table.iterrows():
    fpath = os.path.join(dir_dat, r.fname)
    export_data_bids(fpath, dir_out, task, r.run_id, r.subj_id,
                     event_id, r)

# scans
for s in subjects:
    all_scans = []
    for k in ['eeg', 'ieeg']:
        subj_scans = os.listdir(os.path.join(dir_out, 'derivatives', 'epochs',
                                             s, k))
        subj_scans = [os.path.join(k, scan) for scan in subj_scans if '.npy' in scan]
        all_scans.extend(subj_scans)
    all_scans = pd.DataFrame({'filename': all_scans})
    all_scans.sort_values('filename', inplace=True)
    fname_scans = os.path.join(dir_out, 'derivatives', 'epochs',
                               s,  '%s_scans.tsv' % s)
    all_scans.to_csv(fname_scans, index=False, sep='\t')
