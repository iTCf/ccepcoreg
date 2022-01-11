import os.path


def make_subj_table(fname_results_table, dir_base):
    import pandas as pd
    import os.path as op
    res_tab = pd.read_csv(fname_results_table)

    res_tab = res_tab[['subj', 'fname', 'stim_int', 'stim_dur', 'stim_freq',
                       'stim_cont', 'stim_angle_cond',
                       'stim_gw_cond']]

    subj_count = res_tab.subj.astype('category').cat.codes.values+1
    subj_id = ['sub-%s' % str(s).zfill(2) for s in subj_count]
    res_tab['subj_id'] = subj_id

    run_count = []
    v = 1
    for ix_s, s in enumerate(subj_count):
        if ix_s == 0:
            v = 1
        else:
            if subj_count[ix_s -1] == s:
                v = v + 1
            else:
                v = 1
        run_count.append(v)
    run_id = ['run-%s' % str(s).zfill(2) for s in run_count]
    res_tab['run_id'] = run_id
    fname_save = op.join(dir_base, 'runs_table.csv')
    res_tab.to_csv(fname_save, index=False)


def make_runs_table(dir_base):
    import pandas as pd
    import os
    import natsort
    import re

    dir_data = os.path.join(dir_base, 'data', 'fif')
    files = os.listdir(dir_data)
    files = natsort.natsorted(files)

    subj_id, subj_code, run_id, fname = [], [], [], []

    for f in files:
        spl = f.split('_')
        subj_id.append(re.findall(r'\d', spl[0])[0].zfill(2))
        subj_code.append(spl[0])
        run_id.append('%s_%s' % (spl[1], spl[2]))
        fname.append(os.path.join(dir_data, f))
    runs_table = pd.DataFrame({'subj_id': subj_id, 'subj_code': subj_code,
                               'run_id': run_id, 'fname': fname})
    runs_table['run_nr'] = runs_table.groupby('subj_id').cumcount() + 1

    fname_save = os.path.join(dir_base, 'share', 'runs_table.csv')
    runs_table.to_csv(fname_save, index=False)


def make_coordsys_ieeg_json(dir_json_ieeg, subj_id, task):
    import os
    import json

    for system in ['T1w', 'MNI152NLin2009aSym']:
        coordys = {"iEEGCoordinateUnits": "m"}

        if 'T1w' in system:
            coordys["iEEGCoordinateSystem"] = 'Other'
            coordys["iEEGCoordinateSystemDescription"] = 'Native MRI space'
            coordys['iEEGCoordinateProcessingReference'] = 'Narizzano, M., Arnulfo, G., Ricci, S., Toselli, B., Tisdall, M., Canessa, A., Fato, M.M., Cardinale, F., 2017. SEEG assistant: a 3DSlicer extension to support epilepsy surgery. BMC Bioinformatics 18. https://doi.org/10.1186/s12859-017-1545-8'
            if 'maskface' in system:
                coordys['IntendedFor'] = '/derivatives/%s/epochs/anat/%s_%s.nii' % (subj_id, subj_id, system)

            else:
                coordys['IntendedFor'] = '/%s/anat/%s_%s.nii' % (subj_id, subj_id, system)

        elif 'MNI' in system:
            coordys['iEEGCoordinateSystem'] = 'ICBM 2009a Nonlinear Symmetric'
            coordys['iEEGCoordinateProcessingReference'] = 'Narizzano, M., Arnulfo, G., Ricci, S., Toselli, B., Tisdall, M., Canessa, A., Fato, M.M., Cardinale, F., 2017. SEEG assistant: a 3DSlicer extension to support epilepsy surgery. BMC Bioinformatics 18. https://doi.org/10.1186/s12859-017-1545-8 /// Avants, B.B., Tustison, N.J., Song, G., Cook, P.A., Klein, A., Gee, J.C., 2011. A reproducible evaluation of ANTs similarity metric performance in brain image registration. NeuroImage 54, 2033–2044. https://doi.org/10.1016/j.neuroimage.2010.09.025'
            coordys['iEEGCoordinateProcessingDescription'] = 'Skull stripping and SyN Registration (ANTs) to ICBM152'

        fpath_json_sys = os.path.join(dir_json_ieeg, '%s_task-%s_space-%s_coordsystem.json' % (subj_id, task, system))
        with open(fpath_json_sys, 'w') as f:
            json.dump(coordys, f, indent=4)


def make_coordsys_eeg_json(dir_coordsys_eeg_json, fname_dig, subj_id, task):
    import os.path as op
    import json
    from mne.viz._3d import _fiducial_coords
    import pandas as pd

    dig = pd.read_csv(fname_dig, delim_whitespace=True, names=['kind', 'ch_name', 'x', 'y', 'z'])

    coordys = {"EEGCoordinateSystem": "T1w", "EEGCoordinateUnits": "mm", "AnatomicalLandmarkCoordinates": {}}

    coordys['AnatomicalLandmarkCoordinates']['LPA'] = [float(n) for n in dig.loc[dig.ch_name == '1', ['x', 'y', 'z']].values.squeeze()]  # has to be float and not np.float to be able to json dump
    coordys['AnatomicalLandmarkCoordinates']['NAS'] = [float(n) for n in dig.loc[dig.ch_name == '2', ['x', 'y', 'z']].values.squeeze()]
    coordys['AnatomicalLandmarkCoordinates']['RPA'] = [float(n) for n in dig.loc[dig.ch_name == '3', ['x', 'y', 'z']].values.squeeze()]

    coordys['AnatomicalLandmarkCoordinateSystem'] = "T1w"
    coordys['IntendedFor'] = '/%s/anat/%s_T1.nii' % (subj_id, subj_id)
    coordys['AnatomicalLandmarkCoordinateUnits'] = "mm"

    fname_save = op.join(dir_coordsys_eeg_json, '%s_task-%s_coordsystem.json' % (subj_id, task))
    with open(fname_save, 'w') as f:
        json.dump(coordys, f, indent=4)


def make_electrodes_ieeg_tsv(dir_electrodes_ieeg_tsv, subj_id, seeg_coords, taskname):
    import pandas as pd
    import os.path as op

    for system in ['T1w', 'MNI152NLin2009aSym']:
        if 'T1w' in system:
            elec = pd.DataFrame({'name': seeg_coords.name, 'x': seeg_coords.x_mri/1e3, 'y': seeg_coords.y_mri/1e3,
                                 'z': seeg_coords.z_mri/1e3})
        elif 'MNI' in system:
            elec = pd.DataFrame({'name': seeg_coords.name, 'x': seeg_coords.x_norm_mri/1e3, 'y': seeg_coords.y_norm_mri/1e3,
                                 'z': seeg_coords.z_norm_mri/1e3})
        else:
            elec = pd.DataFrame({'name': seeg_coords.name, 'x': seeg_coords.x_surf/1e3, 'y': seeg_coords.y_surf/1e3,
                                 'z': seeg_coords.z_surf/1e3})
        elec['size'] = 7.5
        elec['manufacturer'] = 'Dixi Medical'
        elec['material'] = 'PtIr'
        elec.sort_values(['name'])
        elec = elec.round(5)
        fname_save = op.join(dir_electrodes_ieeg_tsv, '%s_task-%s_space-%s_electrodes.tsv' % (subj_id, taskname, system))
        elec.to_csv(fname_save, sep='\t', index=False)
    print('Done creating electrodes.tsv')


def make_electrodes_eeg_tsv(dir_electrodes_eeg_tsv, fname_dig, subj_id, task):
    import pandas as pd
    import os.path as op
    from info import ch185

    dig = pd.read_csv(fname_dig, delim_whitespace=True, names=['kind', 'ch_name', 'x', 'y', 'z'])
    dig = dig.loc[dig.kind == 'eeg']
    dig = dig.loc[dig.ch_name.isin(ch185)]

    elec = dig[['ch_name', 'x', 'y', 'z']]
    elec.columns = ['name', 'x', 'y', 'z']
    elec['material'] = 'HydroCel CleanLeads'
    elec = elec.round(2)

    fname_save = op.join(dir_electrodes_eeg_tsv, '%s_task-%s_electrodes.tsv' % (subj_id, task))
    elec.to_csv(fname_save, sep='\t', index=False)


def make_events_json(fname_events_json):
    import json
    events_json = {'electrical_stimulation_site': 'Electrodes involved in the stimulation',
                   'electrical_stimulation_current': 'Stimulation current (A)',
                   'electrical_stimulation_frequency': 'Frequency of stimulation (Hz)',
                   'electrical_stimulation_type': 'Kind of wave'}

    with open(fname_events_json, 'w') as f:
        json.dump(events_json, f, indent=4)


def export_data_bids(fpath_epo, dir_out, task, run, subject_id, event_id, r):
    import mne
    from mne_bids import make_bids_basename
    from mne_bids.write import _participants_json, _participants_tsv
    from itcfpy.read_write import load_scoreg
    import os.path as op
    import pandas as pd
    import numpy as np
    import json

    bids_basename = f'{subject_id}_task-{task}_{run}'
    print(bids_basename)

    # epo = mne.read_epochs(fpath_epo)
    for k in ['HDEEG', 'SEEG_bipolar']:
        kind = 'eeg' if k == 'HDEEG' else 'ieeg'
        epo = load_scoreg(fpath_epo, kind=k)
        if epo is None:
            print(f'FAILED {kind}')
            fname_log = op.join(dir_out, 'derivatives', 'epochs',
                                subject_id, kind,
                                bids_basename + '_epochs.FAILED')
            with open(fname_log, 'w') as f:
                f.write(fpath_epo)
            return
        epo = epo.crop(-0.3, 0.7)

        desc_spl = os.path.split(fpath_epo)[-1].split('_')
        desc = []
        for i in [1, 3, 4, 5]:
            v = desc_spl[i]
            if i > 1 & (len(v) > 3):
                v = v.replace('0', '0.')  # add decimal
            desc.append(v)

        # participants
        fname_participants = op.join(dir_out, 'participants')
        _participants_tsv(epo, subject_id.replace('sub-', ''), fname_participants + '.tsv', verbose=False)
        _participants_json(fname_participants + '.json', overwrite=True, verbose=False)

        # channels tsv
        ch_names = epo.ch_names.copy()
        if 'STI' in ch_names:
            ch_names.remove('STI')
        status = ['good' if c not in epo.info['bads'] else 'bad' for c in ch_names]

        chans_eeg = {'name': ch_names, 'type': [kind]*len(ch_names),
                     'units': ['V']*len(ch_names), 'low_cutoff': ['0.5']*len(ch_names),
                     'high_cutoff': ['45']*len(ch_names),
                     'sampling_frequency': [1000]*len(ch_names), 'status': status,
                     'reference': ['average']*len(ch_names)}

        chans_ieeg = {'name': ch_names, 'type': [kind]*len(ch_names),
                      'units': ['V']*len(ch_names), 'low_cutoff': ['0.5']*len(ch_names),
                      'high_cutoff': ['300']*len(ch_names),
                      'sampling_frequency': [1000]*len(ch_names), 'status': status,
                      'reference': ['bipolar']*len(ch_names)}

        chans = chans_eeg if kind == 'eeg' else chans_ieeg
        chans_tsv = pd.DataFrame(chans)
        fname_chans_tsv = op.join(dir_out, 'derivatives', 'epochs', subject_id,
                                  kind, bids_basename + '_channels.tsv')
        chans_tsv.to_csv(fname_chans_tsv, index=False, sep='\t')

        # epochs.tsv
        stim_ch = desc_spl[1]
        n_epo = len(epo)
        duration = np.abs(epo.times[0]) + epo.times[-1]
        tr_type = '%s %s %s %s %s %s' % (desc[0], desc[1], desc[2], desc[3], r.stim_angle_cond, r.stim_gw_cond)
        epo_tsv = pd.DataFrame({'duration': [duration] * n_epo,
                                'zero_time': [np.abs(epo.times[0])] * n_epo,
                                'trial_type': [tr_type] * n_epo})

        fname_epo_tsv = op.join(dir_out, 'derivatives', 'epochs',
                                subject_id, kind,
                                bids_basename + '_epochs.tsv')
        epo_tsv.to_csv(fname_epo_tsv, index=False, sep='\t')

        # epochs json
        epo_json = {'Description': f'Stimulation of {tr_type} (channel, intensity, duration, frequency, angle, grey/white matter)',
                    'Sources': '/%s/%s/%s_%s.npy' % (subject_id, kind, bids_basename, kind),
                    'BaselineCorrection': True,
                    'BaselineCorrectionMethod': 'mean subtraction',
                    'BaselinePeriod': [-0.3, 0]}

        fname_epo_json = op.join(dir_out, 'derivatives', 'epochs',
                                 subject_id, kind,
                                 bids_basename + '_epochs.json')

        with open(fname_epo_json, 'w') as f:
            json.dump(epo_json, f, indent=4)

        # data
        dat = epo.get_data()  # check omit trigger
        fname_dat = op.join(dir_out, 'derivatives', 'epochs',
                            subject_id, kind,
                            bids_basename + '_epochs.npy')
        np.save(fname_dat, dat)


def make_mri_json(subj_id, dir_out):
    import os.path as op
    import json
    fname_anonymi_json = op.join(dir_out, subj_id, 'anat', '%s_T1w.json' % subj_id)
    anonymi_info = {'ImageProcessingSoftware': "AnonyMI - Mikulan, E., Russo, S., Zauli, F.M., d’Orio, P., Parmigiani,"
                                              " S., Favaro, J., Knight, W., Squarza, S., Perri, P., Cardinale, F., "
                                              "Avanzini, P., Pigorini, A., 2021. A comparative study between "
                                              "state-of-the-art MRI deidentification and AnonyMI, a new method "
                                              "combining re-identification risk reduction and geometrical preservation."
                                              " Human Brain Mapping 42, 5523–5534. https://doi.org/10.1002/hbm.25639 -"
                                              "https://github.com/iTCf/anonymi"}

    with open(fname_anonymi_json, 'w') as f:
        json.dump(anonymi_info, f, indent=4)


def load_bids(dir_bids, subj_id, task, run_id, kind='eeg'):
    import mne
    import os.path as op
    import numpy as np
    import pandas as pd
    import json

    bids_fname_base = op.join(dir_bids, 'derivatives', 'epochs', subj_id, kind,
                              '%s_task-%s_%s' % (subj_id, task,  run_id))

    fname_eeg = bids_fname_base + '_epochs.npy'
    fname_chans = bids_fname_base + '_channels.tsv'

    if kind == 'eeg':
        fname_elecs = bids_fname_base.replace(run_id, '') + 'electrodes.tsv'
        fname_coordsys = bids_fname_base.replace(run_id, 'coordsystem.json')
    elif kind == 'ieeg':
        fname_elecs = bids_fname_base.replace(run_id, '') + 'space-T1w_electrodes.tsv'
        fname_coordsys = bids_fname_base.replace(run_id, 'space-T1w_coordsystem.json')
    else:
        print('Unknown kind')

    with open(fname_coordsys) as json_file:
        coordsys = json.load(json_file)

    data = np.load(fname_eeg)
    chans = pd.read_csv(fname_chans, sep='\t')
    ch_names = chans.name.tolist()
    elecs = pd.read_csv(fname_elecs, sep='\t')

    ch_types = ['eeg']*len(chans) if kind == 'eeg' else ['seeg']*len(chans)
    info = mne.create_info(ch_names, sfreq=1000,  # todo: srate from bids file
                           ch_types=ch_types)
    epo = mne.EpochsArray(data, info, tmin=-0.3)  # todo: tmin from bids file

    if kind == 'eeg':
        dig_ch_pos = dict(zip(ch_names, elecs[['x', 'y', 'z']].values))
        fiducials = coordsys['AnatomicalLandmarkCoordinates']
        mont = mne.channels.make_dig_montage(dig_ch_pos, nasion=fiducials['NAS'],
                                             rpa=fiducials['RPA'], lpa=fiducials['LPA'],
                                             coord_frame='head')
        epo.set_montage(mont)

    ch_status = chans.status.tolist()
    bads = [c for c, s in zip(ch_names, ch_status) if s == 'bad']
    epo.info['bads'] = bads
    epo.baseline = (-0.3, 0)  # todo: baseline from bids file
    return epo


def load_trans(fname_trans):
    import h5py
    import mne

    trans_ori = h5py.File(fname_trans).get('trans')[()]
    trans = mne.transforms.Transform(fro='head', to='mri', trans=trans_ori)
    return trans


def copy_mris(fname_runs, fname_codes, dir_fs, dir_save):
    import pandas as pd
    import shutil
    import os.path as op

    runs_table = pd.read_csv(fname_runs)
    subj_codes = pd.read_csv(fname_codes)

    subjects = runs_table.subj.unique()

    for s in subjects:
        subj_name = subj_codes.loc[subj_codes.code == s, 'subj'].values[0].upper()
        subj_id = runs_table.loc[runs_table.subj == s, 'subj_id'].values[0]
        fname_mri = op.join(dir_fs, subj_name, 'mri', 'T1.mgz')
        fname_save = op.join(dir_save, f'{subj_id}_T1.mgz')
        print(f'saving{fname_mri} as {fname_save}:')
        shutil.copy(fname_mri, fname_save)
        print('\n')


