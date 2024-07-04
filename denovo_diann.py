import os

import pandas as pd

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import argparse
import numpy as np
from pyteomics import mgf, mass
from dataclasses import dataclass, asdict

import tensorflow as tf
import tensorflow.keras as k
from tensorflow_addons.layers import InstanceNormalization

from utils import *
from BasicClass import Ion
import pickle
import gzip


print(tf.__version__, tf.config.list_physical_devices('GPU'))


def read_spec(header, data_dir, count=-1, previous_loc=0):
    spectra = []

    chosen_header = header.iloc[previous_loc:]

    for peptide in chosen_header.iterrows():

        c = int(peptide['charge'])

        pep = str(peptide['mod_sequence'])

        mass = Ion.precursorion2mass(float(peptide['precursor_mz']), c)
        hcd = 0

        with open(os.path.join(data_dir, peptide['MSGP File Name']), 'rb') as f:
            f.seek(peptide['MSGP Datablock Pointer'])
            spec = pickle.loads(gzip.decompress(f.read(peptide['MSGP Datablock Length'])))

        related_ms1 = spec['related_ms1']
        related_ms2 = spec['related_ms2']

        mz_arrays = []
        intensity_arrays = []
        for i, row in related_ms2.iterrows():
            mzs = row['mz']
            intensities = row['intensity']

            mz_arrays.append(mzs)
            intensity_arrays.append(intensities)

        mz_arrays = np.concatenate(mz_arrays)
        intensity_arrays = np.concatenate(intensity_arrays)

        sorted_indices = np.argsort(mz_arrays)
        mz_arrays = mz_arrays[sorted_indices]
        intensity_arrays = intensity_arrays[sorted_indices]

        spectra.append({'pep': pep, 'charge': c, 'type': 3, 'nmod': 0, 'mod': np.zeros(len(pep), 'int32'),
                    'mass': mass, 'mz': mz_arrays, 'it': intensity_arrays, 'nce': hcd})

        if count > 0 and len(spectra) >= count:
            break

    return spectra

# post correction step
def post_correction(matrix, mass, c, ppm=10):
    positional_score = np.max(matrix, axis=-1)
    seq = decode(matrix)
    pep = topep(seq)
    seq = seq[:len(pep)]
    tol = mass * ppm / 1000000

    for i, char in enumerate(pep):
        if char in '*[]':
            pep = pep[:i]
            positional_score[i:] = 1
            seq = seq[:i]
            break

    if len(pep) < 1:
        return '', -1, positional_score

    msp = m1(topep(seq), c)
    delta = msp - mass
    pos = 0
    a = seq[0]

    if abs(delta) < tol:
        return topep(seq), -1, positional_score

    for i in range(len(seq) - 1):  # no last pos
        mi = mass_list[seq[i]]
        for j in range(1, 21):
            if j == 8:
                continue  # ignore 'I'

            d = msp - mass + (mass_list[j] - mi) / c

            if abs(d) < abs(delta):
                delta = d
                pos = i
                a = j

    if abs(delta) < tol:  # have good match
        candi = np.int32(seq == seq[pos])
        if np.sum(candi) > 1.5:  # ambiguis
            pos = np.argmin((1 - candi) * 10 + candi *
                            np.max(matrix[:len(seq)], axis=-1))

        seq[pos] = a
        positional_score[pos] = 1

        return topep(seq), pos, positional_score
    else:
        return topep(seq), -1, positional_score

# hyper parameter
@dataclass(frozen = True)
class hyper():
    lmax: int = 30
    outlen: int = lmax + 2
    m1max: int = 2048
    mz_max: int = 2048
    pre: float = 0.1
    low: float = 0
    vdim: int = int(mz_max / pre)
    dim: int = vdim + 0
    maxc: int = 8
    sp_dim: int = 4

    mode: int = 3
    scale: float = 0.3

# convert spectra into model input format
def input_processor(spectra):
    nums = len(spectra)

    inputs = config({
        'y': np.zeros([nums, hyper.sp_dim, hyper.dim], 'float32'),
        'info': np.zeros([nums, 2], 'float32'),
        'charge': np.zeros([nums, hyper.maxc], 'float32')
    })

    for i, sp in enumerate(spectra):
        mass, c, mzs, its = sp['mass'], sp['charge'], sp['mz'], sp['it']
        mzs = mzs / 1.00052

        its = normalize(its, hyper.mode)

        inputs.info[i][0] = mass / hyper.m1max
        inputs.info[i][1] = sp['type']
        inputs.charge[i][c - 1] = 1

        precursor_index = min(hyper.dim - 1, round((mass * c - c + 1) / hyper.pre))

        vectorlize(mzs, its, mass, c, hyper.pre, hyper.dim, hyper.low, 0, out=inputs.y[i][0], use_max=1)
        inputs.y[i][1][:precursor_index] = inputs.y[i][0][:precursor_index][::-1] # reverse it

        vectorlize(mzs, its, mass, c, hyper.pre, hyper.dim, hyper.low, 0, out=inputs.y[i][2], use_max=0)
        inputs.y[i][3][:precursor_index] = inputs.y[i][2][:precursor_index][::-1] # reverse mz

    return tuple([inputs[key] for key in inputs])

def denovo(model, spectra, batch_size):
    predict_peps = []
    scores = []
    positional_scores = []
    charges = [sp['charge'] for sp in spectra]
    peps = [sp['pep'] for sp in spectra]

    predictions = model.predict(data_seq(spectra, input_processor, batch_size, xonly=True), verbose=1)

    for rst, sp in zip(predictions, spectra):
        ms, c = sp['mass'], sp['charge']

        # run post correction
        pep, pos, positional_score = post_correction(rst, ms, c)

        predict_peps.append(pep)
        positional_scores.append(positional_score)
        scores.append(np.prod(positional_score))

    ppm_diffs = asnp32([ppm(sp['mass'], m1(pp, c)) for sp, pp, c in zip(spectra, predict_peps, charges)])
    return peps, predict_peps, scores, positional_scores, ppm_diffs, spectra


parser = argparse.ArgumentParser()
parser.add_argument('--input_header', type=str,
                    help='input file header', default='')
parser.add_argument('--input_folder', type=str,
                    help='input data folder', default='')
parser.add_argument('--output', type=str,
                    help='output file path', default='example.tsv')
parser.add_argument('--model', type=str,
                    help='Pretained model path', default='model.h5')
parser.add_argument('--loop_size', type=int,
                    help='number of spectra in memory', default=10000)
parser.add_argument('--batch_size', type=int,
                    help='number of spectra per step', default=128)

args = parser.parse_args()

print('Loading model....')
tf.keras.backend.clear_session()
model = k.models.load_model(args.model, compile=0)

print("Starting reading header of:", args.input_header)
input_header = pd.read_csv(args.input_header, index_col='feature_id')

f = open(args.output, 'w+')
f.writelines(['TITLE\tDENOVO\tScore\tPPM Difference\tPositional Score\n'])

# sequencing loop
i = 0
while True:
    spectra = read_spec(input_header, args.input_folder, count=args.loop_size, previous_loc=i)
    if len(spectra) <= 0:
        break

    print("De novo spectra from", i, "to", i + len(spectra))
    i += len(spectra)

    peps, ppeps, scores, pscores, ppms, _ = denovo(model, spectra, args.batch_size)

    f.writelines("\t".join([p, pp, f4(s), str(ppm), str(list(pscore)[:len(pp)])]) + "\n"
                 for p, pp, s, pscore, ppm in zip(peps, ppeps, scores, pscores, ppms))

f.close()
print('Finished,', i, 'spectra in total')
