import fasttext
import pandas as pd
from tqdm import tqdm
import numpy as np
from Bio import SeqIO
import random
import itertools
from hilbertcurve.hilbertcurve import HilbertCurve
import time
from Bio.SeqUtils import GC, molecular_weight
import tensorflow as tf
import os
import re

def seq2kmer(in_fa, random_choice=False, rand_n=False):
    """
    Function to convert DNA sequences to their kmer counts
    :param in_fa:
    :param random_choice:
    :param rand_n:
    :return:
    """
    print(f'Reading {os.path.basename(in_fa)} into dictionary  and removing N\'s')
    multi_fa = SeqIO.to_dict(SeqIO.parse(in_fa, 'fasta'))
    clean_multi_fa = {}
    for k, v in tqdm(multi_fa.items()):
        if 'N' not in str(v.seq).upper():
            clean_multi_fa[f'{k}'] = v
    if random_choice is True:
        random.seed(12)
        rand_clean_idx = random.sample(list(clean_multi_fa), k=rand_n)
        clean_fa = {key: clean_multi_fa[key] for key in rand_clean_idx}
        print(f'Number of clean records taken randomly: {rand_n}')
    else:
        clean_fa = clean_multi_fa
        print(f'Number of records before N removal: {len(multi_fa)}\nNumber of records after N removal {len(clean_fa)}')

    #seq_len = len(v.seq)

    # Creates a list of all possible 1-, 2- and 3mers.
    nuc = [''.join(n) for n in itertools.product(['A', 'C', 'G', 'T'],
                                                 repeat=1)]
    nuc += [''.join(n) for n in itertools.product(['A', 'C', 'G', 'T'],
                                                  repeat=2)]
    nuc += [''.join(n) for n in itertools.product(['A', 'C', 'G', 'T'],
                                                  repeat=3)]
    nuc += [''.join(n) for n in itertools.product(['A', 'C', 'G', 'T'],
                                                  repeat=4)]

    kmer_mat = np.zeros((len(clean_fa), len(nuc)))



    print(f'Counting kmers for each record in {os.path.basename(in_fa)}')
    for n, record in enumerate(tqdm(clean_fa.items())):

        kmer_count = {f'{nu}':0. for nu in nuc}

        seq = str(record[1].seq).upper()
        # Counts all occurrences of each kmer in each sequence and gives the proportion of its occurrences.
        for i in range(len(seq)):
            kmer_count[f'{seq[i]}'] += (1 / len(seq))

        for i in range(len(seq) - 1):
            kmer_count[f'{seq[i:i+2]}'] += (1 / int(len(seq)-1))

        for i in range(len(seq) - 2):
            kmer_count[f'{seq[i:i+3]}'] += (1 / int(len(seq)-2))

        for i in range(len(seq) - 3):
            kmer_count[f'{seq[i:i+4]}'] += (1 / int(len(seq)-3))

        for k, v in kmer_count.items():
            if len(k) == 1:
                v /= len(seq)
            elif len(k) == 2:
                v /= (len(seq) - 1)
            elif len(k) == 3:
                v /= (len(seq) - 2)
            elif len(k) == 4:
                v /= (len(seq) - 3)

        for index, value in enumerate(kmer_count.items()):
            #print(index, value)
            kmer_mat[n][index] = value[1]

    return kmer_mat

def bed_extract(in_bed, outfile, window=1000, step=1, label=None,
                ignore_sex=False, append_chr=False):
    """
    Function to extract regions from a bed file with the desired window size
    and step.
    :param in_bed: Bed file with regions of interest.
    :param outfile: Outfile that will be a new bed file with the regions at a desired length
    :param window: The desired window length. Default 1000bp
    :param step: The length of each step you want the window to slide.
    Default 1.
    :param label: The desired label you want the region to have. E.g.
    'enhancer'. Default None.
    :return: None, writes out to a file as part of the function call.
    """
    f5 = pd.read_csv(in_bed,
                 sep='\t',
                 header=None,
                 index_col=None)
    if ignore_sex:
        idx = f5.index[f5.iloc[:, 0] == 'chrX'].tolist()
        idx += f5.index[f5.iloc[:, 0] == 'chrY'].tolist()
        f5.drop(idx, inplace=True)
    with open(outfile,'w') as bed:
        for i in tqdm(f5.itertuples()):
            chrom, start, stop = i[1:4]
            if stop - start < window:
                #print(chrom, start, stop)
                midpoint = start + ((stop - start) //2)
                #print(midpoint-(window//2), midpoint+(window//2))
                if append_chr:
                    bed.write(f'chr{chrom}\t{midpoint - (window // 2)}\t{midpoint + (window // 2)}\t{label}\n')
                else:
                    bed.write(f'{chrom}\t{midpoint-(window//2)}\t{midpoint+(window//2)}\t{label}\n')

            elif stop - start >= window:
                #print(start, stop)
                #print(stop-start)
                for k in range(0, ((stop - start) + 1), step):
                    if (start + k + window) <= stop:
                        if append_chr:
                            bed.write(f'chr{chrom}\t{start + k}\t{start + k + window}\t{label}\n')
                        else:
                            bed.write(f'{chrom}\t{start+k}\t{start+k+window}\t{label}\n')
    return f'Finished. Saved to {outfile}'


def gtf_annotation_extract(annotation_file, output_file, append_chr=True):
    """
    Reformat an Ensembl GTF file into a 4 column bed file. chr, start, stop,
    type
    :param annotation_file: The Ensembl GTF file to be reformatted.
    :param output_file: The reformatted output file.
    :param append_chr: Whether or not to append to 'chr' to the chromosome
    column. Default True
    :return: None. Outputs to a file.
    """
    print(f'Reading in annotation file {annotation_file}')
    anno = pd.read_csv(annotation_file,
                       sep='\t',
                       comment='#',
                       header=None,
                       index_col=None)
    print(f'Writing reformatted annotations to {output_file}')
    with open(output_file, 'w') as out:
        for i in tqdm(anno.itertuples()):
            #print(i)
            chrom, start, stop, type = i[1], i[4], i[5], i[3]
            assert type != (None or ''), 'Something\'s missing from the type ' \
                                        'column'
            #print(chrom, start, stop, type)
            if append_chr == True:
                out.write(f'chr{chrom}\t{start}\t{stop}\t{type}\n')
            else:
                out.write(f'{chrom}\t{start}\t{stop}\t{type}\n')
    return f'Finished. Saved to {output_file}'


def clean_negative_set(in_bed, out_bed, num_auto):
    """

    :param in_bed:
    :param out_bed:
    :param num_auto:
    :return:
    """
    chrom = [f'chr{i}' for i in range(1,(num_auto + 1))]
    neg_set = pd.read_csv(in_bed,
                          sep='\t',
                          index_col=None,
                          header=None)
    with open(out_bed, 'w') as bed:
        for i in tqdm(neg_set.itertuples()):
            if i[1] in chrom:
                bed.write(f'{i[1]}\t{i[2]}\t{i[3]}\n')
    return f'Finished. Saved {out_bed}'

def neg_bed_extract(in_bed, outfile, window=1000, step=1, label=None):
    """
    Function to extract regions from a bed file with the desired window size
    and step.
    :param in_bed: Bed file with regions of interest.
    :param outfile: Outfile that will be a new bed file with the regions at a desired length
    :param window: The desired window length. Default 1000bp
    :param step: The length of each step you want the window to slide.
    Default 1.
    :param label: The desired label you want the region to have. E.g.
    'enhancer'. Default None.
    :return: None, writes out to a file as part of the function call.
    """
    neg = pd.read_csv(in_bed,
                 sep='\t',
                 header=None,
                 index_col=None)
    with open(outfile,'w') as bed:
        for i in tqdm(neg.itertuples()):
            chrom, start, stop = i[1:4]
            if stop - start < window:
                continue
                #print(chrom, start, stop)
                #midpoint = start + ((stop - start) //2)
                #print(midpoint-(window//2), midpoint+(window//2))
                #bed.write(f'{chrom}\t{midpoint-(window//2)}\
                # t{midpoint+(window//2)}\t{label}\n')

            elif stop - start >= window:
                #print(start, stop)
                #print(stop-start)
                for k in range(0, ((stop - start) + 1), step):
                    if (start + k + window) <= stop:
                        bed.write(f'{chrom}\t{start+k}\t{start+k+window}\t{label}\n')
    return f'Finished. Saved to {outfile}'

def one_hot(seq):
    """

    :param seq: Input sequence to be one hot encoded.
    :return: np.array representing the one-hot encoded sequence.
    """
    array = np.zeros(shape=(int(len(seq)), 4))
    for n in range(len(seq)):
        if seq[n] == 'A':
            array[n][0] = 1.
        elif seq[n] == 'C':
            array[n][1] = 1.
        elif seq[n] == 'G':
            array[n][2] = 1.
        elif seq[n] == 'T':
            array[n][3] = 1

    return array

def seq2img3D(in_fa, hc_p=1 , hc_n=2,
            random_choice=False,
            rand_n=None):
    """
    Function to convert dna sequence to an n x n image with 256 channels.
    :param in_fa: Multifasta file to be converted
    :param hc_p: Order for the hilbert curve.
    :param hc_n: Number of dimensions for hilbert curve. Default=2
    :param random_choice: Bool. If true, will take a random selection of
    records from the dictionary equal to size of rand_n.
    :param rand_n: Number of records to use if random_choice is True.
    :return: ndarray of shape N, C, H, W. Where N is the
    number of records in the multifasta file,
    C is the number of channels, since we have 256 unique 4mer combos,
    we have 256 channels to the "image", H height 2**hc_p. W width 2**hc_p.
    """
    # Generate the Hilbert Curve
    print(f'Generating hilbert curve of order {hc_p} with {hc_n} dimensions.\n')
    HC = HilbertCurve(n=hc_n, p=hc_p)
    points = HC.points_from_distances(distances=list(range(int(2**hc_p)**2)))

    # Generate the mapping dictionary for values. Here, we are using 4mers so
    # all possible 4mers we can have is 256. 4 nucleotides to the power of
    # 4mer = 256 possible kmers.
    print(f'Generating mapping dictionary for all possible 4mers\n')
    def mapping_dict():
        nuc = [''.join(n) for n in
               itertools.product(['A', 'C', 'G', 'T'], repeat=4)]
        nuc_dict = {}
        for k, i in enumerate(nuc):
            # print(k, i)
            nuc_dict[f'{i}'] = [0. for i in range(len(nuc))]
            nuc_dict[f'{i}'][k] = 1.
            nuc_dict[f'{i}'] = np.array(nuc_dict[f'{i}'])
        return nuc_dict


    print(f'Reading in {in_fa} to a dictionary\nRemoving records with N\'s\n')
    multi_fa = SeqIO.to_dict(SeqIO.parse(in_fa, 'fasta'))
    clean_multi_fa = {}
    for k, v in tqdm(multi_fa.items()):
        if 'N' not in str(v.seq).upper():
            clean_multi_fa[f'{k}'] = v
    if random_choice is True:
        random.seed(12)
        rand_clean_idx = random.sample(list(clean_multi_fa), k=rand_n)
        clean_fa = {key: clean_multi_fa[key] for key in rand_clean_idx}
        print(f'Number of clean records taken randomly: {rand_n}')
    else:
        clean_fa = clean_multi_fa
        print(f'Number of records before N removal: {len(multi_fa)}\nNumber of '
          f'records after N removal: {len(clean_fa)}\n')
    start_time = time.time()
    nuc_dict = mapping_dict()
    print(f'Generating array dataset of shape '
          f'{len(clean_fa)}, 256, {2**hc_p}, {2**hc_p}\n')
    img_mat = np.zeros((len(clean_fa), 256, 2**hc_p, 2**hc_p))
    nuc_channel = list(nuc_dict.keys())
    print(f'Beginning sequencing to image conversion for {in_fa}.')
    for k, record in enumerate(tqdm(clean_fa.items())):
        for i in range(int(len(str(record[1].seq).upper())-4 + 1)):
            nc = nuc_channel.index(f'{str(record[1].seq).upper()[i:i+4]}')
            img_mat[k][nc][points[i][1]][points[i][0]] = 1
    end_time = time.time()
    total_time = end_time - start_time
    print(f'Time taken to create image array: {total_time/60} mins')
    return img_mat

def fasta2txt(in_fa, out_txt, n, random_choice=False, rand_n=None):
    """
    Function to reformat a multifasta file into a text file to be used by Tensorflow's tf.data.TextLineDataset
    :param in_fa: Input multi-fasta file
    :param out_txt: Output text file
    :param n: size of each word. The string will be separated at every n point
    :param random_choice: Takes a random sample of the multi-fasta file that meets filtering criteria.
    :param rand_n: Number of random samples to take
    :return: None. Writes a new file in `out_txt`.
    """
    print(f'Reading in {os.path.basename(in_fa)} to a dictionary\nRemoving records with N\'s\n')
    multi_fa = SeqIO.to_dict(SeqIO.parse(in_fa, 'fasta'))
    clean_multi_fa = {}
    for k, v in tqdm(multi_fa.items()):
        if 'N' not in str(v.seq).upper() and k not in clean_multi_fa:
            clean_multi_fa[f'{k}'] = v
        #elif f'{k}' in clean_multi_fa:
        #    print(f'Key: {k} already in dictionary, adding ".2" to the key.')
        #    clean_multi_fa[f'{k}.2'] = v
        #elif f'{k}.2' in clean_multi_fa:
        #    print(f'Key: {k} already in dictionary, adding ".3" to the key.')
        #    clean_multi_fa[f'{k}.3'] = v
    if random_choice is True:
        random.seed(12)
        rand_clean_idx = random.sample(list(clean_multi_fa), k=rand_n)
        clean_fa = {key: clean_multi_fa[key] for key in rand_clean_idx}
        print(f'Number of clean records taken randomly: {rand_n}')
    else:
        clean_fa = clean_multi_fa
        print(f'Number of records before N removal: {len(multi_fa)}\nNumber of records after N removal: {len(clean_fa)}\n')
    with open(out_txt, 'w') as txt:
        print(f'Writing reformatted sequences to {os.path.basename(out_txt)}.')
        for k, v in tqdm(clean_fa.items()):
            seq = str(v.seq).upper()
            string_seq = [str(seq[i:i+n]) for i in range(0, len(seq)-n+1, n)]
            new_string = ' '.join(string_seq)
            txt.write(f'{new_string}\n')
    txt.close()
    return 'Finished reformatting fasta sequences.'

def seq2img2D(in_fa, hc_p=1 , hc_n=2,
            random_choice=False,
            rand_n=None):
    """
    Function to convert dna sequence to an n x n image where each pixel value is
    determined by what 4mer is in that position.

    :param in_fa: Multifasta file to be converted
    :param hc_p: Order for the hilbert curve. Default = 1
    :param hc_n: Number of dimensions for hilbert curve. Default=2
    :param random_choice: Bool. If true, will take a random selection of
    records from the dictionary equal to size of rand_n.
    :param rand_n: Number of records to use if random_choice is True.
    :return: ndarray of shape N, H, W. Where N is the
    number of records in the multifasta file, H height 2**hc_p. W width 2**hc_p.
    """
    # Generate the Hilbert Curve
    print(f'Generating hilbert curve of order {hc_p} with {hc_n} dimensions.\n')
    HC = HilbertCurve(n=hc_n, p=hc_p)
    points = HC.points_from_distances(distances=list(range(int(2**hc_p)**2)))

    # Generate the mapping dictionary for values. Here, we are using 4mers so
    # all possible 4mers we can have is 256. 4 nucleotides to the power of
    # 4mer = 256 possible kmers.
    print(f'Generating mapping dictionary for all possible 4mers\n')
    def mapping_dict():
        nuc = [''.join(n) for n in
               itertools.product(['A', 'C', 'G', 'T'], repeat=4)]
        nuc_dict = {}
        for k, i in enumerate(nuc):
            # print(k, i)
            nuc_dict[f'{i}'] = k + 1.
        return nuc_dict


    print(f'Reading in {in_fa} to a dictionary\nRemoving records with N\'s\n')
    multi_fa = SeqIO.to_dict(SeqIO.parse(in_fa, 'fasta'))
    clean_multi_fa = {}
    for k, v in tqdm(multi_fa.items()):
        if 'N' not in str(v.seq).upper() and k not in clean_multi_fa:
            clean_multi_fa[f'{k}'] = v
        elif f'{k}' in clean_multi_fa:
            print(f'Key: {k} already in dictionary, adding ".2" to the key.')
            clean_multi_fa[f'{k}.2'] = v
        elif f'{k}.2' in clean_multi_fa:
            print(f'Key: {k} already in dictionary, adding ".3" to the key.')
            clean_multi_fa[f'{k}.3'] = v
    if random_choice is True:
        random.seed(12)
        rand_clean_idx = random.sample(list(clean_multi_fa), k=rand_n)
        clean_fa = {key: clean_multi_fa[key] for key in rand_clean_idx}
        print(f'Number of clean records taken randomly: {rand_n}')
    else:
        clean_fa = clean_multi_fa
        print(f'Number of records before N removal: {len(multi_fa)}\nNumber of '
          f'records after N removal: {len(clean_fa)}\n')
    start_time = time.time()
    nuc_dict = mapping_dict()
    print(f'Generating array dataset of shape '
          f'{len(clean_fa)}, {2**hc_p}, {2**hc_p}\n')
    img_mat = np.zeros((len(clean_fa), 2**hc_p, 2**hc_p))
    nuc_channel = list(nuc_dict.keys())
    print(f'Beginning sequencing to image conversion for {in_fa}.')
    for k, record in enumerate(tqdm(clean_fa.items())):
        for i in range(int(len(str(record[1].seq).upper())-4 + 1)):
            nc = nuc_channel.index(f'{str(record[1].seq).upper()[i:i+4]}')
            img_mat[k][points[i][1]][points[i][0]] = nc
    end_time = time.time()
    total_time = end_time - start_time
    print(f'Time taken to create image array: {total_time/60} mins')
    return img_mat / 256


def fa_stats(in_fa):
    """
    Function to compute some basic descriptive stats from a multifasta file.
    :param in_fa: Path to multifasta file
    :return: A dictionary of stats that can be converted to a DataFrame
    """
    stats = {}
    lengths = {}
    gc = {}
    mol_weight = {}
    perc_Ns = {}
    words = [''.join(c) for c in itertools.product('ACGT', repeat=10)]
    seq_words = {}
    for w in words:
        seq_words[f'{w}'] = 0


    for i, record in tqdm(enumerate(SeqIO.parse(in_fa, 'fasta'))):
        for n in range(int(len(record.seq)-9)):
            word = str(record.seq).upper()[n:n+10]
            if word in seq_words:
                seq_words[f'{word}'] += 1

        lengths[f"enhancer_{i}"] = len(str(record.seq))
        gc[f"enhancer_{i}"] = GC(record.seq)
        perc_Ns[f'enhancer_{i}'] = (str(record.seq).upper().count('N') / len\
            (str(record.seq)) * 100)
        if 'N' not in str(record.seq).upper():
            mol_weight[f'enhancer_{i}'] = \
                molecular_weight(seq = str(record.seq),seq_type='DNA')
        elif 'N' in str(record.seq).upper():
            mol_weight[f'enhancer_{i}'] = np.nan

    stats["Lengths"] = lengths
    stats["GC content"] = gc
    stats['Percentage of Ns'] = perc_Ns
    stats['Molecular weight'] = mol_weight
    #stats["Unique words"] = seq_words
    return stats

def seq2onehot(in_fa, random_choice=False, rand_n=None):
    """
    Function to generate one-hot encoded matrices from fasta sequences
    :param in_fa: Path to multifasta file.
    :param random_choice: If true will take a random sample from the fasta record. Default False
    :param rand_n: If random_choice = True, will take rand_n number of random samples.
    :return: Matrix with  shape N, L, 4.
    N = number of records in the fasta file.
    L = length of the sequence (rows) and
    4 is A, C, G, T.
    """
    print(f'Reading {in_fa} into dictionary  and removing N\'s')
    multi_fa = SeqIO.to_dict(SeqIO.parse(in_fa, 'fasta'))
    clean_multi_fa = {}
    for k, v in tqdm(multi_fa.items()):
        if 'N' not in str(v.seq).upper():
            clean_multi_fa[f'{k}'] = v
    if random_choice is True:
        random.seed(12)
        rand_clean_idx = random.sample(list(clean_multi_fa), k=rand_n)
        clean_fa = {key: clean_multi_fa[key] for key in rand_clean_idx}
        print(f'Number of clean records taken randomly: {rand_n}')
    else:
        clean_fa = clean_multi_fa
        print(f'Number of records before N removal: {len(multi_fa)}\nNumber of records after N removal {len(clean_fa)}')
    seq_len = len(v.seq)
    start_time = time.time()
    one_hot_mat = np.zeros((len(clean_fa), seq_len, 4))

    print(f'Beginning one-hot encoding of {in_fa}')
    for k, record in enumerate(tqdm(clean_fa.items())):
        for i in range(len(str(record[1].seq).upper())):
            assert str(record[1].seq).upper()[i] != 'N', 'Something went ' \
                                                         'wrong prior. You ' \
                                                         'need to make sure ' \
                                                         'there are no Ns in ' \
                                                         'the seq'
            if str(record[1].seq).upper()[i] == 'A':
                one_hot_mat[k][i][0] = 1.
            elif str(record[1].seq).upper()[i] == 'C':
                one_hot_mat[k][i][1] = 1.
            elif str(record[1].seq).upper()[i] == 'G':
                one_hot_mat[k][i][2] = 1.
            elif str(record[1].seq).upper()[i] == 'T':
                one_hot_mat[k][i][3] = 1.
    end_time = time.time()
    total_time = end_time - start_time
    print(f'Time taken to create one-hot matrix: {total_time/60} mins')
    return one_hot_mat

def seqToWordVec(in_fa, in_vec, model_path, word_size=10, random_choice=False,
                 rand_n=None):
    """
    Function to convert DNA sequence to a word vector representation.
    :param in_fa: Input multifasta file.
    :param in_vec: vector file generated by FastText training.
    :param word_size: Word size used to generate the corpus. Defaults to 10
    :param model_path: Path to the trained FastText model binary.
    :param random_choice: Default False. If True, will take a random sample
    == rand_n.
    :param rand_n: Number of random samples to take.
    :return: Matrix of word vectors.
    """
    print(f'Loading model:\t{model_path}\n')
    model = fasttext.load_model(model_path)
    print(f'Loading vectors:\t{in_vec}')
    word_vec = pd.read_csv(in_vec,
                           header=None,
                           index_col=0,
                           sep=' ',
                           skiprows=1)
    word_vec = word_vec.iloc[:, :100]

    word_dict = {}
    for i in tqdm(word_vec.itertuples()):
        word_dict[f'{i[0]}'] = np.reshape(np.array(list(i[1:])),
                                          newshape=(100,))

    print(f'Reading {in_fa} into dictionary  and removing N\'s')
    multi_fa = SeqIO.to_dict(SeqIO.parse(in_fa, 'fasta'))
    clean_multi_fa = {}
    for k, v in tqdm(multi_fa.items()):
        if 'N' not in str(v.seq).upper():
            clean_multi_fa[f'{k}'] = v
    if random_choice is True:
        random.seed(12)
        rand_clean_idx = random.sample(list(clean_multi_fa), k=rand_n)
        clean_fa = {key: clean_multi_fa[key] for key in rand_clean_idx}
        print(f'Number of clean records taken randomly: {rand_n}')
    else:
        clean_fa = clean_multi_fa
        print(f'Number of records before N removal: {len(multi_fa)}\nNumber of records after N removal {len(clean_fa)}')


    start_time = time.time()


    print(f'Beginning word vector represenations of {in_fa}')
    mat = []
    for k, record in enumerate(tqdm(clean_fa.items())):
        seq_arr = []
        for i in range(0, len(str(record[1].seq).upper()), word_size):
            assert str(record[1].seq).upper()[i] != 'N', 'Something went ' \
                                                         'wrong prior. You ' \
                                                         'need to make sure ' \
                                                         'there are no Ns in ' \
                                                         'the seq'

            if str(record[1].seq.upper()[i:i+word_size]) not in \
                        word_dict.keys():
                print(f'{str(record[1].seq).upper()[i:i+word_size]} not in '
                          f'dictionary. Creating vector '
                  f'from subwords.')
                seq_arr.append(model.get_word_vector(str(record[
                                                               1].seq.upper()[
                                                         i:i+word_size])))
            else:
                seq_arr.append(word_dict[str(record[1].seq).upper()[
                                       i:i+word_size]])
        seq_arr = np.array(seq_arr)
        mat.append(seq_arr.T)

    mat = np.array(mat)

    return mat

VECTOR_FILE = '/Users/callummacphillamy/PhD/Reference_Genomes/hg19/hg19.skipgram.30.vec'
MODEL_PATH = '/Users/callummacphillamy/PhD/Reference_Genomes/hg19/hg19.skipgram.30.bin'

def seq2FlatWordVec(in_fa, word_size=10, random_choice=False,
                 rand_n=None):
    """
    Function to convert DNA sequence to a word vector representation.
    :param in_fa: Input multifasta file.
    :param in_vec: vector file generated by FastText training.
    :param word_size: Word size used to generate the corpus. Defaults to 10
    :param model_path: Path to the trained FastText model binary.
    :param random_choice: Default False. If True, will take a random sample
    == rand_n.
    :param rand_n: Number of random samples to take.
    :return: Matrix of word vectors.
    """
    #print(f'Loading model:\t{model_path}\n')
    model = fasttext.load_model(MODEL_PATH)
    #print(f'Loading vectors:\t{in_vec}')
    word_vec = pd.read_csv(VECTOR_FILE,
                           header=None,
                           index_col=0,
                           sep=' ',
                           skiprows=1)
    word_vec = word_vec.iloc[:, :30]

    word_dict = {}
    for i in tqdm(word_vec.itertuples()):
        word_dict[f'{i[0]}'] = np.reshape(np.array(list(i[1:])),
                                          newshape=(30,))

    print(f'Reading {in_fa} into dictionary  and removing N\'s')
    multi_fa = SeqIO.to_dict(SeqIO.parse(in_fa, 'fasta'))
    clean_multi_fa = {}
    for k, v in tqdm(multi_fa.items()):
        if 'N' not in str(v.seq).upper():
            clean_multi_fa[f'{k}'] = v
    if random_choice is True:
        random.seed(12)
        rand_clean_idx = random.sample(list(clean_multi_fa), k=rand_n)
        clean_fa = {key: clean_multi_fa[key] for key in rand_clean_idx}
        print(f'Number of clean records taken randomly: {rand_n}')
    else:
        clean_fa = clean_multi_fa
        print(f'Number of records before N removal: {len(multi_fa)}\nNumber of records after N removal {len(clean_fa)}')


    start_time = time.time()


    print(f'Beginning word vector represenations of {in_fa}')
    mat = []
    for k, record in enumerate(tqdm(clean_fa.items())):
        seq_arr = []
        for i in range(0, len(str(record[1].seq).upper()), word_size):
            assert str(record[1].seq).upper()[i] != 'N', 'Something went ' \
                                                         'wrong prior. You ' \
                                                         'need to make sure ' \
                                                         'there are no Ns in ' \
                                                         'the seq'

            if str(record[1].seq.upper()[i:i+word_size]) not in \
                    word_dict.keys():
                print(f'{str(record[1].seq).upper()[i:i+word_size]} not in '
                      f'dictionary. Creating vector '
                      f'from subwords.')
                seq_arr.append(model.get_word_vector(str(record[
                                                             1].seq.upper()[
                                                         i:i+word_size])))
            else:
                seq_arr.append(word_dict[str(record[1].seq).upper()[
                                         i:i+word_size]])
        seq_arr = np.array(seq_arr)
        mat.append(np.matrix.flatten(seq_arr.T))

    mat = np.array(mat)

    return mat

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, X_arr, y_arr, list_IDs, batch_size=32, dim=(1000,4), n_channels=1,
                 n_classes=2, shuffle=True):
        'Initialization'
        self.X_arr = X_arr
        self.y_arr = y_arr
        self.dim = dim
        self.batch_size = batch_size
        #self.labels = labels
        self.list_IDs = list_IDs # list of the indexes of the array
        #self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.list_IDs)) / self.batch_size)

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        X, y = self.__data_generation(list_IDs_temp)
        #X = X.reshape((self.batch_size, *self.dim, self.n_channels))
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'
        X = np.empty((self.batch_size, *self.dim))#, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            X[i,] = self.X_arr[ID]

            y[i] = self.y_arr[ID]
        #X = X.reshape((self.batch_size, *self.dim, self.n_channels))
        return X, y

class ImgDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, X_arr, y_arr, list_IDs, batch_size=32, dim=(32,32),
                 n_channels=1,
                 n_classes=2, shuffle=True):
        'Initialization'
        self.X_arr = X_arr
        self.y_arr = y_arr
        self.dim = dim
        self.batch_size = batch_size
        #self.labels = labels
        self.list_IDs = list_IDs # list of the indexes of the array
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.list_IDs)) / self.batch_size)

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        X, y = self.__data_generation(list_IDs_temp)
        #X = X.reshape((self.batch_size, *self.dim, self.n_channels))
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            X[i,] = self.X_arr[ID]

            y[i] = self.y_arr[ID]
        #X = X.reshape((self.batch_size, *self.dim, self.n_channels))
        return X, y

class WVDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, X_arr, y_arr, list_IDs, batch_size=32, dim=(100,200),
                 n_classes=2, shuffle=True):
        'Initialization'
        self.X_arr = X_arr
        self.y_arr = y_arr
        self.dim = dim
        self.batch_size = batch_size
        #self.labels = labels
        self.list_IDs = list_IDs # list of the indexes of the array
        #self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.list_IDs)) / self.batch_size)

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        X, y = self.__data_generation(list_IDs_temp)
        #X = X.reshape((self.batch_size, *self.dim, self.n_channels))
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'
        X = np.empty((self.batch_size, *self.dim)) #, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            X[i,] = self.X_arr[ID]

            y[i] = self.y_arr[ID]
        #X = X.reshape((self.batch_size, *self.dim, self.n_channels))
        return X, y

def makeGappedKmers(word_length=int(), mismatch=int()):
    assert word_length > mismatch, 'Word length must be greater than the number of mismatches allowed. E.g. word_length = 3 and mismatch = 3 will yield a kmer that looks like this: `...`'
    print(f'Generating list of all possible gapped kmer combinations for word_legnth: {word_length}; mismatch: {mismatch}.')
    words = list(itertools.product(['A', 'C', 'G', 'T', '.'], repeat=word_length))
    gapped = {}
    for word in words:
        kmer = ''.join(word)
        if kmer.count('.') == mismatch:
            gapped[f'{kmer}'] = 0
    print(f'{len(gapped)} gapped kmers generated')
    return gapped

def seq2GappedKmer(in_fa=str(), word_length=int(), mismatch=int(), random_choice=False, rand_n=False):
    """
    Function to convert DNA sequences to their kmer counts
    :param in_fa:
    :param random_choice:
    :param rand_n:
    :return:
    """
    print(f'Reading {os.path.basename(in_fa)} into dictionary  and removing N\'s')
    multi_fa = SeqIO.to_dict(SeqIO.parse(in_fa, 'fasta'))
    clean_multi_fa = {}
    for k, v in tqdm(multi_fa.items()):
        if 'N' not in str(v.seq).upper():
            clean_multi_fa[f'{k}'] = v
    if random_choice is True:
        random.seed(12)
        rand_clean_idx = random.sample(list(clean_multi_fa), k=rand_n)
        clean_fa = {key: clean_multi_fa[key] for key in rand_clean_idx}
        print(f'Number of clean records taken randomly: {rand_n}')
    else:
        clean_fa = clean_multi_fa
        print(f'Number of records before N removal: {len(multi_fa)}\nNumber of records after N removal {len(clean_fa)}')

    #seq_len = len(v.seq)

    # Create a list of all possible gapped kmers.
    possible_g_kmers = makeGappedKmers(word_length, mismatch)
    possible_g_kmers = list(possible_g_kmers.keys())


    kmer_mat = np.zeros((len(clean_fa), len(possible_g_kmers)))



    print(f'Counting gapped kmers for each record in {os.path.basename(in_fa)}')
    for n, record in enumerate(tqdm(clean_fa.items())):

        kmer_count = {f'{gk}':0. for gk in possible_g_kmers}

        seq = str(record[1].seq).upper()
        # Counts all occurrences of each kmer in each sequence and gives the proportion of its occurrences.
        for km in kmer_count.keys():
            matches = [m.start() for m in re.finditer(f'(?={km})', seq)]
            kmer_count[f'{km}'] = len(matches)

        for index, value in enumerate(kmer_count.items()):
            #print(index, value)
            kmer_mat[n][index] = value[1]

    return kmer_mat

def seq2kmerMatrix(in_fa, random_choice=False, rand_n=False):
    """
    Function to convert DNA sequences to their kmer counts
    :param in_fa:
    :param random_choice:
    :param rand_n:
    :return:
    """
    print(f'Reading {os.path.basename(in_fa)} into dictionary  and removing N\'s')
    multi_fa = SeqIO.to_dict(SeqIO.parse(in_fa, 'fasta'))
    clean_multi_fa = {}
    for k, v in tqdm(multi_fa.items()):
        if 'N' not in str(v.seq).upper():
            clean_multi_fa[f'{k}'] = v
    if random_choice is True:
        random.seed(12)
        rand_clean_idx = random.sample(list(clean_multi_fa), k=rand_n)
        clean_fa = {key: clean_multi_fa[key] for key in rand_clean_idx}
        print(f'Number of clean records taken randomly: {rand_n}')
    else:
        clean_fa = clean_multi_fa
        print(f'Number of records before N removal: {len(multi_fa)}\nNumber of records after N removal {len(clean_fa)}')

    #seq_len = len(v.seq)

    # Creates a list of all possible 1-, 2- and 3mers.
    nuc = [''.join(n) for n in itertools.product(['A', 'C', 'G', 'T'],
                                                 repeat=1)]
    nuc += [''.join(n) for n in itertools.product(['A', 'C', 'G', 'T'],
                                                  repeat=2)]
    nuc += [''.join(n) for n in itertools.product(['A', 'C', 'G', 'T'],
                                                  repeat=3)]
    #nuc += [''.join(n) for n in itertools.product(['A', 'C', 'G', 'T'],
                                                  #repeat=4)]

    kmer_mat = np.zeros((len(clean_fa), len(nuc)))



    print(f'Counting kmers for each record in {os.path.basename(in_fa)}')
    for n, record in enumerate(tqdm(clean_fa.items())):

        kmer_count = {f'{nu}':0. for nu in nuc}

        seq = str(record[1].seq).upper()
        # Counts all occurrences of each kmer in each sequence and gives the proportion of its occurrences.
        for i in range(len(seq)):
            kmer_count[f'{seq[i]}'] += (1 / len(seq))

        for i in range(len(seq) - 1):
            kmer_count[f'{seq[i:i+2]}'] += (1 / int(len(seq)-1))

        for i in range(len(seq) - 2):
            kmer_count[f'{seq[i:i+3]}'] += (1 / int(len(seq)-2))

        #for i in range(len(seq) - 3):
        #    kmer_count[f'{seq[i:i+4]}'] += (1 / int(len(seq)-3))

        for k, v in kmer_count.items():
            if len(k) == 1:
                v /= len(seq)
            elif len(k) == 2:
                v /= (len(seq) - 1)
            elif len(k) == 3:
                v /= (len(seq) - 2)
            elif len(k) == 4:
                v /= (len(seq) - 3)

        for index, value in enumerate(kmer_count.items()):
            #print(index, value)
            kmer_mat[n][index] = value[1]

    return kmer_mat

def seq24hot(in_fa=str(), random_choice=False, rand_n=False):
    print(f'Reading {os.path.basename(in_fa)} into dictionary  and removing N\'s')
    multi_fa = SeqIO.to_dict(SeqIO.parse(in_fa, 'fasta'))
    clean_multi_fa = {}
    for k, v in tqdm(multi_fa.items()):
        if 'N' not in str(v.seq).upper():
            clean_multi_fa[f'{k}'] = v
    if random_choice is True:
        random.seed(12)
        rand_clean_idx = random.sample(list(clean_multi_fa), k=rand_n)
        clean_fa = {key: clean_multi_fa[key] for key in rand_clean_idx}
        print(f'Number of clean records taken randomly: {rand_n}')
    else:
        clean_fa = clean_multi_fa
        print(f'Number of records before N removal: {len(multi_fa)}\nNumber of records after N removal {len(clean_fa)}')

    #seq_len = len(v.seq)
    values_view = clean_fa.values()
    value_iterator = iter(values_view)
    seq_len = next(value_iterator)
    kmer_mat = np.zeros((len(clean_fa), len(seq_len)))



    print(f'Counting kmers for each record in {os.path.basename(in_fa)}')
    for n, record in enumerate(tqdm(clean_fa.items())):
        #print(n, record)
        #break
        seq = str(record[1].seq).upper()
        #break
        for nu in range(len(seq)):
            #print(seq[nu])
            if seq[nu] == 'A':
                kmer_mat[n][nu] = 0.25
            elif seq[nu] == 'C':
                kmer_mat[n][nu] = 0.5
            elif seq[nu] == 'G':
                kmer_mat[n][nu] = 0.75
            elif seq[nu] == 'T':
                kmer_mat[n][nu] = 1.

    return kmer_mat    