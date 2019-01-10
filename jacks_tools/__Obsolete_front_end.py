
import io, os, csv, random, sys, logging
from jacks.jacks import infer_JACKS, LOG, infer_JACKS_gene
from jacks.io_preprocess import (load_data_and_preprocess, writeJacksWResults, writeJacksXResults,
    pickleJacksFullResults, writeFoldChanges)
import numpy as np
import pandas as pd

def prepareFile(filename, hdr):
    #Count any lines before the headers (should be skipped)
    f = io.open(filename)
    skip_lines, line = 0, f.readline()
    while hdr not in line and skip_lines < 100: skip_lines += 1; line = f.readline()
    f.close()

    if skip_lines >= 100:
        raise Exception('Could not find line with header ' + hdr + ' in ' + filename)

    #Check for comma vs tab delimited
    delim = ',' if (filename.split('.')[-1] == 'csv') else '\t'

    #Reopen the file and skip to the start of the data
    f = io.open(filename); [f.readline() for i in range(skip_lines)]
    return f, delim

#output:  {input_filename:[(sample_id, colname)]}
def createSampleSpec(infile, repfile, rep_hdr, sample_hdr, ctrl_sample_or_hdr):
    f, delim = prepareFile(repfile, rep_hdr)
    rdr = csv.DictReader(f, delimiter=delim) # iterating yields ordered dicts keyed to first row of csv
    sample_spec = {infile:[]}
    ctrl_per_sample = (ctrl_sample_or_hdr in rdr.fieldnames)
    ctrl_spec = {}
    for row in rdr:
        sample_spec[infile].append((row[sample_hdr],row[rep_hdr]))
        if ctrl_per_sample:
            if row[sample_hdr] in ctrl_spec:
                if ctrl_spec[row[sample_hdr]] != row[ctrl_sample_or_hdr]:
                    err_msg = '%s vs %s for %s\n' % (ctrl_spec[row[sample_hdr]], row[ctrl_sample_or_hdr], row[sample_hdr])
                    raise Exception(err_msg + 'Different controls for replicates of the sample not supported.')
            else: ctrl_spec[row[sample_hdr]] = row[ctrl_sample_or_hdr]
    f.close()
    return sample_spec, ctrl_per_sample, ctrl_spec



#output:  {grna: gene}
def createGeneSpec(guidemappingfile, sgrna_hdr, gene_hdr):
    f, delim = prepareFile(guidemappingfile, sgrna_hdr)
    rdr = csv.DictReader(f, delimiter=delim)
    gene_spec = {row[sgrna_hdr]:row[gene_hdr] for row in rdr}
    f.close()
    return gene_spec



def loadSgrnaReference(filename):
    f = io.open(filename)
    x_ref = {row['sgrna']:row for row in csv.DictReader(f,delimiter='\t')}
    f.close()
    return x_ref



def bootstrap(gene_index, testdata, ctrldata, gene_grnas,
                    fixed_x=None, n_iter=50, apply_w_hp=False,):

    # Different numbers of guides per genes, so storing x1 x2 results in
    # dict keyed by genes containing ndarrays.
    # ndarray shape will be 2, nguides, nguides-1
    guide_eff = {}
    # storing gene results in the same way
    # array shapes will be (2, nguides, nconditions)
    gene_res = {}

    for gene in gene_index:

        nguides = len(gene_index[gene])
        guide_eff[gene] = x_array = np.ndarray(shape=(2, nguides, nguides - 1))
        gene_res[gene] = w_array = np.ndarray(shape=(2, nguides, testdata.shape[1]))

        for drop_i in range(nguides):
            Ig = gene_index[gene]
            # drop guides
            Ig = [Ig[i] for i in range(nguides) if i != drop_i]
            guide_names = [gene_grnas[gene][i] for i in range(nguides) if i != drop_i]

            data, data_err = testdata[Ig, :, 0], testdata[Ig, :, 1]

            if testdata.shape == ctrldata.shape:  # each line has a matching control:
                gene_ctrl_dat = ctrldata[Ig, :, 0]
                gene_ctrl_err = ctrldata[Ig, :, 1]
            else:
                gene_ctrl_dat = ctrldata[Ig, 0]
                gene_ctrl_err = ctrldata[Ig, 1]

            y, tau, x1, x2, w1, w2 = infer_JACKS_gene(
                data, data_err, gene_ctrl_dat, gene_ctrl_err,
                n_iter, fixed_x=fixed_x, apply_w_hp=apply_w_hp
            )
            w_std = np.sqrt(w2 - w1**2.0)
            x_array[:, drop_i, :] = np.array([x1, x2])
            w_array[:, drop_i, :] = np.array([w1, w_std])

    return guide_eff, gene_res

def run(countfile, replicatefile, guidemappingfile, rep_hdr, sample_hdr, ctrl_sample_or_hdr,
         sgrna_hdr, gene_hdr, outprefix, sgrna_reference_file = None, apply_w_hp = False,
        boot_strap=False):

    LOG.setLevel(logging.WARNING)

    print('Loading sample specification')
    sample_spec, ctrl_per_sample, ctrl_spec = createSampleSpec(
        countfile, replicatefile, rep_hdr, sample_hdr, ctrl_sample_or_hdr
    )

    outfile_w = outprefix + '_gene_JACKS_results.txt'
    outfile_w2 = outprefix + '_genestd_JACKS_results.txt'
    outfile_x = outprefix + '_grna_JACKS_results.txt'
    outfile_lfc = outprefix + '_logfoldchange_means.txt'
    outfile_lfc_std = outprefix + '_logfoldchange_std.txt'
    outfile_pickle = outprefix + '_JACKS_results_full.pickle'

    # Load the mappings from guides to genes
    print('Loading gene mappings')
    gene_spec = createGeneSpec(guidemappingfile, sgrna_hdr, gene_hdr)

    # Load the data and preprocess
    print('Loading data and pre-processing')
    data, meta, sample_ids, genes, gene_index = load_data_and_preprocess(sample_spec, gene_spec)
    gene_grnas = {gene: [x for x in meta[gene_index[gene] ,0]] for gene in gene_index}
    writeFoldChanges(outfile_lfc, data, meta, sample_ids)
    writeFoldChanges(outfile_lfc_std, data, meta, sample_ids, write_std=True)

    if sgrna_reference_file:
        print('Loading sgrna reference values')
        x_ref = loadSgrnaReference(sgrna_reference_file)

        print('Checking sgrna reference identifiers against gene mappings')

        for guide in gene_spec:
            if guide not in x_ref:
                raise Exception('%s has no sgrna reference in %s' % (guide, sgrna_reference_file))

        x_reference = {'X1': np.array([eval(x_ref[x]['X1']) for x in meta[:,0]]),
                       'X2': np.array([eval(x_ref[x]['X2']) for x in meta[:,0]])}
    else:
        x_reference = None

    # Run all samples against their controls
    print('Running JACKS_ inference')
    if ctrl_per_sample:  # Different control samples specified per test sample
        test_sample_idxs = [i for i ,x in enumerate(sample_ids) if ctrl_spec[x] != x]
        testdata = data[: ,test_sample_idxs ,:]
        ctrldata = data[: ,[sample_ids.index(ctrl_spec[sample_ids[idx]]) for idx in test_sample_idxs] ,:]
    else:  # Same control sample for all tests
        ctrldata = data[: ,sample_ids.index(ctrl_sample_or_hdr) ,:]
        test_sample_idxs = [i for i ,x in enumerate(sample_ids) if x != ctrl_sample_or_hdr]
        testdata = data[: ,test_sample_idxs ,:]
    if not boot_strap:
        jacks_results = infer_JACKS(gene_index, testdata, ctrldata, fixed_x=x_reference, apply_w_hp=apply_w_hp)
        # Write out the results
        print('Writing JACKS_ results')
        sample_ids_without_ctrl = [sample_ids[idx] for idx in test_sample_idxs]
        writeJacksWResults( outfile_w, jacks_results, sample_ids_without_ctrl)
        writeJacksWResults( outfile_w2, jacks_results, sample_ids_without_ctrl, write_w2=True)
        writeJacksXResults( outfile_x, jacks_results, gene_grnas )
        pickleJacksFullResults( outfile_pickle, jacks_results, sample_ids_without_ctrl, gene_grnas )
    else:
        [sample_ids[idx] for idx in test_sample_idxs]
        boots_x_w = bootstrap(gene_index, testdata, ctrldata, gene_grnas,
                               fixed_x=x_reference, apply_w_hp=apply_w_hp,)

        if outprefix:
            import pickle
            with open(outprefix + '.boostrap.pickle', 'wb') as f:
                pickle.dump(boots_x_w, f)
        return boots_x_w

# if __name__ == '__main__':
#
    # running both ramsay's screens, with sensitive guides removed...
    # os.chdir('/Users/johnc.thomas/Dropbox/crispr/ramsay/jacks')
    # os.chdir('d:/Dropbox/crispr/ramsay/jacks')
    # bootstrappin = False
    # os.chdir('/Users/johnc.thomas/Dropbox/crispr/dosage')
    # countfile = '../counts_all/hap1_DUB_screens_matylda.tsv'
    # replicatefile = 'repmap_48h.txt'
    # outprefix = 'hap1_DUB_1n2.48h'
    # # sgrna_reference_file = 'both_screens_D14.1_grna_JACKS_results.txt'
    # sgrna_reference_file = None
    #
    # rep_hdr = 'Replicate'
    # sample_hdr = 'Sample'
    # ctrl_sample_or_hdr = 'Control' # Give header if per sample, control sample name otherwise
    # guidemappingfile = countfile
    # gene_hdr = 'gene'
    # sgrna_hdr = 'guide'
    #
    # if os.path.isfile(outprefix+'_JACKS_results_full.pickle'):
    #     input('Results file '+outprefix+'_JACKS_results_full.pickle already exists\npress enter to overwrite.')
    #
    # run(countfile,replicatefile,guidemappingfile,
    #     rep_hdr,sample_hdr,ctrl_sample_or_hdr,
    #     sgrna_hdr,gene_hdr,
    #     outprefix,sgrna_reference_file,
    #     boot_strap=bootstrappin)


    #countfile, replicatefile, guidemappingfile, rep_hdr, sample_hdr, ctrl_sample_or_hdr,
     #        sgrna_hdr, gene_hdr, outprefix, sgrna_reference_file = None, apply_w_hp = False):
    #os.chdir('/Users/johnc.thomas/Dropbox/crispr/becca1/drug-rescued/')
    # os.chdir('/Users/johnc.thomas/Dropbox/crispr/becca1/drug-rescued/')
    # countfile = "./counts/AB.sumrep.counts.tsv"
    # replicatefile = 'repmap.bothcell.d14_1.txt'
    # guidemappingfile = countfile
    # gene_hdr = 'gene'
    # rep_hdr = 'Replicate'
    # sample_hdr = 'Sample'
    # ctrl_sample_or_hdr = 'Control' # Give header if per sample, control sample name otherwise
    # sgrna_hdr = 'sgrna'
    # outprefix = 'both-sum.d14_2_fixSD'
    # # sgrna_reference_file = '/Users/johnc.thomas/Dropbox/crispr/gRNA_efficacy_w_X2.txt'
    # sgrna_reference_file = None
    #
    # if os.path.isfile(outprefix+'_JACKS_results_full.pickle'):
    #     input('Results file '+outprefix+'_JACKS_results_full.pickle already exists\npress enter to overwrite.')
    #
    # run(countfile,replicatefile,guidemappingfile,
    #     rep_hdr,sample_hdr,ctrl_sample_or_hdr,
    #     sgrna_hdr,gene_hdr,
    #     outprefix,sgrna_reference_file)
