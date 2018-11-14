import os
#import subprocess
#from subprocess import run
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests
try:
    from adjustText import adjust_text
except ModuleNotFoundError:
    def adjust_text(*args, **kwargs):
        pass

#v0.2


def plot_volcano(esstab, savefn=None):

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    pos = esstab['jacks_score'] > 0
    neg = ~pos

    faces = dict(facecolors='none', edgecolors='b')
    plt.scatter(esstab.loc[pos, 'jacks_score'], esstab.loc[pos, 'fdr_pos'], **faces)
    plt.scatter(esstab.loc[neg, 'jacks_score'], esstab.loc[neg, 'fdr_neg'], **faces)
    plt.yscale('log')

    plt.plot([min(esstab['jacks_score']), max(esstab['jacks_score'])],
             [0.05, 0.05], 'k--')

    # get lowest not zero and set zeroes to 1/10 that value
    min_pos = min(esstab.loc[esstab['fdr_pos'] > 0, 'fdr_pos'])
    min_neg = min(esstab.loc[esstab['fdr_neg'] > 0, 'fdr_neg'])
    print(min_neg, min_pos)
    for fdri, fdr in enumerate(['fdr_pos', 'fdr_neg']):
        esstab.loc[esstab[fdr] == 0, fdr] = (min_pos, min_neg)[fdri] / 10

    plt.xlabel('Essentiality')
    plt.ylabel('FDR')

    plt.ylim(min(min_pos, min_neg) / 10)
    plt.gca().invert_yaxis()
    if savefn :
        plt.savefig(savefn)
    return fig, ax


def plot_2d_score(x_score, y_score, formatters=None, labels=None):
    """Produce biplot of 2 essentiality series.
    args:
        x_ess, y_ess -- pd.Series with shared index that gives x and y values
            to be plotted
        labels -- list of gene names to be labeled on the plot
        formatters -- list of tuples as below. format_dict is passed to
            plt.plot(..., **format_dict)
            [(list_of_genes, format_dict), ...]
    Formatters"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    # plot
    if not formatters:
        formatters = [(x_score.index, {})]
    for genes, formats in formatters:
        if 'marker' not in formats:
            formats['marker'] ='o'
        plt.plot(x_score.loc[genes], y_score.loc[genes], linestyle='none', **formats)

    lim = ax.axis()
    minlim = min(lim[0], lim[2])
    maxlim = max(lim[1], lim[3])
    plt.plot([minlim, maxlim], [minlim, maxlim], 'k--', alpha=0.2)
    plt.plot([0, 0], [minlim, maxlim], 'g--')
    plt.plot([minlim, maxlim], [0, 0], 'g--')

    # labels
    txt = []
    # label significants and control
    if labels:
        for lab in set(labels):
            txt.append(plt.text(x_score[lab], y_score[lab], lab))

    plt.tight_layout()
    adjust_text(txt, arrowprops=dict(arrowstyle='->'))

    return fig, ax

def tabulate(prefix):
    """Return 3 tables giving the:
        1. jacks_score|fdr_pos|fdr_neg|std,
        2. guide efficacy data,
    and 3. fold changes for each gene.

    Tables are multiindexed by sample name and then results columns for those
    samples.

    fdr in the

    Prefix is the used to identify the results files. So prefix
    should contain the path to the files if they aren't in os.getcwd()"""

    kwtab = dict(sep='\t', index_col=0)

    # othertab = pd.DataFrame(columns=("IC10","IC90","D14"), index=essen['D14'].index)
    # Tables produced by jacks have columns that are the groups
    genes = pd.read_table(prefix + '_gene_JACKS_results.txt', sep='\t', index_col=0)
    genes_index = sorted(genes.index)
    genes = genes.reindex(genes_index)
    genesstd = pd.read_table(prefix + '_genestd_JACKS_results.txt', sep='\t', index_col=0)
    genesstd = genesstd.reindex(genes_index)
    ps = genes / genesstd
    ps = ps.apply(norm.cdf)

    # multiindex DF for each experiment giving results
    sig_cols = pd.MultiIndex.from_product((ps.columns, ['jacks_score', 'fdr_pos', 'fdr_neg', 'std']),
                                          names=('exp', 'stat'))
    sig_df = pd.DataFrame(index=genes_index, columns=sig_cols)

    for exp in ps.columns:
        sig_df.loc[:, (exp, 'fdr_neg')] = multipletests(ps[exp], method='fdr_bh')[1]
        sig_df.loc[:, (exp, 'fdr_pos')] = multipletests(1 - ps[exp], method='fdr_bh')[1]
        sig_df.loc[:, (exp, 'jacks_score')] = genes[exp]
        sig_df.loc[:, (exp, 'std')] = genesstd



    # get guide data, foldchange and efficacies
    guide_cols = pd.MultiIndex.from_product((ps.columns, ['foldchange', 'fold_std', 'eff', 'eff_std']),
                                            names=['exp', 'stat'])
    fchange_df = pd.DataFrame(columns=guide_cols)
    foldchange = pd.read_table(prefix + '_logfoldchange_means.txt', **kwtab)
    foldstd = pd.read_table(prefix + '_logfoldchange_std.txt', **kwtab)
    eff_tab = pd.read_table(prefix + '_grna_JACKS_results.txt', **kwtab)

    for exp in ps.columns:
        fchange_df.loc[:, (exp, 'foldchange')] = foldchange[exp]
        fchange_df.loc[:, (exp, 'fold_std')] = foldstd[exp]
    fchange_df.loc[:, 'gene'] = foldchange['gene']

    efficacies = pd.DataFrame(columns=('eff', 'eff_std'))
    efficacies.loc[:, 'eff'] = eff_tab['X1']
    efficacies.loc[:, 'eff_std'] = (eff_tab['X2'] - eff_tab['X1'] ** 2) ** 0.5
    efficacies.loc[:, 'gene'] = fchange_df['gene']

    return sig_df, efficacies, fchange_df

if __name__ == '__main__':
    pass
   #  os.chdir('/Users/johnc.thomas/Dropbox/crispr/dub1_hap1_u2os/dosage/jacks_output/')
   # #pd.read_table('hap1_DUB_1n2.48h_gene_JACKS_results.txt', sep='\t', index_col=0)
   #
   #  for prefix in 'hap1_DUB_1n2.48h', 'hap1_DUB_1n2.dmso':
   #      ess, eff, fc = tabulate(prefix)
   #      ess.to_csv(prefix+'.ess_table.csv')
   #      eff.to_csv(prefix+'.efficacy.csv')
   #      fc.to_csv(prefix+'.logFC.csv')

    # pass
    # #
    # # os.chdir('/Users/johnc.thomas/becca1')
    # # scripts_dir = '/Users/johnc.thomas/Applications/guide_trimming_scripts'
    # #
    # # jkfn = './jacks/48hrWRef2_JACKS_results_full.pickle'
    # #
    # # # check efficacy file guides vs becs data
    # # guide_eff = pd.read_table('/Users/johnc.thomas/Dropbox/crispr/gRNA_efficacy_w_X2.txt', sep='\t')
    # #
    # # with open(jkfn, 'rb') as f:
    # #     jkres = jacks_results, cell_lines, grnas = pickle.load(f)
    # #
    # # with open('./jacks/all_counts.tsv') as f:
    # #     counts = pd.read_table(f, index_col=0)
    # # counts.head()
    # #
    # # ess_tables, guide_tables = tabulate_jacksres(*jkres)
    # # # for group, tab in ess_tables.items():
    # # #     print(tab.isna().sum())