import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#import pickle
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests#
try:
    from adjustText import adjust_text
except ModuleNotFoundError:
    def adjust_text(*args, **kwargs):
        pass

__version__ = '0.5'

#todo: sepfucntions for each table, one function that calls all, option to only return scores
#todo: put eff and fold change on the same table (currently nans)
#todo pass ax to plot2d
#todo bootstrapping?

# in 0.5
# refactor "esstab" to "score_table"

def plot_volcano(score_table, savefn=None, ax=None,
                 label_deplet = 0, label_enrich = 0, other_labels=None,
                 p_thresh = 0.05):
    """Supply pandas table with 'jacks_score' and 'fdr_pos/neg' columns.
    Returns fig, ax if no ax is supplied."""
    if ax is None:
        fig, ax_ = plt.subplots(1, 1, figsize=(10, 10))
    else:
        plt.sca(ax)

    score_table = score_table.copy()

    pos = score_table['jacks_score'] > 0
    neg = ~pos

    # get lowest not zero and set zeroes to 1/10 that value
    min_pos = min(score_table.loc[score_table['fdr_pos'] > 0, 'fdr_pos'])
    min_neg = min(score_table.loc[score_table['fdr_neg'] > 0, 'fdr_neg'])
    #print(min_neg, min_pos)
    for fdri, fdr in enumerate(['fdr_pos', 'fdr_neg']):
        score_table.loc[score_table[fdr] == 0, fdr] = (min_pos, min_neg)[fdri] / 10

    for mask, fdr in (pos, 'fdr_pos'), (neg, 'fdr_neg'):
        score_table.loc[mask, 'fdr'] = score_table.loc[mask, fdr]


    score_table.loc[:, 'fdr'] = score_table[fdr].apply(lambda x: -np.log10(x))

    faces = dict(facecolors='none', edgecolors='b')
    # plt.scatter(score_table.loc[pos, 'jacks_score'], score_table.loc[pos, 'fdr_pos'], **faces)
    # plt.scatter(score_table.loc[neg, 'jacks_score'], score_table.loc[neg, 'fdr_neg'], **faces)
    plt.scatter(score_table.jacks_score, score_table.fdr, **faces)

    p = -np.log10(p_thresh)

    plt.plot([min(score_table['jacks_score']), max(score_table['jacks_score'])],
             [p, p], 'k--')

    # label the top and bottom most, if specified
    texts = []
    texts_done = []
    # get subtables
    dep = score_table.sort_values('jacks_score').head(label_deplet)
    enr = score_table.sort_values('jacks_score').tail(label_enrich)
    for stab in dep, enr:
        for lab, row in stab.iterrows():
            if row.fdr < p:
                continue
            texts_done.append(lab)
            texts.append(plt.text(row.jacks_score, row.fdr, lab))
    # label additional genes
    if other_labels:
        for lab, row in score_table.loc[other_labels, :].iterrows():
            if lab in texts_done:
                continue
            texts.append(
                plt.text(score_table.loc[lab, 'jacks_score'],
                         score_table.loc[lab, 'fdr'],
                         lab)
            )
    if texts:
        adjust_text(texts, arrowprops=dict(arrowstyle='->', color='red'))

    plt.xlabel('JACKS score')
    plt.ylabel('-log10(FDR)')

    if savefn :
        plt.savefig(savefn)
    if ax is None:
        return fig, ax_
    else:
        return ax


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
    fig, ax = plt.subplots(1, 1, figsize=(8,8))
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
    genesstd = pd.read_table(prefix + '_gene_std_JACKS_results.txt', sep='\t', index_col=0)
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
        sig_df.loc[:, (exp, 'std')] = genesstd[exp]



    # get guide data, foldchange and efficacies
    guide_cols = pd.MultiIndex.from_product((ps.columns, ['foldchange', 'fold_std', 'eff', 'eff_std']),
                                            names=['exp', 'stat'])
    fchange_df = pd.DataFrame(columns=guide_cols)
    foldchange = pd.read_table(prefix + '_logfoldchange_means.txt', **kwtab)
    foldstd = pd.read_table(prefix + '_logfoldchange_std.txt', **kwtab)
    eff_tab = pd.read_table(prefix + '_grna_JACKS_results.txt', **kwtab)

    for exp in ps.columns:
        fchange_df.loc[:, (exp, 'lfc')] = foldchange[exp]
        fchange_df.loc[:, (exp, 'fold_std')] = foldstd[exp]
    fchange_df.loc[:, 'gene'] = foldchange['gene']

    efficacies = pd.DataFrame(columns=('eff', 'eff_std'))
    efficacies.loc[:, 'eff'] = eff_tab['X1']
    efficacies.loc[:, 'eff_std'] = (eff_tab['X2'] - eff_tab['X1'] ** 2) ** 0.5
    efficacies.loc[:, 'gene'] = fchange_df['gene']

    return sig_df, efficacies, fchange_df

