
# coding: utf-8

# In[3]:


import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.patheffects
import matplotlib.pyplot as plt
import seaborn as sns
import sys

from decimal import Decimal
from matplotlib import gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.font_manager import FontProperties
from matplotlib import transforms
from scipy import stats
from scipy.spatial import distance
from scipy.cluster import hierarchy
from statsmodels.sandbox.stats import multicomp

# import utils
sys.path.append("../../utils")
from misc_utils import *
from norm_utils import *
from snp_utils import *
from del_utils import *

mpl.rcParams['figure.dpi'] = 90


# ## style pre-sets

# In[4]:


NOTEBOOK_PRESET = {"style": "ticks", "font": "Helvetica", "font_scale": 1.2, "context": "notebook"}
NOTEBOOK_FONTSIZE = 10


# In[5]:


PAPER_PRESET = {"style": "ticks", "font": "Helvetica", "context": "paper", 
                "rc": {"font.size":8,"axes.titlesize":8,
                       "axes.labelsize":8, 'axes.linewidth':0.5,
                       "legend.fontsize":8, "xtick.labelsize":8,
                       "ytick.labelsize":8, "xtick.major.size": 3.0,
                       "ytick.major.size": 3.0, "axes.edgecolor": "black",
                       "xtick.major.pad": 3.0, "ytick.major.pad": 3.0}}
PAPER_FONTSIZE = 8


# ## palette pre-sets

# In[6]:


husl = sns.color_palette("husl", 9)
BETTER_TYPE_PALETTE = {"CONTROL": husl[3], "CONTROL_SNP": husl[4], "WILDTYPE": husl[5], "FLIPPED": husl[6], 
                       "SNP": husl[7], "DELETION": husl[0], "SCRAMBLED": "lightgray", "RANDOM": "darkgray"}


# In[ ]:


TSS_CLASS_PALETTE = {"Enhancer": sns.color_palette("deep")[1], 
                     "intergenic": sns.color_palette("deep")[2], "protein_coding": sns.color_palette("deep")[5], 
                     "div_lnc": sns.color_palette("deep")[3], "div_pc": sns.color_palette("deep")[0]}


# In[ ]:


COLOR_DICT = {"A": "crimson", "C": "mediumblue", "G": "orange", "T": "forestgreen"}


# ## label pre-sets

# In[7]:


BETTER_TYPE_ORDER1 = ["CONTROL", "CONTROL_SNP", "WILDTYPE", "FLIPPED", "SNP", "SCRAMBLED", "RANDOM"]
BETTER_TYPE_ORDER2 = ["CONTROL", "CONTROL_SNP", "WILDTYPE", "FLIPPED", "SNP", "DELETION", "SCRAMBLED", "RANDOM"]


# In[ ]:


TSS_CLASS_ORDER = ["Enhancer", "intergenic", "div_lnc", "protein_coding", "div_pc"]


# ## class

# In[ ]:


class Scale(matplotlib.patheffects.RendererBase):
    def __init__(self, sx, sy=None):
        self._sx = sx
        self._sy = sy

    def draw_path(self, renderer, gc, tpath, affine, rgbFace):
        affine = affine.identity().scale(self._sx, self._sy)+affine
        renderer.draw_path(gc, tpath, affine, rgbFace)


# ## plotting functions

# In[ ]:


def add_margin(ax,x=0.05,y=0.05):
    # This will, by default, add 5% to the x and y margins. You 
    # can customise this using the x and y arguments when you call it.

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    xmargin = (xlim[1]-xlim[0])*x
    ymargin = (ylim[1]-ylim[0])*y

    ax.set_xlim(xlim[0]-xmargin,xlim[1]+xmargin)
    ax.set_ylim(ylim[0]-ymargin,ylim[1]+ymargin)


# In[8]:


def mimic_r_boxplot(ax):
    for i, patch in enumerate(ax.artists):
        r, g, b, a = patch.get_facecolor()
        col = (r, g, b, 1)
        patch.set_facecolor((r, g, b, .5))
        patch.set_edgecolor((r, g, b, 1))

        # Each box has 6 associated Line2D objects (to make the whiskers, fliers, etc.)
        # Loop over them here, and use the same colour as above
        line_order = ["lower", "upper", "whisker_1", "whisker_2", "med", "fliers"]
        for j in range(i*6,i*6+6):
            elem = line_order[j%6]
            line = ax.lines[j]
            if "whisker" in elem:
                line.set_visible(False)
            line.set_color(col)
            line.set_mfc(col)
            line.set_mec(col)
            if "fliers" in elem:
                line.set_alpha(0.5)


# In[ ]:


def annotate_pval(ax, x1, x2, y, h, text_y, val, fontsize):
    from decimal import Decimal
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1, c="black", linewidth=0.5)
    if val < 0.0001:
        #text = "{:.2e}".format(Decimal(val))
        text = "**"
    elif val < 0.05:
        #text = "%.4f" % val
        text = "*"
    else:
        text = "n.s."
    ax.text((x1+x2)*.5, text_y, text, ha='center', va='bottom', color="black", size=fontsize)


# In[ ]:


def neg_control_plot(df, order, palette, fontsize, cell_type, ax, figsize, ylabel, sharey, title, save, plotname):
    df_sub = df[df["better_type"].isin(order)].drop_duplicates()
    
    if ax == None:
        plt.figure(figsize=figsize)
        ax = sns.boxplot(data=df_sub, x="better_type", y="overall_mean", order=order, palette=palette, linewidth=1,
                         saturation=1, flierprops = dict(marker='o', markersize=5))
    else:
        sns.boxplot(data=df_sub, x="better_type", y="overall_mean", order=order, palette=palette, linewidth=1,
                    saturation=1, flierprops = dict(marker='o', markersize=5), ax=ax)
    ax.set_xticklabels(order, rotation=30)

    mimic_r_boxplot(ax)

    # calc p-vals b/w dists
    rand_dist = np.asarray(df[df["better_type"] == "random"]["overall_mean"])
    ctrl_dist = np.asarray(df[df["better_type"] == "control"]["overall_mean"])

    rand_dist = rand_dist[~np.isnan(rand_dist)]
    ctrl_dist = ctrl_dist[~np.isnan(ctrl_dist)]

    rand_u, rand_pval = stats.mannwhitneyu(rand_dist, ctrl_dist, alternative="two-sided", use_continuity=False)
    
    if sharey:
        ax.set_ylim((-10, 10))
        ax.yaxis.set_ticks(np.arange(-10, 11, 5))
        y_1 = 8
        y_2 = 6
        text_y_1 = 7.5
        text_y_2 = 5.5
    else:
        ax.set_ylim((np.min(rand_dist)-2, np.max(wt_dist)+3.5))
        y_1 = np.max(wt_dist)+1.85
        y_2 = np.max(wt_dist)+0.75
        text_y_1 = np.max(wt_dist)+1.65
        text_y_2 = np.max(wt_dist)+0.55
    
    # statistical annotation
    annotate_pval(ax, 0, 2, y_1, 0, text_y_1, rand_pval, fontsize)
    annotate_pval(ax, 1, 2, y_2, 0, text_y_2, scram_pval, fontsize)
    
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    if title:
        ax.set_title("%s" % (cell_type))
    if save:
        plt.savefig("%s/%s.pdf" % (figs_dir, plotname), dpi="figure", bbox_inches="tight")


# In[ ]:


def plot_activ_and_tiles(figsize, df, reps, color, palette, x_margin_percent, tss, x_tick_size, save, plotname):
    fig = plt.figure(figsize=(figsize))

    gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1], hspace=0)
    activ_ax = plt.subplot(gs[0])
    tile_ax = plt.subplot(gs[1])

    ## plot activities ##
    df["adjusted_tile_start"] = df["actual_start"] + ((df["actual_end"] - df["actual_start"])/2)
    cols = list(reps)
    cols.extend(["element_id", "element", "adjusted_tile_start", "combined_sig"])
    df_sub = df[cols]

    # sort and melt
    df_sub = df_sub.sort_values(by="adjusted_tile_start")
    df_melt = pd.melt(df_sub, id_vars=["element_id", "element", "adjusted_tile_start", "combined_sig"])
    sns.swarmplot(data=df_melt, x="adjusted_tile_start", y="value", ax=activ_ax, color="lightslategrey", size=5,
                  hue="combined_sig", palette=palette)
    sns.boxplot(data=df_melt, x="adjusted_tile_start", y="value", ax=activ_ax, 
                showcaps=False, showfliers=False, whiskerprops={'linewidth':0}, 
                zorder=1, hue="combined_sig", palette=palette, dodge=False)
    
    # fix boxplot colors
    for i,artist in enumerate(activ_ax.artists):
        # Set the linecolor on the artist to the facecolor, and set the facecolor to None
        col = artist.get_facecolor()
        artist.set_edgecolor(col)
        artist.set_facecolor('None')

        # Each box has 6 associated Line2D objects (to make the whiskers, fliers, etc.)
        # Loop over them here, and use the same colour as above
        for j in range(i*3,i*3+3):
            line = activ_ax.lines[j]
            line.set_color(col)
            line.set_mfc(col)
            line.set_mec(col)
        
    add_margin(activ_ax, x=x_margin_percent, y=0)
    activ_ax.xaxis.set_visible(False)
    activ_ax.set_ylabel("MPRA activity")
    activ_ax.legend_.remove()

    ## plot tiles ##
    for i, elem_id in enumerate(df.sort_values(by="tile_number").element_id):
        tile_num = df[df["element_id"] == elem_id]["tile_number"].iloc[0]
        tile_start = df[df["element_id"] == elem_id]["actual_start"].iloc[0]
        tile_end = df[df["element_id"] == elem_id]["actual_end"].iloc[0]
        tile_strand = df[df["element_id"] == elem_id]["strand"].iloc[0]
        if i % 2 == 0:
            y = 0.5
        else:
            y = 0
        tile_ax.plot((tile_start, tile_end), (y, y), color="black", linewidth=5, solid_capstyle="butt")

    tile_ax.get_xaxis().get_major_formatter().set_useOffset(False)
    tile_ax.get_xaxis().get_major_formatter().set_scientific(False)

    tile_ax.plot((tss, tss), (0.75, 1.4), '-', color=color)

    if tile_strand == "+":
        tile_ax.arrow(tss, 1.4, 40, 0, fc=color, ec=color, head_width=0.45, head_length=30, linewidth=1)
    else:
        tile_ax.arrow(tss, 1.4, -40, 0, fc=color, ec=color, head_width=0.45, head_length=30, linewidth=1)

    #tile_ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    tile_ax.set_xticks(np.arange(df.actual_start.min(), df.actual_start.max()+200, 144))
    plt.setp(tile_ax.get_xticklabels(), fontsize=x_tick_size)
    tile_ax.set_ylim((-0.5, 1.75))
    tile_ax.yaxis.set_visible(False)
    tile_ax.spines["left"].set_visible(False)
    tile_ax.spines["right"].set_visible(False)
    tile_ax.spines["bottom"].set_visible(False)
    
    if save:
        fig.savefig(plotname, dpi="figure", bbox_inches="tight")
    plt.show()


# In[1]:


def plot_dendrogram(linkage, max_dist, title):
    
    plt.figure(figsize=(25, 8))
    dg = hierarchy.dendrogram(linkage, show_leaf_counts=True)

    dists = []
    for i, d, c in zip(dg['icoord'], dg['dcoord'], dg['color_list']):
                x = 0.5 * sum(i[1:3])
                y = d[1]
                plt.plot(x, y, 'o', c=c)
                if y > max_dist:
                    plt.annotate("%.3g" % y, (x, y), xytext=(0, -5),
                                 textcoords='offset points',
                                 va='top', ha='center')
                dists.append(y)

    plt.axhline(y=max_dist)
    plt.title(title)
    plt.show()
    return dists


# In[ ]:


def pearsonfunc(x, y, **kws):
    r, p = stats.pearsonr(x, y)
    ax = plt.gca()
    ax.annotate("pearson r = {:.2f}\np = {:.2e}".format(r, Decimal(p)),
                xy=(.1, .9), xycoords=ax.transAxes)

def spearmanfunc(x, y, **kws):
    r, p = stats.spearmanr(x, y)
    ax = plt.gca()
    ax.annotate("spearman r = {:.2f}\np = {:.2e}".format(r, Decimal(p)),
                xy=(.1, .9), xycoords=ax.transAxes)


# In[ ]:


def plot_peaks_and_tfbs(figsize, seq_len, seq_name, cell, scores, yerrs, motif_vals, bases, plotname, save):
    
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(3, 1, height_ratios=[4, 3, 1], hspace=0.2)
    peak_ax = plt.subplot(gs[0])
    motif_ax = plt.subplot(gs[1])
    
    # plot deletion values
    xs = list(range(0, seq_len))
    peak_ax.bar(xs, scores, yerr=yerrs, color="lightgray", edgecolor="gray", linewidth=0.5, ecolor="gray", 
                error_kw={"elinewidth": 0.75})
    
    # labels
    peak_ax.set_xlim((-0.5, seq_len))
    peak_ax.set_xlabel("")
    peak_ax.set_ylabel("log2(del/WT)", fontsize=5)
    peak_ax.xaxis.set_visible(False)
    peak_ax.set_title("filtered scores and peaks: %s (%s)" % (seq_name, cell))
    
    # plot motif nums
    xs = list(range(0, seq_len))
    max_motif_val = np.nanmax(np.abs(motif_vals))
    motif_ax.axhline(y=0, color="darkgrey", linewidth=0.5, linestyle="dashed")
    motif_ax.plot(xs, motif_vals, color="black", linewidth=0.75, zorder=10)
    
    # labels
    motif_ax.set_xlim((-0.5, seq_len))
    motif_ax.set_ylim((-max_motif_val-1, max_motif_val+1))
    motif_ax.set_xlabel("nucleotide number")
    motif_ax.set_ylabel(r'$\Delta$ motifs', fontsize=5)
    motif_ax.xaxis.set_visible(False)
    
    plt.show()
    if save:
        fig.savefig("%s.pdf" % (plotname), dpi="figure", bbox_inches="tight", transparent=True)
    plt.close()


# In[ ]:


def paired_swarmplots_w_pval(n_rows, n_cols, figsize, snp_df, data_df, fontsize, figs_dir, plotname, save):
    fig, axarr = plt.subplots(figsize=figsize, squeeze=False)
    pal = {"ref": "grey", "alt": sns.color_palette()[2]}
    median_width = 0.3
    
    # make axes objects
    axes = []
    counter = 0
    for r in range(n_rows):
        for c in range(n_cols):
            if counter < len(snp_df):
                ax = plt.subplot2grid((n_rows, n_cols), (r, c))
                axes.append(ax)
            counter += 1

    # add plots
    counter = 0
    for i, row in snp_df.iterrows():
        ax = axes[counter]
        wt_id = row.wt_id
        snp_id = row.unique_id
        df = data_df[data_df["unique_id"].isin([wt_id, snp_id])]
        df = df.sort_values(by="wt_or_snp", ascending=False)
        if not "NA" in str(row.combined_padj) and not pd.isnull(row.combined_padj):
            sns.swarmplot(data=df, x="wt_or_snp", y="rep_mean", ax=ax, palette=pal)
            
            for tick, text in zip(ax.get_xticks(), ax.get_xticklabels()):
                snp = text.get_text()

                # calculate the median value for all replicates of either X or Y
                median_val = df[df["wt_or_snp"]==snp]["rep_mean"].median()

                # plot horizontal lines across the column, centered on the tick
                ax.plot([tick-median_width/2, tick+median_width/2], [median_val, median_val],
                        lw=2, color='k', zorder=10)
            
        else:
            sns.swarmplot(data=df, x="wt_or_snp", y="rep_mean", ax=ax, color="lightgray")
            
            for tick, text in zip(ax.get_xticks(), ax.get_xticklabels()):
                snp = text.get_text()

                # calculate the median value for all replicates of either X or Y
                median_val = df[df["wt_or_snp"]==snp]["rep_mean"].median()

                # plot horizontal lines across the column, centered on the tick
                ax.plot([tick-median_width/2, tick+median_width/2], [median_val, median_val],
                        lw=2, color='k', zorder=10)
        if len(row.SNP) > 50:
            ax.set_title("SNP: long haplotype", fontsize=fontsize)
        else:
            ax.set_title("SNP: %s" % row.SNP, fontsize=fontsize)
        ax.set_ylim((df.rep_mean.min()-2, df.rep_mean.max()+3))
        ax.set_ylabel("")
        ax.set_xlabel("")

        # statistical annotation
        x1, x2 = 0, 1   # columns (first column: 0, see plt.xticks())
        y, h, col = df["rep_mean"].max() + 0.75, 0, "black"
        ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=0.5, c=col)
        if not "NA" in str(row.combined_padj) and not pd.isnull(row.combined_padj):
            if row.combined_padj < 0.0001:
                #text = "{:.2e}".format(Decimal(val))
                text = "**"
            elif row.combined_padj < 0.05:
                #text = "%.4f" % val
                text = "*"
            else:
                text = "n.s."
        else:
            text = "tile activities not sig"
        ax.text((x1+x2)*.5, y+h, text, ha='center', va='bottom', color=col, size=fontsize)
            

        counter += 1

    plt.tight_layout()
    plt.subplots_adjust(wspace = .3)
    plt.show()
    if save:
        fig.savefig("%s/%s.pdf" % (figs_dir, plotname), dpi="figure", bbox_inches="tight")


# In[ ]:


def plot_peaks_and_snps(figsize, seq_len, seq_name, widths, scores, yerrs, scaled_scores, snp_vals, snp_sigs, bases, plotname, figs_dir, save):
    sns.set(style="ticks", font="Helvetica", context="paper", rc={"font.size":7,"axes.titlesize":7,
                                                              "axes.labelsize":7, 'axes.linewidth':0.5,
                                                              "legend.fontsize":6, "xtick.labelsize":6,
                                                              "ytick.labelsize":6, "xtick.major.size": 3.0,
                                                              "ytick.major.size": 3.0, "axes.edgecolor": "black",
                                                              "xtick.major.pad": 3.0, "ytick.major.pad": 3.0})
    
    snp_pal = {"sig": "firebrick", "not sig": "darkgray", "NA__too_many_rep_NAs": "darkgray", "NA": "white"}
    
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(3, 1, height_ratios=[5, 1, 2], hspace=0.1)
    peak_ax = plt.subplot(gs[0])
    snp_ax = plt.subplot(gs[2])
    seq_ax = plt.subplot(gs[1])
    
    ### peaks figure ###
    # plot peak locations
    for w in widths:
        peak_ax.axvline(x=w[0], color="gray", linestyle="solid", linewidth=0.5, zorder=1)
        snp_ax.axvline(x=w[0], color="gray", linestyle="solid", linewidth=0.5, zorder=1)
        peak_ax.axvline(x=w[1], color="gray", linestyle="solid", linewidth=0.5, zorder=1)
        snp_ax.axvline(x=w[1], color="gray", linestyle="solid", linewidth=0.5, zorder=1)
        peak_ax.axvspan(w[0], w[1], alpha=0.5, color="gainsboro", zorder=1)
        snp_ax.axvspan(w[0], w[1], alpha=0.5, color="gainsboro", zorder=1)
    
    # plot deletion values
    xs = list(range(0, seq_len))
    peak_ax.bar(xs, scores, yerr=yerrs, color="lightgray", edgecolor="gray", linewidth=0.5, ecolor="gray", 
                error_kw={"elinewidth": 0.75})
    
    # labels
    peak_ax.set_xlim((-0.5, seq_len))
    peak_ax.set_xlabel("")
    peak_ax.set_ylabel("log2(del/WT)")
    peak_ax.xaxis.set_visible(False)
    peak_ax.set_title(seq_name)
    
    # plot snp values
    xs = list(range(0, seq_len))
    snp_colors = [snp_pal[x] for x in snp_sigs]
    snp_ax.scatter(xs, snp_vals, s=12, color=snp_colors, edgecolor="black", linewidth=0.5, zorder=10)
    for i in range(seq_len):
        l2fc = snp_vals[i]
        snp_ax.plot([i, i], [0, l2fc], lw=1, color="k")
    snp_ax.axhline(y=0, lw=1, color="k", zorder=1)
    
    # labels
    snp_ax.set_xlim((-0.5, seq_len))
    snp_ax.set_xlabel("nucleotide number")
    snp_ax.set_ylabel("log2(alt/ref)")
    snp_ax.xaxis.set_visible(False)
    
    ### seq logo ###
    mpl.rcParams["font.family"] = "Arial"
    scaled_scores = scale_range(scaled_scores, 0.5, 2.0)
    
    font = FontProperties()
    font.set_size(6)
    font.set_weight("bold")
    
    seq_ax.set_xticks(range(1,len(scaled_scores)+1))
    seq_ax.set_ylim((0, 2))
    seq_ax.axis("off")
    trans_offset = transforms.offset_copy(seq_ax.transData, 
                                          fig=fig, 
                                          x=1, 
                                          y=0, 
                                          units="dots")
    
    for i in range(0, len(scaled_scores)):
        score = scaled_scores[i]
        base = bases[i]
        color = COLOR_DICT[base]
        txt = seq_ax.text(i+0.25, 0, base, transform=trans_offset,fontsize=6, color=color, 
                          ha="center", fontproperties=font)
        txt.set_path_effects([Scale(1.0, score)])
        fig.canvas.draw()
        trans_offset = transforms.offset_copy(seq_ax.transData, fig=fig, x=1, y=0, units='points')
    
    #plt.tight_layout()
    plt.show()
    if save:
        fig.savefig("%s/%s" % (figs_dir, plotname), dpi="figure", bbox_inches="tight")
    plt.close()


# In[ ]:


def getOverlap(a, b):
    return max(a[0], b[0]) - min(a[1], b[1])


# In[ ]:


def plot_peaks_and_fimo(figsize, seq_len, seq_name, widths, scores, yerrs, scaled_scores, bases, motif_pos, motif_names, plotname, figs_dir, save):
     
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(3, 1, height_ratios=[5, 1, 2], hspace=0.1)
    peak_ax = plt.subplot(gs[0])
    seq_ax = plt.subplot(gs[1])
    motif_ax = plt.subplot(gs[2])
    
    ### peaks figure ###
    # plot peak locations
    for w in widths:
        peak_ax.axvline(x=w[0], color="gray", linestyle="solid", linewidth=0.5, zorder=1)
        peak_ax.axvline(x=w[1], color="gray", linestyle="solid", linewidth=0.5, zorder=1)
        peak_ax.axvspan(w[0], w[1], alpha=0.5, color="gainsboro", zorder=1)
    
    # plot deletion values
    xs = list(range(0, seq_len))
    peak_ax.bar(xs, scores, yerr=yerrs, color="lightgray", edgecolor="gray", linewidth=0.5, ecolor="gray", 
                error_kw={"elinewidth": 0.75})
    
    # labels
    peak_ax.set_xlim((-0.5, seq_len))
    peak_ax.set_xlabel("")
    peak_ax.set_ylabel("log2(del/WT)")
    peak_ax.xaxis.set_visible(False)
    peak_ax.set_title(seq_name)
    
    # plot motif locations
    xs = list(range(0, seq_len))
    prev_plotted = {}
    
    # iterate through things plotted at each prev_y value
    # if any overlaps, move
    for i, pos in enumerate(motif_pos):
        #print("")
        #print("i: %s, pos: %s" % (i, pos))
        plotted = False
        if i == 0:
            #print("first motif, plotting at y=0")
            motif_ax.plot([pos[0], pos[1]], [0, 0], color="darkgrey", linewidth=2, solid_capstyle="butt")
            plotted = True
            prev_plotted[0] = [pos]
            continue
        for prev_y in sorted(prev_plotted.keys(), reverse=True):
            vals = prev_plotted[prev_y]
            overlaps = []
            for prev_pos in vals:
                overlaps.append(getOverlap(prev_pos, pos))
            if any(x < 0 for x in overlaps):
                #print("motif overlaps w/ %s, continuing" % (prev_y))
                continue
            else:
                if not plotted:
                    #print("motif doesn't overlap anything at y=%s, plotting" % prev_y)
                    motif_ax.plot([pos[0], pos[1]], [prev_y, prev_y], color="darkgrey", linewidth=2, 
                                  solid_capstyle="butt")
                    if prev_y not in prev_plotted:
                        prev_plotted[prev_y] = [pos]
                    else:
                        new_vals = list(prev_plotted[prev_y])
                        new_vals.extend([pos])
                        prev_plotted[prev_y] = new_vals
                    plotted = True
        if not plotted:
            prev_y -= 0.25
            #print("motif overlaps at all prev_y, plotting at %s" % prev_y)
            motif_ax.plot([pos[0], pos[1]], [prev_y, prev_y], color="darkgrey", linewidth=2, 
                          solid_capstyle="butt")
            if prev_y not in prev_plotted:
                prev_plotted[prev_y] = [pos]
            else:
                new_vals = list(prev_plotted[prev_y])
                new_vals.extend([pos])
                prev_plotted[prev_y] = new_vals
            plotted = True
        #print(prev_plotted)
        
    min_y = np.min(list(prev_plotted.keys()))

    # labels
    motif_ax.set_xlim((-0.5, seq_len))
    motif_ax.set_ylim((min_y - 0.25, 0.25))
    motif_ax.set_xlabel("nucleotide number")
    motif_ax.set_ylabel("")
    motif_ax.xaxis.set_visible(False)
    motif_ax.yaxis.set_visible(False)
    motif_ax.axis("off")
    
    ### seq logo ###
    mpl.rcParams["font.family"] = "Arial"
    scaled_scores = scale_range(scaled_scores, 0.5, 2.0)
    
    font = FontProperties()
    font.set_size(6)
    font.set_weight("bold")
    
    seq_ax.set_xticks(range(1,len(scaled_scores)+1))
    seq_ax.set_ylim((0, 2))
    seq_ax.axis("off")
    trans_offset = transforms.offset_copy(seq_ax.transData, 
                                          fig=fig, 
                                          x=1, 
                                          y=0, 
                                          units="dots")
    
    for i in range(0, len(scaled_scores)):
        score = scaled_scores[i]
        base = bases[i]
        color = COLOR_DICT[base]
        txt = seq_ax.text(i+0.25, 0, base, transform=trans_offset,fontsize=6, color=color, 
                          ha="center", fontproperties=font)
        txt.set_path_effects([Scale(1.0, score)])
        fig.canvas.draw()
        trans_offset = transforms.offset_copy(seq_ax.transData, fig=fig, x=1, y=0, units='points')
    
    #plt.tight_layout()
    plt.show()
    if save:
        fig.savefig("%s/%s" % (figs_dir, plotname), dpi="figure", bbox_inches="tight")
    plt.close()

