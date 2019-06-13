
# coding: utf-8

# # 00__motifs

# In[1]:


import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import sys

from scipy.stats import spearmanr

# import utils
sys.path.append("../../../utils")
from plotting_utils import *
from misc_utils import *
from norm_utils import *

get_ipython().magic('matplotlib inline')
get_ipython().magic("config InlineBackend.figure_format = 'svg'")
mpl.rcParams['figure.autolayout'] = False


# In[2]:


sns.set(**PAPER_PRESET)
fontsize = PAPER_FONTSIZE


# In[3]:


np.random.seed(2019)


# ## variables

# In[4]:


data_dir = "../../../data/02__mpra/02__activs"
human_max_f = "%s/human_TSS_vals.max_tile.txt" % data_dir
mouse_max_f = "%s/mouse_TSS_vals.max_tile.txt" % data_dir


# In[5]:


results_dir = "../../../data/02__mpra/03__results"
results_f = "%s/native_cis_trans_effects_data.txt" % results_dir


# In[6]:


motif_info_dir = "../../../misc/02__motif_info"
motif_map_f = "%s/00__lambert_et_al_files/00__metadata/curated_motif_map.txt" % motif_info_dir


# In[7]:


motif_dir = "../../../data/04__mapped_motifs"
human_motifs_f = "%s/hg19_human_curated_tfs_out/fimo.txt.gz" % motif_dir
mouse_motifs_f = "%s/mm9_human_curated_tfs_out/fimo.txt.gz" % motif_dir


# In[8]:


expr_dir = "../../../data/03__rna_seq/04__TF_expr"
orth_expr_f = "%s/orth_TF_expression.txt" % expr_dir
human_expr_f = "%s/hESC_TF_expression.txt" % expr_dir
mouse_expr_f = "%s/mESC_TF_expression.txt" % expr_dir


# In[9]:


orth_f = "../../../misc/01__ensembl_orthologs/ensembl96_human_mouse_orths.txt.gz"


# ## 1. import data

# In[10]:


results = pd.read_table(results_f, sep="\t")
results.head()


# In[11]:


human_max = pd.read_table(human_max_f, sep="\t")
mouse_max = pd.read_table(mouse_max_f, sep="\t")
human_max.head()


# In[12]:


motif_map = pd.read_table(motif_map_f)
motif_map.head()


# In[13]:


human_motifs = pd.read_table(human_motifs_f, sep="\t")
human_motifs.head()


# In[14]:


mouse_motifs = pd.read_table(mouse_motifs_f, sep="\t")
mouse_motifs.head()


# In[15]:


orth_expr = pd.read_table(orth_expr_f, sep="\t")
orth_expr.head()


# In[16]:


human_expr = pd.read_table(human_expr_f, sep="\t")
human_expr.head()


# In[17]:


mouse_expr = pd.read_table(mouse_expr_f, sep="\t")
mouse_expr.head()


# In[18]:


orth = pd.read_table(orth_f, sep="\t")
orth.head()


# ## 2. parse motif files

# In[19]:


human_motifs = human_motifs.merge(motif_map, left_on="#pattern name", right_on="index", how="left")
human_motifs["hg19_id"] = human_motifs["sequence name"].str.split("__", expand=True)[1]
human_motifs["tile_num"] = human_motifs["sequence name"].str.split(";", expand=True)[0].str.split("__", expand=True)[2]
human_motifs["tss_strand"] = human_motifs["sequence name"].str[-2]
human_motifs.sample(5)


# In[20]:


mouse_motifs = mouse_motifs.merge(motif_map, left_on="#pattern name", right_on="index", how="left")
mouse_motifs["mm9_id"] = mouse_motifs["sequence name"].str.split("__", expand=True)[1]
mouse_motifs["tss_strand"] = mouse_motifs["sequence name"].str[-2]
mouse_motifs["tile_num"] = mouse_motifs["sequence name"].str.split(";", expand=True)[0].str.split("__", expand=True)[2]
mouse_motifs.sample(5)


# In[21]:


# limit motif tiles to those that are max tiles (since we mapped motifs in both tiles)
human_max_motifs = human_max.merge(human_motifs, left_on=["tss_id", "tss_tile_num"],
                                   right_on=["hg19_id", "tile_num"], how="left").reset_index()
human_max_motifs = human_max_motifs[~pd.isnull(human_max_motifs["element"])]
human_max_motifs.head()


# In[22]:


# limit motif tiles to those that are max tiles (since we mapped motifs in both tiles)
mouse_max_motifs = mouse_max.merge(mouse_motifs, left_on=["tss_id", "tss_tile_num"],
                                   right_on=["mm9_id", "tile_num"], how="left").reset_index()
mouse_max_motifs = mouse_max_motifs[~pd.isnull(mouse_max_motifs["element"])]
mouse_max_motifs.head()


# ## 3. find motifs enriched in trans effects

# ### human sequences
# [since we have to look at motifs in each sequence separately, analyze them separately]

# In[23]:


uniq_human_tfs = human_max_motifs["gene_name"].unique()
len(uniq_human_tfs)


# In[ ]:


results.trans_status_detail_human.value_counts()


# In[ ]:


human_trans_results = {}
for i, tf in enumerate(uniq_human_tfs):
    if i % 50 == 0:
        print(i)
        
    # do directional analysis: first, high in human
    sub_motifs = human_max_motifs[human_max_motifs["gene_name"] == tf]["hg19_id"].unique()
    
    sub_trans = results[results["trans_status_detail_human"] == "trans effect\n(higher in human)"]["hg19_id"].unique()
    sub_no_trans = results[results["trans_status_detail_human"] != "trans effect\n(higher in human)"]["hg19_id"].unique()
    
    n_hu_trans_w_motif = len([x for x in sub_trans if x in sub_motifs])
    n_hu_trans_wo_motif = len([x for x in sub_trans if x not in sub_motifs])
    n_hu_no_trans_w_motif = len([x for x in sub_no_trans if x in sub_motifs])
    n_hu_no_trans_wo_motif = len([x for x in sub_no_trans if x not in sub_motifs])
    
    arr = np.zeros((2, 2))
    arr[0, 0] = n_hu_trans_w_motif
    arr[0, 1] = n_hu_trans_wo_motif
    arr[1, 0] = n_hu_no_trans_w_motif
    arr[1, 1] = n_hu_no_trans_wo_motif
    
    hu_odds, hu_p = stats.fisher_exact(arr)
    
    # next, high in mouse
    sub_trans = results[results["trans_status_detail_human"] == "trans effect\n(higher in mouse)"]["hg19_id"].unique()
    sub_no_trans = results[results["trans_status_detail_human"] != "trans effect\n(higher in mouse)"]["hg19_id"].unique()
    
    n_mo_trans_w_motif = len([x for x in sub_trans if x in sub_motifs])
    n_mo_trans_wo_motif = len([x for x in sub_trans if x not in sub_motifs])
    n_mo_no_trans_w_motif = len([x for x in sub_no_trans if x in sub_motifs])
    n_mo_no_trans_wo_motif = len([x for x in sub_no_trans if x not in sub_motifs])
    
    arr = np.zeros((2, 2))
    arr[0, 0] = n_mo_trans_w_motif
    arr[0, 1] = n_mo_trans_wo_motif
    arr[1, 0] = n_mo_no_trans_w_motif
    arr[1, 1] = n_mo_no_trans_wo_motif
    
    mo_odds, mo_p = stats.fisher_exact(arr)
    human_trans_results[tf] = {"high_in_HUES64_odds": hu_odds, "high_in_HUES64_pval": hu_p, 
                               "n_high_in_HUES64_trans_w_motif": n_hu_trans_w_motif, 
                               "n_high_in_HUES64_trans_wo_motif": n_hu_trans_wo_motif, 
                               "n_no_high_in_HUES64_trans_w_motif": n_hu_no_trans_w_motif,
                               "n_no_high_in_HUES64_trans_wo_motif": n_hu_no_trans_wo_motif,
                               "high_in_mESC_odds": mo_odds, "high_in_mESC_pval": mo_p, 
                               "n_high_in_mESC_trans_w_motif": n_mo_trans_w_motif, 
                               "n_high_in_mESC_trans_wo_motif": n_mo_trans_wo_motif, 
                               "n_no_high_in_mESC_trans_w_motif": n_mo_no_trans_w_motif,
                               "n_no_high_in_mESC_trans_wo_motif": n_mo_no_trans_wo_motif}
    
human_trans_results = pd.DataFrame.from_dict(human_trans_results, orient="index").reset_index()
human_trans_results.sort_values(by="high_in_HUES64_pval").head()


# In[ ]:


human_trans_results["high_in_HUES64_padj"] = multicomp.multipletests(human_trans_results["high_in_HUES64_pval"], method="fdr_bh")[1]
len(human_trans_results[human_trans_results["high_in_HUES64_padj"] < 0.05])


# In[ ]:


human_trans_results["high_in_mESC_padj"] = multicomp.multipletests(human_trans_results["high_in_mESC_pval"], method="fdr_bh")[1]
len(human_trans_results[human_trans_results["high_in_mESC_padj"] < 0.05])


# In[ ]:


list(human_trans_results[human_trans_results["high_in_HUES64_padj"] < 0.05]["index"])


# ### mouse sequences

# In[ ]:


uniq_mouse_tfs = mouse_max_motifs["gene_name"].unique()
len(uniq_mouse_tfs)


# In[ ]:


results.trans_status_detail_mouse.value_counts()


# In[ ]:


mouse_trans_results = {}
for i, tf in enumerate(uniq_mouse_tfs):
    if i % 50 == 0:
        print(i)
        
    # do directional analysis: first, high in human
    sub_motifs = mouse_max_motifs[mouse_max_motifs["gene_name"] == tf]["mm9_id"].unique()
    
    sub_trans = results[results["trans_status_detail_mouse"] == "trans effect\n(higher in human)"]["mm9_id"].unique()
    sub_no_trans = results[results["trans_status_detail_mouse"] != "trans effect\n(higher in human)"]["mm9_id"].unique()
    
    n_hu_trans_w_motif = len([x for x in sub_trans if x in sub_motifs])
    n_hu_trans_wo_motif = len([x for x in sub_trans if x not in sub_motifs])
    n_hu_no_trans_w_motif = len([x for x in sub_no_trans if x in sub_motifs])
    n_hu_no_trans_wo_motif = len([x for x in sub_no_trans if x not in sub_motifs])
    
    arr = np.zeros((2, 2))
    arr[0, 0] = n_hu_trans_w_motif
    arr[0, 1] = n_hu_trans_wo_motif
    arr[1, 0] = n_hu_no_trans_w_motif
    arr[1, 1] = n_hu_no_trans_wo_motif
    
    hu_odds, hu_p = stats.fisher_exact(arr)
    
    # next, high in mouse
    sub_trans = results[results["trans_status_detail_mouse"] == "trans effect\n(higher in mouse)"]["mm9_id"].unique()
    sub_no_trans = results[results["trans_status_detail_mouse"] != "trans effect\n(higher in mouse)"]["mm9_id"].unique()
    
    n_mo_trans_w_motif = len([x for x in sub_trans if x in sub_motifs])
    n_mo_trans_wo_motif = len([x for x in sub_trans if x not in sub_motifs])
    n_mo_no_trans_w_motif = len([x for x in sub_no_trans if x in sub_motifs])
    n_mo_no_trans_wo_motif = len([x for x in sub_no_trans if x not in sub_motifs])
    
    arr = np.zeros((2, 2))
    arr[0, 0] = n_mo_trans_w_motif
    arr[0, 1] = n_mo_trans_wo_motif
    arr[1, 0] = n_mo_no_trans_w_motif
    arr[1, 1] = n_mo_no_trans_wo_motif
    
    mo_odds, mo_p = stats.fisher_exact(arr)
    mouse_trans_results[tf] = {"high_in_HUES64_odds": hu_odds, "high_in_HUES64_pval": hu_p, 
                               "n_high_in_HUES64_trans_w_motif": n_hu_trans_w_motif, 
                               "n_high_in_HUES64_trans_wo_motif": n_hu_trans_wo_motif, 
                               "n_no_high_in_HUES64_trans_w_motif": n_hu_no_trans_w_motif,
                               "n_no_high_in_HUES64_trans_wo_motif": n_hu_no_trans_wo_motif,
                               "high_in_mESC_odds": mo_odds, "high_in_mESC_pval": mo_p, 
                               "n_high_in_mESC_trans_w_motif": n_mo_trans_w_motif, 
                               "n_high_in_mESC_trans_wo_motif": n_mo_trans_wo_motif, 
                               "n_no_high_in_mESC_trans_w_motif": n_mo_no_trans_w_motif,
                               "n_no_high_in_mESC_trans_wo_motif": n_mo_no_trans_wo_motif}
    
mouse_trans_results = pd.DataFrame.from_dict(mouse_trans_results, orient="index").reset_index()
mouse_trans_results.sort_values(by="high_in_mESC_pval").head()


# In[ ]:


mouse_trans_results["high_in_HUES64_padj"] = multicomp.multipletests(mouse_trans_results["high_in_HUES64_pval"], method="fdr_bh")[1]
len(mouse_trans_results[mouse_trans_results["high_in_HUES64_padj"] < 0.05])


# In[ ]:


mouse_trans_results["high_in_mESC_padj"] = multicomp.multipletests(mouse_trans_results["high_in_mESC_pval"], method="fdr_bh")[1]
len(mouse_trans_results[mouse_trans_results["high_in_mESC_padj"] < 0.05])


# In[ ]:


list(mouse_trans_results[mouse_trans_results["high_in_HUES64_padj"] < 0.05]["index"])


# ## 4. plot trans enrichment results

# In[ ]:


sig_human = human_trans_results[human_trans_results["high_in_HUES64_padj"] < 0.05][["index", "high_in_HUES64_odds"]].set_index("index")
sig_human = sig_human.sort_values(by="high_in_HUES64_odds", ascending=False)
sig_human


# In[ ]:


grid_kws = {"height_ratios": (.75, .25), "hspace": 3}
f, (ax, cbar_ax) = plt.subplots(2, gridspec_kw=grid_kws, figsize=(3, 1.5))
ax = sns.heatmap(sig_human.T, ax=ax, cbar_ax=cbar_ax, cbar_kws={"orientation": "horizontal"}, cmap="Greys", 
                 linewidth=0.25, xticklabels=sig_human.index, yticklabels="", vmin=1)
ax.set_xlabel("")
f.savefig("trans_motif_enrichments.human_high_in_HUES64.pdf", dpi="figure", bbox_inches="tight")


# In[ ]:


sig_mouse = mouse_trans_results[mouse_trans_results["high_in_HUES64_padj"] < 0.05][["index", "high_in_HUES64_odds"]].set_index("index")
sig_mouse = sig_mouse.sort_values(by="high_in_HUES64_odds", ascending=False)
sig_mouse


# In[ ]:


grid_kws = {"height_ratios": (.75, .25), "hspace": 3}
f, (ax, cbar_ax) = plt.subplots(2, gridspec_kw=grid_kws, figsize=(3, 1.5))
ax = sns.heatmap(sig_mouse.T, ax=ax, cbar_ax=cbar_ax, cbar_kws={"orientation": "horizontal"}, cmap="Greys", 
                 linewidth=0.25, xticklabels=sig_mouse.index, yticklabels="", vmin=1)
ax.set_xlabel("")
f.savefig("trans_motif_enrichments.mouse_high_in_HUES64.pdf", dpi="figure", bbox_inches="tight")


# In[ ]:


sig_both = sig_human.append(sig_mouse)
sig_both = sig_both.sort_values(by="high_in_HUES64_odds", ascending=False)


# In[ ]:


grid_kws = {"height_ratios": (.75, .25), "hspace": 3}
f, (ax, cbar_ax) = plt.subplots(2, gridspec_kw=grid_kws, figsize=(3, 1.5))
ax = sns.heatmap(sig_both.T, ax=ax, cbar_ax=cbar_ax, cbar_kws={"orientation": "horizontal"}, cmap="Greys", 
                 linewidth=0.25, xticklabels=sig_both.index, yticklabels="", vmin=1)
ax.set_xlabel("")
f.savefig("trans_motif_enrichments.high_in_HUES64.pdf", dpi="figure", bbox_inches="tight")


# In[ ]:





# ## 5. plot trans effect sizes for each motif

# In[41]:


human_max_motifs_sub = human_max_motifs[["hg19_id", "gene_name"]].drop_duplicates()
mouse_max_motifs_sub = mouse_max_motifs[["mm9_id", "gene_name"]].drop_duplicates()


# In[42]:


results.columns


# In[43]:


# plot trans effects of each motif +/- individually
sig_human = sig_human.reset_index()
for sig_tf in sig_human["index"].unique():
    print(sig_tf)
    
    sub_motif_ids = human_max_motifs[human_max_motifs["gene_name"] == sig_tf]["hg19_id"].unique()
    data_w_motif = results[results["hg19_id"].isin(sub_motif_ids)]
    data_no_motif = results[~results["hg19_id"].isin(sub_motif_ids)]
    print("# w/ motif: %s, # w/o: %s" % (len(data_w_motif), len(data_no_motif)))
    
    data_w_motif["motif_status"] = sig_tf
    data_no_motif["motif_status"] = "no motif"
    tmp = data_w_motif.append(data_no_motif)
    
    # calc p-vals b/w dists
    dist1 = np.asarray(data_w_motif["logFC_trans_human"])
    dist2 = np.asarray(data_no_motif["logFC_trans_human"])

    dist1 = dist1[~np.isnan(dist1)]
    dist2 = dist2[~np.isnan(dist2)]

    u, pval = stats.mannwhitneyu(dist1, dist2, alternative="two-sided", use_continuity=False)
    print(pval)
    
    # boxplot
    fig, ax = plt.subplots(figsize=(1.5, 1.5), nrows=1, ncols=1)
    order = ["no motif", sig_tf]
    pal = {"no motif": "gray", sig_tf: sns.color_palette("Set2")[2]}
    sns.boxplot(data=tmp, x="motif_status", y="logFC_trans_human", ax=ax, order=order, palette=pal,
                flierprops = dict(marker='o', markersize=5), linewidth=1)
    mimic_r_boxplot(ax)
    ax.set_xlabel("")
    ax.set_ylabel("trans effect size\nfor human sequence")
    plt.show()
    fig.savefig("%s_trans_effect_boxplot.pdf" % sig_tf, dpi="figure", bbox_inches="tight")


# In[44]:


# plot trans effects of each motif +/- individually
sig_mouse = sig_mouse.reset_index()
for sig_tf in sig_mouse["index"].unique():
    print(sig_tf)
    
    sub_motif_ids = mouse_max_motifs[mouse_max_motifs["gene_name"] == sig_tf]["mm9_id"].unique()
    data_w_motif = results[results["mm9_id"].isin(sub_motif_ids)]
    data_no_motif = results[~results["mm9_id"].isin(sub_motif_ids)]
    print("# w/ motif: %s, # w/o: %s" % (len(data_w_motif), len(data_no_motif)))
    
    data_w_motif["motif_status"] = sig_tf
    data_no_motif["motif_status"] = "no motif"
    tmp = data_w_motif.append(data_no_motif)
    
    # calc p-vals b/w dists
    dist1 = np.asarray(data_w_motif["logFC_trans_mouse"])
    dist2 = np.asarray(data_no_motif["logFC_trans_mouse"])

    dist1 = dist1[~np.isnan(dist1)]
    dist2 = dist2[~np.isnan(dist2)]

    u, pval = stats.mannwhitneyu(dist1, dist2, alternative="two-sided", use_continuity=False)
    print(pval)
    
    # boxplot
    fig, ax = plt.subplots(figsize=(1.5, 1.5), nrows=1, ncols=1)
    order = ["no motif", sig_tf]
    pal = {"no motif": "gray", sig_tf: sns.color_palette("Set2")[2]}
    sns.boxplot(data=tmp, x="motif_status", y="logFC_trans_mouse", ax=ax, order=order, palette=pal,
                flierprops = dict(marker='o', markersize=5), linewidth=1)
    mimic_r_boxplot(ax)
    ax.set_xlabel("")
    ax.set_ylabel("trans effect size\nfor mouse sequence")
    plt.show()
    fig.savefig("%s_trans_effect_boxplot.pdf" % sig_tf, dpi="figure", bbox_inches="tight")


# ## 6. examine expression changes related to trans effects

# In[45]:


len(orth_expr)


# In[46]:


trans_tfs = list(sig_both.reset_index()["index"])
print(len(trans_tfs))
trans_tfs


# In[47]:


fig, ax = plt.subplots(figsize=(2.2, 2.2), nrows=1, ncols=1)

sig = orth_expr[orth_expr["sig"] == "sig"]
not_sig = orth_expr[orth_expr["sig"] == "not sig"]
trans_sig = sig[sig["gene_name_human"].isin(trans_tfs)]
trans_not_sig = not_sig[not_sig["gene_name_human"].isin(trans_tfs)]

ax.scatter(sig["mean_tpm_human"], 
           sig["mean_tpm_mouse"],
           color=sns.color_palette("Set2")[2], alpha=0.75, s=10, 
           linewidths=0.5, edgecolors="white")

ax.scatter(not_sig["mean_tpm_human"], 
           not_sig["mean_tpm_mouse"],
           color="gray", alpha=0.9, s=10, 
           linewidths=0.5, edgecolors="white")

ax.scatter(trans_sig["mean_tpm_human"], 
           trans_sig["mean_tpm_mouse"],
           color="black", alpha=0.9, s=12, 
           linewidths=0.5, edgecolors="black")

ax.scatter(trans_not_sig["mean_tpm_human"], 
           trans_not_sig["mean_tpm_mouse"],
           color="darkgray", alpha=0.9, s=12, 
           linewidths=0.5, edgecolors="darkgray")

ax.set_xscale("symlog")
ax.set_yscale("symlog")
ax.plot([-0.75, 400000], [-0.75, 400000], "k", linestyle="dashed")
ax.set_xlim((-0.75, 400000))
ax.set_ylim((-0.75, 400000))

ax.set_xlabel("human TF tpm in hESC")
ax.set_ylabel("mouse TF tpm in mESC")

# annotate corr
no_nan = orth_expr[(~pd.isnull(orth_expr["mean_tpm_human"])) & 
                   (~pd.isnull(orth_expr["mean_tpm_mouse"]))]
r, p = spearmanr(no_nan["mean_tpm_human"], no_nan["mean_tpm_mouse"])
ax.text(0.05, 0.97, "r = {:.2f}".format(r), ha="left", va="top", fontsize=fontsize,
        transform=ax.transAxes)
ax.text(0.05, 0.90, "# sig = %s" % (len(sig)), ha="left", va="top", fontsize=fontsize, 
        color=sns.color_palette("Set2")[2],
        transform=ax.transAxes)
ax.text(0.05, 0.83, "# not sig = %s" % (len(not_sig)), ha="left", va="top", fontsize=fontsize, color="gray",
        transform=ax.transAxes)
fig.savefig("TF_human_v_mouse_scatter.w_trans_outline.pdf", dpi="figure", bbox_inches="tight")


# ## 6. plot expression changes and % ID

# In[48]:


orth_sub = orth[["Gene stable ID", "Gene name", "Mouse gene stable ID", "Mouse gene name", "Mouse homology type",
                 "%id. target Mouse gene identical to query gene", "%id. query gene identical to target Mouse gene"]]
orth_sub.columns = ["gene_id_human", "gene_name_human", "gene_id_mouse", "gene_name_mouse", "homology_type",
                    "perc_mouse_to_human", "perc_human_to_mouse"]
orth_sub.head()


# In[49]:


orth_expr = orth_expr.merge(orth_sub, on=["gene_id_human", "gene_name_human", "gene_id_mouse", "gene_name_mouse"],
                            how="left")
orth_expr.sample(5)


# In[50]:


orth_expr = orth_expr.drop_duplicates()


# In[51]:


avg_log2FoldChange = orth_expr.log2FoldChange.mean()
print(avg_log2FoldChange)
avg_perc_mouse_to_human = orth_expr.perc_mouse_to_human.mean()
print(avg_perc_mouse_to_human)
avg_perc_human_to_mouse = orth_expr.perc_human_to_mouse.mean()
print(avg_perc_human_to_mouse)


# In[52]:


trans_log2FoldChange = orth_expr[orth_expr["gene_name_human"].isin(trans_tfs)][["gene_name_human", 
                                                                                "log2FoldChange"]].set_index("gene_name_human")
trans_perc_mouse_to_human = orth_expr[orth_expr["gene_name_human"].isin(trans_tfs)][["gene_name_human", 
                                                                                "perc_mouse_to_human"]].set_index("gene_name_human")
trans_perc_human_to_mouse = orth_expr[orth_expr["gene_name_human"].isin(trans_tfs)][["gene_name_human", 
                                                                                "perc_human_to_mouse"]].set_index("gene_name_human")

trans_log2FoldChange


# In[53]:


log2FoldChange = trans_log2FoldChange.append(pd.DataFrame.from_dict({"average": {"log2FoldChange": avg_log2FoldChange}}, 
                                                                    orient="index")).reset_index()
perc_mouse_to_human = trans_perc_mouse_to_human.append(pd.DataFrame.from_dict({"average": {"perc_mouse_to_human": avg_perc_mouse_to_human}}, 
                                                                    orient="index")).reset_index()
perc_human_to_mouse = trans_perc_human_to_mouse.append(pd.DataFrame.from_dict({"average": {"perc_human_to_mouse": avg_perc_human_to_mouse}}, 
                                                                    orient="index")).reset_index()
log2FoldChange


# In[54]:


ordered_trans_tfs = list(log2FoldChange.sort_values(by="log2FoldChange", ascending=False)["index"])
ordered_trans_tfs.pop(ordered_trans_tfs.index('average'))
ordered_trans_tfs


# In[55]:


pal = {k:sns.color_palette("Set2")[2] for k in trans_tfs}
pal["average"] = "gray"

order = ["average"]
order.extend(ordered_trans_tfs)


# In[56]:


fig = plt.figure(figsize=(3, 1.25))

ax = sns.barplot(data=log2FoldChange, x="index", y="log2FoldChange", order=order, palette=pal)
ax.set_xlabel("")
ax.set_ylabel("log2(hESC tpm/mESC tpm)")
_ = ax.set_xticklabels(order, rotation=50, ha='right', va='top')
fig.savefig("log2FoldChange_trans_tfs.pdf", dpi="figure", bbox_inches="tight")


# In[57]:


fig = plt.figure(figsize=(3, 1.25))

ax = sns.barplot(data=perc_mouse_to_human, x="index", y="perc_mouse_to_human", order=order, palette=pal)
ax.set_xlabel("")
ax.set_ylabel("% identity\n(mouse to human)")
_ = ax.set_xticklabels(order, rotation=50, ha='right', va='top')
fig.savefig("perc_mouse_to_human_trans_tfs.pdf", dpi="figure", bbox_inches="tight")


# In[58]:


fig = plt.figure(figsize=(3, 1.25))

ax = sns.barplot(data=perc_human_to_mouse, x="index", y="perc_human_to_mouse", order=order, palette=pal)
ax.set_xlabel("")
ax.set_ylabel("% identity\n(human to mouse)")
_ = ax.set_xticklabels(order, rotation=50, ha='right', va='top')
fig.savefig("perc_human_to_mouse_trans_tfs.pdf", dpi="figure", bbox_inches="tight")


# In[59]:


orth_expr[orth_expr["gene_name_human"] == "VSX1"]


# ## 7. find enrichment of trans motifs in enhancers

# In[68]:


# # hypergeometric
# enh_results = {}
# for i, tf in enumerate(uniq_human_tfs):
#     if i % 50 == 0:
#         print(i)
        
#     # look for enrichment of this motif in either human or mouse eRNAs
#     sub_human_motifs = list(human_max_motifs[human_max_motifs["gene_name"] == tf]["hg19_id"].unique())
#     sub_mouse_motifs = list(mouse_max_motifs[mouse_max_motifs["gene_name"] == tf]["mm9_id"].unique())
    
#     sub_human_enhs = list(results[results["cleaner_biotype_hg19"] == "eRNA"]["hg19_id"].unique())
#     sub_mouse_enhs = list(results[results["cleaner_biotype_mm9"] == "eRNA"]["mm9_id"].unique())
    
#     sub_human_no_enhs = list(results[results["cleaner_biotype_hg19"] != "eRNA"]["hg19_id"].unique())
#     sub_mouse_no_enhs = list(results[results["cleaner_biotype_mm9"] != "eRNA"]["mm9_id"].unique())
    
#     # variable still called human but includes both human & mouse IDs
#     sub_human_motifs.extend(sub_mouse_motifs)
#     sub_human_enhs.extend(sub_mouse_enhs)
#     sub_human_no_enhs.extend(sub_mouse_no_enhs)
    
#     n_enh_w_motif = len([x for x in sub_human_enhs if x in sub_human_motifs])
#     n_enh_wo_motif = len([x for x in sub_human_enhs if x not in sub_human_motifs])
#     n_no_enh_w_motif = len([x for x in sub_human_no_enhs if x in sub_human_motifs])
#     n_no_enh_wo_motif = len([x for x in sub_human_no_enhs if x not in sub_human_motifs])
    

    
#     pval = stats.hypergeom.sf(n_enh_w_motif-1, n_enh_w_motif+n_enh_wo_motif+n_no_enh_w_motif+n_no_enh_wo_motif, 
#                               n_enh_w_motif+n_enh_wo_motif, n_enh_w_motif+n_no_enh_w_motif)
    
#     enh_results[tf] = {"high_in_enh_pval": p, 
#                        "n_enh_w_motif": n_enh_w_motif, 
#                        "n_enh_wo_motif": n_enh_wo_motif, 
#                        "n_no_enh_w_motif": n_no_enh_w_motif,  
#                        "n_no_enh_wo_motif": n_no_enh_wo_motif}
    
# enh_results = pd.DataFrame.from_dict(enh_results, orient="index").reset_index()
# enh_results.sort_values(by="high_in_enh_pval").head()


# In[69]:


enh_results = {}
for i, tf in enumerate(uniq_human_tfs):
    if i % 50 == 0:
        print(i)
        
    # look for enrichment of this motif in either human or mouse eRNAs
    sub_human_motifs = list(human_max_motifs[human_max_motifs["gene_name"] == tf]["hg19_id"].unique())
    sub_mouse_motifs = list(mouse_max_motifs[mouse_max_motifs["gene_name"] == tf]["mm9_id"].unique())
    
    sub_human_enhs = list(results[results["cleaner_biotype_hg19"] == "eRNA"]["hg19_id"].unique())
    sub_mouse_enhs = list(results[results["cleaner_biotype_mm9"] == "eRNA"]["mm9_id"].unique())
    
    sub_human_no_enhs = list(results[results["cleaner_biotype_hg19"] != "eRNA"]["hg19_id"].unique())
    sub_mouse_no_enhs = list(results[results["cleaner_biotype_mm9"] != "eRNA"]["mm9_id"].unique())
    
    # variable still called human but includes both human & mouse IDs
    sub_human_motifs.extend(sub_mouse_motifs)
    sub_human_enhs.extend(sub_mouse_enhs)
    sub_human_no_enhs.extend(sub_mouse_no_enhs)
    
    n_enh_w_motif = len([x for x in sub_human_enhs if x in sub_human_motifs])
    n_enh_wo_motif = len([x for x in sub_human_enhs if x not in sub_human_motifs])
    n_no_enh_w_motif = len([x for x in sub_human_no_enhs if x in sub_human_motifs])
    n_no_enh_wo_motif = len([x for x in sub_human_no_enhs if x not in sub_human_motifs])
    
    arr = np.zeros((2, 2))
    arr[0, 0] = n_enh_w_motif
    arr[0, 1] = n_enh_wo_motif
    arr[1, 0] = n_no_enh_w_motif
    arr[1, 1] = n_no_enh_wo_motif
    
    odds, p = stats.fisher_exact(arr, alternative="greater")
    
    enh_results[tf] = {"high_in_enh_odds": odds, "high_in_enh_pval": p, 
                       "n_enh_w_motif": n_enh_w_motif, 
                       "n_enh_wo_motif": n_enh_wo_motif, 
                       "n_no_enh_w_motif": n_no_enh_w_motif,  
                       "n_no_enh_wo_motif": n_no_enh_wo_motif}
    
enh_results = pd.DataFrame.from_dict(enh_results, orient="index").reset_index()
enh_results.sort_values(by="high_in_enh_pval").head()


# In[70]:


enh_results["high_in_enh_padj"] = multicomp.multipletests(enh_results["high_in_enh_pval"], method="fdr_bh")[1]
len(enh_results[enh_results["high_in_enh_padj"] < 0.05])


# In[71]:


enh_results[enh_results["index"].isin(ordered_trans_tfs)]


# In[72]:


tmp = enh_results[enh_results["index"].isin(ordered_trans_tfs)]
tmp[tmp['high_in_enh_padj'] < 0.05]


# In[ ]:





# In[ ]:




