
# coding: utf-8

# # 03__activs_general
# 
# in this notebook, i look at activities (determined by MPRAnalyze) for sequences -- neg. ctrls, pos. ctrls, by biotype, etc. i also correlate these activities w/ endogenous expression. finally, since every TSS has 2 tiles, i take the maximum activity tile per TSS in order to continue on with cross-species comparisons.

# In[1]:


import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import math
import matplotlib.pyplot as plt
import numpy as np
import re
import seaborn as sns
import sys

from scipy.stats import spearmanr

# import utils
sys.path.append("../../../utils")
from plotting_utils import *

get_ipython().magic('matplotlib inline')
get_ipython().magic("config InlineBackend.figure_format = 'svg'")
mpl.rcParams['figure.autolayout'] = False


# In[2]:


sns.set(**PAPER_PRESET)
fontsize = PAPER_FONTSIZE


# In[3]:


np.random.seed(2019)


# ## functions

# In[49]:


def cleaner_biotype(row, biotype_col):
    try:
        if row["name"] == "random_sequence":
            return "negative control"
        elif "samp" in row.element:
            return "positive control"
        else:
            if row[biotype_col] in ["protein_coding", "div_pc"]:
                return "mRNA"
            elif row[biotype_col] in ["intergenic", "antisense", "div_lnc"]:
                return "lncRNA"
            elif row[biotype_col] == "enhancer":
                return "eRNA"
            elif row[biotype_col] == "no cage activity":
                return "no CAGE activity"
            else:
                return "other"
    except:
        if row[biotype_col] in ["protein_coding", "div_pc"]:
            return "mRNA"
        elif row[biotype_col] in ["intergenic", "antisense", "div_lnc"]:
            return "lncRNA"
        elif row[biotype_col] == "enhancer":
            return "eRNA"
        elif row[biotype_col] == "no cage activity":
            return "no CAGE activity"
        else:
            return "other"


# In[45]:


def is_sig(row, col):
    if row[col] < 0.05:
        return "sig"
    else:
        return "not sig"


# In[58]:


def fix_cage_exp(row, col):
    if row[col] == "no cage activity":
        return 0
    else:
        return float(row[col])


# ## variables

# In[4]:


data_dir = "../../../data/02__mpra/02__activs"
alpha_f = "%s/alpha_per_elem.quantification.txt" % data_dir


# In[5]:


index_f = "../../../data/01__design/01__index/TWIST_pool4_v8_final.with_element_id.txt.gz"


# In[6]:


tss_map_f = "../../../data/01__design/00__mpra_list/mpra_tss.with_ids.UPDATED.txt"


# ## 1. import files

# In[7]:


alpha = pd.read_table(alpha_f, sep="\t")
alpha.reset_index(inplace=True)
alpha.head()


# In[8]:


index = pd.read_table(index_f, sep="\t")


# In[9]:


index_elem = index[["element", "tile_type", "element_id", "name", "tile_number", "chrom", "strand", "actual_start", 
                    "actual_end", "dupe_info"]]
index_elem = index_elem.drop_duplicates()


# In[10]:


tss_map = pd.read_table(tss_map_f, sep="\t", index_col=0)
tss_map.head()


# ## 2. merge alphas w/ index

# In[11]:


pos_ctrls = alpha[alpha["index"].str.contains("__samp")]
pos_ctrls["HUES64_log"] = np.log10(pos_ctrls["HUES64"])
pos_ctrls["mESC_log"] = np.log10(pos_ctrls["mESC"])
len(pos_ctrls)


# In[12]:


alpha = alpha[~alpha["index"].str.contains("__samp")]
len(alpha)


# In[13]:


data = alpha.merge(index_elem, left_on="index", right_on="element", how="left")
data.drop("index", axis=1, inplace=True)
data.head()


# In[14]:


data["HUES64_log"] = np.log10(data["HUES64"])
data["mESC_log"] = np.log10(data["mESC"])
data.sample(5)


# ## 3. compare negative controls to TSSs & positive controls
# maybe delete this section

# In[19]:


neg_ctrls = data[data["tile_type"] == "RANDOM"]
others = data[data["tile_type"] != "RANDOM"]


# In[20]:


plt.figure(figsize=(1.5, 1.25))
sns.kdeplot(neg_ctrls["HUES64_log"], label="negative controls (n=%s)" % len(neg_ctrls), 
             color="gray", cumulative=True)
sns.kdeplot(others["HUES64_log"], label="sequences (n=%s)" % len(others), 
             color=sns.color_palette("Set2")[1], cumulative=True)
sns.kdeplot(pos_ctrls["HUES64_log"], label="positive controls (n=%s)" % len(pos_ctrls), 
             color="black", cumulative=True)
plt.legend(loc=2, bbox_to_anchor=(1.05, 1))
plt.ylabel("density")
plt.xlabel("log10 MPRA activity in hESCs")


# In[21]:


plt.figure(figsize=(1.5, 1.25))
sns.kdeplot(neg_ctrls["mESC_log"], label="negative controls (n=%s)" % len(neg_ctrls), 
             color="gray", cumulative=True)
sns.kdeplot(others["mESC_log"], label="sequences (n=%s)" % len(others), 
             color=sns.color_palette("Set2")[0], cumulative=True)
sns.kdeplot(pos_ctrls["mESC_log"], label="positive controls (n=%s)" % len(pos_ctrls), 
             color="black", cumulative=True)
plt.legend(loc=2, bbox_to_anchor=(1.05, 1))
plt.ylabel("density")
plt.xlabel("log10 MPRA activity in mESCs")


# ## 4. compare activities across biotypes + controls

# In[23]:


data["tss_id"] = data["name"].str.split("__", expand=True)[1]
data["species"] = data["name"].str.split("_", expand=True)[0]
data["tss_tile_num"] = data["name"].str.split("__", expand=True)[2]
data.sample(5)


# In[24]:


pos_ctrls.columns = ["element", "HUES64", "mESC", "HUES64_pval", "mESC_pval", "HUES64_padj", "mESC_padj", 
                     "HUES64_log", "mESC_log"]
pos_ctrls.head()


# In[25]:


human_df = data[(data["species"] == "HUMAN") | (data["name"] == "random_sequence")]
mouse_df = data[(data["species"] == "MOUSE") | (data["name"] == "random_sequence")]

human_df_w_ctrls = human_df.append(pos_ctrls)
mouse_df_w_ctrls = mouse_df.append(pos_ctrls)

human_df_w_ctrls = human_df_w_ctrls.merge(tss_map[["hg19_id", "biotype_hg19", "stem_exp_hg19", "orig_species"]], 
                                          left_on="tss_id", right_on="hg19_id", how="left")
mouse_df_w_ctrls = mouse_df_w_ctrls.merge(tss_map[["mm9_id", "biotype_mm9", "stem_exp_mm9", "orig_species"]], 
                                          left_on="tss_id", right_on="mm9_id", how="left")
mouse_df_w_ctrls.sample(5)


# In[27]:


human_df_w_ctrls["cleaner_biotype"] = human_df_w_ctrls.apply(cleaner_biotype, biotype_col="biotype_hg19", axis=1)
mouse_df_w_ctrls["cleaner_biotype"] = mouse_df_w_ctrls.apply(cleaner_biotype, biotype_col="biotype_mm9", axis=1)
human_df_w_ctrls.cleaner_biotype.value_counts()


# In[34]:


ctrl_order = ["negative control", "eRNA", "lncRNA", "mRNA", "positive control"]

human_ctrl_pal = {"negative control": "gray", "no CAGE activity": "gray", "eRNA": sns.color_palette("Set2")[1],
                  "lncRNA": sns.color_palette("Set2")[1], "mRNA": sns.color_palette("Set2")[1], 
                  "positive control": "black"}

mouse_ctrl_pal = {"negative control": "gray", "no CAGE activity": "gray", "eRNA": sns.color_palette("Set2")[0],
                  "lncRNA": sns.color_palette("Set2")[0], "mRNA": sns.color_palette("Set2")[0], 
                  "positive control": "black"}


# In[35]:


fig = plt.figure(figsize=(2.35, 1.5))
ax = sns.boxplot(data=human_df_w_ctrls, x="cleaner_biotype", y="HUES64", flierprops = dict(marker='o', markersize=5),
                 order=ctrl_order, palette=human_ctrl_pal)
mimic_r_boxplot(ax)

ax.set_xticklabels(ctrl_order, rotation=50, ha='right', va='top')
ax.set_xlabel("")
ax.set_yscale("symlog")
ax.set_ylabel("MPRA activity in hESCs")

for i, label in enumerate(ctrl_order):
    n = len(human_df_w_ctrls[human_df_w_ctrls["cleaner_biotype"] == label])
    color = human_ctrl_pal[label]
    ax.annotate(str(n), xy=(i, -0.7), xycoords="data", xytext=(0, 0), 
                textcoords="offset pixels", ha='center', va='bottom', 
                color=color, size=fontsize)

ax.set_ylim((-1, 20))
fig.savefig("better_neg_ctrl_boxplot.human.pdf", dpi="figure", bbox_inches="tight")


# In[37]:


fig = plt.figure(figsize=(2.35, 1.5))
ax = sns.boxplot(data=mouse_df_w_ctrls, x="cleaner_biotype", y="mESC", flierprops = dict(marker='o', markersize=5),
                 order=ctrl_order, palette=mouse_ctrl_pal)
mimic_r_boxplot(ax)

ax.set_xticklabels(ctrl_order, rotation=50, ha='right', va='top')
ax.set_xlabel("")
ax.set_yscale("symlog")
ax.set_ylabel("MPRA activity in mESCs")

for i, label in enumerate(ctrl_order):
    n = len(mouse_df_w_ctrls[mouse_df_w_ctrls["cleaner_biotype"] == label])
    color = mouse_ctrl_pal[label]
    ax.annotate(str(n), xy=(i, -0.7), xycoords="data", xytext=(0, 0), 
                textcoords="offset pixels", ha='center', va='bottom', 
                color=color, size=fontsize)

ax.set_ylim((-1, 20))
fig.savefig("better_neg_ctrl_boxplot.mouse.pdf", dpi="figure", bbox_inches="tight")


# ## 5. compare activities across tiles

# In[38]:


df = data[data["tss_tile_num"].isin(["tile1", "tile2"])]
human_df = df[df["species"] == "HUMAN"]
mouse_df = df[df["species"] == "MOUSE"]

human_df = human_df.merge(tss_map[["hg19_id", "biotype_hg19", "stem_exp_hg19", "orig_species"]], left_on="tss_id", 
                          right_on="hg19_id", how="right")
mouse_df = mouse_df.merge(tss_map[["mm9_id", "biotype_mm9", "stem_exp_mm9", "orig_species"]], left_on="tss_id", 
                          right_on="mm9_id", how="right")
mouse_df.sample(5)


# In[44]:


for df, species, colname, color in zip([human_df, mouse_df], ["hESCs", "mESCs"], ["HUES64", "mESC"], [sns.color_palette("Set2")[1], sns.color_palette("Set2")[0]]):
    fig = plt.figure(figsize=(2, 1.5))
    ax = sns.boxplot(data=df, x="tss_tile_num", y=colname, flierprops = dict(marker='o', markersize=5),
                     color=color)
    mimic_r_boxplot(ax)

    # calc p-vals b/w dists
    tile1_dist = np.asarray(df[df["tss_tile_num"] == "tile1"][colname])
    tile2_dist = np.asarray(df[df["tss_tile_num"] == "tile2"][colname])

    tile1_dist = tile1_dist[~np.isnan(tile1_dist)]
    tile2_dist = tile2_dist[~np.isnan(tile2_dist)]

    tile_u, tile_pval = stats.mannwhitneyu(tile1_dist, tile2_dist, alternative="two-sided", use_continuity=False)
    print(tile_pval)

    annotate_pval(ax, 0.2, 0.8, 1, 0, 1, tile_pval, fontsize)
    ax.set_yscale('symlog')
    ax.set_ylabel("MPRA activity in %s" % species)
    ax.set_xlabel("")
    ax.set_title(species)


# ## 6. find max activity per tile

# In[46]:


human_df_sort = human_df[["element", "tss_id", "biotype_hg19", "tss_tile_num", "HUES64", "HUES64_log", "HUES64_padj"]].sort_values(by=["tss_id", "HUES64_log"], ascending=False)
human_df_max = human_df_sort.groupby(["tss_id", "biotype_hg19"]).head(1)
human_df_max["HUES64_sig"] = human_df_max.apply(is_sig, col="HUES64_padj", axis=1)
human_df_max.head(10)


# In[47]:


mouse_df_sort = mouse_df[["element", "tss_id", "biotype_mm9", "tss_tile_num", "mESC", "mESC_log", "mESC_padj"]].sort_values(by=["tss_id", "mESC_log"], ascending=False)
mouse_df_max = mouse_df_sort.groupby(["tss_id", "biotype_mm9"]).head(1)
mouse_df_max["mESC_sig"] = mouse_df_max.apply(is_sig, col="mESC_padj", axis=1)
mouse_df_max.head(10)


# In[50]:


human_df_max["cleaner_biotype"] = human_df_max.apply(cleaner_biotype, biotype_col="biotype_hg19", axis=1)
mouse_df_max["cleaner_biotype"] = mouse_df_max.apply(cleaner_biotype, biotype_col="biotype_mm9", axis=1)


# In[52]:


ctrls = human_df_w_ctrls[human_df_w_ctrls["cleaner_biotype"].isin(["negative control", "positive control"])]
len(ctrls)


# In[53]:


human_tmp = human_df_max.append(ctrls)
mouse_tmp = mouse_df_max.append(ctrls)


# In[55]:


# re-make boxplots with neg/pos ctrls
fig = plt.figure(figsize=(2.35, 1.25))
ax = sns.boxplot(data=human_tmp, x="cleaner_biotype", y="HUES64", flierprops = dict(marker='o', markersize=5),
                 order=ctrl_order, palette=human_ctrl_pal)
mimic_r_boxplot(ax)

ax.set_xticklabels(ctrl_order, rotation=50, ha='right', va='top')
ax.set_xlabel("")
ax.set_yscale("symlog")
ax.set_ylabel("MPRA activity\n(hESCs)")

for i, label in enumerate(ctrl_order):
    n = len(human_tmp[human_tmp["cleaner_biotype"] == label])
    color = human_ctrl_pal[label]
    ax.annotate(str(n), xy=(i, -0.7), xycoords="data", xytext=(0, 0), 
                textcoords="offset pixels", ha='center', va='bottom', 
                color=color, size=fontsize)

ax.set_ylim((-1, 20))
fig.savefig("better_neg_ctrl_boxplot.human_max.pdf", dpi="figure", bbox_inches="tight")


# In[56]:


# re-make boxplots with neg/pos ctrls
fig = plt.figure(figsize=(2.35, 1.25))
ax = sns.boxplot(data=mouse_tmp, x="cleaner_biotype", y="mESC", flierprops = dict(marker='o', markersize=5),
                 order=ctrl_order, palette=mouse_ctrl_pal)
mimic_r_boxplot(ax)

ax.set_xticklabels(ctrl_order, rotation=50, ha='right', va='top')
ax.set_xlabel("")
ax.set_yscale("symlog")
ax.set_ylabel("MPRA activity\n(mESCs)")

for i, label in enumerate(ctrl_order):
    n = len(mouse_tmp[mouse_tmp["cleaner_biotype"] == label])
    color = mouse_ctrl_pal[label]
    ax.annotate(str(n), xy=(i, -0.7), xycoords="data", xytext=(0, 0), 
                textcoords="offset pixels", ha='center', va='bottom', 
                color=color, size=fontsize)

ax.set_ylim((-1, 20))
fig.savefig("better_neg_ctrl_boxplot.mouse_max.pdf", dpi="figure", bbox_inches="tight")


# ## 7. correlate MPRA activities w/ endogenous activities
# 
# consider re-doing w/ RNA-seq

# In[57]:


human_tmp = human_df_max.merge(human_df[["element", "stem_exp_hg19"]].drop_duplicates(), on="element")
len(human_tmp)


# In[59]:


human_tmp["stem_exp_hg19_fixed"] = human_tmp.apply(fix_cage_exp, col="stem_exp_hg19", axis=1)
human_tmp.sample(5)


# In[60]:


biotypes_sub = ["eRNA", "lncRNA", "mRNA"]


# In[67]:


fig, axes = plt.subplots(figsize=(4, 1.25), nrows=1, ncols=len(biotypes_sub), sharex=False, sharey=True)
for i, biotype in enumerate(biotypes_sub):
    print(biotype)
    ax = axes[i]
    sub = human_tmp[human_tmp["cleaner_biotype"] == biotype]
    
    sub["stem_exp_hg19_log"] = np.log10(sub["stem_exp_hg19_fixed"] + 1)
    sub = sub[~pd.isnull(sub["HUES64_log"])]
    print(len(sub))
    
    sns.regplot(data=sub, x="stem_exp_hg19_log", y="HUES64_log", color=human_ctrl_pal[biotype], 
                scatter_kws={"s": 15, "alpha": 0.75, "linewidth": 0.5, "edgecolor": "white"}, fit_reg=True, ax=ax)
    
    # get coeffs of linear fit
    slope, intercept, r_value, p_value, std_err = stats.linregress(sub['stem_exp_hg19_log'], sub['HUES64_log'])
    print(r_value)
    
    ax.text(0.95, 0.95, "r = {:.2f}".format(r_value), ha="right", va="top", fontsize=fontsize,
        transform=ax.transAxes)
    ax.text(0.95, 0.85, "n = %s" % (len(sub)), ha="right", va="top", fontsize=fontsize,
            transform=ax.transAxes)
    
    ax.set_xlabel("")
    if i == 0:
        ax.set_ylabel("log10(MPRA activity)\n(hESCs)")
    else:
        ax.set_ylabel("")
        
plt.text(0.5, -0.05, "log10(CAGE expression + 1) (hESCs)", ha="center", va="top", 
             transform=plt.gcf().transFigure, fontsize=fontsize)
plt.subplots_adjust(wspace=0.1)
plt.show()

fig.savefig("cage_corr_human.biotype_sub.pdf", dpi="figure", bbox_inches="tight")


# In[68]:


mouse_tmp = mouse_df_max.merge(mouse_df[["element", "stem_exp_mm9"]].drop_duplicates(), on="element")
mouse_tmp["stem_exp_mm9_fixed"] = mouse_tmp.apply(fix_cage_exp, col="stem_exp_mm9", axis=1)
len(mouse_tmp)


# In[70]:


fig, axes = plt.subplots(figsize=(4, 1.25), nrows=1, ncols=len(biotypes_sub), sharex=False, sharey=True)
for i, biotype in enumerate(biotypes_sub):
    print(biotype)
    ax = axes[i]
    sub = mouse_tmp[mouse_tmp["cleaner_biotype"] == biotype]
    
    sub["stem_exp_mm9_log"] = np.log10(sub["stem_exp_mm9_fixed"] + 1)
    sub = sub[~pd.isnull(sub["mESC_log"])]
    print(len(sub))
    
    sns.regplot(data=sub, x="stem_exp_mm9_log", y="mESC_log", color=mouse_ctrl_pal[biotype], 
                scatter_kws={"s": 15, "alpha": 0.75, "linewidth": 0.5, "edgecolor": "white"}, fit_reg=True, ax=ax)
    
    # get coeffs of linear fit
    slope, intercept, r_value, p_value, std_err = stats.linregress(sub['stem_exp_mm9_log'], sub['mESC_log'])
    print(r_value)
    
    ax.text(0.95, 0.95, "r = {:.2f}".format(r_value), ha="right", va="top", fontsize=fontsize,
        transform=ax.transAxes)
    ax.text(0.95, 0.85, "n = %s" % (len(sub)), ha="right", va="top", fontsize=fontsize,
            transform=ax.transAxes)
    
    ax.set_xlabel("")
    if i == 0:
        ax.set_ylabel("log10(MPRA activity)\n(mESCs)")
    else:
        ax.set_ylabel("")
        
plt.text(0.5, -0.05, "log10(CAGE expression) (mESCs)", ha="center", va="top", 
             transform=plt.gcf().transFigure, fontsize=fontsize)
plt.subplots_adjust(wspace=0.1)
plt.show()

fig.savefig("cage_corr_mouse.biotype_sub.pdf", dpi="figure", bbox_inches="tight")


# ## 8. write files

# In[71]:


human_df_filename = "%s/human_TSS_vals.both_tiles.txt" % data_dir
mouse_df_filename = "%s/mouse_TSS_vals.both_tiles.txt" % data_dir
human_df_max_filename = "%s/human_TSS_vals.max_tile.txt" % data_dir
mouse_df_max_filename = "%s/mouse_TSS_vals.max_tile.txt" % data_dir


# In[72]:


human_df.to_csv(human_df_filename, sep="\t", index=False)
mouse_df.to_csv(mouse_df_filename, sep="\t", index=False)
human_df_max.to_csv(human_df_max_filename, sep="\t", index=False)
mouse_df_max.to_csv(mouse_df_max_filename, sep="\t", index=False)


# In[ ]:




