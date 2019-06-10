
# coding: utf-8

# # 07__cis_effects

# In[1]:


import warnings
warnings.filterwarnings('ignore')

import itertools
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import sys

from itertools import combinations 
from scipy.stats import linregress
from scipy.stats import spearmanr
from scipy.stats import pearsonr

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

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


# ## functions

# In[4]:


def is_ctrl(row):
    if "CONTROL" in row["index"]:
        return "control"
    else:
        return "TSS"


# In[5]:


def cis_status(row):
    if row.fdr < 0.05:
        return "significant cis effect"
    else:
        return "no cis effect"


# In[6]:


def cis_status_detail(row):
    if row.fdr < 0.05:
        if row.logFC < 0:
            return "cis effect\n(higher in human)"
        else:
            return "cis effect\n(higher in mouse)"
    else:
        return "no cis effect"


# In[7]:


def cis_status_one(row):
    if row.fdr_cis_HUES64 < 0.05 or row.fdr_cis_mESC < 0.05:
        return "significant cis effect"
    else:
        return "no cis effect"


# In[8]:


def cis_status_detail_one(row):
    if row.fdr_cis_HUES64 < 0.05:
        if row.fdr_cis_mESC < 0.05:
            # 2 sig trans effects: check both
            if row.logFC_cis_HUES64 < 0 and row.logFC_cis_mESC < 0:
                return "cis effect\n(higher in human)"
            elif row.logFC_cis_HUES64 > 0 and row.logFC_cis_mESC > 0:
                return "cis effect\n(higher in mouse)"
            else:
                return "cis effect\n(direction interaction)"
        else:
            # only sig in human, only check human
            if row.logFC_cis_HUES64 < 0:
                return "cis effect\n(higher in human)"
            else:
                return "cis effect\n(higher in mouse)"
    else:
        if row.fdr_cis_mESC < 0.05:
            # only sig in mouse, only check mouse
            if row.logFC_cis_mESC < 0:
                return "cis effect\n(higher in human)"
            else:
                return "cis effect\n(higher in mouse)"
        else:
            # no sig effects
            return "no cis effect"


# In[9]:


def cis_logFC_one(row):
    if row.fdr_cis_HUES64 < 0.05:
        if row.fdr_cis_mESC < 0.05:
            # 2 sig trans effects: take max of both
            l2fcs = list(row[["logFC_cis_HUES64", "logFC_cis_mESC"]])
            return signed_max(l2fcs)
        else:
            # only sig in human, return human val
            return row.logFC_cis_HUES64
    else:
        if row.fdr_cis_mESC < 0.05:
            # only sig in mouse, return mouse val
            return row.logFC_cis_mESC
        else:
            # no sig effects: take max of both
            l2fcs = list(row[["logFC_cis_HUES64", "logFC_cis_mESC"]])
            return signed_max(l2fcs)


# In[10]:


def signed_max(nums):
    abs_nums = np.abs(nums)
    max_idx = np.argmax(abs_nums)
    return nums[max_idx]


# ## variables

# In[11]:


results_dir = "../../../data/02__mpra/03__results"
results_f = "%s/native_effects_data.txt" % results_dir


# In[12]:


data_dir = "../../../data/02__mpra/02__activs"
HUES64_cis_f = "%s/HUES64_cis_results.txt" % data_dir
mESC_cis_f = "%s/mESC_cis_results.txt" % data_dir


# In[13]:


tss_map_f = "../../../data/01__design/01__mpra_list/mpra_tss.with_ids.UPDATED.txt"


# In[14]:


align_f = "../../../misc/00__tss_metadata/tss_map.seq_alignment.txt"


# ## 1. import data

# In[15]:


results = pd.read_table(results_f)
results.head()


# In[16]:


HUES64_cis = pd.read_table(HUES64_cis_f).reset_index()
mESC_cis = pd.read_table(mESC_cis_f).reset_index()
mESC_cis.head()


# In[17]:


tss_map = pd.read_table(tss_map_f, index_col=0)
tss_map.head()


# In[18]:


align = pd.read_table(align_f)
align.head()


# ## 2. plot cis controls vs. TSS

# In[19]:


HUES64_cis["ctrl_status"] = HUES64_cis.apply(is_ctrl, axis=1)
mESC_cis["ctrl_status"] = mESC_cis.apply(is_ctrl, axis=1)
mESC_cis.ctrl_status.value_counts()


# In[20]:


fig = plt.figure(figsize=(1.5, 1))
sns.distplot(HUES64_cis[HUES64_cis["ctrl_status"] == "control"]["pval"], hist=False, color="gray",
             label="negative controls (n=%s)" % len(HUES64_cis[HUES64_cis["ctrl_status"] == "control"]))
sns.distplot(HUES64_cis[HUES64_cis["ctrl_status"] != "control"]["pval"], hist=False, color="black",
             label="TSSs (n=%s)" % len(HUES64_cis[HUES64_cis["ctrl_status"] != "control"]))

plt.ylabel("density")
plt.xlabel("hESC cis effect p-value")
plt.legend(loc=2, bbox_to_anchor=(1.05, 1.05))
fig.savefig("HUES64_cis_ctrl_pval_dist.pdf", dpi="figure", bbox_inches="tight")


# In[21]:


fig = plt.figure(figsize=(1.5, 1))
sns.distplot(mESC_cis[mESC_cis["ctrl_status"] == "control"]["pval"], hist=False, color="gray",
             label="negative controls (n=%s)" % len(mESC_cis[mESC_cis["ctrl_status"] == "control"]))
sns.distplot(mESC_cis[mESC_cis["ctrl_status"] != "control"]["pval"], hist=False, color="black",
             label="TSSs (n=%s)" % len(mESC_cis[mESC_cis["ctrl_status"] != "control"]))

plt.ylabel("density")
plt.xlabel("mESC cis effect p-value")
plt.legend(loc=2, bbox_to_anchor=(1.05, 1.05))
fig.savefig("mESC_cis_cis_ctrl_pval_dist.pdf", dpi="figure", bbox_inches="tight")


# In[22]:


HUES64_cis["abs_logFC"] = np.abs(HUES64_cis["logFC"])
mESC_cis["abs_logFC"] = np.abs(mESC_cis["logFC"])


# In[23]:


order = ["control", "TSS"]
pal = {"control": "gray", "TSS": "black"}


# In[24]:


fig = plt.figure(figsize=(1, 1.75))
ax = sns.boxplot(data=HUES64_cis, x="ctrl_status", y="abs_logFC", flierprops = dict(marker='o', markersize=5), 
                 order=order, palette=pal)
mimic_r_boxplot(ax)

ax.set_xticklabels(["negative\ncontrols", "TSSs"], rotation=50, ha='right', va='top')
ax.set_xlabel("")
ax.set_ylabel("cis effect size in hESCs")

for i, label in enumerate(order):
    n = len(HUES64_cis[HUES64_cis["ctrl_status"] == label])
    color = pal[label]
    ax.annotate(str(n), xy=(i, -0.8), xycoords="data", xytext=(0, 0), 
                textcoords="offset pixels", ha='center', va='bottom', 
                color=color, size=fontsize)

ax.set_ylim((-1, 6))

# calc p-vals b/w dists
dist1 = np.asarray(HUES64_cis[HUES64_cis["ctrl_status"] == "control"]["abs_logFC"])
dist2 = np.asarray(HUES64_cis[HUES64_cis["ctrl_status"] != "control"]["abs_logFC"])

dist1 = dist1[~np.isnan(dist1)]
dist2 = dist2[~np.isnan(dist2)]

u, pval = stats.mannwhitneyu(dist1, dist2, alternative="two-sided", use_continuity=False)
print(pval)

annotate_pval(ax, 0.2, 0.8, 1.5, 0, 1.3, pval, fontsize)
fig.savefig("HUES64_cis_ctrl_effectsize_boxplot.pdf", dpi="figure", bbox_inches="tight")


# In[25]:


fig = plt.figure(figsize=(1, 1.75))
ax = sns.boxplot(data=mESC_cis, x="ctrl_status", y="abs_logFC", flierprops = dict(marker='o', markersize=5), 
                 order=order, palette=pal)
mimic_r_boxplot(ax)

ax.set_xticklabels(["negative\ncontrols", "TSSs"], rotation=50, ha='right', va='top')
ax.set_xlabel("")
ax.set_ylabel("cis effect size in mESCs")

for i, label in enumerate(order):
    n = len(mESC_cis[mESC_cis["ctrl_status"] == label])
    color = pal[label]
    ax.annotate(str(n), xy=(i, -0.8), xycoords="data", xytext=(0, 0), 
                textcoords="offset pixels", ha='center', va='bottom', 
                color=color, size=fontsize)

ax.set_ylim((-1, 6.5))

# calc p-vals b/w dists
dist1 = np.asarray(mESC_cis[mESC_cis["ctrl_status"] == "control"]["abs_logFC"])
dist2 = np.asarray(mESC_cis[mESC_cis["ctrl_status"] != "control"]["abs_logFC"])

dist1 = dist1[~np.isnan(dist1)]
dist2 = dist2[~np.isnan(dist2)]

u, pval = stats.mannwhitneyu(dist1, dist2, alternative="two-sided", use_continuity=False)
print(pval)

annotate_pval(ax, 0.2, 0.8, 1.5, 0, 1.3, pval, fontsize)
fig.savefig("mESC_cis_ctrl_effectsize_boxplot.pdf", dpi="figure", bbox_inches="tight")


# ## 3. classify cis effects

# In[26]:


HUES64_cis["cis_status"] = HUES64_cis.apply(cis_status, axis=1)
HUES64_cis["cis_status_detail"] = HUES64_cis.apply(cis_status_detail, axis=1)
HUES64_cis.cis_status_detail.value_counts()


# In[27]:


mESC_cis["cis_status"] = mESC_cis.apply(cis_status, axis=1)
mESC_cis["cis_status_detail"] = mESC_cis.apply(cis_status_detail, axis=1)
mESC_cis.cis_status_detail.value_counts()


# ## 4. merge cis effects w/ native effects data

# In[28]:


HUES64_cis["hg19_id"] = HUES64_cis["index"].str.split("__", expand=True)[0]
HUES64_cis["biotype_hg19"] = HUES64_cis["index"].str.split("__", expand=True)[1]
HUES64_cis["mm9_id"] = HUES64_cis["index"].str.split("__", expand=True)[2]
HUES64_cis["biotype_mm9"] = HUES64_cis["index"].str.split("__", expand=True)[3]
HUES64_cis.head()


# In[29]:


mESC_cis["hg19_id"] = mESC_cis["index"].str.split("__", expand=True)[0]
mESC_cis["biotype_hg19"] = mESC_cis["index"].str.split("__", expand=True)[1]
mESC_cis["mm9_id"] = mESC_cis["index"].str.split("__", expand=True)[2]
mESC_cis["biotype_mm9"] = mESC_cis["index"].str.split("__", expand=True)[3]
mESC_cis.head()


# In[30]:


HUES64_cis_sub = HUES64_cis[["hg19_id", "biotype_hg19", "mm9_id", "biotype_mm9", "fdr", "logFC", "abs_logFC",
                             "cis_status", "cis_status_detail"]]
HUES64_cis_sub.columns = ["hg19_id", "biotype_hg19", "mm9_id", "biotype_mm9", "fdr_cis_HUES64", "logFC_cis_HUES64", 
                          "abs_logFC_cis_HUES64", "cis_status_HUES64", "cis_status_detail_HUES64"]


# In[31]:


mESC_cis_sub = mESC_cis[["hg19_id", "biotype_hg19", "mm9_id", "biotype_mm9", "fdr", "logFC", "abs_logFC",
                         "cis_status", "cis_status_detail"]]
mESC_cis_sub.columns = ["hg19_id", "biotype_hg19", "mm9_id", "biotype_mm9", "fdr_cis_mESC", "logFC_cis_mESC", 
                        "abs_logFC_cis_mESC", "cis_status_mESC", "cis_status_detail_mESC"]


# In[32]:


data = results.merge(HUES64_cis_sub, 
                     left_on=["hg19_id", "biotype_hg19", 
                              "mm9_id", "biotype_mm9"], 
                     right_on=["hg19_id", "biotype_hg19", 
                               "mm9_id", "biotype_mm9"]).merge(mESC_cis_sub,
                                                               left_on=["hg19_id", "biotype_hg19", 
                                                                        "mm9_id", "biotype_mm9"], 
                                                               right_on=["hg19_id", "biotype_hg19", 
                                                                         "mm9_id", "biotype_mm9"])
print(len(data))
data.sample(5)


# In[33]:


data["cis_status_one"] = data.apply(cis_status_one, axis=1)
data["cis_status_detail_one"] = data.apply(cis_status_detail_one, axis=1)


# In[34]:


data.cis_status_one.value_counts()


# In[35]:


data.cis_status_detail_one.value_counts()


# In[36]:


## remove cis effects w/ direction interactions for now! (there aren't any right now)
data_filt = data[~data["cis_status_detail_one"].str.contains("interaction")]


# ## 5. plot cis effects on scatter plot

# In[37]:


# limit to those that are significant in at least 1 context
data_filt = data_filt[(data_filt["HUES64_padj_hg19"] < 0.05) | (data_filt["mESC_padj_mm9"] < 0.05)]
len(data_filt)


# In[38]:


fig, ax = plt.subplots(figsize=(1.75, 1.75), nrows=1, ncols=1)

not_sig = data_filt[data_filt["fdr_cis_HUES64"] >= 0.01]
sig = data_filt[data_filt["fdr_cis_HUES64"] < 0.01]

ax.scatter(sig["HUES64_hg19"], sig["HUES64_mm9"], s=10, alpha=0.75, 
           color="black", linewidths=0.5, edgecolors="white")
ax.scatter(not_sig["HUES64_hg19"], not_sig["HUES64_mm9"], s=8, alpha=0.5, 
           color="gray", linewidths=0.5, edgecolors="white")

plt.xlabel("human activity in hESCs")
plt.ylabel("mouse activity in hESCs")
ax.set_xscale("symlog")
ax.set_yscale("symlog")
ax.plot([-0.2, 25], [-0.2, 25], linestyle="dashed", color="k")
ax.set_xlim((-0.2, 25))
ax.set_ylim((-0.2, 25))

# annotate corr
no_nan = data_filt[(~pd.isnull(data_filt["HUES64_hg19"])) & 
                   (~pd.isnull(data_filt["HUES64_mm9"]))]
r, p = spearmanr(no_nan["HUES64_hg19"], no_nan["HUES64_mm9"])
ax.text(0.05, 0.97, "r = {:.2f}".format(r), ha="left", va="top", fontsize=fontsize,
        transform=ax.transAxes)
ax.text(0.05, 0.90, "n = %s" % (len(data_filt)), ha="left", va="top", fontsize=fontsize,
        transform=ax.transAxes)
fig.savefig("cis_HUES64_scatter.pdf", dpi="figure", bbox_inches="tight")


# In[39]:


fig, ax = plt.subplots(figsize=(1.75, 1.75), nrows=1, ncols=1)

not_sig = data_filt[data_filt["fdr_cis_mESC"] >= 0.01]
sig = data_filt[data_filt["fdr_cis_mESC"] < 0.01]

ax.scatter(sig["mESC_hg19"], sig["mESC_mm9"], s=10, alpha=0.75, 
           color="black", linewidths=0.5, edgecolors="white")
ax.scatter(not_sig["mESC_hg19"], not_sig["mESC_mm9"], s=8, alpha=0.5, 
           color="gray", linewidths=0.5, edgecolors="white")

plt.xlabel("human activity in mESCs")
plt.ylabel("mouse activity in mESCs")
ax.set_xscale("symlog")
ax.set_yscale("symlog")
ax.plot([-0.2, 25], [-0.2, 25], linestyle="dashed", color="k")
ax.set_xlim((-0.2, 25))
ax.set_ylim((-0.2, 25))

# annotate corr
no_nan = data_filt[(~pd.isnull(data_filt["mESC_hg19"])) & 
                   (~pd.isnull(data_filt["mESC_mm9"]))]
r, p = spearmanr(no_nan["mESC_hg19"], no_nan["mESC_mm9"])
ax.text(0.05, 0.97, "r = {:.2f}".format(r), ha="left", va="top", fontsize=fontsize,
        transform=ax.transAxes)
ax.text(0.05, 0.90, "n = %s" % (len(data_filt)), ha="left", va="top", fontsize=fontsize,
        transform=ax.transAxes)
fig.savefig("cis_mESC_scatter.pdf", dpi="figure", bbox_inches="tight")


# In[40]:


# plot effect size agreement b/w the two cell lines
fig, ax = plt.subplots(figsize=(1.75, 1.75), nrows=1, ncols=1)

ax.scatter(data_filt["logFC_cis_HUES64"], data_filt["logFC_cis_mESC"], s=10, alpha=0.75, 
           color="slategray", linewidths=0.5, edgecolors="white")

plt.xlabel("cis effect size in hESCs")
plt.ylabel("cis effect size in mESCs")

ax.plot([-6, 6], [-6, 6], linestyle="dashed", color="k")
ax.set_xlim((-6, 6))
ax.set_ylim((-6, 6))

# annotate corr
no_nan = data_filt[(~pd.isnull(data_filt["logFC_cis_HUES64"])) & 
                   (~pd.isnull(data_filt["logFC_cis_mESC"]))]
r, p = spearmanr(no_nan["logFC_cis_HUES64"], no_nan["logFC_cis_mESC"])
ax.text(0.05, 0.97, "r = {:.2f}".format(r), ha="left", va="top", fontsize=fontsize,
        transform=ax.transAxes)
ax.text(0.05, 0.90, "n = %s" % (len(data_filt)), ha="left", va="top", fontsize=fontsize,
        transform=ax.transAxes)
fig.savefig("cis_effect_bw_cells_scatter.pdf", dpi="figure", bbox_inches="tight")


# ## 6. plot cis effect sizes across biotypes

# In[43]:


# first determine which logFC to use since there are 2 options
data["logFC_cis_max"] = data.apply(cis_logFC_one, axis=1)
data["abs_logFC_cis_max"] = np.abs(data["logFC_cis_max"])

# re-filter
data_filt = data[~data["cis_status_detail_one"].str.contains("interaction")]
data_filt = data_filt[(data_filt["HUES64_padj_hg19"] < 0.05) | (data_filt["mESC_padj_mm9"] < 0.05)]
len(data_filt)


# In[44]:


clean_order = ["eRNA", "lincRNA", "lncRNA", "mRNA"]


# In[46]:


fig = plt.figure(figsize=(1.75, 1.5))
ax = sns.boxplot(data=data_filt, x="cleaner_biotype_hg19", y="abs_logFC_cis_max", 
                 flierprops = dict(marker='o', markersize=5), 
                 order=clean_order, color=sns.color_palette("Set2")[1])
mimic_r_boxplot(ax)

ax.set_xticklabels(clean_order, rotation=50, ha='right', va='top')
ax.set_xlabel("")
ax.set_ylabel("maximum cis effect size")

for i, label in enumerate(clean_order):
    n = len(data_filt[data_filt["cleaner_biotype_hg19"] == label])
    color = sns.color_palette("Set2")[1]
    ax.annotate(str(n), xy=(i, -0.7), xycoords="data", xytext=(0, 0), 
                textcoords="offset pixels", ha='center', va='bottom', 
                color=color, size=fontsize)

ax.set_ylim((-0.8, 6))

# calc p-vals b/w dists
dist1 = np.asarray(data_filt[data_filt["cleaner_biotype_hg19"] == "eRNA"]["abs_logFC_cis_max"])
dist2 = np.asarray(data_filt[data_filt["cleaner_biotype_hg19"] == "lncRNA"]["abs_logFC_cis_max"])
dist3 = np.asarray(data_filt[data_filt["cleaner_biotype_hg19"] == "mRNA"]["abs_logFC_cis_max"])

dist1 = dist1[~np.isnan(dist1)]
dist2 = dist2[~np.isnan(dist2)]
dist3 = dist3[~np.isnan(dist3)]

u12, pval12 = stats.mannwhitneyu(dist1, dist2, alternative="two-sided", use_continuity=False)
print(pval12)

u13, pval13 = stats.mannwhitneyu(dist1, dist3, alternative="two-sided", use_continuity=False)
print(pval13)

u23, pval23 = stats.mannwhitneyu(dist2, dist3, alternative="two-sided", use_continuity=False)
print(pval23)

# annotate_pval(ax, 0.2, 0.8, 1.75, 0, 1.74, pval12, fontsize)
# annotate_pval(ax, 1.2, 1.8, 1.75, 0, 1.74, pval23, fontsize)
# annotate_pval(ax, 0.2, 1.8, 2.4, 0, 2.19, pval13, fontsize)

fig.savefig("cis_clean_biotype_hg19_effectsize_boxplot.pdf", dpi="figure", bbox_inches="tight")


# In[47]:


full_order = ["enhancer", "intergenic", "div_lnc", "protein_coding", "div_pc"]
full_labels = ["eRNA", "lincRNA", "div. lncRNA", "mRNA", "div. mRNA"]


# In[48]:


fig = plt.figure(figsize=(2.75, 1.5))
ax = sns.boxplot(data=data_filt, x="biotype_hg19", y="abs_logFC_cis_max", 
                 flierprops = dict(marker='o', markersize=5), 
                 order=full_order, color=sns.color_palette("Set2")[1])
mimic_r_boxplot(ax)

ax.set_xticklabels(full_labels, rotation=50, ha='right', va='top')
ax.set_xlabel("")
ax.set_ylabel("maximum cis effect size")

for i, label in enumerate(full_order):
    n = len(data_filt[data_filt["biotype_hg19"] == label])
    color = sns.color_palette("Set2")[1]
    ax.annotate(str(n), xy=(i, -0.7), xycoords="data", xytext=(0, 0), 
                textcoords="offset pixels", ha='center', va='bottom', 
                color=color, size=fontsize)

ax.set_ylim((-0.8, 6))

fig.savefig("cis_biotype_hg19_effectsize_boxplot.pdf", dpi="figure", bbox_inches="tight")


# In[49]:


fig = plt.figure(figsize=(1.75, 1.5))
ax = sns.boxplot(data=data_filt, x="cleaner_biotype_mm9", y="abs_logFC_cis_max", 
                 flierprops = dict(marker='o', markersize=5), 
                 order=clean_order, color=sns.color_palette("Set2")[0])
mimic_r_boxplot(ax)

ax.set_xticklabels(clean_order, rotation=50, ha='right', va='top')
ax.set_xlabel("")
ax.set_ylabel("maximum cis effect size")

for i, label in enumerate(clean_order):
    n = len(data_filt[data_filt["cleaner_biotype_mm9"] == label])
    color = sns.color_palette("Set2")[0]
    ax.annotate(str(n), xy=(i, -0.7), xycoords="data", xytext=(0, 0), 
                textcoords="offset pixels", ha='center', va='bottom', 
                color=color, size=fontsize)

ax.set_ylim((-0.8, 6.2))

# calc p-vals b/w dists
dist1 = np.asarray(data_filt[data_filt["cleaner_biotype_mm9"] == "eRNA"]["abs_logFC_cis_max"])
dist2 = np.asarray(data_filt[data_filt["cleaner_biotype_mm9"] == "lncRNA"]["abs_logFC_cis_max"])
dist3 = np.asarray(data_filt[data_filt["cleaner_biotype_mm9"] == "mRNA"]["abs_logFC_cis_max"])

dist1 = dist1[~np.isnan(dist1)]
dist2 = dist2[~np.isnan(dist2)]
dist3 = dist3[~np.isnan(dist3)]

u12, pval12 = stats.mannwhitneyu(dist1, dist2, alternative="two-sided", use_continuity=False)
print(pval12)

u13, pval13 = stats.mannwhitneyu(dist1, dist3, alternative="two-sided", use_continuity=False)
print(pval13)

u23, pval23 = stats.mannwhitneyu(dist2, dist3, alternative="two-sided", use_continuity=False)
print(pval23)

# annotate_pval(ax, 0.2, 0.8, 1.85, 0, 1.55, pval12, fontsize)
# annotate_pval(ax, 1.2, 1.8, 1.65, 0, 1.64, pval23, fontsize)
# annotate_pval(ax, 0.2, 1.8, 2.2, 0, 2., pval13, fontsize)

fig.savefig("cis_clean_biotype_mm9_effectsize_boxplot.pdf", dpi="figure", bbox_inches="tight")


# In[50]:


fig = plt.figure(figsize=(2.75, 1.5))
ax = sns.boxplot(data=data_filt, x="biotype_mm9", y="abs_logFC_cis_max", 
                 flierprops = dict(marker='o', markersize=5), 
                 order=full_order, color=sns.color_palette("Set2")[0])
mimic_r_boxplot(ax)

ax.set_xticklabels(full_labels, rotation=50, ha='right', va='top')
ax.set_xlabel("")
ax.set_ylabel("maximum cis effect size")

for i, label in enumerate(full_order):
    n = len(data_filt[data_filt["biotype_mm9"] == label])
    color = sns.color_palette("Set2")[0]
    ax.annotate(str(n), xy=(i, -0.7), xycoords="data", xytext=(0, 0), 
                textcoords="offset pixels", ha='center', va='bottom', 
                color=color, size=fontsize)

ax.set_ylim((-0.8, 6.2))

fig.savefig("cis_biotype_mm9_effectsize_boxplot.pdf", dpi="figure", bbox_inches="tight")


# In[53]:


switch_order = ["CAGE turnover", "eRNA", "lincRNA", "lncRNA", "mRNA"]


# In[54]:


fig = plt.figure(figsize=(1.75, 1.5))
ax = sns.boxplot(data=data_filt, x="biotype_switch_clean", y="abs_logFC_cis_max", 
                 flierprops = dict(marker='o', markersize=5), 
                 order=switch_order, color=sns.color_palette("Set2")[2])
mimic_r_boxplot(ax)

ax.set_xticklabels(switch_order, rotation=50, ha='right', va='top')
ax.set_xlabel("")
ax.set_ylabel("maximum cis effect size")

for i, label in enumerate(switch_order):
    n = len(data_filt[data_filt["biotype_switch_clean"] == label])
    color = sns.color_palette("Set2")[2]
    ax.annotate(str(n), xy=(i, -0.7), xycoords="data", xytext=(0, 0), 
                textcoords="offset pixels", ha='center', va='bottom', 
                color=color, size=fontsize)

ax.set_ylim((-0.8, 6.2))

# calc p-vals b/w dists
dist1 = np.asarray(data_filt[data_filt["biotype_switch_clean"] == "CAGE turnover"]["abs_logFC_cis_max"])
dist2 = np.asarray(data_filt[data_filt["biotype_switch_clean"] == "eRNA"]["abs_logFC_cis_max"])
dist3 = np.asarray(data_filt[data_filt["biotype_switch_clean"] == "lncRNA"]["abs_logFC_cis_max"])
dist4 = np.asarray(data_filt[data_filt["biotype_switch_clean"] == "mRNA"]["abs_logFC_cis_max"])

dist1 = dist1[~np.isnan(dist1)]
dist2 = dist2[~np.isnan(dist2)]
dist3 = dist3[~np.isnan(dist3)]
dist4 = dist4[~np.isnan(dist4)]

u12, pval12 = stats.mannwhitneyu(dist1, dist2, alternative="two-sided", use_continuity=False)
print(pval12)

u23, pval23 = stats.mannwhitneyu(dist2, dist3, alternative="two-sided", use_continuity=False)
print(pval23)

u34, pval34 = stats.mannwhitneyu(dist3, dist4, alternative="two-sided", use_continuity=False)
print(pval34)

# annotate_pval(ax, 0.2, 0.8, 1.85, 0, 1.84, pval12, fontsize)
# annotate_pval(ax, 1.2, 1.8, 1.85, 0, 1.84, pval23, fontsize)
# annotate_pval(ax, 2.2, 2.8, 1.85, 0, 1.85, pval34, fontsize)

fig.savefig("cis_clean_biotype_switch_effectsize_boxplot.pdf", dpi="figure", bbox_inches="tight")


# In[55]:


full_switch_order = ["CAGE turnover", "enhancer", "intergenic", "div_lnc", "protein_coding", "div_pc"]
full_switch_labels = ["CAGE turnover", "eRNA", "lincRNA", "div. lncRNA", "mRNA", "div. mRNA"]


# In[56]:


fig = plt.figure(figsize=(3, 1.5))
ax = sns.boxplot(data=data_filt, x="biotype_switch", y="abs_logFC_cis_max", 
                 flierprops = dict(marker='o', markersize=5), 
                 order=full_switch_order, color=sns.color_palette("Set2")[2])
mimic_r_boxplot(ax)

ax.set_xticklabels(full_switch_labels, rotation=50, ha='right', va='top')
ax.set_xlabel("")
ax.set_ylabel("maximum cis effect size")

for i, label in enumerate(full_switch_order):
    n = len(data_filt[data_filt["biotype_switch"] == label])
    color = sns.color_palette("Set2")[2]
    ax.annotate(str(n), xy=(i, -0.7), xycoords="data", xytext=(0, 0), 
                textcoords="offset pixels", ha='center', va='bottom', 
                color=color, size=fontsize)

ax.set_ylim((-0.8, 6.2))

# # calc p-vals b/w dists
# dist1 = np.asarray(data_filt[data_filt["biotype_switch_clean"] == "CAGE turnover"]["abs_logFC_native"])
# dist2 = np.asarray(data_filt[data_filt["biotype_switch_clean"] == "eRNA"]["abs_logFC_native"])
# dist3 = np.asarray(data_filt[data_filt["biotype_switch_clean"] == "lncRNA"]["abs_logFC_native"])
# dist4 = np.asarray(data_filt[data_filt["biotype_switch_clean"] == "mRNA"]["abs_logFC_native"])

# dist1 = dist1[~np.isnan(dist1)]
# dist2 = dist2[~np.isnan(dist2)]
# dist3 = dist3[~np.isnan(dist3)]
# dist4 = dist4[~np.isnan(dist4)]

# u12, pval12 = stats.mannwhitneyu(dist1, dist2, alternative="two-sided", use_continuity=False)
# print(pval12)

# u23, pval23 = stats.mannwhitneyu(dist2, dist3, alternative="two-sided", use_continuity=False)
# print(pval23)

# u34, pval34 = stats.mannwhitneyu(dist3, dist4, alternative="two-sided", use_continuity=False)
# print(pval34)

# annotate_pval(ax, 0.2, 0.8, 1.75, 0, 1.74, pval12, fontsize)
# annotate_pval(ax, 1.2, 1.8, 1.75, 0, 1.74, pval23, fontsize)
# annotate_pval(ax, 2.2, 2.8, 1.75, 0, 1.75, pval34, fontsize)

fig.savefig("cis_biotype_switch_effectsize_boxplot.pdf", dpi="figure", bbox_inches="tight")


# ## 7. find % of significant cis effects across biotypes

# In[57]:


tots = data_filt.groupby("biotype_switch")["hg19_id"].agg("count").reset_index()
sig = data_filt[data_filt["cis_status_one"] != "no cis effect"].groupby("biotype_switch")["hg19_id"].agg("count").reset_index()
full_sig = tots.merge(sig, on="biotype_switch", how="left").fillna(0)
full_sig["percent_sig"] = (full_sig["hg19_id_y"]/full_sig["hg19_id_x"])*100
full_sig.head()


# In[58]:


# get a hypergeometric p-value for each biotype
res = {}

tot_genes = np.sum(full_sig["hg19_id_x"])
tot_native = np.sum(full_sig["hg19_id_y"])
for biotype in full_switch_order:
    row = full_sig[full_sig["biotype_switch"] == biotype].iloc[0]
    n_biotype = row.hg19_id_x
    n_native = row.hg19_id_y
    
    pval = stats.hypergeom.sf(n_native-1, tot_genes, n_biotype, tot_native)
    res[biotype] = {"pval": pval}
    
res = pd.DataFrame.from_dict(res, orient="index").reset_index()
full_sig = full_sig.merge(res, left_on="biotype_switch", right_on="index")
full_sig["padj"] = multicomp.multipletests(full_sig["pval"], method="fdr_bh")[1]
full_sig.head()


# In[54]:


fig = plt.figure(figsize=(3, 1.5))
ax = sns.barplot(data=full_sig, x="biotype_switch", y="percent_sig", 
                 order=full_switch_order, color=sns.color_palette("Set2")[2])

ax.set_xticklabels(full_switch_labels, rotation=50, ha='right', va='top')
ax.set_xlabel("")
ax.set_ylabel("% of TSSs with cis effects")

for i, label in enumerate(full_switch_order):
    n = full_sig[full_sig["biotype_switch"] == label]["hg19_id_x"].iloc[0]
    ax.annotate(str(n), xy=(i, 2), xycoords="data", xytext=(0, 0), 
                textcoords="offset pixels", ha='center', va='bottom', 
                color="white", size=fontsize)
    
    p_sig = full_sig[full_sig["biotype_switch"] == label]["percent_sig"].iloc[0]
    fdr = full_sig[full_sig["biotype_switch"] == label]["padj"].iloc[0]
    if fdr < 0.01:
        txt = "**"
        ax.annotate(txt, xy=(i, p_sig), xycoords="data", xytext=(0, -5), textcoords="offset pixels", ha='center', 
                    va='bottom', color="black", size=10)
    elif fdr < 0.05:
        txt = "*"
        ax.annotate(txt, xy=(i, p_sig), xycoords="data", xytext=(0, -5), textcoords="offset pixels", ha='center', 
                    va='bottom', color="black", size=10)
    else:
        txt = "n.s."
        ax.annotate(txt, xy=(i, p_sig), xycoords="data", xytext=(0, 0.25), textcoords="offset pixels", ha='center', 
                    va='bottom', color="black", size=fontsize)

ax.set_ylim((0, 80))

fig.savefig("perc_sig_cis_biotype_switch.pdf", dpi="figure", bbox_inches="tight")


# In[59]:


tots = data_filt.groupby("biotype_switch_clean")["hg19_id"].agg("count").reset_index()
sig = data_filt[data_filt["cis_status_one"] != "no cis effect"].groupby("biotype_switch_clean")["hg19_id"].agg("count").reset_index()
clean_sig = tots.merge(sig, on="biotype_switch_clean", how="left").fillna(0)
clean_sig["percent_sig"] = (clean_sig["hg19_id_y"]/clean_sig["hg19_id_x"])*100
clean_sig.head()


# In[60]:


# get a fisher's exact p-value for each biotype
res = {}

tot_genes = np.sum(clean_sig["hg19_id_x"])
tot_native = np.sum(clean_sig["hg19_id_y"])
for biotype in switch_order:
    row = clean_sig[clean_sig["biotype_switch_clean"] == biotype].iloc[0]
    n_biotype = row.hg19_id_x
    n_native = row.hg19_id_y
    
    pval = stats.hypergeom.sf(n_native-1, tot_genes, n_biotype, tot_native)
    res[biotype] = {"pval": pval}
    
res = pd.DataFrame.from_dict(res, orient="index").reset_index()
clean_sig = clean_sig.merge(res, left_on="biotype_switch_clean", right_on="index")
clean_sig["padj"] = multicomp.multipletests(clean_sig["pval"], method="fdr_bh")[1]
clean_sig.head()


# In[61]:


fig = plt.figure(figsize=(1.75, 1.5))
ax = sns.barplot(data=clean_sig, x="biotype_switch_clean", y="percent_sig", 
                 order=switch_order, color=sns.color_palette("Set2")[2])

ax.set_xticklabels(switch_order, rotation=50, ha='right', va='top')
ax.set_xlabel("")
ax.set_ylabel("% of TSSs with cis effects")

for i, label in enumerate(switch_order):
    n = clean_sig[clean_sig["biotype_switch_clean"] == label]["hg19_id_x"].iloc[0]
    ax.annotate(str(n), xy=(i, 2), xycoords="data", xytext=(0, 0), 
                textcoords="offset pixels", ha='center', va='bottom', 
                color="white", size=fontsize)
    
    p_sig = clean_sig[clean_sig["biotype_switch_clean"] == label]["percent_sig"].iloc[0]
    fdr = clean_sig[clean_sig["biotype_switch_clean"] == label]["padj"].iloc[0]
    if fdr < 0.01:
        txt = "**"
        ax.annotate(txt, xy=(i, p_sig), xycoords="data", xytext=(0, -5), textcoords="offset pixels", ha='center', 
                    va='bottom', color="black", size=10)
    elif fdr < 0.05:
        txt = "*"
        ax.annotate(txt, xy=(i, p_sig), xycoords="data", xytext=(0, -5), textcoords="offset pixels", ha='center', 
                    va='bottom', color="black", size=10)
    else:
        txt = "n.s."
        ax.annotate(txt, xy=(i, p_sig), xycoords="data", xytext=(0, 0.25), textcoords="offset pixels", ha='center', 
                    va='bottom', color="black", size=fontsize)

ax.set_ylim((0, 80))

fig.savefig("perc_sig_cis_clean_biotype_switch.pdf", dpi="figure", bbox_inches="tight")


# ## 8. look into complete v. partial cis gain/losses

# In[62]:


fig = plt.figure(figsize=(1.5, 1))
sns.distplot(data_filt["abs_logFC_cis_max"], hist=False, color="black",
             label="TSSs (n=%s)" % len(data_filt))

plt.ylabel("density")
plt.xlabel("maximum cis effect size")
plt.legend(loc=2, bbox_to_anchor=(1.05, 1.05))
fig.savefig("cis_effectsize_dist.pdf", dpi="figure", bbox_inches="tight")


# In[63]:


fig, ax = plt.subplots(figsize=(2.2, 2.2), nrows=1, ncols=1)

sig_human_only = data_filt[(data_filt["HUES64_padj_hg19"] < 0.01) & (data_filt["mESC_padj_mm9"] >= 0.01)]
sig_mouse_only = data_filt[(data_filt["HUES64_padj_hg19"] >= 0.01) & (data_filt["mESC_padj_mm9"] < 0.01)]
sig_both = data_filt[(data_filt["HUES64_padj_hg19"] < 0.01) & (data_filt["mESC_padj_mm9"] < 0.01)]
sig_neither = data_filt[(data_filt["HUES64_padj_hg19"] >= 0.01) & (data_filt["mESC_padj_mm9"] >= 0.01)]

ax.scatter(sig_neither["HUES64_hg19"], sig_neither["HUES64_mm9"], s=10, alpha=0.75, 
           color="gray", linewidths=0.5, edgecolors="white")
ax.scatter(sig_both["HUES64_hg19"], sig_both["HUES64_mm9"], s=8, alpha=0.5, 
           color=sns.color_palette("Set2")[2], linewidths=0.5, edgecolors="white")
ax.scatter(sig_human_only["HUES64_hg19"], sig_human_only["HUES64_mm9"], s=8, alpha=0.5, 
           color=sns.color_palette("Set2")[1], linewidths=0.5, edgecolors="white")
ax.scatter(sig_mouse_only["HUES64_hg19"], sig_mouse_only["HUES64_mm9"], s=8, alpha=0.5, 
           color=sns.color_palette("Set2")[0], linewidths=0.5, edgecolors="white")

plt.xlabel("human activity in hESCs")
plt.ylabel("mouse activity in hESCs")
ax.set_xscale("symlog")
ax.set_yscale("symlog")
ax.plot([-0.2, 25], [-0.2, 25], linestyle="dashed", color="k")
ax.set_xlim((-0.2, 25))
ax.set_ylim((-0.2, 25))


# In[64]:


fig, ax = plt.subplots(figsize=(2.2, 2.2), nrows=1, ncols=1)

sig_human_only = data_filt[(data_filt["HUES64_padj_hg19"] < 0.01) & (data_filt["mESC_padj_mm9"] >= 0.01)]
sig_mouse_only = data_filt[(data_filt["HUES64_padj_hg19"] >= 0.01) & (data_filt["mESC_padj_mm9"] < 0.01)]
sig_both = data_filt[(data_filt["HUES64_padj_hg19"] < 0.01) & (data_filt["mESC_padj_mm9"] < 0.01)]
sig_neither = data_filt[(data_filt["HUES64_padj_hg19"] >= 0.01) & (data_filt["mESC_padj_mm9"] >= 0.01)]

ax.scatter(sig_neither["mESC_hg19"], sig_neither["mESC_mm9"], s=10, alpha=0.75, 
           color="gray", linewidths=0.5, edgecolors="white")
ax.scatter(sig_both["mESC_hg19"], sig_both["mESC_mm9"], s=8, alpha=0.5, 
           color=sns.color_palette("Set2")[2], linewidths=0.5, edgecolors="white")
ax.scatter(sig_human_only["mESC_hg19"], sig_human_only["mESC_mm9"], s=8, alpha=0.5, 
           color=sns.color_palette("Set2")[1], linewidths=0.5, edgecolors="white")
ax.scatter(sig_mouse_only["mESC_hg19"], sig_mouse_only["mESC_mm9"], s=8, alpha=0.5, 
           color=sns.color_palette("Set2")[0], linewidths=0.5, edgecolors="white")

plt.xlabel("human activity in mESCs")
plt.ylabel("mouse activity in mESCs")
ax.set_xscale("symlog")
ax.set_yscale("symlog")
ax.plot([-0.2, 25], [-0.2, 25], linestyle="dashed", color="k")
ax.set_xlim((-0.2, 25))
ax.set_ylim((-0.2, 25))


# In[65]:


fig = plt.figure(figsize=(1.5, 1))
sns.distplot(sig_human_only["abs_logFC_cis_max"], hist=True, bins=15, color="black",
             label="TSSs only sig in human context (n=%s)" % len(sig_human_only))

plt.ylabel("density")
plt.xlabel("maximum cis effect size")
plt.legend(loc=2, bbox_to_anchor=(1.05, 1.05))
plt.axvline(x=2.5, color="black", zorder=1)


# In[66]:


fig = plt.figure(figsize=(1.5, 1))
sns.distplot(sig_mouse_only["abs_logFC_cis_max"], hist=True, bins=15, color="black",
             label="TSSs only sig in mouse context (n=%s)" % len(sig_mouse_only))

plt.ylabel("density")
plt.xlabel("maximum cis effect size")
plt.legend(loc=2, bbox_to_anchor=(1.05, 1.05))
plt.axvline(x=2.5, color="black", zorder=1)


# In[63]:


###### think about this a little more later, gets confusing because now we have 2 conditions we are checking #####


# ## 9. compare cis effects to native effects

# In[70]:


fig, ax = plt.subplots(figsize=(2.2, 2.2), nrows=1, ncols=1)

ax.scatter(data_filt["logFC_cis_max"], data_filt["logFC_native"], s=10, alpha=0.75, 
           color="slategray", linewidths=0.5, edgecolors="white")

plt.xlabel("maximum cis effect size")
plt.ylabel("native effect size")

ax.plot([-7, 7], [-7, 7], linestyle="dashed", color="k")
ax.set_xlim((-7, 7))
ax.set_ylim((-7, 7))

# annotate corr
no_nan = data_filt[(~pd.isnull(data_filt["logFC_cis_max"])) & 
                   (~pd.isnull(data_filt["logFC_native"]))]
r, p = spearmanr(no_nan["logFC_cis_max"], no_nan["logFC_native"])
ax.text(0.05, 0.97, "r = {:.2f}".format(r), ha="left", va="top", fontsize=fontsize,
        transform=ax.transAxes)
ax.text(0.05, 0.90, "n = %s" % (len(data_filt)), ha="left", va="top", fontsize=fontsize,
        transform=ax.transAxes)
fig.savefig("cis_v_native_scatter.pdf", dpi="figure", bbox_inches="tight")


# In[71]:


data_filt.columns


# In[72]:


no_native_sub = data_filt[data_filt["native_status"] == "no native effect"]
native_sub = data_filt[data_filt["native_status"] != "no native effect"]


# In[73]:


order = ["no cis effect", "significant cis effect"]
pal = {"no cis effect": "gray", "significant cis effect": "black"}


# In[74]:


fig, ax = plt.subplots(figsize=(1, 1), nrows=1, ncols=1)

sns.countplot(data=no_native_sub, x="cis_status_one", ax=ax, order=order, palette=pal)
ax.set_xticklabels(order, va="top", ha="right", rotation=50)
ax.set_xlabel("")
fig.savefig("cis_countplot.no_native.pdf", dpi="figure", bbox_inches="tight")


# In[75]:


fig, ax = plt.subplots(figsize=(1, 1), nrows=1, ncols=1)

sns.countplot(data=native_sub, x="cis_status_one", ax=ax, order=order[::-1], palette=pal)
ax.set_xticklabels(order[::-1], va="top", ha="right", rotation=50)
ax.set_xlabel("")
fig.savefig("cis_countplot.native.pdf", dpi="figure", bbox_inches="tight")


# In[76]:


len(native_sub)


# In[77]:


len(native_sub[native_sub["cis_status_one"] == "significant cis effect"])


# In[78]:


len(native_sub[native_sub["cis_status_one"] == "significant cis effect"])/len(native_sub)


# In[79]:


native_human_sub = data_filt[data_filt["native_status_detail"].str.contains("human")]
native_mouse_sub = data_filt[data_filt["native_status_detail"].str.contains("mouse")]


# In[80]:


order = ["cis effect\n(higher in human)", "cis effect\n(higher in mouse)",
         "no cis effect"]
pal = {"no cis effect": "gray", 
       "cis effect\n(higher in human)": sns.color_palette("Set2")[1],
       "cis effect\n(higher in mouse)": sns.color_palette("Set2")[0]}


# In[81]:


fig, ax = plt.subplots(figsize=(1.3, 1), nrows=1, ncols=1)

sns.countplot(data=native_human_sub, x="cis_status_detail_one", ax=ax, order=order, palette=pal)
ax.set_xticklabels(order, va="top", ha="right", rotation=50)
ax.set_xlabel("")
fig.savefig("cis_countplot_detail.native_human.pdf", dpi="figure", bbox_inches="tight")


# In[82]:


order = ["cis effect\n(higher in mouse)", "cis effect\n(higher in human)",
         "no cis effect"]


# In[83]:


fig, ax = plt.subplots(figsize=(1.3, 1), nrows=1, ncols=1)

sns.countplot(data=native_mouse_sub, x="cis_status_detail_one", ax=ax, order=order, palette=pal)
ax.set_xticklabels(order, va="top", ha="right", rotation=50)
ax.set_xlabel("")
fig.savefig("cis_countplot_detail.native_mouse.pdf", dpi="figure", bbox_inches="tight")


# ## 10. examine how cis effects correlate with sequence alignment

# In[84]:


data_filt = data_filt.merge(align, on=["hg19_id", "mm9_id"])
data_filt.head()


# In[85]:


data_filt.columns


# In[86]:


fig = plt.figure(figsize=(2, 2))
ax = sns.regplot(data=data_filt, x="seq_alignment_score", y="abs_logFC_cis_max", color="slategray", fit_reg=True,
                 scatter_kws={"s": 15, "alpha": 0.75, "linewidth": 0.5, "edgecolor": "white"}, 
                 line_kws={"color": "black"})

# # highlight the complete
# sub = data_filt[data_filt["native_status_complete"].str.contains("complete")]
# ax.scatter(sub["seq_alignment_score"], sub["abs_logFC_cis_max"], color="black", s=15, alpha=1, linewidth=0.5,
#            edgecolor="white")

# annotate corr
no_nan = data_filt[(~pd.isnull(data_filt["seq_alignment_score"])) & 
                   (~pd.isnull(data_filt["abs_logFC_cis_max"]))]
r, p = spearmanr(no_nan["seq_alignment_score"], no_nan["abs_logFC_cis_max"])
ax.text(0.95, 0.97, "r = {:.2f}".format(r), ha="right", va="top", fontsize=fontsize,
        transform=ax.transAxes)
ax.text(0.95, 0.90, "n = %s" % (len(no_nan)), ha="right", va="top", fontsize=fontsize,
        transform=ax.transAxes)
        
plt.xlabel("pairwise alignment score")
plt.ylabel("maximum cis effect size")
fig.savefig("cis_effect_v_alignment.pdf", dpi="figure", bbox_inches="tight")


# In[87]:


p


# In[88]:


fig = plt.figure(figsize=(2, 2))
ax = sns.regplot(data=data_filt, x="seq_alignment_score", y="abs_logFC_cis_max", color="slategray", fit_reg=True,
                 scatter_kws={"s": 15, "alpha": 0.75, "linewidth": 0.5, "edgecolor": "white"}, 
                 line_kws={"color": "black"})

# highlight the complete
sub = data_filt[data_filt["native_status_complete"].str.contains("complete")]
ax.scatter(sub["seq_alignment_score"], sub["abs_logFC_cis_max"], color="black", s=15, alpha=1, linewidth=0.5,
           edgecolor="white")

# annotate corr
no_nan = data_filt[(~pd.isnull(data_filt["seq_alignment_score"])) & 
                   (~pd.isnull(data_filt["abs_logFC_cis_max"]))]
r, p = spearmanr(no_nan["seq_alignment_score"], no_nan["abs_logFC_cis_max"])
ax.text(0.95, 0.97, "r = {:.2f}".format(r), ha="right", va="top", fontsize=fontsize,
        transform=ax.transAxes)
ax.text(0.95, 0.90, "n = %s" % (len(no_nan)), ha="right", va="top", fontsize=fontsize,
        transform=ax.transAxes)
        
plt.xlabel("pairwise alignment score")
plt.ylabel("maximum cis effect size")
fig.savefig("cis_effect_v_alignment.completes_highlighted.pdf", dpi="figure", bbox_inches="tight")


# ## 11. write results file

# In[89]:


data.columns


# In[90]:


data.to_csv("%s/native_and_cis_effects_data.txt" % results_dir, sep="\t", index=False)


# In[ ]:




