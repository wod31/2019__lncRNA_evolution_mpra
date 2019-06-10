
# coding: utf-8

# # 08__trans_effects

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


# ## functions

# In[4]:


def is_ctrl(row):
    if "CONTROL" in row["index"]:
        return "control"
    else:
        return "TSS"


# In[5]:


def trans_status(row):
    if row.fdr < 0.05:
        return "significant trans effect"
    else:
        return "no trans effect"


# In[6]:


def trans_status_detail(row):
    if row.fdr < 0.05:
        if row.logFC < 0:
            return "trans effect\n(higher in human)"
        else:
            return "trans effect\n(higher in mouse)"
    else:
        return "no trans effect"


# In[7]:


def trans_status_one(row):
    if row.fdr_trans_human < 0.05 or row.fdr_trans_mouse < 0.05:
        return "significant trans effect"
    else:
        return "no trans effect"


# In[8]:


def trans_status_detail_one(row):
    if row.fdr_trans_human < 0.05:
        if row.fdr_trans_mouse < 0.05:
            # 2 sig trans effects: check both
            if row.logFC_trans_human < 0 and row.logFC_trans_mouse < 0:
                return "trans effect\n(higher in human)"
            elif row.logFC_trans_human > 0 and row.logFC_trans_mouse > 0:
                return "trans effect\n(higher in mouse)"
            else:
                return "trans effect\n(direction interaction)"
        else:
            # only sig in human, only check human
            if row.logFC_trans_human < 0:
                return "trans effect\n(higher in human)"
            else:
                return "trans effect\n(higher in mouse)"
    else:
        if row.fdr_trans_mouse < 0.05:
            # only sig in mouse, only check mouse
            if row.logFC_trans_mouse < 0:
                return "trans effect\n(higher in human)"
            else:
                return "trans effect\n(higher in mouse)"
        else:
            # no sig effects
            return "no trans effect"


# In[9]:


def trans_logFC_one(row):
    if row.fdr_trans_human < 0.05:
        if row.fdr_trans_mouse < 0.05:
            # 2 sig trans effects: take max of both
            l2fcs = list(row[["logFC_trans_human", "logFC_trans_mouse"]])
            return signed_max(l2fcs)
        else:
            # only sig in human, return human val
            return row.logFC_trans_human
    else:
        if row.fdr_trans_mouse < 0.05:
            # only sig in mouse, return mouse val
            return row.logFC_trans_mouse
        else:
            # no sig effects: take max of both
            l2fcs = list(row[["logFC_trans_human", "logFC_trans_mouse"]])
            return signed_max(l2fcs)


# In[10]:


def signed_max(nums):
    abs_nums = np.abs(nums)
    max_idx = np.argmax(abs_nums)
    return nums[max_idx]


# In[11]:


def cis_trans_status(row):
    if row.cis_status_one == "no cis effect" and row.trans_status_one == "no trans effect":
        return "no cis or trans effects"
    elif row.cis_status_one != "no cis effect" and row.trans_status_one == "no trans effect":
        return "cis effect only"
    elif row.cis_status_one == "no cis effect" and row.trans_status_one != "no trans effect":
        return "trans effect only"
    else:
        if "human" in row.trans_status_detail_one and "human" in row.cis_status_detail_one:
            return "cis and trans effects\n(directional)"
        elif "mouse" in row.trans_status_detail_one and "mouse" in row.cis_status_detail_one:
            return "cis and trans effects\n(directional)"
        else:
            return "cis and trans effects\n(compensatory)"


# In[12]:


def cis_trans_status_detail(row):
    if row.cis_status_one == "no cis effect" and row.trans_status_one == "no trans effect":
        return "no cis or trans effects"
    elif row.cis_status_one != "no cis effect" and row.trans_status_one == "no trans effect":
        if "human" in row.cis_status_detail_one:
            return "cis effect only\n(higher in human)"
        else:
            return "cis effect only\n(higher in mouse)"
    elif row.cis_status_one == "no cis effect" and row.trans_status_one != "no trans effect":
        if "human" in row.trans_status_detail_one:
            return "trans effect only\n(higher in human)"
        else:
            return "trans effect only\n(higher in mouse)"
    else:
        if "human" in row.trans_status_detail_one and "human" in row.cis_status_detail_one:
            return "cis and trans effects\n(directional: higher in human)"
        elif "mouse" in row.trans_status_detail_one and "mouse" in row.cis_status_detail_one:
            return "cis and trans effects\n(directional: higher in mouse)"
        else:
            return "cis and trans effects\n(compensatory)"


# In[13]:


def cis_trans_status_short(row):
    if row.cis_trans_status == "no cis or trans effects":
        return "no cis or trans effects"
    else:
        return "cis and/or trans effects"


# ## variables

# In[14]:


results_dir = "../../../data/02__mpra/03__results"
results_f = "%s/native_and_cis_effects_data.txt" % results_dir


# In[15]:


data_dir = "../../../data/02__mpra/02__activs"
human_trans_f = "%s/human_trans_results.txt" % data_dir
mouse_trans_f = "%s/mouse_trans_results.txt" % data_dir


# In[16]:


tss_map_f = "../../../data/01__design/01__mpra_list/mpra_tss.with_ids.UPDATED.txt"


# ## 1. import data

# In[17]:


results = pd.read_table(results_f)
results.head()


# In[18]:


human_trans = pd.read_table(human_trans_f).reset_index()
mouse_trans = pd.read_table(mouse_trans_f).reset_index()
mouse_trans.head()


# In[19]:


tss_map = pd.read_table(tss_map_f, index_col=0)
tss_map.head()


# ## 2. plot trans controls vs. TSSs

# In[20]:


human_trans["ctrl_status"] = human_trans.apply(is_ctrl, axis=1)
mouse_trans["ctrl_status"] = mouse_trans.apply(is_ctrl, axis=1)
mouse_trans.sample(5)


# In[21]:


mouse_trans.ctrl_status.value_counts()


# In[22]:


fig = plt.figure(figsize=(1.5, 1))
sns.distplot(human_trans[human_trans["ctrl_status"] == "control"]["pval"], hist=False, color="gray",
             label="negative controls (n=%s)" % len(human_trans[human_trans["ctrl_status"] == "control"]))
sns.distplot(human_trans[human_trans["ctrl_status"] != "control"]["pval"], hist=False, color="black",
             label="TSSs (n=%s)" % len(human_trans[human_trans["ctrl_status"] != "control"]))

plt.ylabel("density")
plt.xlabel("human sequence trans effect p-value")
plt.legend(loc=2, bbox_to_anchor=(1.05, 1.05))
fig.savefig("human_trans_ctrl_pval_dist.pdf", dpi="figure", bbox_inches="tight")


# In[23]:


fig = plt.figure(figsize=(1.5, 1))
sns.distplot(mouse_trans[mouse_trans["ctrl_status"] == "control"]["pval"], hist=False, color="gray",
             label="negative controls (n=%s)" % len(mouse_trans[mouse_trans["ctrl_status"] == "control"]))
sns.distplot(mouse_trans[mouse_trans["ctrl_status"] != "control"]["pval"], hist=False, color="black",
             label="TSSs (n=%s)" % len(mouse_trans[mouse_trans["ctrl_status"] != "control"]))

plt.ylabel("density")
plt.xlabel("mouse sequence trans effect p-value")
plt.legend(loc=2, bbox_to_anchor=(1.05, 1.05))
fig.savefig("mouse_trans_ctrl_pval_dist.pdf", dpi="figure", bbox_inches="tight")


# In[24]:


human_trans["abs_logFC"] = np.abs(human_trans["logFC"])
mouse_trans["abs_logFC"] = np.abs(mouse_trans["logFC"])


# In[25]:


order = ["control", "TSS"]
pal = {"control": "gray", "TSS": "black"}


# In[26]:


fig = plt.figure(figsize=(1, 1.5))
ax = sns.boxplot(data=human_trans, x="ctrl_status", y="abs_logFC", flierprops = dict(marker='o', markersize=5), 
                 order=order, palette=pal)
mimic_r_boxplot(ax)

ax.set_xticklabels(["negative\ncontrols", "TSSs"], rotation=50, ha='right', va='top')
ax.set_xlabel("")
ax.set_ylabel("human sequence trans effect size")

for i, label in enumerate(order):
    n = len(human_trans[human_trans["ctrl_status"] == label])
    color = pal[label]
    ax.annotate(str(n), xy=(i, -0.6), xycoords="data", xytext=(0, 0), 
                textcoords="offset pixels", ha='center', va='bottom', 
                color=color, size=fontsize)

ax.set_ylim((-0.75, 3.2))

# calc p-vals b/w dists
dist1 = np.asarray(human_trans[human_trans["ctrl_status"] == "control"]["abs_logFC"])
dist2 = np.asarray(human_trans[human_trans["ctrl_status"] != "control"]["abs_logFC"])

dist1 = dist1[~np.isnan(dist1)]
dist2 = dist2[~np.isnan(dist2)]

u, pval = stats.mannwhitneyu(dist1, dist2, alternative="two-sided", use_continuity=False)
print(pval)

annotate_pval(ax, 0.2, 0.8, 0.75, 0, 0.65, pval, fontsize)
fig.savefig("human_trans_ctrl_effectsize_boxplot.pdf", dpi="figure", bbox_inches="tight")


# In[27]:


fig = plt.figure(figsize=(1, 1.5))
ax = sns.boxplot(data=mouse_trans, x="ctrl_status", y="abs_logFC", flierprops = dict(marker='o', markersize=5), 
                 order=order, palette=pal)
mimic_r_boxplot(ax)

ax.set_xticklabels(["negative\ncontrols", "TSSs"], rotation=50, ha='right', va='top')
ax.set_xlabel("")
ax.set_ylabel("mouse sequence trans effect size")

for i, label in enumerate(order):
    n = len(mouse_trans[mouse_trans["ctrl_status"] == label])
    color = pal[label]
    ax.annotate(str(n), xy=(i, -0.6), xycoords="data", xytext=(0, 0), 
                textcoords="offset pixels", ha='center', va='bottom', 
                color=color, size=fontsize)

ax.set_ylim((-0.75, 3.2))

# calc p-vals b/w dists
dist1 = np.asarray(mouse_trans[mouse_trans["ctrl_status"] == "control"]["abs_logFC"])
dist2 = np.asarray(mouse_trans[mouse_trans["ctrl_status"] != "control"]["abs_logFC"])

dist1 = dist1[~np.isnan(dist1)]
dist2 = dist2[~np.isnan(dist2)]

u, pval = stats.mannwhitneyu(dist1, dist2, alternative="two-sided", use_continuity=False)
print(pval)

annotate_pval(ax, 0.2, 0.8, 0.75, 0, 0.65, pval, fontsize)
fig.savefig("mouse_trans_ctrl_effectsize_boxplot.pdf", dpi="figure", bbox_inches="tight")


# ## 3. classify trans effects

# In[28]:


human_trans["trans_status"] = human_trans.apply(trans_status, axis=1)
human_trans["trans_status_detail"] = human_trans.apply(trans_status_detail, axis=1)
human_trans.trans_status_detail.value_counts()


# In[29]:


mouse_trans["trans_status"] = mouse_trans.apply(trans_status, axis=1)
mouse_trans["trans_status_detail"] = mouse_trans.apply(trans_status_detail, axis=1)
mouse_trans.trans_status_detail.value_counts()


# ## 4. merge trans effects w/ native + cis effects

# In[30]:


human_trans["hg19_id"] = human_trans["index"].str.split("__", expand=True)[0]
human_trans["biotype_hg19"] = human_trans["index"].str.split("__", expand=True)[1]
human_trans["mm9_id"] = human_trans["index"].str.split("__", expand=True)[2]
human_trans["biotype_mm9"] = human_trans["index"].str.split("__", expand=True)[3]
human_trans.head()


# In[31]:


mouse_trans["hg19_id"] = mouse_trans["index"].str.split("__", expand=True)[0]
mouse_trans["biotype_hg19"] = mouse_trans["index"].str.split("__", expand=True)[1]
mouse_trans["mm9_id"] = mouse_trans["index"].str.split("__", expand=True)[2]
mouse_trans["biotype_mm9"] = mouse_trans["index"].str.split("__", expand=True)[3]
mouse_trans.head()


# In[32]:


human_trans_sub = human_trans[["hg19_id", "biotype_hg19", "mm9_id", "biotype_mm9", "fdr", "logFC", "abs_logFC",
                               "trans_status", "trans_status_detail"]]
human_trans_sub.columns = ["hg19_id", "biotype_hg19", "mm9_id", "biotype_mm9", "fdr_trans_human", "logFC_trans_human", 
                           "abs_logFC_trans_human", "trans_status_human", "trans_status_detail_human"]


# In[33]:


mouse_trans_sub = mouse_trans[["hg19_id", "biotype_hg19", "mm9_id", "biotype_mm9", "fdr", "logFC", "abs_logFC",
                               "trans_status", "trans_status_detail"]]
mouse_trans_sub.columns = ["hg19_id", "biotype_hg19", "mm9_id", "biotype_mm9", "fdr_trans_mouse", "logFC_trans_mouse", 
                           "abs_logFC_trans_mouse", "trans_status_mouse", "trans_status_detail_mouse"]


# In[34]:


data = results.merge(human_trans_sub, 
                     left_on=["hg19_id", "biotype_hg19", 
                              "mm9_id", "biotype_mm9"], 
                     right_on=["hg19_id", "biotype_hg19", 
                               "mm9_id", "biotype_mm9"]).merge(mouse_trans_sub,
                                                               left_on=["hg19_id", "biotype_hg19", 
                                                                        "mm9_id", "biotype_mm9"], 
                                                               right_on=["hg19_id", "biotype_hg19", 
                                                                         "mm9_id", "biotype_mm9"])
print(len(data))
data.sample(5)


# In[35]:


data["trans_status_one"] = data.apply(trans_status_one, axis=1)
data["trans_status_detail_one"] = data.apply(trans_status_detail_one, axis=1)


# In[36]:


data.trans_status_one.value_counts()


# In[37]:


data.trans_status_detail_one.value_counts()


# ## 5. plot trans effects on scatter plot

# In[38]:


# limit to those that are significant in at least 1 context
data_filt = data[(data["HUES64_padj_hg19"] < 0.05) | (data["mESC_padj_mm9"] < 0.05)]
len(data_filt)


# In[39]:


fig, ax = plt.subplots(figsize=(2.2, 2.2), nrows=1, ncols=1)

not_sig = data_filt[data_filt["fdr_trans_human"] >= 0.01]
sig = data_filt[data_filt["fdr_trans_human"] < 0.01]

ax.scatter(sig["HUES64_hg19"], sig["mESC_hg19"], s=10, alpha=0.75, 
           color="black", linewidths=0.5, edgecolors="white")
ax.scatter(not_sig["HUES64_hg19"], not_sig["mESC_hg19"], s=8, alpha=0.5, 
           color="gray", linewidths=0.5, edgecolors="white")

plt.xlabel("human activity in hESCs")
plt.ylabel("human activity in mESCs")
ax.set_xscale("symlog")
ax.set_yscale("symlog")
ax.plot([-0.2, 25], [-0.2, 25], linestyle="dashed", color="k")
ax.set_xlim((-0.2, 25))
ax.set_ylim((-0.2, 25))

# annotate corr
no_nan = data_filt[(~pd.isnull(data_filt["HUES64_hg19"])) & 
                   (~pd.isnull(data_filt["mESC_hg19"]))]
r, p = spearmanr(no_nan["HUES64_hg19"], no_nan["mESC_hg19"])
ax.text(0.05, 0.97, "r = {:.2f}".format(r), ha="left", va="top", fontsize=fontsize,
        transform=ax.transAxes)
ax.text(0.05, 0.90, "n = %s" % (len(data_filt)), ha="left", va="top", fontsize=fontsize,
        transform=ax.transAxes)
fig.savefig("trans_human_scatter.pdf", dpi="figure", bbox_inches="tight")


# In[40]:


fig, ax = plt.subplots(figsize=(2.2, 2.2), nrows=1, ncols=1)

not_sig = data_filt[data_filt["fdr_trans_mouse"] >= 0.01]
sig = data_filt[data_filt["fdr_trans_mouse"] < 0.01]

ax.scatter(sig["mESC_mm9"], sig["HUES64_mm9"], s=10, alpha=0.75, 
           color="black", linewidths=0.5, edgecolors="white")
ax.scatter(not_sig["mESC_mm9"], not_sig["HUES64_mm9"], s=8, alpha=0.5, 
           color="gray", linewidths=0.5, edgecolors="white")

plt.xlabel("mouse activity in mESCs")
plt.ylabel("mouse activity in hESCs")
ax.set_xscale("symlog")
ax.set_yscale("symlog")
ax.plot([-0.2, 25], [-0.2, 25], linestyle="dashed", color="k")
ax.set_xlim((-0.2, 25))
ax.set_ylim((-0.2, 25))

# annotate corr
no_nan = data_filt[(~pd.isnull(data_filt["mESC_mm9"])) & 
                   (~pd.isnull(data_filt["HUES64_mm9"]))]
r, p = spearmanr(no_nan["mESC_mm9"], no_nan["HUES64_mm9"])
ax.text(0.05, 0.97, "r = {:.2f}".format(r), ha="left", va="top", fontsize=fontsize,
        transform=ax.transAxes)
ax.text(0.05, 0.90, "n = %s" % (len(data_filt)), ha="left", va="top", fontsize=fontsize,
        transform=ax.transAxes)
fig.savefig("trans_mouse_scatter.pdf", dpi="figure", bbox_inches="tight")


# In[41]:


fig, ax = plt.subplots(figsize=(1.75, 1.75), nrows=1, ncols=1)

neg_ctrls = human_trans[human_trans["ctrl_status"] == "control"]
tss = human_trans[human_trans["ctrl_status"] != "control"]

ax.scatter(tss["logFC"], -np.log10(tss["fdr"]), s=10, alpha=0.75, 
           color="black", linewidths=0.5, edgecolors="white")
ax.scatter(neg_ctrls["logFC"], -np.log10(neg_ctrls["fdr"]), s=8, alpha=0.5, 
           color="gray", linewidths=0.5, edgecolors="white")

plt.xlabel("log2(human seq. in mESCs/human seq. in hESCs)\n(trans effect size)")
plt.ylabel("-log10(FDR)")
ax.axhline(y=-np.log10(0.05), color="black", linestyle="dashed")


fig.savefig("human_trans_volcano.ctrls_highlighted.pdf", dpi="figure", bbox_inches="tight")


# In[42]:


fig, ax = plt.subplots(figsize=(1.75, 1.75), nrows=1, ncols=1)

neg_ctrls = mouse_trans[mouse_trans["ctrl_status"] == "control"]
tss = mouse_trans[mouse_trans["ctrl_status"] != "control"]

ax.scatter(tss["logFC"], -np.log10(tss["fdr"]), s=10, alpha=0.75, 
           color="black", linewidths=0.5, edgecolors="white")
ax.scatter(neg_ctrls["logFC"], -np.log10(neg_ctrls["fdr"]), s=8, alpha=0.5, 
           color="gray", linewidths=0.5, edgecolors="white")

plt.xlabel("log2(mouse seq. in mESCs/mouse seq. in hESCs)\n(trans effect size)")
plt.ylabel("-log10(FDR)")
ax.axhline(y=-np.log10(0.05), color="black", linestyle="dashed")


fig.savefig("mouse_trans_volcano.ctrls_highlighted.pdf", dpi="figure", bbox_inches="tight")


# In[43]:


def trans_sig_status(row):
    if row.fdr_trans_human < 0.05 and row.fdr_trans_mouse < 0.05:
        return "sig_both"
    elif row.fdr_trans_human < 0.05 and row.fdr_trans_mouse >= 0.05:
        return "sig_human"
    elif row.fdr_trans_human >= 0.05 and row.fdr_trans_mouse < 0.05:
        return "sig_mouse"
    else:
        return "not_sig_both"
    
data_filt["trans_sig_status"] = data_filt.apply(trans_sig_status, axis=1)
data_filt.trans_sig_status.value_counts()


# In[44]:


# plot effect size agreement b/w the two seqs
fig, ax = plt.subplots(figsize=(1.75, 1.75), nrows=1, ncols=1)

sig_human = data_filt[data_filt["trans_sig_status"] == "sig_human"]
sig_mouse = data_filt[data_filt["trans_sig_status"] == "sig_mouse"]
sig_both = data_filt[data_filt["trans_sig_status"] == "sig_both"]
not_sig = data_filt[data_filt["trans_sig_status"] == "not_sig_both"]

ax.scatter(not_sig["logFC_trans_human"], not_sig["logFC_trans_mouse"], s=10, alpha=0.75, 
           color="gray", linewidths=0.5, edgecolors="white")
ax.scatter(sig_human["logFC_trans_human"], sig_human["logFC_trans_mouse"], s=10, alpha=0.75, 
           color=sns.color_palette("Set2")[1], linewidths=0.5, edgecolors="white")
ax.scatter(sig_mouse["logFC_trans_human"], sig_mouse["logFC_trans_mouse"], s=10, alpha=0.75, 
           color=sns.color_palette("Set2")[0], linewidths=0.5, edgecolors="white")
ax.scatter(sig_both["logFC_trans_human"], sig_both["logFC_trans_mouse"], s=12, alpha=1, 
           color="black", linewidths=0.5, edgecolors="white")

plt.xlabel("trans effect size in human")
plt.ylabel("trans effect size in mouse")

ax.axhline(y=0, color="black", linestyle="dashed")
ax.axvline(x=0, color="black", linestyle="dashed")
ax.set_xlim((-3, 2))
ax.set_ylim((-2, 2.5))

# annotate corr
no_nan = data_filt[(~pd.isnull(data_filt["logFC_trans_human"])) & 
                   (~pd.isnull(data_filt["logFC_trans_mouse"]))]
r, p = spearmanr(no_nan["logFC_trans_human"], no_nan["logFC_trans_mouse"])
ax.text(0.05, 0.97, "r = {:.2f}".format(r), ha="left", va="top", fontsize=fontsize,
        transform=ax.transAxes)
ax.text(0.05, 0.90, "n = %s" % (len(data_filt)), ha="left", va="top", fontsize=fontsize,
        transform=ax.transAxes)
fig.savefig("trans_effect_bw_seqs_scatter.sig_status_color.pdf", dpi="figure", bbox_inches="tight")


# In[45]:


# plot effect size agreement b/w the two seqs
fig, ax = plt.subplots(figsize=(1.75, 1.75), nrows=1, ncols=1)

sig_human = data_filt[data_filt["trans_status_detail_one"].str.contains("human")]
sig_mouse = data_filt[data_filt["trans_status_detail_one"].str.contains("mouse")]
sig_int = data_filt[data_filt["trans_status_detail_one"].str.contains("interaction")]
not_sig = data_filt[data_filt["trans_status_detail_one"] == "no trans effect"]

ax.scatter(not_sig["logFC_trans_human"], not_sig["logFC_trans_mouse"], s=10, alpha=0.75, 
           color="gray", linewidths=0.5, edgecolors="white")
ax.scatter(sig_human["logFC_trans_human"], sig_human["logFC_trans_mouse"], s=10, alpha=0.75, 
           color=sns.color_palette("Set2")[1], linewidths=0.5, edgecolors="white")
ax.scatter(sig_mouse["logFC_trans_human"], sig_mouse["logFC_trans_mouse"], s=10, alpha=0.75, 
           color=sns.color_palette("Set2")[0], linewidths=0.5, edgecolors="white")
ax.scatter(sig_int["logFC_trans_human"], sig_int["logFC_trans_mouse"], s=10, alpha=0.75, 
           color=sns.color_palette("Set2")[3], linewidths=0.5, edgecolors="black")

plt.xlabel("trans effect size in human")
plt.ylabel("trans effect size in mouse")

ax.plot([-3, 3], [-3, 3], linestyle="dashed", color="k")
ax.set_xlim((-3, 3))
ax.set_ylim((-3, 3))

# annotate corr
no_nan = data_filt[(~pd.isnull(data_filt["logFC_trans_human"])) & 
                   (~pd.isnull(data_filt["logFC_trans_mouse"]))]
r, p = spearmanr(no_nan["logFC_trans_human"], no_nan["logFC_trans_mouse"])
ax.text(0.05, 0.97, "r = {:.2f}".format(r), ha="left", va="top", fontsize=fontsize,
        transform=ax.transAxes)
ax.text(0.05, 0.90, "n = %s" % (len(data_filt)), ha="left", va="top", fontsize=fontsize,
        transform=ax.transAxes)
fig.savefig("trans_effect_bw_seqs_scatter.pdf", dpi="figure", bbox_inches="tight")


# In[46]:


## remove trans effects w/ direction interactions for now! (there aren't any right now)
data_filt = data[~data["trans_status_detail_one"].str.contains("interaction")]


# ## 6. plot trans effect sizes across biotypes

# In[47]:


# first determine which logFC to use since there are 2 options
data["logFC_trans_max"] = data.apply(trans_logFC_one, axis=1)
data["abs_logFC_trans_max"] = np.abs(data["logFC_trans_max"])


# In[48]:


# re-filter
data_filt = data[~data["trans_status_detail_one"].str.contains("interaction")]
print(len(data_filt))
data_filt = data_filt[(data_filt["HUES64_padj_hg19"] < 0.05) | (data_filt["mESC_padj_mm9"] < 0.05)]
len(data_filt)


# In[49]:


# # since we have 2 options here, let's plot the maximum l2fc
# data_filt["abs_logFC_trans_max"] = data_filt[["abs_logFC_trans_human", "abs_logFC_trans_mouse"]].max(axis=1)
# data_filt.sample(5)


# In[50]:


clean_order = ["eRNA", "lncRNA", "mRNA"]


# In[51]:


fig = plt.figure(figsize=(1.75, 1.5))
ax = sns.boxplot(data=data_filt, x="cleaner_biotype_hg19", y="abs_logFC_trans_max", 
                 flierprops = dict(marker='o', markersize=5), 
                 order=clean_order, color=sns.color_palette("Set2")[1])
mimic_r_boxplot(ax)

ax.set_xticklabels(clean_order, rotation=50, ha='right', va='top')
ax.set_xlabel("")
ax.set_ylabel("maximum trans effect size")

for i, label in enumerate(clean_order):
    n = len(data_filt[data_filt["cleaner_biotype_hg19"] == label])
    color = sns.color_palette("Set2")[1]
    ax.annotate(str(n), xy=(i, -0.4), xycoords="data", xytext=(0, 0), 
                textcoords="offset pixels", ha='center', va='bottom', 
                color=color, size=fontsize)

ax.set_ylim((-0.5, 2.5))

# calc p-vals b/w dists
dist1 = np.asarray(data_filt[data_filt["cleaner_biotype_hg19"] == "eRNA"]["abs_logFC_trans_max"])
dist2 = np.asarray(data_filt[data_filt["cleaner_biotype_hg19"] == "lncRNA"]["abs_logFC_trans_max"])
dist3 = np.asarray(data_filt[data_filt["cleaner_biotype_hg19"] == "mRNA"]["abs_logFC_trans_max"])

dist1 = dist1[~np.isnan(dist1)]
dist2 = dist2[~np.isnan(dist2)]
dist3 = dist3[~np.isnan(dist3)]

u12, pval12 = stats.mannwhitneyu(dist1, dist2, alternative="two-sided", use_continuity=False)
print(pval12)

u13, pval13 = stats.mannwhitneyu(dist1, dist3, alternative="two-sided", use_continuity=False)
print(pval13)

u23, pval23 = stats.mannwhitneyu(dist2, dist3, alternative="two-sided", use_continuity=False)
print(pval23)

annotate_pval(ax, 0.2, 0.8, 0.8, 0, 0.74, pval12, fontsize)
annotate_pval(ax, 1.2, 1.8, 0.8, 0, 0.79, pval23, fontsize)
annotate_pval(ax, 0.2, 1.8, 1.2, 0, 1.1, pval13, fontsize)

fig.savefig("trans_clean_biotype_hg19_effectsize_boxplot.pdf", dpi="figure", bbox_inches="tight")


# In[52]:


full_order = ["enhancer", "intergenic", "div_lnc", "protein_coding", "div_pc"]
full_labels = ["eRNA", "lincRNA", "div. lncRNA", "mRNA", "div. mRNA"]


# In[53]:


fig = plt.figure(figsize=(2.75, 1.5))
ax = sns.boxplot(data=data_filt, x="biotype_hg19", y="abs_logFC_trans_max", 
                 flierprops = dict(marker='o', markersize=5), 
                 order=full_order, color=sns.color_palette("Set2")[1])
mimic_r_boxplot(ax)

ax.set_xticklabels(full_labels, rotation=50, ha='right', va='top')
ax.set_xlabel("")
ax.set_ylabel("maximum trans effect size")

for i, label in enumerate(full_order):
    n = len(data_filt[data_filt["biotype_hg19"] == label])
    color = sns.color_palette("Set2")[1]
    ax.annotate(str(n), xy=(i, -0.7), xycoords="data", xytext=(0, 0), 
                textcoords="offset pixels", ha='center', va='bottom', 
                color=color, size=fontsize)

ax.set_ylim((-0.8, 2.2))

fig.savefig("trans_biotype_hg19_effectsize_boxplot.pdf", dpi="figure", bbox_inches="tight")


# In[54]:


fig = plt.figure(figsize=(1.75, 1.5))
ax = sns.boxplot(data=data_filt, x="cleaner_biotype_mm9", y="abs_logFC_trans_max", 
                 flierprops = dict(marker='o', markersize=5), 
                 order=clean_order, color=sns.color_palette("Set2")[0])
mimic_r_boxplot(ax)

ax.set_xticklabels(clean_order, rotation=50, ha='right', va='top')
ax.set_xlabel("")
ax.set_ylabel("maximum trans effect size")

for i, label in enumerate(clean_order):
    n = len(data_filt[data_filt["cleaner_biotype_mm9"] == label])
    color = sns.color_palette("Set2")[0]
    ax.annotate(str(n), xy=(i, -0.7), xycoords="data", xytext=(0, 0), 
                textcoords="offset pixels", ha='center', va='bottom', 
                color=color, size=fontsize)

ax.set_ylim((-0.8, 2.2))

# calc p-vals b/w dists
dist1 = np.asarray(data_filt[data_filt["cleaner_biotype_mm9"] == "eRNA"]["abs_logFC_trans_max"])
dist2 = np.asarray(data_filt[data_filt["cleaner_biotype_mm9"] == "lncRNA"]["abs_logFC_trans_max"])
dist3 = np.asarray(data_filt[data_filt["cleaner_biotype_mm9"] == "mRNA"]["abs_logFC_trans_max"])

dist1 = dist1[~np.isnan(dist1)]
dist2 = dist2[~np.isnan(dist2)]
dist3 = dist3[~np.isnan(dist3)]

u12, pval12 = stats.mannwhitneyu(dist1, dist2, alternative="two-sided", use_continuity=False)
print(pval12)

u13, pval13 = stats.mannwhitneyu(dist1, dist3, alternative="two-sided", use_continuity=False)
print(pval13)

u23, pval23 = stats.mannwhitneyu(dist2, dist3, alternative="two-sided", use_continuity=False)
print(pval23)

annotate_pval(ax, 0.2, 0.8, 0.8, 0, 0.74, pval12, fontsize)
annotate_pval(ax, 1.2, 1.8, 0.8, 0, 0.79, pval23, fontsize)
annotate_pval(ax, 0.2, 1.8, 1.2, 0, 1.1, pval13, fontsize)

fig.savefig("trans_clean_biotype_mm9_effectsize_boxplot.pdf", dpi="figure", bbox_inches="tight")


# In[55]:


fig = plt.figure(figsize=(2.75, 1.5))
ax = sns.boxplot(data=data_filt, x="biotype_mm9", y="abs_logFC_trans_max", 
                 flierprops = dict(marker='o', markersize=5), 
                 order=full_order, color=sns.color_palette("Set2")[0])
mimic_r_boxplot(ax)

ax.set_xticklabels(full_labels, rotation=50, ha='right', va='top')
ax.set_xlabel("")
ax.set_ylabel("maximum trans effect size")

for i, label in enumerate(full_order):
    n = len(data_filt[data_filt["biotype_mm9"] == label])
    color = sns.color_palette("Set2")[0]
    ax.annotate(str(n), xy=(i, -0.7), xycoords="data", xytext=(0, 0), 
                textcoords="offset pixels", ha='center', va='bottom', 
                color=color, size=fontsize)

ax.set_ylim((-0.8, 2.2))

fig.savefig("trans_biotype_mm9_effectsize_boxplot.pdf", dpi="figure", bbox_inches="tight")


# In[56]:


switch_order = ["CAGE turnover - eRNA", "CAGE turnover - lincRNA", "CAGE turnover - lncRNA", "CAGE turnover - mRNA",
                "eRNA", "lincRNA", "lncRNA", "mRNA"]


# In[57]:


fig = plt.figure(figsize=(2.75, 1.5))
ax = sns.boxplot(data=data_filt, x="biotype_switch_clean", y="abs_logFC_trans_max", 
                 flierprops = dict(marker='o', markersize=5), 
                 order=switch_order, color=sns.color_palette("Set2")[2])
mimic_r_boxplot(ax)

ax.set_xticklabels(switch_order, rotation=50, ha='right', va='top')
ax.set_xlabel("")
ax.set_ylabel("maximum trans effect size")

for i, label in enumerate(switch_order):
    n = len(data_filt[data_filt["biotype_switch_clean"] == label])
    color = sns.color_palette("Set2")[2]
    ax.annotate(str(n), xy=(i, -0.4), xycoords="data", xytext=(0, 0), 
                textcoords="offset pixels", ha='center', va='bottom', 
                color=color, size=fontsize)

ax.set_ylim((-0.5, 2.3))

# calc p-vals b/w dists
dist1 = np.asarray(data_filt[data_filt["biotype_switch_clean"] == "CAGE turnover"]["abs_logFC_trans_max"])
dist2 = np.asarray(data_filt[data_filt["biotype_switch_clean"] == "eRNA"]["abs_logFC_trans_max"])
dist3 = np.asarray(data_filt[data_filt["biotype_switch_clean"] == "lncRNA"]["abs_logFC_trans_max"])
dist4 = np.asarray(data_filt[data_filt["biotype_switch_clean"] == "mRNA"]["abs_logFC_trans_max"])

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

# annotate_pval(ax, 0.2, 0.8, 0.9, 0, 0.8, pval12, fontsize)
# annotate_pval(ax, 1.2, 1.8, 0.9, 0, 0.9, pval23, fontsize)
# annotate_pval(ax, 2.2, 2.8, 0.9, 0, 0.8, pval34, fontsize)

fig.savefig("trans_clean_biotype_switch_effectsize_boxplot.pdf", dpi="figure", bbox_inches="tight")


# In[58]:


full_switch_order = ["CAGE turnover - enhancer", "CAGE turnover - intergenic", "CAGE turnover - div_lnc",
                     "CAGE turnover - protein_coding", "CAGE turnover - div_pc", "enhancer", "intergenic", 
                     "div_lnc", "protein_coding", "div_pc"]
full_switch_labels = ["CAGE turnover - eRNA", "CAGE turnover - lincRNA", "CAGE turnover - div. lncRNA",
                     "CAGE turnover - mRNA", "CAGE turnover - div. mRNA", "eRNA", "lincRNA", "div. lncRNA", 
                      "mRNA", "div. mRNA"]


# In[59]:


fig = plt.figure(figsize=(3, 1.5))
ax = sns.boxplot(data=data_filt, x="biotype_switch", y="abs_logFC_trans_max", 
                 flierprops = dict(marker='o', markersize=5), 
                 order=full_switch_order, color=sns.color_palette("Set2")[2])
mimic_r_boxplot(ax)

ax.set_xticklabels(full_switch_labels, rotation=50, ha='right', va='top')
ax.set_xlabel("")
ax.set_ylabel("maximum trans effect size")

for i, label in enumerate(full_switch_order):
    n = len(data_filt[data_filt["biotype_switch"] == label])
    color = sns.color_palette("Set2")[2]
    ax.annotate(str(n), xy=(i, -0.4), xycoords="data", xytext=(0, 0), 
                textcoords="offset pixels", ha='center', va='bottom', 
                color=color, size=fontsize)

ax.set_ylim((-0.5, 2.3))

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

fig.savefig("trans_biotype_switch_effectsize_boxplot.pdf", dpi="figure", bbox_inches="tight")


# ## 7. find % significant trans effects across biotypes

# In[60]:


native_sig = data_filt[data_filt["native_status"] != "no native effect"]
tots = native_sig.groupby("biotype_switch")["hg19_id"].agg("count").reset_index()
sig = native_sig[native_sig["trans_status_one"] != "no trans effect"].groupby("biotype_switch")["hg19_id"].agg("count").reset_index()
full_sig = tots.merge(sig, on="biotype_switch", how="left").fillna(0)
full_sig["percent_sig"] = (full_sig["hg19_id_y"]/full_sig["hg19_id_x"])*100
full_sig.head()


# In[61]:


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


# In[62]:


fig = plt.figure(figsize=(3, 1.5))
ax = sns.barplot(data=full_sig, x="biotype_switch", y="percent_sig", 
                 order=full_switch_order, color=sns.color_palette("Set2")[2])

ax.set_xticklabels(full_switch_labels, rotation=50, ha='right', va='top')
ax.set_xlabel("")
ax.set_ylabel("% of TSSs with trans effects")

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

ax.set_ylim((0, 40))

fig.savefig("perc_sig_trans_biotype_switch.pdf", dpi="figure", bbox_inches="tight")


# In[63]:


tots = native_sig.groupby("biotype_switch_clean")["hg19_id"].agg("count").reset_index()
sig = native_sig[native_sig["trans_status_one"] != "no trans effect"].groupby("biotype_switch_clean")["hg19_id"].agg("count").reset_index()
clean_sig = tots.merge(sig, on="biotype_switch_clean", how="left").fillna(0)
clean_sig["percent_sig"] = (clean_sig["hg19_id_y"]/clean_sig["hg19_id_x"])*100
clean_sig.head()


# In[64]:


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


# In[65]:


fig = plt.figure(figsize=(2.75, 1.5))
ax = sns.barplot(data=clean_sig, x="biotype_switch_clean", y="percent_sig", 
                 order=switch_order, color=sns.color_palette("Set2")[2])

ax.set_xticklabels(switch_order, rotation=50, ha='right', va='top')
ax.set_xlabel("")
ax.set_ylabel("% of TSSs with trans effects")

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

ax.set_ylim((0, 25))

fig.savefig("perc_sig_trans_clean_biotype_switch.pdf", dpi="figure", bbox_inches="tight")


# ## 8. compare trans effects to native effects

# In[66]:


# plot effect size agreement b/w the two cell lines
fig, ax = plt.subplots(figsize=(2.2, 2.2), nrows=1, ncols=1)

ax.scatter(data_filt["logFC_trans_max"], data_filt["logFC_native"], s=10, alpha=0.75, 
           color=sns.color_palette("Set2")[2], linewidths=0.5, edgecolors="white")

plt.xlabel("maximum trans effect size")
plt.ylabel("native effect size")

ax.plot([-6, 6], [-6, 6], linestyle="dashed", color="k")
ax.set_xlim((-6, 6))
ax.set_ylim((-6, 6))

# annotate corr
no_nan = data_filt[(~pd.isnull(data_filt["logFC_trans_max"])) & 
                   (~pd.isnull(data_filt["logFC_native"]))]
r, p = spearmanr(no_nan["logFC_trans_max"], no_nan["logFC_native"])
ax.text(0.05, 0.97, "r = {:.2f}".format(r), ha="left", va="top", fontsize=fontsize,
        transform=ax.transAxes)
ax.text(0.05, 0.90, "n = %s" % (len(data_filt)), ha="left", va="top", fontsize=fontsize,
        transform=ax.transAxes)
fig.savefig("trans_v_native_scatter.pdf", dpi="figure", bbox_inches="tight")


# In[126]:


# plot effect size agreement b/w the two cell lines
fig, ax = plt.subplots(figsize=(2.2, 2.2), nrows=1, ncols=1)

ax.scatter(data_filt["logFC_trans_max"], data_filt["logFC_native"], s=10, alpha=0.75, 
           color=sns.color_palette("Set2")[2], linewidths=0.5, edgecolors="white")

plt.xlabel("maximum trans effect size")
plt.ylabel("native effect size")

ax.set_xlim((-2, 2))
ax.set_ylim((-6, 6))


# In[67]:


no_native_sub = data_filt[data_filt["native_status"] == "no native effect"]
native_sub = data_filt[data_filt["native_status"] != "no native effect"]


# In[68]:


order = ["no trans effect", "significant trans effect"]
pal = {"no trans effect": "gray", "significant trans effect": "black"}


# In[69]:


fig, ax = plt.subplots(figsize=(1, 1), nrows=1, ncols=1)

sns.countplot(data=no_native_sub, x="trans_status_one", ax=ax, order=order, palette=pal)
ax.set_xticklabels(order, va="top", ha="right", rotation=50)
ax.set_xlabel("")
fig.savefig("trans_countplot.no_native.pdf", dpi="figure", bbox_inches="tight")


# In[70]:


fig, ax = plt.subplots(figsize=(1, 1), nrows=1, ncols=1)

sns.countplot(data=native_sub, x="trans_status_one", ax=ax, order=order[::-1], palette=pal)
ax.set_xticklabels(order[::-1], va="top", ha="right", rotation=50)
ax.set_xlabel("")
fig.savefig("trans_countplot.native.pdf", dpi="figure", bbox_inches="tight")


# In[71]:


len(native_sub)


# In[72]:


len(native_sub[native_sub["trans_status_one"] == "significant trans effect"])


# In[73]:


len(native_sub[native_sub["trans_status_one"] == "significant trans effect"])/len(native_sub)


# In[74]:


native_human_sub = data_filt[data_filt["native_status_detail"].str.contains("human")]
native_mouse_sub = data_filt[data_filt["native_status_detail"].str.contains("mouse")]


# In[75]:


order = ["trans effect\n(higher in human)", "trans effect\n(higher in mouse)",
         "no trans effect"]
pal = {"no trans effect": "gray", 
       "trans effect\n(higher in human)": sns.color_palette("Set2")[1],
       "trans effect\n(higher in mouse)": sns.color_palette("Set2")[0]}


# In[76]:


fig, ax = plt.subplots(figsize=(1.3, 1), nrows=1, ncols=1)

sns.countplot(data=native_human_sub, x="trans_status_detail_one", ax=ax, order=order, palette=pal)
ax.set_xticklabels(order, va="top", ha="right", rotation=50)
ax.set_xlabel("")
fig.savefig("trans_countplot_detail.native_human.pdf", dpi="figure", bbox_inches="tight")


# In[77]:


native_human_sub.trans_status_detail_one.value_counts()


# In[78]:


wrong_trans_dir_human = native_human_sub[native_human_sub["trans_status_detail_one"] == "trans effect\n(higher in mouse)"]


# In[79]:


wrong_trans_dir_human.cis_status_detail_one.value_counts()


# In[80]:


order = ["trans effect\n(higher in mouse)", "trans effect\n(higher in human)",
         "no trans effect"]


# In[81]:


fig, ax = plt.subplots(figsize=(1.3, 1), nrows=1, ncols=1)

sns.countplot(data=native_mouse_sub, x="trans_status_detail_one", ax=ax, order=order, palette=pal)
ax.set_xticklabels(order, va="top", ha="right", rotation=50)
ax.set_xlabel("")
fig.savefig("trans_countplot_detail.native_mouse.pdf", dpi="figure", bbox_inches="tight")


# In[82]:


native_mouse_sub.trans_status_detail_one.value_counts()


# In[83]:


wrong_trans_dir_mouse = native_mouse_sub[native_mouse_sub["trans_status_detail_one"] == "trans effect\n(higher in human)"]


# In[84]:


wrong_trans_dir_mouse.cis_status_detail_one.value_counts()


# ## 9. compare trans effects to cis effects

# In[85]:


# plot effect size agreement b/w the two cell lines
fig, ax = plt.subplots(figsize=(2.2, 2.2), nrows=1, ncols=1)

ax.scatter(data_filt["logFC_cis_max"], data_filt["logFC_trans_max"], s=10, alpha=0.75, 
           color="slategray", linewidths=0.5, edgecolors="white")

plt.xlabel("maximum cis effect size")
plt.ylabel("maximum trans effect size")

ax.axhline(y=0, color="black", linestyle="dashed")
ax.axvline(x=0, color="black", linestyle="dashed")
ax.set_xlim((-7, 7))
ax.set_ylim((-2.4, 2))

# annotate corr
no_nan = data_filt[(~pd.isnull(data_filt["logFC_cis_max"])) & 
                   (~pd.isnull(data_filt["logFC_trans_max"]))]
r, p = spearmanr(no_nan["logFC_cis_max"], no_nan["logFC_trans_max"])
print(p)
ax.text(0.05, 0.90, "r = {:.2f}".format(r), ha="left", va="top", fontsize=fontsize,
        transform=ax.transAxes)
ax.text(0.05, 0.83, "n = %s" % (len(data_filt)), ha="left", va="top", fontsize=fontsize,
        transform=ax.transAxes)
fig.savefig("cis_v_trans_scatter.pdf", dpi="figure", bbox_inches="tight")


# In[129]:


# plot effect size agreement b/w the two cell lines
fig, ax = plt.subplots(figsize=(2.2, 2.2), nrows=1, ncols=1)

sub = data_filt[data_filt["native_status"] == "no native effect"]
ax.scatter(sub["logFC_cis_max"], sub["logFC_trans_max"], s=10, alpha=0.75, 
           color="slategray", linewidths=0.5, edgecolors="white")

plt.xlabel("maximum cis effect size")
plt.ylabel("maximum trans effect size")

ax.axhline(y=0, color="black", linestyle="dashed")
ax.axvline(x=0, color="black", linestyle="dashed")
ax.set_xlim((-7, 7))
ax.set_ylim((-2.4, 2))

# annotate corr
no_nan = sub[(~pd.isnull(sub["logFC_cis_max"])) & 
             (~pd.isnull(sub["logFC_trans_max"]))]
r, p = spearmanr(no_nan["logFC_cis_max"], no_nan["logFC_trans_max"])
print(p)
ax.text(0.05, 0.90, "r = {:.2f}".format(r), ha="left", va="top", fontsize=fontsize,
        transform=ax.transAxes)
ax.text(0.05, 0.83, "n = %s" % (len(sub)), ha="left", va="top", fontsize=fontsize,
        transform=ax.transAxes)
fig.savefig("cis_v_trans_scatter.no_native.pdf", dpi="figure", bbox_inches="tight")


# In[131]:


print(len(sub[(sub["logFC_cis_max"] > 0) & (sub["logFC_trans_max"] > 0)]))
print(len(sub[(sub["logFC_cis_max"] < 0) & (sub["logFC_trans_max"] < 0)]))
print(len(sub[(sub["logFC_cis_max"] > 0) & (sub["logFC_trans_max"] < 0)]))
print(len(sub[(sub["logFC_cis_max"] < 0) & (sub["logFC_trans_max"] > 0)]))


# In[130]:


# plot effect size agreement b/w the two cell lines
fig, ax = plt.subplots(figsize=(2.2, 2.2), nrows=1, ncols=1)

sub = data_filt[data_filt["native_status"] != "no native effect"]
ax.scatter(sub["logFC_cis_max"], sub["logFC_trans_max"], s=10, alpha=0.75, 
           color="slategray", linewidths=0.5, edgecolors="white")

plt.xlabel("maximum cis effect size")
plt.ylabel("maximum trans effect size")

ax.axhline(y=0, color="black", linestyle="dashed")
ax.axvline(x=0, color="black", linestyle="dashed")
ax.set_xlim((-7, 7))
ax.set_ylim((-2.4, 2))

# annotate corr
no_nan = sub[(~pd.isnull(sub["logFC_cis_max"])) & 
             (~pd.isnull(sub["logFC_trans_max"]))]
r, p = spearmanr(no_nan["logFC_cis_max"], no_nan["logFC_trans_max"])
print(p)
ax.text(0.05, 0.90, "r = {:.2f}".format(r), ha="left", va="top", fontsize=fontsize,
        transform=ax.transAxes)
ax.text(0.05, 0.83, "n = %s" % (len(data_filt)), ha="left", va="top", fontsize=fontsize,
        transform=ax.transAxes)
fig.savefig("cis_v_trans_scatter.sig_native.pdf", dpi="figure", bbox_inches="tight")


# In[86]:


data_filt["cis_trans_status"] = data_filt.apply(cis_trans_status, axis=1)
data_filt.cis_trans_status.value_counts()


# In[87]:


data_filt["cis_trans_status_detail"] = data_filt.apply(cis_trans_status_detail, axis=1)
data_filt.cis_trans_status_detail.value_counts()


# In[88]:


# plot effect size agreement b/w the two cell lines
fig, ax = plt.subplots(figsize=(2.2, 2.2), nrows=1, ncols=1)

sub = data_filt[data_filt["cis_trans_status"].str.contains("cis and trans effects")]

sub_human = sub[sub["cis_trans_status_detail"] == "cis and trans effects\n(directional: higher in human)"]
sub_mouse = sub[sub["cis_trans_status_detail"] == "cis and trans effects\n(directional: higher in mouse)"]
sub_comp = sub[sub["cis_trans_status_detail"] == "cis and trans effects\n(compensatory)"]

ax.scatter(sub_human["logFC_cis_max"], sub_human["logFC_trans_max"], s=10, alpha=0.9, 
           color=sns.color_palette("Set2")[1], linewidths=0.5, edgecolors="white")
ax.scatter(sub_mouse["logFC_cis_max"], sub_mouse["logFC_trans_max"], s=10, alpha=0.9, 
           color=sns.color_palette("Set2")[0], linewidths=0.5, edgecolors="white")
ax.scatter(sub_comp["logFC_cis_max"], sub_comp["logFC_trans_max"], s=10, alpha=0.9, 
           color=sns.color_palette("Set2")[2], linewidths=0.5, edgecolors="white")

plt.xlabel("maximum cis effect size")
plt.ylabel("maximum trans effect size")

ax.axhline(y=0, color="black", linestyle="dashed")
ax.axvline(x=0, color="black", linestyle="dashed")
ax.set_xlim((-7.2, 7.2))
ax.set_ylim((-2.4, 2.4))

# # annotate Ns
sub_comp_left = sub_comp[sub_comp["cis_status_detail_one"] == "cis effect\n(higher in human)"]
sub_comp_right = sub_comp[sub_comp["cis_status_detail_one"] == "cis effect\n(higher in mouse)"]
ax.text(0.03, 0.97, "n = %s" % (len(sub_comp_left)), ha="left", va="top", fontsize=fontsize, 
        color=sns.color_palette("Set2")[2], transform=ax.transAxes)
ax.text(0.03, 0.03, "n = %s" % (len(sub_human)), ha="left", va="bottom", fontsize=fontsize, 
        color=sns.color_palette("Set2")[1], transform=ax.transAxes)
ax.text(0.97, 0.03, "n = %s" % (len(sub_comp_right)), ha="right", va="bottom", fontsize=fontsize, 
        color=sns.color_palette("Set2")[2], transform=ax.transAxes)
ax.text(0.97, 0.97, "n = %s" % (len(sub_mouse)), ha="right", va="top", fontsize=fontsize, 
        color=sns.color_palette("Set2")[0], transform=ax.transAxes)
fig.savefig("cis_v_trans_scatter.cis_trans_only.pdf", bbox_inches="tight", dpi="figure")


# In[89]:


# plot effect size agreement b/w the two cell lines
fig, ax = plt.subplots(figsize=(2.2, 2.2), nrows=1, ncols=1)

sub = data_filt[data_filt["cis_trans_status"].str.contains("cis and trans effects")]

sub_native = sub[sub["native_status"] != "no native effect"]
sub_no_native = sub[sub["native_status"] == "no native effect"]

ax.scatter(sub_native["logFC_cis_max"], sub_native["logFC_trans_max"], s=10, alpha=0.75, 
           color="black", linewidths=0.5, edgecolors="white")
ax.scatter(sub_no_native["logFC_cis_max"], sub_no_native["logFC_trans_max"], s=10, alpha=0.75, 
           color="gray", linewidths=0.5, edgecolors="white")

plt.xlabel("maximum cis effect size")
plt.ylabel("maximum trans effect size")

ax.axhline(y=0, color="black", linestyle="dashed")
ax.axvline(x=0, color="black", linestyle="dashed")
ax.set_xlim((-7.2, 7.2))
ax.set_ylim((-2.4, 2.4))


# In[90]:


data_filt["cis_trans_short"] = data_filt.apply(cis_trans_status_short, axis=1)
data_filt.cis_trans_short.value_counts()


# In[91]:


no_native_sub = data_filt[data_filt["native_status"] == "no native effect"]
native_sub = data_filt[data_filt["native_status"] != "no native effect"]


# In[92]:


order = ["no cis or trans effects", "cis and/or trans effects"]
pal = {"no cis or trans effects": "gray", "cis and/or trans effects": "black"}


# In[93]:


fig, ax = plt.subplots(figsize=(1, 1), nrows=1, ncols=1)

sns.countplot(data=no_native_sub, x="cis_trans_short", ax=ax, order=order, palette=pal)
ax.set_xticklabels(order, va="top", ha="right", rotation=50)
ax.set_xlabel("")
fig.savefig("cis_trans_countplot.no_native.pdf", dpi="figure", bbox_inches="tight")


# In[94]:


fig, ax = plt.subplots(figsize=(1, 1), nrows=1, ncols=1)

sns.countplot(data=native_sub, x="cis_trans_short", ax=ax, order=order[::-1], palette=pal)
ax.set_xticklabels(order[::-1], va="top", ha="right", rotation=50)
ax.set_xlabel("")
fig.savefig("cis_trans_countplot.native.pdf", dpi="figure", bbox_inches="tight")


# In[95]:


order = ["cis effect only", "trans effect only", "cis and trans effects\n(directional)", 
         "cis and trans effects\n(compensatory)", "no cis or trans effects"]
pal = {"cis effect only": "black", "trans effect only": "black", "cis and trans effects\n(directional)": "black", 
         "cis and trans effects\n(compensatory)": "black", "no cis or trans effects": "gray"}


# In[96]:


fig, ax = plt.subplots(figsize=(1.5, 1), nrows=1, ncols=1)

sns.countplot(data=no_native_sub, x="cis_trans_status", ax=ax, order=order[::-1], palette=pal)
ax.set_xticklabels(order[::-1], va="top", ha="right", rotation=50)
ax.set_xlabel("")
fig.savefig("cis_trans_countplot_more.no_native.pdf", dpi="figure", bbox_inches="tight")


# In[97]:


fig, ax = plt.subplots(figsize=(1.5, 1), nrows=1, ncols=1)

sns.countplot(data=native_sub, x="cis_trans_status", ax=ax, order=order, palette=pal)
ax.set_xticklabels(order, va="top", ha="right", rotation=50)
ax.set_xlabel("")
fig.savefig("cis_trans_countplot_more.native.pdf", dpi="figure", bbox_inches="tight")


# In[98]:


native_sub.cis_trans_status.value_counts()


# In[99]:


len(native_sub)


# In[100]:


len(native_sub[native_sub["cis_trans_short"] == "cis and/or trans effects"])


# In[101]:


len(native_sub[native_sub["cis_trans_short"] == "cis and/or trans effects"])/len(native_sub)


# In[102]:


native_human_sub = data_filt[data_filt["native_status_detail"].str.contains("human")]
native_mouse_sub = data_filt[data_filt["native_status_detail"].str.contains("mouse")]


# In[103]:


data_filt.cis_trans_status_detail.value_counts()


# In[104]:


order = ["cis effect only\n(higher in human)", "cis effect only\n(higher in mouse)", 
         "trans effect only\n(higher in human)", "trans effect only\n(higher in mouse)",
         "cis and trans effects\n(directional: higher in human)", 
         "cis and trans effects\n(directional: higher in mouse)",
         "cis and trans effects\n(compensatory)",
         "no cis or trans effects"]
pal = {"no cis or trans effects": "gray", 
       "trans effect only\n(higher in human)": sns.color_palette("Set2")[1],
       "trans effect only\n(higher in mouse)": sns.color_palette("Set2")[0],
       "cis effect only\n(higher in human)": sns.color_palette("Set2")[1],
       "cis effect only\n(higher in mouse)": sns.color_palette("Set2")[0],
       "cis and trans effects\n(directional: higher in human)": sns.color_palette("Set2")[1], 
       "cis and trans effects\n(directional: higher in mouse)": sns.color_palette("Set2")[0],
       "cis and trans effects\n(compensatory)": sns.color_palette("Set2")[2]}


# In[105]:


native_human_sub.cis_trans_status_detail.value_counts()


# In[106]:


fig, ax = plt.subplots(figsize=(3, 1), nrows=1, ncols=1)

sns.countplot(data=native_human_sub, x="cis_trans_status_detail", ax=ax, order=order, palette=pal)
ax.set_xticklabels(order, va="top", ha="right", rotation=50)
ax.set_xlabel("")
fig.savefig("cis_trans_countplot_detail.native_human.pdf", dpi="figure", bbox_inches="tight")


# In[107]:


tmp = native_human_sub[native_human_sub["cis_trans_status_detail"].str.contains("compensatory")]
print(len(tmp))
print(len(tmp[tmp["abs_logFC_trans_max"] < tmp["abs_logFC_cis_max"]]))
tmp.cis_status_detail_one.value_counts()


# In[108]:


native_mouse_sub.cis_trans_status_detail.value_counts()


# In[109]:


fig, ax = plt.subplots(figsize=(3, 1), nrows=1, ncols=1)

sns.countplot(data=native_mouse_sub, x="cis_trans_status_detail", ax=ax, order=order, palette=pal)
ax.set_xticklabels(order, va="top", ha="right", rotation=50)
ax.set_xlabel("")
fig.savefig("cis_trans_countplot_detail.native_mouse.pdf", dpi="figure", bbox_inches="tight")


# In[110]:


tmp = native_mouse_sub[native_mouse_sub["cis_trans_status_detail"].str.contains("compensatory")]
print(len(tmp))
print(len(tmp[tmp["abs_logFC_trans_max"] < tmp["abs_logFC_cis_max"]]))
tmp.cis_status_detail_one.value_counts()


# ## 10. look at cis/trans per biotype

# ### first directional

# In[111]:


tots = data_filt.groupby("biotype_switch")["hg19_id"].agg("count").reset_index()
sig = data_filt[data_filt["cis_trans_status"].str.contains("directional")].groupby("biotype_switch")["hg19_id"].agg("count").reset_index()
full_sig = tots.merge(sig, on="biotype_switch", how="left").fillna(0)
full_sig["percent_sig"] = (full_sig["hg19_id_y"]/full_sig["hg19_id_x"])*100
full_sig.head()


# In[112]:


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


# In[113]:


fig = plt.figure(figsize=(3, 1.5))
ax = sns.barplot(data=full_sig, x="biotype_switch", y="percent_sig", 
                 order=full_switch_order, color=sns.color_palette("Set2")[2])

ax.set_xticklabels(full_switch_labels, rotation=50, ha='right', va='top')
ax.set_xlabel("")
ax.set_ylabel("% of TSSs with cis/trans\ndirectional effects")

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

ax.set_ylim((0, 20))

fig.savefig("perc_sig_cis_trans_directional_biotype_switch.pdf", dpi="figure", bbox_inches="tight")


# In[114]:


tots = data_filt.groupby("biotype_switch_clean")["hg19_id"].agg("count").reset_index()
sig = data_filt[data_filt["cis_trans_status"].str.contains("directional")].groupby("biotype_switch_clean")["hg19_id"].agg("count").reset_index()
clean_sig = tots.merge(sig, on="biotype_switch_clean", how="left").fillna(0)
clean_sig["percent_sig"] = (clean_sig["hg19_id_y"]/clean_sig["hg19_id_x"])*100
clean_sig.head()


# In[115]:


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


# In[116]:


fig = plt.figure(figsize=(1.75, 1.5))
ax = sns.barplot(data=clean_sig, x="biotype_switch_clean", y="percent_sig", 
                 order=switch_order, color=sns.color_palette("Set2")[2])

ax.set_xticklabels(switch_order, rotation=50, ha='right', va='top')
ax.set_xlabel("")
ax.set_ylabel("% of TSSs with cis/trans\ndirectional effects")

for i, label in enumerate(switch_order):
    n = clean_sig[clean_sig["biotype_switch_clean"] == label]["hg19_id_x"].iloc[0]
    ax.annotate(str(n), xy=(i, 0), xycoords="data", xytext=(0, 0), 
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

ax.set_ylim((0, 15))

fig.savefig("perc_sig_cis_trans_directional_clean_biotype_switch.pdf", dpi="figure", bbox_inches="tight")


# ### then compensatory

# In[117]:


tots = data_filt.groupby("biotype_switch")["hg19_id"].agg("count").reset_index()
sig = data_filt[data_filt["cis_trans_status"].str.contains("compensatory")].groupby("biotype_switch")["hg19_id"].agg("count").reset_index()
full_sig = tots.merge(sig, on="biotype_switch", how="left").fillna(0)
full_sig["percent_sig"] = (full_sig["hg19_id_y"]/full_sig["hg19_id_x"])*100
full_sig.head()


# In[118]:


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


# In[119]:


fig = plt.figure(figsize=(3, 1.5))
ax = sns.barplot(data=full_sig, x="biotype_switch", y="percent_sig", 
                 order=full_switch_order, color=sns.color_palette("Set2")[2])

ax.set_xticklabels(full_switch_labels, rotation=50, ha='right', va='top')
ax.set_xlabel("")
ax.set_ylabel("% of TSSs with cis/trans\ncompensatory effects")

for i, label in enumerate(full_switch_order):
    n = full_sig[full_sig["biotype_switch"] == label]["hg19_id_x"].iloc[0]
    ax.annotate(str(n), xy=(i, 0), xycoords="data", xytext=(0, 0), 
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

ax.set_ylim((0, 10))

fig.savefig("perc_sig_cis_trans_compensatory_biotype_switch.pdf", dpi="figure", bbox_inches="tight")


# In[120]:


tots = data_filt.groupby("biotype_switch_clean")["hg19_id"].agg("count").reset_index()
sig = data_filt[data_filt["cis_trans_status"].str.contains("compensatory")].groupby("biotype_switch_clean")["hg19_id"].agg("count").reset_index()
clean_sig = tots.merge(sig, on="biotype_switch_clean", how="left").fillna(0)
clean_sig["percent_sig"] = (clean_sig["hg19_id_y"]/clean_sig["hg19_id_x"])*100
clean_sig.head()


# In[121]:


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


# In[122]:


fig = plt.figure(figsize=(1.75, 1.5))
ax = sns.barplot(data=clean_sig, x="biotype_switch_clean", y="percent_sig", 
                 order=switch_order, color=sns.color_palette("Set2")[2])

ax.set_xticklabels(switch_order, rotation=50, ha='right', va='top')
ax.set_xlabel("")
ax.set_ylabel("% of TSSs with cis/trans\ncompensatory effects")

for i, label in enumerate(switch_order):
    n = clean_sig[clean_sig["biotype_switch_clean"] == label]["hg19_id_x"].iloc[0]
    ax.annotate(str(n), xy=(i, 0), xycoords="data", xytext=(0, 0), 
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

ax.set_ylim((0, 10))

fig.savefig("perc_sig_cis_trans_compensatory_clean_biotype_switch.pdf", dpi="figure", bbox_inches="tight")


# In[123]:


data_filt[data_filt["cis_trans_status"].str.contains("compensatory")]


# In[124]:


data.columns


# In[125]:


data.to_csv("%s/native_cis_trans_effects_data.txt" % results_dir, sep="\t", index=False)

