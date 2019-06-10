
# coding: utf-8

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


# ## variables

# In[4]:


data_dir = "../../../data/02__mpra/02__activs"
human_max_f = "%s/human_TSS_vals.max_tile.txt" % data_dir
mouse_max_f = "%s/mouse_TSS_vals.max_tile.txt" % data_dir


# In[5]:


results_dir = "../../../data/02__mpra/03__results"
results_f = "%s/native_cis_trans_effects_data.txt" % results_dir


# In[6]:


cov_dir = "../../../data/05__motif_coverage"
human_bp_cov_f = "%s/hg19_evo_fimo_sig.bp_covered.txt" % cov_dir
human_max_cov_f = "%s/hg19_evo_fimo_sig.max_coverage.txt" % cov_dir
mouse_bp_cov_f = "%s/mm9_evo_fimo_sig.bp_covered.txt" % cov_dir
mouse_max_cov_f = "%s/mm9_evo_fimo_sig.max_coverage.txt" % cov_dir


# ## 1. import data

# In[7]:


human_max = pd.read_table(human_max_f, sep="\t")
mouse_max = pd.read_table(mouse_max_f, sep="\t")
human_max.head()


# In[8]:


results = pd.read_table(results_f)
results.head()


# In[9]:


human_bp_cov = pd.read_table(human_bp_cov_f, header=None)
human_bp_cov.columns = ["chr", "start", "end", "name", "tile", "strand", "n1", "n2", "n3", "perc_bp_cov"]

mouse_bp_cov = pd.read_table(mouse_bp_cov_f, header=None)
mouse_bp_cov.columns = ["chr", "start", "end", "name", "tile", "strand", "n1", "n2", "n3", "perc_bp_cov"]

human_bp_cov.head()


# In[10]:


human_max_cov = pd.read_table(human_max_cov_f, header=None)
human_max_cov.columns = ["name", "max_cov"]

mouse_max_cov = pd.read_table(mouse_max_cov_f, header=None)
mouse_max_cov.columns = ["name", "max_cov"]

human_max_cov.head()


# ## 2. join coverage w/ results

# In[11]:


human_bp_cov["hg19_id"] = human_bp_cov["name"].str.split("__", expand=True)[1]
human_bp_cov["tss_tile_num"] = human_bp_cov["name"].str.split(";", expand=True)[0].str.split("__", expand=True)[2]
human_max_cov["hg19_id"] = human_max_cov["name"].str.split("__", expand=True)[1]
human_max_cov["tss_tile_num"] = human_max_cov["name"].str.split(";", expand=True)[0].str.split("__", expand=True)[2]
human_max_cov.sample(5)


# In[12]:


mouse_bp_cov["mm9_id"] = mouse_bp_cov["name"].str.split("__", expand=True)[1]
mouse_bp_cov["tss_tile_num"] = mouse_bp_cov["name"].str.split(";", expand=True)[0].str.split("__", expand=True)[2]
mouse_max_cov["mm9_id"] = mouse_max_cov["name"].str.split("__", expand=True)[1]
mouse_max_cov["tss_tile_num"] = mouse_max_cov["name"].str.split(";", expand=True)[0].str.split("__", expand=True)[2]
mouse_max_cov.sample(5)


# In[13]:


len(human_max)


# In[14]:


human_max = human_max.merge(human_bp_cov[["hg19_id", "tss_tile_num", "perc_bp_cov"]],
                            left_on=["tss_id", "tss_tile_num"], right_on=["hg19_id", "tss_tile_num"])
len(human_max)


# In[15]:


human_max = human_max.merge(human_max_cov[["hg19_id", "tss_tile_num", "max_cov"]],
                            left_on=["hg19_id", "tss_tile_num"], right_on=["hg19_id", "tss_tile_num"])
len(human_max)


# In[16]:


# FOR LATER - WHY DO THESE 2 FILES HAVE DIFF # OF SEQS? SHOULD BE SAME
len(human_max_cov)


# In[17]:


len(human_bp_cov)


# In[18]:


len(mouse_max)


# In[19]:


mouse_max = mouse_max.merge(mouse_bp_cov[["mm9_id", "tss_tile_num", "perc_bp_cov"]],
                            left_on=["tss_id", "tss_tile_num"], right_on=["mm9_id", "tss_tile_num"])
len(mouse_max)


# In[20]:


mouse_max = mouse_max.merge(mouse_max_cov[["mm9_id", "tss_tile_num", "max_cov"]],
                            left_on=["mm9_id", "tss_tile_num"], right_on=["mm9_id", "tss_tile_num"])
len(mouse_max)


# In[21]:


results = results.merge(human_max[["hg19_id", "perc_bp_cov", "max_cov"]], on="hg19_id", how="left")
results = results.merge(mouse_max[["mm9_id", "perc_bp_cov", "max_cov"]], on="mm9_id", how="left",
                        suffixes=("_hg19", "_mm9"))
print(len(results))
results.sample(5)


# In[22]:


results["bp_cov_change"] = results["perc_bp_cov_mm9"] - results["perc_bp_cov_hg19"]
results["max_cov_change"] = results["max_cov_mm9"] - results["max_cov_hg19"]


# In[23]:


results["abs_bp_cov_change"] = np.abs(results["bp_cov_change"])
results["abs_max_cov_change"] = np.abs(results["max_cov_change"])


# ## 3. plot

# In[24]:


data_filt = results[(results["HUES64_padj_hg19"] < 0.05) | (results["mESC_padj_mm9"] < 0.05)]
data_filt = data_filt[~data_filt["cis_status_detail_one"].str.contains("interaction")]
len(data_filt)


# In[25]:


data_filt.fillna(0, inplace=True)


# In[26]:


fig = plt.figure(figsize=(2, 2))
ax = sns.regplot(data=data_filt, x="bp_cov_change", y="logFC_cis_max", color="slategray", fit_reg=True,
                 scatter_kws={"s": 15, "alpha": 0.75, "linewidth": 0.5, "edgecolor": "white"}, 
                 line_kws={"color": "black"})

# annotate corr
no_nan = data_filt[(~pd.isnull(data_filt["bp_cov_change"])) & 
                   (~pd.isnull(data_filt["logFC_cis_max"]))]
r, p = spearmanr(no_nan["bp_cov_change"], no_nan["logFC_cis_max"])
ax.text(0.95, 0.97, "r = {:.2f}".format(r), ha="right", va="top", fontsize=fontsize,
        transform=ax.transAxes)
ax.text(0.95, 0.90, "n = %s" % (len(no_nan)), ha="right", va="top", fontsize=fontsize,
        transform=ax.transAxes)
        
plt.xlabel("delta # bp covered")
plt.ylabel("maximum cis effect size")
fig.savefig("cis_effect_v_delta_bp_cov.pdf", dpi="figure", bbox_inches="tight")


# In[27]:


fig = plt.figure(figsize=(2, 2))
ax = sns.regplot(data=data_filt, x="abs_bp_cov_change", y="abs_logFC_cis_max", color="slategray", fit_reg=True,
                 scatter_kws={"s": 15, "alpha": 0.75, "linewidth": 0.5, "edgecolor": "white"}, 
                 line_kws={"color": "black"})

# annotate corr
no_nan = data_filt[(~pd.isnull(data_filt["abs_bp_cov_change"])) & 
                   (~pd.isnull(data_filt["abs_logFC_cis_max"]))]
r, p = spearmanr(no_nan["abs_bp_cov_change"], no_nan["abs_logFC_cis_max"])
ax.text(0.95, 0.97, "r = {:.2f}".format(r), ha="right", va="top", fontsize=fontsize,
        transform=ax.transAxes)
ax.text(0.95, 0.90, "n = %s" % (len(no_nan)), ha="right", va="top", fontsize=fontsize,
        transform=ax.transAxes)
        
plt.xlabel("abs. delta # bp covered")
plt.ylabel("maximum abs. cis effect size")
fig.savefig("cis_effect_v_delta_bp_cov.abs.pdf", dpi="figure", bbox_inches="tight")


# In[28]:


fig = plt.figure(figsize=(2, 2))
ax = sns.regplot(data=data_filt, x="max_cov_change", y="logFC_cis_max", color="slategray", fit_reg=True,
                 scatter_kws={"s": 15, "alpha": 0.75, "linewidth": 0.5, "edgecolor": "white"}, 
                 line_kws={"color": "black"})

# annotate corr
no_nan = data_filt[(~pd.isnull(data_filt["max_cov_change"])) & 
                   (~pd.isnull(data_filt["logFC_cis_max"]))]
r, p = spearmanr(no_nan["max_cov_change"], no_nan["logFC_cis_max"])
ax.text(0.95, 0.97, "r = {:.2f}".format(r), ha="right", va="top", fontsize=fontsize,
        transform=ax.transAxes)
ax.text(0.95, 0.90, "n = %s" % (len(no_nan)), ha="right", va="top", fontsize=fontsize,
        transform=ax.transAxes)
        
plt.xlabel("delta max coverage")
plt.ylabel("maximum cis effect size")
fig.savefig("cis_effect_v_delta_max_cov.pdf", dpi="figure", bbox_inches="tight")


# In[29]:


fig = plt.figure(figsize=(2, 2))
ax = sns.regplot(data=data_filt, x="abs_max_cov_change", y="abs_logFC_cis_max", color="slategray", fit_reg=True,
                 scatter_kws={"s": 15, "alpha": 0.75, "linewidth": 0.5, "edgecolor": "white"}, 
                 line_kws={"color": "black"})

# annotate corr
no_nan = data_filt[(~pd.isnull(data_filt["abs_max_cov_change"])) & 
                   (~pd.isnull(data_filt["abs_logFC_cis_max"]))]
r, p = spearmanr(no_nan["abs_max_cov_change"], no_nan["abs_logFC_cis_max"])
ax.text(0.95, 0.97, "r = {:.2f}".format(r), ha="right", va="top", fontsize=fontsize,
        transform=ax.transAxes)
ax.text(0.95, 0.90, "n = %s" % (len(no_nan)), ha="right", va="top", fontsize=fontsize,
        transform=ax.transAxes)
        
plt.xlabel("abs. delta max coverage")
plt.ylabel("maximum abs. cis effect size")
fig.savefig("cis_effect_v_delta_max_cov.abs.pdf", dpi="figure", bbox_inches="tight")


# In[ ]:




