
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

# In[30]:


data_dir = "../../../data/02__mpra/02__activs"
human_max_f = "%s/human_TSS_vals.max_tile.txt" % data_dir
mouse_max_f = "%s/mouse_TSS_vals.max_tile.txt" % data_dir


# In[12]:


results_dir = "../../../data/02__mpra/03__results"
results_f = "%s/native_cis_trans_effects_data.txt" % results_dir


# In[20]:


motif_dir = "../../../data/04__mapped_motifs"
human_motifs_f = "%s/hg19_human_curated_tfs_out/fimo.txt.gz" % motif_dir
mouse_motifs_f = "%s/mm9_human_curated_tfs_out/fimo.txt.gz" % motif_dir


# In[14]:


expr_dir = "../../../data/03__rna_seq/04__TF_expr"
orth_expr_f = "%s/orth_TF_expression.txt" % expr_dir
human_expr_f = "%s/hESC_TF_expression.txt" % expr_dir
mouse_expr_f = "%s/mESC_TF_expression.txt" % expr_dir


# ## 1. import data

# In[21]:


results = pd.read_table(results_f, sep="\t")
results.head()


# In[31]:


human_max = pd.read_table(human_max_f, sep="\t")
mouse_max = pd.read_table(mouse_max_f, sep="\t")
human_max.head()


# In[22]:


human_motifs = pd.read_table(human_motifs_f, sep="\t")
human_motifs.head()


# In[23]:


mouse_motifs = pd.read_table(mouse_motifs_f, sep="\t")
mouse_motifs.head()


# In[24]:


orth_expr = pd.read_table(orth_expr_f, sep="\t")
orth_expr.head()


# In[25]:


human_expr = pd.read_table(human_expr_f, sep="\t")
human_expr.head()


# In[26]:


mouse_expr = pd.read_table(mouse_expr_f, sep="\t")
mouse_expr.head()


# ## 2. parse motif files

# In[28]:


human_motifs["hg19_id"] = human_motifs["sequence name"].str.split("__", expand=True)[1]
human_motifs["tile_num"] = human_motifs["sequence name"].str.split(";", expand=True)[0].str.split("__", expand=True)[2]
human_motifs["tss_strand"] = human_motifs["sequence name"].str[-2]
human_motifs.head()


# In[29]:


mouse_motifs["mm9_id"] = mouse_motifs["sequence name"].str.split("__", expand=True)[1]
mouse_motifs["tss_strand"] = mouse_motifs["sequence name"].str[-2]
mouse_motifs["tile_num"] = mouse_motifs["sequence name"].str.split(";", expand=True)[0].str.split("__", expand=True)[2]
mouse_motifs.head()


# In[32]:


# limit motif tiles to those that are max tiles (since we mapped motifs in both tiles)
human_max_motifs = human_max.merge(human_motifs, left_on=["tss_id", "tss_tile_num"],
                                   right_on=["hg19_id", "tile_num"], how="left").reset_index()
human_max_motifs = human_max_motifs[~pd.isnull(human_max_motifs["element"])]
human_max_motifs.head()


# In[33]:


# limit motif tiles to those that are max tiles (since we mapped motifs in both tiles)
mouse_max_motifs = mouse_max.merge(mouse_motifs, left_on=["tss_id", "tss_tile_num"],
                                   right_on=["mm9_id", "tile_num"], how="left").reset_index()
mouse_max_motifs = mouse_max_motifs[~pd.isnull(mouse_max_motifs["element"])]
mouse_max_motifs.head()


# ## 3. find motifs enriched in trans effects

# In[35]:


uniq_human_tfs = human_max_motifs["#pattern name"].unique()
len(uniq_human_tfs)


# In[38]:


human_max_motifs.sample().iloc[0]


# In[40]:


human_motifs.sample(5)


# In[ ]:




