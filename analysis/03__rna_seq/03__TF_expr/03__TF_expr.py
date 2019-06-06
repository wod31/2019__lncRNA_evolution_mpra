
# coding: utf-8

# # 03__TF_expr

# In[1]:


import warnings
warnings.filterwarnings('ignore')

import itertools
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


# ## variables

# In[4]:


hESC_expr_f = "../../../data/03__rna_seq/03__diff_expr/hESC.tpm.txt"
mESC_expr_f = "../../../data/03__rna_seq/03__diff_expr/mESC.tpm.txt"
orth_expr_f = "../../../data/03__rna_seq/03__diff_expr/orth.tpm.txt"
orth_de_f = "../../../data/03__rna_seq/03__diff_expr/orth.DESeq2.txt"


# In[5]:


orth_f = "../../../misc/01__ensembl_orthologs/ensembl96_human_mouse_orths.txt.gz"
human_gene_map_f = "../../../misc/01__ensembl_orthologs/gencode.v25lift37.GENE_ID_TO_NAME_AND_BIOTYPE_MAP.txt"
mouse_gene_map_f = "../../../misc/01__ensembl_orthologs/gencode.vM13.GENE_ID_TO_NAME_AND_BIOTYPE_MAP.txt"


# In[6]:


motif_info_dir = "../../../misc/02__motif_info"
human_map_f = "%s/01__meme_files/human_cisbp_id_map.updated.txt" % motif_info_dir
mouse_map_f = "%s/01__meme_files/mouse_cisbp_id_map.updated.txt" % motif_info_dir


# ## 1. import data

# In[7]:


hESC_expr = pd.read_table(hESC_expr_f).reset_index()
mESC_expr = pd.read_table(mESC_expr_f).reset_index()
hESC_expr.head()


# In[8]:


orth_expr = pd.read_table(orth_expr_f).reset_index()
orth_expr.head()


# In[9]:


orth_de = pd.read_table(orth_de_f).reset_index()
orth_de.head()


# In[10]:


orth = pd.read_table(orth_f)
orth.head()


# In[11]:


human_gene_map = pd.read_table(human_gene_map_f, header=None)
human_gene_map.columns = ["gene_id", "biotype", "gene_name"]
human_gene_map.head()


# In[12]:


mouse_gene_map = pd.read_table(mouse_gene_map_f, header=None)
mouse_gene_map.columns = ["gene_id", "biotype", "gene_name"]
mouse_gene_map.head()


# In[13]:


human_map = pd.read_table(human_map_f, header=None)
human_map.columns = ["motif_id", "gene_name"]
human_map.head()


# In[14]:


mouse_map = pd.read_table(mouse_map_f, header=None)
mouse_map.columns = ["motif_id", "gene_name"]
mouse_map.head()


# ## 2. do some QC on RNA-seq

# In[15]:


human_gene_map["index"] = human_gene_map["gene_id"].str.split(".", expand=True)[0]
mouse_gene_map["index"] = mouse_gene_map["gene_id"].str.split(".", expand=True)[0]
mouse_gene_map.head()


# In[16]:


hESC_expr = hESC_expr.merge(human_gene_map, on="index", how="left")
hESC_expr.sample(5)


# In[17]:


mESC_expr = mESC_expr.merge(mouse_gene_map, on="index", how="left")
mESC_expr.sample(5)


# In[18]:


human_genes_to_check = ["XIST", "SRY", "RPS4Y1", "DDX3Y", "POU5F1", "NANOG", "SOX2", "EOMES", "SOX17", "FOXA2"]


# In[20]:


human_sub = hESC_expr[hESC_expr["gene_name"].isin(human_genes_to_check)]
human_sub = pd.melt(human_sub[["gene_name", "rep1", "rep2"]], id_vars="gene_name")
human_sub.head()


# In[27]:


fig = plt.figure(figsize=(4, 1))

ax = sns.barplot(data=human_sub, x="gene_name", y="value", hue="variable", palette="Paired", 
                 order=human_genes_to_check)
#ax.set_yscale('symlog')
ax.set_xticklabels(human_genes_to_check, va="top", ha="right", rotation=50)
ax.set_ylabel("tpm")
ax.set_title("expression of human genes in hESCs")
ax.set_xlabel("")

plt.legend(loc=2, bbox_to_anchor=(1.05, 1))


# In[39]:


mouse_genes_to_check = ["Xist", "Sry", "Eif2s3y", "Ddx3y", "Pou5f1", "Nanog", "Sox2", "Eomes", "Sox17", "Foxa2"]


# In[40]:


mouse_sub = mESC_expr[mESC_expr["gene_name"].isin(mouse_genes_to_check)]
mouse_sub = pd.melt(mouse_sub[["gene_name", "rep1", "rep2", "rep3"]], id_vars="gene_name")
mouse_sub.head()


# In[41]:


mouse_sub.gene_name.unique()


# In[42]:


fig = plt.figure(figsize=(4, 1))

ax = sns.barplot(data=mouse_sub, x="gene_name", y="value", hue="variable", palette="Paired", 
                 order=mouse_genes_to_check)
#ax.set_yscale('symlog')
ax.set_xticklabels(mouse_genes_to_check, va="top", ha="right", rotation=50)
ax.set_ylabel("tpm")
ax.set_title("expression of mouse genes in mESCs")
ax.set_xlabel("")

plt.legend(loc=2, bbox_to_anchor=(1.05, 1))


# ## 3. look at expression of TFs in hESCs

# In[44]:


uniq_human_TFs = human_map["gene_name"].unique()
print(len(uniq_human_TFs))

TFs_in_seq = [x for x in uniq_human_TFs if x in list(hESC_expr["gene_name"])]
print(len(TFs_in_seq))

TFs_missing = [x for x in uniq_human_TFs if x not in list(hESC_expr["gene_name"])]
print(len(TFs_missing))


# In[45]:


TFs_missing


# the above 21 TFs are missing from the RNA-seq so will not be included (I hand checked a few and couldn't find easy aliases, might look more later)

# In[47]:


hESC_TFs = hESC_expr[hESC_expr["gene_name"].isin(uniq_human_TFs)]
print(len(hESC_TFs))
hESC_TFs.head()


# In[ ]:


fig = plt.figure(figsize=(2, 2))



# ## 4. look at DE expression in mouse

# In[43]:


orth_expr["gene_id_human"] = orth_expr["index"].str.split("__", expand=True)[0]
orth_expr["gene_id_mouse"] = orth_expr["index"].str.split("__", expand=True)[1]
orth_expr.head()


# In[13]:


orth_sub = orth[["Gene stable ID", "Mouse gene stable ID", "Gene name", "Mouse gene name"]].drop_duplicates()
orth_sub.columns = ["gene_id_human", "gene_id_mouse", "gene_name_human", "gene_name_mouse"]
orth_expr = orth_expr.merge(orth_sub, on=["gene_id_human", "gene_id_mouse"]).drop_duplicates()
orth_expr.head()


# In[ ]:




