
# coding: utf-8

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


align_f = "../../../misc/00__tss_metadata/tss_map.seq_alignment.txt"


# ## 1. import data

# In[9]:


results = pd.read_table(results_f, sep="\t")
print(len(results))
results.head()


# In[10]:


human_max = pd.read_table(human_max_f, sep="\t")
mouse_max = pd.read_table(mouse_max_f, sep="\t")
human_max.head()


# In[11]:


motif_map = pd.read_table(motif_map_f)
motif_map.head()


# In[12]:


human_motifs = pd.read_table(human_motifs_f, sep="\t")
human_motifs.head()


# In[13]:


mouse_motifs = pd.read_table(mouse_motifs_f, sep="\t")
mouse_motifs.head()


# In[14]:


align = pd.read_table(align_f)
align.head()


# ## 2. parse motif files

# In[15]:


human_motifs = human_motifs.merge(motif_map, left_on="#pattern name", right_on="index", how="left")
human_motifs["hg19_id"] = human_motifs["sequence name"].str.split("__", expand=True)[1]
human_motifs["tile_num"] = human_motifs["sequence name"].str.split(";", expand=True)[0].str.split("__", expand=True)[2]
human_motifs["tss_strand"] = human_motifs["sequence name"].str[-2]
human_motifs.sample(5)


# In[16]:


mouse_motifs = mouse_motifs.merge(motif_map, left_on="#pattern name", right_on="index", how="left")
mouse_motifs["mm9_id"] = mouse_motifs["sequence name"].str.split("__", expand=True)[1]
mouse_motifs["tss_strand"] = mouse_motifs["sequence name"].str[-2]
mouse_motifs["tile_num"] = mouse_motifs["sequence name"].str.split(";", expand=True)[0].str.split("__", expand=True)[2]
mouse_motifs.sample(5)


# In[17]:


# limit motif tiles to those that are max tiles (since we mapped motifs in both tiles)
human_max_motifs = human_max.merge(human_motifs, left_on=["tss_id", "tss_tile_num"],
                                   right_on=["hg19_id", "tile_num"], how="left").reset_index()
human_max_motifs = human_max_motifs[~pd.isnull(human_max_motifs["element"])]
human_max_motifs.head()


# In[18]:


# limit motif tiles to those that are max tiles (since we mapped motifs in both tiles)
mouse_max_motifs = mouse_max.merge(mouse_motifs, left_on=["tss_id", "tss_tile_num"],
                                   right_on=["mm9_id", "tile_num"], how="left").reset_index()
mouse_max_motifs = mouse_max_motifs[~pd.isnull(mouse_max_motifs["element"])]
mouse_max_motifs.head()


# ## 4. calculate % aligned sequence that overlaps motifs

# In[ ]:


motif_align_res = {}
for i, row in align.iterrows():
    if i % 50 == 0:
        print(i)
    hg19_id = row.hg19_id
    mm9_id = row.mm9_id
    seq_str = row.seq_alignment_string
    
    seq_str_human = seq_str.split("\n")[0]
    seq_str_align = seq_str.split("\n")[1]
    seq_str_mouse = seq_str.split("\n")[2]
    
    
    human_sig_motifs = human_max_motifs[human_max_motifs["q-value"] < 0.05]
    mouse_sig_motifs = mouse_max_motifs[mouse_max_motifs["q-value"] < 0.05]
    
    human_motifs = human_sig_motifs[human_sig_motifs["hg19_id"] == hg19_id][["start", "stop", "gene_name"]].drop_duplicates()
    mouse_motifs = mouse_sig_motifs[mouse_sig_motifs["mm9_id"] == mm9_id][["start", "stop", "gene_name"]].drop_duplicates()
    
    human_motif_starts = list(human_motifs["start"])
    human_motif_ends = list(human_motifs["stop"])
    
    mouse_motif_starts = list(mouse_motifs["start"])
    mouse_motif_ends = list(mouse_motifs["stop"])
    
    n_human_motifs = len(human_motif_starts)
    n_mouse_motifs = len(mouse_motif_starts)
    
    human_motif_ranges = [list(range(int(human_motif_starts[x]), int(human_motif_ends[x])+1)) for x in range(n_human_motifs)]
    mouse_motif_ranges = [list(range(int(mouse_motif_starts[x]), int(mouse_motif_ends[x])+1)) for x in range(n_mouse_motifs)]
    
    human_motif_ranges = [item for sublist in human_motif_ranges for item in sublist]
    mouse_motif_ranges = [item for sublist in mouse_motif_ranges for item in sublist]
    
    # first iterate through human sequence
    human_counter = 0
    human_coverage_list = []
    human_coverage_dict = {}
    for j, b in enumerate(seq_str_human):
        if b != "-":
            human_counter += 1
            cov_count = human_motif_ranges.count(human_counter)
            #print("b: %s, counter: %s, cov: %s" % (b, human_counter, cov_count))
            human_coverage_list.append(cov_count)
        else:
            if j == 0:
                cov_count = 0
        human_coverage_dict[j] = cov_count
    
    # then iterate through mouse sequence
    mouse_counter = 0
    mouse_coverage_list = []
    mouse_coverage_dict = {}
    for j, b in enumerate(seq_str_mouse):
        if b != "-":
            mouse_counter += 1
            cov_count = mouse_motif_ranges.count(mouse_counter)
            mouse_coverage_list.append(cov_count)
        else:
            if j == 0:
                cov_count = 0
        mouse_coverage_dict[j] = cov_count
                
    # then iterate through alignment
    n_align_w_motif = 0
    n_align_wo_motif = 0
    n_unalign_w_motif = 0
    n_unalign_wo_motif = 0
    for j, a in enumerate(seq_str_align):
        human_cov = human_coverage_dict[j]
        mouse_cov = mouse_coverage_dict[j]
        if a == "|":
            if human_cov > 0 or mouse_cov > 0:
                n_align_w_motif += 1
            else:
                n_align_wo_motif += 1
        else:
            if human_cov > 0 or mouse_cov > 0:
                n_unalign_w_motif += 1
            else:
                n_unalign_wo_motif += 1
                
    comp_id = "%s__%s" % (hg19_id, mm9_id)
    motif_align_res[comp_id] = {"n_align_w_motif": n_align_w_motif, "n_align_wo_motif": n_align_wo_motif,
                                "n_unalign_w_motif": n_unalign_w_motif, "n_unalign_wo_motif": n_unalign_wo_motif}


# In[ ]:


len(motif_align_res)


# In[ ]:


motif_align_res = pd.DataFrame.from_dict(motif_align_res, orient="index").reset_index()
motif_align_res.sample(5)


# In[ ]:


motif_align_res["perc_motif_regions_aligned"] = (motif_align_res["n_align_w_motif"]/(motif_align_res["n_align_w_motif"]+motif_align_res["n_unalign_w_motif"])) * 100
motif_align_res["perc_no_motif_regions_aligned"] = (motif_align_res["n_align_wo_motif"]/(motif_align_res["n_align_wo_motif"]+motif_align_res["n_unalign_wo_motif"])) * 100
motif_align_res.sort_values(by="perc_motif_regions_aligned").head()


# In[ ]:


motif_align_res["hg19_id"] = motif_align_res["index"].str.split("__", expand=True)[0]
motif_align_res["mm9_id"] = motif_align_res["index"].str.split("__", expand=True)[1]
motif_align_res.sample(5)


# In[ ]:


print(len(results))
motif_results = results.merge(motif_align_res, on=["hg19_id", "mm9_id"])
print(len(motif_results))
motif_results.head()


# In[ ]:


data_filt = motif_results[(motif_results["HUES64_padj_hg19"] < 0.05) | (motif_results["mESC_padj_mm9"] < 0.05)]
data_filt = data_filt[~data_filt["cis_status_detail_one"].str.contains("interaction")]
data_filt.drop_duplicates()
len(data_filt)


# In[ ]:


data_filt.columns


# In[ ]:


fig = plt.figure(figsize=(2, 2))
ax = sns.regplot(data=data_filt, x="perc_motif_regions_aligned", y="abs_logFC_cis_max", color="slategray", fit_reg=True,
                 scatter_kws={"s": 15, "alpha": 0.75, "linewidth": 0.5, "edgecolor": "white"}, 
                 line_kws={"color": "black"})

# annotate corr
no_nan = data_filt[(~pd.isnull(data_filt["perc_motif_regions_aligned"])) & 
                   (~pd.isnull(data_filt["abs_logFC_cis_max"]))]
r, p = spearmanr(no_nan["perc_motif_regions_aligned"], no_nan["abs_logFC_cis_max"])
ax.text(0.95, 0.97, "r = {:.2f}".format(r), ha="right", va="top", fontsize=fontsize,
        transform=ax.transAxes)
ax.text(0.95, 0.90, "n = %s" % (len(no_nan)), ha="right", va="top", fontsize=fontsize,
        transform=ax.transAxes)
        
plt.xlabel("% motif regions aligned")
plt.ylabel("maximum cis effect size")
fig.savefig("cis_effect_v_motif_align.pdf", dpi="figure", bbox_inches="tight")


# In[ ]:


fig = plt.figure(figsize=(2, 2))
ax = sns.regplot(data=data_filt, x="perc_no_motif_regions_aligned", y="abs_logFC_cis_max", color="slategray", fit_reg=True,
                 scatter_kws={"s": 15, "alpha": 0.75, "linewidth": 0.5, "edgecolor": "white"}, 
                 line_kws={"color": "black"})

# annotate corr
no_nan = data_filt[(~pd.isnull(data_filt["perc_no_motif_regions_aligned"])) & 
                   (~pd.isnull(data_filt["abs_logFC_cis_max"]))]
r, p = spearmanr(no_nan["perc_no_motif_regions_aligned"], no_nan["abs_logFC_cis_max"])
ax.text(0.95, 0.97, "r = {:.2f}".format(r), ha="right", va="top", fontsize=fontsize,
        transform=ax.transAxes)
ax.text(0.95, 0.90, "n = %s" % (len(no_nan)), ha="right", va="top", fontsize=fontsize,
        transform=ax.transAxes)
        
plt.xlabel("% no motif regions aligned")
plt.ylabel("maximum cis effect size")
fig.savefig("cis_effect_v_no_motif_align.pdf", dpi="figure", bbox_inches="tight")


# In[ ]:




