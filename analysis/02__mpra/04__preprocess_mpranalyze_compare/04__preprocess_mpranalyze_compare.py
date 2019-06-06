
# coding: utf-8

# # 04__preprocess_mpranalyze_compare
# 
# in this notebook, i re-shape the counts data to run MPRAnalyze comparison mode. importantly, i also include the negative controls for comparison mode that I made in the previous notebook (01). i only set MPRAnalyze comparison mode to run on the TSSs that we are interested in: that is, the MAXIMUM tile for each orthologous TSS pair.

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


# ## functions

# In[4]:


def ctrl_status(row):
    if "CONTROL" in row.comp_id:
        return True
    else:
        return False


# ## variables

# In[5]:


mpranalyze_dir = "../../../data/02__mpra/01__counts/mpranalyze_files"


# In[6]:


dna_counts_f = "%s/dna_counts.mpranalyze.for_quantification.txt" % mpranalyze_dir
rna_counts_f = "%s/rna_counts.mpranalyze.for_quantification.txt" % mpranalyze_dir


# In[7]:


data_dir = "../../../data/02__mpra/02__activs"


# In[8]:


human_max_f = "%s/human_TSS_vals.max_tile.txt" % data_dir
mouse_max_f = "%s/mouse_TSS_vals.max_tile.txt" % data_dir


# In[9]:


tss_map_f = "../../../data/01__design/00__mpra_list/mpra_tss.with_ids.UPDATED.txt"


# ## 1. import data

# In[10]:


dna_counts = pd.read_table(dna_counts_f)
dna_counts.head()


# In[11]:


rna_counts = pd.read_table(rna_counts_f)
rna_counts.head()


# In[12]:


human_max = pd.read_table(human_max_f)
mouse_max = pd.read_table(mouse_max_f)
human_max.head()


# In[13]:


tss_map = pd.read_table(tss_map_f, index_col=0)
tss_map.head()


# ## 2. remove any sequences in TSS map that we removed at initial MPRAnalyze (low counts)

# In[14]:


# filter out any elements we removed at initial steps (low dna counts)
human_max = human_max[human_max["element"].isin(dna_counts["element"])]
mouse_max = mouse_max[mouse_max["element"].isin(dna_counts["element"])]


# ## 3. get positive ctrl dna/rna counts

# In[15]:


dna_counts_ctrl = dna_counts[dna_counts["element"].str.contains("samp")]
print(len(dna_counts_ctrl))
rna_counts_ctrl = rna_counts[rna_counts["element"].str.contains("samp")]
print(len(rna_counts_ctrl))


# # first make files needed for seq. comparison (native and cis effects)

# ## 1. merge max. ortholog pairs w/ counts

# In[16]:


dna_counts_human_max = human_max[["element", "tss_id"]].merge(dna_counts, on="element")
dna_counts_mouse_max = mouse_max[["element", "tss_id"]].merge(dna_counts, on="element")
dna_counts_human_max.head()


# In[17]:


print(len(dna_counts_human_max))
print(len(dna_counts_mouse_max))


# In[18]:


rna_counts_human_max = human_max[["element", "tss_id"]].merge(rna_counts, on="element")
rna_counts_mouse_max = mouse_max[["element", "tss_id"]].merge(rna_counts, on="element")
rna_counts_human_max.head()


# In[19]:


print(len(rna_counts_human_max))
print(len(rna_counts_mouse_max))


# ## 2. merge human/mouse counts into 1 dataframe

# In[20]:


tss_map_mpra = tss_map.merge(rna_counts_human_max, left_on="hg19_id", 
                             right_on="tss_id").merge(rna_counts_mouse_max, left_on="mm9_id", right_on="tss_id",
                                                      suffixes=("__seq:human", "__seq:mouse"))
tss_map_mpra.drop_duplicates(inplace=True)
tss_map_mpra.head(5)


# In[21]:


tss_map_dna = tss_map.merge(dna_counts_human_max, left_on="hg19_id", 
                            right_on="tss_id").merge(dna_counts_mouse_max, left_on="mm9_id", right_on="tss_id",
                                                     suffixes=("__seq:human", "__seq:mouse"))
tss_map_dna.drop_duplicates(inplace=True)
tss_map_dna.head(5)


# ## 3. assign each pair an ID

# In[22]:


HUES64_rna_cols = [x for x in tss_map_mpra.columns if "samp:HUES64" in x]
mESC_rna_cols = [x for x in tss_map_mpra.columns if "samp:mESC" in x]
all_dna_cols = [x for x in tss_map_dna.columns if "samp:dna" in x]

human_cols = ["hg19_id", "biotype_hg19", "mm9_id", "biotype_mm9"]
human_cols.extend(HUES64_rna_cols)

mouse_cols = ["hg19_id", "biotype_hg19", "mm9_id", "biotype_mm9"]
mouse_cols.extend(mESC_rna_cols)

dna_cols = ["hg19_id", "biotype_hg19", "mm9_id", "biotype_mm9"]
dna_cols.extend(all_dna_cols)

tss_map_mpra_human = tss_map_mpra[human_cols]
tss_map_mpra_mouse = tss_map_mpra[mouse_cols]

tss_map_dna = tss_map_dna[dna_cols]

tss_map_mpra_human.head()


# In[23]:


tss_map_mpra["comp_id"] = tss_map_mpra["hg19_id"] + "__" + tss_map_mpra["biotype_hg19"] + "__" + tss_map_mpra["mm9_id"] + "__" + tss_map_mpra["biotype_mm9"]
tss_map_mpra_human["comp_id"] = tss_map_mpra_human["hg19_id"] + "__" + tss_map_mpra_human["biotype_hg19"] + "__" + tss_map_mpra_human["mm9_id"] + "__" + tss_map_mpra_human["biotype_mm9"] 
tss_map_mpra_mouse["comp_id"] = tss_map_mpra_mouse["hg19_id"] + "__" + tss_map_mpra_mouse["biotype_hg19"] + "__" + tss_map_mpra_mouse["mm9_id"] + "__" + tss_map_mpra_mouse["biotype_mm9"]
tss_map_dna["comp_id"] = tss_map_dna["hg19_id"] + "__" + tss_map_dna["biotype_hg19"] + "__" + tss_map_dna["mm9_id"] + "__" + tss_map_dna["biotype_mm9"]

tss_map_mpra_human.drop(["hg19_id", "biotype_hg19", "mm9_id", "biotype_mm9"], axis=1, inplace=True)
tss_map_mpra_mouse.drop(["hg19_id", "biotype_hg19", "mm9_id", "biotype_mm9"], axis=1, inplace=True)
tss_map_dna.drop(["hg19_id", "biotype_hg19", "mm9_id", "biotype_mm9"], axis=1, inplace=True)

human_cols = ["comp_id"]
human_cols.extend(HUES64_rna_cols)

mouse_cols = ["comp_id"]
mouse_cols.extend(mESC_rna_cols)

dna_cols = ["comp_id"]
dna_cols.extend(all_dna_cols)

tss_map_mpra_human = tss_map_mpra_human[human_cols]
tss_map_mpra_mouse = tss_map_mpra_mouse[mouse_cols]
tss_map_dna = tss_map_dna[dna_cols]

tss_map_mpra_human.head()


# In[24]:


# also add dataframe for native comparisons
native_cols = ["comp_id"]
native_human_cols = [x for x in tss_map_mpra.columns if "HUES64" in x and "human" in x]
native_mouse_cols = [x for x in tss_map_mpra.columns if "mESC" in x and "mouse" in x]
native_cols.extend(native_human_cols)
native_cols.extend(native_mouse_cols)
tss_map_mpra_native = tss_map_mpra[native_cols]
tss_map_mpra_native.head()


# In[25]:


# remove duplicates
tss_map_dna.drop_duplicates(inplace=True)
print(len(tss_map_dna))
print(len(tss_map_dna["comp_id"].unique()))

tss_map_mpra_human.drop_duplicates(inplace=True)
print(len(tss_map_mpra_human))
print(len(tss_map_mpra_human["comp_id"].unique()))

tss_map_mpra_mouse.drop_duplicates(inplace=True)
print(len(tss_map_mpra_mouse))
print(len(tss_map_mpra_mouse["comp_id"].unique()))

tss_map_mpra_native.drop_duplicates(inplace=True)
print(len(tss_map_mpra_native))
print(len(tss_map_mpra_native["comp_id"].unique()))


# ## 4. pair positive controls together to serve as negative controls
# for each down-sampled control element (there are 4), randomly choose 100 pairs to serve as human/mouse

# In[26]:


ctrl_ids = rna_counts_ctrl.element.unique()
ctrl_ids[0:5]


# In[27]:


ctrl_seqs = set([x.split("__")[0] for x in ctrl_ids])
samp_ids = set([x.split("__")[1] for x in ctrl_ids])


# In[28]:


all_samp_id_pairs = list(itertools.combinations(samp_ids, 2))
all_samp_id_pairs_str = ["%s__%s" % (x[0], x[1]) for x in all_samp_id_pairs]
all_samp_id_pairs_str[0:5]


# In[29]:


sampled_samp_id_pairs = np.random.choice(all_samp_id_pairs_str, size=100)
sampled_samp_id_pairs[0:5]


# In[30]:


neg_ctrls_dna = pd.DataFrame()
neg_ctrls_human = pd.DataFrame()
neg_ctrls_mouse = pd.DataFrame()
neg_ctrls_native = pd.DataFrame()

for i, seq in enumerate(ctrl_seqs):
    print("ctrl #: %s" % (i+1))
    
    for j, samp_id_pair in enumerate(sampled_samp_id_pairs):
        if j % 50 == 0:
            print("...samp pair #: %s" % (j+1))
            
        samp1 = samp_id_pair.split("__")[0] # arbitrarily call 'human' seq
        samp2 = samp_id_pair.split("__")[1] # arbitrarily call 'mouse' seq
        
        human_elem = "%s__%s" % (seq, samp1)
        mouse_elem = "%s__%s" % (seq, samp2)
        
        human_sub_dna = dna_counts_ctrl[dna_counts_ctrl["element"] == human_elem]
        mouse_sub_dna = dna_counts_ctrl[dna_counts_ctrl["element"] == mouse_elem]
        
        human_sub_rna = rna_counts_ctrl[rna_counts_ctrl["element"] == human_elem]
        mouse_sub_rna = rna_counts_ctrl[rna_counts_ctrl["element"] == mouse_elem]
        
        # re-name columns w/ species name
        human_dna_cols = ["element"]
        mouse_dna_cols = ["element"]
        human_rna_cols = ["element"]
        mouse_rna_cols = ["element"]
        
        human_dna_cols.extend(["%s__seq:human" % x for x in human_sub_dna.columns if x != "element"])
        mouse_dna_cols.extend(["%s__seq:mouse" % x for x in mouse_sub_dna.columns if x != "element"])
        
        human_rna_cols.extend(["%s__seq:human" % x for x in human_sub_rna.columns if x != "element"])
        mouse_rna_cols.extend(["%s__seq:mouse" % x for x in mouse_sub_rna.columns if x != "element"])
        
        human_sub_dna.columns = human_dna_cols
        mouse_sub_dna.columns = mouse_dna_cols
        human_sub_rna.columns = human_rna_cols
        mouse_sub_rna.columns = mouse_rna_cols
        
        # add comp_id to each df
        comp_id = "CONTROL:%s__SAMP_PAIR:%s" % ((i+1), (j+1))
        human_sub_dna["comp_id"] = comp_id
        mouse_sub_dna["comp_id"] = comp_id
        human_sub_rna["comp_id"] = comp_id
        mouse_sub_rna["comp_id"] = comp_id
        
        # merge each df into 1
        human_sub_dna.drop("element", axis=1, inplace=True)
        mouse_sub_dna.drop("element", axis=1, inplace=True)
        human_sub_rna.drop("element", axis=1, inplace=True)
        mouse_sub_rna.drop("element", axis=1, inplace=True)
        
        sub_dna = human_sub_dna.merge(mouse_sub_dna, on="comp_id")
        sub_rna = human_sub_rna.merge(mouse_sub_rna, on="comp_id")
        
        # subset rna appropriately into each negative control bucket
        sub_rna_human_cols = [x for x in sub_rna.columns if x == "comp_id" or "HUES64" in x]
        sub_rna_mouse_cols = [x for x in sub_rna.columns if x == "comp_id" or "mESC" in x]
        sub_rna_native_cols = [x for x in sub_rna.columns if x == "comp_id" or ("HUES64" in x and "human" in x) or ("mESC" in x and "mouse" in x)]
        
        sub_rna_human = sub_rna[sub_rna_human_cols]
        sub_rna_mouse = sub_rna[sub_rna_mouse_cols]
        sub_rna_native = sub_rna[sub_rna_native_cols]
        
        # append
        neg_ctrls_dna = neg_ctrls_dna.append(sub_dna)
        neg_ctrls_human = neg_ctrls_human.append(sub_rna_human)
        neg_ctrls_mouse = neg_ctrls_mouse.append(sub_rna_mouse)
        neg_ctrls_native = neg_ctrls_native.append(sub_rna_native)


# In[31]:


all_dna = tss_map_dna.append(neg_ctrls_dna)
all_dna.set_index("comp_id", inplace=True)
len(all_dna)


# In[32]:


all_rna_human = tss_map_mpra_human.append(neg_ctrls_human)
all_rna_human.set_index("comp_id", inplace=True)
len(all_rna_human)


# In[33]:


all_rna_mouse = tss_map_mpra_mouse.append(neg_ctrls_mouse)
all_rna_mouse.set_index("comp_id", inplace=True)
len(all_rna_mouse)


# In[34]:


all_rna_native = tss_map_mpra_native.append(neg_ctrls_native)
all_rna_native.set_index("comp_id", inplace=True)
len(all_rna_native)


# In[35]:


# also make file w/ everything together to test interactions!
tmp_human = all_rna_human.reset_index()
tmp_mouse = all_rna_mouse.reset_index()
all_rna = tmp_human.merge(tmp_mouse, on="comp_id")
all_cols = all_rna.columns
all_rna.set_index("comp_id", inplace=True)
len(all_rna)


# ## 5. make annotation files

# In[36]:


dna_col_ann = {}
human_col_ann = {}
mouse_col_ann = {}
native_col_ann = {}
all_col_ann = {}

for cols, ann in zip([all_dna_cols, human_cols, mouse_cols, native_cols, all_cols], 
                     [dna_col_ann, human_col_ann, mouse_col_ann, native_col_ann, all_col_ann]):
    for col in cols:
        if col == "comp_id":
            continue
        samp = col.split("__")[0].split("_")[-1][-1]
        cond = col.split(":")[1].split("_")[0]
        barc = col.split(":")[2].split("_")[0]
        seq = col.split(":")[-1]
        ann[col] = {"sample": samp, "condition": cond, "barcode": barc, "seq": seq}

dna_col_ann = pd.DataFrame.from_dict(dna_col_ann, orient="index")
human_col_ann = pd.DataFrame.from_dict(human_col_ann, orient="index")
mouse_col_ann = pd.DataFrame.from_dict(mouse_col_ann, orient="index")
native_col_ann = pd.DataFrame.from_dict(native_col_ann, orient="index")
all_col_ann = pd.DataFrame.from_dict(all_col_ann, orient="index")
native_col_ann.sample(5)


# In[37]:


all_col_ann.sample(5)


# ## 6. make control ID files

# In[38]:


ctrls = all_rna.reset_index()[["comp_id", "samp:HUES64_rep1__barc:10__seq:human"]]
ctrls["ctrl_status"] = ctrls.apply(ctrl_status, axis=1)
ctrls.drop("samp:HUES64_rep1__barc:10__seq:human", axis=1, inplace=True)
ctrls.ctrl_status.value_counts()


# In[39]:


ctrls.head()


# ## 7. write seq comparison files

# In[40]:


dna_col_ann.to_csv("%s/dna_col_ann.all_comp.mpranalyze.txt" % mpranalyze_dir, sep="\t")
human_col_ann.to_csv("%s/HUES64_col_ann.seq_comp.mpranalyze.txt" % mpranalyze_dir, sep="\t")
mouse_col_ann.to_csv("%s/mESC_col_ann.seq_comp.mpranalyze.txt" % mpranalyze_dir, sep="\t")
native_col_ann.to_csv("%s/native_col_ann.seq_comp.mpranalyze.txt" % mpranalyze_dir, sep="\t")
all_col_ann.to_csv("%s/all_col_ann.seq_comp.mpranalyze.txt" % mpranalyze_dir, sep="\t")

ctrls.to_csv("%s/ctrl_status.all_comp.mpranalyze.txt" % mpranalyze_dir, sep="\t", index=False)

all_dna.to_csv("%s/dna_counts.all_comp.mpranalyze.txt" % mpranalyze_dir, sep="\t", index=True)
all_rna_human.to_csv("%s/HUES64_rna_counts.seq_comp.mpranalyze.txt" % mpranalyze_dir, sep="\t", index=True)
all_rna_mouse.to_csv("%s/mESC_rna_counts.seq_comp.mpranalyze.txt" % mpranalyze_dir, sep="\t", index=True)
all_rna_native.to_csv("%s/native_rna_counts.seq_comp.mpranalyze.txt" % mpranalyze_dir, sep="\t", index=True)
all_rna.to_csv("%s/all_rna_counts.seq_comp.mpranalyze.txt" % mpranalyze_dir, sep="\t", index=True)


# # then make files for cell line comparisons (trans effects)

# ## 1. run trans effects separately for human seqs & mouse seqs, so subset counts dataframe

# In[41]:


human_columns = [x for x in all_rna.columns if "seq:human" in x]
mouse_columns = [x for x in all_rna.columns if "seq:mouse" in x]


# In[42]:


human_trans = all_rna[human_columns]
mouse_trans = all_rna[mouse_columns]


# In[43]:


print(len(human_trans))


# In[44]:


print(len(mouse_trans))


# ## 2. subset annotation dataframe

# In[45]:


tmp = all_col_ann.reset_index()
tmp.head()


# In[46]:


human_trans_col_ann = tmp[tmp["index"].isin(human_columns)].set_index("index")
del human_trans_col_ann.index.name
human_trans_col_ann.sample(5)


# In[47]:


mouse_trans_col_ann = tmp[tmp["index"].isin(mouse_columns)].set_index("index")
del mouse_trans_col_ann.index.name
mouse_trans_col_ann.sample(5)


# In[48]:


print(len(human_columns))
print(len(human_trans_col_ann))
print(len(mouse_columns))
print(len(mouse_trans_col_ann))


# ## 3. write cell comparison files

# In[49]:


human_trans_col_ann.to_csv("%s/human_col_ann.cell_comp.mpranalyze.txt" % mpranalyze_dir, sep="\t")
mouse_trans_col_ann.to_csv("%s/mouse_col_ann.cell_comp.mpranalyze.txt" % mpranalyze_dir, sep="\t")

human_trans.to_csv("%s/human_rna_counts.cell_comp.mpranalyze.txt" % mpranalyze_dir, sep="\t", index=True)
mouse_trans.to_csv("%s/mouse_rna_counts.cell_comp.mpranalyze.txt" % mpranalyze_dir, sep="\t", index=True)

