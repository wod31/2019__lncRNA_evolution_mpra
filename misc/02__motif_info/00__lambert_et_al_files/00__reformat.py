
# coding: utf-8

# # notebook to reformat cisbp files to meme and other issues
# [all based on the files in the 2018 Lambert et al review "The Human Transcription Factors"]

# In[1]:


import numpy as np
import os
import pandas as pd
import sys


# ## variables

# In[2]:


tf_info_f = "00__metadata/TF_info.txt"
motif_info_f = "00__metadata/motif_info.txt"


# In[3]:


pwm_dir = "01__pwms"


# ## 1. import data

# In[4]:


tf_info = pd.read_table(tf_info_f, sep="\t")
tf_info.head()


# In[5]:


motif_info = pd.read_table(motif_info_f, sep="\t")
motif_info = motif_info[~pd.isnull(motif_info['CIS-BP ID'])]
motif_info.head()


# ## 2. read in motif files

# In[6]:


motifs = {}


# In[54]:


files = os.listdir(pwm_dir)
files = [f for f in files if "README" not in f]
print("n files: %s" % (len(files)))
for f in files:
    
    motif = f.split(".")[0]
    with open("%s/%s" % (pwm_dir, f)) as fp:
        for line in fp:
            if line.startswith("Pos"):
                continue
            info = line.split()
            if info[0] == "1":
                pwm = []

            info = line.split()
            
            # round the pwm info to 5 decimal points
            info = [round(float(x), 5) for x in info]
            
            pwm.append(info[1:])
    motifs[motif] = pwm


# In[55]:


list(motifs.keys())[0:5]


# In[56]:


motifs['HKR1']


# ## 3. map motifs to curated TFs

# In[57]:


curated_tfs = set(tf_info[tf_info["Is TF?"] == "Yes"]["Ensembl ID"])
len(curated_tfs)


# In[69]:


curated_motif_map = {}
curated_pwms = {}

for key in motifs:
    gene = motif_info[motif_info["CIS-BP ID"].str.contains(key)]["Ensembl ID"].iloc[0]
    gene_name = motif_info[motif_info["CIS-BP ID"].str.contains(key)]["HGNC symbol"].iloc[0]
    if gene in curated_tfs:
        pwm = motifs[key]
        
        # make sure the pwm sums to 1 in all rows!
        arr = np.asarray(pwm, dtype=np.float64)
        s = arr.sum(axis=1)
        if (s < 0.99).any() or (s > 1.01).any():
            print("bad motif: %s | len: %s | sums: %s" % (key, len(pwm), s))
        else:
            curated_pwms[key] = motifs[key]
            curated_motif_map[key] = {"gene_id": gene, "gene_name": gene_name}


# In[70]:


curated_motif_map = pd.DataFrame.from_dict(curated_motif_map, orient="index").reset_index()
curated_motif_map.head()


# In[71]:


len(curated_motif_map["gene_id"].unique())


# In[72]:


len(curated_motif_map)


# ## 4. convert to MEME format (for FIMO)

# In[73]:


out_f = "../01__meme_files/human_curated_tfs.txt"

with open(out_f, "w") as f:
    f.write("MEME version 4\n\n")
    f.write("ALPHABET= ACGT\n\n")
    f.write("strands: + -\n\n")
    f.write("Background letter frequencies (from uniform background):\nA 0.25000 C 0.25000 G 0.25000 T 0.25000\n\n")

    # now write the motifs
    for key in curated_pwms:
        #print(key)

        # first find its name
        motif_name = curated_motif_map[curated_motif_map["index"] == key]["gene_name"].iloc[0]
        f.write("MOTIF %s %s\n\n" % (key, motif_name))

        pwm = curated_pwms[key]
        n_bp = len(pwm)
        f.write("letter-probability matrix: alength= 4 w= %s\n" % n_bp)
        for pos in pwm:
            f.write("  %s\t%s\t%s\t%s\n" % (round(float(pos[0]), 5), round(float(pos[1]), 5), 
                                            round(float(pos[2]), 5), round(float(pos[3]), 5)))
        f.write("\n")
f.close()


# ## 5. check how many TFs in this list have orthologs according to Ensembl

# In[74]:


orths_f = "../../04__ensembl_orthologs/ensembl_human_mouse_orthologs.txt.gz"
orths = pd.read_table(orths_f, sep="\t")
orths.head()


# In[75]:


curated_motif_map_orth = curated_motif_map.merge(orths, left_on="gene_id", right_on="Gene stable ID", how="left")
curated_motif_map_orth.head()


# In[78]:


curated_motif_map_orth = curated_motif_map_orth[["index", "gene_id", "gene_name", 
                                                 "Mouse gene stable ID", "Mouse gene name"]]
curated_motif_map_orth.columns = ["motif_id", "human_gene_id", "human_gene_name", "mouse_gene_id", "mouse_gene_name"]
curated_motif_map_orth.sample(5)


# In[80]:


n_tot = len(curated_motif_map_orth["human_gene_name"].unique())
n_tot


# In[82]:


n_orth = len(curated_motif_map_orth[~pd.isnull(curated_motif_map_orth["mouse_gene_id"])]["human_gene_name"].unique())
n_orth


# In[83]:


# compare to lambert metadata file
cons_f = "00__metadata/TF_conservation.txt"
cons = pd.read_table(cons_f, sep=",")
cons.head()


# In[84]:


len(cons)


# In[85]:


cons["gene_id"] = cons["GENE"].str.split("#", expand=True)[1]
cons.head()


# In[86]:


cons_sub = cons[cons["gene_id"].isin(curated_motif_map["gene_id"])]
len(cons_sub)


# In[89]:


cons_sub_mouse_orth = cons_sub[cons_sub["Mouse"] > 0]
len(cons_sub_mouse_orth)


# In[88]:


cons_sub_mouse_orth[cons_sub_mouse_orth["GENE"].str.contains("HKR1")]


# In[97]:


tmp = curated_motif_map_orth[pd.isnull(curated_motif_map_orth["mouse_gene_id"])]
len(tmp)


# In[102]:


tmp[tmp["human_gene_name"] == "ZNF3"]


# In[103]:


len(cons_sub_mouse_orth[cons_sub_mouse_orth["gene_id"].isin(tmp["human_gene_id"])])


# In[94]:


orths[orths["Gene stable ID"] == "ENSG00000184635"]


# In[ ]:




