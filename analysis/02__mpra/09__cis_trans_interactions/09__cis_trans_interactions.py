
# coding: utf-8

# # 09__cis_trans_interactions

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


def cis_trans_int_status(row):
    if row.fdr < 0.1:
        return "significant cis/trans interaction"
    else:
        return "no cis/trans interaction"


# In[6]:


def cis_trans_int_status_detail(row):
    if row.fdr < 0.1:
        if row.logFC < 0:
            return "cis effect\n(higher in human)"
        else:
            return "cis effect\n(higher in mouse)"
    else:
        return "no cis effect"


# ## variables

# In[7]:


results_dir = "../../../data/02__mpra/03__results"
results_f = "%s/native_cis_trans_effects_data.txt" % results_dir


# In[8]:


data_dir = "../../../data/02__mpra/02__activs"
cis_trans_int_f = "%s/cis_trans_interaction_results.txt" % data_dir


# ## 1. import data

# In[9]:


results = pd.read_table(results_f)
results.head()


# In[10]:


cis_trans_int = pd.read_table(cis_trans_int_f).reset_index()
cis_trans_int.head()


# ## 2. plot controls

# In[11]:


cis_trans_int["ctrl_status"] = cis_trans_int.apply(is_ctrl, axis=1)
cis_trans_int.ctrl_status.value_counts()


# In[12]:


fig = plt.figure(figsize=(1.5, 1))
sns.distplot(cis_trans_int[cis_trans_int["ctrl_status"] == "control"]["pval"], hist=False, color="gray",
             label="negative controls (n=%s)" % len(cis_trans_int[cis_trans_int["ctrl_status"] == "control"]))
sns.distplot(cis_trans_int[cis_trans_int["ctrl_status"] != "control"]["pval"], hist=False, color="black",
             label="TSSs (n=%s)" % len(cis_trans_int[cis_trans_int["ctrl_status"] != "control"]))

plt.ylabel("density")
plt.xlabel("cis/trans interaction p-value")
plt.legend(loc=2, bbox_to_anchor=(1.05, 1.05))
fig.savefig("cis_trans_int_ctrl_pval_dist.pdf", dpi="figure", bbox_inches="tight")


# In[13]:


cis_trans_int["abs_logFC"] = np.abs(cis_trans_int["logFC"])


# In[14]:


order = ["control", "TSS"]
pal = {"control": "gray", "TSS": "black"}


# In[15]:


fig = plt.figure(figsize=(1, 1.5))
ax = sns.boxplot(data=cis_trans_int, x="ctrl_status", y="abs_logFC", flierprops = dict(marker='o', markersize=5), 
                 order=order, palette=pal)
mimic_r_boxplot(ax)

ax.set_xticklabels(["negative\ncontrols", "TSSs"], rotation=50, ha='right', va='top')
ax.set_xlabel("")
ax.set_ylabel("cis/trans interaction effect size")

for i, label in enumerate(order):
    n = len(cis_trans_int[cis_trans_int["ctrl_status"] == label])
    color = pal[label]
    ax.annotate(str(n), xy=(i, -0.5), xycoords="data", xytext=(0, 0), 
                textcoords="offset pixels", ha='center', va='bottom', 
                color=color, size=fontsize)

ax.set_ylim((-0.6, 4))

# calc p-vals b/w dists
dist1 = np.asarray(cis_trans_int[cis_trans_int["ctrl_status"] == "control"]["abs_logFC"])
dist2 = np.asarray(cis_trans_int[cis_trans_int["ctrl_status"] != "control"]["abs_logFC"])

dist1 = dist1[~np.isnan(dist1)]
dist2 = dist2[~np.isnan(dist2)]

u, pval = stats.mannwhitneyu(dist1, dist2, alternative="two-sided", use_continuity=False)
print(pval)

annotate_pval(ax, 0.2, 0.8, 0.77, 0, 0.7, pval, fontsize)
fig.savefig("cis_trans_int_ctrl_effectsize_boxplot.pdf", dpi="figure", bbox_inches="tight")


# ## 3. classify cis/trans interaction effects

# In[16]:


cis_trans_int["cis_trans_int_status"] = cis_trans_int.apply(cis_trans_int_status, axis=1)
cis_trans_int.cis_trans_int_status.value_counts()


# In[17]:


cis_trans_int[cis_trans_int["cis_trans_int_status"].str.contains("significant")]


# In[18]:


results[(results["hg19_id"] == "h.1305") & (results["mm9_id"] == "m.1177")][["logFC_cis_HUES64", "logFC_cis_mESC"]]


# In[19]:


results[(results["hg19_id"] == "h.2113") & (results["mm9_id"] == "m.1925")][["logFC_cis_HUES64", "logFC_cis_mESC"]]


# almost all significant examples are ones where the mouse seq. does not work in human but then does [better] in mouse -- these have a logFC > 0. why are there so few examples of the opposite?

# ## 4. merge w/ existing data

# In[20]:


cis_trans_int["hg19_id"] = cis_trans_int["index"].str.split("__", expand=True)[0]
cis_trans_int["biotype_hg19"] = cis_trans_int["index"].str.split("__", expand=True)[1]
cis_trans_int["mm9_id"] = cis_trans_int["index"].str.split("__", expand=True)[2]
cis_trans_int["biotype_mm9"] = cis_trans_int["index"].str.split("__", expand=True)[3]
cis_trans_int.head()


# In[21]:


cis_trans_int_sub = cis_trans_int[["hg19_id", "biotype_hg19", "mm9_id", "biotype_mm9", "fdr", "logFC", "abs_logFC",
                                   "cis_trans_int_status"]]
cis_trans_int_sub.columns = ["hg19_id", "biotype_hg19", "mm9_id", "biotype_mm9", "fdr_cis_trans_int", 
                             "logFC_cis_trans_int", "abs_logFC_cis_trans_int", "cis_trans_int_status"]


# In[22]:


data = results.merge(cis_trans_int_sub, 
                     left_on=["hg19_id", "biotype_hg19", 
                              "mm9_id", "biotype_mm9"], 
                     right_on=["hg19_id", "biotype_hg19", 
                               "mm9_id", "biotype_mm9"])
print(len(data))
data.sample(5)


# In[23]:


# limit to those that are significant in at least 1 context
data_filt = data[(data["HUES64_padj_hg19"] < 0.05) | (data["mESC_padj_mm9"] < 0.05)]
len(data_filt)


# In[24]:


data_filt.cis_trans_int_status.value_counts()


# ## 5. model

# In[25]:


import statsmodels.api as sm
import statsmodels.formula.api as smf

from scipy import stats
stats.chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df)


# In[26]:


full_mod = smf.ols(formula='logFC_native ~ logFC_cis_max + logFC_trans_max + logFC_cis_max : logFC_trans_max', 
                   data=data_filt).fit()
cis_mod = smf.ols(formula='logFC_native ~ logFC_cis_max', 
                  data=data_filt).fit()
trans_mod = smf.ols(formula='logFC_native ~ logFC_trans_max', 
                    data=data_filt).fit()
no_inter_mod = smf.ols(formula='logFC_native ~ logFC_cis_max + logFC_trans_max', 
                       data=data_filt).fit()


# In[27]:


# cis model alone
cis_mod.summary()


# In[28]:


# trans model alone
trans_mod.summary()


# In[29]:


# cis and trans mod together
no_inter_mod.summary()


# In[30]:


full_mod.summary()


# In[31]:


def lrtest(llmin, llmax):
    lr = 2 * (llmax - llmin)
    p = stats.chisqprob(lr, 1) # llmax has 1 dof more than llmin
    return lr, p


# In[32]:


# does adding trans info help cis model?
cisllf = cis_mod.llf
cistransllf = no_inter_mod.llf
lr, p = lrtest(cisllf, cistransllf)
print('LR test, p value: {:.2f}, {:.10f}'.format(lr, p))


# In[34]:


# does adding cis-trans info help model?
cistransllf = no_inter_mod.llf
cistransinterllf = full_mod.llf
lr, p = lrtest(cistransllf, cistransinterllf)
print('LR test, p value: {:.2f}, {:.10f}'.format(lr, p))


# In[35]:


df = full_mod.conf_int(alpha=0.05, cols=None)
df["estimate"] = list(full_mod.params)
df = df.reset_index()
df


# In[36]:


fig = plt.figure(figsize=(2, 1))

ax = sns.barplot(data=df[df["index"] != "Intercept"], y="index", x="estimate", color=sns.color_palette("Set2")[2])

ax.set_xlabel("variance explained")
ax.set_ylabel("")


# In[37]:


cis_mod.rsquared


# In[38]:


no_inter_mod.rsquared


# In[39]:


full_mod.rsquared


# In[40]:


cis_var = cis_mod.rsquared
cis_var


# In[41]:


trans_var = no_inter_mod.rsquared - cis_var
trans_var


# In[42]:


int_var = full_mod.rsquared - no_inter_mod.rsquared
int_var


# In[54]:


var_exp = {"cis effects": [cis_var], "trans effects": [trans_var], "cis-trans interactions": [int_var]}
var_exp = pd.DataFrame.from_dict(var_exp, orient="index").reset_index()

fig = plt.figure(figsize=(1.6, 1.75))
ax = sns.barplot(data=var_exp, x="index", y=0, color=sns.color_palette("Set2")[2])
ax.set_xlabel("")
ax.set_ylabel("variance explained")
ax.set_xticklabels(var_exp["index"], rotation=50, ha='right', va='top')

colors = {0: "white", 1: "black", 2: "black"}
for i, label in enumerate(var_exp["index"]):
    n = var_exp[var_exp["index"] == label][0].iloc[0]
    ax.annotate("{:.3f}".format(n), xy=(i, 0.07), xycoords="data", xytext=(0, 0), 
                textcoords="offset pixels", ha='center', va='bottom', 
                color=colors[i], size=fontsize)
    
fig.savefig("var_explained.one_by_one.pdf", dpi="figure", bbox_inches="tight")


# In[ ]:




