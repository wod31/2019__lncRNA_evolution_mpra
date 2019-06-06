
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd


# ## label pre-sets

# ## label functions

# In[3]:


def better_type(row):
    if row["tile_type"] == "WILDTYPE":
        if "HUMAN" in row["element_id"]:
            if "PROMOTER" in row["element_id"]:
                return "WT human xist/firre/tug1 promoter"
            elif "EVO_TSS" in row["element_id"]:
                return "WT other human tss"
        elif "MOUSE" in row["element_id"]:
            if "PROMOTER" in row["element_id"]:
                return "WT mouse xist/firre/tug1 promoter"
            elif "EVO_TSS" in row["element_id"]:
                return "WT other mouse tss"
    elif row["tile_type"] == "FLIPPED":
        if "HUMAN" in row["element_id"]:
            if "PROMOTER" in row["element_id"]:
                return "WT flipped human xist/firre/tug1 promoter"
            elif "EVO_TSS" in row["element_id"]:
                return "WT flipped other human tss"
        elif "MOUSE" in row["element_id"]:
            if "PROMOTER" in row["element_id"]:
                return "WT flipped mouse xist/firre/tug1 promoter"
            elif "EVO_TSS" in row["element_id"]:
                return "WT flipped other mouse tss"
    elif "DELETION" in row["tile_type"]:
        if "HUMAN" in row["element_id"]:
            return "human deletion"
        elif "MOUSE" in row["element_id"]:
            return "mouse deletion"
    elif row["tile_type"] == "CONTROL":
        return "control"
    elif row["tile_type"] == "RANDOM":
        return "random"
    else:
        return "?"


# ## short pandas functions

# In[4]:


def get_item(row, d, key_col):
    try:
        return d[row[key_col]]
    except:
        return "no pvalue calculated"


# In[ ]:


def active_in_only_one(row):
    if row["combined_class"].count("sig active") == 1:
        return True
    else:
        return False
    
def active_in_only_two(row):
    if row["combined_class"].count("sig active") == 2:
        return True
    else:
        return False

def active_in_only_three(row):
    if row["combined_class"].count("sig active") == 3:
        return True
    else:
        return False


# In[ ]:


def get_cage_id(row):
    if row.oligo_type != "RANDOM":
        cage_id = row.seq_name.split("__")[1].split(",")[0]
    else:
        cage_id = "none"
    return cage_id


# ## other utils

# In[1]:


def calculate_tissue_specificity(df):
    array = df.as_matrix()
    array_max = np.max(array, axis=1)
    tmp = array.T / array_max
    tmp = 1 - tmp.T
    specificities = np.sum(tmp, axis=1)/(array.shape[1])
    return specificities


# In[2]:


def scale_range(data, minTo, maxTo):
    """
    function to scale data linearly to a new min/max value set
    
    parameters
    ----------
    data: array like, list of numbers
    minTo: float, minimum of new range desired
    maxTo: float, maximum of new range desired
    
    returns
    -------
    scaled_data: array like, new list of numbers (appropriately scaled)
    """
    minFrom = np.nanmin(data)
    maxFrom = np.nanmax(data)
    
    scaled_data = []
    
    for point in data:
        new_point = minTo + (maxTo - minTo) * ((point - minFrom)/(maxFrom - minFrom))
        scaled_data.append(new_point)
    
    return scaled_data


# In[ ]:




