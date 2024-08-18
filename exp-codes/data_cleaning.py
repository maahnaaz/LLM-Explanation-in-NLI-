import torch
import pandas as pd
import json

from jsonformer import Jsonformer
from transformers import AutoModelForCausalLM, AutoTokenizer
# Converting the json files to csv
df_list = []
for i in range(n):
    path = f"../expthenlabel03.json"
    with open(path, "r") as file:
        data = file.read()
        df = (pd.DataFrame((eval(data.replace("}{", "},{")))) )
    df_list.append( pd.DataFrame((eval(data.replace("}{", "},{")))) )
df = pd.concat(df_list, ignore_index=True)

