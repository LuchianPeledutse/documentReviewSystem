import pickle
import pathlib

retriever_metrics_folder = pathlib.Path(".")
label_names = [item.name for item in retriever_metrics_folder.iterdir() if item.is_dir()]

label_dict = {name: idx for idx, name in enumerate(label_names)}

with open("label_dict.pkl", "wb") as label_pkl_file:
    pickle.dump(label_dict, label_pkl_file)

