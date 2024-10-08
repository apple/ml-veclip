from datasets import Dataset, load_from_disk
import json
import glob

ds = Dataset.from_dict({"path": list(glob.glob("data/wit_*.json"))})
ds.save_to_disk("tmp")
# now reload from disk and read the files
ds = load_from_disk("tmp")

def read(batch):
    urls = []
    captions = []
    for path in batch["path"]:
        with open(path) as f:
            for k, v in json.load(f).items():
                urls.append(k)
                captions.append(v)

    new_batch = {"url": urls, "caption": captions}
    
    return new_batch

ds = ds.map(read, batched=True, batch_size=1, remove_columns=["path"])

print(ds)

ds.push_to_hub("nielsr/wit_300m")