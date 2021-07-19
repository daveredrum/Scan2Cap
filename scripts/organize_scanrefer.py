import os
import sys
import json

sys.path.append(os.path.join(os.getcwd())) # HACK add the root folder
from lib.config import CONF

SCANREFER = json.load(open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered.json")))

organized = {}
for data in SCANREFER:
    scene_id = data["scene_id"]
    object_id = data["object_id"]
    ann_id = data["ann_id"]

    # store
    if scene_id not in organized:
        organized[scene_id] = {}

    if object_id not in organized[scene_id]:
        organized[scene_id][object_id] = {}

    if ann_id not in organized[scene_id][object_id]:
        organized[scene_id][object_id][ann_id] = None

    organized[scene_id][object_id][ann_id] = data

with open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_organized.json"), "w") as f:
    json.dump(organized, f, indent=4)
    