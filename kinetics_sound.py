# %%
from pathlib import Path
import json

# %%
# https://arxiv.org/pdf/1705.08168.pdf Appendix A1
kinetics_sound = "blowing nose, bowling, chopping wood, ripping paper, shuffling cards, singing, tapping pen, typing, blowing out, dribbling ball, laughing, mowing the lawn by pushing lawnmower, shoveling snow, stomping, tap dancing, tapping guitar, tickling, fingerpicking, patting, playing accordion, playing bagpipes, playing bass guitar, playing clarinet, playing drums, playing guitar, playing harmonica, playing keyboard, playing organ, playing piano, playing saxophone, playing trombone, playing trumpet, playing violin, playing xylophone"
kinetics_sound = kinetics_sound.split(", ")
assert len(kinetics_sound) == 34

# %%
k700 = Path("/media/mark/sol/kinetics-dataset/k700-2020")
traindir = k700 / "train"
valdir = k700 / "val"
testdir = k700 / "test"
categories = list(sorted([d.name for d in traindir.iterdir() if d.is_dir()]))
print(len(categories))
print(categories[:10])

# %%
overlaps = 0
found_categories = []
for c in categories:
    if c not in kinetics_sound:
        continue
    print(c)
    found_categories.append(c)
    overlaps += 1
print(overlaps)

# %%
missing = set(kinetics_sound).difference(found_categories)
for c in missing:
    print(c)
"""
patting
mowing the lawn by pushing lawnmower
fingerpicking
dribbling ball
blowing out
typing
stomping
"""

# %%
for c in categories:
    # keyword = "mowing"
    # keyword = "blowing"
    keyword = "pat"
    if keyword in c:
        print(c)

# %%
k700_renamed_categories = "mowing lawn, blowing leaves, dribbling basketball".split(
    ", ")
selected_categories = found_categories + k700_renamed_categories
print(len(selected_categories))
# %%
selected_categories
# yapf: disable
[ 'blowing nose', 'bowling', 'chopping wood', 'laughing', 'playing accordion', 'playing bagpipes', 'playing bass guitar', 'playing clarinet', 'playing drums', 'playing guitar', 'playing harmonica', 'playing keyboard', 'playing organ', 'playing piano', 'playing saxophone', 'playing trombone', 'playing trumpet', 'playing violin', 'playing xylophone', 'ripping paper', 'shoveling snow', 'shuffling cards', 'singing', 'tap dancing', 'tapping guitar', 'tapping pen', 'tickling', 'mowing lawn', 'blowing leaves', 'dribbling basketball'
]
# yapf: enable

# %%
num_videos = {}
data_subset = dict(train=dict(), val=dict())


def video_paths(split: Path, category: str):
    return list(
        sorted(str(v) for v in (split / category).iterdir() if v.suffix == ".mp4"))


for c in selected_categories:
    train_vids = video_paths(traindir, c)
    val_vids = video_paths(valdir, c)
    # test_vids = videos(testdir, c)
    num_videos[c] = dict(train=len(train_vids), val=len(val_vids))
    data_subset["train"][c] = train_vids
    data_subset["val"][c] = val_vids
num_videos

# %%
kinetics_subset = Path.cwd() / "kinetics_subset.json"
assert not kinetics_subset.exists(), f"file {kinetics_subset} exists"
kinetics_subset.write_text(json.dumps(data_subset, indent=2))

# %%
# yapf: disable
{'blowing nose': {'train': 741, 'val': 48},
 'bowling': {'train': 964, 'val': 49},
 'chopping wood': {'train': 971, 'val': 48},
 'laughing': {'train': 959, 'val': 49},
 'playing accordion': {'train': 979, 'val': 50},
 'playing bagpipes': {'train': 986, 'val': 49},
 'playing bass guitar': {'train': 739, 'val': 49},
 'playing clarinet': {'train': 810, 'val': 50},
 'playing drums': {'train': 977, 'val': 49},
 'playing guitar': {'train': 871, 'val': 49},
 'playing harmonica': {'train': 833, 'val': 48},
 'playing keyboard': {'train': 953, 'val': 50},
 'playing organ': {'train': 979, 'val': 49},
 'playing piano': {'train': 960, 'val': 48},
 'playing saxophone': {'train': 957, 'val': 49},
 'playing trombone': {'train': 912, 'val': 50},
 'playing trumpet': {'train': 955, 'val': 44},
 'playing violin': {'train': 921, 'val': 50},
 'playing xylophone': {'train': 981, 'val': 48},
 'ripping paper': {'train': 859, 'val': 46},
 'shoveling snow': {'train': 975, 'val': 49},
 'shuffling cards': {'train': 946, 'val': 49},
 'singing': {'train': 778, 'val': 49},
 'tap dancing': {'train': 970, 'val': 49},
 'tapping guitar': {'train': 971, 'val': 50},
 'tapping pen': {'train': 948, 'val': 47},
 'tickling': {'train': 779, 'val': 50},
 'mowing lawn': {'train': 976, 'val': 49},
 'blowing leaves': {'train': 722, 'val': 50},
 'dribbling basketball': {'train': 971, 'val': 49}}
# yapf: enable
# %%
