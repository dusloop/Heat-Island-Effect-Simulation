import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.cluster import DBSCAN

# 1) setting random heatmap simulation
ROWS = 150
COLS = 90

NUMBER_OF_HOTSPOTS = 4

# feature of hotspot
HOTSPOT_MIN_TEMPERATURE = 0.0
HOTSPOT_MAX_TEMPERATURE = 1.0
HOTSPOT_MIN_SCALE = 3
HOTSPOT_MAX_SCALE = 12

# feature of position which is not hotspot
MAP_AVERAGE_TEMPERATURE = 0.1
MAP_STD = 0.05

# setting clustering
HOTSPOT_THRESHOLD = 0.7

# 2) setting DBSCAN clustering parameter
# eps (epsilon): regard pixels in particular distance as neighbor, adapt depend on scale
DBSCAN_EPS = 15
# min_samples: min number of neighbor points  to be the center.
DBSCAN_MIN_SAMPLES = 10

# 3) create random heatmap
# np.clip: constrain values, it consists of array, a_min, a_max
# array: list that wil be restricted
# np.random.normal: consist of loc, scale, size
# loc: average of normal distribution
# scale: standard deviation which makes normal distribution more dilated
# size: (n, m) -> 2D array n*m
data = np.clip(np.random.normal(MAP_AVERAGE_TEMPERATURE, MAP_STD, (ROWS, COLS)), HOTSPOT_MIN_TEMPERATURE,
               HOTSPOT_MAX_TEMPERATURE * 0.3)

# create simulation data
actualHotspot = []
for _ in range(NUMBER_OF_HOTSPOTS):
    centerY = random.randint(0, ROWS - 1)
    centerX = random.randint(0, COLS - 1)
    spreadScale = random.uniform(HOTSPOT_MIN_SCALE, HOTSPOT_MAX_SCALE)

    actualHotspot.append({
        'y': centerY,
        'x': centerX,
        'scale': spreadScale
    })

# set contribution of temp
for hotspot in actualHotspot:
    centerY = hotspot['y']
    centerX = hotspot['x']
    spreadScale = hotspot['scale']

    for y in range(ROWS):
        for x in range(COLS):
            distance = (x - centerX) ** 2 + (y - centerY) ** 2
            # exp: e ** p
            tempContribution = np.exp(-distance / (2 * spreadScale ** 2))
            data[y, x] = np.clip(data[y, x] + tempContribution,
                                 HOTSPOT_MIN_TEMPERATURE, HOTSPOT_MAX_TEMPERATURE)


# 4) search hotspot with brute force
def findHotspots(data, threshold):
    # nparray.shape -> check dimension
    rows, cols = data.shape
    hotspotCandidates = []
    for y in range(rows):
        for x in range(cols):
            if data[y, x] >= threshold:
                hotspotCandidates.append([x, y])
    return np.array(hotspotCandidates)

# operate brute force
hotspotCandidates = findHotspots(data, HOTSPOT_THRESHOLD)

# 5) operate DBSCAN(Density-Based Spatial Clustering of Applications with Noise) clustering, 
#    check density of coordinates
# 1. get hotspot label that is categorized with number
# 2. get hotspot label of each cluster
# 3. check center of each cluster
if len(hotspotCandidates) > 0:
    # create DBSCAN model and start learning by using fit
    db = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES).fit(hotspotCandidates)

    # noise: -1, not be clustered
    labels = db.labels_

    clusteredHotspotCenter = []

    # abstract unique clustered labels
    uniqueLabels = set(labels)
    uniqueLabels.discard(-1)  # remove noise

    for k in uniqueLabels:
        # abstract all clustered points, categorize each cluster
        classMemberMask = (labels == k)
        clusterPoints = hotspotCandidates[classMemberMask]

        # calculate center of clustered hotspot by averaging x, y
        center_x = int(np.mean(clusterPoints[:, 0]))
        center_y = int(np.mean(clusterPoints[:, 1]))

        clusteredHotspotCenter.append({'x': center_x, 'y': center_y, 'cluster_size': len(clusterPoints)})
else:
    clusteredHotspotCenter = []
    print("Warning: No hotspot candidates found above the threshold.")

# 6) visualize
# 0. create basic map
fig, ax = plt.subplots(figsize=(8, 10))
cax = ax.imshow(
    data,
    cmap='plasma',
    aspect='auto',
    interpolation='bicubic',
    vmin=HOTSPOT_MIN_TEMPERATURE,
    vmax=HOTSPOT_MAX_TEMPERATURE
)
fig.colorbar(cax, orientation='vertical')
ax.set_title(f'Heatmap Simulation with DBSCAN Clustering', fontsize=14)
ax.set_xticks(np.arange(0, COLS, 10))
ax.set_yticks(np.arange(0, ROWS, 15))

# 1. scatter hotspot candidates
if len(hotspotCandidates) > 0:
    ax.scatter(
        hotspotCandidates[:, 0],  # x
        hotspotCandidates[:, 1],  # y
        s=5,
        marker='.',
        color='lightcoral',
        alpha=0.3,
        label=f'Brute Force Candidates ({len(hotspotCandidates)} points)'
    )

# 2. express hotspot center which is calculated by DBSCAN
final_hotspotX = [h['x'] for h in clusteredHotspotCenter]
final_hotspotY = [h['y'] for h in clusteredHotspotCenter]

ax.scatter(
    final_hotspotX,
    final_hotspotY,
    s=250,
    marker='D',  # Diamond
    color='blue',
    edgecolor='black',
    linewidths=1.5,
    label=f'Clustered Hotspot Center ({len(clusteredHotspotCenter)} found)'
)

# 3. scatter actual hotspot center
actualHotspotX = [h['x'] for h in actualHotspot]
actualHotspotY = [h['y'] for h in actualHotspot]

ax.scatter(
    actualHotspotX,
    actualHotspotY,
    s=300,
    marker='X',
    color='yellow',
    edgecolor='black',
    linewidths=1.5,
    label='Actual Hotspot Center'
)

ax.legend()
plt.show()
