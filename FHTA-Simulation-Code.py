import numpy as np
import matplotlib.pyplot as plt
import random

ROWS = 150
COLS = 90

NUMBER_OF_HOTSPOTS = 4

# feature of hotspot
HOTSPOT_MIN_TEMPERATURE = 0.0
HOTSPOT_MAX_TEMPERATURE = 1.0
HOTSPOT_MIN_SCALE = 3
HOTSPOT_MAX_SCALE = 12

# feature of coldspot
COLDSPOT_TEMPERATURE = 0.1
COLDSPOT_STD = 0.05

# first, create coldspot
data = np.clip(np.random.normal(COLDSPOT_TEMPERATURE, COLDSPOT_STD, (ROWS, COLS)), HOTSPOT_MIN_TEMPERATURE, HOTSPOT_MAX_TEMPERATURE * 0.3)
initialData = data.copy()

# second, create hotspot and heat around area
hotspots = []
for _ in range(NUMBER_OF_HOTSPOTS):
    centerY = random.randint(0, ROWS - 1)
    centerX = random.randint(0, COLS - 1)
    spreadScale = random.uniform(HOTSPOT_MIN_SCALE, HOTSPOT_MAX_SCALE)

    hotspots.append({
        'y': centerY,
        'x': centerX,
        'scale': spreadScale
    })

# third, set temperature contribution
for hotspot in hotspots:
    centerY = hotspot['y']
    centerX = hotspot['x']
    spreadScale = hotspot['scale']

    for y in range(ROWS):
        for x in range(COLS):
            distance = (x - centerX) ** 2 + (y - centerY) ** 2
            temperatureContribution = np.exp(-distance / (2 * spreadScale ** 2))
            data[y, x] = np.clip(data[y, x] + temperatureContribution * (HOTSPOT_MAX_TEMPERATURE - data[y, x]), HOTSPOT_MIN_TEMPERATURE, HOTSPOT_MAX_TEMPERATURE)

basicFig, basicAx = plt.subplots(figsize=(8, 10))
basicCax = basicAx.imshow(
    data,
    cmap='plasma',
    aspect='auto',
    interpolation='bicubic',
    vmin=HOTSPOT_MIN_TEMPERATURE,
    vmax=HOTSPOT_MAX_TEMPERATURE
)
cbar = basicFig.colorbar(basicCax, orientation='vertical')
basicAx.set_title('heatmap simulation', fontsize=14)
basicAx.set_xticks(np.arange(0, COLS, 10))
basicAx.set_yticks(np.arange(0, ROWS, 15))

# get hotspot center position
hotspotX = [h['x'] for h in hotspots]
hotspotY = [h['y'] for h in hotspots]

modifiedFig, modifiedAx = plt.subplots(figsize=(8, 10))
modifiedCax = modifiedAx.imshow(
    initialData,
    cmap='plasma',
    aspect='auto',
    interpolation='bicubic',
    vmin=HOTSPOT_MIN_TEMPERATURE,
    vmax=HOTSPOT_MAX_TEMPERATURE
)
modifiedAx.scatter(
    hotspotX,
    hotspotY,
    s=300,
    marker='X',
    color='red',
    edgecolor='black',
    linewidths=1.5,
    label='Hotspot Center'
)
cbar2 = modifiedFig.colorbar(modifiedCax, orientation='vertical')
modifiedAx.set_title('modified heatmap simulation', fontsize=14)
modifiedAx.set_xticks(np.arange(0, COLS, 10))
modifiedAx.set_yticks(np.arange(0, ROWS, 15))
modifiedAx.legend()

plt.show()
