import numpy as np
import matplotlib.pyplot as plt
import random
# 군집화 (Clustering)를 위해 scikit-learn 라이브러리에서 DBSCAN을 불러옵니다.
from sklearn.cluster import DBSCAN

# 군집의 중심을 계산하기 위해 NumPy를 사용합니다.

# --- 시뮬레이션 설정 ---
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

# --- 탐색 및 군집화 설정 ---
# 1. 브루트 포스 임계값: 핫스팟 후보를 찾을 최소 온도 (이전과 동일)
HOTSPOT_THRESHOLD = 0.7

# 2. DBSCAN 군집화 매개변수
# eps (epsilon): 한 점으로부터 이 거리(픽셀) 이내에 있는 점들을 이웃으로 간주합니다.
#               핫스팟의 퍼짐 정도(scale)에 따라 조정해야 합니다.
DBSCAN_EPS = 15
# min_samples: 중심점(Core point)이 되기 위해 필요한 최소 이웃 수.
DBSCAN_MIN_SAMPLES = 10

# --- 1. 시뮬레이션 데이터 생성 (Coldspot) ---
data = np.clip(np.random.normal(COLDSPOT_TEMPERATURE, COLDSPOT_STD, (ROWS, COLS)), HOTSPOT_MIN_TEMPERATURE,
               HOTSPOT_MAX_TEMPERATURE * 0.3)

# --- 2. 시뮬레이션 데이터 생성 (Hotspot) ---
actual_hotspots = []
for _ in range(NUMBER_OF_HOTSPOTS):
    centerY = random.randint(0, ROWS - 1)
    centerX = random.randint(0, COLS - 1)
    spreadScale = random.uniform(HOTSPOT_MIN_SCALE, HOTSPOT_MAX_SCALE)

    actual_hotspots.append({
        'y': centerY,
        'x': centerX,
        'scale': spreadScale
    })

# --- 3. 온도 기여도 설정 ---
for hotspot in actual_hotspots:
    centerY = hotspot['y']
    centerX = hotspot['x']
    spreadScale = hotspot['scale']

    for y in range(ROWS):
        for x in range(COLS):
            distance = (x - centerX) ** 2 + (y - centerY) ** 2
            temperatureContribution = np.exp(-distance / (2 * spreadScale ** 2))
            data[y, x] = np.clip(data[y, x] + temperatureContribution * (HOTSPOT_MAX_TEMPERATURE - data[y, x]),
                                 HOTSPOT_MIN_TEMPERATURE, HOTSPOT_MAX_TEMPERATURE)


# --- 4. 브루트 포스 핫스팟 탐색 ---
def find_hotspots_bruteforce(data, threshold):
    rows, cols = data.shape
    hotspot_candidates = []
    for y in range(rows):
        for x in range(cols):
            if data[y, x] >= threshold:
                # [x, y] 순서로 저장 (좌표계 관례에 따름)
                hotspot_candidates.append([x, y])
    return np.array(hotspot_candidates)


# 브루트 포스 탐색 실행
hotspot_coordinates = find_hotspots_bruteforce(data, HOTSPOT_THRESHOLD)

# --- 5. DBSCAN 군집화 실행 ---
if len(hotspot_coordinates) > 0:
    # DBSCAN 모델 생성 및 학습
    db = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES).fit(hotspot_coordinates)

    # 각 데이터 포인트가 속한 군집 레이블 (-1은 노이즈)
    labels = db.labels_

    # 군집화된 핫스팟 중심점 계산
    final_hotspots = []

    # 고유한 군집 레이블 추출 (노이즈 -1 제외)
    unique_labels = set(labels)
    unique_labels.discard(-1)  # 노이즈 레이블 제거

    for k in unique_labels:
        # 현재 군집(k)에 속하는 모든 포인트 추출
        class_member_mask = (labels == k)
        cluster_points = hotspot_coordinates[class_member_mask]

        # 군집의 중심(Centroid) 계산 (x와 y 좌표의 평균)
        center_x = int(np.mean(cluster_points[:, 0]))
        center_y = int(np.mean(cluster_points[:, 1]))

        final_hotspots.append({'x': center_x, 'y': center_y, 'cluster_size': len(cluster_points)})
else:
    final_hotspots = []
    print("Warning: No hotspot candidates found above the threshold.")

# --- 6. 시각화 (탐색된 핫스팟 중심 표시) ---

# 원본 데이터 시각화
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

# A. 브루트 포스 탐색된 핫스팟 후보 위치 (배경)
if len(hotspot_coordinates) > 0:
    ax.scatter(
        hotspot_coordinates[:, 0],  # x 좌표
        hotspot_coordinates[:, 1],  # y 좌표
        s=5,
        marker='.',
        color='lightcoral',
        alpha=0.3,
        label=f'Brute Force Candidates ({len(hotspot_coordinates)} points)'
    )

# B. DBSCAN으로 찾은 최종 핫스팟 중심 위치 (파란색 다이아몬드)
final_hotspotX = [h['x'] for h in final_hotspots]
final_hotspotY = [h['y'] for h in final_hotspots]

ax.scatter(
    final_hotspotX,
    final_hotspotY,
    s=250,
    marker='D',  # Diamond
    color='blue',
    edgecolor='black',
    linewidths=1.5,
    label=f'Clustered Hotspot Center ({len(final_hotspots)} found)'
)

# C. 실제 핫스팟 중심 위치 (비교를 위해 노란색 X로 표시)
actual_hotspotX = [h['x'] for h in actual_hotspots]
actual_hotspotY = [h['y'] for h in actual_hotspots]

ax.scatter(
    actual_hotspotX,
    actual_hotspotY,
    s=300,
    marker='X',
    color='yellow',
    edgecolor='black',
    linewidths=1.5,
    label='Actual Hotspot Center (Simulation Input)'
)

ax.legend()
plt.show()
