import matplotlib.pyplot as plt
import pandas as pd

def MakePrefixSumAverageList(l):
    global lenOfList

    prefixSum = 0
    prefixSumAverageList = []
    for i in range(lenOfList):
        prefixSum += float(l['기온'][i])
        prefixSumAverageList.append(prefixSum/(i+1))

    return prefixSumAverageList

def LabelTemperature(l):
    global lenOfList

    for i in range(lenOfList):
        plt.text(timeList[i], l[i] + 0.025, f'{round(l[i], 2)}℃',
                 ha='center', va='bottom', fontsize=9, color='black')

buildingTemperatureFile = pd.read_csv('', encoding='EUC-KR')
parkTemperatureFile = pd.read_csv('', encoding='EUC-KR')

timeList = []
lenOfList = len(buildingTemperatureFile['측정 시각'])
buildingTemperaturePrefixSum = MakePrefixSumAverageList(buildingTemperatureFile)
parkTemperaturePrefixSum = MakePrefixSumAverageList(parkTemperatureFile)

plt.rc('font', family='Malgun Gothic', size=10)

for i in range(lenOfList):
    if int(buildingTemperatureFile["측정 시각"][i][3:].replace(':', '')) < int(parkTemperatureFile["측정 시각"][i][3:].replace(':', '')) or i==0: # 오후 12시는 제외함
        timeList.append(f'{buildingTemperatureFile["측정 시각"][i]}~{parkTemperatureFile["측정 시각"][i]}')
    else:
        timeList.append(f'{parkTemperatureFile["측정 시각"][i]}~{buildingTemperatureFile["측정 시각"][i]}')

plt.plot(timeList, buildingTemperaturePrefixSum, label='건물 밀집 지역', marker='o', c='magenta')
plt.plot(timeList, parkTemperaturePrefixSum, label='공원 지역', marker='o', c='yellow')

plt.title('건물 밀집 지역과 공원의 평균 기온 비교')
plt.xlabel('측정 시각')
plt.ylabel('기온 (°C)')
plt.grid(True)
plt.legend()

LabelTemperature(buildingTemperaturePrefixSum)
LabelTemperature(parkTemperaturePrefixSum)

plt.show()
