![Alt text](image.png)
## 🏆 2023 DATA·AI 분석 경진대회
### 대회개요
1. 대회일정 : 2023. 05.30(화) ~ 10. 27(금) / 시상식 11.30(목) * 일정은 상황에 따라 변경 될 수 있습니다.
2. 대회목적 : DATA 및 AI 기반의 과학·사회적 문제 해결 및 DNA 저변 확대
3. 세부부문 :
- 과학기술 문제형 : 연구데이터를 활용한 과학기술 현안 문제 해결
- 사회현안 문제형 : 인공지능 데이터를 활용한 지역 및 사회현안 문제해결
4. 단      계 :
- 문제발굴 : 연구데이터 또는 인공지능 데이터 기반의 해결이 필요한 과학·사회적 문제를 발굴하고 관련 데이터를 공개하여 다양한 연구자의 참여 유도
- 문제해결 : 데이터 기반으로 주어진 문제에 최적 성능을 보이는 데이터 분석 또는 AI 모델 개발
5. 주      최 :과학기술정보통신부, 국가과학기술연구회, 대전광역시, 국회도서관
6. 주      관 : 한국과학기술정보연구원(KISTI)
7. 후      원 : (사)한국콘텐츠학회
<br/>

## 🚂 기차 대차의 이상 소음원 위치 판별
### 문제 설명
#### 문제 개요
* 문제 정의 : 이동하는 기차 대차의 이상 음향 발생 위치 판별로 사고 위험 예방
* 목적 및 배경 : 국민의 생활에서 발생하는 다양한 비언어적 소리를 데이터로 축적하여 공공 교통안전 현안 문제를 해결하기 위해 소리데이터 융합·활용하여 사회문제 해결 필요

#### 최종 성과물
* 실시간으로 이동하는 기차 결함의 위치를 소리데이터로 활용하여 이상 유/무를 판별하여 위험 상황을 단계별로 인지 할 수 있도록 알람 표시

### 데이터 설명
#### 데이터 정보
* 데이터명 : 철도 소음데이터
* 유형 : 오디오/이미지
* 포맷 : tdms, json
* 용량 : 87.3GB
* 건수 : 2대의 서로 다른 전동차 (수소전동차 : 64건, 차세대전동차 : 118건)
* 문제해결을 위해 필요한 설명 : 철도 고장 발생 시 열차의 탈선, 전복 등의 사고로 이어질 수 있고, 필연적으로 대규모의 인명 피해를 동반한다. 철도 고장 및 고장의 전조증상은 음향 또는 초음파로 나타나는 경우가 많으므로 이에 대한 데이터를 구축하면 정상 및 비정상을 판별할 수 있는 인공지능 알고리즘을 개발할 수 있다. 이 알고리즘을 통해 고장 발생을 사전에 탐지하여 대처함으로써 대형 사고를 미연에 방지할 수 있다.
* ※ 수집 장비 : 초음파, 음향 카메라
#### 데이터 샘플
```
{
	“title_s206”: “test_01.tdms”
	“title_batcam2”: “test_1.tdms”
	“Creator”: “SM Ins.”
	“Year”: “2022”
	“Date”: “1103”
	“Train”: “전동차”
		“Length”: 122
		“Car_num”: 6
	“Horn”: “Yes”
		“Position”: 62
	“Place”: “시험선로”
	“Photo_sensor_positions”: [0, 33.2]
	“S206_position”: 22
	“Batcam_position”: 23
}
```

## 📚 모델 구조
AudioAnomalyDetection/<br>
│<br>
├── data_handling/<br>
│   ├── \_\_init\_\_.py<br>
│   ├── file_loaders.py      # TDMS 및 JSON 파일 로딩 함수들<br>
│   └── data_savers.py       # 이상 소음 데이터를 저장하는 함수들<br>
│<br>
├── data_analysis/<br>
│   ├── \_\_init\_\_.py<br>
│   ├── audio_analysis.py    # 오디오 데이터 분석 및 시각화 함수들<br>
│   └── anomaly_detection.py # 이상 징후 탐지 및 정보 계산 함수들<br>
│<br>
├── machine_learning/<br>
│   ├── \_\_init\_\_.py<br>
│   ├── model.py             # Autoencoder 모델 생성 및 컴파일 함수들<br>
│   └── training.py          # 모델 학습 및 학습 과정 시각화 함수들<br>
│<br>
├── utilities/<br>
│   ├── \_\_init\_\_.py<br>
│   └── utils.py             # 범용적 유틸리티 함수들 (시간 형식 변환, 파일 경로 생성 등)<br>
│<br>
├── main.py                  # 메인 실행 로직. 다른 모듈들을 사용하여 전체 프로세스를 조정<br>
│<br>
├── constants.py             # 프로젝트 전체에 걸쳐 사용되는 상수들 (예: 샘플 레이트)<br>
│<br>
├── requirements.txt         # 프로젝트에 필요한 Python 패키지 목록<br>
│<br>
└── README.md                # 프로젝트 설명, 설정 방법, 사용 방법 등에 대한 문서<br>

## 추가중