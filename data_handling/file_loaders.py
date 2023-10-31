# file_loaders.py
import os
import json
import datetime
import numpy as np
from nptdms import TdmsFile
from sklearn.decomposition import PCA


def load_tdms_file(tdms_file_path):
    """
    주어진 경로에서 TDMS 파일을 로드.
    
    매개변수:
        tdms_file_path (str): 로드할 TDMS 파일의 경로.
        
    반환:
        nptdms.TdmsFile: 로드된 TDMS 파일 객체.
    """
    if not os.path.exists(tdms_file_path):
        raise FileNotFoundError(f"The specified TDMS file does not exist: {tdms_file_path}")
    return TdmsFile.read(tdms_file_path)


def get_representative_channel(tdms_file):
    """
    TDMS 파일에서 주성분 분석(PCA)을 통해 가장 대표적인 채널 데이터를 선택하는 함수.

    매개변수:
        tdms_file (nptdms.TdmsFile): 분석할 TDMS 파일.

    반환:
        tuple: 선택된 대표 채널의 데이터(numpy.ndarray)와 이름(str).
    """
    channels = tdms_file["RawData"].channels()
    selected_channels = [channel for channel in channels if channel.name.startswith("Channel")]
    data_matrix = [channel.data for channel in selected_channels]
    data_matrix = np.array(data_matrix).T
    pca = PCA(n_components=1)
    principal_components = pca.fit_transform(data_matrix)
    sample_with_highest_score_index = np.argmax(principal_components)
    representative_channel_index = np.argmax(data_matrix[sample_with_highest_score_index, :])
    
    # 대표 채널의 데이터와 이름을 반환
    return selected_channels[representative_channel_index].data, selected_channels[representative_channel_index].name



def find_corresponding_json(tdms_file_path, base_directory):
    """
    주어진 TDMS 파일 경로에 대응하는 JSON 파일을 검색.

    매개변수:
        tdms_file_path (str): TDMS 파일의 경로.
        base_directory (str): 검색을 시작할 기본 디렉토리의 경로.

    반환:
        str 또는 None: 대응하는 JSON 파일의 경로 또는 파일을 찾지 못한 경우 None.
    """
    tdms_directory = os.path.join(base_directory, "tdms")
    json_directory = os.path.join(base_directory, "json")
    
    # TDMS 파일 이름 추출
    tdms_file_name = os.path.basename(tdms_file_path)
    # 확장자 없는 파일 이름
    file_name_without_ext = os.path.splitext(tdms_file_name)[0]

    # JSON 디렉토리 내의 모든 폴더를 순회하며 대응하는 JSON 파일 검색
    for folder in os.listdir(tdms_directory):
        if not folder.endswith("차세대전동차"):
            continue

        tdms_folder_path = os.path.join(tdms_directory, folder, "BATCAM2")
        json_folder_path = os.path.join(json_directory, folder)
        
        for file in os.listdir(json_folder_path):
            if file.endswith(".json"):
                with open(os.path.join(json_folder_path, file), 'r', encoding='utf-8') as json_file:
                    data = json.load(json_file)
                    title_batcam2 = data.get("title_batcam2")
                    
                    # JSON 파일의 title_batcam2가 TDMS 파일 이름과 일치하는지 확인
                    if title_batcam2:
                        title_without_ext = os.path.splitext(title_batcam2)[0]  # 확장자 제거
                        if title_without_ext == file_name_without_ext:
                            matched_file_path = os.path.join(json_folder_path, file)
                            return matched_file_path, file  # 경로와 파일 이름을 반환
    
    # 대응하는 JSON 파일을 찾지 못한 경우
    return None, None


def get_total_duration(tdms_file, sample_rate=25600):
    """
    주어진 TDMS 파일의 총 재생 시간을 계산.

    매개변수:
        tdms_file (nptdms.TdmsFile): 총 재생 시간을 계산할 TDMS 파일.
        sample_rate (int): 샘플 레이트. 기본값: 25600.

    반환:
        str: 'HH:MM:SS' 형식의 총 재생 시간.
    """
    # 대표 채널에서 데이터 길이 가져오기
    _, channel_name = get_representative_channel(tdms_file)
    channel_data_length = len(tdms_file["RawData"][channel_name].data)

    # 총 시간(초) 계산
    total_seconds = channel_data_length / sample_rate

    # 초를 'HH:MM:SS' 형식으로 변환
    formatted_time = str(datetime.timedelta(seconds=total_seconds))

    return formatted_time
