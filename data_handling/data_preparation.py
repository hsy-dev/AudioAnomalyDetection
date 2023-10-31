import os
import json
import numpy as np
from nptdms import TdmsFile
from .file_loaders import find_corresponding_json, get_representative_channel
from data_analysis.audio_analysis import plot_audio, window_data


def load_and_prepare_data(tdms_file_path, base_directory, sample_rate):
    """
    데이터 로딩 및 준비과정을 처리하는 함수.

    Parameters:
    tdms_file_path (str): 분석할 TDMS 파일의 경로.
    base_directory (str): 기본 디렉토리 경로.
    sample_rate (int): 샘플링 레이트.

    Returns:
    tuple: 정규화된 데이터, 파일명(확장자 제외), 총 지속시간, 센서 정보
    """
    # JSON 파일 찾기 및 로딩
    print("\033[1m" + "Step 1: 입력된 TDMS 파일과 매칭되는 JSON 파일 탐색" + "\033[0m")
    json_file_path, matched_file_name = find_corresponding_json(tdms_file_path, base_directory)

    if json_file_path is None:
        print(f"No corresponding JSON file found for {tdms_file_path}. Please check the file path and try again.")
        return None  # JSON 파일이 없는 경우, 함수를 종료함.
    else:
        print(f"Match found: {matched_file_name}")  # 일치하는 파일 이름 출력

    with open(json_file_path, 'r', encoding='utf-8') as json_file:
        sensor_info = json.load(json_file)

    # tdms 파일 로드
    tdms_file = TdmsFile(tdms_file_path)

    # 로드된 파일에서 파일 이름만 추출
    file_name = os.path.basename(tdms_file_path)

    # 파일 확장자 제거
    file_name_without_ext = os.path.splitext(file_name)[0]

    print("\n" + '-'*80)
    print("\033[1m" + "\nStep 2: 대표 채널 선택 및 오디오 시각화" + "\033[0m")
    # 대표 채널 추출 및 오디오 데이터 시각화
    data, channel_name = get_representative_channel(tdms_file)
    total_duration_seconds = len(data) / sample_rate  # 샘플링 레이트로 나누어 총 시간(초) 계산
    total_duration = f"{int(total_duration_seconds // 60)}:{int(total_duration_seconds % 60):02}"  # 분:초 형식으로 변환

    plot_audio(file_name_without_ext, channel_name, data)

    # 데이터를 윈도우 형태로 변경 및 정규화
    windowed_data = window_data(data)
    min_val = np.min(windowed_data)
    max_val = np.max(windowed_data)
    normalized_data = (windowed_data - min_val) / (max_val - min_val)

    # input_dim 계산
    input_dim = normalized_data.shape[1] if len(normalized_data.shape) > 1 else normalized_data.shape[0]

    return normalized_data, input_dim, file_name_without_ext, total_duration, sensor_info