# data_savers.py
import os
from scipy.io import wavfile

def save_results(windowed_data, anomalies, save_directory, sample_rate=25600):
    """
    이상 소음 데이터를 .wav 파일로 저장.

    매개변수:
        windowed_data (numpy.ndarray): 원본 윈도우 분할 데이터.
        anomalies (list): 이상이 감지된 윈도우의 인덱스 목록.
        save_directory (str): 결과를 저장할 디렉토리 경로.
        sample_rate (int): 오디오의 샘플링 레이트. 기본값: 25600.

    반환:
        list: 저장된 .wav 파일들의 경로 리스트.
    """
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    
    saved_files = []  # 저장된 파일 경로를 저장할 리스트
    for idx, anomaly_index in enumerate(anomalies):
        anomaly_window = windowed_data[anomaly_index]  # 넘파이 배열에서 특정 인덱스의 데이터 가져오기
        file_name = f"{save_directory}/anomaly_window_{anomaly_index}_idx_{idx}.wav"
        wavfile.write(file_name, sample_rate, anomaly_window)
        saved_files.append(file_name)  # 파일 경로를 리스트에 추가
    
    return saved_files  # 저장된 파일 경로의 리스트 반환


def provide_user_feedback(saved_files, save_directory):
    """
    사용자에게 저장된 파일에 대한 피드백을 제공.

    매개변수:
        saved_files (list): 저장된 파일들의 경로 리스트.
        save_directory (str): 파일이 저장된 디렉토리.
    """
    print("\n" + '-'*80)
    print("\033[1m" + "\nStep 6: 이상 소음 데이터 파일 저장" + "\033[0m")
    print(f"파일이 '{save_directory}'에 성공적으로 저장되었습니다.")
    print("저장된 파일:")
    for file_path in saved_files:
        print(file_path)
