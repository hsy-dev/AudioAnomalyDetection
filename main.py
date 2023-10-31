import os
from utilities import provide_user_feedback
from data_handling.file_loaders import (
    load_tdms_file, find_corresponding_json, get_representative_channel, get_total_duration
)
from data_handling.data_preparation import prepare_data
from model_training.train_and_evaluate_model import train_and_evaluate_model
from anomaly_detection.detect_and_display_anomalies import detect_and_display_anomalies
from data_handling.data_savers import save_results

def main(tdms_file_path, base_directory, save_directory, sample_rate=25600):
    # 데이터 로드 및 준비
    tdms_file = load_tdms_file(tdms_file_path)
    json_file_path = find_corresponding_json(tdms_file_path, base_directory)
    data, channel_name = get_representative_channel(tdms_file)
    total_duration = get_total_duration(tdms_file, sample_rate)

    # 데이터 전처리
    normalized_data, input_dim = prepare_data(data)

    # 모델 훈련 및 평가
    autoencoder, history = train_and_evaluate_model(normalized_data, input_dim)

    # 이상 징후 탐지 및 정보 제공
    anomalies, anomaly_df = detect_and_display_anomalies(autoencoder, normalized_data, total_duration)

    # 결과 저장 및 사용자 피드백 제공
    saved_files = save_results(normalized_data, anomalies, save_directory, sample_rate)
    provide_user_feedback(anomaly_df, saved_files, save_directory)

if __name__ == "__main__":
    base_directory = "C:/_Programming/Project/Data"  # 'json'이 포함되지 않은 경로
    tdms_file_path = base_directory + "/tdms/221108_차세대전동차/BATCAM2/test_08.tdms"
    save_directory = "output"
    main(tdms_file_path, base_directory, save_directory)

