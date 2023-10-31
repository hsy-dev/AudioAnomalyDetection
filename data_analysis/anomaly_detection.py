import numpy as np
import pandas as pd

def detect_anomaly_info(window_indices, sensor_position, sample_rate, car_length, total_cars, window_size):
    """
    탐지된 이상 징후에 대한 상세 정보(위치, 시간, 차량 번호 등)를 계산.

    매개변수:
        window_indices (numpy.ndarray): 이상이 감지된 윈도우의 인덱스 배열.
        sensor_position (float): 센서의 위치(미터).
        sample_rate (int): 데이터의 샘플링 레이트.
        car_length (float): 각 차량의 길이(미터).
        total_cars (int): 전체 차량 수.
        window_size (int): 각 데이터 윈도우의 크기(초).

    반환:
        list: 각각의 리스트에는 이상 감지된 시간, 위치, 차량 번호, 및 윈도우 인덱스 포함.
    """
    anomaly_positions = []
    anomaly_times = []
    anomaly_cars = []
    anomaly_windows = []

    for idx in window_indices:
        # 이상치가 발생한 위치 계산
        position = sensor_position + (idx * window_size)  # 초를 미터로 변환

        # 이상치가 발생한 시간 계산
        time = idx * window_size  # 초로 변환
        # 시간을 분:초 형식으로 변환
        minutes = int(time // 60)
        seconds = int(time % 60)
        formatted_time = f"{minutes}:{seconds:02}"
        anomaly_times.append(formatted_time)

        # 이상치가 발생한 차 번호 계산
        car_number = int(position / car_length) + 1  # 몇 번째 칸인지 계산
        car_number = min(car_number, total_cars)  # 전체 차량 수를 초과하지 않도록 조정
        anomaly_cars.append(car_number)

        # 윈도우 정보 추가
        anomaly_windows.append(idx)
        
        anomaly_positions.append(position)

    return anomaly_times, anomaly_positions, anomaly_cars, anomaly_windows



def detect_and_display_anomalies(model, normalized_data, file_name_without_ext, total_duration, sensor_info):
    """
    이상 징후를 탐지하고 표시하는 함수.

    Parameters:
    model (Model): 학습된 모델.
    normalized_data (np.array): 정규화된 데이터.
    file_name_without_ext (str): 확장자를 제외한 파일 이름.
    total_duration (str): 오디오의 총 지속시간.
    sensor_info (dict): 센서 정보.
    save_directory (str): 결과를 저장할 디렉토리.
    sample_rate (int): 샘플링 레이트.

    Returns:
    None
    """


    sample_rate = 25600

    print("\n" + '-'*80)
    print("\033[1m" + "\nStep 5: 이상 징후 탐지 및 정보 계산" + "\033[0m")
    # 재구성 오차 계산 및 이상 징후 탐지
    reconstructions = model.predict(normalized_data)
    window_errors = np.mean(np.abs(normalized_data - reconstructions), axis=1)
    threshold = np.mean(window_errors) + 2 * np.std(window_errors)
    anomalous_windows = np.where(window_errors > threshold)[0]

    # 이상 징후 정보 계산
    anomaly_times, anomaly_positions, anomaly_cars, anomaly_windows = detect_anomaly_info(
        anomalous_windows,
        # sensor_info["Batcam_position"],
        23,
        sample_rate=sample_rate,
        car_length=20,
        # total_cars=sensor_info["Car_num"]
        total_cars=6,
        window_size=1)
    
    # sensor_info의 데이터 타입과 내용 출력
    # print(type(sensor_info))
    # print(sensor_info)

    # 총 시간을 각 anomaly_times에 추가
    anomaly_times_with_duration = [f"{time} / {total_duration}" for time in anomaly_times]

    # 이상 소음원 정보를 DataFrame으로 변환 및 출력
    anomaly_df = pd.DataFrame({
        'Time (anomaly/total)': anomaly_times_with_duration,
        'Position (m)': anomaly_positions,
        'Car Number': anomaly_cars,
        'Window Index': anomaly_windows
    })

    # DataFrame 출력
    print(anomaly_df)

    # 출력된 데이터프레임에서 이상 소음원에 대한 정보를 추출
    num_anomalies = len(anomaly_df)
    unique_cars = anomaly_df['Car Number'].unique()

    # 사용자에게 정보 제공
    print(f"\n분석 결과, '{file_name_without_ext}'파일에서 총 {num_anomalies}개의 이상 소음원이 감지되었습니다.")
    for car in unique_cars:
        anomalies_per_car = anomaly_df[anomaly_df['Car Number'] == car]
        print(f"차량 번호 {car}에서 {len(anomalies_per_car)}개의 이상 소음원이 발견되었습니다.")

    # 필요한 값을 반환
    return anomaly_df, anomaly_windows