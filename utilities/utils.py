import os

def ensure_directory_exists(dir_path):
    """
    주어진 경로에 디렉토리가 없으면 생성하는 함수.

    매개변수:
        dir_path (str): 확인할 디렉토리 경로.
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def format_duration(seconds):
    """
    초 단위의 시간을 'HH:MM:SS' 형식의 문자열로 변환하는 함수.

    매개변수:
        seconds (int): 변환할 시간(초).
    
    반환:
        str: 'HH:MM:SS' 형식의 시간 문자열.
    """
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"
