import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

def plot_audio(file_name, channel_name, data, sample_rate=25600):
    """
    오디오 데이터를 시각화합니다. 이 함수는 시간 도메인, 주파수 도메인, 스펙트로그램을 표시합니다.

    매개변수:
        file_name (str): 시각화에 사용될 파일의 이름.
        channel_name (str): 시각화에 사용될 채널의 이름.
        data (numpy.ndarray): 시각화할 오디오 데이터.
        sample_rate (int): 오디오의 샘플링 레이트. 기본값: 25600.
    """
    plt.figure(figsize=(20, 5))
    
    # 그래프 상단에 해당 파일 이름과 채널 이름을 전체 제목으로 표시
    plt.suptitle(f"{file_name} - {channel_name}", fontsize=30)

    # 시간 도메인 그래프
    plt.subplot(1, 3, 1)
    plt.plot(np.arange(len(data)) / sample_rate, data)
    plt.title("Time Domain")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")

    # 주파수 도메인 그래프 (FFT)
    plt.subplot(1, 3, 2)
    frequencies = np.fft.rfftfreq(len(data), d=1/sample_rate)
    fft_values = np.abs(np.fft.rfft(data))
    plt.plot(frequencies, fft_values)
    plt.title("Frequency Domain")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")

    # 스펙트로그램
    plt.subplot(1, 3, 3)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(data)), ref=np.max)
    librosa.display.specshow(D, sr=sample_rate, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.xlabel("Time (seconds)")
    plt.ylabel("Frequency (Hz)")

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)  # 전체 제목과 그래프 사이의 간격 조정
    plt.show()


def window_data(data, sample_rate=25600, window_size=1):
    """
    주어진 데이터를 윈도우 단위로 분할. 이를 통해 각 윈도우에서의 이상 감지 분석 가능.

    매개변수:
        data (numpy.ndarray): 분할할 원본 데이터.
        sample_rate (int): 데이터의 샘플링 레이트. 기본값: 25600.
        window_size (int): 각 데이터 윈도우의 크기(초). 기본값: 1.

    반환:
        numpy.ndarray: 윈도우로 분할된 데이터.
    """
    window_length = sample_rate * window_size
    num_windows = len(data) // window_length
    return np.array([data[i * window_length:(i + 1) * window_length] for i in range(num_windows)])

