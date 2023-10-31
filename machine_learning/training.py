# training.py

import tensorflow as tf
import matplotlib.pyplot as plt
from .model import Autoencoder

class PrintEpoch(tf.keras.callbacks.Callback):
    """
    에폭 종료 시 로깅 정보를 출력하는 콜백 클래스.

    상속:
        tensorflow.keras.callbacks.Callback

    메서드:
        on_epoch_end(epoch, logs=None): 에폭이 끝날 때 호출되며, 로깅 정보를 출력.
    """
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % 5 == 0:  # 에폭이 5의 배수일 때
            print(f"Epoch {epoch + 1}/{self.params['epochs']} - "
                  f"loss: {logs['loss']:.4f} - val_loss: {logs['val_loss']:.4f}")

def train_model(model, data, epochs=50, batch_size=256, validation_split=0.1):
    """
    주어진 모델을 데이터로 학습시키는 함수.

    매개변수:
        model (Model): 학습시킬 모델.
        data (np.ndarray): 학습 데이터.
        epochs (int): 에포크 수.
        batch_size (int): 배치 크기.
        validation_split (float): 검증 데이터의 비율.

    반환:
        History: 학습 과정 데이터.
    """
    model.compile(optimizer='adam', loss='binary_crossentropy')
    
    history = model.fit(data, data,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_split=validation_split,
                        verbose=0,  # 에폭 로깅을 위한 콜백을 사용함.
                        callbacks=[PrintEpoch()])
    return history


def train_and_evaluate_model(normalized_data):
    """
    모델을 학습시키고 평가하는 함수.

    Parameters:
    normalized_data (np.array): 정규화된 데이터.

    Returns:
    tuple: 학습된 모델, 학습 히스토리
    """
    # 모델 생성 및 컴파일
    input_dim = normalized_data.shape[1]  # 윈도우 크기에 따른 입력 차원
    autoencoder = Autoencoder(input_dim)

    print("\n" + '-'*80)
    print("\033[1m" + "\nStep 3:모델 학습 진행 상황" + "\033[0m")
    # 모델 학습
    history = train_model(autoencoder, normalized_data, epochs=30, batch_size=512, validation_split=0.1)

    print("\n" + '-'*80)
    print("\033[1m" + "\nStep 4: 모델 학습 과정 시각화" + "\033[0m")
    # 모델 학습 과정 시각화
    plt.figure(figsize=(14, 5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.show()

    return autoencoder, history