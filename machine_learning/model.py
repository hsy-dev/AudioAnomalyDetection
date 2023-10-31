# model.py

import tensorflow as tf
from tensorflow.keras import layers, Model

class Autoencoder(Model):
    """
    주어진 입력 차원에 대해 인코더의 마지막 차원을 1/512로 줄이는 오토인코더 모델.

    상속:
        tensorflow.keras.Model

    매개변수:
        input_dim (int): 입력 데이터의 차원.

    메서드:
        call(x): 모델의 전방향 패스를 수행.
    """
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        
        # Encoder
        self.encoder = tf.keras.Sequential([
            # 인코더의 레이어 구성
            layers.Dense(input_dim // 128, activation="relu"),  # 줄인 차원: 원래 데이터의 1/128
            layers.Dense(input_dim // 256, activation="relu"),
            layers.Dense(input_dim // 512, activation="relu")
        ])

        # Decoder
        self.decoder = tf.keras.Sequential([
            layers.Dense(input_dim // 256, activation="relu"),
            layers.Dense(input_dim // 128, activation="relu"),
            layers.Dense(input_dim, activation='sigmoid')
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
