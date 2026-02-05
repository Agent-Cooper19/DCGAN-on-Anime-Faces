import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=0'

import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import time
import json

# 图像参数
IMG_SIZE = 64
CHANNELS = 3
LATENT_DIM = 100  # 减小潜在维度加速训练


def create_folders():
    for folder in ["saved_model", "samples", "logs"]:
        os.makedirs(folder, exist_ok=True)


def load_and_preprocess_image(path):
    """简化版数据加载"""
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = (img - 127.5) / 127.5  # 归一化到[-1, 1]
    return img


def create_generator():
    """简化生成器"""
    model = models.Sequential([
        layers.Dense(8 * 8 * 256, use_bias=False, input_shape=(LATENT_DIM,)),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),

        layers.Reshape((8, 8, 256)),

        layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),

        layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),

        layers.Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),

        layers.Conv2DTranspose(CHANNELS, (5, 5), strides=(1, 1), padding='same', use_bias=False, activation='tanh'),
    ])
    return model


def create_discriminator():
    """简化判别器"""
    model = models.Sequential([
        layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[IMG_SIZE, IMG_SIZE, CHANNELS]),
        layers.LeakyReLU(alpha=0.2),
        layers.Dropout(0.3),

        layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'),
        layers.LeakyReLU(alpha=0.2),
        layers.Dropout(0.3),

        layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same'),
        layers.LeakyReLU(alpha=0.2),
        layers.Dropout(0.3),

        layers.Flatten(),
        layers.Dense(1),
    ])
    return model


def train_simple_gan():
    """简化的训练循环"""
    create_folders()

    # 参数 - 针对大数据集优化
    BATCH_SIZE = 64  # 减小batch_size
    EPOCHS = 40
    SAMPLE_INTERVAL = 5

    # 加载图片路径
    image_paths = glob("data/*.jpg") + glob("data/*.jpeg") + glob("data/*.png")
    print(f"找到 {len(image_paths):,} 张训练图片")

    if len(image_paths) == 0:
        print("错误: data/文件夹中没有图片!")
        return

    # 创建数据集
    print("创建数据集中...")
    dataset = tf.data.Dataset.from_tensor_slices(image_paths)
    dataset = dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(10000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    print("数据集创建完成")

    # 创建模型
    print("构建模型中...")
    generator = create_generator()
    discriminator = create_discriminator()

    # 优化器
    generator_optimizer = optimizers.Adam(0.0002, 0.5)
    discriminator_optimizer = optimizers.Adam(0.0002, 0.5)

    # 损失函数
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    # 训练历史
    history = {'d_loss': [], 'g_loss': []}

    print(f"\n开始训练 {EPOCHS} 轮...")
    print(f"批大小: {BATCH_SIZE}, 每 {SAMPLE_INTERVAL} 轮保存样本")
    print("=" * 60)

    start_time = time.time()

    for epoch in range(EPOCHS):
        epoch_start = time.time()
        epoch_d_loss = []
        epoch_g_loss = []

        # 训练每个batch
        for batch_idx, real_images in enumerate(dataset):
            batch_size = tf.shape(real_images)[0]

            # 训练判别器
            noise = tf.random.normal([batch_size, LATENT_DIM])
            with tf.GradientTape() as disc_tape:
                generated_images = generator(noise, training=True)

                real_output = discriminator(real_images, training=True)
                fake_output = discriminator(generated_images, training=True)

                # 标签平滑
                real_labels = tf.ones_like(real_output) * 0.9
                fake_labels = tf.zeros_like(fake_output) + 0.1

                real_loss = cross_entropy(real_labels, real_output)
                fake_loss = cross_entropy(fake_labels, fake_output)
                disc_loss = (real_loss + fake_loss) / 2

            disc_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
            discriminator_optimizer.apply_gradients(zip(disc_gradients, discriminator.trainable_variables))

            # 训练生成器
            noise = tf.random.normal([batch_size, LATENT_DIM])
            with tf.GradientTape() as gen_tape:
                generated_images = generator(noise, training=True)
                fake_output = discriminator(generated_images, training=True)
                gen_labels = tf.ones_like(fake_output) * 0.9
                gen_loss = cross_entropy(gen_labels, fake_output)

            gen_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
            generator_optimizer.apply_gradients(zip(gen_gradients, generator.trainable_variables))

            epoch_d_loss.append(disc_loss.numpy())
            epoch_g_loss.append(gen_loss.numpy())

            # 每10个batch显示进度
            if batch_idx % 50 == 0:
                print(f"  Epoch {epoch + 1:2d}, Batch {batch_idx:4d}: "
                      f"D={np.mean(epoch_d_loss[-10:]):.4f}, G={np.mean(epoch_g_loss[-10:]):.4f}")

        # 记录损失
        avg_d_loss = np.mean(epoch_d_loss)
        avg_g_loss = np.mean(epoch_g_loss)
        history['d_loss'].append(avg_d_loss)
        history['g_loss'].append(avg_g_loss)

        # 显示epoch进度
        epoch_time = time.time() - epoch_start
        elapsed = time.time() - start_time
        remaining = (EPOCHS - epoch - 1) * (elapsed / (epoch + 1))

        h = int(remaining // 3600)
        m = int((remaining % 3600) // 60)
        s = int(remaining % 60)

        print(f"Epoch {epoch + 1:2d}/{EPOCHS} | "
              f"D: {avg_d_loss:.4f} | G: {avg_g_loss:.4f} | "
              f"Time: {epoch_time:.1f}s | ETA: {h:02d}:{m:02d}:{s:02d}")

        # 保存样
        if (epoch + 1) % SAMPLE_INTERVAL == 0:
            noise = np.random.normal(size=(25, LATENT_DIM))
            generated_images = generator.predict(noise, verbose=0)
            generated_images = (generated_images + 1) / 2.0

            fig, axes = plt.subplots(5, 5, figsize=(10, 10))
            for img, ax in zip(generated_images, axes.flatten()):
                ax.imshow(img)
                ax.axis('off')

            plt.savefig(f"samples/generated_epoch_{epoch + 1:04d}.png", dpi=100, bbox_inches='tight')
            plt.close()
            print(f"  ↳ 保存样本: samples/generated_epoch_{epoch + 1:04d}.png")

        # 保存检查点
        if (epoch + 1) % 10 == 0:
            generator.save(f"saved_model/generator_epoch_{epoch + 1}.keras")

    # 保存最终模型
    generator.save("saved_model/generator_final.keras")
    discriminator.save("saved_model/discriminator_final.keras")

    # 保存训练历史
    with open("logs/training_history.json", "w") as f:
        json.dump({
            'd_loss': history['d_loss'],
            'g_loss': history['g_loss'],
            'parameters': {
                'batch_size': BATCH_SIZE,
                'latent_dim': LATENT_DIM,
                'epochs': EPOCHS,
                'num_images': len(image_paths)
            }
        }, f, indent=2)

    print(f"\n训练完成! 总时间: {(time.time() - start_time) / 60:.1f}分钟")
    print(f"模型保存在: saved_model/")
    print(f"样本保存在: samples/")


if __name__ == "__main__":
    train_simple_gan()