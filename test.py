import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import os

print("🚀 简单测试：生成25张动漫人脸")

# 1. 加载模型
model_path = "saved_model/generator_final.keras"
if not os.path.exists(model_path):
    print(f"❌ 找不到模型文件: {model_path}")
    print("请先训练模型")
    exit()

print(f"✅ 加载模型: {model_path}")
model = load_model(model_path)

# 2. 生成25张图片（与samples文件夹一样）
latent_dim = model.input_shape[1]  # 自动获取噪声维度
n_images = 25

print(f"📊 模型信息:")
print(f"   • 噪声维度: {latent_dim}")
print(f"   • 生成数量: {n_images}张")

# 生成噪声
noise = np.random.normal(size=(n_images, latent_dim))

# 生成图片
print("🎨 生成图片中...")
images = model.predict(noise, verbose=0)

# 转换到[0, 1]范围
images = (images + 1) / 2.0

# 3. 保存图片（与samples文件夹格式一样）
# 创建5x5网格
n = int(np.sqrt(n_images))  # n=5

fig, axes = plt.subplots(n, n, figsize=(10, 10))
axes = axes.flatten()

for img, ax in zip(images, axes):
    ax.imshow(img)
    ax.axis('off')

plt.tight_layout()

# 保存文件
output_file = "test_generated.png"
plt.savefig(output_file, dpi=100, bbox_inches='tight')
plt.close()

print(f"✅ 图片已保存: {output_file}")
print("🎉 测试完成！")