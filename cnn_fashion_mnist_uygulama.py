
"""
ÖDEV: Derin Öğrenme ve Evrişimsel Sinir Ağlarının (CNN) Temel Yapısının İncelenmesi
Hazırlayan: MOHAMMAD ALSAYEH
Öğrenci No: 02220224654
"""

import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Rastgelelik kontrolü
tf.keras.utils.set_random_seed(42)
np.random.seed(42)

print("="*65)
print("3. BÖLÜM: KONVOLÜSYON İŞLEMİNİN KODLANMASI")
print("="*65)

conv = tf.keras.layers.Conv2D(
    filters=16,
    kernel_size=(3, 3),
    activation='relu',
    kernel_initializer='he_normal'
)

dummy_input = tf.random.normal((1, 128, 128, 1))
output = conv(dummy_input)
weights, biases = conv.get_weights()

print("Giriş boyutu:", dummy_input.shape)
print("Çıkış boyutu:", output.shape)
print("Filtre ağırlıklarının boyutu:", weights.shape)
print("Bias boyutu:", biases.shape)
print("İlk filtre:\n", weights[:, :, 0, 0])

# 16 Filtrenin Başlangıç Ağırlıklarını Görselleştirme
fig, axes = plt.subplots(4, 4, figsize=(7, 7))
fig.suptitle("Conv2D 16 Adet 3x3 Filtrenin Başlangıç Ağırlıkları (he_normal)", fontweight="bold")
vmin, vmax = weights.min(), weights.max()
for i, ax in enumerate(axes.flat):
    f_mat = weights[:, :, 0, i]
    im = ax.imshow(f_mat, cmap="RdBu_r", vmin=vmin, vmax=vmax)
    ax.set_title(f"Filtre {i+1}", fontsize=9)
    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0, 1, 2])
    for r in range(3):
        for c in range(3):
            ax.text(c, r, f"{f_mat[r, c]:.2f}", ha="center", va="center", fontsize=7)
plt.tight_layout()
plt.savefig("bolum3_konvolusyon_filtreleri.png", dpi=200)
plt.close()

print("\n" + "="*65)
print("4. BÖLÜM: POOLING İŞLEMİNİN İNCELENMESİ")
print("="*65)

input_data = tf.constant([
    [[1., 3., 2., 4.],
     [5., 6., 1., 2.],
     [7., 2., 8., 3.],
     [4., 1., 5., 9.]]
])
input_data = tf.reshape(input_data, (1, 4, 4, 1))
pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2)
pool_output = pool(input_data)

print("Giriş:\n", input_data.numpy()[0, :, :, 0])
print("Pooling sonucu:\n", pool_output.numpy()[0, :, :, 0])

print("\n" + "="*65)
print("5. BÖLÜM: ReLU VE SOFTMAX FONKSİYONLARININ İNCELENMESİ")
print("="*65)

x = tf.constant([-5., -2., -1., 0., 2., 4., 6.])
relu = tf.keras.layers.ReLU()
y = relu(x)
print("ReLU Giriş:", x.numpy())
print("ReLU Çıkışı:", y.numpy())

logits = tf.constant([[2.0, 1.0, 0.1]])
softmax = tf.keras.layers.Softmax()
probabilities = softmax(logits)
print("Softmax sonuçları:", probabilities.numpy())
print("Olasılıkların toplamı:", tf.reduce_sum(probabilities).numpy())

print("\n" + "="*65)
print("6. & 8. BÖLÜM: FASHION-MNIST CNN EĞİTİMİ VE FİLTRE DENEYİ")
print("="*65)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
x_train = (x_train.astype("float32") / 255.0)[..., np.newaxis]
x_test  = (x_test.astype("float32") / 255.0)[..., np.newaxis]

print("Eğitim verisi:", x_train.shape)
print("Test verisi:", x_test.shape)

def build_and_train_cnn(first_filters):
    tf.keras.utils.set_random_seed(42)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(first_filters, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()
    start_t = time.time()
    history = model.fit(
        x_train, y_train,
        epochs=5,
        batch_size=64,
        validation_split=0.1,
        verbose=1
    )
    elapsed = time.time() - start_t
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    train_acc = history.history['accuracy'][-1]
    return history, train_acc, test_acc, test_loss, elapsed

print("\n--- DENEY 1: İlk Katman 8 Filtre ---")
hist_8, tr_acc_8, te_acc_8, te_loss_8, time_8 = build_and_train_cnn(8)

print("\n--- DENEY 2: İlk Katman 16 Filtre ---")
hist_16, tr_acc_16, te_acc_16, te_loss_16, time_16 = build_and_train_cnn(16)

# Eğitim sonuçlarının görselleştirilmesi
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(hist_8.history['accuracy'], label='Training (8 Filtre)')
plt.plot(hist_8.history['val_accuracy'], label='Validation (8 Filtre)')
plt.plot(hist_16.history['accuracy'], '--', label='Training (16 Filtre)')
plt.plot(hist_16.history['val_accuracy'], '--', label='Validation (16 Filtre)')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(hist_8.history['loss'], label='Training (8 Filtre)')
plt.plot(hist_8.history['val_loss'], label='Validation (8 Filtre)')
plt.plot(hist_16.history['loss'], '--', label='Training (16 Filtre)')
plt.plot(hist_16.history['val_loss'], '--', label='Validation (16 Filtre)')
plt.title('Loss')
plt.xlabel('Epoch')
plt.legend()

plt.tight_layout()
plt.savefig("bolum6_egitim_grafikleri.png", dpi=200)
plt.show()

print("\n" + "="*65)
print("DENEY KARŞILAŞTIRMA TABLOSU")
print("="*65)
print(f"{'Ölçüt':<25} | {'Deney 1 (8 Filtre)':<20} | {'Deney 2 (16 Filtre)':<20}")
print("-" * 70)
print(f"{'Eğitim Doğruluğu':<25} | {tr_acc_8:<20.4f} | {tr_acc_16:<20.4f}")
print(f"{'Test Doğruluğu':<25} | {te_acc_8:<20.4f} | {te_acc_16:<20.4f}")
print(f"{'Eğitim Süresi (sn)':<25} | {time_8:<20.2f} | {time_16:<20.2f}")
