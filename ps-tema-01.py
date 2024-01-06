import numpy as np
import matplotlib.pyplot as plt
from scipy import misc, ndimage
from scipy.fft import dctn, idctn
from skimage import color, data
import cv2

X = misc.ascent()
plt.imshow(X, cmap=plt.cm.gray)
plt.show()

Y1 = dctn(X, type=1)
Y2 = dctn(X, type=2)
Y3 = dctn(X, type=3)
Y4 = dctn(X, type=4)
freq_db_1 = 20*np.log10(abs(Y1))
freq_db_2 = 20*np.log10(abs(Y2))
freq_db_3 = 20*np.log10(abs(Y3))
freq_db_4 = 20*np.log10(abs(Y4))

plt.subplot(221).imshow(freq_db_1)
plt.subplot(222).imshow(freq_db_2)
plt.subplot(223).imshow(freq_db_3)
plt.subplot(224).imshow(freq_db_4)
plt.show()

k = 120

Y_ziped = Y2.copy()
Y_ziped[k:] = 0
X_ziped = idctn(Y_ziped)

plt.imshow(X_ziped, cmap=plt.cm.gray)
plt.show()


Q_down = 10

X_jpeg = X.copy()
X_jpeg = Q_down*np.round(X_jpeg/Q_down);

plt.subplot(121).imshow(X, cmap=plt.cm.gray)
plt.title('Original')
plt.subplot(122).imshow(X_jpeg, cmap=plt.cm.gray)
plt.title('Down-sampled')
plt.show()

Q_jpeg = [[16, 11, 10, 16, 24, 40, 51, 61],
          [12, 12, 14, 19, 26, 28, 60, 55],
          [14, 13, 16, 24, 40, 57, 69, 56],
          [14, 17, 22, 29, 51, 87, 80, 62],
          [18, 22, 37, 56, 68, 109, 103, 77],
          [24, 35, 55, 64, 81, 104, 113, 92],
          [49, 64, 78, 87, 103, 121, 120, 101],
          [72, 92, 95, 98, 112, 100, 103, 99]]

# Encoding
x = X[:8, :8]
y = dctn(x)
y_jpeg = Q_jpeg*np.round(y/Q_jpeg)

# Decoding
x_jpeg = idctn(y_jpeg)

# Results
y_nnz = np.count_nonzero(y)
y_jpeg_nnz = np.count_nonzero(y_jpeg)

plt.subplot(121).imshow(x, cmap=plt.cm.gray)
plt.title('Original')
plt.subplot(122).imshow(x_jpeg, cmap=plt.cm.gray)
plt.title('JPEG')
plt.show()

print('Componente în frecvență:' + str(y_nnz) +
      '\nComponente în frecvență după cuantizare: ' + str(y_jpeg_nnz))


# Sarcini

#1. [6p] Completați algoritmul JPEG incluzând toate blocurile din imagine.

#2. [4p] Extindeți la imagini color (incluzând transformarea din RGB în Y'CbCr). Exemplificați pe `scipy.misc.face` folosită în tema anterioară.

#3. [6p] Extindeți algoritmul pentru compresia imaginii până la un prag MSE impus de utilizator.

#4. [4p] Extindeți algoritmul pentru compresie video. Demonstrați pe un clip scurt din care luați fiecare cadru și îl tratați ca pe o imagine.


# 1)
# Sarcina 1: Completați algoritmul JPEG incluzând toate blocurile din imagine.
# Pentru a procesa toate blocurile din imagine, trebuie să iterăm peste imagine în blocuri de 8x8 pixeli, aplicăm DCT, cuantizăm, și apoi aplicăm IDCT pentru reconstrucție.
# Functie pentru procesarea unui bloc de 8x8
def process_block(block, q_matrix):
    y = dctn(block, norm='ortho')
    y_quantized = np.round(y / q_matrix)
    return idctn(y_quantized * q_matrix, norm='ortho')

# Procesarea întregii imagini, bloc cu bloc
height, width = X.shape
compressed_image = np.zeros_like(X)
for i in range(0, height, 8):
    for j in range(0, width, 8):
        block = X[i:i+8, j:j+8]
        compressed_image[i:i+8, j:j+8] = process_block(block, Q_jpeg)

# Afișarea rezultatului
plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.imshow(X, cmap=plt.cm.gray)
plt.title('Original Image')
plt.subplot(122)
plt.imshow(compressed_image, cmap=plt.cm.gray)
plt.title('Compressed Image with JPEG')
plt.show()


# 2)
# Sarcina 2: Extindeți la imagini color (incluzând transformarea din RGB în Y'CbCr).
# Pentru imagini color, trebuie să convertim imaginea din RGB în formatul Y'CbCr, să procesăm fiecare canal separat și apoi să convertim înapoi în RGB.

# Generarea unei imagini color
color_image = np.random.rand(256, 256, 3)

# Conversia din RGB în Y'CbCr
ycbcr_image = color.rgb2ycbcr(color_image)

# Procesarea fiecărui canal separat
compressed_ycbcr_image = np.zeros_like(ycbcr_image)
for channel in range(3):
    for i in range(0, ycbcr_image.shape[0], 8):
        for j in range(0, ycbcr_image.shape[1], 8):
            block = ycbcr_image[i:i+8, j:j+8, channel]
            compressed_ycbcr_image[i:i+8, j:j+8, channel] = process_block(block, Q_jpeg)

# Conversia înapoi în RGB
compressed_rgb_image = color.ycbcr2rgb(compressed_ycbcr_image)

# Afișarea rezultatului
plt.figure(figsize=(15, 10))
plt.subplot(121)
plt.imshow(color_image)
plt.title('Original RGB Image')
plt.subplot(122)
plt.imshow(np.clip(compressed_rgb_image, 0, 1))  # Clip pentru a evita valorile în afara intervalului [0, 1]
plt.title('Compressed RGB Image')
plt.show()


# 3)
# Sarcina 3: Extindeți algoritmul pentru compresia imaginii până la un prag MSE impus de utilizator.
# Aici trebuie să ajustăm cuantizarea astfel încât eroarea medie pătratică (MSE) dintre imaginea originală și cea comprimată să fie sub un prag specificat.

def calculate_mse(original, compressed):
    """Calculează Mean Squared Error (MSE) între două imagini."""
    return np.mean((original - compressed) ** 2)

def adjust_quantization_for_mse(original, q_matrix, mse_threshold):
    """Ajustează matricea de cuantizare pentru a atinge un prag MSE dat."""
    height, width = original.shape
    adjusted_image = np.zeros_like(original)
    adjustment_factor = 1

    while True:
        adjusted_q_matrix = q_matrix * adjustment_factor
        for i in range(0, height, 8):
            for j in range(0, width, 8):
                block = original[i:i+8, j:j+8]
                adjusted_image[i:i+8, j:j+8] = process_block(block, adjusted_q_matrix)

        current_mse = calculate_mse(original, adjusted_image)
        if current_mse <= mse_threshold:
            break
        adjustment_factor += 0.1  # Creșterea graduală a factorului de ajustare

    return adjusted_image, adjusted_q_matrix

# Setăm un prag MSE
mse_threshold = 1000

# Ajustăm cuantizarea pentru a îndeplini pragul MSE
adjusted_image, adjusted_q_matrix = adjust_quantization_for_mse(X, Q_jpeg, mse_threshold)

# Afișarea rezultatului
plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.imshow(X, cmap=plt.cm.gray)
plt.title('Original Image')
plt.subplot(122)
plt.imshow(adjusted_image, cmap=plt.cm.gray)
plt.title('Adjusted Compressed Image')
plt.show()

# Afișarea MSE-ului final
final_mse = calculate_mse(X, adjusted_image)
print(f"Final MSE: {final_mse}")


# 4)
# Sarcina 4: Extindeți algoritmul pentru compresie video.
# Pentru compresie video, fiecare cadru din videoclip este tratat ca o imagine separată și comprimat folosind algoritmul JPEG. Aceasta necesită un set de date video și procesarea cadrelor într-o buclă.

# Calea către videoclipul .mp4
input_video_path = "d77331c0-57d7-4dbd-8200-3dcde4aeabf3.avi"
output_video_path = 'compressed.avi'

# Funcția de procesare a întregii imagini
def process_entire_image(image, q_matrix):
    height, width = image.shape
    compressed_image = np.zeros_like(image)
    for i in range(0, height, 8):
        for j in range(0, width, 8):
            block = image[i:i+8, j:j+8]
            compressed_image[i:i+8, j:j+8] = process_block(block, q_matrix)
    return compressed_image

# Deschide videoclipul original
cap = cv2.VideoCapture(input_video_path)

# Obțineți dimensiunile și FPS-ul videoclipului
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Pregătește scriitorul de videoclip pentru a salva videoclipul comprimat
fourcc = 0x58564944  # Codul numeric pentru 'XVID'
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height), isColor=False)


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convertește frame-ul în grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Aplică compresia JPEG pe frame
    compressed_frame = process_entire_image(gray_frame, Q_jpeg)

    # Scrie frame-ul comprimat în fișierul de ieșire
    out.write(compressed_frame)

# Eliberează resursele
cap.release()
out.release()