from keras.models import load_model
from cv2 import cv2
import numpy as np

#IMAGE_FILE = './corrosion/2 (1).jpg'
IMAGE_FILE = './test.png'
model = load_model('model-009.model')

rust_clsfr = cv2.CascadeClassifier('rust.xml')

labels_dict = {0: 'errosion corrosion', 1: 'galvanic corrosion',
               2: 'normal non-corrosion', 3: 'pitting corrosion', 4: 'uniform corrosion'}
color_dict = {0: (0, 255, 0), 1: (0, 0, 255)}

img = cv2.imread(IMAGE_FILE)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#rust = rust_clsfr.detectMultiScale(gray, 1.3, 5)
rust = rust_clsfr.detectMultiScale(
    gray, scaleFactor=1.3, minNeighbors=5, minSize=(75, 75))

for (i, (x, y, w, h)) in enumerate(rust):
    rust_img = gray[y:y+w, x:x+w]
    resized = cv2.resize(rust_img, (100, 100))
    normalized = resized/255.0
    reshaped = np.reshape(normalized, (1, 100, 100, 1))
    result = model.predict(reshaped)
    label = np.argmax(result, axis=1)[0]
    # Surround cascade with rectangle
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.putText(img, labels_dict[label], (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

if label == 0:
    print('=====================================================================')
    print('')
    print('Kondisi Pipa : KOROSI EROSI')
    print('')
    print('Silahkan ikuti instruksi dibawah :')
    print('')
    print('1. Kurangi aliran fluida apabila terlalu deras.')
    print('2. Beri lapisan inhibitor untuk pipa.')
    print('3. Beri lapisan pelindung dari zat agresif.')
    print('4. Jika kondisi telah parah, ganti logam dengan logam yang lebih homogen.')
    print('')
    print('=====================================================================')
elif label == 1:
    print('=====================================================================')
    print('')
    print('Kondisi Pipa : KOROSI GALVANIS')
    print('')
    print('Silahkan ikuti instruksi dibawah :')
    print('')
    print('1. Beri isolator yang lebih tebal untuk menghilangkan aliran elektrolit.')
    print('2. Tambahkan inhibitor anti-korosi pada fluida pipa.')
    print('2. Jika Kondisi telah parah, tambahkan proteksi katodik.')
    print('')
    print('=====================================================================')
elif label == 2:
    print('=====================================================================')
    print('')
    print('Kondisi Pipa : NORMAL')
    print('')
    print('Silahkan ikuti instruksi dibawah :')
    print('')
    print('Lakukan perawatan berkala pada pipa anda untuk mencegah terjadinya korosi.')
    print('')
    print('=====================================================================')
elif label == 3:
    print('=====================================================================')
    print('')
    print('Jenis Korosi : KOROSI SUMUR')
    print('')
    print('Silahkan ikuti instruksi dibawah :')
    print('')
    print('1. Beri lapisan inhibitor untuk pipa.')
    print('2. Beri lapisan pelindung dari zat agresif.')
    print('3. Jika kondisi telah parah, ganti logam dengan logam yang lebih homogen.')
    print('')
    print('=====================================================================')
else:
    print('=====================================================================')
    print('')
    print('Kondisi Pipa : KOROSI SERAGAM')
    print('')
    print('Silahkan ikuti instruksi dibawah :')
    print('')
    print('1. Lapisi dengan pelapis dengan campuran inhibitor.')
    print('2. Jika kondisi telah parah, ganti logam dengan paduan tembaga 0,4%.')
    print('')
    print('=====================================================================')

# Display the cascade to the user
cv2.imshow(labels_dict[label] + "s", img)
cv2.waitKey(0)
#cv2.imwrite('output_image.jpg', img)
