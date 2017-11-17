from keras.preprocessing.image import img_to_array, load_img, array_to_img
from keras.models import load_model
import scipy as sp
import numpy as np
from scipy import signal
from scipy import fftpack



model = load_model('../weights/adadelta_default_es_model.h5')
model.load_weights('../weights/adadelta_default_es.h5')

# im = load_img('/home/cuervo/thesis/data/pic_set/test/scn_6/image_800.jpg', grayscale=True)

AF = open('/home/cuervo/thesis/data/final_pu/no_dc/scn_4_snr_15.dat', 'rb')

# spectrogram(...) returns also the frequency bins and the times:
# f, t, Sxx = signal.spectrogram(...)
# but we won't use them\
for i in range(64):
    data = sp.fromfile(AF, dtype=sp.complex64, count=7700)
    _, _, Sxx = signal.spectrogram(data,
                                   fs=10e6,
                                   mode='magnitude',
                                   return_onesided=False,
                                   nperseg=64,
                                   detrend=False,
                                   noverlap=0)
# The spectrum will be reversed, so we shift it
    Sxx = sp.fftpack.fftshift(Sxx, axes=0)
    Sxx = 20 * np.log10(Sxx)
    avgd = np.average(Sxx, axis=1)
    if i == 0:
        stacked = np.array(avgd)
    else:
        stacked = np.vstack([stacked, avgd])

from scipy.misc import imsave, toimage
imsave("test.jpg", stacked)

# print("stacked= ", stacked)
# stacked = np.expand_dims(stacked, axis=2)
# image = array_to_img(stacked, scale=False)
image = toimage(stacked, channel_axis=2)
sample = img_to_array(image)
# print("image= ", sample)
# print("sample= ",sample)
sample = np.expand_dims(sample, axis=0)


im = load_img('test.jpg', grayscale=True)
x = img_to_array(im)
# print("X= ",x)
x = np.expand_dims(x, axis=0)

print(model.predict(x))
print(np.argmax(model.predict(x)))

print(sample.shape)
print(model.predict(sample))
print(np.argmax(model.predict(sample)))

# model.save('../weights/adadelta_default_es_model.h5')
