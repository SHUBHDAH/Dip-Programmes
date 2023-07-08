from numpy import asarray
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
img = cv.imread('j.jpg', cv.IMREAD_GRAYSCALE)
assert img is not None, "file could not be read, check with os.path.exists()"
f = np.fft.fft2(img)
domaindata = asarray(img)
gaussdata = asarray(f)

# plt.scatter(img[:, 0], img[:, 1], c=img[:, 2], cmap='hot')
# plt.show()
# plt.scatter(f[:, 0], f[:, 1], c=f[:, 2], cmap='hot')
# plt.show()

print("Normal image data",domaindata)
print("fourier transform data",gaussdata)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20*np.log(np.abs(fshift))
plt.subplot(121),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()

plt.scatter(img[:, 0], img[:, 1], c=img[:, 2], cmap='hot')
plt.show()
plt.scatter(f[:, 0], f[:, 1], c=f[:, 2], cmap='hot')
plt.show()
