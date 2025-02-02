# Fingerprint Matching in Python
import os 
import cv2

sample = cv2.imread("SOKOFingerprints/SOCOFing/Altered/Altered-Hard/49__M_Right_index_finger_Obl.BMP")
sample = cv2.resize(sample, None, fx=2.5, fy=2.5)

cv2.imshow("Sample", sample)
cv2.waitKey(0)
cv2.destroyAllWindows()