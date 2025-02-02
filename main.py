# Fingerprint Matching in Python
import os 
import cv2

sample = cv2.imread("SOKOFingerprints/SOCOFing/Altered/Altered-Hard/49__M_Right_index_finger_Obl.BMP")
# sample = cv2.resize(sample, None, fx=2.5, fy=2.5)

# cv2.imshow("Sample", sample)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

best_score = 0
Image = None
filename = None
kp1, kp1, mp = None, None, None

counter = 0
for file in [file for file in os.listdir("SOKOFingerprints/SOCOFing/Real")][:1000]:
    if counter % 10 == 0:
        print(counter)
        print(file)
    counter += 1
    fingerprint_image = cv2.imread("SOKOFingerprints/SOCOFing/Real" + file)
    sift = cv2.SIFT_create()

    keypoints_one, descriptors_one = sift.detectAndCompute(sample, None)
    keypoints_two, descriptors_two = sift.detectAndCompute(fingerprint_image, None)
    
    matches = cv2.FlannBasedMatcher({"algorithm": 1, "trees": 10}, {}).knnMatch(descriptors_one, descriptors_two, k=2)

    match_points = []

    for p, q in matches:
        if p.distance < 0.1 * q.distance:
            match_points.append(p)

    keypoints = 0
    if len(keypoints_one) < len(keypoints_two):
        keypoints = len(keypoints_one)
    else:
        keypoints = len(keypoints_two)

    if len(match_points) / keypoints * 100 > best_score:
        best_score = len(match_points) / keypoints * 100
        filename = file
        image = fingerprint_image
        kp1, kp2, mp = keypoints_one, keypoints_two, match_points

print("BEST MATCH: " + filename)
print("SCORE " +  str(best_score))

result = cv2.drawMatches(sample, kp1, image, kp2, mp, None)
result = cv2.resize(result, fx=4, fy=4)
cv2.imshow("Result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()