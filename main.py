# Fingerprint Matching in Python
import os 
import cv2

sample = cv2.imread("SOKOFingerprints/SOCOFing/Altered/Altered-Hard/49__M_Right_index_finger_Obl.BMP")

best_score = 0
Image = None
filename = None
kp1, kp2, mp = None, None, None

counter = 0
for file in os.listdir("SOKOFingerprints/SOCOFing/Real")[:1000]:
    if counter % 10 == 0:
        print(counter)
        print(file)
    counter += 1

    file_path = os.path.join("SOKOFingerprints/SOCOFing/Real", file)
    fingerprint_image = cv2.imread(file_path)

    if fingerprint_image is None:
        print(f"Error: Could not load image {file_path}")
        continue

    sift = cv2.SIFT_create()

    keypoints_one, descriptors_one = sift.detectAndCompute(sample, None)
    keypoints_two, descriptors_two = sift.detectAndCompute(fingerprint_image, None)

    if descriptors_one is None or descriptors_two is None:
        print(f"Skipping {file}: No descriptors found.")
        continue
    
    matches = cv2.FlannBasedMatcher({"algorithm": 1, "trees": 10}, {}).knnMatch(descriptors_one, descriptors_two, k=2)
    match_points = [p for p, q in matches if p.distance < 0.1 * q.distance]

    keypoints = min(len(keypoints_one), len(keypoints_two))

    if keypoints > 0:
        match_score = len(match_points) / keypoints * 100
        if match_score > best_score:
            best_score = match_score
            filename = file
            image = fingerprint_image
            kp1, kp2, mp = keypoints_one, keypoints_two, match_points

print(f"BEST MATCH: {filename}" )
print(f"SCORE: {best_score}")

if image is not None:
    result = cv2.drawMatches(sample, kp1, image, kp2, mp, None)
    result = cv2.resize(result, (0,0), fx=4, fy=4)
    cv2.imshow("Result", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No valid matches found.")