# Fingerprint Matching in Python
import os 
import cv2

sample = cv2.imread("SOKOFingerprints/SOCOFing/Altered/Altered-Hard/570__M_Left_ring_finger_Obl.BMP")

if sample is None:
    print("Error: Could not load sample image.")
    exit()

sample = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
sample = cv2.resize(sample, (400, 400))
sample = cv2.equalizeHist(sample)

best_score = 0
best_image = None
best_filename = None
kp1, kp2, mp = None, None, None

total_precision = 0
total_recall = 0
match_count = 0

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
    matches = sorted(matches, key=lambda x:x[0].distance)
    strong_matches = matches[:int(len(matches) * 0.7)]
    match_points = [p for p, q in strong_matches if p.distance < 0.6 * q.distance]

    keypoints = min(len(keypoints_one), len(keypoints_two))

    # precision and recall analysis
    if len(strong_matches) > 0 and keypoints > 0:
        precision = len(match_points) / len(strong_matches)
        recall = len(match_points) / keypoints
        total_precision += precision
        total_recall += recall
        match_count += 1

        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}")

    

    if keypoints > 0:
        match_score = len(match_points) / keypoints * 100
        if match_score > best_score:
            best_score = match_score
            best_filename = file
            best_image = fingerprint_image
            kp1, kp2, mp = keypoints_one, keypoints_two, match_points

if match_count > 0:
    avg_precision = total_precision / match_count
    avg_recall = total_recall / match_count
    print(f"\n🔥 **Final Metrics Across All Matches:** 🔥")
    print(f"✅ **Average Precision:** {avg_precision:.4f}")
    print(f"✅ **Average Recall:** {avg_recall:.4f}")

if best_filename:
    print(f"BEST MATCH: {best_filename}" )
    print(f"SCORE: {best_score}")

    result = cv2.drawMatches(sample, kp1, best_image, kp2, mp, None)
    result = cv2.resize(result, None, fx=4, fy=4)
    cv2.imshow("Result", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No valid matches found.")