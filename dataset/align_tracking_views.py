# process based on Satya Mallicks implementation of Feature based image alignment
# https://www.learnopencv.com/image-alignment-feature-based-using-opencv-c-python/

import cv2
import numpy as np

MAX_FEATURES = 100000
GOOD_MATCHES_PERCENT = 0.21


def alignImages(im1, im2):
    # Convert images to grayscale
    im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    # Detect ORB features and compute descriptors.
    orb = cv2.ORB_create(MAX_FEATURES)
    keypoints1, descriptros1 = orb.detectAndCompute(im1Gray, None)
    keypoints2, descriptros2 = orb.detectAndCompute(im2Gray, None)

    # Match features.
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptros1, descriptros2, None)

    # Sort matches by score
    #matches.sort(key=lambda x: x.distance, reverse=False)
    matches = sorted(matches, key=lambda x: x.distance)

    # Remove sub-par matches
    numGoodMatches = int(len(matches) * GOOD_MATCHES_PERCENT)
    numTopMatches = int((len(matches) * GOOD_MATCHES_PERCENT) / 10)
    matches_filtered = matches[:numGoodMatches]

    # Draw top 1% of matches
    imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches[:numTopMatches], None)
    cv2.imwrite("matches.jpg", imMatches)

    # Extract location of good matches
    points1 = np.zeros((len(matches_filtered), 2), dtype=np.float32)
    points2 = np.zeros((len(matches_filtered), 2), dtype=np.float32)

    for i, match in enumerate(matches_filtered):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    # Use homography
    height, width, channels = im2.shape
    im1Reg = cv2.warpPerspective(im1, h, (width, height))

    # Write aligned image to disk.
    outFilename = "aligned_test.jpg"
    print("[INFO]      Saving aligned image  :  ", outFilename)
    cv2.imwrite(outFilename, im1Reg)

    return im1Reg, h


if __name__ == '__main__':
    # Read reference image
    refFilename = "top_down_view.PNG"
    print("[INFO]      Reading reference image : ", refFilename)
    imReference = cv2.imread(refFilename, cv2.IMREAD_COLOR)

    # Read image to be aligned
    imFilename = "angled_view.PNG"
    print("[INFO]      Reading image to align : ", imFilename);
    im = cv2.imread(imFilename, cv2.IMREAD_COLOR)

    print("[INFO]      Aligning images...")
    # Registered image will be restored in imReg.
    # The estimated homography will be stored in h
    imReg, h = alignImages(imReference, im)

    # Write aligned image to disk.
    outFilename = "aligned.jpg"
    print("[INFO]      Saving aligned image  :  ", outFilename)
    cv2.imwrite(outFilename, imReg)

    # Print estimated homography
    print("[INFO]      Estimated homography  :  ", h)
