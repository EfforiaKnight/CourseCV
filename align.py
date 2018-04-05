import cv2
import numpy as np
import argparse


def align_images(img, ref, max_matches, good_match_percent):
    # Convert images to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)

    # Detect ORB features and compute descriptors.
    orb = cv2.ORB_create(max_matches)
    keypoints_img, descriptors_img = orb.detectAndCompute(img_gray, None)
    keypoints_ref, descriptors_ref = orb.detectAndCompute(ref_gray, None)

    # Match features.
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors_img, descriptors_ref, None)

    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)

    # Remove not so good matches
    num_good_matches = int(len(matches) * good_match_percent)
    matches = matches[:num_good_matches]

    # Draw top matches
    img_matches = cv2.drawMatches(img, keypoints_img, ref, keypoints_ref, matches, None)
    cv2.imwrite("matches.jpg", img_matches)

    # Extract location of good matches
    points_img = np.zeros((len(matches), 2), dtype=np.float32)
    points_ref = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points_img[i, :] = keypoints_img[match.queryIdx].pt
        points_ref[i, :] = keypoints_ref[match.trainIdx].pt

    # Find homography
    h, mask = cv2.findHomography(points_img, points_ref, cv2.RANSAC)

    # Use homography
    height, width, channels = ref.shape
    img_reg = cv2.warpPerspective(img, h, (width, height))

    return img_reg, h

def verify_signatures(img_ref, img_reg):
    img_diff = cv2.absdiff(img_ref, img_reg)
    cv2.imwrite("img_diff.jpg", img_diff)


if __name__ == '__main__':
    # Construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--img", required=True, help="Path to the scanned image")
    ap.add_argument("-r", "--img-ref", required=True, help="Path to the reference image")
    ap.add_argument("--max-matches", default=500, type=int, help="Max matches for ORB feature detector")
    ap.add_argument("--good-match-percent", default=0.15, type=float, help="Percent of good matches to keep")
    args = ap.parse_args()

    # Read reference image
    print("Reading reference image : ", args.img_ref)
    img_ref = cv2.imread(args.img_ref, cv2.IMREAD_COLOR)

    # Read image to be aligned
    print("Reading image to align : ", args.img);  
    img = cv2.imread(args.img, cv2.IMREAD_COLOR)

    print("Aligning images ...")
    # Registered image will be restored in imReg. 
    # The estimated homography will be stored in h. 
    img_reg, h = align_images(img, img_ref, args.max_matches, args.good_match_percent)
    
    # Verify signatures
    print("Verifying signatures ...")
    verify_signatures(img_ref, img_reg)

    # Write aligned image to disk. 
    out_filename = "aligned.jpg"
    print("Saving aligned image : ", out_filename); 
    cv2.imwrite(out_filename, img_reg)

    # Print estimated homography
    print("Estimated homography matrix: \n",  h)
  