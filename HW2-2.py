import numpy as np
import cv2

imgname1 = 'test1.png'
imgname2 = 'test1.png'

flann_params= dict(algorithm = 6,
 table_number = 6, # 12
 key_size = 12, # 20
 multi_probe_level = 1) #2

sift = cv2.xfeatures2d.SIFT_create()

img1 = cv2.imread(imgname1)
kp1, des1 = sift.detectAndCompute(img1,None)

img2 = cv2.imread(imgname2)
rows, cols, ch = img2.shape
pts1 = np.float32([[0, 0], [cols - 1, 0], [0, rows - 1]])
pts2 = np.float32([[cols * 0.3, rows * 0.3], [cols * 0.9, rows * 0.2], [cols * 0.1, rows * 0.9]])
M = cv2.getAffineTransform(pts1, pts2)
img2 = cv2.warpAffine(img2, M, (cols, rows))

kp2, des2 = sift.detectAndCompute(img2,None)


bf = cv2.BFMatcher()
raw_matches = bf.knnMatch(des1,des2, k=2)

good = []
for m,n in raw_matches:
 if m.distance < 0.7*n.distance:
     good.append(m)

if len(good)>10:
 src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
 dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
 M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)

 matchesMask = mask.ravel().tolist()

else:
 print("Not enough matches are found - %d/%d" % (len(good),10))
 matchesMask = None
draw_params = dict(matchColor = (0,255,0),
 singlePointColor = (0,0,255),
 matchesMask = matchesMask,
 flags = 2)

vis = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)

cv2.imshow("", vis)
cv2.waitKey()
cv2.destroyAllWindows()