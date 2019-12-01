"""
图片的人脸标定
"""
# coding: utf-8
import cv2
import dlib
# draw_all_landmarks_id
detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor('/home/tanhui/notebook/shape_predictor_68_face_landmarks.dat')
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

img = cv2.imread('2.jpg')
cv2.imshow('landmarks1', img)
print(img.shape,'sssss')
dets = detector(img, 0)  # dlib人脸检测
print(dets[0])

for i, d in enumerate(dets):
    cv2.rectangle(img, (d.left(), d.top()), (d.right(), d.bottom()), (0, 255, 0), 1)
    shape = predictor(img, d)  # dlib人脸特征点检测
    for k in range(0, 68):  # 68个特征点
        cv2.circle(img,(shape.part(k).x, shape.part(k).y), 1, (0, 0, 255), -1)  #-1表示填充
        #cv2.putText(img_out,'%d' % k,(shape.part(k).x,shape.part(k).y),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,0,255),1) #标记点号
        print (shape.part(k).x,shape.part(k).y)
cv2.imwrite('face_landmarks.jpg', img)
# print('success')

cv2.imshow('landmarks', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
