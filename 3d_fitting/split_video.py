import cv2
from facenet_pytorch import MTCNN
import numpy as np
device = 'cuda:0'

mtcnn = MTCNN(select_largest=False, device=device)

vname = 'media/src_adam.mp4'

dst = 'media/frame_avata_adam'

cap  = cv2.VideoCapture(f'{vname}')


count=0
bboxes = []
while count<90:
    ret,frame = cap.read()
    if not ret or frame is None:
        break
    img_arr = frame[:, :, ::-1]
    orig_h, orig_w = img_arr.shape[:2]
    bbox, probs = mtcnn.detect(img_arr)
    if bbox is None:
        continue
    bboxes.append(bbox[0])
    count+=1

bboxes = np.array(bboxes)
bboxes_mean = np.mean(bboxes,axis=0).astype(np.int16)
box_w = bboxes_mean[2]-bboxes_mean[0]
box_h = bboxes_mean[3]-bboxes_mean[1]


edge = min(box_w,box_h)

center_x =  int((bboxes_mean[2]+ bboxes_mean[0])/2)

center_y =  int((bboxes_mean[3]+ bboxes_mean[1])/2)


x1,y1,x2,y2 = int(center_x-edge*1.07), int(center_y-edge*1.07), int(center_x+edge*1.07), int(center_y+edge*1.07)


print(x1,y1,x2,y2)
# #yuminhong
# # x1,y1,x2,y2 = 200,500,920,1080 

# #obama 382 13 831 462
# # obama2 285 27 976 718
# #obama3 301 32 913 644
# #jane   200,500,920,1080 

# #obama_4 382 13 831 462
# #xiaoyan 250 629 859 1238
# #evelyn 302 407 861 966
#digital_a vata 544 1 1373 830
#digital_a_kpop 621 13 1314 706
#digital adam 210 14 437 241


count=0
cap  = cv2.VideoCapture(f'{vname}')

while 1:
    ret,frame = cap.read()
    if not ret or frame is None:
        break

    frame = frame[y1:y2,x1:x2]

    print(frame.shape)
    resized = cv2.resize(frame,(512,512))
    resized2= cv2.resize(frame,(256,256))

    print(resized.shape)

    cv2.imwrite(f'{dst}/{count:04d}.jpg',resized2)
    #cv2.imwrite(f'media/frame_xiaoyan_512/{count:04d}.jpg',resized)

    count+=1
    if count>6000:
        break
    # cv2.imshow('v', frame)
    
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()