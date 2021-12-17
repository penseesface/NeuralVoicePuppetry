import face_alignment
import os
import cv2
device = 'cuda:0'

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device=device)


input = cv2.imread('media/frames_obama2_fit_gt/train/0000_crop.jpg')
input = cv2.resize(input,(256,256))
print(input.shape)
input = input[:,:,::-1]
lms = fa.get_landmarks(input)

img = input.copy()

for point in lms[0]:
    print(point)
    cv2.circle(img,(int(point[0]),int(point[1])),2,(255,255,0),1)

cv2.imshow('img_show',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# src = 'media/frames_zhubo_fit_gt/train'

# print(os.listdir(src))

# src_expression = [x for x in os.listdir(src) if x.endswith('crop.jpg')]

# src_expression = sorted(src_expression)

# print(src_expression)



# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# fps = 30
  
# height = int(512)
# width = int(512)
# print("fps", fps)
# print("frame_height", height)
# print("frame_width", width)

# V_writer = cv2.VideoWriter(f'media/background/videorendered.mp4',fourcc, fps, (width,height))


# for path in src_expression:

#     img =  cv2.imread(f'{src}/{path}')

#     V_writer.write(img)

# V_writer.release()
