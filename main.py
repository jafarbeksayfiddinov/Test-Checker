import easyocr
import cv2
from matplotlib import pyplot as plt
from sympy.physics.vector import cross

IMAGE_PATH='d1.jpg'
# reader = easyocr.Reader(['en'],gpu=False)
# result = reader.readtext(IMAGE_PATH)
# for i in result:
#     print(i[1])


img=cv2.imread(IMAGE_PATH)
rows,cols,_=img.shape
print(rows,cols)

i=0
j=0
fig=1

while i+400<=rows:
    while j+400<=cols:
        cut_image=img[i:i+400,j:j+400]
        reader = easyocr.Reader(['en'], gpu=False)
        result = reader.readtext(image=cut_image)
        # print(result)

        for k in result:
            print(k[1])
        if len(result)!=0:
            top_left=tuple(result[0][0][0])
            bottom_right=tuple(result[0][0][2])
            text=result[0][1]
            font=cv2.FONT_HERSHEY_SIMPLEX

            img1=cv2.rectangle(cut_image,top_left,bottom_right,(0,255,0),5)
            img1=cv2.putText(img1,text,top_left,font,.5,(255,255,255),cv2.LINE_AA)

            cv2.imwrite(f"changed{fig}.jpg",img1)

        cv2.imwrite(f"crop{fig}.jpg",cut_image)

        # cv2.imshow("Crop:"+str(fig),cut_image)
        # cv2.waitKey(1000)
        fig+=1
        j+=350
    i+=350
    j=0

# top_left=tuple(result[0][0][0])
# bottom_right=n ym(result[0][0][2])
# text=result[0][1]
# font=cv2.FONT_HERSHEY_SIMPLEX

# img=cv2.rectangle(img,top_left,bottom_right,(0,255,0),5)
# img=cv2.putText(img,text,top_left,font,.5,(255,255,255),cv2.LINE_AA)
# plt.imshow(img)
# plt.show()