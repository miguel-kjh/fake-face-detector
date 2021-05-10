import cv2
import os
import re

from matplotlib import pyplot as plt
from retinaface.pre_trained_models import get_model
from retinaface.utils import vis_annotations
from PIL import Image

model = get_model("resnet50_2020-07-20", max_size=1024, device = "cuda")
model.eval()

face_cascade = cv2.CascadeClassifier('faces_detectors/haarcascade_frontalface_default.xml')

def isImage(filename: str):
    return re.search(".jpg$", filename)

def getFaces(image, annotation, filesave):

  for face in annotation:
    box = face['bbox']
    #print(box)
    copy_img = image[box[1]:box[3], box[0]:box[2]]
    #plt.figure(index)
    #plt.imshow(copy_img)
    #plt.show()
    print(filesave)
    cv2.imwrite(filesave, copy_img)

def main():
    folder = "Database_real_and_fake_face_160x160"
    folder_faces = "Database_real_and_fake_only_face"
    for path in reversed(os.listdir(folder)):
        print(path)
        img_path = os.path.join(folder, path)
        img_path_face = os.path.join(folder_faces, path)
        for file in os.listdir(img_path):
            # img = Image.open(file).convert("RGB")
            if file not in os.listdir(img_path_face):
                image_real = cv2.imread(os.path.join(img_path, file))
                image = cv2.cvtColor(image_real, cv2.COLOR_BGR2RGB)

                annotation = model.predict_jsons(image)
                if annotation:
                    try:
                        getFaces(image_real, annotation, os.path.join(img_path_face, file))
                    except:
                        print("Error!!")
                        continue
                #plt.imshow(vis_annotations(image, annotation))
                #plt.show()


                """faces = face_cascade.detectMultiScale(image, 1.1, 4)
                # Draw rectangle around the faces
                for (x, y, w, h) in faces:
                    cv2.rectangle(image_real, (x, y), (x+w, y+h), (255, 0, 0), 2)
                # Display the output
                plt.imshow(image_real)
                plt.show()"""

if __name__ == '__main__':
    main()