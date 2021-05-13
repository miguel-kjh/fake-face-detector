import cv2
import os

from matplotlib import pyplot as plt
from retinaface.pre_trained_models import get_model
from retinaface.utils import vis_annotations
from PIL import Image

model = get_model("resnet50_2020-07-20", max_size=1024, device = "cuda")
model.eval()

def getFaces(image, annotation, filesave):

  for face in annotation:
    box = face['bbox']
    copy_img = image[box[1]:box[3], box[0]:box[2]]
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

if __name__ == '__main__':
    main()