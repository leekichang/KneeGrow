import os
import cv2
from tqdm import tqdm

class Annotator(object):
    def __init__(self, path):
        self.path = path
        self.files = [os.path.join(path, file) for file in os.listdir(path) if file.endswith('.png')]
        self.imgs = []

    def load_imgs(self):
        for file in tqdm(self.files):
            self.imgs.append(cv2.imread(file))

    def show_img(self, idx):
        cv2.imshow(self.files[idx], self.imgs[idx])
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == '__main__':
    date = '2023-07-26'
    path = f'./data/{date}/frames'
    annotator = Annotator(path)
    annotator.load_imgs()
    annotator.show_img(0)
    