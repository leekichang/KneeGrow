import os
import cv2
import pickle
import numpy as np
from tqdm import tqdm

import utils

class Annotator(object):
    def __init__(self, path):
        self.path = path
        self.files = [file for file in os.listdir(f'{path}/frames') if file.endswith('.png')]
        self.image_list = []
        
        self.image_index = 0
        self.image_points = [{'p1':(400,540), 'p2':(960,540), 'p3':(1520, 540)}]
        self.selected_point = None
        self.selected_point_key = None
        
        self.window_name = "Image Annotation"
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.on_mouse_event)
    
    def key2point(self, key):
        return self.image_points[self.image_index][key]
        
    def load_imgs(self):
        files = [os.path.join(f'{path}/frames', file) for file in self.files]
        for file in tqdm(files):
            self.image_list.append(cv2.imread(file))

    def show_img(self, idx):
        cv2.imshow(self.files[idx], self.image_list[idx])
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def on_mouse_event(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            for key in self.image_points[self.image_index]:
                point = self.image_points[self.image_index][key]
                if abs(point[0] - x) < 10 and abs(point[1] - y) < 10:
                    self.selected_point = point
                    self.selected_point_key = key
                    break
        elif event == cv2.EVENT_LBUTTONUP:
            if self.selected_point is not None:
                self.image_points[self.image_index][self.selected_point_key] = self.selected_point
                self.draw_points(self.image_list[self.image_index].copy())
            self.selected_point = None
            self.selected_point_key = None
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.selected_point is not None:
                self.selected_point = (x, y)
    
    def draw_points(self, img):
        for key in self.image_points[self.image_index].keys():
            point = self.image_points[self.image_index][key]
            cv2.circle(img, point, 5, (0, 0, 255), -1)
    
    def calc_angle(self):
        vector1 = np.array(self.image_points[self.image_index]['p1']) - np.array(self.image_points[self.image_index]['p2'])
        vector2 = np.array(self.image_points[self.image_index]['p2']) - np.array(self.image_points[self.image_index]['p3'])
        dot_product = np.dot(vector1, vector2)
        norm_product = np.linalg.norm(vector1) * np.linalg.norm(vector2)
        angle_rad = np.arccos(dot_product / norm_product)
        angle_deg = np.degrees(angle_rad)
        return angle_deg
    
    
    def draw_angle_arc(self, img):
        vector1 = np.array(self.image_points[self.image_index]['p1']) - np.array(self.image_points[self.image_index]['p2'])
        vector2 = np.array(self.image_points[self.image_index]['p2']) - np.array(self.image_points[self.image_index]['p3'])
        dot_product = np.dot(vector1, vector2)
        norm_product = np.linalg.norm(vector1) * np.linalg.norm(vector2)
        angle_rad = np.arccos(dot_product / norm_product)

        angle_deg = np.degrees(angle_rad)
        angle_text = f"Angle: {angle_deg:.2f} deg"

        if len(self.image_points[self.image_index]) == 3:
            if 0 <= angle_rad <= np.pi:
                cv2.ellipse(img, self.image_points[self.image_index]['p2'], (50, 50), 0, 0, np.degrees(angle_rad), (0, 255, 255), 2)
            else:
                cv2.ellipse(img, self.image_points[self.image_index]['p2'], (50, 50), 0, 0, np.degrees(angle_rad), (0, 255, 255), -2)

        # 텍스트를 추가하여 각도 정보를 시각화
        cv2.putText(img, angle_text, (self.image_points[self.image_index]['p2'][0] - 40, self.image_points[self.image_index]['p2'][1] + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 4)

    
    def annotate_image(self):
        while self.image_index < len(self.image_list):
            img = self.image_list[self.image_index].copy()
            
            if self.selected_point is not None:
                cv2.circle(img, self.selected_point, 7, (0, 255, 0), -1)

            if len(self.image_points[self.image_index]) >= 2:
                cv2.line(img, self.image_points[self.image_index]['p1'], self.image_points[self.image_index]['p2'], (0, 255, 0), 2)
            if len(self.image_points[self.image_index]) == 3:
                cv2.line(img, self.image_points[self.image_index]['p2'], self.image_points[self.image_index]['p3'], (0, 255, 0), 2)
            self.draw_points(img)
            self.draw_angle_arc(img)

            cv2.imshow(self.window_name, img)
            key = cv2.waitKey(1)

            if key == 27:  # 'esc' key
                self.image_points[self.image_index] = {'p1':(400,540), 'p2':(960,540), 'p3':(1520, 540)}
            elif key == 13:  # 'enter' key
                if len(self.image_points[self.image_index]) == 3:
                    with open(f'{self.path}/anno/{self.files[self.image_index].split(".")[0]}.pickle', 'wb') as f:
                        pickle.dump({'points':self.image_points[self.image_index], 'angle':self.calc_angle()}, f, protocol=pickle.HIGHEST_PROTOCOL)
                    print(f'{self.image_index+1}/{len(self.image_list)}: {self.image_points[self.image_index]}, angle: {self.calc_angle():.2f}°')
                    self.image_index += 1
                    self.image_points.append(self.image_points[self.image_index - 1].copy())
                    if self.image_index >= len(self.image_list):
                        break
                else:
                    cv2.destroyAllWindows()
        return self.image_points
    
    
    
if __name__ == '__main__':
    args = utils.parse_args()
    path = f'./data/{args.date}'
    annotator = Annotator(path)
    annotator.load_imgs()
    
    result = annotator.annotate_image()
    