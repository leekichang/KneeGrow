import os
import cv2
import pickle
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import utils

def plot_ROMs(ROMs, args):
    t = np.linspace(0, len(ROMs)/6, len(ROMs))
    plt.plot(t, ROMs)
    plt.xlabel('Time (sec)')
    plt.ylabel('ROM (°)')
    plt.title(f'ROM of {args.date}')
    plt.show()

def images_to_video(files, output_path, infos, frame_rate=15, codec="mp4v"):
    image_list = [cv2.imread(file) for file in tqdm(files)]
    # 이미지 크기를 기준으로 비디오의 가로와 세로 크기 설정    
    height, width, _ = image_list[0].shape

    # 비디오 라이터 객체 생성
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(output_path, fourcc, frame_rate, (width, height))

    # 이미지들을 비디오에 추가
    for idx, image in enumerate(tqdm(image_list)):
        cv2.line(image, infos['points'][idx]['p1'], infos['points'][idx]['p2'], (0, 255, 0), 2)
        cv2.line(image, infos['points'][idx]['p2'], infos['points'][idx]['p3'], (0, 255, 0), 2)
        
        for key in infos['points'][idx].keys():
            cv2.circle(image, infos['points'][idx][key], 5, (0,0,255), -1)
        angle_text = f"Angle: {infos['ROMs'][idx]:.2f} deg"
        cv2.putText(image, angle_text, (infos['points'][idx]['p2'][0] - 40, infos['points'][idx]['p2'][1] + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 4)
        out.write(image)

    # 비디오 라이터 해제
    out.release()

if __name__ == '__main__':
    args = utils.parse_args()
    path = f'./data/{args.date}'
    
    infos = {'ROMs':[], 'points':[]}
    files = [file for file in os.listdir(f'{path}/anno') if file.endswith('.pickle')]
    for file in files:
        with open(file=f'{path}/anno/{file}', mode='rb') as f:
            info = pickle.load(f)
        infos['points'].append(info['points'])
        infos['ROMs'].append(info['angle'])
    print(f'{"Maximum Extension:":>19} {min(infos["ROMs"]):>5.2f}°\n{"Maximum Flexion:":>19} {max(infos["ROMs"]):>5.2f}°')
    plot_ROMs(infos['ROMs'], args)
    
    frame_path = [os.path.join(f'{path}/frames', file) for file in os.listdir(f'{path}/frames') if file.endswith('.png')]
    output_video_path = f"./{args.date}_video.mp4"
    images_to_video(frame_path, output_video_path, infos)