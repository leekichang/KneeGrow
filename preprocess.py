import os
import cv2
from tqdm import tqdm

import utils

def save_frames(input_file, output_path, frame_rate_divisor=2):
    """
    동영상 파일의 frame rate를 낮추어 프레임을 이미지로 저장하는 함수
    
    Parameters:
        input_file (str): 입력 동영상 파일 경로 (예: 'video.mov')
        output_path (str): 이미지를 저장할 디렉토리 경로 (예: './frames/')
        frame_rate_divisor (int): frame rate를 낮추기 위한 간격 (기본값: 2)
    """
    # VideoCapture 객체 생성
    cap = cv2.VideoCapture(input_file)
    
    # 동영상 파일이 올바르게 열렸는지 확인
    if not cap.isOpened():
        print("Error: 동영상 파일을 열 수 없습니다.")
        return
    
    # 프레임 수와 해상도 정보 가져오기
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 저장할 이미지 파일 경로 설정
    file_name = input_file.split('/')[-1].split('.')[0]  # 파일 이름 추출 (확장자 제외)
    image_file_format = '.png'  # 이미지 파일 형식 (확장자)
    
    # 이미지 저장 디렉토리 생성
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # 프레임별로 이미지 저장 (frame_rate_divisor만큼 간격으로 저장)
    frame_number, n_frames = 0, 0
    while True:
        # 프레임 읽기
        ret, frame = cap.read()
        if not ret:
            break  # 프레임을 더 이상 읽을 수 없으면 종료
        
        # frame_rate_divisor만큼 간격으로 이미지 저장
        if frame_number % frame_rate_divisor == 0:
            image_file_path = os.path.join(output_path, f"{file_name}_{frame_number:04d}{image_file_format}")
            cv2.imwrite(image_file_path, frame)
            n_frames += 1
        frame_number += 1
    
    # VideoCapture 객체 해제
    cap.release()
    print(f"프레임별 이미지 저장이 완료되었습니다. 총 {n_frames}개의 이미지가 저장되었습니다.")

if __name__ == '__main__':
    args = utils.parse_args()
    path = f'./data/{args.date}'
    video_files = [os.path.join(f'{path}/videos', file).replace("\\", "/") for file in os.listdir(f'{path}/videos/')]
    # 동영상 파일 경로와 이미지 저장 디렉토리 경로 설정
    for file in video_files:
        print(file)
        output_image_path = f'{path}/frames/'
        # frame rate를 낮춰 프레임을 이미지로 저장 (frame_rate_divisor를 조정하여 frame rate 조절)
        frame_rate_divisor = 5  # frame rate를 낮추기 위한 간격
        save_frames(file, output_image_path, frame_rate_divisor)
