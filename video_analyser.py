import datetime
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import cv2
import torch
from PIL import Image
import numpy as np
from detect import predict, send_alert

def preprocess_frame(frame):
    """ Converte o frame do OpenCV para uma imagem PIL e aplica transformações. """
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Converte para RGB
    image_pil = Image.fromarray(image)  # Converte para PIL Image
    return image_pil

def main():
    video_path = 'video/video.mp4'

    if not os.path.exists(video_path):
        print(f"Erro: O arquivo de vídeo '{video_path}' não foi encontrado.")
        return

    video_capture = cv2.VideoCapture(video_path)
    if not video_capture.isOpened():
        print(f"Erro: Não foi possível abrir o arquivo de vídeo '{video_path}'.")
        return

    total_frames = 0
    intervalo = 2    
    last_alert = datetime.datetime.now() - datetime.timedelta(hours=1)

    print("Iniciando processamento do vídeo...")

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Fim do vídeo ou erro ao carregar o frame.")
            break

        total_frames += 1
        
        if total_frames % intervalo == 0:
            image_pil = preprocess_frame(frame)  # Converte o frame para imagem PIL
            is_danger = predict(image_pil)  # Chama a função de predição
            
            if is_danger:    
                # Verifica se já passou 10 segundos do ultimo alerta
                if (datetime.datetime.now() - last_alert).seconds < 10:
                    print(f"Alerta já enviado há {(datetime.datetime.now() - last_alert).seconds} segundos.")
                    
                else:
                    last_alert = datetime.datetime.now()
                    send_alert()  # Envia alerta por e-mail

            cv2.imshow('Video', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
