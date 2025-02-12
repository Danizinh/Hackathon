import cv2
from detect import predict, send_alert
from video_analyser import preprocess_frame

def capture_video():
    # Inicia a captura de vídeo da webcam
    cap = cv2.VideoCapture(0)

    # Verifica se a captura foi aberta corretamente
    if not cap.isOpened():
        print("Erro ao acessar a webcam.")
        return

    try:
        while True:
            # Captura frame por frame
            ret, frame = cap.read()

            if not ret:
                break

            # Exibe o frame capturado
            cv2.imshow('Webcam', frame)
            
            image_pil = preprocess_frame(frame)  # Converte o frame para imagem PIL
            is_danger = predict(image_pil)  # Chama a função de predição
            
            if is_danger:
                send_alert()  # Envia alerta por e-mail

            # Parar o loop ao pressionar a tecla 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        pass

    # Libera a captura de vídeo e fecha todas as janelas
    cap.release()
    cv2.destroyAllWindows()

# Chama a função para capturar e exibir vídeo da webcam
if __name__ == "__main__":
    capture_video()