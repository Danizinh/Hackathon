# Detecta Objetos Cortantes em Vídeos

Este projeto utiliza redes neurais para detectar objetos cortantes em vídeos capturados por webcam ou por vídeos gravados. Ele usa um modelo pré-treinado para classificar imagens e alertar caso um objeto cortante seja detectado.

## Requisitos

Certifique-se de ter instalado:

- Python 3.8+
- pip
- Ambiente virtual configurado

## Instalação

### 1. Clonar o repositório

```sh
# Clone este repositório
git clone https://github.com/Danizinh/Hackathon.git
```

### 2. Criar ambiente virtual

#### No Windows (cmd/powershell):

```sh
python -m venv venv
venv\Scripts\activate
```

#### No Linux/macOS:

```sh
python3 -m venv venv
source venv/bin/activate
```

### 3. Instalar dependências

```sh
pip install -r requirements.txt
```

## Como Executar

### 1. Treinar o modelo (opcional)

Se o modelo ainda não foi treinado, você pode iniciar o treinamento:

```sh
python detect.py
```

Se o modelo já estiver salvo (`modelo_cortantes.pth`), esta etapa pode ser ignorada.

### 2. Iniciar a detecção por webcam

```sh
python camera_analyser.py
```

Aperte `q` para encerrar a execução.

### 3. Analisar um vídeo existente

Salve o vídeo a ser analizado na pasta "video", como o nome: video.mp4

```sh
python video_analyser.py
```

Aperte `q` para encerrar a execução.

## Estrutura do Projeto

```
├── detect.py            # Código principal para treinar, detectar objetos e enviar alertas
├── camera_analyser.py   # Captura de vídeo e análise em tempo real
├── video_analyser.py    # Analisa um vídeo gravado
├── requirements.txt     # Dependências do projeto
├── dataset/             # Diretório onde as imagens de treinamento devem ser armazenadas
└── modelo_cortantes.pth # Modelo treinado (gerado após o treinamento)
```

## Observações

- O script `detect.py` irá perguntar se deseja treinar o modelo caso o arquivo `modelo_cortantes.pth` não esteja presente.
- O script `video_analyser.py` verifica um vídeo já existente e analisa cada frame.
- O script `camera_analyser.py` usa a webcam para capturar vídeo e analisar cada frame.
- Independente se for um vídeo gravado ou da câmera em tempo real, quando for detectado um objeto cortante, o sistema enviará um alerta (mock). Caso já tenha sido enviado um alerta recentemente (menos de 10 segundos), o sistema ignorará este alerta.
