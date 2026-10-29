from pathlib import Path

import cv2
import pandas as pd

PASTA = Path(__file__).resolve().parent


def centro_da_esfera(frame):
    # Recorta a faixa do vídeo onde a esfera se movimenta
    altura = frame.shape[0]
    inicio = int(0.42 * altura)
    regiao = frame[inicio:int(0.78 * altura)]

    # Separa a esfera colorida do fundo e remove pequenos ruídos
    hsv = cv2.cvtColor(regiao, cv2.COLOR_BGR2HSV)
    mascara = cv2.inRange(hsv, (0, 43, 35), (179, 255, 255))
    elemento = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, elemento) # Abertura
    contornos, _ = cv2.findContours( # Pega limites das formas brancas
        mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    candidatos = []
    for contorno in contornos:
        area = cv2.contourArea(contorno) # Calcula área em pixels
        _, _, largura, altura = cv2.boundingRect(contorno)
        # Descarta objetos pequenos ou alongados, como o barbante
        if area < 2000 or max(largura, altura) / min(largura, altura) > 1.4:
            continue

        candidatos.append((area, contorno))


    _, esfera = max(candidatos, key=lambda item: item[0]) # Candidato de maior area
    momentos = cv2.moments(esfera)
    x = momentos['m10'] / momentos['m00']
    y = inicio + momentos['m01'] / momentos['m00']
    return x, y


def main():
    video = PASTA / 'IMG_2957.MOV'
    cap = cv2.VideoCapture(str(video)) # Abre para leitura
    fps = cap.get(cv2.CAP_PROP_FPS)

    dados = []
    quadro = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            x, y = centro_da_esfera(frame)
            # Converte o número do quadro em tempo, em segundos.
            dados.append((round(x, 3), round(y, 3), round(quadro / fps, 6)))
            quadro += 1

    finally:
        cap.release()

    pd.DataFrame(dados, columns=['x', 'y', 't']).to_csv(
        PASTA / 'dados_pendulo.csv', index=False
    )

if __name__ == '__main__':
    main()
