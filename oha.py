from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # Salva o gráfico sem abrir uma janela.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, savgol_filter

PASTA = Path(__file__).resolve().parent


def oha(t, x0, A, b, omega, phi):
    return x0 + A * np.exp(-b * t) * np.cos(omega * t + phi)


def chutes_iniciais(t, x):
    # Suaviza os dados e localiza máximos e mínimos para estimar o período.
    dt = np.median(np.diff(t))
    suave = savgol_filter(x, 11, 2)
    distancia = int(0.9 / dt)
    proeminencia = 0.15 * np.ptp(x)
    picos, _ = find_peaks(suave, distance=distancia, prominence=proeminencia)
    vales, _ = find_peaks(-suave, distance=distancia, prominence=proeminencia)
    periodos = np.concatenate([np.diff(t[picos]), np.diff(t[vales])])
    omega = 2 * np.pi / np.median(periodos)

    # Estima o amortecimento pela queda das amplitudes dos extremos.
    meio = (np.median(x[picos]) + np.median(x[vales])) / 2
    extremos = np.sort(np.concatenate([picos, vales]))
    amplitudes = np.abs(x[extremos] - meio)
    inclinacao = np.polyfit(t[extremos], np.log(amplitudes), 1)[0]
    b = -inclinacao

    # Estima amplitude, fase e equilíbrio com seno e cosseno.
    matriz = np.column_stack([np.cos(omega * t), np.sin(omega * t), np.ones_like(t)])
    c, s, x0 = np.linalg.lstsq(matriz, x, rcond=None)[0]
    return [x0, np.hypot(c, s), b, omega, np.arctan2(-s, c)]
        # x0, A0, b, omega, fase inicial


def main():
    # Lê o CSV gerado pelo primeiro programa e inicia o tempo em zero.
    dados = pd.read_csv(PASTA / 'dados_pendulo.csv')
    t = dados['t'].to_numpy(dtype=float)
    x = dados['x'].to_numpy(dtype=float)
    t = t - t[0]

    # Ajusta a oscilação amortecida aos dados experimentais.
    parametros, _ = curve_fit(
        oha, t, x, p0=chutes_iniciais(t, x),
        bounds=([-np.inf, 0, 0, 0, -2 * np.pi],
                [np.inf, np.inf, 1, 20, 2 * np.pi]), # Limites
        maxfev=40000,
    )
    x0, A, b, omega, phi = parametros

    # Calcula Q, período, frequência e qualidade do ajuste.
    Q = omega / (2 * b)
    periodo = 2 * np.pi / omega
    r2 = 1 - np.sum((x - oha(t, *parametros)) ** 2) / np.sum((x - x.mean()) ** 2)
    pd.DataFrame([{
        'x0_pixels': x0,
        'amplitude_inicial_pixels': A,
        'amortecimento_b_s-1': b,
        'omega_rad_s': omega,
        'fase_rad': phi,
        'periodo_s': periodo,
        'frequencia_Hz': 1 / periodo,
        'fator_qualidade_Q': Q,
        'R2': r2,
    }]).to_csv(PASTA / 'parametros_pendulo.csv', index=False, float_format='%.9f')

    # Compara as posições medidas com a curva ajustada.
    fig, ax = plt.subplots()
    ax.plot(t, x - x0, '.', markersize=2, label='Dados do vídeo')
    tempo = np.linspace(t.min(), t.max(), 2500)
    ax.plot(tempo, oha(tempo, *parametros) - x0, label=f'Ajuste: Q = {Q:.2f}')
    ax.set_xlabel('Tempo (s)')
    ax.set_ylabel('Posição relativa ao equilíbrio (pixels)')
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(PASTA / 'grafico_pendulo.png', dpi=170)
    plt.close(fig)


if __name__ == '__main__':
    main()
