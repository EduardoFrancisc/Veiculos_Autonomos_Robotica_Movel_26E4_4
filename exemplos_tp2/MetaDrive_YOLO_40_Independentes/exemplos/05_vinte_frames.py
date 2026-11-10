"""
Exemplo 05/40 — Capturar vinte imagens

O QUE FAZ:
Obtem 20 imagens reais da camera frontal e grava quadros selecionados.

COMO FUNCIONA, PASSO A PASSO:
Executa steps com politica aleatoria e extrai o ultimo frame de observacao.
O codigo pode ser aberto e executado individualmente. Suas funcoes estao
neste mesmo arquivo, sem importacao de um modulo de apoio do curso.
O terminal apresenta dados concretos e as imagens/relatorios ficam na pasta
resultados/05_vinte_frames/ quando a operacao estiver concluida.

IMPORTANCIA PARA SISTEMAS AUTONOMOS:
A sequencia de cameras tem custo proprio, mesmo sem rede profunda.
O exercicio permite discutir a diferenca entre observacao visual, estimativa
por processamento de imagem e medida fisica validada. Nem toda classe COCO
representa uma placa, e dados de pixels nao devem receber unidade fisica sem
calibracao da camera e da geometria da pista.

DEPENDENCIAS: consulte requirements.txt; para cenas reais simuladas, e necessario
MetaDrive com renderizacao grafica funcional. Em Google Colab, configurar EGL ou
Xvfb; use_render=True e apropriado para desktop com janela.
EXERCICIO AO ALUNO: modifique os parametros, repita com a mesma semente,
compare o resultado e descreva a mudanca visual e temporal observada.
"""
from __future__ import annotations
import argparse
import csv
import json
import time
from pathlib import Path
from collections import Counter
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parent.parent
DADOS = ROOT / "dados"
RESULTADOS = ROOT / "resultados" / Path(__file__).stem
RESULTADOS.mkdir(parents=True, exist_ok=True)

def imagem_demo():
    """Fotografia externa fornecida com o pacote; NAO e frame do MetaDrive."""
    caminho = DADOS / 'frontal_kitti.png'
    frame = cv2.imread(str(caminho), cv2.IMREAD_COLOR)
    if frame is None:
        raise FileNotFoundError(f'Imagem nao encontrada: {caminho}')
    return cv2.resize(frame, (640, 360))


def salvar(nome, imagem):
    caminho = RESULTADOS / nome
    if not cv2.imwrite(str(caminho), imagem):
        raise OSError(f'Falha ao gravar {caminho}')
    print('Imagem salva:', caminho)
    return caminho


def salvar_csv(nome, registros):
    caminho=RESULTADOS/nome
    if not registros:
        raise ValueError('Nao ha registros a salvar; o teste falhou antes do primeiro step.')
    colunas=list(registros[0].keys())
    with caminho.open('w', newline='', encoding='utf-8') as f:
        w=csv.DictWriter(f, fieldnames=colunas);w.writeheader();w.writerows(registros)
    print('CSV:', caminho, '| registros:', len(registros))
    return caminho


def escrever_texto(nome, texto):
    p=RESULTADOS/nome;p.write_text(texto,encoding='utf-8')
    print('Relatorio:',p)

def abrir_metadrive(render=False, largura=640, altura=360, semente=42, densidade=0.15):
    """Instancia MetaDrive REAL com camera frontal RGB (nao usa frames fabricados).

    Desktop com janela: --render; Google Colab: sem janela e EGL/Xvfb configurado.
    A observacao da camera e obtida diretamente de env.step().
    """
    try:
        from metadrive.envs.metadrive_env import MetaDriveEnv
        from metadrive.component.sensors.rgb_camera import RGBCamera
    except ImportError as e:
        raise RuntimeError('MetaDrive nao instalado. Leia LEIA_PRIMEIRO.md; pip install metadrive-simulator') from e
    cfg={
        'use_render': bool(render),
        'image_observation': True,
        'norm_pixel': False,
        'stack_size': 1,
        'sensors': {'rgb_camera': (RGBCamera, largura, altura)},
        'vehicle_config': {'image_source': 'rgb_camera'},
        'traffic_density': float(densidade),
        'horizon': 1100,
        'start_seed': int(semente),
        'num_scenarios': 1024,  # margem para seeds de reinicio em ate 500+ episodios
        'log_level': 50,
    }
    return MetaDriveEnv(cfg)


def obter_frame_bgr(obs):
    """Recupera o ultimo frame RGB da observacao e converte para BGR uint8."""
    if not isinstance(obs, dict) or 'image' not in obs:
        raise RuntimeError(f'Observacao inesperada. Esperado dict com image; recebido {type(obs)}')
    frame=np.asarray(obs['image'])
    if frame.ndim == 4: frame=frame[..., -1]
    if frame.ndim != 3 or frame.shape[-1] not in (3,4):
        raise RuntimeError(f'Formato inesperado do frame: {frame.shape}')
    frame=frame[..., :3]
    if np.issubdtype(frame.dtype, np.floating):
        frame=np.clip(frame*(255. if float(np.max(frame)) <= 1.01 else 1.),0,255).astype(np.uint8)
    else: frame=np.clip(frame,0,255).astype(np.uint8)
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


def sequencia_acoes(n, semente=42):
    """Mesmas acoes para todos os benchmarks; politica aleatoria ilustrativa."""
    rng=np.random.default_rng(semente)
    return [(float(rng.uniform(-0.12,0.12)),float(rng.uniform(0.65,0.95))) for _ in range(n)]


def velocidade_agente(env):
    """Velocidade informada pelo simulador; sem inferir unidade a partir de pixels."""
    try:return float(env.agent.speed)
    except (AttributeError, TypeError, ValueError): return float('nan')

def args_comuns():
    parser=argparse.ArgumentParser(description=__doc__.splitlines()[1] if __doc__ else 'Exemplo independente')
    parser.add_argument('--render',action='store_true',help='Usar janela desktop. No Colab, omitir.')
    parser.add_argument('--steps',type=int,default=None,help='Substitui o numero de steps didatico')
    parser.add_argument('--imgsz',type=int,default=640,help='Resolucao de inferencia YOLO')
    parser.add_argument('--skip',type=int,default=1,help='YOLO a cada N frames')
    parser.add_argument('--offline',action='store_true',help='Somente imagem de amostra; NAO e benchmark do MetaDrive')
    return parser.parse_args()


if __name__ == "__main__":
    a=args_comuns()
    env=abrir_metadrive(a.render);acoes=sequencia_acoes(20)
    try:
        obs,info=env.reset(seed=42)
        for i,ac in enumerate(acoes):
            obs,r,done,trunc,info=env.step(ac)
            if i in (0,9,19):salvar(f'frame_{i:02d}.png',obter_frame_bgr(obs))
            if done or trunc:obs,info=env.reset(seed=42+i+1)
    finally:env.close()
    print('Foram feitos 20 steps (com reinicio se episodio terminar).')
