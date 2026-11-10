"""
Exemplo 28/40 — Executar YOLO em 20 steps

O QUE FAZ:
Liga MetaDrive ao YOLO em fluxo curto.

COMO FUNCIONA, PASSO A PASSO:
Cada env.step gera um novo frame entregue ao YOLOv8n.
O codigo pode ser aberto e executado individualmente. Suas funcoes estao
neste mesmo arquivo, sem importacao de um modulo de apoio do curso.
O terminal apresenta dados concretos e as imagens/relatorios ficam na pasta
resultados/28_yolo_20_steps/ quando a operacao estiver concluida.

IMPORTANCIA PARA SISTEMAS AUTONOMOS:
O reconhecimento de objetos sinteticos pode falhar por diferenca de dominio.
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

CLASSES_TRAFEGO={'person','bicycle','car','motorcycle','bus','truck','traffic light','stop sign'}

def carregar_yolo():
    """Carrega YOLOv8n treinado em COCO; primeiro uso pode baixar pesos (internet)."""
    try:from ultralytics import YOLO
    except ImportError as e:raise RuntimeError('Execute pip install ultralytics') from e
    return YOLO('yolov8n.pt')


def detectar_yolo(modelo, frame, imgsz=640, conf=0.25, memoria=None):
    """Inferencia YOLO, classes COCO de interesse e caixas rotuladas.

    COCO contem 'stop sign' e 'traffic light', mas nao todas as placas de transito.
    Deteccoes vazias sao validas. Em imagens MetaDrive ha domain gap visual.
    """
    resultados=modelo.predict(source=frame,imgsz=imgsz,conf=conf,verbose=False)
    det=resultados[0]
    saida=frame.copy(); contagem=Counter()
    if memoria is not None: memoria.clear()
    if det.boxes is not None:
        boxes=det.boxes.xyxy.cpu().numpy()
        cls=det.boxes.cls.cpu().numpy().astype(int)
        probs=det.boxes.conf.cpu().numpy()
        for coords,id_,p in zip(boxes,cls,probs):
            nome=str(det.names[int(id_)])
            if nome not in CLASSES_TRAFEGO:continue
            contagem[nome]+=1
            x1,y1,x2,y2=map(int,coords)
            if memoria is not None:memoria.append((x1,y1,x2,y2,nome,float(p)))
            cv2.rectangle(saida,(x1,y1),(x2,y2),(30,220,80),2)
            cv2.putText(saida,f'{nome} {float(p):.2f}',(x1,max(20,y1-7)),
                        cv2.FONT_HERSHEY_SIMPLEX,.48,(30,220,80),2)
    return saida,dict(contagem)

def executar_episodios(steps, render=False, detectar_faixa=False, yolo=False, skip=1,
                     imgsz=640, seed=42, tamanho=(640,360), save_video=True):
    """Executa steps EXATOS (com reset apos fim do episodio).

    FPS inclui env.step + pipeline + inferencia, exclui download/inicializacao.
    Politica aleatoria e seed equivalentes entre cenarios: nao e conducao segura.
    Frame skip conserva ultima saida do YOLO e marca detec_atualizada=0.
    """
    from time import perf_counter
    modelo=carregar_yolo() if yolo else None
    if modelo is not None:
        # Aquecimento separado do cronometro; reduz viés de primeira inferencia.
        dummy=np.zeros((tamanho[1],tamanho[0],3),np.uint8)
        for _ in range(2):modelo.predict(source=dummy,imgsz=imgsz,verbose=False)
    env=abrir_metadrive(render,tamanho[0],tamanho[1],seed)
    writer=None
    if save_video:
        out=RESULTADOS/f'feed_{"yolo" if yolo else "baseline"}_{steps}.mp4'
        writer=cv2.VideoWriter(str(out),cv2.VideoWriter_fourcc(*'mp4v'),12,tamanho)
        if not writer.isOpened():raise RuntimeError('Nao foi possivel criar MP4. Verifique codecs.')
    obs=None; episodio=0; ultima_contagem={}; ultima_boxes=[]; linhas=[]
    acoes=sequencia_acoes(steps,seed)
    try:
        obs,info=env.reset(seed=seed)
        for idx,acao in enumerate(acoes):
            t0=perf_counter()
            obs,recompensa,done,trunc,info=env.step(acao)
            sim_ms=(perf_counter()-t0)*1000
            frame=obter_frame_bgr(obs)
            faixa_ms=0.;yolo_ms=0.
            anotado=frame.copy()
            desvio_px=None;curvatura_px=None
            if detectar_faixa:
                ta=perf_counter();res=detectar_faixas(frame)
                faixa_ms=(perf_counter()-ta)*1000
                anotado=res['imagem'];desvio_px=res['desvio_px'];curvatura_px=res['curvatura_raio_px']
            atualizou=0
            if modelo is not None:
                if idx%max(1,int(skip))==0:
                    ty=perf_counter()
                    anotado, ultima_contagem=detectar_yolo(modelo,anotado,imgsz=imgsz,memoria=ultima_boxes)
                    # Em GPU, esperar para medir custo real.
                    try:
                        import torch
                        if torch.cuda.is_available():torch.cuda.synchronize()
                    except ImportError:pass
                    yolo_ms=(perf_counter()-ty)*1000
                    atualizou=1
                elif ultima_boxes:
                    # Reutiliza apenas coordenadas anteriores NA IMAGEM ATUAL; nao repete frame antigo.
                    for x1,y1,x2,y2,nome,p in ultima_boxes:
                        cv2.rectangle(anotado,(x1,y1),(x2,y2),(0,165,255),2)
                        cv2.putText(anotado,f'{nome} {p:.2f} (antigo)',(x1,max(20,y1-7)),
                                    cv2.FONT_HERSHEY_SIMPLEX,.45,(0,165,255),2)
                    cv2.putText(anotado,'YOLO: boxes anteriores (skip)',(8,54),
                                cv2.FONT_HERSHEY_SIMPLEX,.55,(0,200,255),2)
            total_ms=(perf_counter()-t0)*1000
            vel=velocidade_agente(env)
            registro={'step':idx,'episodio':episodio,'velocidade_env':vel,
                      'desvio_px':desvio_px,'curvatura_raio_px':curvatura_px,
                      'sim_ms':round(sim_ms,4),'faixas_ms':round(faixa_ms,4),
                      'yolo_ms':round(yolo_ms,4),'total_ms':round(total_ms,4),
                      'fps_instantaneo':round(1000/max(total_ms,1e-8),3),
                      'deteccao_atualizada':atualizou}
            for n in ['car','person','bicycle','motorcycle','bus','truck','traffic light','stop sign']:
                registro[n]=int(ultima_contagem.get(n,0)) if modelo is not None else 0
            linhas.append(registro)
            if writer:
                cv2.putText(anotado,f'Step {idx+1}/{steps} | {1000/max(total_ms,1):.1f} FPS',
                            (8,frame.shape[0]-12),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),2)
                writer.write(anotado)
            if idx in (0,steps//2,steps-1): salvar(f'step_{idx:04d}.png',anotado)
            if (idx+1)%100==0:print(f'Step {idx+1}/{steps}, fps instantaneo {1000/max(total_ms,1e-8):.1f}')
            if done or trunc:
                episodio+=1
                # Um novo episodio continua a coleta; o total segue EXATO.
                if idx<steps-1:obs,info=env.reset(seed=seed+episodio)
    finally:
        if writer:writer.release()
        env.close()
    perfil=f'{"yolo" if yolo else "sem_yolo"}_skip{skip}_img{imgsz}_{"faixas" if detectar_faixa else "sem_faixas"}'
    csvp=salvar_csv(f'metricas_{steps}_steps_{perfil}.csv',linhas)
    tempo=sum(x['total_ms'] for x in linhas)/1000
    resumo={'steps':len(linhas),'fps_efetivo':len(linhas)/max(tempo,1e-9),
            'latencia_media_ms':float(np.mean([x['total_ms'] for x in linhas])),
            'p95_ms':float(np.percentile([x['total_ms'] for x in linhas],95)),
            'faixas_validas':sum(x['desvio_px'] is not None for x in linhas),
            'updates_yolo':sum(x['deteccao_atualizada'] for x in linhas),
            'metadados':'Medidas de wall-clock no hardware que executou o teste; velocidade_env vem do MetaDrive.'}
    (RESULTADOS/f'resumo_{steps}_{perfil}.json').write_text(json.dumps(resumo,indent=2,ensure_ascii=False),encoding='utf8')
    print('RESUMO REAL:',resumo)
    return linhas,resumo,csvp

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
    linhas,res,csvp=executar_episodios(a.steps or 20,a.render,detectar_faixa=False,yolo=True,skip=1,imgsz=640)
    print('Experimento concluido, dados reais no CSV.')
