"""Apoio aos 36 exemplos: caminhos, figuras, vídeo e medição.
Instalação: python -m pip install -r requirements.txt
As operações estudadas permanecem nos exemplos; este módulo evita repetir
código de arquivos, gráficos e estatísticas. Não usa GPU nem rede.
"""
from pathlib import Path
import argparse
import csv
import json
import os
import platform
import sys
from uuid import uuid4
from time import perf_counter_ns
import cv2
import numpy as np
import matplotlib.pyplot as plt

RAIZ = Path(__file__).resolve().parents[1]

def iniciar(arquivo):
    p = argparse.ArgumentParser(description=Path(arquivo).stem)
    p.add_argument('--imagem', type=Path, default=RAIZ/'dados/frontal_kitti.png')
    p.add_argument('--video', type=Path, default=RAIZ/'dados/direcao_udacity.mp4')
    p.add_argument('--saida', type=Path, default=RAIZ/'resultados')
    p.add_argument('--sem-gui', action='store_true', help='Salva figuras sem abrir janelas.')
    p.add_argument('--mostrar', action='store_true', help='Abre feed cv2.imshow no desktop.')
    p.add_argument('--quadros', type=int, default=120, help='Limite total, incluindo aquecimento.')
    p.add_argument('--aquecimento', type=int, default=10)
    p.add_argument('--fps-alvo', type=float, default=30.0)
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--d', type=int, default=5)
    p.add_argument('--sigma-cor', type=float, default=50.0)
    p.add_argument('--sigma-espaco', type=float, default=50.0)
    p.add_argument('--seed', type=int, default=42)
    a = p.parse_args()
    if a.quadros <= a.aquecimento or a.aquecimento < 0:
        p.error('--quadros deve ser maior que --aquecimento; aquecimento >= 0.')
    if a.fps_alvo <= 0 or a.threads < 1 or a.d < 1:
        p.error('FPS, threads e diâmetro devem ser positivos.')
    cv2.setNumThreads(a.threads)
    destino = a.saida/Path(arquivo).stem
    destino.mkdir(parents=True, exist_ok=True)
    return a, destino

def carregar_imagem(caminho):
    caminho = Path(caminho).expanduser().resolve()
    if not caminho.is_file():
        raise FileNotFoundError(f'Imagem não encontrada: {caminho}. Use --imagem CAMINHO.')
    # imdecode também permite caminhos com acentos no Windows.
    img = cv2.imdecode(np.fromfile(caminho, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f'Arquivo não decodificável como imagem: {caminho}')
    return img

def abrir_video(caminho):
    caminho = Path(caminho).expanduser().resolve()
    cap = cv2.VideoCapture(str(caminho))
    if not cap.isOpened():
        raise OSError(f'Vídeo não aberto: {caminho}. Confira caminho e codec.')
    return cap

def metadados(img, caminho=None):
    h, w = img.shape[:2]
    m = {'resolucao_largura_altura':[w,h], 'shape':list(img.shape),
         'dtype':str(img.dtype), 'canais':1 if img.ndim == 2 else img.shape[2],
         'bytes_pixels':int(img.nbytes), 'elementos':int(img.size),
         'bytes_por_elemento':int(img.itemsize)}
    if caminho is not None:
        m['bytes_arquivo'] = Path(caminho).stat().st_size
    print(json.dumps(m, ensure_ascii=False, indent=2))
    return m

def cinza(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def quadrantes(plano):
    h,w = plano.shape[:2]; my,mx = h//2,w//2
    if h < 2 or w < 2:
        raise ValueError('A imagem precisa ter pelo menos 2 x 2 pixels.')
    # O segundo intervalo inclui o resto em dimensões ímpares.
    return {'superior_esquerdo':plano[:my,:mx],
            'superior_direito':plano[:my,mx:],
            'inferior_esquerdo':plano[my:,:mx],
            'inferior_direito':plano[my:,mx:]}

def salvar_json(caminho, obj):
    Path(caminho).write_text(json.dumps(obj,ensure_ascii=False,indent=2), encoding='utf-8')

def salvar_csv(caminho, linhas):
    if not linhas:
        raise ValueError('Não há linhas para salvar.')
    with Path(caminho).open('w',newline='',encoding='utf-8') as f:
        wr = csv.DictWriter(f,fieldnames=list(linhas[0]))
        wr.writeheader(); wr.writerows(linhas)

def ler_csv(caminho):
    with Path(caminho).open(encoding='utf-8',newline='') as f:
        return list(csv.DictReader(f))

def salvar_imagem(caminho, img):
    caminho = Path(caminho)
    ok, dados = cv2.imencode(caminho.suffix, img)
    if not ok:
        raise OSError('Falha ao codificar a imagem.')
    dados.tofile(caminho)

def finalizar_figura(fig, caminho, a):
    fig.savefig(caminho,dpi=160,bbox_inches='tight',facecolor='white')
    if not a.sem_gui:
        plt.show()
    plt.close(fig)

def painel(imagens, titulos, caminho, a, colunas=None):
    n = len(imagens); cols = colunas or min(n,3); rows=(n+cols-1)//cols
    fig,axs = plt.subplots(rows,cols,figsize=(4.7*cols,3.0*rows),squeeze=False)
    for ax,im,titulo in zip(axs.flat,imagens,titulos):
        if im.ndim == 2:
            ax.imshow(im,cmap='gray',vmin=0,vmax=255)
        else:
            ax.imshow(cv2.cvtColor(im,cv2.COLOR_BGR2RGB))
        ax.set_title(titulo,fontsize=11);ax.axis('off')
    for ax in list(axs.flat)[n:]: ax.axis('off')
    fig.tight_layout(); finalizar_figura(fig,caminho,a)

def histograma_comparado(antes, depois, caminho, a, canal='cinza'):
    fig,ax = plt.subplots(figsize=(8,4))
    for im,label,ls in [(antes,'Antes','-'),(depois,'Após CLAHE','--')]:
        h = np.bincount(im.ravel(),minlength=256)/im.size
        ax.plot(np.arange(256),h,label=label,color='black',linestyle=ls,linewidth=1.3)
    ax.set(xlabel=f'Intensidade do canal {canal} (0–255)',ylabel='Frequência relativa',xlim=(0,255))
    ax.legend();ax.grid(alpha=.2); fig.tight_layout();finalizar_figura(fig,caminho,a)

def clahe_colorido(img, clip=2.0, grade=(8,8)):
    lab = cv2.cvtColor(img,cv2.COLOR_BGR2LAB)
    l,aa,bb = cv2.split(lab)
    novo_l = cv2.createCLAHE(clipLimit=clip,tileGridSize=grade).apply(l)
    return cv2.cvtColor(cv2.merge([novo_l,aa,bb]),cv2.COLOR_LAB2BGR), l, novo_l

def pipeline(frame, tamanho=(640,480), d=5, sigma_cor=50, sigma_espaco=50):
    # Cronômetro monotônico de alta resolução; cada marca delimita uma etapa.
    t0=perf_counter_ns()
    reduzido=cv2.resize(frame,tamanho,interpolation=cv2.INTER_AREA)
    t1=perf_counter_ns()
    hsv=cv2.cvtColor(reduzido,cv2.COLOR_BGR2HSV)
    t2=perf_counter_ns()
    v=hsv[:,:,2]
    t3=perf_counter_ns()
    filtrado=cv2.bilateralFilter(v,d,sigma_cor,sigma_espaco)
    t4=perf_counter_ns()
    tempos={'resize_ms':(t1-t0)/1e6, 'hsv_ms':(t2-t1)/1e6,
            'extracao_v_ms':(t3-t2)/1e6,'bilateral_ms':(t4-t3)/1e6,
            'proc_ms':(t4-t0)/1e6}
    return reduzido,v,filtrado,tempos

def ambiente():
    cpu = platform.processor()
    info = Path('/proc/cpuinfo')
    if info.exists():
        for line in info.read_text(errors='replace').splitlines():
            if line.startswith('model name'):
                cpu=line.split(':',1)[1].strip();break
    modelo=''
    arq=Path('/proc/device-tree/model')
    if arq.exists(): modelo=arq.read_bytes().decode(errors='replace').strip('\x00')
    return {'sistema':platform.platform(), 'arquitetura':platform.machine(),
            'cpu':cpu or 'não identificada','modelo_placa':modelo or 'não identificado',
            'python':sys.version.split()[0], 'opencv':cv2.__version__,
            'numpy':np.__version__, 'threads_opencv':cv2.getNumThreads(),
            'gpu_utilizada':False, 'saida_terminal_tty':sys.stdout.isatty()}

def resumo_tempos(linhas, fps_alvo):
    n=len(linhas)
    if n==0: raise ValueError('Nenhum frame medido; reduza o aquecimento.')
    proc=np.array([float(r['proc_ms']) for r in linhas])
    loop=np.array([float(r['loop_ms']) for r in linhas])
    prazo=1000/fps_alvo
    return {'frames_medidos':n,'fps_alvo':fps_alvo,'prazo_ms':prazo,
            'proc_media_ms':float(proc.mean()),'proc_mediana_ms':float(np.median(proc)),
            'proc_p95_ms':float(np.percentile(proc,95)),
            'proc_p99_ms':float(np.percentile(proc,99)),
            'fps_processamento':float(1000*n/proc.sum()),
            'fps_loop':float(1000*n/loop.sum()),'loop_p95_ms':float(np.percentile(loop,95)),
            'loop_p99_ms':float(np.percentile(loop,99)),
            'frames_acima_prazo':int((loop>prazo).sum()),
            'percentual_acima_prazo':float(100*np.mean(loop>prazo)),
            'interpretacao':'FPS de capacidade offline; não é a taxa nominal da câmera.'}

def rodar_stream(a, out, gravar=True, verbose=True, tamanho=(640,480)):
    cap=abrir_video(a.video)
    fps_nominal=cap.get(cv2.CAP_PROP_FPS)
    fps_saida=fps_nominal if fps_nominal>0 else a.fps_alvo
    writer=None; linhas=[];idx=0;resumo=None
    # Gravar em arquivo novo e publicar só depois de fechar e validar.
    # Isso preserva a versão anterior se a execução for interrompida.
    caminho_feed=out/f'_gravacao_{uuid4().hex}.mp4'
    try:
        while idx < a.quadros:
            encerrar=False
            inicio=perf_counter_ns()
            ret,frame=cap.read()
            lido=perf_counter_ns()
            if not ret: break
            reduzido,v,filtrado,t= pipeline(frame,tamanho,a.d,a.sigma_cor,a.sigma_espaco)
            aquecendo=idx<a.aquecimento
            if not aquecendo:
                if gravar or a.mostrar:
                    # Original e V filtrado lado a lado; a cor não é reconstruída.
                    view=np.hstack([reduzido,cv2.cvtColor(filtrado,cv2.COLOR_GRAY2BGR)])
                    cv2.rectangle(view,(0,0),(view.shape[1],48),(0,0,0),-1)
                    texto=f'Frame {idx}: {t["proc_ms"]:.2f} ms | FPS proc {1000/max(t["proc_ms"],1e-9):.1f}'
                    cv2.putText(view,texto,(12,30),cv2.FONT_HERSHEY_SIMPLEX,.62,(255,255,255),1,cv2.LINE_AA)
                    if gravar:
                        if writer is None:
                            writer=cv2.VideoWriter(str(caminho_feed),cv2.VideoWriter_fourcc(*'mp4v'),fps_saida,
                                                   (view.shape[1],view.shape[0]))
                            if not writer.isOpened(): raise OSError('VideoWriter não abriu. Verifique suporte a MP4.')
                        writer.write(view)
                    if a.mostrar:
                        cv2.imshow('Original | V bilateral',view)
                        encerrar=(cv2.waitKey(1)&0xFF == ord('q'))
                if verbose:
                    print(f'frame={idx:04d} | processamento={t["proc_ms"]:.3f} ms | FPS_proc={1000/max(t["proc_ms"],1e-9):.2f}')
                fim=perf_counter_ns()
                linhas.append({'frame':idx,'leitura_ms':(lido-inicio)/1e6,**t,
                               'loop_ms':(fim-inicio)/1e6})
                if len(linhas)==1:
                    # Fora do cronômetro: fotografia do feed para relatório e Colab.
                    painel([reduzido,filtrado],['Original 640 × 480','Canal V após bilateral'],out/'primeiro_feed.png',a)
            idx+=1
            if encerrar: break
    finally:
        cap.release()
        if writer is not None: writer.release()
        if a.mostrar: cv2.destroyAllWindows()
    conferido=False
    if gravar and writer is not None:
        # Fora do cronômetro: confirmar o encerramento do container MP4.
        leitor=cv2.VideoCapture(str(caminho_feed))
        try:
            ok_primeiro,primeiro=leitor.read()
            total=int(leitor.get(cv2.CAP_PROP_FRAME_COUNT))
            ok_ultimo=False
            if total>0:
                leitor.set(cv2.CAP_PROP_POS_FRAMES,total-1)
                ok_ultimo,ultimo=leitor.read()
            if not ok_primeiro or not ok_ultimo or total!=len(linhas):
                raise OSError('O MP4 não foi finalizado corretamente; confira codec e espaço em disco.')
            conferido=True
        finally:
            leitor.release()
        caminho_feed.replace(out/'feed_processado.mp4')
    salvar_csv(out/'tempos_frames.csv',linhas)
    resumo=resumo_tempos(linhas,a.fps_alvo)
    resumo.update({'fps_nominal_video':fps_nominal,'aquecimento_descartado':min(idx,a.aquecimento),
                   'frames_lidos':idx,'tamanho':list(tamanho),'d':a.d,
                   'sigma_cor':a.sigma_cor,'sigma_espaco':a.sigma_espaco,
                   'feed_gravado':gravar,'ambiente':ambiente(),
                   'feed_primeiro_ultimo_conferidos':conferido,
                   'escopo_loop':'leitura + pipeline + montagem/gravação/exibição se habilitadas + impressão; exclui inicialização, figuras, CSV, flush final e validação do MP4'})
    salvar_json(out/'resumo.json',resumo)
    print(json.dumps(resumo,ensure_ascii=False,indent=2))
    return linhas,resumo

def item_a(a,out):
    img=carregar_imagem(a.imagem)
    m=metadados(img,a.imagem)
    y=cinza(img)
    brilho={k:float(r.mean()) for k,r in quadrantes(y).items()}
    print('Brilho médio (cinza, 0–255):',brilho)
    clahe=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
    eq=clahe.apply(y)
    # Sombras reduzem o contraste faixa/asfalto e obstáculo/fundo; limiares
    # fixos podem falhar. Saturação apaga detalhes. CLAHE amplia contraste
    # local, mas também pode reforçar ruído: verificar a etapa de percepção.
    painel([y,eq],['Original em cinza','Após CLAHE'],out/'original_clahe.png',a)
    histograma_comparado(y,eq,out/'histogramas.png',a)
    colorida,l,leq=clahe_colorido(img)
    painel([img,colorida],['Original colorida','CLAHE no canal L de Lab'],out/'clahe_colorido.png',a)
    salvar_imagem(out/'equalizada_cinza.png',eq)
    salvar_imagem(out/'equalizada_colorida.png',colorida)
    rel={'imagem':str(a.imagem),'metadados':m,'definicao_brilho':'média do canal cinza produzido por BGR2GRAY',
         'brilho_quadrantes':brilho,'clahe':{'clipLimit':2.0,'tileGridSize':[8,8]}}
    salvar_json(out/'item_a.json',rel)
    return rel
