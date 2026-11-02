"""Exemplo 34 — Gerar uma discussão baseada no CSV

COMENTÁRIO GERAL
Executa o fluxo, lê novamente o CSV e calcula FPS como N/tempo total, percentis e percentual acima do prazo. Gera um texto com os dados reais desta execução.

O QUE INSTALAR
Python 3.10 ou superior; OpenCV, NumPy e Matplotlib.
No VS Code: python -m pip install -r requirements.txt
No Colab: executar a célula de instalação do caderno incluído no pacote.
Bibliotecas da biblioteca padrão não precisam de instalação separada.

COMO EXECUTAR
Na pasta do pacote: python exemplos/34_relatorio_fps.py
Sem janelas: acrescentar --sem-gui. Dados próprios: --imagem ARQUIVO --video ARQUIVO.
Todas as saídas são salvas automaticamente em resultados/34_relatorio_fps/.

O QUE OBSERVAR
O relatório identifica o hardware e exige medição no Raspberry Pi para concluir a viabilidade nessa placa.
"""
import matplotlib.pyplot as plt
from apoio import (iniciar, ler_csv, finalizar_figura, ambiente, resumo_tempos, rodar_stream)

a,out=iniciar(__file__)
rodar_stream(a,out,gravar=False,verbose=False)
linhas=ler_csv(out/'tempos_frames.csv');r=resumo_tempos(linhas,a.fps_alvo);hw=ambiente()
placa='raspberry pi 4' in hw['modelo_placa'].lower()
texto=f"""Relatório de desempenho — execução local

Hardware: {hw['cpu']}; placa: {hw['modelo_placa']}; arquitetura: {hw['arquitetura']}.
OpenCV: {hw['opencv']}; threads: {hw['threads_opencv']}; GPU: não utilizada.
Pipeline: resize 640×480, BGR→HSV, V e bilateral d={a.d}.
Frames medidos: {r['frames_medidos']}; aquecimento solicitado: {a.aquecimento}.
Média do processamento: {r['proc_media_ms']:.3f} ms; FPS do processamento: {r['fps_processamento']:.2f}.
FPS do loop: {r['fps_loop']:.2f}; p95 do loop: {r['loop_p95_ms']:.3f} ms.
Meta: {a.fps_alvo:g} FPS; prazo: {r['prazo_ms']:.3f} ms.
Frames acima do prazo: {r['frames_acima_prazo']} ({r['percentual_acima_prazo']:.2f}%).

O resultado descreve a capacidade offline deste hardware sob este protocolo.
Identificação como Raspberry Pi 4: {'sim' if placa else 'não confirmada'}.
Para avaliar essa placa, repetir a medição nela, registrar clock, temperatura,
resfriamento, SO, backend de captura e versões, e usar um teste prolongado.
Mesmo com FPS médio suficiente, verificar p95/p99 e o custo das etapas
seguintes de percepção. O teste não mede detector de faixas ou obstáculos.
"""
(out/'relatorio.txt').write_text(texto,encoding='utf-8');print(texto)
fig,ax=plt.subplots(figsize=(8,4));ms=[float(x['loop_ms']) for x in linhas]
ax.hist(ms,bins=20,color='0.65',edgecolor='white');ax.axvline(r['prazo_ms'],color='black',linestyle='--',label='Prazo')
ax.set(xlabel='Tempo do loop (ms)',ylabel='Frames');ax.legend()
finalizar_figura(fig,out/'latencia_histograma.png',a)
