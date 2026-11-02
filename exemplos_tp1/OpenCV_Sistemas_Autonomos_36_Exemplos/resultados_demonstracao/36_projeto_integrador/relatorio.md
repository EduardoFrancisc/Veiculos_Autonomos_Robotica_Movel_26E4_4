# Relatório do projeto integrador

## 1. Dados e ambiente
Imagem: /workspace/scratch/73415f6cd7b0/OpenCV_Sistemas_Autonomos_36_Exemplos/dados/frontal_kitti.png. Vídeo: /workspace/scratch/73415f6cd7b0/OpenCV_Sistemas_Autonomos_36_Exemplos/dados/direcao_udacity.mp4.
Hardware: AMD EPYC 9V74 80-Core Processor; placa: não identificado; SO: Linux-6.18.44-x86_64-with-glibc2.39.
OpenCV 4.12.0; Python 3.12.14; threads 1.
Registrar a origem/licença das mídias no relatório. O pacote fornece KITTI e Udacity.

## 2. Item A
Resolução (W,H): [1224, 370].
dtype: uint8; canais: 3.
Bytes da matriz: 1358640; bytes do arquivo: 893783.
Brilho dos quadrantes (cinza): {'superior_esquerdo': 119.03189365836424, 'superior_direito': 44.87524288994877, 'inferior_esquerdo': 97.58974562798092, 'inferior_direito': 98.54704999116764}.
CLAHE: clipLimit=2 e tileGridSize=(8,8). Figuras em item_a/.
Discutir onde o contraste melhorou visualmente e onde o ruído foi reforçado.

## 3. Item B
Pipeline: resize 640×480 → HSV → V → bilateral d=5, sigmas 50.0/50.0.
Frames medidos: 211; FPS nominal do vídeo: 25.00.
Média do processamento: 4.941 ms; FPS processamento: 202.38.
FPS do loop: 117.84; p95/p99 do loop: 10.140/10.527 ms.
Meta: 30 FPS; prazo: 33.333 ms.
Percentual de frames acima do prazo: 0.00%.
A taxa média offline do loop atingiu a meta nesta execução.
O loop inclui gravação/impressão; veja o escopo completo em item_b/resumo.json.

## 4. Viabilidade embarcada
Esta medição pertence ao hardware descrito acima. Se não foi feita no Raspberry
Pi 4, a viabilidade nessa placa permanece por verificar. Repetir com o mesmo
vídeo, parâmetros e protocolo; registrar clock, temperatura e throttling.
Confrontar FPS do fluxo, percentis e deadlines com a meta da aplicação, deixando
margem para detecção/rastreamento. Não extrapolar os números de um PC ou Colab.

## 5. Interpretação técnica
Sombras diminuem o contraste faixa/asfalto e obstáculo/fundo; saturação perde
detalhes e compressão modifica transições. CLAHE redistribui contraste local e
bilateral reduz variações locais preservando algumas bordas. Comparar imagens
e métricas; a melhoria da percepção só pode ser afirmada após validação própria.
