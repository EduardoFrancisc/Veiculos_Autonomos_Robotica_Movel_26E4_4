# Fundamentos OpenCV e representação de imagens em sistemas autônomos
Prof. Vitor Amadeu Souza · 36 exemplos incrementais

## Comece aqui
1. Extraia o ZIP mantendo a pasta e seus arquivos juntos.
2. No VS Code, abra a pasta do pacote e selecione um interpretador Python 3.10+.
3. Crie um ambiente virtual e instale os requisitos conforme o guia abaixo.
4. Execute os exemplos em ordem. Cada exemplo lê sua própria entrada e não exige
   executar o anterior. `apoio.py` concentra apenas funções comuns de I/O e medição.
5. As saídas são salvas em `resultados/NOME_DO_EXEMPLO/`.

### Windows / PowerShell
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python exemplos/02_leitura_metadados.py
python exemplos/29_item_b_stream.py --mostrar --sem-gui
```
Se a política do PowerShell impedir a ativação, use o interpretador diretamente:
```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe exemplos/29_item_b_stream.py --sem-gui
```

### Ubuntu / Linux / Raspberry Pi OS
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python exemplos/28_item_a_completo.py
python exemplos/29_item_b_stream.py --mostrar --sem-gui
```
Se `venv` não estiver disponível no sistema Debian/Ubuntu:
`sudo apt install python3-venv`.
No Raspberry Pi, prefira SO de 64 bits e Python compatível. Se não existir wheel
para sua versão/arquitetura, use os pacotes da distribuição em ambiente separado:
```bash
sudo apt update
sudo apt install python3-opencv python3-numpy python3-matplotlib python3-venv
python3 -m venv --system-site-packages .venv-pi
source .venv-pi/bin/activate
python exemplos/35_raspberry_pi.py --sem-gui
```
Não misture OpenCV do apt e do pip no mesmo ambiente; registre as versões usadas.
Em execução sem desktop, instale `requirements-colab.txt` em vez de `requirements.txt`
e use `--sem-gui`; não use `--mostrar` com OpenCV headless.

### Google Colab
1. Abra `Caderno_36_Exemplos_OpenCV.ipynb` no Colab via **Arquivo → Fazer upload de notebook**.
2. Use runtime de CPU; GPU não é necessária para estes exemplos.
3. Execute a primeira célula e envie o ZIP quando solicitado.
4. Execute a instalação e os exemplos. Cada célula contém o código completo.
5. Figuras aparecem abaixo das células. O feed é salvo e convertido para H.264
   quando FFmpeg está disponível. `cv2.imshow` é apenas para desktop.

## Executar com seus próprios dados
```bash
python exemplos/28_item_a_completo.py --imagem "dados/minha_rua.jpg"
python exemplos/29_item_b_stream.py --video "dados/meu_video.mp4" --sem-gui --quadros 300 --aquecimento 20
python exemplos/36_projeto_integrador.py --imagem "dados/minha_rua.jpg" --video "dados/meu_video.mp4" --sem-gui
```
Arquivos padrão: `dados/frontal_kitti.png` e `dados/direcao_udacity.mp4`.
Origem e licenças: `documentacao/FONTES_E_LICENCAS.md`.

## Parâmetros comuns
`--sem-gui`: salva figuras sem abrir janelas; `--mostrar`: abre feed no desktop;
`--saida PASTA`: destino das saídas; `--quadros N`: limite total de frames;
`--aquecimento N`: frames iniciais fora das estatísticas; `--threads N`: threads do OpenCV;
`--fps-alvo N`: meta para calcular o prazo; `--d N`, `--sigma-cor N`, `--sigma-espaco N`:
parâmetros do bilateral; `--seed N`: semente dos exemplos com ruído.
O limite precisa ser maior que o aquecimento. Um vídeo curto limita o número de amostras.

## Progressão
| Nº | Conceito | Arquivo em exemplos/ |
|---|---|---|
| 01 | Conferir o ambiente de execução | `01_ambiente.py` |
| 02 | Carregar a imagem e imprimir os metadados | `02_leitura_metadados.py` |
| 03 | Entender pixel, linha, coluna e recorte | `03_pixels_coordenadas.py` |
| 04 | Exibir as cores corretamente | `04_bgr_rgb.py` |
| 05 | Separar os canais B, G e R | `05_canais_bgr.py` |
| 06 | Distinguir cinza e canal V | `06_cinza_e_v.py` |
| 07 | Evitar estouro na aritmética uint8 | `07_uint8_saturacao.py` |
| 08 | Normalizar e voltar a uint8 | `08_normalizacao.py` |
| 09 | Redimensionar e preservar proporção | `09_redimensionamento.py` |
| 10 | Medir o brilho por quadrante | `10_brilho_quadrantes.py` |
| 11 | Inspecionar regiões escuras e saturadas | `11_qualidade_basica.py` |
| 12 | Construir um histograma de 256 níveis | `12_histograma.py` |
| 13 | Interpretar a distribuição acumulada | `13_cdf.py` |
| 14 | Comparar equalização global e original | `14_equalizacao_global.py` |
| 15 | Aplicar CLAHE à imagem em cinza | `15_clahe_cinza.py` |
| 16 | Equalizar o canal L de Lab | `16_clahe_colorido.py` |
| 17 | Investigar o limite de contraste do CLAHE | `17_clahe_cliplimit.py` |
| 18 | Investigar a grade do CLAHE | `18_clahe_grade.py` |
| 19 | Simular uma sombra na pista | `19_iluminacao_sombra.py` |
| 20 | Observar efeitos da compressão JPEG | `20_compressao_jpeg.py` |
| 21 | Adicionar ruído com semente conhecida | `21_ruido_gaussiano.py` |
| 22 | Comparar Gaussian, mediana e bilateral | `22_comparar_filtros.py` |
| 23 | Interpretar H, S e V | `23_hsv.py` |
| 24 | Filtrar V preservando transições | `24_bilateral_v.py` |
| 25 | Medir o custo do diâmetro do bilateral | `25_custo_bilateral.py` |
| 26 | Abrir o vídeo e ler o primeiro frame | `26_abrir_video.py` |
| 27 | Ler frames sequencialmente | `27_iterar_frames.py` |
| 28 | Consolidar todas as saídas do item A | `28_item_a_completo.py` |
| 29 | Executar o pipeline completo em vídeo | `29_item_b_stream.py` |
| 30 | Encontrar a etapa que consome mais tempo | `30_benchmark_etapas.py` |
| 31 | Comparar resolução e diâmetro | `31_comparar_configuracoes.py` |
| 32 | Relacionar contraste e bordas da pista | `32_bordas_faixas.py` |
| 33 | Examinar contornos sem atribuir classes | `33_contornos_obstaculos.py` |
| 34 | Gerar uma discussão baseada no CSV | `34_relatorio_fps.py` |
| 35 | Registrar condições da plataforma embarcada | `35_raspberry_pi.py` |
| 36 | Executar os itens A e B e organizar a entrega | `36_projeto_integrador.py` |

## Correspondência com a tarefa
| Exigência | Exemplos de referência | Evidência |
|---|---|---|
| Imagem frontal e metadados | 02, 28, 36 | Terminal e JSON |
| Brilho em quatro quadrantes | 10, 28, 36 | CSV/JSON e painel |
| CLAHE e histogramas num único plot | 15, 16, 28, 36 | histogramas.png |
| Original/equalizada lado a lado | 15, 16, 28, 36 | PNG e exibição Matplotlib |
| Comentário técnico sobre percepção | 15, 28, 36 | Comentários no código |
| Vídeo → 640×480 → HSV → V → bilateral | 24, 29, 36 | Feed MP4 e primeiro_feed.png |
| Tempo por frame e FPS | 29, 30, 34, 36 | Terminal, CSV e resumo.json |
| Discussão de Raspberry Pi 4 | 34, 35, 36 | Relatório e condições do hardware |

## Como interpretar FPS corretamente
`fps_processamento = 1000 × N / soma(proc_ms)`; não calcule a média de `1000/t_i`.
`fps_loop = 1000 × N / soma(loop_ms)` usa leitura e pipeline, além de gravação,
montagem/exibição e impressão quando habilitadas. Figuras, CSV, inicialização
e flush final/validação do MP4 ficam fora desse escopo. O CSV mantém os tempos por frame e etapa.
O arquivo de vídeo é lido tão rápido quanto possível: esse número é capacidade
offline, não a taxa nominal da câmera ou uma prova de latência de captura real.
Compare percentis e frames acima do prazo. Para 30 FPS, prazo = 33,33 ms.

## Resultados incluídos
`resultados_demonstracao/` contém saídas realmente executadas no ambiente de
preparação do material. O hardware está identificado nos JSON. Esses números
não são medições do Raspberry Pi 4 e não representam um detector de obstáculos.
O relatório/modelo orienta a repetição na placa e a redação baseada em evidências.

## Problemas comuns
- `ModuleNotFoundError`: confirme o interpretador do VS Code e a instalação nesse ambiente.
- Arquivo não encontrado: mantenha a estrutura do ZIP ou use caminhos completos entre aspas.
- Janela OpenCV indisponível: use `--sem-gui`, abra o vídeo salvo ou use OpenCV desktop.
- Vídeo não abre: confira arquivo, codec e `cap.isOpened`; teste um MP4 H.264 no VLC.
- Feed MP4 sem reprodução no navegador: converter com FFmpeg ou abrir no VLC.
- Lentidão: medir as etapas; testar d menor e SO/threads; a entrega B continua 640×480.

O código autoral pode ser adaptado para aula com atribuição. As mídias de terceiros
têm suas próprias licenças, descritas em FONTES_E_LICENCAS.md.
