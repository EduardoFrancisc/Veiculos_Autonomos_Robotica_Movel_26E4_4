"""Exemplo 36 — Executar os itens A e B e organizar a entrega

COMENTÁRIO GERAL
Reúne a inspeção da imagem e o processamento do vídeo em pastas separadas, produz métricas e um roteiro preenchido com os resultados medidos. É a referência final de execução.

O QUE INSTALAR
Python 3.10 ou superior; OpenCV, NumPy e Matplotlib.
No VS Code: python -m pip install -r requirements.txt
No Colab: executar a célula de instalação do caderno incluído no pacote.
Bibliotecas da biblioteca padrão não precisam de instalação separada.

COMO EXECUTAR
Na pasta do pacote: python exemplos/36_projeto_integrador.py
Sem janelas: acrescentar --sem-gui. Dados próprios: --imagem ARQUIVO --video ARQUIVO.
Todas as saídas são salvas automaticamente em resultados/36_projeto_integrador/.

O QUE OBSERVAR
A conclusão sobre Raspberry Pi exige dados obtidos nessa placa. Anexe mídia, plots, CSV e uma discussão técnica.
"""
from apoio import (iniciar, rodar_stream, item_a)

a,out=iniciar(__file__)
pa=out/'item_a';pb=out/'item_b';pa.mkdir(exist_ok=True);pb.mkdir(exist_ok=True)
ra=item_a(a,pa)
linhas,rb=rodar_stream(a,pb,gravar=True,verbose=True)
# Comentário técnico: sombra reduz contraste local; brilho excessivo satura
# detalhes; JPEG pode alterar bordas. Ajustar CLAHE e bilateral exige observar
# a imagem e validar a percepção. Nenhum desses filtros classifica obstáculos.
hw=rb['ambiente'];condicao=rb['fps_loop']>=a.fps_alvo
texto=f"""# Relatório do projeto integrador

## 1. Dados e ambiente
Imagem: {a.imagem}. Vídeo: {a.video}.
Hardware: {hw['cpu']}; placa: {hw['modelo_placa']}; SO: {hw['sistema']}.
OpenCV {hw['opencv']}; Python {hw['python']}; threads {hw['threads_opencv']}.
Registrar a origem/licença das mídias no relatório. O pacote fornece KITTI e Udacity.

## 2. Item A
Resolução (W,H): {ra['metadados']['resolucao_largura_altura']}.
dtype: {ra['metadados']['dtype']}; canais: {ra['metadados']['canais']}.
Bytes da matriz: {ra['metadados']['bytes_pixels']}; bytes do arquivo: {ra['metadados']['bytes_arquivo']}.
Brilho dos quadrantes (cinza): {ra['brilho_quadrantes']}.
CLAHE: clipLimit=2 e tileGridSize=(8,8). Figuras em item_a/.
Discutir onde o contraste melhorou visualmente e onde o ruído foi reforçado.

## 3. Item B
Pipeline: resize 640×480 → HSV → V → bilateral d={a.d}, sigmas {a.sigma_cor}/{a.sigma_espaco}.
Frames medidos: {rb['frames_medidos']}; FPS nominal do vídeo: {rb['fps_nominal_video']:.2f}.
Média do processamento: {rb['proc_media_ms']:.3f} ms; FPS processamento: {rb['fps_processamento']:.2f}.
FPS do loop: {rb['fps_loop']:.2f}; p95/p99 do loop: {rb['loop_p95_ms']:.3f}/{rb['loop_p99_ms']:.3f} ms.
Meta: {a.fps_alvo:g} FPS; prazo: {rb['prazo_ms']:.3f} ms.
Percentual de frames acima do prazo: {rb['percentual_acima_prazo']:.2f}%.
A taxa média offline do loop {'atingiu' if condicao else 'não atingiu'} a meta nesta execução.
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
"""
(out/'relatorio.md').write_text(texto,encoding='utf-8');print(texto)
