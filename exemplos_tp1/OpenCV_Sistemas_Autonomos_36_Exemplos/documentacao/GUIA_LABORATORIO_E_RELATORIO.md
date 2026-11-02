# Guia do laboratório e do relatório

## Sugestão de condução da aula
1. Pedir que os alunos descrevam uma região clara e uma região escura antes de
   usar filtros. Executar 02 e 10 e relacionar os números com a fotografia.
2. Comparar 14 e 15. Não escolher CLAHE apenas pela aparência global: observar
   o pedestre, a sombra e a textura do pavimento. Experimentar 17 e 18.
3. Construir a pipeline do item B com 23 e 24. Só então executar 26, 27 e 29.
4. Ler CSV/JSON e comparar escopos com 30 e 34. Fazer o aluno identificar o
   hardware medido antes de escrever uma conclusão sobre o Raspberry Pi.
5. Consolidar com 36. Solicitar uma descrição concreta dos resultados e o
   vínculo entre qualidade visual e percepção de faixas/obstáculos.

## Item A — roteiro de execução
```bash
python exemplos/28_item_a_completo.py --imagem "dados/frontal_kitti.png"
```
Conferir no terminal resolução (W,H), dtype, canais, bytes da matriz e arquivo.
Conferir as médias dos quatro quadrantes, calculadas no canal cinza.
Abrir `original_clahe.png`, `histogramas.png` e a extensão colorida.
Verificar o comentário técnico nos exemplos 15, 28, 36 e em `apoio.py`.

Perguntas para a discussão:
- Qual região ganhou contraste e onde isso é visível?
- Alguma textura/ruído também foi reforçada?
- A diferença entre quadrantes pode ser explicada também pela composição da cena?
- Os histogramas comparam o mesmo canal, intervalo e normalização?

## Item B — roteiro de execução
```bash
python exemplos/29_item_b_stream.py --video "dados/direcao_udacity.mp4" --quadros 300 --aquecimento 10 --sem-gui
```
No desktop com OpenCV GUI: acrescentar `--mostrar`.
No Colab: usar o feed salvo e sua conversão para H.264; não usar cv2.imshow.
Conferir a ordem resize 640×480, BGR→HSV, extração V e bilateral.
Guardar `tempos_frames.csv`, `resumo.json`, `primeiro_feed.png` e o MP4.
Os primeiros 10 frames saem das estatísticas. Como a mídia tem 221 frames,
o limite 300 termina no fim do vídeo e produz 211 frames medidos.

## Estrutura sugerida para o relatório
1. Objetivo e dados: origem da imagem/vídeo, condições e justificativa da escolha.
2. Ambiente: CPU/placa, RAM, SO, versões e número de threads OpenCV.
3. Item A: metadados, quatro médias, parâmetros CLAHE, painel e histogramas.
4. Item B: fluxograma, parâmetros, feed, tempo médio, FPS do processamento e do
   loop, p95/p99 e percentual acima do prazo da meta.
5. Discussão: interpretar regiões e dados; relacionar sombra, saturação, JPEG e
   ruído à percepção. Explicar o que foi medido e onde.
6. Viabilidade embarcada: concluir conforme dados da plataforma-alvo ou explicar
   o experimento necessário para essa conclusão.
7. Referências e arquivos: citar mídias, documentação e anexar código/CSV.

## Modelo para a discussão de viabilidade
“Na plataforma [modelo/CPU], usando [versões], [threads] e o pipeline
640×480→HSV→V→bilateral com [parâmetros], foram medidos [N] frames após
[N] de aquecimento. O FPS do loop foi [valor], o p95 foi [valor] ms e
[percentual]% dos frames ultrapassaram o prazo de [valor] ms para [meta] FPS.
A medição incluiu [escopo]. Com base nesses dados, [conclusão compatível com
os resultados]. A margem para as etapas seguintes e a estabilidade ao longo
do tempo precisam ser consideradas.”

Se os dados são de PC/Colab, não substituir o nome por Raspberry Pi 4. A
discussão específica da placa exige a repetição nela ou resultados equivalentes
obtidos por um protocolo devidamente descrito.

## Medição prolongada no Raspberry Pi 4
- Usar a versão de 4 GB e registrar SO, CPU, clock e refrigeração.
- Usar vídeo longo ou captura contínua; a mídia curta do pacote não avalia
  estabilidade térmica por vários minutos.
- Medir com e sem visualização/gravação conforme o caso de uso e declarar o escopo.
- Registrar temperatura e throttling ao longo do experimento; repetir rodadas.
- Manter 640×480 para a entrega. Explorar resoluções alternativas apenas como extensão.
- As chamadas deste pacote usam CPU. GPU do Colab não as acelera automaticamente.

## Evidências incluídas no pacote
36 scripts foram executados com sucesso no ambiente de preparação. As células
de código do notebook também foram compiladas. A interação com a janela de um
desktop e o upload no Colab dependem do ambiente do usuário; as execuções de
referência foram feitas sem GUI. Consulte VERIFICACAO.json e os JSON de ambiente.
