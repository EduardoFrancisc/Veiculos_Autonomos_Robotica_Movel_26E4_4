# EX4 - Item B: YOLOv8n no loop do MetaDrive

## Configuracao
- MetaDriveEnv com use_render=True, image_observation=True.
- Camera frontal 640x360; inferencia YOLOv8n (COCO) a cada frame.
- Classes monitoradas: ['bicycle', 'bus', 'car', 'motorcycle', 'person', 'stop sign', 'traffic light', 'truck'].

## FPS
- Baseline sem YOLO: 7.25 FPS (137.8 ms/frame)
- YOLO ativo (skip=1, imgsz=640): 7.06 FPS (141.6 ms/frame)
- Degradacao: 2.6%

## Deteccoes (500 steps YOLO)
- Objetos totais: 533
- Updates YOLO: 500/500
- Serie temporal: serie_temporal_deteccoes_yolo_ativo.png

## Benchmarks de otimizacao (100 steps)
- skip2_img640: 9.75 FPS (skip=2, imgsz=640, lat=102.6 ms)
- skip1_img320: 9.73 FPS (skip=1, imgsz=320, lat=102.8 ms)
- skip2_img320: 10.72 FPS (skip=2, imgsz=320, lat=93.3 ms)

## Discussao
A adicao de um detector profundo no loop de simulacao aumenta a latencia
por frame e reduz o FPS efetivo. O YOLOv8n em CPU domina o custo quando
executado em todos os frames e em resolucao 640. Estrategias que minimizam
a queda de FPS: (a) frame skip (inferir a cada N frames e reutilizar as
ultimas caixas nos frames intermediarios), (b) reducao da resolucao de
inferencia (imgsz=320), (c) combinacao de skip + resolucao reduzida,
(d) execucao em GPU com batch/async e (e) detector mais leve ou pruning.
O frame skip reduz proporcionalmente o custo do YOLO, enquanto a resolucao
reduzida diminui o custo por inferencia; a combinacao oferece o melhor
compromisso, ao custo de menor fidelidade temporal/espacial das deteccoes.
