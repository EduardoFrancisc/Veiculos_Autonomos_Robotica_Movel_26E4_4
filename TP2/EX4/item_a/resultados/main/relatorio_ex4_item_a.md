# EX4 - Item A: MetaDrive em modo observacao

## Configuracao
- MetaDriveEnv com use_render=True, image_observation=True.
- Camera frontal RGB (640x360), frames capturados a cada step.
- Detector de faixas do TP1 aplicado sobre os frames simulados.
- Desvio lateral exibido em tempo real no feed sobreposto.

## Resultados
- Steps executados: 200
- Faixas validas: 90/200
- Velocidade media do agente: 7.504
- Desvio medio (px): -22.77
- CSV: metricas_200_steps.csv (200 registros)
- Feed: feed_metadrive_faixas_200.mp4
- Comparacao visual: comparacao_real_vs_simulado.png, comparacao_sintetica_vs_simulado.png
- Validade detector -> real (KITTI): False; simulado (MetaDrive): True

## Observacao
Desvio/curvatura estao no plano da imagem (pixels) por ausencia de
calibracao metrica da camera MetaDrive; valores em metros exigiriam
homografia calibrada para a cena simulada.
