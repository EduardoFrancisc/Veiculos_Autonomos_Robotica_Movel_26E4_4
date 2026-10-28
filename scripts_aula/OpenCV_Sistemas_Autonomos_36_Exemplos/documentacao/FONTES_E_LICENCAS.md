# Fontes técnicas, dados e licenças

Consultas técnicas conferidas em 6 de outubro de 2026. Os exemplos foram
executados com Python 3.12.14, OpenCV 4.12.0, NumPy 2.2.6 e Matplotlib 3.10.8.
As referências da documentação usam a série 4.x e podem apontar para uma
versão posterior. O pacote registra as versões de cada execução.

## Imagem frontal KITTI
- Arquivo incluído: `dados/frontal_kitti.png` (1.224 × 370 pixels).
- Identificação: amostra `training/image_2/000000.png` do benchmark de objetos,
  apresentada no repositório dtc-KITTI-For-Beginners.
- Fonte original: https://www.cvlibs.net/datasets/kitti/
- Espelho da amostra: https://github.com/dtczhl/dtc-KITTI-For-Beginners
- URL do arquivo efetivamente obtido:
  https://raw.githubusercontent.com/dtczhl/dtc-KITTI-For-Beginners/master/img/image_000000.png
- Autores do dataset: Andreas Geiger, Philip Lenz, Raquel Urtasun; KIT e TTI-C.
- Licença das imagens KITTI: Creative Commons Attribution-NonCommercial-ShareAlike 3.0.
  https://creativecommons.org/licenses/by-nc-sa/3.0/
- As figuras derivadas dessa imagem mantêm a atribuição e a licença KITTI.
  A licença do código do espelho não substitui a licença do dataset.
- Referência: Geiger, A.; Lenz, P.; Urtasun, R. Are we ready for Autonomous Driving?
  The KITTI Vision Benchmark Suite. CVPR, 2012.

## Vídeo real de direção Udacity
- Arquivo incluído: `dados/direcao_udacity.mp4`.
- Arquivo de origem: `test_videos/solidWhiteRight.mp4`, projeto CarND-LaneLines-P1.
- Repositório: https://github.com/udacity/CarND-LaneLines-P1
- URL: https://raw.githubusercontent.com/udacity/CarND-LaneLines-P1/master/test_videos/solidWhiteRight.mp4
- Licença do projeto: MIT; texto preservado em `dados/LICENCA_UDACITY.txt`.
- `dados/pista_udacity.png` é o primeiro frame extraído desse vídeo.
- O vídeo tem 221 frames, resolução 960×540 e FPS nominal 25 na leitura usada.

## Transformações e resultados
- Exemplo 19: sombra artificial, fator 0,30 em uma região da imagem real.
- Exemplos 21 e 22: ruído artificial gaussiano, semente 42.
- Exemplo 20: recodificação JPEG da mesma entrada em qualidades diferentes.
- Figuras de matrizes, histograma e degrau: construções numéricas para ensino.
- CSV e resumos de desempenho: medições reais do ambiente identificado nos JSON.
  Não são resultados do Raspberry Pi 4 nem de um detector de faixas/obstáculos.

## Documentação e fonte primária
1. História do OpenCV: https://opencv.org/anniversary/
2. Operações básicas: https://docs.opencv.org/4.x/d3/df2/tutorial_py_basic_ops.html
3. Imagens e codecs: https://docs.opencv.org/4.x/d4/da8/group__imgcodecs.html
4. Memória NumPy: https://numpy.org/doc/stable/reference/generated/numpy.ndarray.nbytes.html
5. Conversões de cor: https://docs.opencv.org/4.x/de/d25/imgproc_color_conversions.html
6. Histogramas/equalização: https://docs.opencv.org/4.x/d5/daf/tutorial_py_histogram_equalization.html
7. CLAHE: https://docs.opencv.org/4.x/d6/db6/classcv_1_1CLAHE.html
8. Filtros: https://docs.opencv.org/4.x/d4/d13/tutorial_py_filtering.html
9. Tomasi, C.; Manduchi, R. Bilateral Filtering for Gray and Color Images. ICCV, 1998.
   Fonte dos autores: https://users.soe.ucsc.edu/~manduchi/Papers/ICCV98.pdf
10. Vídeo: https://docs.opencv.org/4.x/dd/d43/tutorial_py_video_display.html
11. Raspberry Pi 4: https://www.raspberrypi.com/products/raspberry-pi-4-model-b/specifications/
12. Geometria: https://docs.opencv.org/4.x/da/d6e/tutorial_py_geometric_transformations.html
13. Canny: https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html

As mídias e suas derivações seguem as licenças acima. Os exemplos foram
escritos para este material e podem ser adaptados em aula, mantendo as
atribuições das mídias e as referências técnicas.
