## NP size measurement code and related distribution analysis

This code is for NP size measurement from Hitachi HT 7700 TEM images. 'NP_size.py' is first used to extract size information of measured nanoparticles, and subsequently 'Histogram.ipynb' is used to generate the histogram plot from the extracted size information.

### Installation

To use the 'NP_size.py', put images in a folder starting with name "sphere/cube/rod/star" and run the code. The program will open all the jpg/tif files, then label and measures the sizes of the particles. The particles measured will be highlighted by green contours with index numbers. Press 'n' on keyboard to open the next image. Press 'Esc' to quit. After finishing, all the labelled NP images and the corresponding extracted measurements will be saved in newly generated files.

Additional controls for 'NP_size.py':
1. Use the trackbar on top to change the size of the binding shapes for more precise measurements.
2. Enlarge part of the image by dragging on the original image.
3. Change the parameters on top of the codes.

To generate plot that visualizes the distribution of the length, width, and the aspect ratio of the measured nanoparticles, run 'Histogram.ipynb' in jupyter notebook.

Demo test: the 'long_NR.txt' file
