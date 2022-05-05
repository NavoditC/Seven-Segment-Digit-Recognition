Task:

On some days the scale having the seven segment display might have a streak of sunlight such that the brightness across the displayed numerals varies. In order to address this issue, the following procedure is followed.

Procedure:

The user needs to select the relevant part of the scale from the image using the mouse. Alternatively, manual cropping could also be done but using the mouse for cropping seems to be more elegant!
Gamma correction is then performed to adjust the brightness such that uniformity across the digits being displayed is achieved.

Gamma correction was performed with a value of gamma equal to 2.0 for the sample image taken from the dataset. In case this problem is there on a particular day while taking the readings, the snippet of code could be placed in the original code in which images are read one after the other making use of a for loop.

Note: G0110044.JPG is a sample image from the dataset when such an issue was encountered.


