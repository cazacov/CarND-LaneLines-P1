## Submission

### Pipline description
My pipeline consisted of 11 steps.
1. Input image is copied and converted to grayscale,
2. Then the image is smotthed with Gaussian blur. Kernel size 5.
3. Edges are found using Canny function from the OpenCV library.
4. All points outside the region of interest are set to black. As region of interest, I use a trapezium with the height of 50% of the image.
5. I use Hough algorithm to find lines on the masked image form the previous step. During my tests, I noticed that images can have different resolution. Test images are all 960x540, while challenge video has resolution 1280x720. Hard-coding minimum line length would be a bad idea, so I decided to calculate it as image_width / 5.
6. Lines with a low slope (< 20 degrees) cannot be lane lines, so I discard them in function filter_low_slope.
7. I need to find exactly two lines limiting the current lane. In function find_left_right all lines are split in two sets: one with lines going from bottom left corner of the image to the middle and second set of lines going from the right bottom corner. From every set I then select the line with the biggest length.
8. It could happen that in the step above only one line or nothing at all was found. In this case, I retry steps 2-7 using another grayscaling algorithm.
9. Now we have two lines, but they do not end at the bottom of the image. The function extend_lines_to_border extends them downwards to the image border.
10. In addition, the top ends of lines jump up and down folowing the noise in the middle of the image. The function cut_lines cuts those to 90% of distance between bottom border and the line intersection point.
11. Finally, the draw_lines function is used to draw lane lines on the original image.

I decided not to change the draw_lines function because I expect function to do exactly what its name says and do not contain any hidden non-obvious logic. The decision what lines to choose is made in a separate function called find_left_right. Instead of draw_lines you can draw a polygon with my function draw_poly. I personally find polygon more pretty than lines.


### Potental shortcommings
The region of interest is fixed in the bottom 50% of the image. That could be a problem if the car goes downhills and line intersection point will be somewhere close to the top of the image.

The standard grayscaling algorithm does not provide contrast image on the challenge video. Because the wrong lines are detected mostly because of the noise from the green leaves on the blue sky background, my algorithm tries to switch to R channel of RGB that is not so noisy. If that does not help it also tries to find lines in H channel of the HLS color space. That helps a little, but not much.

### Possible improvements
High contrast grayscale image is critical for the line detection with Canny edge detection and Hough transformation. Some image pre-processing to increase contrast could help.

My code works well on test images, but cannot find lane lines on some video frames. It does not have any memory and analyzes every frame as a separate image. Adding some time smoothing in line coordinates could help to fill small gaps. Something like taking moving average of last 10 frames will probably provide good results as long as lines were found on  6 of 10 images or more.
