After the digits have been identified by making use of contour detection techniques, we need to form bounding boxes over them to be used in the seven-segment digit recognition algorithm.

After contour detection two steps need to be followed:
1) These contours need to be sorted according to their x-coordinate of appearance in the image
2) The user would have to follow an iterative procedure to impose appropriate height and width constraints on the bounding boxes so that the digits can be enclosed
