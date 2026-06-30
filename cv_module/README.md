# Week 3: Shape & Color Detection

What do we need to know?
We need to understand how machines recognize geometric shapes (squares, circles) instead of faces, and how to detect dominant colors. We use "Contours" to find shape boundaries and the "K-Means" machine learning algorithm to analyze and classify color pixels.

    How do we do it?
1. Shape Detection: Convert image to grayscale, detect edges using Canny, use findContours, and approxPolyDP to count sides (3 sides = Triangle, 4 sides = Square/Rectangle, etc.).
2. Color Detection: Take every pixel as data (R, G, B) and use K-Means Clustering to group them into the top 3 most common colors.
3. Drawing: Draw Bounding Boxes around detected shapes and save the output.

    How do we apply this practically?
By writing the `shape_color_detector.py` script. When a test image is passed, it analyzes it, prints a text report, and saves a new image with drawn shapes to `test_outputs`.