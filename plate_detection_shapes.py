import cv2
import numpy as np
import matplotlib.pyplot as plt


def undistort_image(img):
    """
    With the parameters we calculated when we calibrated the camera we
    undistort our image.
    ARGS:
    - img: raw image
    RETURN:
    - img_un: undistorted image
    """

    with np.load("calib.npz") as X:
        mtx, _, dist = [X[i] for i in ("intrinsic", "extrinsic", "dist")]

    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    dst = cv2.undistort(img, mtx, dist, None, newCameraMatrix=newcameramtx)

    return dst


def get_plate(image, number):
    """
    Function to detect the plate in a
    given image
    Args:
    - image: Image with the plate
    Returns:
    - plate_img: image with the zoomed
    in plate
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # blur = cv2.bilateralFilter(gray, 11,90, 90)

    # Apply erosion
    erosion = cv2.erode(blur, None, iterations=0)

    # Canny edge detection
    edges = cv2.Canny(erosion, 50, 150)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours by area
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Initialize the picture with the plate
    plate_img = None

    for contour in contours:
        # Approximate the contour
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Get the number of vertices
        num_vertices = len(approx)

        # Classify shapes based on the number of vertices
        if num_vertices == 4:
            # Draw the shape
            cv2.drawContours(image, [approx], 0, (0, 255, 0), 2)

            # Get bounding box coordinates
            x, y, w, h = cv2.boundingRect(approx)

            # Zoom in on the detected object
            zoomed_in = image[y : y + h + 3, x : x + w + 3]

            # calculate ratio of square
            h, w = zoomed_in.shape[:2]
            ratio = w / h

            # Ratio of the used plates
            if ratio > 4.5 and ratio < 5.5:
                print(ratio)

                plate_img = zoomed_in
                cv2.imwrite(f"Plate_zoomed_{number}.jpg", plate_img)

                # # Display the zoomed-in region
                cv2.imshow("Zoomed In", zoomed_in)

                cv2.waitKey(500)
                # plt.imshow(cv2.cvtColor(zoomed_in, cv2.COLOR_BGR2RGB))
                # plt.title('Zoomed In')
                # plt.show()

    # Display the result
    cv2.imshow("Shapes Detection image 4", image)
    cv2.imwrite(f"Contours_{number}.jpg", image)

    # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # plt.title('Shapes Detection')
    # plt.show()

    cv2.waitKey(1500)
    cv2.destroyAllWindows()

    return plate_img


def classify_shape(corners):
    """
    Function to classify a shape
    given its corners. THe number is higher
    than expected because the cornerSubPix
    function returns first a True or False
    """
    if len(corners) == 5:
        # With len 5 it could be either a circle
        # or a rectangle
        circulo = False
        for i in range(1, len(corners)):
            for j in range(i + 1, len(corners)):
                if redondeo_coord(corners[i][0], corners[i][1]) == redondeo_coord(
                    corners[j][0], corners[j][1]
                ):
                    circulo = True

        if circulo:
            print("Circle")
            return "Circle"
        else:
            print("Rectangle")
            return "Rectangle"

    if len(corners) == 4:
        print("Triangle ")
        return "Triangle"

    return None


def redondeo_coord(cordx, cordy, decimales=2):
    """
    Function to round the coordinates x and y
    of a point
    """

    redondeo_cord1 = round(cordx, decimales)
    redondeo_cord2 = round(cordy, decimales)

    return (redondeo_cord1, redondeo_cord2)


def get_pattern(img, number):
    """
    Function to get the pattern in the figure.
    Args:
    - Img: image with the zoomed in plate
    Returns:
    - Pattern: list with the found pattern
    """
    pattern = []

    # Transform to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # blur = cv2.GaussianBlur(gray, (3,3), 0.5)

    # erosion = cv2.erode(blur, None, iterations=2)

    ret, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    # Find de countours
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    for i in contours:
        size = cv2.contourArea(i)
        rect = cv2.minAreaRect(i)

        if size < 10000:
            gray = np.float32(gray)
            mask = np.zeros(gray.shape, dtype="uint8")
            cv2.fillPoly(mask, [i], (255, 255, 255))

            # Find the corners
            dst = cv2.cornerHarris(mask, 5, 3, 0.04)
            ret, dst = cv2.threshold(dst, 0.1 * dst.max(), 255, 0)
            dst = np.uint8(dst)
            ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
            corners = cv2.cornerSubPix(
                gray, np.float32(centroids), (5, 5), (-1, -1), criteria
            )

            # Determine the shape given the corners
            shape = classify_shape(corners)
            if shape:
                pattern.append(shape)

            # Show the shape in the image
            img[dst > 0.1 * dst.max()] = [0, 0, 255]
            cv2.imshow("image", img)
            cv2.imwrite(f"Corner_detection_{number}.jpg", img)
            cv2.waitKey(500)
            cv2.destroyAllWindows

    return pattern


def check_pattern(pattern, CORRECT_PATTERN):
    """
    Function to determine wheter of not
    the found pattern on the plate is
    the same as the "password"
    """
    print(f"Detected pattern: {pattern}")
    if pattern == CORRECT_PATTERN:
        print("Correct")
        return True
    print("Incorrect")
    return False


if __name__ == "__main__":
    # Variable
    CORRECT_PATTERN = ["Rectangle", "Circle", "Triangle", "Rectangle"]

    # Load image
    img = cv2.imread("image.jpg")
    number = 1
    # Undistort image
    # img = undistort_image(img)

    # Get the plate image
    # plate_img = get_plate(img, number)

    # Find the pattern of the plate
    # pattern = get_pattern(plate_img, number)
    pattern = get_pattern(img, number)

    # Check is the found pattern is correct
    # check_pattern(pattern, CORRECT_PATTERN)
