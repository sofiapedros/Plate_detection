import cv2
import glob
import copy
import numpy as np
import imageio
from picamera2 import Picamera2


def load_images(filenames):
    return [imageio.imread(filename) for filename in filenames]


def get_chessboard_points(chessboard_shape, dx, dy):
    return [
        [(i % chessboard_shape[0]) * dx, (i // chessboard_shape[0]) * dy, 0]
        for i in range(np.prod(chessboard_shape))
    ]


if __name__ == "__main__":
    filenames = list(sorted(glob.glob("./prueba2/*.jpg")))
    imgs = load_images(filenames)

    # We will execute findChessboardCorners for each image to find the corners
    corners = [cv2.findChessboardCorners(i, (7, 7)) for i in imgs]
    corners2 = copy.deepcopy(corners)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 17, 0.01)
    # Cada una de las imagenes la volvemos a blanco y negro
    imgs_grey = [cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) for img in imgs]

    # For each image and corners we are going to use cornerSubPix
    cornersRefined = [
        cv2.cornerSubPix(i, cor[1], (7, 7), (-1, -1), criteria) if cor[0] else []
        for i, cor in zip(imgs_grey, corners2)
    ]

    imgs2 = copy.deepcopy(imgs)

    for i in range(len(imgs2)):
        cv2.imwrite(
            f"img_{i+1}.jpg",
            cv2.drawChessboardCorners(
                imgs2[i],
                patternSize=[8, 6],
                corners=corners2[i][1],
                patternWasFound=corners2[i][0],
            ),
        )

    # We are going to draw the corners if we have found them
    tmp = [
        cv2.drawChessboardCorners(img, (7, 7), cor[1], cor[0])
        for img, cor in zip(imgs2, corners)
        if cor[0]
    ]

    cb_points = get_chessboard_points((7, 7), 17, 17)

    # We are going to retrieve existing corners (cor[0] == True)
    valid_corners = [cor[1] for cor in corners if cor[0]]

    num_valid_images = len(valid_corners)

    # Matrix with the coordinates of the corners
    real_points = get_chessboard_points((7, 7), 17, 17)

    # We are going to convert our coordinates list in the reference system to numpy array
    object_points = np.asarray(
        [real_points for i in range(num_valid_images)], dtype=np.float32
    )

    # Convert the corners list to array
    image_points = np.asarray(valid_corners, dtype=np.float32)

    # ASIGNMENT: Calibrate the left camera

    rms, intrinsics, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objectPoints=object_points,
        imagePoints=image_points,
        imageSize=(7, 7),
        cameraMatrix=imgs2[0].shape[::-1],
        distCoeffs=None,
    )
    # Calculate extrinsecs matrix using Rodigues on each rotation vector addid its translation vector
    extrinsics = list(
        map(lambda rvec, tvec: np.hstack((cv2.Rodrigues(rvec)[0], tvec)), rvecs, tvecs)
    )
    # Save the calibration file
    np.savez("calib", intrinsic=intrinsics, extrinsic=extrinsics, dist=dist_coeffs)

    # Lets print some outputs
    print("Corners standard intrinsics:\n", intrinsics)
    print("Corners standard dist_coefs:\n", dist_coeffs)
    print("root mean sqaure reprojection error:\n", rms)
