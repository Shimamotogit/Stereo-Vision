import cv2
import numpy as np
import os
import time

def prepare_chessboard_points(chessboard_size, square_size):
    """
    チェスボードの3Dポイントを準備

    Args:
        chessboard_size (tuple) : チェスボードのコーナー数（列, 行）
        square_size (float)     : チェスボードの各正方形のサイズ（任意の単位）

    Returns:
        numpy.ndarray           : チェスボードの3D座標
    """

    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    objp *= square_size
    return objp

def capture_stereo_images(cap_left, cap_right, chessboard_size, objpoints, imgpoints_left, imgpoints_right, image_folder):
    """
    ステレオカメラで画像をキャプチャし、チェスボードのコーナーを検出

    Args:
        cap_left  (cv2.VideoCapture) : 左カメラのキャプチャオブジェクト
        cap_right (cv2.VideoCapture) : 右カメラのキャプチャオブジェクト
        chessboard_size (tuple)      : チェスボードのコーナー数（列, 行）
        objpoints (list)             : 3Dポイントのリスト
        imgpoints_left  (list)       : 左カメラの検出した2Dポイントリスト
        imgpoints_right (list)       : 右カメラの検出した2Dポイントリスト
        image_folder (str)           : 画像を保存するフォルダのパス
    """

    print("チェスボードをカメラに向けてください's'で撮影、'q'で終了")

    while True:
        ret_left,  frame_left  = cap_left.read()
        ret_right, frame_right = cap_right.read()

        if not ret_left or not ret_right:
            print("カメラからフレームを取得できません")
            continue

        combined_frame = np.hstack((frame_left, frame_right))
        cv2.imshow("Stereo Cameras (Left | Right)", combined_frame)

        key = cv2.waitKey(1)
        if key == ord('s'):
            detect_and_save_chessboard(
                frame_left, frame_right, chessboard_size, objpoints, imgpoints_left, imgpoints_right, image_folder
            )
        elif key == ord('q'):
            break

    cv2.destroyAllWindows()

def detect_and_save_chessboard(
    frame_left, frame_right, chessboard_size, objpoints, imgpoints_left, imgpoints_right, image_folder
    ):
    """
    チェスボードを検出し、ポイントと画像を保存

    Args:
        frame_left  (numpy.ndarray) : 左カメラから取得したフレーム
        frame_right (numpy.ndarray) : 右カメラから取得したフレーム
        chessboard_size (tuple)     : チェスボードのコーナー数（列, 行）
        objpoints (list)            : 3Dポイントのリスト
        imgpoints_left  (list)      : 左カメラの検出した2Dポイントリスト
        imgpoints_right (list)      : 右カメラの検出した2Dポイントリスト
        image_folder (str)          : 画像を保存するフォルダのパス
    """

    gray_left  = cv2.cvtColor(frame_left,  cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)

    retL, cornersL = cv2.findChessboardCorners(
        gray_left, chessboard_size,
        cv2.CALIB_CB_ADAPTIVE_THRESH +
        cv2.CALIB_CB_FAST_CHECK +
        cv2.CALIB_CB_NORMALIZE_IMAGE
    )
    retR, cornersR = cv2.findChessboardCorners(
        gray_right, chessboard_size,
        cv2.CALIB_CB_ADAPTIVE_THRESH +
        cv2.CALIB_CB_FAST_CHECK +
        cv2.CALIB_CB_NORMALIZE_IMAGE
    )

    if retL and retR:
        objpoints.append(objp)
        imgpoints_left.append(cornersL)
        imgpoints_right.append(cornersR)

        timestamp = time.time()
        cv2.imwrite(f"{image_folder}/left_{timestamp}.jpg",  frame_left)
        cv2.imwrite(f"{image_folder}/right_{timestamp}.jpg", frame_right)
        print("チェスボードを検出して画像を保存")
    else:
        print("チェスボードが見つかりません")

def perform_stereo_calibration(objpoints, imgpoints_left, imgpoints_right, cap_left_shape):
    """
    ステレオキャリブレーションを実行します

    Args:
        objpoints (list)       : 3Dポイントのリスト
        imgpoints_left  (list) : 左カメラの検出した2Dポイントリスト
        imgpoints_right (list) : 右カメラの検出した2Dポイントリスト
        cap_left_shape (tuple) : 左カメラの画像の解像度（高さ, 幅）

    Returns:
        tuple                  : 各カメラの行列、歪み係数、回転行列、並進ベクトル
    """

    print("キャリブレーションを実行中...")

    _, cameraMatrixL, distCoeffsL, _, _ = cv2.calibrateCamera(
        objpoints, imgpoints_left, cap_left_shape[::-1], None, None
    )
    _, cameraMatrixR, distCoeffsR, _, _ = cv2.calibrateCamera(
        objpoints, imgpoints_right, cap_left_shape[::-1], None, None
    )

    _, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpoints_left, imgpoints_right,
        cameraMatrixL, distCoeffsL,
        cameraMatrixR, distCoeffsR,
        cap_left_shape[::-1],
        flags=cv2.CALIB_FIX_INTRINSIC
    )

    return cameraMatrixL, distCoeffsL, cameraMatrixR, distCoeffsR, R, T

def save_calibration_results(filename, cameraMatrixL, distCoeffsL, cameraMatrixR, distCoeffsR, R, T):
    """
    キャリブレーション結果を保存

    Args:
        filename (str): 保存するファイルのパス
        cameraMatrixL (numpy.ndarray) : 左カメラのカメラ行列
        distCoeffsL   (numpy.ndarray) : 左カメラの歪み係数
        cameraMatrixR (numpy.ndarray) : 右カメラのカメラ行列
        distCoeffsR   (numpy.ndarray) : 右カメラの歪み係数
        R (numpy.ndarray)             : 回転行列
        T (numpy.ndarray)             : 並進ベクトル
    """

    cv_file = cv2.FileStorage(filename, cv2.FILE_STORAGE_WRITE)
    cv_file.write("camera_matrix_left",  cameraMatrixL)
    cv_file.write("camera_matrix_right", cameraMatrixR)
    cv_file.write("dist_coeffs_left",  distCoeffsL)
    cv_file.write("dist_coeffs_right", distCoeffsR)
    cv_file.write("R", R)
    cv_file.write("T", T)
    cv_file.release()

    print("キャリブレーション結果を保存しました")

def main():
    """
    メイン処理
    """

    chessboard_size = (8, 5)
    square_size = 1.0
    camera_id_left = 0
    camera_id_right = 1
    image_folder = "calib_images"
    os.makedirs(image_folder, exist_ok=True)

    global objp  # detect_and_save_chessboard内で使用
    objp = prepare_chessboard_points(chessboard_size, square_size)

    objpoints = []
    imgpoints_left  = []
    imgpoints_right = []

    cap_left  = cv2.VideoCapture(camera_id_left,  cv2.CAP_DSHOW)
    cap_right = cv2.VideoCapture(camera_id_right, cv2.CAP_DSHOW)

    if not cap_left.isOpened() or not cap_right.isOpened():
        print("カメラを起動できませんでした")
        return

    try:
        capture_stereo_images(
            cap_left, cap_right, chessboard_size, objpoints, imgpoints_left, imgpoints_right, image_folder
        )
    finally:
        cap_left.release()
        cap_right.release()

    if len(objpoints) > 0:
        cameraMatrixL, distCoeffsL, cameraMatrixR, distCoeffsR, R, T = perform_stereo_calibration(
            objpoints, imgpoints_left, imgpoints_right, cap_left.shape
        )
        save_calibration_results('stereo_calib.xml', cameraMatrixL, distCoeffsL, cameraMatrixR, distCoeffsR, R, T)
    else:
        print("キャリブレーションを実行するための十分なデータがありません")

if __name__ == "__main__":
    main()
