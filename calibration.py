import cv2
import numpy as np
import os
import time

# キャリブレーション用のチェスボード設定
chessboard_size = (9, 6)
square_size     = 1.0  # 正方形1つのサイズ（任意の単位）

# カメラのID設定
camera_id_left  = 0  # 左カメラのID
camera_id_right = 1  # 右カメラのID

# 保存用フォルダ
image_folder = "calib_images"
os.makedirs(image_folder, exist_ok=True)

# キャリブレーション用のポイント準備
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
objp *= square_size

objpoints       = []  # 3Dポイント
imgpoints_left  = []  # 左カメラの2Dポイント
imgpoints_right = []  # 右カメラの2Dポイント

# カメラを起動
cap_left  = cv2.VideoCapture(camera_id_left)
cap_right = cv2.VideoCapture(camera_id_right)

if not cap_left.isOpened() or not cap_right.isOpened():
    print("カメラを起動できませんでした。")
    exit()

print("チェスボードをカメラに向けてください。's'で撮影、'q'で終了。")

while True:
    ret_left,  frame_left  = cap_left.read()
    ret_right, frame_right = cap_right.read()

    if not ret_left or not ret_right:
        print("カメラからフレームを取得できませんでした。")
        continue

    combined_frame = np.hstack((frame_left, frame_right))
    cv2.imshow("Stereo Cameras (Left | Right)", combined_frame)

    key = cv2.waitKey(1)
    if key == ord('s'):
        gray_left  = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)

        # チェスボードのコーナーを検出
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
        print(retL, retR)

        if retL and retR:
            objpoints.append(objp)
            imgpoints_left.append(cornersL)
            imgpoints_right.append(cornersR)

            # 検出結果を保存
            timestamp = time.time()
            cv2.imwrite(f"{image_folder}/left_{timestamp}.jpg",  frame_left)
            cv2.imwrite(f"{image_folder}/right_{timestamp}.jpg", frame_right)
            print("チェスボードを検出して画像を保存しました。")
        else:
            print("チェスボードが見つかりませんでした。")
    elif key == ord('q'):  # 終了
        break

# カメラを解放
cap_left.release()
cap_right.release()
cv2.destroyAllWindows()

# ステレオキャリブレーション
print("キャリブレーションを実行中...")
_, cameraMatrixL, distCoeffsL, _, _ = cv2.calibrateCamera(objpoints, imgpoints_left,  gray_left.shape[::-1],  None, None)
_, cameraMatrixR, distCoeffsR, _, _ = cv2.calibrateCamera(objpoints, imgpoints_right, gray_right.shape[::-1], None, None)

_, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
    objpoints, imgpoints_left, imgpoints_right,
    cameraMatrixL, distCoeffsL,
    cameraMatrixR, distCoeffsR,
    gray_left.shape[::-1],
    flags=cv2.CALIB_FIX_INTRINSIC
)

# キャリブレーション結果を保存
cv_file = cv2.FileStorage('stereo_calib.xml', cv2.FILE_STORAGE_WRITE)
cv_file.write("camera_matrix_left",  cameraMatrixL)
cv_file.write("camera_matrix_right", cameraMatrixR)
cv_file.write("dist_coeffs_left",    distCoeffsL)
cv_file.write("dist_coeffs_right",   distCoeffsR)
cv_file.write("R", R)
cv_file.write("T", T)
cv_file.release()
print("キャリブレーション結果を保存しました。")
