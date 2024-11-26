import cv2
import numpy as np
import matplotlib.pyplot as plt

# ステレオカメラキャリブレーションデータの読み込み（事前に準備されたキャリブレーションファイルを利用）
stereo_params = cv2.FileStorage('stereo_calib.xml', cv2.FILE_STORAGE_READ)
camera_matrix_left  = stereo_params.getNode("camera_matrix_left").mat()
camera_matrix_right = stereo_params.getNode("camera_matrix_right").mat()
dist_coeffs_left  = stereo_params.getNode("dist_coeffs_left").mat()
dist_coeffs_right = stereo_params.getNode("dist_coeffs_right").mat()
R = stereo_params.getNode("R").mat()
T = stereo_params.getNode("T").mat()

# カメラの画像を取得
cap_left  = cv2.VideoCapture(0)  # 左カメラ
cap_right = cv2.VideoCapture(1)  # 右カメラ

if not (cap_left.isOpened() and cap_right.isOpened()):
    print("カメラが開けませんでした。")
    exit()

ret_left,  img_left  = cap_left.read()
ret_right, img_right = cap_right.read()

if not (ret_left and ret_right):
    print("画像が取得できませんでした。")
    exit()

# ステレオ整列（Rectification）
h, w = img_left.shape[:2]
R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
    camera_matrix_left, dist_coeffs_left,
    camera_matrix_right, dist_coeffs_right,
    (w, h), R, T
)

map_left_x, map_left_y = cv2.initUndistortRectifyMap(
    camera_matrix_left,  dist_coeffs_left,  R1, P1, (w, h), cv2.CV_32FC1
)
map_right_x, map_right_y = cv2.initUndistortRectifyMap(
    camera_matrix_right, dist_coeffs_right, R2, P2, (w, h), cv2.CV_32FC1
)

rectified_left  = cv2.remap(
    img_left,  map_left_x,  map_left_y,  cv2.INTER_LINEAR
)
rectified_right = cv2.remap(
    img_right, map_right_x, map_right_y, cv2.INTER_LINEAR
)

# アルゴリズム選択（BM または SGBM）
algorithm = "SGBM"  # "BM" or "SGBM" を選択

if algorithm == "BM":
    # StereoBM_createの設定
    stereo = cv2.StereoBM_create(
        numDisparities=16 * 4,  # 視差範囲: 視差の最大値 - 最小値。16の倍数で指定。
                                # 高い値にすると広範囲の視差を計算可能だが計算コスト増。
        blockSize=15            # ブロックサイズ: マッチングに使用する領域サイズ（奇数）。
                                # 小さい値: 細かいディテールが見えるがノイズが増加。
                                # 大きい値: ノイズを軽減するが滑らかになりすぎる。
    )
elif algorithm == "SGBM":
    # StereoSGBM_createの設定
    stereo = cv2.StereoSGBM_create(
        minDisparity=0,            # 最小視差: 計算する視差の最小値。通常は0。
        numDisparities=16 * 4,     # 視差範囲: 視差の最大値 - 最小値。16の倍数で指定。
                                   # BMより精度が高い計算が可能。
        blockSize=5,               # ブロックサイズ: マッチングに使用する領域サイズ（奇数）。
                                   # SGBMでは通常、BMより小さい値を使用。
        P1=8 * 3 * 5 ** 2,         # 平滑化ペナルティ（小さな変化用）。
                                   # 計算式: 8 * チャンネル数 * blockSize^2。
                                   # スムーズな視差画像を作るための調整値。
        P2=32 * 3 * 5 ** 2,        # 平滑化ペナルティ（大きな変化用）。
                                   # 計算式: 32 * チャンネル数 * blockSize^2。
                                   # 境界線など視差の急変する箇所を制御する値。
        disp12MaxDiff=1,           # 左右整合性の最大許容差。
                                   # 左右視差がこの値を超える場合は無効視差と見なす。
        preFilterCap=63,           # 入力画像のキャッピング値。
                                   # 入力画像の値を0～この値に収める。
                                   # 高すぎるとノイズが増える場合あり。
        uniquenessRatio=15,        # 一意性検査の閾値（%）。
                                   # 他の視差候補と比較して、どれだけ差があるかを判定。
                                   # 高い値: 正確だが細部を失う可能性あり。
                                   # 低い値: 誤差を含むが細部を捉える。
        speckleWindowSize=100,     # スペックルノイズ除去のウィンドウサイズ。
                                   # 小さい値だとノイズが残る可能性がある。
                                   # 大きい値だと滑らかさが増すが、詳細が失われる。
        speckleRange=32            # スペックルノイズ除去時の視差変化許容範囲。
                                   # この値を超える変化を持つ領域は無効視差と見なす。
    )

else:
    print("不明なアルゴリズムが選択されました。")
    exit()

# 視差マップの生成
gray_left  = cv2.cvtColor(rectified_left,  cv2.COLOR_BGR2GRAY)
gray_right = cv2.cvtColor(rectified_right, cv2.COLOR_BGR2GRAY)

disparity = stereo.compute(gray_left, gray_right).astype(np.float32) / 16.0  # SGBMではスケールが必要

# 視差マップの可視化
plt.figure(figsize=(10, 7))
plt.title(f"Disparity Map ({algorithm})")
plt.imshow(disparity, cmap='plasma')
plt.colorbar(label="Disparity Value")
plt.show()

cap_left.release()
cap_right.release()
