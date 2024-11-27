import cv2
import numpy as np

# ステレオカメラキャリブレーションデータの読み込み
stereo_params = cv2.FileStorage('stereo_calib.xml', cv2.FILE_STORAGE_READ)
camera_matrix_left  = stereo_params.getNode("camera_matrix_left").mat()
camera_matrix_right = stereo_params.getNode("camera_matrix_right").mat()
dist_coeffs_left  = stereo_params.getNode("dist_coeffs_left").mat()
dist_coeffs_right = stereo_params.getNode("dist_coeffs_right").mat()
R = stereo_params.getNode("R").mat()
T = stereo_params.getNode("T").mat()

# アルゴリズム選択（BM または SGBM）
algorithm = "BM" # "BM" or "SGBM" を選択
blockSize = 5

if algorithm == "BM":
    # StereoBM_createの設定
    stereo = cv2.StereoBM_create(
        numDisparities = 16 * 8,
        # 視差範囲: 視差の最大値から最小値を引いた時の値、16の倍数で指定する
        # 高い値にすると広範囲の視差を計算可能になるが、計算コストが増える
        
        blockSize = blockSize,
        # ブロックサイズ: マッチングに使用する領域サイズ（奇数）
        # 小さい値: 細かいディテールが見えるがノイズが増える
        # 大きい値: ノイズを軽減するが滑らかになる
    )
elif algorithm == "SGBM":
    # StereoSGBM_createの設定
    stereo = cv2.StereoSGBM_create(
        minDisparity = 0,
        # 最小視差: 計算する視差の最小値、通常は0

        numDisparities = 16 * 4,
        # 視差範囲: 視差の最大値から最小値を引いた時の値、16の倍数で指定する
        # BMより精度が高い計算が可能

        blockSize = blockSize,
        # ブロックサイズ: マッチングに使用する領域サイズ（奇数）
        # SGBMでは通常BMより小さい値を使用する

        P1 = 8 * 3 * blockSize ** 2,
        # 平滑化ペナルティ（小さな変化用）
        # 計算式: 8 * チャンネル数 * blockSize^2
        # スムーズな視差画像を作るための調整値

        P2 = 32 * 3 * blockSize ** 2,
        # 平滑化ペナルティ（大きな変化用）
        # 計算式: 32 * チャンネル数 * blockSize^2
        # 境界線など視差の急変する箇所を制御する値

        disp12MaxDiff = 1,
        # 左右整合性の最大許容差
        # 左右視差がこの値を超える場合は無効視差とみなします

        preFilterCap = 63,
        # 入力画像のキャッピング値
        # 入力画像の値を0～preFilterCapに収める
        # 高すぎるとノイズが増える場合がある

        uniquenessRatio = 15,
        # 一意性検査の閾値（%）
        # 他の視差候補と比較して、どれだけ差があるかを判定する
        # 高い値: 正確だが細部を失う可能性がある
        # 低い値: 誤差を含むが細部を捉える

        speckleWindowSize = 100,
        # スペックルノイズ除去のウィンドウサイズ
        # 小さい値だとノイズが残る可能性がある
        # 大きい値だと滑らかさが増すが、詳細が失われる

        speckleRange = 64,
        # スペックルノイズ除去時の視差変化許容範囲
        # この値を超える変化を持つ領域は無効視差と見なす

        # mode=cv2.STEREO_SGBM_MODE_HH
        # cv2.STEREO_SGBM_MODE_SGBM (標準)
        # cv2.STEREO_SGBM_MODE_HH (ハイパフォーマンスモード、計算量が増える)
        # cv2.STEREO_SGBM_MODE_SGBM_3WAY (3パスモード、計算量が少し増えるが品質向上)
        # cv2.STEREO_SGBM_MODE_HH4 (4方向検索を行う)
        # デフォルト: cv2.STEREO_SGBM_MODE_SGBM
        # 推奨: 品質を優先する場合はcv2.STEREO_SGBM_MODE_HHを選択
    )

else:
    print("不明なアルゴリズムが選択されました。")
    exit()

# カメラの画像を取得
cap_left  = cv2.VideoCapture(0, cv2.CAP_DSHOW) # 左カメラ
cap_right = cv2.VideoCapture(1, cv2.CAP_DSHOW) # 右カメラ

if not (cap_left.isOpened() and cap_right.isOpened()):
    print("カメラが開けませんでした。")
    exit()

ret_left,  frame_left  = cap_left.read()
ret_right, frame_right = cap_right.read()

if not (ret_left and ret_right):
    print("画像が取得できませんでした。")
    exit()

# ステレオ整列用のパラメータを計算
h, w = frame_left.shape[:2]
R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
    camera_matrix_left,  dist_coeffs_left,
    camera_matrix_right, dist_coeffs_right,
    (w, h), R, T
)

# ステレオ整列用の整列マップを計算
map_left_x, map_left_y = cv2.initUndistortRectifyMap(
    camera_matrix_left, dist_coeffs_left, R1, P1, (w, h), cv2.CV_32FC1
)
map_right_x, map_right_y = cv2.initUndistortRectifyMap(
    camera_matrix_right, dist_coeffs_right, R2, P2, (w, h), cv2.CV_32FC1
)

# ステレオ整列の整列処理
rectified_left = cv2.remap(
    frame_left, map_left_x, map_left_y, cv2.INTER_LINEAR
)
rectified_right = cv2.remap(
    frame_right, map_right_x, map_right_y, cv2.INTER_LINEAR
)

# 右視差マッチャー
stereo_right = cv2.ximgproc.createRightMatcher(stereo)

# WLSフィルタの設定
wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereo)
wls_filter.setLambda(8000)     # スムージングの強さ（大きくするほど滑らかに）
wls_filter.setSigmaColor(1.5)  # 色空間の影響範囲（エッジの保持度合いを調整）

smoothing_frame_value = 5 #指定されたフレーム数の平均を用いて視差マップを平滑化する
                          #高い値ほど平滑化されるが、残像が残る。利用しない場合は'1'
disp_visual_list = []

# リアルタイム処理ループ
while True:
    ret_left,  frame_left  = cap_left.read()
    ret_right, frame_right = cap_right.read()

    if not (ret_left and ret_right):
        print("カメラからの画像取得に失敗しました。")
        break

    # ステレオ整列（Rectification）
    rectified_left  = cv2.remap(frame_left,  map_left_x,  map_left_y,  cv2.INTER_LINEAR)
    rectified_right = cv2.remap(frame_right, map_right_x, map_right_y, cv2.INTER_LINEAR)

    # グレースケール変換
    gray_left  = cv2.cvtColor(rectified_left,  cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(rectified_right, cv2.COLOR_BGR2GRAY)

    # 視差マップの生成
    left_disparity  = stereo.compute(gray_left, gray_right).astype(np.float32) / 16.0
    right_disparity = stereo_right.compute(gray_right, gray_left).astype(np.float32) / 16.0

    # WLSフィルタの適用
    filtered_disparity = wls_filter.filter(
        left_disparity, gray_left, disparity_map_right=right_disparity
    )

    # フィルタ後の視差マップを可視化
    disp_visual = cv2.normalize(filtered_disparity, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    # disp_visual = cv2.normalize(left_disparity, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    disp_visual_list.append(disp_visual)
    disp_visual_list = disp_visual_list[-smoothing_frame_value:]
    disp_average_visual = np.mean(disp_visual_list, axis=0).astype(np.uint8)

    disp_colormap = cv2.applyColorMap(disp_average_visual, cv2.COLORMAP_PLASMA)

    # 結果をリアルタイム表示
    cv2.imshow("Rectified Left", rectified_left)
    cv2.imshow("Rectified Right", rectified_right)
    cv2.imshow("Filtered Disparity Map", disp_colormap)

    # 'q'キーで終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# リソースの解放
cap_left.release()
cap_right.release()
cv2.destroyAllWindows()