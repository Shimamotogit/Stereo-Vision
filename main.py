import cv2
import numpy as np

def load_stereo_params(filename="stereo_calib.xml"):
    """
    ステレオカメラキャリブレーションデータを読み込む
    
    Args:
        filename (str): キャリブレーションデータのファイルパス

    Returns:
        dict: ステレオカメラのキャリブレーションデータ（行列、回転、平行移動など）
    """

    stereo_params = cv2.FileStorage(filename, cv2.FILE_STORAGE_READ)
    params = {
        "camera_matrix_left":  stereo_params.getNode("camera_matrix_left").mat(),
        "camera_matrix_right": stereo_params.getNode("camera_matrix_right").mat(),
        "dist_coeffs_left":  stereo_params.getNode("dist_coeffs_left").mat(),
        "dist_coeffs_right": stereo_params.getNode("dist_coeffs_right").mat(),
        "R": stereo_params.getNode("R").mat(),
        "T": stereo_params.getNode("T").mat(),
    }
    stereo_params.release()
    return params

def initialize_stereo_matcher(algorithm, blockSize):
    """
    ステレオマッチャー（BMまたはSGBM）を初期化する関数
    
    Args:
        algorithm (str): "BM" または "SGBM" を指定
        blockSize (int): ブロックサイズ（奇数）
        
    Returns:
        cv2.StereoMatcher: 初期化されたステレオマッチャー
    """

    if algorithm == "BM":
        stereo = cv2.StereoBM_create(
            numDisparities=16 * 8,
            blockSize=blockSize
        )

    elif algorithm == "SGBM":
        stereo = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=16 * 4,
            blockSize=blockSize,
            P1=8 * 3 * blockSize ** 2,
            P2=32 * 3 * blockSize ** 2,
            disp12MaxDiff=1,
            preFilterCap=63,
            uniquenessRatio=15,
            speckleWindowSize=100,
            speckleRange=64,
        )

    else:
        raise ValueError("不明なアルゴリズムが選択されました")

    return stereo

def compute_rectification_maps(params, image_size):
    """
    ステレオ整列用のマップを計算する関数

    Args:
        params (dict): キャリブレーションデータ
        image_size (tuple): 入力画像のサイズ (幅, 高さ)
    Returns:
        tuple: 左右画像の整列マップ (map_left_x, map_left_y, map_right_x, map_right_y)
    """

    R1, R2, P1, P2, _, _, _ = cv2.stereoRectify(
        params["camera_matrix_left"],  params["dist_coeffs_left"],
        params["camera_matrix_right"], params["dist_coeffs_right"],
        image_size, params["R"], params["T"]
    )   

    map_left_x, map_left_y = cv2.initUndistortRectifyMap(
        params["camera_matrix_left"], params["dist_coeffs_left"], R1, P1, image_size, cv2.CV_32FC1
    )
    map_right_x, map_right_y = cv2.initUndistortRectifyMap(
        params["camera_matrix_right"], params["dist_coeffs_right"], R2, P2, image_size, cv2.CV_32FC1
    )

    return map_left_x, map_left_y, map_right_x, map_right_y

def create_wls_filter(stereo):
    """
    WLSフィルタを初期化する関数

    Args:
        stereo (cv2.StereoMatcher): 左視差マッチャー

    Returns:
        cv2.ximgproc_DisparityWLSFilter: 初期化されたWLSフィルタ
    """

    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereo)
    wls_filter.setLambda(8000)
    wls_filter.setSigmaColor(1.5)
    return wls_filter

def process_stereo_frames(cap_left, cap_right, stereo, stereo_right, wls_filter, maps):
    """
    ステレオカメラからフレームを取得して処理するメインループ

    Args:
        cap_left (cv2.VideoCapture): 左カメラキャプチャ
        cap_right (cv2.VideoCapture): 右カメラキャプチャ
        stereo (cv2.StereoMatcher): 左視差マッチャー
        stereo_right (cv2.StereoMatcher): 右視差マッチャー
        wls_filter (cv2.ximgproc_DisparityWLSFilter): WLSフィルタ
        maps (tuple): ステレオ整列マップ
    """

    map_left_x, map_left_y, map_right_x, map_right_y = maps
    disp_visual_list = []

    while True:
        ret_left, frame_left = cap_left.read()
        ret_right, frame_right = cap_right.read()

        if not (ret_left and ret_right):
            print("画像取得に失敗しました")
            break

        rectified_left  = cv2.remap(frame_left,  map_left_x,  map_left_y,  cv2.INTER_LINEAR)
        rectified_right = cv2.remap(frame_right, map_right_x, map_right_y, cv2.INTER_LINEAR)

        gray_left  = cv2.cvtColor(rectified_left,  cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(rectified_right, cv2.COLOR_BGR2GRAY)

        left_disparity  = stereo.compute(gray_left, gray_right).astype(np.float32) / 16.0
        right_disparity = stereo_right.compute(gray_right, gray_left).astype(np.float32) / 16.0

        filtered_disparity = wls_filter.filter(left_disparity, gray_left, disparity_map_right=right_disparity)
        disp_visual = cv2.normalize(filtered_disparity, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

        disp_visual_list.append(disp_visual)
        disp_visual_list = disp_visual_list[-5:]  # 平滑化フレーム数を指定
        disp_average_visual = np.mean(disp_visual_list, axis=0).astype(np.uint8)

        disp_colormap = cv2.applyColorMap(disp_average_visual, cv2.COLORMAP_PLASMA)

        cv2.imshow("Rectified Left",  rectified_left)
        cv2.imshow("Rectified Right", rectified_right)
        cv2.imshow("Filtered Disparity Map", disp_colormap)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def main():
    camera_id_left = 1
    camera_id_right = 2
    params = load_stereo_params()
    stereo = initialize_stereo_matcher("BM", 5)
    stereo_right = cv2.ximgproc.createRightMatcher(stereo)
    wls_filter = create_wls_filter(stereo)

    cap_left  = cv2.VideoCapture(camera_id_left,  cv2.CAP_DSHOW)
    cap_right = cv2.VideoCapture(camera_id_right, cv2.CAP_DSHOW)
    if not (cap_left.isOpened() and cap_right.isOpened()):
        print("カメラが開けませんでした")
        return

    ret, frame = cap_left.read()
    if not ret:
        print("カメラから画像が取得できませんでした")
        return

    image_size = (frame.shape[1], frame.shape[0])
    maps = compute_rectification_maps(params, image_size)

    process_stereo_frames(cap_left, cap_right, stereo, stereo_right, wls_filter, maps)

    cap_left.release()
    cap_right.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
