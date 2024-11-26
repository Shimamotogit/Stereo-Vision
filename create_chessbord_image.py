import cv2
import numpy as np

def generate_chessboard_with_padding(rows=9, cols=6, square_size=50, padding=20, output_file="chessboard_with_padding.png"):
    """
    チェスボード画像を生成し、周囲に白いパディングを追加する関数。

    :param rows: チェスボードの行数（黒と白の合計の数）。
    :param cols: チェスボードの列数（黒と白の合計の数）。
    :param square_size: 各正方形のサイズ（ピクセル単位）。
    :param padding: パディングの幅（ピクセル単位）。
    :param output_file: 出力ファイル名。
    """
    # チェスボードの全体サイズを計算
    board_height = rows * square_size
    board_width = cols * square_size

    # チェスボードの画像を作成
    chessboard = np.zeros((board_height, board_width), dtype=np.uint8)

    # チェスボードの白黒を交互に塗る
    for i in range(rows):
        for j in range(cols):
            if (i + j) % 2 == 0:  # 偶数のマス目を白にする
                x_start, y_start = j * square_size, i * square_size
                x_end, y_end = x_start + square_size, y_start + square_size
                chessboard[y_start:y_end, x_start:x_end] = 255

    # 周囲にパディングを追加
    padded_chessboard = cv2.copyMakeBorder(
        chessboard, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=255
    )

    # ファイルに保存
    cv2.imwrite(output_file, padded_chessboard)
    print(f"チェスボード画像（パディング付き）を生成しました: {output_file}")

# パラメータを指定して生成
generate_chessboard_with_padding(rows=9, cols=6, square_size=50, padding=50, output_file="chessboard.png")
