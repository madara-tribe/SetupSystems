import cv2
import glob

def comb_movie(movie_files,out_path):

    # 形式はmp4
    fourcc = cv2.VideoWriter_fourcc('m','p','4','v')

    # 動画情報の取得
    movie = cv2.VideoCapture(movie_files[0])
    fps = movie.get(cv2.CAP_PROP_FPS)
    height = movie.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = movie.get(cv2.CAP_PROP_FRAME_WIDTH)


    # 出力先のファイルを開く
    out = cv2.VideoWriter(out_path, int(fourcc), fps, (int(width), int(height)))


    for movies in (movie_files):
        print(movies)
        # 動画ファイルの読み込み，引数はビデオファイルのパス
        movie = cv2.VideoCapture(movies)

        # 正常に動画ファイルを読み込めたか確認
        if movie.isOpened() == True:
            # read():1コマ分のキャプチャ画像データを読み込む
            ret, frame = movie.read()
        else:
            ret = False

        while ret:
            # 読み込んだフレームを書き込み
            out.write(frame)
            # 次のフレーム読み込み
            ret, frame = movie.read()

if __name__=='__main__':
    files = sorted(glob.glob("./movies/*.mp4"))
    out_path = "movie_output.mp4"
    comb_movie(files, out_path)