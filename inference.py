import net
import torch
import os
from face_alignment import align
import numpy as np


adaface_models = {
    'ir_50':"pretrained/adaface_ir50_ms1mv2.ckpt",
}

def load_pretrained_model(architecture='ir_50'):
    # load model and pretrained statedict
    assert architecture in adaface_models.keys()
    model = net.build_model(architecture)
    statedict = torch.load(adaface_models[architecture])['state_dict']
    model_statedict = {key[6:]:val for key, val in statedict.items() if key.startswith('model.')}
    model.load_state_dict(model_statedict)
    model.eval()
    return model

def to_input(pil_rgb_image):
    np_img = np.array(pil_rgb_image)
    brg_img = ((np_img[:,:,::-1] / 255.) - 0.5) / 0.5
    brg_img_transposed = np.array([brg_img.transpose(2,0,1)])
    tensor = torch.tensor(brg_img_transposed).float()
    return tensor

if __name__ == '__main__':
    # 事前学習済みモデルをevalモードで読み込み
    model = load_pretrained_model('ir_50')
    # 特徴ベクトルとそのノルムを取得
    # (バッチサイズ、特徴ベクトルの次元数=RGB、画像サイズ、画像サイズ)
    feature, norm = model(torch.randn(2, 3, 112, 112))

    # テスト画像が保存されているディレクトリのパスを指定
    test_image_dir = 'face_alignment/test_images'

    # 特徴ベクトルを格納するための空のリストを初期化
    # 各画像から抽出した特徴ベクトルの格納用
    features = []

    # テスト画像ディレクトリ内のすべてのファイル名をソートして一つずつ処理
    for filename in sorted(os.listdir(test_image_dir)):
        # 各ファイルのフルパスを作成
        target_path = os.path.join(test_image_dir, filename)

        # 指定したパスの画像に対して顔のアライメント（整列）を行い、その結果を取得
        # 顔の位置を検出し、画像内の顔を所定の位置に整列させた画像を返す（顔認識の精度を向上させるための一般的な前処理）
        aligned_rgb_img = align.get_aligned_face(target_path)

        # 整列されたRGB画像をモデルの入力形式（BGRテンソル）に変換
        # `to_input` 関数は、画像をNumPy配列に変換し、色チャンネルをRGBからBGRに並べ替え、値を正規化してからPyTorchテンソルに変換している
        # PyTorchモデルへはこの形式を入力として渡す
        bgr_tensor_input = to_input(aligned_rgb_img)

        # モデルに変換済みの入力画像を渡し、特徴ベクトルを取得
        # モデルは入力テンソルを処理し、顔の特徴を表すベクトルを出力（顔認識や類似度計算のために使用するための特徴量）
        feature, _ = model(bgr_tensor_input)

        # 取得した特徴ベクトルをリストに追加（類似度計算用）
        features.append(feature)

    # 特徴ベクトル間の類似度スコアを計算
    # torch.cat(features): featuresリストのすべてのテンソルを結合し、ベクトル同士の内積（行列積）を計算
    # 各画像ペアのコサイン類似度を示す
    similarity_scores = torch.cat(features) @ torch.cat(features).T

    # 類似度スコアを出力
    print(similarity_scores)
