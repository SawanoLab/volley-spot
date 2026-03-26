# VNL-STES: バレーボール分析における時空間イベント検出ベンチマークデータセットとモデル

> **🔧 GPU互換性向上版（修正フォーク）**
>
> 元のリポジトリ: [jhong93/spot](https://github.com/jhong93/spot)
> 
> **修正内容**: CUDA、MPS（Apple Metal）、CPUデバイスのサポートを追加しました。詳細は[GPU互換性対応](#gpu互換性対応) セクションを参照してください。

**CVPR Workshop 2025**

Hoang Quoc Nguyen, Ankhzaya Jamsrandorj, Vanyi Chao, Yin May Oo, Muhammad Amrulloh Robbani, Kyung-Ryoul Mun, Jinwook Kim

[[プロジェクトページ]](https://hoangqnguyen.github.io/stes/) [[論文]](https://openaccess.thecvf.com/content/CVPR2025W/CVSPORTS/papers/Nguyen_VNL-STES_A_Benchmark_Dataset_and_Model_for_Spatiotemporal_Event_Spotting_CVPRW_2025_paper.pdf)

本プロジェクトは、バレーボール動画において**いつ**（時間）**どこで**（空間）主要なイベントが発生するかをフレームレベルの精度で判定する新しいタスク「精密な時空間イベント検出」を提案します。**VNLデータセット**（1,028試合クリップ、251,110フレーム、6,137の時空間ラベル付きイベント）と**時空間イベント検出モデル（STES）**を提供しています。

STESは従来手法を**+9.86 mTAP**上回り、イベント位置の空間ローカライゼーションで**80.21 mSAP@2-6P**を達成し、2～6ピクセル範囲内のイベント位置を正確に特定します。

## アーキテクチャ

STESは3つのコンポーネントで構成されます：

- **(A) 特徴抽出器**: RegNet-Y 800MFバックボーン + ゲーテッドシフトモジュール（GSM）で時空間特徴を抽出
- **(B) テンポラル集約器**: 多層双方向GRUで長期時間的モデリングおよびイベント分類を実施
- **(C) 空間予測器**: MLPヘッドで正規化された（x, y）イベント座標をバックボーン特徴から回帰

## 性能結果

### 時間的イベント検出（mTAP）

| モデル | @0F | @1F | @2F | @4F | @0-4F |
|--------|-----|-----|-----|-----|-------|
| E2E Spot | 32.37 | 57.95 | 63.37 | 67.58 | 55.32 |
| T-DEED | 30.41 | 63.39 | 69.00 | 71.51 | 58.58 |
| E2E Spatial | 44.56 | 70.03 | 72.15 | 72.51 | 64.81 |
| **STES（提案手法）** | **46.76** | **73.64** | **76.29** | **77.06** | **68.44** |

### 空間的イベント検出（mSAP）

| メトリック | E2E Spatial | STES（提案手法） |
|-----------|-------------|-----------------|
| mSAP@2P | 57.16 | **69.63** |
| mSAP@4P | 79.86 | **84.52** |
| mSAP@6P | 83.82 | **86.47** |
| mSAP@2-6P | 73.61 | **80.21** |

## VNLデータセット

データセットは2022-2023シーズンの8試合で構成され、1,028のラリークリップに分割され、6つのイベントクラスが含まれています：

| アクション | 数 | パーセンテージ |
|-----------|----|----|
| サーブ | 1,071 | 17.45% |
| レシーブ | 1,558 | 25.39% |
| トス | 1,393 | 22.70% |
| スパイク | 1,321 | 21.53% |
| ブロック | 550 | 8.96% |
| 得点 | 244 | 3.97% |

**データセット分割**: 訓練811 / 検証102 / テスト115 ラリー

データセットのダウンロード（リサイズ398x224、13GB）: https://hoangqnguyen.github.io/stes/

## セットアップ

```bash
pip install -r requirements.txt
```

動作確認環境: Linux/macOS (Python 3.10+, PyTorch 2.x, CUDA/MPS/CPU対応)

## データの準備

### フレーム抽出

高さ224ピクセルにリサイズされたビデオフレームを抽出してください：

```
<frame_dir>/
├── video1/
│   ├── 000000.jpg
│   ├── 000001.jpg
│   └── ...
├── video2/
│   └── ...
```

### データセットメタデータ

`data/<dataset_name>/` 下に以下のファイルを配置してください：

- **`class.txt`** ― 1行に1つのクラス名
- **`train.json`**, **`val.json`**, **`test.json`** ― アノテーションファイル

アノテーション形式:
```json
[
    {
        "video": "video_id",
        "num_frames": 3600,
        "num_events": 12,
        "events": [
            {
                "frame": 525,
                "label": "spike",
                "comment": "",
                "x": 0.45,
                "y": 0.62
            }
        ],
        "fps": 25,
        "width": 1920,
        "height": 1080
    }
]
```

## 訓練

### ゼロから訓練

```bash
python train_e2e_spatial.py <dataset> <frame_dir> \
  -m rny008_gsm -t gru \
  --clip_len 64 --batch_size 16 \
  --num_epochs 150 \
  -s exp/<experiment_name> \
  --predict_location \
  --num_workers 4
```

### 事前学習 + ファインチューン

```bash
# ステージ1: 事前学習
python train_e2e_spatial.py vnl_1.5 data/vnl_1.5/frames_224p \
  -m rny008_gsm -t gru --clip_len 64 --batch_size 16 \
  --num_epochs 100 -s exp/pretrain_finetune --predict_location

# ステージ2: ファインチューン（最後のチェックポイントから再開）
python train_e2e_spatial.py vnl_2.0 data/vnl_2.0/frames_224p \
  -m rny008_gsm -t gru --clip_len 64 --batch_size 16 \
  --num_epochs 200 -s exp/pretrain_finetune --predict_location --resume
```

実行可能なスクリプトは `run.sh` および `pretrain_finetune.sh` を参照してください。

## 評価

```bash
python eval.py <model_dir_or_pred_file> -s <split>
```

予測は訓練中に `pred-{split}.{epoch}.recall.json.gz` 形式で保存されます。

## 動画上での推論

```bash
python inference_on_mp4.py --checkpoint_path <path_to_checkpoint> --video <path_to_video>
```

## デモアプリケーション

ビデオファイルをアップロードして検出されたイベントを可視化するGradioウェブインターフェース：

```bash
python app.py
```

## プロジェクト構成

```
├── train_e2e_spatial.py    # メイン訓練スクリプト
├── eval.py                 # 平均精度（mAP）での評価
├── inference_on_mp4.py     # ビデオファイルでの推論
├── app.py                  # Gradioデモアプリ
├── model/
│   ├── common.py           # ベースモデルクラス、MLP
│   ├── modules.py          # GRU予測ヘッド、位置予測器、チャネルアテンション
│   ├── shift.py            # TSMおよびGSM時間シフトモジュール
│   └── min_gru.py          # 最小限のGRUバリアント
├── dataset/
│   ├── frame.py            # フレームベースのデータセットクラス
│   └── transform.py        # データ拡張処理
├── util/
│   ├── device.py           # GPU/MPS/CPU デバイス選択ユーティリティ
│   ├── score.py            # mAP計算と位置対応メトリクス
│   ├── eval.py             # フレーム予測の後処理
│   ├── dataset.py          # データセット登録とヘルパー関数
│   └── io.py               # JSON/GZ入出力ユーティリティ
├── run.sh                  # 訓練スクリプト
├── pretrain_finetune.sh    # 2段階事前学習+ファインチューンスクリプト
└── requirements.txt
```

## GPU互換性対応

本フォークは、**CUDA、MPS（Apple Metal）、CPU対応**を追加し、メイントレーニングコードの変更なしに複数デバイスを自動的にサポートします。

### 新規ファイル

- **`util/device.py`**: デバイス選択ユーティリティ
  - `select_device(prefer: str)` ― 最適なデバイスを自動選択（CUDA > MPS > CPU）
  - `get_autocast_context(device)` ― デバイス対応の混合精度計算
  - `get_grad_scaler(device)` ― デバイス対応の勾配スケーリング

### 修正ファイル

| ファイル | 変更内容 |
|---------|---------|
| `train_e2e_spatial.py` | デバイスユーティリティをインポート；`torch.cuda.amp.autocast()` を `get_autocast_context()` に置き換え |
| `model/common.py` | `device == "cuda"` のチェック代わりに`get_grad_scaler()`を使用 |
| `model/shift.py` | CUDA固有の `FloatTensor` をデバイス非依存の `x.new_zeros()` に置き換え |
| `requirements.txt` | `transformers>=4.30.0` を追加（ConvNeXt V2用） |

### 使用方法

コードはデバイスを自動検出して使用します：
- **NVIDIA CUDA**: GPUマシン上
- **Apple Metal（MPS）**: M1/M2/M3+ Mac上
- **CPU**: フォールバック

コード変更は不要です。通常通り訓練/推論を実行してください：

```bash
python train_e2e_spatial.py <dataset> <frame_dir> -m rny008_gsm -t gru ...
# 最適なデバイスが自動的に選択されます
```

## 元のリポジトリ

- **ソース**: [jhong93/spot](https://github.com/jhong93/spot)
- **ライセンス**: BSD 2-Clause（[LICENSE](LICENSE) を参照）
- **引用**: 下記の引用セクションを参照

## 引用

```bibtex
@inproceedings{nguyen2025vnlstes,
    author={Nguyen, Hoang Quoc and Jamsrandorj, Ankhzaya and Chao, Vanyi and Oo, Yin May and Robbani, Muhammad Amrulloh and Mun, Kyung-Ryoul and Kim, Jinwook},
    title={VNL-STES: A Benchmark Dataset and Model for Spatiotemporal Event Spotting in Volleyball Analytics},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)},
    year={2025}
}
```

## ライセンス

このプロジェクトはBSD 2-Clauseライセンスの下でリリースされています。詳細は[LICENSE](LICENSE)ファイルを参照してください。

BSD 2-Clauseライセンスの概要：
- ✅ 自由に改変、使用、配布ができます
- ✅ 商用利用が可能です
- ⚠️ 著作権表示と免責事項を保持する必要があります

### 著作権表示

**元の著作権（Original Copyright）:**
Copyright 2022 James Hong, Haotian Zhang, Matthew Fisher, Michael Gharbi, Kayvon Fatahalian

**修正内容（Modifications）:**
Copyright 2025 (or current year) SawanoLab
