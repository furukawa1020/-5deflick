# M5Deflick

M5UnitV2などのカメラで大きなクッション型かなフリック盤面を見て、殴った位置と方向をかな入力に変換する試作です。

## 考え方

1. カメラ映像でクッション画像の4隅をクリックしてキャリブレーションします。
2. 盤面を正面向きの座標に変換します。
3. 色で各かなキーの現在位置を軽く追跡します。
4. 拳や手の動きを動体検出で拾います。
5. 動きが入ったキーと、キー中心から見た進入側をフリック方向にします。
6. `あ` の左側を殴ると `い`、上なら `う`、右なら `え`、下なら `お` になります。

## セットアップ

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 起動

ブラウザUIで使う場合:

```powershell
python m5deflick_web.py --source unitv2
```

起動したら `http://127.0.0.1:8787` を開きます。

UnitV2をUSBでつないでいる場合は、まずこれを試します。

```powershell
python m5deflick.py --source unitv2
```

通常のWebカメラで試す場合:

```powershell
python m5deflick.py --source 0
```

UnitV2のプレビューURLやMJPEG URLが分かっている場合:

```powershell
python m5deflick.py --source-url http://10.254.239.1/your_stream_path
```

最初は入力せずコンソールへ出すだけです。実際にアクティブなアプリへ文字入力する場合は:

```powershell
python m5deflick.py --source unitv2 --output unicode
```

## 操作

ブラウザUIでは、`4隅` ボタンでキャリブレーションを開始し、Camera映像上で左上、右上、右下、左下の順にクリックします。`Arm Mask` に腕や拳が白く出ていれば検出されています。

- 起動直後に盤面の4隅を、左上、右上、右下、左下の順にクリックします。
- `r`: キャリブレーションをやり直します。
- `b`: 背景を取り直します。
- `q` または `Esc`: 終了します。

## 調整

殴っても反応しない場合:

```powershell
python m5deflick.py --source unitv2 --min-motion-area 700 --deadzone 0.16
```

誤爆が多い場合:

```powershell
python m5deflick.py --source unitv2 --min-motion-area 2500 --cooldown 0.6
```

色が照明で拾いにくい場合:

```powershell
python m5deflick.py --source unitv2 --color-min-area 500 --color-search-inflate 0.65
```

色追跡を切って、固定レイアウトだけで見る場合:

```powershell
python m5deflick.py --source unitv2 --zone-mode layout
```

方向を「入ってきた側」ではなく「動いたベクトル」で見たい場合:

```powershell
python m5deflick.py --source unitv2 --direction-mode motion
```

## メモ

UnitV2の公式ドキュメントでは、USB接続時にPCとUnitV2がネットワーク接続され、ブラウザから `unitv2.py` または `10.254.239.1` でアクセスできると説明されています。スクリプトは `unitv2` 指定時にそのホストから映像URL候補を探しますが、ファームウェアや表示アプリによってURLが違う場合は `--source-url` で直接指定してください。

クッション自体が大きく揺れると動体検出が盤面全体を拾います。その場合はカメラとクッションを少し固定するか、四隅にマーカーを付けて自動追跡する方式に進化させるのが次の段階です。
