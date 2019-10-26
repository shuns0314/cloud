# todo
## preprocessing
- 前処理の色分けとかをどうするか。
- ~~train test split の階層化~~
- ~~maskの穴を消す~~
- ~~gamma 補正~~

## Network
- 簡易に分析できるネットワークの構築（Unetの小さいやつとか、YOLOとか）
- Resnet38
- Resnet50
- Essential
- YOLOとか四角予測結果を返すもの

## Data Augmentation
- Data Augmentationの比率の適正化
- ~~transrate~~
- ~~gamma~~
- ~~equalize~~

## Utility
- ~~arg.classの実装~~
- json or yaml

## parameta
- ~~schedulerの実装~~
- ~~Bayesian Optimization~~

## loss
- Lovasz Hinge loss を使えるようにする。

## Ensemble
- Snapshot Ensemble

## post-processing
- ~~矩形に変換~~
- 矩形、囲むだけ、何もせずをhyperparameterにして探索するとか...

## TTA
- test time augmentation

## memo
- 座標と枠の個数とかを予測
- 人と違うところ画像サイズ、DA、前処理？

## 画像チェック
### train
- 1e40a05, 3b9a092, 3c1e099, 06e5dd6, 8bd81ce, 17fe76e, 41f92e5, 42ac1b7, 076de5e, 171e62f, 400a38d, 563fc48, 838cd7a, 1588d4c,
 5265e81, 24884e7, 046586a, a2dc5c0, b092cc1, c26c635, c0306e5, e04fea3, e5f2f24, eda52f2, f32724b, fa645da, fd5aa5d

### test
- 3cb4ce5, 5d7cd45, 25c5403, 35c7561, 43d43a8, 66e236e, 73f1a66, 4375f9e, 60853e7, 94275be, 638185f, bf4ddbb, d6dd4f6, ea32456