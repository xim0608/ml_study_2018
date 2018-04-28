# 機械学習輪講2018
## 2018年前期 - 3年生の機械学習輪講用レポジトリ
- [Dockerfile](https://github.com/oikenfight/ml_study/blob/develop/Dockerfile)、[docker-compose.yml](https://github.com/oikenfight/ml_study/blob/develop/docker-compose.yml)ファイルは[おいちゃんさん](https://github.com/oikenfight/)が書かれたものです。

## 使い方
### 初回起動時
1. `$ docker-compose up` で起動
2. 下のような表示が出るので、最後の行のURLをコピーしてブラウザで開く
```
my-notebook_1  |     Copy/paste this URL into your browser when you connect for the first time,
my-notebook_1  |     to login with a token:
my-notebook_1  |         http://localhost:8888/?token=[your access token]
```
3. 終了するときは、Ctrl+C

### 2回目以降
1. `docker-compose start`で起動
2. http://localhost:8888 を開く (閲覧履歴等を削除してcookieが消えた場合は初回起動時の方法でやり直す)
3. `docker-compose stop`で停止
