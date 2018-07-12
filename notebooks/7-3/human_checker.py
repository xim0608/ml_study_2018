import human_keras as human
import sys, os
from PIL import Image
import numpy as np

#コマンドラインからファイル名を得る
if len(sys.argv) <= 1:
    print("human-checker.py (ファイル名)")
    quit()

image_size = 32
categories = ["woman", "man"]

#入力画像をNumpyに変換
data = []
files = []
for fname in sys.argv[1:]:
    img = Image.open(fname)
    img = img.convert("RGB")
    img = img.resize((image_size, image_size))
    in_data = np.asarray(img)
    data.append(in_data)
    files.append(fname)
data = np.array(data)

#CNNのモデルを構築
model = human.build_model(data.shape[1:])
model.load_weights("humanmodel.hdf5")

#データを予測
html = ""
pre = model.predict(data)
for i, p in enumerate(pre):
    label = p.argmax()
    print("+ 入力：", files[i])
    print("| 性別：", categories[label])
    html += """
        <h3>入力：{0}</h3>
        <div>
          <p><img src="{1}" width=300></p>
          <p>性別：{2}</p>
        </div>
    """.format(os.path.basename(files[i]),
        files[i],
        categories[i])

#レポートを保存
html = "<html><body style='text-align:center;'>" + \
    "<style> p { margin:0; padding:0; } </style>" + \
    html + "</body></html>"
with open("human-result.html", "w") as f:
    f.write(html)
