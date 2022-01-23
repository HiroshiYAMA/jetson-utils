# ゲーミングノートPCに Ubuntu18.04をインストールする手順

## ブータブル USBメモリの作成
Ubuntu18.04 のインストールイメージ(*.iso)をダウンロードする。
例えば、[これ](http://cdimage.ubuntulinux.jp/releases/18.04.3/ubuntu-ja-18.04.3-desktop-amd64.iso)。

インストールイメージを USBメモリに書き込む。例えば、
Windows10で [Rufus](https://rufus.ie/ja/)等を使う。

## ノートPCの BIOS設定を変更
とにかくセキュアブートを disable にする。  
Windowsとのデュアルブートするなら、Fast bootも disableにする。

ブートデバイス順は、
1. USB memory
2. USB CD-ROM/DVD
3. 内蔵 SSD/HDD

で OK。
変えなくてもノートPC起動時にブートデバイスを選択できれば、それでも良い。

## 試用 Ubuntu(Try Ubuntu)で起動
ブータブル USBメモリをノートPCに挿して、電源 ON。
通常はGRUBメニューのインストール(Install)だと思うが、これだとインストールに失敗するので、試用Ubuntu(Try Ubuntu)を使う。

いきなり試用Ubuntuを選択して起動するならそれで OKだが、
画面表示が乱れたり、ブート途中で固まったりするようなら、
GRUBメニューで試用Ubuntu(Try Ubuntu)にカーソルを合わせてから、おもむろに 'e'キーを押す。
そうすると、起動オプションを変更出来る。

### 起動オプションの変更(**もし、変更しなくても Ubuntu が起動するなら変更しない**)
何行目かに **quiet splash** があるので、
これを **nomodeset acpi=off** に変更する。
Ctrl + 'x'キーで変更完了。

これで無事起動するようになる。

## Ubuntuインストール
[このサイト](https://qiita.com/Shunmo17/items/d2161a570847bb8b8f74)の手順で Ubuntuをインストールする。

ただし、最後に再起動する前に起動オプションを変更する。

### 起動オプションの変更(**もし、変更しなくても Ubuntu が起動したなら変更しなくて良い**)
/mntに chrootした状態で、
```
/etc/default/grub
```
を編集する。エディタは何でも OK。

きっと ***/etc/default/grub*** は、
```bash
GRUB_CMDLINE_LINUX_DEFAULT="quiet splash"
```
ってなっているので、
```bash
GRUB_CMDLINE_LINUX_DEFAULT="nomodeset"
# 又は、上記設定で上手く起動出来なければ、acpi=offを付ける
GRUB_CMDLINE_LINUX_DEFAULT="nomodeset acpi=off"
# nomodeset を付けると起動しない PC もあるので、その場合は、空っぽで OK
GRUB_CMDLINE_LINUX_DEFAULT=""
```
に変更する。
その後、
```bash
update-grub
```
として、GRUBの設定を更新する。  
これで再起動しても大丈夫。  
ただし、**acpi=off**を付けて起動した場合、シャットダウンや再起動は電源 offにする途中で停止するので、その時は電源ボタンを長押しする。

上記の一連の手順をさらっと書くと、  
無事に試用Ubuntu(Try Ubuntu)が起動したら、端末(ターミナル)を開いて、
```bash
sudo ubiquity -b 
```
として、普通にインストールする。

その後、端末(ターミナル)にて、
```bash
# root(/)パーティションをマウント
sudo mount /dev/nvme0n1p2 /mnt

# /boot/efiパーティションをマウント
sudo mkdir -p /mnt/boot/efi
sudo mount /dev/nvme0n1p1 /mnt/boot/efi

# その他
for i in /dev /dev/pts /proc /sys; do sudo mount -B $i /mnt$i; done
sudo modprobe efivars
```

```bash
# GRUBをインストール
sudo apt-get install --reinstall grub-efi-amd64-signed
sudo grub-install --no-nvram --root-directory=/mnt
```

```bash
# GRUBを更新
sudo chroot /mnt
update-grub
cd /boot/efi/EFI
cp -R ubuntu/* BOOT/
cd BOOT
cp grubx64.efi bootx64.efi
```

```bash
vi /etc/default/grub

# GRUB_CMDLINE_LINUX_DEFAULTの設定をこうする
GRUB_CMDLINE_LINUX_DEFAULT="nomodeset"
# 又は、上記設定で上手く起動出来なければ、acpi=offを付ける
GRUB_CMDLINE_LINUX_DEFAULT="nomodeset acpi=off"
# nomodeset を付けると起動しない PC もあるので、その場合は、空っぽで OK
GRUB_CMDLINE_LINUX_DEFAULT=""
```

```bash
update-grub
exit    # chrootを抜ける
```

ここまでやると、なんとなく Ubuntuが動くようになるが、
- 画面の解像度が低い(1024x768くらい)
- スライドパッドが使えない
- 大抵の内蔵無線LAN(Wi-Fi)デバイスが使えない

ことが多いので、Ubuntuのアップデートと NVIDIAのデバイスドライバのインストールをする。

## Ubuntuのアップデート
これすると、大抵の内蔵無線LANデバイスが使えるようになる。

有線LAN又は、無線LANアダプタ(USB)を使ってネットワークに接続する。もちろん、インターネットにアクセス出来ること。

おもむろに Ubuntuをアップデートする。
```bash
sudo apt update
sudo apt upgrade
sudo reboot
```
再起動後、無事に大抵の内蔵無線LANデバイスが使えるようになっている。
それでも内蔵無線LANデバイスが使えない場合は、Linuxカーネルのバージョンアップ([Mainline](http://baker-street.jugem.jp/?eid=450)) & メーカー提供のデバイスドライバのインストールをやってみる。

## NVIDIAのデバイスドライバのインストール
巷にいろいろな方法が紹介されているが、Ubuntuアップデート後は Ubuntuに既にインストールされているアプリ ***ソフトウェアとアップデート(Software & Update)*** を使えば OK。

***ソフトウェアとアップデート(Software & Update)*** を起動して、**追加のドライバー** タブを選択すると、しばらく検索した後、いくつかの NVIDIAのデバイスドライバーが表示される。
その一覧の中からバージョン 460.32以上のものを選択して、**変更の適用** ボタンを押す。

**ただし、バージョン 470.63.~ 以上は使わないこと。**
GPU のクロック制御(電源管理 PowerMizer)が不調になる場合がある。

なので、デバイスドライバのインストール後、うっかり `sudo apt upgrade` でやってしまわないように `apt-mark` で `hold` しておくのが良い。
```bash
sudo apt-mark hold *-460
# 又は
sudo apt-mark hold *-470
```

再起動後、無事にスライドパッドが使えるようになっていて、画面の解像度もより高解像度に出来るようになっている。

**この方法でダメな場合は、** [ここ](https://zenn.dev/190ikp/articles/how_to_install_nvidia_drivers)のやり方でインストールしてみる。
インストール後、Tensor7.2.3のインストールに悪さするので、途中で作成した cuda.list を削除する。
```bash
sudo rm /etc/apt/sources.list.d/cuda.list
```

### NVIDIA GPU のクロック制御(電源管理 PowerMizer)は大丈夫？
たまにまともに動作しない PC がいる。
- クロックが上がらない。常に最少電力モード
- 一旦電源プラグが抜けると、その後クロックが上がらなくなる

そんな時は、
***/etc/modprobe.d/nvidia-user-registry.conf*** っていうファイルに
下記のように書く。その後、PC 再起動。
```
options nvidia NVreg_RegistryDwords="PowerMizerEnable=0x1;PerfLevelSrc=0x3333;PowerMizerDefault=0x2;PowerMizerDefaultAC=0x2"
```

## **ここまでの作業、お疲れ様でした。これで大抵のノートPCで普通に Ubuntu18.04が使えるようになります。**

---

# DeepStreamのインストール手順
JetPack 4.5.1に入っていた GStreamerのエレメントたち(nvvideo4linux2系)が軒並み無いので、jetson-inference(jetson-utils)のために DeepStream をインストールする。
もしかすると、自前で NVDEC, NVENCなエレメントたちをビルドするのでも良いかもしれない。

おおよそ[dGPU Setup for Ubuntu](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Quickstart.html#dgpu-setup-for-ubuntu)の手順に従ってインストールする。

## 事前に諸々のソフトを好みに応じてインストールしておく
```bash
sudo apt install \
    net-tools arp-scan less ssh \
    build-essential cmake git curl \
    vim vim-gtk3 p7zip-full python python3-pip \
    exfat-fuse exfat-utils
```

## これは DeepStreamに必要なやつ
```bash
sudo apt install \
    libssl1.0.0 \
    libgstreamer1.0-0 \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    libgstrtspserver-1.0-0 \
    libjansson4
```

## Install NVIDIA driver 460.32 ~ 470.52
これは既にインストール済みなのでスキップ。

## CUDA 11.1.1のインストール
### インストールする前に確認
***ソフトウェアとアップデート(Software & Update)*** を使ってうまく NVIDIA のデバイスドライバがインストール出来なかった場合は、
きっと[ここ](https://zenn.dev/190ikp/articles/how_to_install_nvidia_drivers)のやり方でインストールしたと思うので、
その場合、引き続きその Web ページに従って、cuda-toolkit-\<version\> パッケージをインストールれば OK。
下記のインストール手順はスキップして、TensorRT のインストールへ Go!!

### インストール手順
[ここ](https://developer.nvidia.com/cuda-11.1.1-download-archive?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1804&target_type=deblocal)の通りにする。

下記を選択する。
- Linux
- x86_64
- Ubuntu
- 18.04
- deb(local)

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin

sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600

wget https://developer.download.nvidia.com/compute/cuda/11.1.1/local_installers/cuda-repo-ubuntu1804-11-1-local_11.1.1-455.32.00-1_amd64.deb

sudo dpkg -i cuda-repo-ubuntu1804-11-1-local_11.1.1-455.32.00-1_amd64.deb

sudo apt-key add /var/cuda-repo-ubuntu1804-11-1-local/7fa2af80.pub

sudo apt-get update
sudo apt-get -y install cuda
```

インストール後、NVIDIAのデバイスドライバーがバージョン 455.32に下がっちゃうので、
再度、 ***ソフトウェアとアップデート(Software & Update)*** の **追加のドライバー** タブにて、バージョン 460.32 ~ 470.52のものをインストールする。

## TensorRT 7.2.3のインストール
[ここ](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-723/install-guide/index.html#installing-debian)の通りにする。
TensorRT local repo fileは、[これ](https://developer.nvidia.com/compute/machine-learning/tensorrt/secure/7.2.3/local_repos/nv-tensorrt-repo-ubuntu1804-cuda11.1-trt7.2.3.4-ga-20210226_1-1_amd64.deb)を使う。

```bash
sudo dpkg -i nv-tensorrt-repo-ubuntu1804-cuda11.1-trt7.2.3.4-ga-20210226_1-1_amd64.deb

sudo apt-key add /var/nv-tensorrt-repo-ubuntu1804-cuda11.1-trt7.2.3.4-ga-20210226/7fa2af80.pub

sudo apt-get update
sudo apt-get install tensorrt

sudo apt-get install python3-libnvinfer-dev
sudo apt-get install onnx-graphsurgeon

dpkg -l | grep TensorRT
```

### PyTorch 1.8.1のインストール
最新は 1.9.0だが、[APIの挙動が結構変わってそう](https://github.com/pytorch/pytorch/releases/tag/v1.9.0)なので、安全のため 1.8.1にする。

[ここ](https://pytorch.org/get-started/locally/)の通りにする。
下記を選択、
- LTS(1.8.1)
- Linux
- Pip
- Python
- CUDA 11.1

```bash
pip3 install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html
```

C++から使うには、
- LibTorch
- C++/Java

を選択。
(cxx11 ABI)の方をダウンロードかな。

### ONNX 1.7.0のインストール
バージョン 1.8.0以上はインストール失敗するので、バージョン 1.7.0にする。

```bash
pip3 install onnx==1.7.0
```

## librdkafkaのインストール
[ここ](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Quickstart.html#id2)の通り。

```bash
git clone https://github.com/edenhill/librdkafka.git

cd librdkafka
git reset --hard 7101c2310341ab3f4675fc565f64f0967e135a6a

./configure
make
sudo make install

sudo mkdir -p /opt/nvidia/deepstream/deepstream-5.1/lib
sudo cp /usr/local/lib/librdkafka* /opt/nvidia/deepstream/deepstream-5.1/lib
```

## DeepStream SDK 5.1のインストール
[ここ](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Quickstart.html#id3)の通り。

**deepstream-5.1_5.1.0-1_amd64.deb** をダウンロード後、
```bash
sudo apt-get install ./deepstream-5.1_5.1.0-1_amd64.deb
```

---
# jetson-inference ビルド & Go
x86_64系でビルド出来るようにしたブランチが GitHubにある。
現在、Jetson用のブランチと統合されている。
ブランチ名は、
- jetson-inference: bgmv2(本流) 又は、Br_collo
- jetson-utils(./utils): master

ONNXファイルは、
[ここ](https://drive.google.com/drive/folders/1x8zIfqA6NZu9Sr_TzrURc3qADCGbaj6i?usp=sharing)
と
[ここ(推奨)](https://drive.google.com/drive/folders/1-_LG7KkBWLa5THeC4e3QsotW7-l1_cn_?usp=sharing)
にある。

MP4等の動画ファイルの入出力に関して、
- 入力はループ再生するために seek を使うと GStreamer のパイプラインが停止しちゃうので、
ちょっと強引な方法(GStreamer のパイプラインを作り直す)で対応。
そのため、先頭に戻る時に少しの間だけ映像の更新が止まる。
- 出力は GStreamer 直接ではうまく動作しないので、OpenCV を使って対応。
OpenCV 経由なら GStreamer のパイプラインを使用可能。ハードウェアエンコード OK。
- MXF ファイル入力は、OpenCV(バックエンド:FFMpeg)にて対応。
- Jetson 環境に比べて、GStreamer, OpenCV とのフレームバッファの受け渡しが遅い。特に動画保存。

という状態なので、
- 動画(マスク等)保存はリアルタイム動作をあまり期待できない。
- 2UVC入力、が最も安定して動作する。
- UVC + 背景動画ファイル、はハイスペックな PC が望ましい。

### ちなみに処理速度は、
| Name | GPU arch | CUDA cores | Tensor cores |
| --- | --- | --- | --- |
| RTX 2070 Super with MAX-Q | Turing | 2560 | 320 |
| RTX 3080 Laptop | Ampere | 6144 | 192 |
| Jetson Xavier NX | Volta | 384 | 48 |

の時、
Nsight Systems で計測して、

#### 2UVC入力で、
***ONNX -> TensorRT だと、***
| model | 2070 speed (pha only) | 2070 speed (fgr + pha) | 3080 (pha only) | 3080 (fgr + pha) | Xavier NX (pha only) | Xavier NX (fgr + pha) |
| --- | --- | --- | --- | --- | --- | --- |
| low (mobilenetv2 1920x1080 Sc025 Th100) | - | - | - | - | - | 33.6msec以内 |
| mid (resnet50 1920x1080 Sc025 Th100) | - | - | - | - | - | 50msec |
| low (resnet50 1920x1080 Sc015 FULL) | 26msecくらい | 31msecくらい | 24.5(9.5)msec | 30msec | - | - |
| mid (resnet50 1920x1080 Sc025 FULL) | 29msecくらい | 34msecくらい | 26(11.5)msec | 31.5msec | - | - |
| high (resnet50 1920x1080 Sc050 FULL) | 39msecくらい | 45msecくらい | 35(17)msec | 40.5msec | - | - |
| **KaijinMatte20K_FHD_UHD** |
| **FP16** |
| (mobilenetv2 1920x1080 Sc025 FULL) | - | - | - | 30(10.5)msec | - | 106msec |
| (mobilenetv2 1920x1080 Sc040 FULL) | - | - | - | 33(11.5)msec | - | 133msec |
| (mobilenetv2 1920x1080 Sc050 FULL) | - | - | - | 36(13)msec | - | 152msec |
| (resnet50 1920x1080 Sc025 FULL) | - | - | - | 31(11.5)msec | - | 124msec |
| (resnet50 1920x1080 Sc040 FULL) | - | - | - | 36(13.5)msec | - | 177msec |
| (resnet50 1920x1080 Sc050 FULL) | - | - | - | 40(18)msec | - | 217msec |

***TorchScript だと、***
| model | 2070 speed (pha only) | 2070 speed (fgr + pha) | 3080 (pha only) | 3080 (fgr + pha) |
| --- | --- | --- | --- | --- |
| mobilenetv2 1920x1080 Sc025 sampling | 24msecくらい | 23msecくらい | 21msec | 22msec |
| resnet50 1920x1080 Sc025 sampling | 28msecくらい | 26msecくらい | 25msec | 26msec |
| **KaijinMatte20K_FHD_UHD** |
| **FP16** |
| mobilenetv2 1920x1080 Sc025 sampling80000 | - | - | - | 23(15)msec |
| mobilenetv2 1920x1080 Sc040 sampling80000 | - | - | - | 27(16)msec |
| mobilenetv2 1920x1080 Sc050 sampling80000 | - | - | - | 30.5(19)msec |
| resnet50 1920x1080 Sc025 sampling80000 | - | - | - | 28(17)msec |
| resnet50 1920x1080 Sc040 sampling80000 | - | - | - | 39(28)msec |
| resnet50 1920x1080 Sc050 sampling80000 | - | - | - | 47(40)msec |
| **FP32** |
| mobilenetv2 1920x1080 Sc025 sampling80000 | - | - | - | 24.5(16)msec |
| mobilenetv2 1920x1080 Sc040 sampling80000 | - | - | - | 32(21)msec |
| mobilenetv2 1920x1080 Sc050 sampling80000 | - | - | - | 37.5(31)msec |
| resnet50 1920x1080 Sc025 sampling80000 | - | - | - | 38(29)msec |
| resnet50 1920x1080 Sc040 sampling80000 | - | - | - | 58(53)msec |
| resnet50 1920x1080 Sc050 sampling80000 | - | - | - | 77(72)msec |

***RVM(TorchScript) だと、***
| model | 3080 |
| --- | --- |
| mobilenetv3 1920x1080 Sc025 | (16)msec |
| mobilenetv3 1920x1080 Sc050 | (22)msec |
| resnet50 1920x1080 Sc025 | (17)msed |
| resnet50 1920x1080 Sc050 | (34.4)msed |

#### UVC + 4K30p(H.264, 29.97fps)で、
| model | 2070 speed (pha only) | 2070 speed (fgr + pha) | 3080 (pha only) | 3080 (fgr + pha) |
| --- | --- | --- | --- | --- |
| low (resnet50 1920x1080 Sc015 FULL) | 33.36msecくらい(*1) | - | -(12.5)msec | - |
| mid (resnet50 1920x1080 Sc025 FULL) | 34.85msecくらい | - | -(13)msec | - |
| high (resnet50 1920x1080 Sc050 FULL) | 46.90msecくらい | - | -(21)msec | - |
| **KaijinMatte20K_FHD_UHD** |
| **FP16** |
| (mobilenetv2 1920x1080 Sc025 FULL) | - | - | - | -(12)msec |
| (mobilenetv2 1920x1080 Sc040 FULL) | - | - | - | -(14)msec |
| (mobilenetv2 1920x1080 Sc050 FULL) | - | - | - | -(16)msec |
| (resnet50 1920x1080 Sc025 FULL) | - | - | - | -(13.5)msec |
| (resnet50 1920x1080 Sc040 FULL) | - | - | - | -(17.5)msec |
| (resnet50 1920x1080 Sc050 FULL) | - | - | - | -(21)msec |

#### UVC + 4K24p(H.264, 23.98fps)で、
| model | 2070 speed (pha only) |
| --- | --- |
| low (resnet50 1920x1080 Sc015 FULL) | 41.70msecくらい(*1) |
| mid (resnet50 1920x1080 Sc025 FULL) | 41.70msecくらい(*1) |
| high (resnet50 1920x1080 Sc050 FULL) | 44.90msecくらい |

(*1) GStreamer での動画ファイル入力時、フレームレートで同期待ちの時間が含まれているため、計測時間は必ずフレームレート以上になる。フレームレートと同じ処理時間なら問題無し。

処理速度の数値: カッコ内の値は、CUDA pinned memoryの代わりにデバイスメモリを使った時のもの。
dGPU(discrete GPU)では効果がある。
Jetson Xavier NX の場合、元々 CPU と GPU とで同じメモリを共有しているため処理速度はほぼ変わらない。

RTX 2070 の1.5倍くらいのスペックがあれば、high でも 30p の可能性あり。  
**最低ラインは、RTX 3060以上かな。出来れば、3080以上。**
と言いたいところだが、 RTX 3080 Laptop は、RTX 2070 Super with MAX-Q の約 1.1倍だった。

## x86_64系のブランチをゲット
```bash
git clone git@github.com:flow-dev/jetson-inference-team.git

cd jetson-inference-team
git submodule update --init

git checkout bgmv2
又は、
git checkout Br_collo

pushd utils
git checkout master
popd
```

## ビルド
[ここ](https://github.com/flow-dev/jetson-inference-team#command)の通り。
***sudo ldconfig*** まで実施すれば OK。

### 注意事項
```bash
cmake ..
# 途中、
# モデルのダウンロードはどれもしない(チェックを全て外す)
# PyTorchのインストールはスキップ
```

---
