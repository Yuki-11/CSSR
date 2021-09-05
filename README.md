# Crack Segmentation for Low-Resolution Images using Joint Learning with Super-Resolution (CSSR)
### [Paper](http://www.mva-org.jp/Proceedings/2021/papers/O1-1-2.pdf) | [Data](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1)
<!-- [![Open CSSR in Colab](https://colab.research.google.com/)<br> -->
<br>

[Crack Segmentation for Low-Resolution Images using Joint Learning with Super-Resolution](http://www.mva-org.jp/Proceedings/2021/papers/O1-1-2.pdf)<br>
 [Yuki Kondo](https://yuki-11.github.io/)\*<sup>1</sup>,
 [Norimichi Ukita](https://www.toyota-ti.ac.jp/Lab/Denshi/iim/ukita/index-j.html)\*<sup>1</sup><br>
 \*<sup>1</sup>[Toyota Technological Institute (TTI-J)](https://www.toyota-ti.ac.jp/english/)
in [MVA 2021](http://www.mva-org.jp/mva2021/) (Oral Presentation, [Best Practical Paper Award](http://www.mva-org.jp/archives.BestPracticalPaperAward.php))

<img src='imgs/results.png'/>

## Our Framework
<img src='imgs/arc.png'/>

## News
* July 27, 2021 -> We received the [Best Practical Paper Award](http://www.mva-org.jp/archives.BestPracticalPaperAward.php) at [MVA 2021](http://www.mva-org.jp/mva2021/)

## What's this?


## Dependencies
* Python >= 3.6
* PyTorch >= 1.8
* numpy >= 1.19


### Usage

1. Clone the repository:

   ```shell
   git clone https://github.com/Yuki-11/CSSR.git
   ```

2. Download [khanhha dataset](https://github.com/khanhha/crack_segmentation):

   ```shell
   cd $CSSR_ROOT
   cd datasets
   curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1xrOqv0-3uMHjZyEUrerOYiYXW_E8SUMP" > /dev/null
   CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"  
   curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1xrOqv0-3uMHjZyEUrerOYiYXW_E8SUMP" -o temp_dataset.zip
   unzip temp_dataset.zip
   rm temp_dataset.zip
   ```

3. Install packages:

   ```shell
   cd $CSSR_ROOT
   pip install -r requirement.txt
   ```

4. Training:
   ```shell
   cd $CSSR_ROOT
   python train.py --config_file <CONFIG FILE>
   ```

5. Test:
   ```shell
   cd $CSSR_ROOT
   python test.py output/<OUTPUT DIRECTORY (OUTPUT_DIR at config.yaml)> <iteration number> 
   ```

### Models and results

Models and results can be downloaded from this link!

## Citations
If you find this work useful, please consider citing it.
```
@inproceedings{CSSR2021,
  title={Crack Segmentation for Low-Resolution Images using Joint Learning with Super-Resolution},
  author={Kondo, Yuki and Ukita, Norimichi},
  booktitle={International Conference on Machine Vision Applications (MVA)},
  year={2021}
}

```