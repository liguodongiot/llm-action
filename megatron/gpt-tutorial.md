
## 数据下载


修改`lsh/cMinhash.cpp`文件：
- 将exc_type改为curexc_type
- 将exc_value改为curexc_value
- 将exc_traceback改为curexc_traceback




```
> python setup.py install
running install
/usr/local/lib/python3.8/dist-packages/setuptools/command/install.py:34: SetuptoolsDeprecationWarning: setup.py install is deprecated. Use build and pip and other standards-based tools.
  warnings.warn(
/usr/local/lib/python3.8/dist-packages/setuptools/command/easy_install.py:144: EasyInstallDeprecationWarning: easy_install command is deprecated. Use build and pip and other standards-based tools.
  warnings.warn(
running bdist_egg
running egg_info
writing lsh.egg-info/PKG-INFO
writing dependency_links to lsh.egg-info/dependency_links.txt
writing requirements to lsh.egg-info/requires.txt
writing top-level names to lsh.egg-info/top_level.txt
reading manifest file 'lsh.egg-info/SOURCES.txt'
adding license file 'LICENSE'
writing manifest file 'lsh.egg-info/SOURCES.txt'
installing library code to build/bdist.linux-x86_64/egg
running install_lib
running build_py
running build_ext
building 'lsh.cMinhash' extension
x86_64-linux-gnu-gcc -pthread -Wno-unused-result -Wsign-compare -DNDEBUG -g -fwrapv -O2 -Wall -g -fstack-protector-strong -Wformat -Werror=format-security -g -fwrapv -O2 -g -fstack-protector-strong -Wformat -Werror=format-security -Wdate-time -D_FORTIFY_SOURCE=2 -fPIC -I/usr/local/lib/python3.8/dist-packages/numpy/core/include -I/usr/include/python3.8 -c lsh/cMinhash.cpp -o build/temp.linux-x86_64-3.8/lsh/cMinhash.o
In file included from /usr/local/lib/python3.8/dist-packages/numpy/core/include/numpy/ndarraytypes.h:1960,
                 from /usr/local/lib/python3.8/dist-packages/numpy/core/include/numpy/ndarrayobject.h:12,
                 from /usr/local/lib/python3.8/dist-packages/numpy/core/include/numpy/arrayobject.h:5,
                 from lsh/cMinhash.cpp:304:
...
creating dist
creating 'dist/lsh-0.3.0-py3.8-linux-x86_64.egg' and adding 'build/bdist.linux-x86_64/egg' to it
removing 'build/bdist.linux-x86_64/egg' (and everything under it)
Processing lsh-0.3.0-py3.8-linux-x86_64.egg
creating /usr/local/lib/python3.8/dist-packages/lsh-0.3.0-py3.8-linux-x86_64.egg
Extracting lsh-0.3.0-py3.8-linux-x86_64.egg to /usr/local/lib/python3.8/dist-packages
Adding lsh 0.3.0 to easy-install.pth file

Installed /usr/local/lib/python3.8/dist-packages/lsh-0.3.0-py3.8-linux-x86_64.egg
Processing dependencies for lsh==0.3.0
Searching for Cython==0.29.34
Best match: Cython 0.29.34
Adding Cython 0.29.34 to easy-install.pth file
Installing cygdb script to /usr/local/bin
Installing cython script to /usr/local/bin
Installing cythonize script to /usr/local/bin

Using /usr/local/lib/python3.8/dist-packages
Searching for numpy==1.22.2
Best match: numpy 1.22.2
Adding numpy 1.22.2 to easy-install.pth file
Installing f2py script to /usr/local/bin
Installing f2py3 script to /usr/local/bin
Installing f2py3.8 script to /usr/local/bin

Using /usr/local/lib/python3.8/dist-packages
Finished processing dependencies for lsh==0.3.0
```


### 下载URL

```
下载地址：https://mega.nz/folder/EZZD0YwJ#9_PlEQzdMVLaNdKv_ICNVQ/folder/cc4RgQQZ
```


```
> python3 blacklist_urls.py ./urls clean_urls.txt

remove blacklisted urls ..
> found 1 files
[MALFORMED URL]: http://ama
[DOMAIN BLACKLIST]: http://plotholes-comic.tumblr.com/post/2559612220/gee-i-sure-hope-tumblr-doesnt-resize-my-comic-so
[DOMAIN BLACKLIST]: http://reicow.tumblr.com/
[DOMAIN BLACKLIST]: http://www.explosm.net/comics/2281/
...
[DOMAIN BLACKLIST]: http://pencilholder.tumblr.com/post/2553331442/coffee
[DOMAIN BLACKLIST]: http://www.liveleak.com/view?i=897_1293900770
[DOMAIN BLACKLIST]: http://christiannightmares.tumblr.com/post/2556591356/christian-nightmares-10-favorite-worst-video
[DOMAIN BLACKLIST]: http://memegenerator.net/Foul-Bachelor-Frog/ImageMacro/4823906/Toilet-Clogged-Shit-in-trash-bag-and-throw-it-in-dumpster
[DOMAIN BLACKLIST]: http://www.liveleak.com/view?i=77d_1293476512
[DOMAIN BLACKLIST]: http://sati1984.tumblr.com/post/2525789905/hungarian-media-authority-punishes-radio-for-playing
[DOMAIN BLACKLIST]: http://ihearttrains.tumblr.com/post/2144304367/i-was-out-for-a-late-night-bike-ride-saw-this-cp
[DOMAIN BLACKLIST]: http://www.amazon.com/gp/product/0765342294?ie=UTF8&tag=reddit2-20
[DOMAIN BLACKLIST]: http://www.amazon.com/gp/product/B000K7VHPA?ie=UTF8&tag=reddit2-20
[DOMAIN BLACKLIST]: http://www.amazon.com/gp/product/B000MEYKD2?ie=UTF8&tag=reddit2-20
[DOMAIN BLACKLIST]: http://dancingalonetopony.tumblr.com/
[DOMAIN BLACKLIST]: http://christiannightmares.tumblr.com/post/2571743396/christian-woman-advertising-the-rapture-on-the
[MALFORMED URL]: http://paris
[DOMAIN BLACKLIST]: http://www.quickmeme.com/Musically-Oblivious-8th-Grader/Ha/animal-collective-no-i-dont-collect-beanie-babbies/
[DOMAIN BLACKLIST]: http://www.dailymotion.com/video/xeizh2_tvxsgr-manufactured-landscapes_people
...
[DOMAIN BLACKLIST]: http://www.explosm.net/comics/2284/
[DOMAIN BLACKLIST]: http://img411.imageshack.us/i/54398042.png/
[DOMAIN BLACKLIST]: http://www.liveleak.com/view?i=d3a_1293898955
[DOMAIN BLACKLIST]: http://s1221.photobucket.com/albums/dd473/Tuekiira/?action=view&current=untitled.jpg
[DOMAIN BLACKLIST]: http://www.amazon.com/Stainless-Steel-Rat-Returns-ebook/dp/B003P2WO6S
[DOMAIN BLACKLIST]: http://onlythebrightest.tumblr.com/post/1683441156
[DOMAIN BLACKLIST]: http://www.dailymotion.com/video/xg5www_bag-raiders-sunlight_music
[DOMAIN BLACKLIST]: http://www.liveleak.com/view?i=7f8_1286326027
[DOMAIN BLACKLIST]: http://nypl.tumblr.com/post/2586612741/in-honor-of-fantasy-king-j-r-r-tolkiens
[EXTENTION BLACKLIST]: http://phil.caint.com/reddit/RedditIrelandWhereAreYouFrom.xls
[DOMAIN BLACKLIST]: http://christiannightmares.tumblr.com/post/2601211787/church-sign-the-peter-in-me-found-at
[DOMAIN BLACKLIST]: http://s401.photobucket.com/albums/pp94/theaudiopervjr/?action=view&current=bestcoastfallon.mp4
[DOMAIN BLACKLIST]: http://www.google.co.uk/images?q=pokebra&hl=en&client=firefox-a&hs=fra&rls=org.mozilla:en-GB:official&prmd=ivns&source=lnms&tbs=isch:1&ei=j58jTee5NeqShAeurPG2Dg&sa=X&oi=mode_link&ct=mode&ved=0CA0Q_AU&biw=1219&bih=752
[MALFORMED URL]: http://opendesktop.org/content/show.php/Egg Window Manager?content=136862
[DOMAIN BLACKLIST]: http://glitterfarm.tumblr.com/post/2556658380/via-iloveawk
[DOMAIN BLACKLIST]: http://qkme.me/I2
[DOMAIN BLACKLIST]: http://www.liveleak.com/view?i=e27_1294160320
[DOMAIN BLACKLIST]: http://www.liveleak.com/view?i=604_1294163370
[DOMAIN BLACKLIST]: http://glitterfarm.tumblr.com/post/2545229659/fuckyeahradicalcartoons-best-political-cartoons
[DOMAIN BLACKLIST]: http://ryan-a.tumblr.com/post/1325972211/nif01
[DOMAIN BLACKLIST]: http://www.google.co.uk/url?sa=t&source=web&cd=2&sqi=2&ved=0CCgQtwIwAQ&url=http%3A%2F%2Fwww.youtube.com%2Fwatch%3Fv%3Db3wT71l4B7c&rct=j&q=what%27s%20mk%20ultra&ei=u2QjTd-EEcWbhQfciv22Dg&usg=AFQjCNHCVya6KzeCOKaXLDlbpdKZzXs5Vw&sig2=voZA0Ips3wKjOQgRd1Dvrw&cad=rja
[DOMAIN BLACKLIST]: http://www.amazon.com/review/R3FK6MTPT1CZW8/ref=cm_cr_pr_perm?&tag=reddit2-20
[DOMAIN BLACKLIST]: http://maps.google.ca/maps?f=q&source=s_q&hl=en&geocode=&q=&sll=45.641048,-73.840313&sspn=0.097454,0.288391&gl=ca&ie=UTF8&ll=45.635332,-73.501292&spn=0.006091,0.018024&t=k&z=17
[DOMAIN BLACKLIST]: http://www.dailymotion.com/video/x9fsm_actiekatten_shortfilms
[DOMAIN BLACKLIST]: http://www.amazon.com/gp/product/B000FA5R5S/ref=kinw_myk_ro_title
[DOMAIN BLACKLIST]: http://memegenerator.net/Philosoraptor/ImageMacro/4928670/Why-arent-hemorrhoids-called-asteroids-
[DOMAIN BLACKLIST]: http://www.amazon.com/gp/product/B000FA5MUI/ref=kinw_myk_ro_title
[DOMAIN BLACKLIST]: http://qkme.me/EH
[DOMAIN BLACKLIST]: http://qkme.me/IK
[DOMAIN BLACKLIST]: http://www.liveleak.com/view?i=f85_1294099720
[DOMAIN BLACKLIST]: http://troyatnight.tumblr.com/post/2595553006/the-snow-melted-and-then-froze-the-rubble-into-a
[DOMAIN BLACKLIST]: http://fuckyeahblackguyonseinfeld.tumblr.com/
[DOMAIN BLACKLIST]: http://www.liveleak.com/view?i=2d8_1294085921
[DOMAIN BLACKLIST]: http://explosm.net/comics/2285
[DOMAIN BLACKLIST]: http://lobstertoes.tumblr.com/post/2517807815/where-can-i-buy
[DOMAIN BLACKLIST]: http://img809.imageshack.us/slideshow/webplayer.php?id=img20110103102932.jpg
[DOMAIN BLACKLIST]: http://accook365.tumblr.com/
...
[DOMAIN BLACKLIST]: http://www.quickmeme.com/meme/BEy/
[DOMAIN BLACKLIST]: http://www.google.ca/images?q=chris+evans&um=1&hl=en&biw=1503&bih=652&tbs=isch:1&tbas=0&source=lnt&sa=X&ei=MAtGTdjkKoT68Aa3zay-AQ&ved=0CAYQpwUoAA
[DOMAIN BLACKLIST]: http://cgi.ebay.co.uk/Final-Fantasy-VIII-Boxset-Collectors-PS1-RARE-VGC-/150552089322?pt=UK_VintageComputing_RL&hash=item230d9a92ea
FINAL | time elapsed (s): 1.34 | number of urls: 71958 | domain blacklisted: 1477 | extention blacklisted: 1 | short urls (<=8): 0 | malformed urls: 40 | duplicate urls: 0
> writing cleaned up url list to clean_urls.txt
done :-)
```



```
> head -n10 clean_urls.txt 
http://theopenend.com/2011/01/13/toe-short-story-the-mosquito-song-%e2%80%93-ch-15/
http://www.readplatform.com/james-blake-55-min-boiler-room-mix/
http://www.mg.co.za/article/2011-01-13-violent-clashes-spread-to-centre-of-tunisian-capital
http://www.playtankworld.com/
http://www.meetup.com/Suffolk-County-Drinkers/
http://www.insanemusclegirls.com/video/doing-what-she-does-best---being-hot/
http://www.mattbors.com/archives/717.html
http://www.nfl.com/videos/nfl-cant-miss-plays/09000d5d81d8d049/Lynch-s-amazing-TD-seals-upset-of-defending-champs
http://www.reuters.com/article/idUSTRE70B26A20110112?feedType=RSS&feedName=topNews&utm_source=feedburner&utm_medium=feed&utm_campaign=Feed:+reuters/topNews+(News+/+US+/+Top+News)
http://mikeos.berlios.de/write-your-own-os.html
```



### 合并数据



```python
import glob
import sys
import json
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default=".",
        help="path where all the json files are located")

    parser.add_argument("--output_file", type=str, default="merged_output.json",
        help="filename where the merged json should go")

    args = parser.parse_args()

    data_path = args.data_path
    out_file = args.output_file

    text_files = glob.glob(data_path + '/*.txt')

    counter = 0

    with open(out_file, 'w') as outfile:
        for fname in text_files:
            counter += 1

            if counter % 1024 == 0:
                print("Merging at ", counter, flush=True)

            with open(fname, 'r') as infile:
                for row in infile:
                    tmp = {}
                    tmp['text'] = row
                    outfile.write(json.dumps(tmp))
                    outfile.write('\n')


    print("Merged file", out_file, flush=True)
```



```
> python3 Megatron-LM/tools/openwebtext/merge_data.py --data_path /workspace/code/scraped/data --output_file /workspace/data/merged_output.json
Merging at  1024
Merging at  2048
Merged file /workspace/data/merged_output.json
```

```
> head -n10 /workspace/data/merged_output.json
{"text": "With every new year, it's murder for Neal Smither and his crew.\n"}
{"text": "\n"}
{"text": "Suicide, too.\n"}
{"text": "\n"}
{"text": "As owner of Crime Scene Cleaners, Smither's job is to clean up the bloody messes left behind when people kill each other or themselves - and those first few weeks after Jan. 1 are his busiest time of year.\n"}
{"text": "\n"}
{"text": "All that holiday frivolity and togetherness may sound good in songs and movies, and a lot of people do indeed get mighty joyful - but experts say there is also a dark flip side of sadness, rage and depression that flares between Thanksgiving and post-New Year's.\n"}
{"text": "\n"}
{"text": "Most people hold their feelings together during the run-up to the new year, but once the holiday letdown sets it in, calls to suicide hot lines nearly double and homicides hit their highest rate of the year. Police officers, crisis counselors and people like Smither put in some long days and nights.\n"}
{"text": "\n"}
```


## 数据预处理

###  清洗数据

执行一下cleanup_dataset.py来把tokens数量少于128的文本都删掉。
```

cd Megatron-LM/tools/openwebtext/
python3 cleanup_dataset.py /workspace/data/merged_output.json /workspace/data/merged_cleand.json
```




执行 ftfy、英语检测并删除少于 128 个标记的文档。 此步骤可以分片并在分片上运行。

```
> python3 cleanup_dataset.py /workspace/data/merged_output.json /workspace/data/merged_cleand.json
building gpt2 dataset ...
will be reading /workspace/data/merged_output.json
and will write the results to /workspace/data/merged_cleand.json
 > filtering /workspace/data/merged_output.json
100%|████████████████████████████████████████████████████████████████████████████████████████████| 1042301/1042301 [00:02<00:00, 457699.45B/s]
100%|██████████████████████████████████████████████████████████████████████████████████████████████| 456318/456318 [00:03<00:00, 131743.88B/s]
> GPT2 tokenizer with 50257 vocab size and eod token 50256 ...
[small document, skipping]: {'text': "With every new year, it's murder for Neal Smither and his crew.\n"}
    skipping  {"text": "\n"}
 No features in text.
[non-english text] {'text': 'Suicide, too.\n'}
    skipping  {"text": "\n"}
 No features in text.
[non-english text] {'text': 'Pin\n'}
    skipping  {"text": "\n"}
 No features in text.
[non-english text] {'text': 'Tweet'}
[FINAL] | elapsed time: 156.46 | documents: 78802 | fixed text: 7891 | non-english: 4536 | non-english chars: 410257 | small docs: 30227 | small docs chars: 5717430
```


```
wc -l merged_output.json 
78802 merged_output.json


wc -l merged_cleand.json 
2456 merged_cleand.json
```


可以使用 cleanup_fix_dataset.py 完成其他清理（例如，删除少于 512 个字符的文档或特定于数据集的清理，如故事、真实新闻数据集）。 可以通过运行 python cleanup_fix_dataset.py --help 找到更多详细信息。 


2. 使用 LSH 查找可能的重复项并将其存储在文件中以供以后处理。 该代码支持保存和加载指纹（fingerprints）以进行重复数据删除，并且还支持多线程以加快处理速度。 更多详细信息可以通过 python find_duplicates.py --help 找到。

```
python find_duplicates.py --inputs /workspace/data/merged_cleand.json merged_cleand_id  --output /workspace/data/output_possible_duplicate_urls
```





### 数据预处理

```
python tools/preprocess_data.py \
       --input /workspace/data/train_data.json \
       --output-prefix /workspace/data/my-gpt2 \
       --vocab-file /workspace/model/gpt2-vocab/gpt2-vocab.json\
       --dataset-impl mmap \
       --tokenizer-type GPT2BPETokenizer \
       --merge-file /workspace/model/gpt2-vocab/gpt2-merges.txt \
       --append-eod \
       --workers 20 \
       --chunk-size 25

```

```
python tools/preprocess_data.py \
>        --input /workspace/data/train_data.json \
>        --output-prefix /workspace/data/my-gpt2 \
>        --vocab-file /workspace/model/gpt2-vocab/gpt2-vocab.json\
>        --dataset-impl mmap \
>        --tokenizer-type GPT2BPETokenizer \
>        --merge-file /workspace/model/gpt2-vocab/gpt2-merges.txt \
>        --append-eod \
>        --workers 20 \
>        --chunk-size 25
Opening /workspace/data/train_data.json
> building GPT2BPETokenizer tokenizer ...
 > padded vocab (size: 50257) with 47 dummy tokens (new size: 50304)
> building GPT2BPETokenizer tokenizer ...
> building GPT2BPETokenizer tokenizer ...
> building GPT2BPETokenizer tokenizer ...
> building GPT2BPETokenizer tokenizer ...
> building GPT2BPETokenizer tokenizer ...
> building GPT2BPETokenizer tokenizer ...
> building GPT2BPETokenizer tokenizer ...
> building GPT2BPETokenizer tokenizer ...
> building GPT2BPETokenizer tokenizer ...
> building GPT2BPETokenizer tokenizer ...
> building GPT2BPETokenizer tokenizer ...
 > padded vocab (size: 50257) with 47 dummy tokens (new size: 50304)
 > padded vocab (size: 50257) with 47 dummy tokens (new size: 50304)
> building GPT2BPETokenizer tokenizer ...
 > padded vocab (size: 50257) with 47 dummy tokens (new size: 50304)
> building GPT2BPETokenizer tokenizer ...
> building GPT2BPETokenizer tokenizer ...
 > padded vocab (size: 50257) with 47 dummy tokens (new size: 50304)
> building GPT2BPETokenizer tokenizer ...
 > padded vocab (size: 50257) with 47 dummy tokens (new size: 50304)
 > padded vocab (size: 50257) with 47 dummy tokens (new size: 50304)
> building GPT2BPETokenizer tokenizer ...
 > padded vocab (size: 50257) with 47 dummy tokens (new size: 50304)
> building GPT2BPETokenizer tokenizer ...
 > padded vocab (size: 50257) with 47 dummy tokens (new size: 50304)
> building GPT2BPETokenizer tokenizer ...
 > padded vocab (size: 50257) with 47 dummy tokens (new size: 50304)
> building GPT2BPETokenizer tokenizer ...
 > padded vocab (size: 50257) with 47 dummy tokens (new size: 50304)
Vocab size: 50257
Output prefix: /workspace/data/my-gpt2
> building GPT2BPETokenizer tokenizer ...
 > padded vocab (size: 50257) with 47 dummy tokens (new size: 50304)
Time to startup: 0.30323338508605957
 > padded vocab (size: 50257) with 47 dummy tokens (new size: 50304)
 > padded vocab (size: 50257) with 47 dummy tokens (new size: 50304)
 > padded vocab (size: 50257) with 47 dummy tokens (new size: 50304)
 > padded vocab (size: 50257) with 47 dummy tokens (new size: 50304)
 > padded vocab (size: 50257) with 47 dummy tokens (new size: 50304)
 > padded vocab (size: 50257) with 47 dummy tokens (new size: 50304)
 > padded vocab (size: 50257) with 47 dummy tokens (new size: 50304)
 > padded vocab (size: 50257) with 47 dummy tokens (new size: 50304)
 > padded vocab (size: 50257) with 47 dummy tokens (new size: 50304)
Processed 100 documents (599.5023090743549 docs/s, 0.5155917905295792 MB/s).
Processed 200 documents (823.8866599553122 docs/s, 0.880650182925332 MB/s).
Processed 300 documents (1185.4008987366815 docs/s, 1.200331609341586 MB/s).
Processed 400 documents (1533.1571463558173 docs/s, 1.6700295716317035 MB/s).
Processed 500 documents (1870.0427840484517 docs/s, 1.9969574995140202 MB/s).
Processed 600 documents (2187.427117123695 docs/s, 2.3795959237741595 MB/s).
Processed 700 documents (2152.781409317778 docs/s, 2.606610975177864 MB/s).
Processed 800 documents (2387.7960244711426 docs/s, 2.818861025855241 MB/s).
Processed 900 documents (2608.6323398315217 docs/s, 3.0303661882286277 MB/s).
Processed 1000 documents (2838.3195442776887 docs/s, 3.2176144414924934 MB/s).
Processed 1100 documents (3016.585243380653 docs/s, 3.4106741220407777 MB/s).
Processed 1200 documents (3236.0855769694435 docs/s, 3.7339411377043383 MB/s).
Processed 1300 documents (3475.8398763827417 docs/s, 3.935396723301203 MB/s).
Processed 1400 documents (3684.5816116836872 docs/s, 4.222750560810705 MB/s).
Processed 1500 documents (3880.979581765468 docs/s, 4.3926025538214795 MB/s).
Processed 1600 documents (3418.7934424893274 docs/s, 3.8987457589126513 MB/s).
Processed 1700 documents (3574.7781279422925 docs/s, 4.042231645357657 MB/s).
Processed 1800 documents (3738.2444121277663 docs/s, 4.194224887440415 MB/s).
Processed 1900 documents (3886.3553392670174 docs/s, 4.393378761729153 MB/s).
Processed 2000 documents (4044.6050768939995 docs/s, 4.54716435296795 MB/s).
Processed 2100 documents (4221.569306909339 docs/s, 4.692880733997719 MB/s).
Processed 2200 documents (4398.378779031662 docs/s, 4.835040337476316 MB/s).
Processed 2300 documents (4568.910999317048 docs/s, 5.0245360709512354 MB/s).
Processed 2400 documents (4740.355416123744 docs/s, 5.179960433973664 MB/s).
Done! Now finalizing.
```



## 模型训练

example/pretrain_gpt.sh 脚本使用单卡 GPU 运行 345M 参数的 GPT 模型进行预训练。 如上所述，单 GPU 训练主要用于调试目的，因为代码针对分布式训练进行了优化。

它遵循与之前的 BERT 脚本基本相同的格式，但有一些显着的差异：使用的tokenization方案是 BPE（需要merge表和 json 词汇文件）而不是 WordPiece，模型架构允许更长的序列（请注意， 最大位置嵌入必须大于或等于最大序列长度），并且 --lr-decay-style 已设置为余弦衰减。 请注意，--data-path 现在包含在预处理中添加的附加 _text_document 后缀，但不包含文件扩展名。

源文件 `arguments.py` 中描述了更多命令行参数。

`example/pretrain_gpt.sh` 可以按照与 BERT 描述相同的方式启动。 设置环境变量并进行任何其他修改，使用适当的安装启动容器，然后运行脚本。


```

```






