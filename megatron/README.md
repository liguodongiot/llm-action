


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




