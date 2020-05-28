Solving the Cocktail Party problem with Catalyst
==============================================================================

We will be implementing the paper `Looking to Listen at the Cocktail Party.`_

<add intro to problem and paper>

.. _Looking to Listen at the Cocktail Party.: https://arxiv.org/abs/1804.03619

Downloading
------------------------------------------------

First step is to download the data. Dataset, also provided in the paper, is called AVSpeech. It contains YouTube link, timestamp of the video and x and y co-ordinate of the face present in the video. We will use all of this information one by one. So, first we need to download the video.

We will be using `youtube-dl.`_ We will also use concurrent.futures from python for multi-threading.
`Complete Source <https://github.com/vitrioil/Speech-Separation/blob/master/src/loader/download.py>`_.

.. _youtube-dl.: https://github.com/ytdl-org/youtube-dl

.. code-block:: python


        def download(link, path, final_name=None):
            command = "youtube-dl {} --output {}.mp4 -f 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4'"
            # check if file already exists
            ...
            p = subprocess.Popen(command.format(link, path), shell=True,
                                 stdout=subprocess.PIPE, cwd=args.vid_dir).communicate()


        def crop(path, start, end, resolution, downloaded_name):
            command = "ffmpeg -y -i {}.mp4 -ss {} -t {} -c:v libx264 -crf 18 -preset veryfast
                      -pix_fmt yuv420p -c:a aac -b:a 128k -strict experimental -r 25 {}"

            command = command.format(downloaded_name, f"{start_minute}:{start_second}",
                                     f"{end_minute}:{end_second}", new_filepath)
            p = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE).communicate()

        def save_video(zargs, resolution=None):
            link, path, start, end, pos_x, pos_y = zargs
            x = int(pos_x*10000)
            y = int(pos_y*10000)
            # store co-ord in file name, as multiple instances of the same video are present.
            downloaded_name = path.as_posix() + f"_{x}_{y}"
            download(link, downloaded_name, final_name=downloaded_name + "_final.mp4")

        def main(args):
            df = pd.read_csv(args.path)
            links = df.iloc[:, 0][args.start:args.end]
            start_times = df.iloc[:, 1][args.start:args.end]
            end_times = df.iloc[:, 2][args.start:args.end]
            pos_x = df.iloc[:, 3][args.start:args.end]
            pos_y = df.iloc[:, 4][args.start:args.end]

            yt_links = ["https://youtube.com/watch\?v\="+l for l in links]
            paths = [Path(os.path.join(args.vid_dir, f)) for f in links]

            link_path = zip(yt_links, paths, start_times, end_times, pos_x, pos_y)
            with concurrent.futures.ThreadPoolExecutor(args.jobs) as executor:
                results = list(tqdm.tqdm(executor.map(save_video, link_path), total=len(links)))

Great, now we have a download script that runs on multiple threads. However, YouTube has a weekly download/watch limit of 2000 videos per week.

Extracting Audio
------------------------------------------------
After downloading we need to extract faces as well as audio. Let's start with extracting audio.
Since, the network expects a signal with a duration of 3 seconds, we need to crop the audio as well, if it is more than 3 seconds long (which it is).

`Source <https://github.com/vitrioil/Speech-Separation/blob/master/src/loader/extract_audio.py>`_.

.. code-block:: python

        def extract(path):
            name = path.stem

            dir_name = path.parents[0]
            audio_dir = args.aud_dir
            audio_path = os.path.join(audio_dir, name)
            video = cv2.VideoCapture(path.as_posix())
            length_orig_video = video.get(cv2.CAP_PROP_FRAME_COUNT)

            #already pre-processed at 25 fps for 3 or more seconds
            length = int(length_orig_video) // 25 // 3
            for i in range(length):
                t = i*3
                command = (f"ffmpeg -y -i {path.as_posix()} -f {args.audio_extension} -ab 64000"
                           f"-vn -ar {args.sampling_rate} -ac {args.audio_channel} - |"
                           f"sox -t {args.audio_extension} - -r 16000 -c 1 -b 8 "
                           f"{audio_path}_part{i}.{args.audio_extension} trim {t} 00:{args.duration:02d}")

                p = subprocess.Popen(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        def main(args):
            file_names = [Path(os.path.join(args.vid_dir, i)) for i in os.listdir(args.vid_dir)\
                          if i.endswith("_final.mp4")]

            with concurrent.futures.ThreadPoolExecutor(args.jobs) as executor:
                results = list(tqdm(executor.map(extract, file_names), total=len(file_names)))

Mixing
------------------------------------------------
<explanation>

<code>

Face Embedding
------------------------------------------------
<explanation>

<code>

Train
------------------------------------------------
<explanation>

<code>

Inference
------------------------------------------------
<explanation>

<code>
