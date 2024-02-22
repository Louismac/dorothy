import os
import wget
import sys

def download_pretrained_rave(rave_model_dir = "pretrained-models"):
    if not os.path.exists(rave_model_dir): # create the folder if it doesn't exist
        os.mkdir(rave_model_dir)
    pretrained_models = ["vintage", "percussion", "VCTK"] # list of available pretrained_models to download in https://acids-ircam.github.io/rave_models_download (you can select less if you want to spend less time on this cell)

    def bar_progress(current, total, width=80): # progress bar for wget
        progress_message = "Downloading: %d%% [%d / %d] bytes" % (current / total * 100, current, total)
        # Don't use print() as it will print in new line every time.
        sys.stdout.write("\r" + progress_message)
        sys.stdout.flush()

    for model in pretrained_models: # download pretrained models and save them in pt_path
        if not os.path.exists(os.path.join(rave_model_dir, f"{model}.ts")): # only download if not already downloaded
            print(f"Downloading {model}.ts...")
            wget.download(f"https://play.forum.ircam.fr/rave-vst-api/get_model/{model}",f"{rave_model_dir}/{model}.ts", bar=bar_progress)
        else:
            print(f"{model}.ts already downloaded")