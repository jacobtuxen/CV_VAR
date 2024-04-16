from SoccerNet.Downloader import SoccerNetDownloader as SNdl
from pathlib import Path
mySNdl = SNdl(LocalDirectory=f"/work3/s194572/SoccerData")
mySNdl.downloadDataTask(task="mvfouls", split=["train","valid","test","challenge"], password="s0cc3rn3t")