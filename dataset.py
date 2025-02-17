import os

NB_DOC_MAX = 1000 
IMDB_CLASSES  = ['neg','pos']
MAX_CHAR_SIZE = 1000

labels = dict(zip(IMDB_CLASSES,[0,1]))

def load_data_film(datapath, classes=IMDB_CLASSES, max_size=NB_DOC_MAX):
    txts = []
    files = []
    filelabels = []
    for label in classes:
        c = 0
        new = [os.path.join(datapath / label, f) for f in os.listdir(datapath / label) if f.endswith(".txt")]
        files += new
        # filelabels += [labels[label]] * len(new) 
        for file in (datapath / label).glob("*.txt"):
            t = file.read_text()
            txts.append(t if len(t)<MAX_CHAR_SIZE else t[:MAX_CHAR_SIZE])
            filelabels.append(labels[label])
            c+=1
            if max_size !=None and c>=max_size: break

    return txts, files, filelabels
    #     c+=1
    #     if train_max_size !=None and c>train_max_size: break

