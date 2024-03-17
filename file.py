import os

# Specify the directory you want to start from
rootDir = "logs/pneumoniamnist/active-softmax/checkpoints"

for dirName, subdirList, fileList in os.walk(rootDir):
    for fname in fileList:
        label, num, label2 = fname.split("_")
        newfname = label + "-" + num + "_" + label2
        src = os.path.join(dirName, fname)
        dst = os.path.join(dirName, newfname)

        os.rename(src, dst)

        # if fname.endswith('.ckpt') and '_250_best' in fname:
        #     src = os.path.join(dirName, fname)
        #     dst = os.path.join(dirName, fname.replace('_250_best', '-250_best'))
        #     os.rename(src, dst)
