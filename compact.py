from utils.processor import Processor
import glob

#for file_path in glob.glob("train/five/open3/npys/"):
#    pcr = Processor(file_path,crop=False, overwrite=True)
#    pcr.compact()
#    print("finished", file_path)
for file_path in glob.glob("train/*/*/npys/"):
    pcr = Processor(file_path,crop=False, overwrite=True)
    pcr.fuse()
    print("finished", file_path)


for file_path in glob.glob("test/*/*/npys/"):
   pcr = Processor(file_path,crop=False, overwrite=True)
   pcr.fuse()
   print("finished", file_path)
print("finished all")
