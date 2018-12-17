from utils.processor import Processor
import glob

# for file_path in glob.glob("data/*/npys/"):
#     pcr = Processor(file_path,crop=False, overwrite=True)
#     pcr.compact()
#     print("finished", file_path)

for file_path in glob.glob("data/ell1/npys/"):
    pcr = Processor(file_path,crop=False, overwrite=True)
    pcr.fuse()
    print("finished", file_path)
print("finished all")
