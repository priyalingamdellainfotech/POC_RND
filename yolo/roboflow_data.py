
from roboflow import Roboflow



rf = Roboflow(api_key="Y3bBLX100940M3YvpLKA", model_format="Testdata", notebook="ultralytics")
rf.workspace().project("Testdata").version(0).download("yolov5")

# rf = Roboflow(model_format="Testdata", notebook="ultralytics", api_key="Y3bBLX100940M3YvpLKA")
#
# project = rf.workspace().project("Testdata")
# dataset = project.version("").download("yolov5")
print()