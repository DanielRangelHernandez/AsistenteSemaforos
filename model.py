from roboflow import Roboflow
rf = Roboflow(api_key="k5et03bqsLBvFs2YySJL")
project = rf.workspace("roboflow-100").project("road-signs-6ih4y")
version = project.version(2)
dataset = version.download("yolov8")
                