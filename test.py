
from ultralytics.utils.benchmarks import benchmark

result=benchmark(
    model="WEC-UOD/runs/detect/   /weights/best.pt",
    data="data.yaml",
    imgsz=640,
    device=0,
    batchsize=1)

# )
# from ultralytics.utils.benchmarks import benchmark
#
# result = benchmark(
#     model="",
#     data="data.yaml",
#     imgsz=640,
#     device=0
# )
print(result)