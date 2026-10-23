import sys
print("Python Executable:", sys.executable)
try:
    import matplotlib
    print("matplotlib:", matplotlib.__version__)
except Exception as e:
    print("matplotlib error:", e)

try:
    import seaborn
    print("seaborn:", seaborn.__version__)
except Exception as e:
    print("seaborn error:", e)

try:
    import torch
    print("torch:", torch.__version__)
except Exception as e:
    print("torch error:", e)

try:
    import sklearn
    print("sklearn:", sklearn.__version__)
except Exception as e:
    print("sklearn error:", e)

try:
    import pandas
    print("pandas:", pandas.__version__)
except Exception as e:
    print("pandas error:", e)

try:
    import reportlab
    print("reportlab:", reportlab.__version__)
except Exception as e:
    print("reportlab error:", e)
