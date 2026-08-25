import numpy as np

data = {
    "full": {
        "demand": [0.00403, 0.01603, 0.00600, 0.00000, 0.00426],
        "equity": [0.2842, 0.5344, 0.6740, 0.2000, 0.3882],
        "coverage": [0.1570, 0.1712, 0.1807, 0.0809, 0.1486],
    },
    "single_objective": {
        "demand": [0.00621, 0.00161, 0.00000, 0.00000, 0.00015],
        "equity": [0.6769, 0.3706, 1.0000, 1.0000, 0.2000],
        "coverage": [0.1605, 0.1379, 0.0416, 0.0523, 0.1165],
    },
    "flat_encoder": {
        "demand": [0.00658, 0.00050, 0.00705, 0.00000, 0.00008],
        "equity": [0.4446, 0.2431, 0.5106, 0.2000, 0.2000],
        "coverage": [0.1605, 0.0630, 0.1712, 0.0809, 0.1177],
    },
}

for method, metrics in data.items():
    print(f"== {method} ==")
    for metric, vals in metrics.items():
        arr = np.array(vals)
        if metric == "demand":
            arr = arr * 100
        mean = arr.mean()
        std = arr.std(ddof=1)
        print(f"  {metric}: {mean:.3f} +/- {std:.3f}  (raw: {arr.tolist()})")
