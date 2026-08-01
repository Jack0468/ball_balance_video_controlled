import glob

files = glob.glob("host_software/**/*.py", recursive=True)
count = 0
for f in files:
    try:
        with open(f, "r", encoding="utf-8") as file:
            content = file.read()
    except Exception as e:
        continue

    if "MAX_BOUND" not in content:
        continue

    print(f"Patching {f}...")
    # Replace the constant definition
    content = content.replace(
        "MAX_BOUND = 200.0 # From ball_dataset.py",
        "MAX_X_BOUND, MAX_Y_BOUND = 93.75, 71.0 # True physical plate bounds",
    )
    content = content.replace(
        "MAX_BOUND = 200.0 # Denormalization constant",
        "MAX_X_BOUND, MAX_Y_BOUND = 93.75, 71.0 # True physical plate bounds",
    )
    content = content.replace(
        "MAX_BOUND         = 200.0   # ResNet denormalisation constant (must match BallDataset)",
        "MAX_X_BOUND, MAX_Y_BOUND = 93.75, 71.0 # True physical plate bounds",
    )
    content = content.replace(
        "MAX_BOUND = 200.0 ", "MAX_X_BOUND, MAX_Y_BOUND = 93.75, 71.0 "
    )
    content = content.replace(
        "MAX_BOUND = 200.0", "MAX_X_BOUND, MAX_Y_BOUND = 93.75, 71.0"
    )

    # Replace numpy array multiplication
    content = content.replace("* MAX_BOUND", "* np.array([MAX_X_BOUND, MAX_Y_BOUND])")

    # Replace scalar multiplication
    content = content.replace(
        "norm_x * np.array([MAX_X_BOUND, MAX_Y_BOUND])", "norm_x * MAX_X_BOUND"
    )
    content = content.replace(
        "norm_y * np.array([MAX_X_BOUND, MAX_Y_BOUND])", "norm_y * MAX_Y_BOUND"
    )

    # Replace tensor divisions
    content = content.replace(
        "touch_x / np.array([MAX_X_BOUND, MAX_Y_BOUND])", "touch_x / MAX_X_BOUND"
    )
    content = content.replace(
        "touch_y / np.array([MAX_X_BOUND, MAX_Y_BOUND])", "touch_y / MAX_Y_BOUND"
    )

    with open(f, "w", encoding="utf-8") as file:
        file.write(content)
    count += 1

print(f"Patched {count} files.")
