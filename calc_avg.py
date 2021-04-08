import os

total_1 = 0
total_2 = 0
count = 0
e = 0
with open("./iccv_geopifu_color_simulated_multiview_quantitative.txt") as f:
    lines_1 = f.readlines()
with open("./iccv_geopifu_color_quantitative.txt") as f:
    lines_2 = f.readlines()

for idx, line in enumerate(lines_1):
    # if "," not in line:
    #     continue
    val_1 = float(lines_1[idx].strip("\n").split(" ")[-1].strip(","))
    val_2 = float(lines_2[idx].strip("\n").split(" ")[-1].strip(","))
    print(lines_1[idx], lines_2[idx])
    count += 1
    total_1 += val_1
    total_2 += val_2

print("Average: ", total_1/count)
print("Average: ", total_2/count)
print("Count: ", count)
