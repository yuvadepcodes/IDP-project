with open("class_names.txt", "w") as f:
    for c in train_data.class_names:
        f.write(c + "\n")