import numpy as np

# Configuration
train_samples = 5
test_samples = 2
sample_shape = (20, 20, 1)  # Height x Width x Channels
num_classes = 2

# Générer les données fictives
train_data = np.random.rand(train_samples, *sample_shape)
test_data = np.random.rand(test_samples, *sample_shape)

# Générer les classes fictives
train_classes = np.random.randint(0, num_classes, train_samples)
test_classes = np.random.randint(0, num_classes, test_samples)

# Convertir les classes en one-hot
train_classes_one_hot = np.eye(num_classes)[train_classes]
test_classes_one_hot = np.eye(num_classes)[test_classes]

# Fonction pour écrire les données dans un fichier
def write_flatten_data(filename, data):
    with open(filename, "w") as f:
        for sample in data:
            flattened = sample.flatten()
            f.write(" ".join(map(str, flattened)) + "\n")

def write_classes(filename, classes):
    with open(filename, "w") as f:
        for class_vec in classes:
            f.write(" ".join(map(str, class_vec.astype(int))) + "\n")

# Sauvegarder les fichiers
write_flatten_data("Test/trainData.txt", train_data)
write_flatten_data("Test/testData.txt", test_data)
write_classes("Test/trainClass.txt", train_classes_one_hot)
write_classes("Test/testClass.txt", test_classes_one_hot)

train_data.shape, test_data.shape, train_classes_one_hot, test_classes_one_hot
