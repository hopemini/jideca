import torch
from torchvision import datasets, transforms
import pickle
import os

def DataPreProcessing(data_path, batch_size ,types, num_classes):
    ## Set path
    data_path = data_path + ("activity" if types == "real" else types)
    data_train_path = data_path + "/image/all_2"
    data_test_path = data_path + "/image/test_" + str(num_classes)

    if not os.path.exists("./data/"):
        os.mkdir("./data/")
    train_save_path = "./data/train_" + ("re" if types == "real" else "se") + ".pkl"
    test_save_path = "./data/test_" + ("re" if types == "real" else "se") + "_" + str(num_classes) + ".pkl"

    ## Set data loader
    train_dataset = datasets.ImageFolder(root=data_train_path,
                               transform=transforms.Compose([
                                   transforms.Resize((256,128)),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5),
                                                        (0.5, 0.5, 0.5)),
                               ]))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=8)

    test_dataset = datasets.ImageFolder(root=data_test_path,
                               transform=transforms.Compose([
                                   transforms.Resize((256,128)),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5),
                                                        (0.5, 0.5, 0.5)),
                               ]))
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=8)

    with open(train_save_path, 'wb') as f:
        pickle.dump(train_loader, f)

    with open(test_save_path, 'wb') as f:
        pickle.dump(test_loader, f)

if __name__ == "__main__":
    data_path = "../data_processing/"
    batch_size = 2400
    num_classes = ["23", "34"]
    types = ["real", "semantic_annotations"]
    for t in types:
        for n in num_classes:
            DataPreProcessing(data_path, batch_size ,t, n)
