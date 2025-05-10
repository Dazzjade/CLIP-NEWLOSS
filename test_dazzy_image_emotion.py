import os
import clip
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset.ImageEmotion import ImageEmotion


# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
# model, process = clip.load('ViT-B/32',device)
model = torch.load("/home/dazzy/CLIP/models/CLIP-SMP_newloss_TEST.pt")

# Load the dataset
root = os.path.expanduser("~/.cache")
train = ImageEmotion(split='train', target='test')
test = ImageEmotion(split='test', target='test')


# train = MELD(split='train',video_len=1,sampling_strategy='random')
# test = MELD(split='test',video_len=1,sampling_strategy='random')


def get_features(dataset):
    all_features = []
    all_labels = []
    i = 0
    with torch.no_grad():
        for images, labels in tqdm(DataLoader(dataset, batch_size=100)):
            features = model.encode_image(images.to(device))

            all_features.append(features)
            all_labels.append(labels)

    return torch.cat(all_features).cpu().numpy(), torch.cat(all_labels).cpu().numpy()

# Calculate the image features
train_features, train_labels = get_features(train)
test_features, test_labels = get_features(test)

print(test_features.shape)
print(test_labels.shape)

# Perform logistic regression
classifier = LogisticRegression(random_state=2020, C=8, max_iter=2000, solver='sag',class_weight=None)
print(f"11111111111111111111111111111111")

classifier.fit(train_features, train_labels)
print("22222222222222222222222222222")
p_feature = classifier.predict_proba(test_features)
p_labels = np.argmax(p_feature,axis = 1)
print("333333333333333333333333333333")
# Evaluate using the logistic regression classifier
# predictions = classifier.predict(test_features)
# accuracy = np.mean((test_labels == predictions).astype(float)) * 100.
acc = accuracy_score(p_labels, test_labels) * 100
print(f"Accuracy = {acc:.3f}")