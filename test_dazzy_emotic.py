import os
import clip
import torch
from dataset.emotic import Emotic
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader
from tqdm import tqdm

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = torch.load("/home/dazzy/CLIP/models/CLIP-SMP_newloss_TEST.pt")
# model, preprocess = clip.load('ViT-B/32', device)


# Load the dataset
root = os.path.expanduser("~/.cache")
train = Emotic(split='train', target='test')
test = Emotic(split='test', target='test')

print(len(train))

def get_features(dataset):
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(DataLoader(dataset, batch_size=100)):
            features = model.encode_image(images.to(device))

            all_features.append(features)
            all_labels.append(labels)

    return torch.cat(all_features).cpu().numpy(), torch.cat(all_labels).cpu().numpy()

# Calculate the image features
train_features, train_labels = get_features(train)
test_features, test_labels = get_features(test)


# Perform logistic regression
classifier = LogisticRegression(random_state=2020, C=2.5, max_iter=10000, solver='sag',class_weight=None)
classifier = OneVsRestClassifier(classifier,n_jobs=-1)
print(f"11111111111111111111111111111111")

classifier.fit(train_features, train_labels)
print("22222222222222222222222222222")
p_labels = classifier.predict_proba(test_features)
print(p_labels)
print("333333333333333333333333333333")
# Evaluate using the logistic regression classifier
# predictions = classifier.predict(test_features)
# accuracy = np.mean((test_labels == predictions).astype(float)) * 100.
mAP = average_precision_score(test_labels, p_labels, average='macro') * 100
print(f"Accuracy = {mAP:.3f}")