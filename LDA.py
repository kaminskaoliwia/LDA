import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

# Data exploration
df = pd.read_csv("data/BankNote_Authentication.csv")

print("Data shape: ", df.shape, end='\n\n')
print(df.head(), end='\n\n')
print(df["class"].value_counts(), end='\n\n')

# Feature ranking by Fisher criterion
features = ["variance", "skewness", "curtosis", "entropy"]

genuine = df[df["class"] == 0]
forged  = df[df["class"] == 1]

scores = {}
for f in features:
    mu0  = genuine[f].mean()
    mu1  = forged[f].mean()
    var0 = genuine[f].var()
    var1 = forged[f].var()
    scores[f] = (mu0 - mu1)**2 / (var0 + var1)

ranking = pd.Series(scores).sort_values(ascending=False)
print("Fisher criterion per feature:")
print(ranking.round(3), end='\n\n')

# Data visualisation
for feature in features:
    plt.hist(genuine[feature], bins=40, alpha=0.6, label="genuine")
    plt.hist(forged[feature],  bins=40, alpha=0.6, label="forged")
    plt.xlabel(feature)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"graphs/{feature}.png")
    plt.clf()

# Scatterplot variance v skewness

plt.scatter(df.loc[df["class"] == 0, "variance"],
            df.loc[df["class"] == 0, "skewness"],
            alpha=0.4, label="genuine")
plt.scatter(df.loc[df["class"] == 1, "variance"],
            df.loc[df["class"] == 1, "skewness"],
            alpha=0.4, label="forged")

plt.xlabel("variance")
plt.ylabel("skewness")
plt.legend()
# plt.show()
plt.savefig("graphs/variance_skewness_scatterplot.png")
plt.clf()

# Scatterplot variance v curtosis

plt.scatter(df.loc[df["class"] == 0, "variance"],
            df.loc[df["class"] == 0, "curtosis"],
            alpha=0.4, label="genuine")
plt.scatter(df.loc[df["class"] == 1, "variance"],
            df.loc[df["class"] == 1, "curtosis"],
            alpha=0.4, label="forged")

plt.xlabel("variance")
plt.ylabel("curtosis")
plt.legend()
# plt.show()
plt.savefig("graphs/variance_curtosis_scatterplot.png")
plt.clf()

# Scatterplot skewness v curtosis

plt.scatter(df.loc[df["class"] == 0, "skewness"],
            df.loc[df["class"] == 0, "curtosis"],
            alpha=0.4, label="genuine")
plt.scatter(df.loc[df["class"] == 1, "skewness"],
            df.loc[df["class"] == 1, "curtosis"],
            alpha=0.4, label="forged")

plt.xlabel("skewness")
plt.ylabel("curtosis")
plt.legend()
# plt.show()
plt.savefig("graphs/skewness_curtosis_scatterplot.png")
plt.clf()

# Accuracy of only variation
threshold_1d = (genuine["variance"].mean() + forged["variance"].mean()) / 2

predicted = (df["variance"] < threshold_1d).astype(int)
correct = (predicted == df["class"]).sum()
accuracy_1d = correct / len(df)

print(f"Threshold on variance alone: {threshold_1d:.3f}")
print(f"Accuracy with variance only: {accuracy_1d:.1%}", end='\n\n')

# S_W and S_B matrices
X = df[features].values
y = df["class"].values

X0 = X[y == 0]  # genuine
X1 = X[y == 1]  # forged

mu0 = X0.mean(axis=0)
mu1 = X1.mean(axis=0)

p = X.shape[1]  # number of features = 4

# Within-class scatter matrix S_W
SW = np.zeros((p, p))
for x in X0:
    d = (x - mu0).reshape(-1, 1)
    SW += d @ d.T # @ = matrix multiplication
for x in X1:
    d = (x - mu1).reshape(-1, 1)
    SW += d @ d.T

# Between-class scatter matrix S_B
mu = X.mean(axis=0)
d0 = (mu0 - mu).reshape(-1, 1)
d1 = (mu1 - mu).reshape(-1, 1)
SB = len(X0) * (d0 @ d0.T) + len(X1) * (d1 @ d1.T)

print("S_W:")
print(SW.round(2), end='\n\n')
print("\nS_B:")
print(SB.round(2), end='\n\n')

# Discriminant axis
# a = np.linalg.inv(SW) @ (mu0 - mu1)

criterion_matrix = np.linalg.inv(SW) @ SB
eigenvalues, eigenvectors = np.linalg.eig(criterion_matrix)

# Eigenvector with largest eigenvalue
idx = np.argmax(eigenvalues)
a = eigenvectors[:, idx]

print("Discriminant direction a:")
print(a.round(4), end='\n\n')

# Normalized discriminant axis
# a = a / np.linalg.norm(a)

# print("Normalized discriminant direction a:")
# print(a.round(4), end='\n\n')

# Projecting data on the discriminant axis
projections = X @ a

proj0 = projections[y == 0]  # genuine
proj1 = projections[y == 1]  # forged

# Classification threshold
threshold = (a @ mu0 + a @ mu1) / 2

print(f"Threshold: {threshold:.4f}")

predictions = (projections < threshold).astype(int)

correct = (predictions == y).sum()
accuracy = correct / len(y)

print(f"Accuracy: {accuracy:.1%}")

# Projection hist
plt.hist(proj0, bins=40, alpha=0.6, label="genuine")
plt.hist(proj1, bins=40, alpha=0.6, label="forged")
plt.axvline(threshold, color="black", linestyle="--", label=f"threshold = {threshold:.2f}")
plt.xlabel("projection onto discriminant axis")
plt.ylabel("count")
plt.title("Data projected onto Fisher discriminant axis")
plt.legend()
# plt.show()
plt.savefig("graphs/projection_hist.png")
plt.clf()

# Verification with sklearn
lda = LinearDiscriminantAnalysis()
lda.fit(X, y)
y_pred = lda.predict(X)

print(f"sklearn accuracy: {(y_pred == y).mean():.1%}", end='\n\n')

cm = confusion_matrix(y, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=["genuine", "forged"])
disp.plot()
plt.title("Confusion matrix - sklearn LDA")
plt.savefig("graphs/confusion_matrix.png")
#plt.show()

print(classification_report(y, y_pred, target_names=["genuine", "forged"]), end='\n\n')
