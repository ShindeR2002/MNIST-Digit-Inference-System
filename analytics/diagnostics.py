import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE

def generate_reports(y_true, y_pred, features):
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10,7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.savefig('images/confusion_matrix.png')
    
    # t-SNE Manifold
    tsne = TSNE(n_components=2)
    reduced = tsne.fit_transform(features[:500])
    plt.scatter(reduced[:,0], reduced[:,1], c=y_true[:500], cmap='tab10')
    plt.savefig('images/tsne_plot.png')