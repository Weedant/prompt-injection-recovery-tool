
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Setup paths
ROOT = Path(__file__).resolve().parents[1]
IMG_DIR = ROOT / "docs" / "images"
IMG_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['figure.dpi'] = 150

def generate_dataset_size_chart():
    """V1 vs V2 dataset size comparison."""
    data = {
        'Version': ['Old (v1)', 'New (v2)'],
        'Total Rows': [65416, 358617]
    }
    df = pd.DataFrame(data)
    
    plt.figure(figsize=(8, 5))
    ax = sns.barplot(x='Version', y='Total Rows', data=df, palette=['#7f8c8d', '#2ecc71'])
    
    plt.title('Dataset Scale Expansion', fontsize=14, fontweight='bold', pad=20)
    plt.ylabel('Thousands of Rows')
    
    # Add number labels on top
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height()):,}', 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha='center', va='center', fontsize=11, fontweight='bold',
                   xytext=(0, 9), textcoords='offset points')
    
    plt.tight_layout()
    plt.savefig(IMG_DIR / "dataset_scale.png")
    print(f"Saved: {IMG_DIR / 'dataset_scale.png'}")

def generate_label_distribution():
    """Pie charts for the three main datasets."""
    files = {
        "Prefilter": "prefilter_merged.csv",
        "Harmful Intent": "harmful_intent_merged.csv",
        "Sandbox Behavior": "sandbox_behavior.csv"
    }
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    colors = ['#3498db', '#e74c3c'] 
    
    for i, (name, fname) in enumerate(files.items()):
        path = ROOT / "data" / "raw" / fname
        if not path.exists():
            continue
            
        df = pd.read_csv(path)
        dist = df['label'].value_counts().sort_index()
        
        labels = ['Benign/Safe', 'Malicious/Compromised']
        if name == "Sandbox Behavior":
            labels = ['Refusal', 'Compliance']
            
        axes[i].pie(dist, labels=labels, autopct='%1.1f%%', 
                  startangle=140, colors=colors, explode=(0.05, 0),
                  textprops={'fontsize': 10, 'fontweight': 'bold'})
        axes[i].set_title(f'{name} Distribution\n(n={len(df):,})', fontsize=12, fontweight='bold')

    plt.suptitle('Class Distribution Across Pipeline Stages', fontsize=16, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.savefig(IMG_DIR / "class_distributions.png")
    print(f"Saved: {IMG_DIR / 'class_distributions.png'}")

def generate_model_performance():
    """Bar chart for ROC-AUC scores."""
    # Data from recent training logs
    results = {
        'Stage': ['Step 2 Prefilter', 'Harmful Intent', 'Step 3 Sandbox (Est)'],
        'ROC-AUC': [0.9859, 0.9257, 0.85] # Placeholder for sandbox
    }
    df = pd.DataFrame(results)
    
    plt.figure(figsize=(9, 6))
    ax = sns.barplot(x='Stage', y='ROC-AUC', data=df, palette='magma')
    
    plt.ylim(0.7, 1.0) # Zoom in on the top performing area
    plt.title('Detection Performance (ROC-AUC)', fontsize=14, fontweight='bold', pad=20)
    
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.2%}', 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha = 'center', va = 'center', xytext = (0, 9), 
                   textcoords = 'offset points', fontweight='bold')

    plt.tight_layout()
    plt.savefig(IMG_DIR / "model_performance.png")
    print(f"Saved: {IMG_DIR / 'model_performance.png'}")

if __name__ == "__main__":
    print("\n[viz] Generating professional documentation plots...")
    generate_dataset_size_chart()
    generate_label_distribution()
    generate_model_performance()
    print("[viz] Success! Images saved in docs/images/")
