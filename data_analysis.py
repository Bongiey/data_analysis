import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
import numpy as np

# Set seaborn style for better visualization
sns.set_style("whitegrid")

def load_iris_data():
    """Load Iris dataset and convert to pandas DataFrame"""
    try:
        iris = load_iris()
        df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
        df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def explore_dataset(df):
    """Explore dataset structure and handle missing values"""
    print("First 5 rows of the dataset:")
    print(df.head())
    print("\nDataset Info:")
    print(df.info())
    
    # Check for missing values
    print("\nMissing Values:")
    print(df.isnull().sum())
    
    # If missing values exist, fill with mean for numerical columns
    for column in df.select_dtypes(include=['float64']).columns:
        if df[column].isnull().any():
            df[column].fillna(df[column].mean(), inplace=True)
            print(f"Filled missing values in {column} with mean")
    
    return df

def analyze_data(df):
    """Perform basic statistical analysis"""
    print("\nBasic Statistics:")
    print(df.describe())
    
    # Group by species and calculate mean for numerical columns
    print("\nMean values by species:")
    print(df.groupby('species').mean())
    
    # Additional observation
    print("\nObservation: The mean measurements vary significantly across species, "
          "suggesting distinct characteristics for each iris type.")

def create_visualizations(df):
    """Create four different types of visualizations"""
    
    # 1. Line Chart: Mean measurements across species
    plt.figure(figsize=(10, 6))
    for column in df.select_dtypes(include=['float64']).columns:
        plt.plot(df['species'].unique(), 
                df.groupby('species')[column].mean(), 
                marker='o', 
                label=column)
    plt.title('Mean Measurements Across Iris Species')
    plt.xlabel('Species')
    plt.ylabel('Measurement (cm)')
    plt.legend()
    plt.savefig('line_chart.png')
    plt.close()
    
    # 2. Bar Chart: Average sepal length by species
    plt.figure(figsize=(8, 6))
    sns.barplot(x='species', y='sepal length (cm)', data=df)
    plt.title('Average Sepal Length by Species')
    plt.xlabel('Species')
    plt.ylabel('Sepal Length (cm)')
    plt.savefig('bar_chart.png')
    plt.close()
    
    # 3. Histogram: Distribution of petal length
    plt.figure(figsize=(8, 6))
    plt.hist(df['petal length (cm)'], bins=20, color='skyblue', edgecolor='black')
    plt.title('Distribution of Petal Length')
    plt.xlabel('Petal Length (cm)')
    plt.ylabel('Frequency')
    plt.savefig('histogram.png')
    plt.close()
    
    # 4. Scatter Plot: Sepal length vs Petal length
    plt.figure(figsize=(8, 6))
    for species in df['species'].unique():
        species_data = df[df['species'] == species]
        plt.scatter(species_data['sepal length (cm)'], 
                   species_data['petal length (cm)'], 
                   label=species, 
                   alpha=0.6)
    plt.title('Sepal Length vs Petal Length')
    plt.xlabel('Sepal Length (cm)')
    plt.ylabel('Petal Length (cm)')
    plt.legend()
    plt.savefig('scatter_plot.png')
    plt.close()

def main():
    # Load data
    df = load_iris_data()
    if df is None:
        return
    
    # Explore and clean data
    df = explore_dataset(df)
    
    # Analyze data
    analyze_data(df)
    
    # Create visualizations
    create_visualizations(df)
    
    print("\nAnalysis complete. Visualizations saved as PNG files.")
    print("Findings:")
    print("- Setosa species has significantly smaller petal measurements compared to Versicolor and Virginica.")
    print("- There is a clear separation in petal length distribution, which could be useful for classification.")
    print("- Sepal length and petal length show a positive correlation, especially within species.")

if __name__ == "__main__":
    main()