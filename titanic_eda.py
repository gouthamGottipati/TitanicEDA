import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)


print("=" * 50)
print("LOADING DATA")
print("=" * 50)
df = pd.read_csv('titanic.csv')
print(f"\nDataset shape: {df.shape}")
print(f"\nFirst 5 rows:")
print(df.head())


print("\n" + "=" * 50)
print("INITIAL DATA EXPLORATION")
print("=" * 50)
print("\nData types and non-null counts:")
df.info()

print("\nSummary statistics:")
print(df.describe())


print("\n" + "=" * 50)
print("MISSING VALUES ANALYSIS")
print("=" * 50)


missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100
missing_df = pd.DataFrame({
    'Missing Count': missing,
    'Percentage': missing_pct
}).sort_values('Missing Count', ascending=False)

print("\nMissing values summary:")
print(missing_df[missing_df['Missing Count'] > 0])


plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=False, cmap='YlOrRd', yticklabels=False)
plt.title('Missing Data Heatmap (Yellow = Missing)', fontsize=14)
plt.tight_layout()
plt.savefig('1_missing_data_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()


print("\n" + "=" * 50)
print("OUTLIER DETECTION")
print("=" * 50)

numerical_cols = ['Age', 'Fare', 'SibSp', 'Parch']

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.ravel()

outlier_summary = {}

for idx, col in enumerate(numerical_cols):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Count outliers
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
    outlier_summary[col] = len(outliers)

    # Create boxplot
    axes[idx].boxplot(df[col].dropna(), vert=True)
    axes[idx].set_title(f'{col} - Outliers: {len(outliers)}', fontsize=12)
    axes[idx].set_ylabel(col)
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('2_outlier_boxplots.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nOutlier counts (using IQR method):")
for col, count in outlier_summary.items():
    print(f"  {col}: {count} outliers")


print("\n" + "=" * 50)
print("DATA CLEANING")
print("=" * 50)


df_clean = df.copy()


print("\nHandling missing values:")


print("  - Filling Age with median by Pclass and Sex")
df_clean['Age'] = df_clean.groupby(['Pclass', 'Sex'])['Age'].transform(
    lambda x: x.fillna(x.median())
)


print("  - Filling Embarked with mode")
df_clean['Embarked'].fillna(df_clean['Embarked'].mode()[0], inplace=True)


print("  - Creating Cabin_Known indicator")
df_clean['Cabin_Known'] = df_clean['Cabin'].notna().astype(int)

df_clean.drop('Cabin', axis=1, inplace=True)


print("\nHandling outliers:")
fare_99th = df_clean['Fare'].quantile(0.99)
print(f"  - Capping Fare at 99th percentile: {fare_99th:.2f}")
df_clean['Fare'] = df_clean['Fare'].clip(upper=fare_99th)

print("\nCleaned dataset shape:", df_clean.shape)
print("Remaining missing values:")
print(df_clean.isnull().sum().sum())


print("\n" + "=" * 50)
print("VISUALIZATION WITH CLEANED DATA")
print("=" * 50)

# Survival distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Survival count
survival_counts = df_clean['Survived'].value_counts()
axes[0].bar(['Not Survived', 'Survived'], survival_counts.values,
            color=['#ff6b6b', '#51cf66'])
axes[0].set_title('Survival Count', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Count')
axes[0].grid(axis='y', alpha=0.3)

# Survival percentage
survival_pct = df_clean['Survived'].value_counts(normalize=True) * 100
axes[1].pie(survival_pct.values, labels=['Not Survived', 'Survived'],
            autopct='%1.1f%%', colors=['#ff6b6b', '#51cf66'], startangle=90)
axes[1].set_title('Survival Percentage', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('3_survival_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\nSurvival rate: {survival_pct[1]:.2f}%")

# Survival by categorical features
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# By Gender
sns.countplot(data=df_clean, x='Sex', hue='Survived', ax=axes[0, 0],
              palette=['#ff6b6b', '#51cf66'])
axes[0, 0].set_title('Survival by Gender', fontsize=12, fontweight='bold')
axes[0, 0].legend(title='Survived', labels=['No', 'Yes'])

# By Passenger Class
sns.countplot(data=df_clean, x='Pclass', hue='Survived', ax=axes[0, 1],
              palette=['#ff6b6b', '#51cf66'])
axes[0, 1].set_title('Survival by Passenger Class', fontsize=12, fontweight='bold')
axes[0, 1].legend(title='Survived', labels=['No', 'Yes'])

# By Embarked Port
sns.countplot(data=df_clean, x='Embarked', hue='Survived', ax=axes[1, 0],
              palette=['#ff6b6b', '#51cf66'])
axes[1, 0].set_title('Survival by Embarkation Port', fontsize=12, fontweight='bold')
axes[1, 0].legend(title='Survived', labels=['No', 'Yes'])

# By Cabin Known
sns.countplot(data=df_clean, x='Cabin_Known', hue='Survived', ax=axes[1, 1],
              palette=['#ff6b6b', '#51cf66'])
axes[1, 1].set_title('Survival by Cabin Info Available', fontsize=12, fontweight='bold')
axes[1, 1].set_xticklabels(['No Cabin Info', 'Has Cabin Info'])
axes[1, 1].legend(title='Survived', labels=['No', 'Yes'])

plt.tight_layout()
plt.savefig('4_survival_by_categories.png', dpi=300, bbox_inches='tight')
plt.show()

# Age distribution by survival
plt.figure(figsize=(10, 6))
df_clean[df_clean['Survived'] == 1]['Age'].plot(kind='kde', label='Survived',
                                                  color='#51cf66', linewidth=2)
df_clean[df_clean['Survived'] == 0]['Age'].plot(kind='kde', label='Not Survived',
                                                  color='#ff6b6b', linewidth=2)
plt.title('Age Distribution by Survival', fontsize=14, fontweight='bold')
plt.xlabel('Age')
plt.ylabel('Density')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('5_age_distribution_survival.png', dpi=300, bbox_inches='tight')
plt.show()

# Correlation matrix
plt.figure(figsize=(10, 8))

numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
corr_matrix = df_clean[numeric_cols].corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, square=True, linewidths=1)
plt.title('Correlation Matrix (Cleaned Data)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('6_correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.show()


print("\n" + "=" * 50)
print("FINAL SUMMARY")
print("=" * 50)

print("\nSurvival rates by key features:")
print("\nBy Gender:")
print(df_clean.groupby('Sex')['Survived'].agg(['sum', 'count', 'mean']))

print("\nBy Passenger Class:")
print(df_clean.groupby('Pclass')['Survived'].agg(['sum', 'count', 'mean']))

print("\nBy Embarkation Port:")
print(df_clean.groupby('Embarked')['Survived'].agg(['sum', 'count', 'mean']))

# Save cleaned dataset
df_clean.to_csv('titanic_cleaned.csv', index=False)
print("\n✓ Cleaned dataset saved as 'titanic_cleaned.csv'")
print("✓ All visualizations saved successfully!")
