import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def create_deck_side_heatmap(df):
    """Create a heatmap showing transport percentage for deck/side combinations"""
    # Create pivot table for transport percentage
    pivot_data = df.groupby(['Deck', 'Side'])['Transported'].agg(['count', 'sum']).reset_index()
    pivot_data['transport_percentage'] = (pivot_data['sum'] / pivot_data['count'] * 100).round(1)
    
    # Create pivot table for heatmap
    heatmap_data = pivot_data.pivot(index='Deck', columns='Side', values='transport_percentage')
    
    # Create the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='RdYlBu_r', 
                cbar_kws={'label': 'Transport Percentage (%)'})
    plt.title('Transport Percentage by Deck and Side', fontsize=16, pad=20)
    plt.xlabel('Side', fontsize=12)
    plt.ylabel('Deck', fontsize=12)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig('data_out/deck_side_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Deck/Side heatmap saved to data_out/deck_side_heatmap.png")
    
    # Print summary statistics
    print("\nDeck/Side Transport Statistics:")
    print(pivot_data.sort_values('transport_percentage', ascending=False))

def create_age_survival_histogram(df):
    """Create a histogram showing age distribution and survival percentage"""
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Age distribution
    ax1.hist(df['Age'].dropna(), bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.set_title('Age Distribution', fontsize=14, pad=10)
    ax1.set_xlabel('Age')
    ax1.set_ylabel('Count')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Survival percentage by individual age
    age_survival = df.groupby('Age')['Transported'].agg(['count', 'sum']).reset_index()
    age_survival['survival_percentage'] = (age_survival['sum'] / age_survival['count'] * 100).round(1)
    
    # Create bar plot
    bars = ax2.bar(age_survival['Age'], age_survival['survival_percentage'], 
                   color='lightcoral', alpha=0.7, edgecolor='black')
    ax2.set_title('Survival Percentage by Age', fontsize=14, pad=10)
    ax2.set_xlabel('Age')
    ax2.set_ylabel('Survival Percentage (%)')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars (only for ages with sufficient data)
    for bar, percentage, count in zip(bars, age_survival['survival_percentage'], age_survival['count']):
        height = bar.get_height()
        if count >= 5:  # Only show labels for ages with 5+ passengers
            ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{percentage}%', ha='center', va='bottom', fontweight='bold', fontsize=8)
            ax2.text(bar.get_x() + bar.get_width()/2., height/2,
                    f'n={count}', ha='center', va='center', fontweight='bold', color='white', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('data_out/age_survival_histogram.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Age survival histogram saved to data_out/age_survival_histogram.png")
    
    # Print age statistics (only for ages with sufficient data)
    print("\nAge Survival Statistics (ages with 5+ passengers):")
    filtered_stats = age_survival[age_survival['count'] >= 5].sort_values('survival_percentage', ascending=False)
    print(filtered_stats[['Age', 'count', 'survival_percentage']]) 