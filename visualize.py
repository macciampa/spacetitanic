import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

def create_deck_side_heatmap(df, stats=False):
    """Create a heatmap showing transport percentage for deck/side combinations"""
    # Create pivot table for transport percentage
    pivot_data = df.groupby(['Deck', 'Side'])['Transported'].agg(['count', 'sum']).reset_index()
    pivot_data['transport_percentage'] = (pivot_data['sum'] / pivot_data['count'] * 100).round(1)
    
    # Create pivot table for heatmap
    heatmap_data = pivot_data.pivot(index='Deck', columns='Side', values='transport_percentage')
    
    # Create pivot table for counts
    count_data = pivot_data.pivot(index='Deck', columns='Side', values='count')
    
    # Create the heatmap
    plt.figure(figsize=(10, 8))
    
    # Create custom annotations with percentage and count
    annot_data = heatmap_data.copy()
    for deck in heatmap_data.index:
        for side in heatmap_data.columns:
            if pd.notna(heatmap_data.loc[deck, side]) and pd.notna(count_data.loc[deck, side]):
                percentage = heatmap_data.loc[deck, side]
                count = int(count_data.loc[deck, side])
                annot_data.loc[deck, side] = f"{percentage}% ({count})"
            else:
                annot_data.loc[deck, side] = ""
    
    sns.heatmap(heatmap_data, annot=annot_data, fmt='', cmap='RdYlBu_r', 
                cbar_kws={'label': 'Transport Percentage (%)'})
    plt.title('Transport Percentage by Deck and Side', fontsize=16, pad=20)
    plt.xlabel('Side', fontsize=12)
    plt.ylabel('Deck', fontsize=12)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig('data_out/visualizations/deck_side_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Deck/Side heatmap saved to data_out/visualizations/deck_side_heatmap.png")
    
    if stats:
        print("\nDeck/Side Transport Statistics:")
        print(pivot_data.sort_values('transport_percentage', ascending=False))

def create_age_survival_histogram(df, stats=False):
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
    plt.savefig('data_out/visualizations/age_survival_histogram.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Age survival histogram saved to data_out/visualizations/age_survival_histogram.png")
    
    if stats:
        print("\nAge Survival Statistics (ages with 5+ passengers):")
        filtered_stats = age_survival[age_survival['count'] >= 5].sort_values('survival_percentage', ascending=False)
        print(filtered_stats[['Age', 'count', 'survival_percentage']])

def create_destination_homeplanet_heatmap(df, stats=False):
    """Create a heatmap showing transport percentage for destination/home planet combinations"""
    # Create pivot table for transport percentage
    pivot_data = df.groupby(['Destination', 'HomePlanet'])['Transported'].agg(['count', 'sum']).reset_index()
    pivot_data['transport_percentage'] = (pivot_data['sum'] / pivot_data['count'] * 100).round(1)
    
    # Create pivot table for heatmap
    heatmap_data = pivot_data.pivot(index='Destination', columns='HomePlanet', values='transport_percentage')
    
    # Create pivot table for counts
    count_data = pivot_data.pivot(index='Destination', columns='HomePlanet', values='count')
    
    # Create the heatmap
    plt.figure(figsize=(12, 8))
    
    # Create custom annotations with percentage and count
    annot_data = heatmap_data.copy()
    for destination in heatmap_data.index:
        for planet in heatmap_data.columns:
            if pd.notna(heatmap_data.loc[destination, planet]) and pd.notna(count_data.loc[destination, planet]):
                percentage = heatmap_data.loc[destination, planet]
                count = int(count_data.loc[destination, planet])
                annot_data.loc[destination, planet] = f"{percentage}% ({count})"
            else:
                annot_data.loc[destination, planet] = ""
    
    sns.heatmap(heatmap_data, annot=annot_data, fmt='', cmap='RdYlBu_r', 
                cbar_kws={'label': 'Transport Percentage (%)'})
    plt.title('Transport Percentage by Destination and Home Planet', fontsize=16, pad=20)
    plt.xlabel('Home Planet', fontsize=12)
    plt.ylabel('Destination', fontsize=12)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig('data_out/visualizations/destination_homeplanet_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Destination/Home Planet heatmap saved to data_out/visualizations/destination_homeplanet_heatmap.png")
    
    if stats:
        print("\nDestination/Home Planet Transport Statistics:")
        print(pivot_data.sort_values('transport_percentage', ascending=False))

def create_spending_survival_histogram(df, stats=False):
    """Create a histogram showing spending distribution and survival percentage by spending bins"""
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: TotalSpent distribution
    ax1.hist(df['TotalSpent'].dropna(), bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
    ax1.set_title('Total Spending Distribution', fontsize=14, pad=10)
    ax1.set_xlabel('Total Amount Spent')
    ax1.set_ylabel('Count')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Survival percentage by spending bins
    # Create spending bins
    spending_bins = [0, 0.1, 25, 50, 100, 200, 300, 500, 750, 1000, 1500, 2000, 3000, 5000, 7500, 10000, 15000, 20000, 30000, 50000, 75000, 100000, float('inf')]
    spending_labels = ['0', '1-25', '26-50', '51-100', '101-200', '201-300', '301-500', '501-750', '751-1K', '1K-1.5K', '1.5K-2K', '2K-3K', '3K-5K', '5K-7.5K', '7.5K-10K', '10K-15K', '15K-20K', '20K-30K', '30K-50K', '50K-75K', '75K-100K', '100K+']
    
    df['SpendingGroup'] = pd.cut(df['TotalSpent'], bins=spending_bins, labels=spending_labels, include_lowest=True)
    spending_survival = df.groupby('SpendingGroup')['Transported'].agg(['count', 'sum']).reset_index()
    spending_survival['survival_percentage'] = (spending_survival['sum'] / spending_survival['count'] * 100).round(1)
    
    # Create bar plot
    bars = ax2.bar(range(len(spending_survival)), spending_survival['survival_percentage'], 
                   color='gold', alpha=0.7, edgecolor='black')
    ax2.set_title('Survival Percentage by Spending Group', fontsize=14, pad=10)
    ax2.set_xlabel('Spending Group')
    ax2.set_ylabel('Survival Percentage (%)')
    ax2.set_xticks(range(len(spending_survival)))
    ax2.set_xticklabels(spending_survival['SpendingGroup'], rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, percentage, count in zip(bars, spending_survival['survival_percentage'], spending_survival['count']):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{percentage}%', ha='center', va='bottom', fontweight='bold', fontsize=9)
        ax2.text(bar.get_x() + bar.get_width()/2., height/2,
                f'n={count}', ha='center', va='center', fontweight='bold', color='white', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('data_out/visualizations/spending_survival_histogram.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Spending survival histogram saved to data_out/visualizations/spending_survival_histogram.png")
    
    if stats:
        print("\nSpending Group Survival Statistics:")
        print(spending_survival[['SpendingGroup', 'count', 'survival_percentage']].sort_values('survival_percentage', ascending=False))

def create_age_transported_distribution(df):
    """Create a histogram showing age distribution by transport status"""
    plt.figure(figsize=(16,6))
    sns.histplot(x=df["Age"], hue="Transported", data=df, kde=True, palette="Set2")
    plt.title("Age Feature Distribution by Transport Status", fontsize=16, pad=20)
    plt.xlabel("Age", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.legend(title="Transported", labels=["No", "Yes"])
    plt.tight_layout()
    plt.savefig('data_out/visualizations/age_transported_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Age transported distribution saved to data_out/visualizations/age_transported_distribution.png")

def create_cabin_num_survival_histogram(df, stats=False):
    """Create a histogram showing cabin number distribution and survival percentage by cabin number"""
    # Convert Num to numeric, handle missing values
    df['Num'] = pd.to_numeric(df['Num'], errors='coerce')
    
    # Determine bin edges with size 100
    min_num = int(df['Num'].min())
    max_num = int(df['Num'].max())
    bin_edges = list(range(min_num, max_num + 101, 100))
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: Cabin number distribution (bins of size 100)
    ax1.hist(df['Num'].dropna(), bins=bin_edges, alpha=0.7, color='slateblue', edgecolor='black')
    ax1.set_title('Cabin Number Distribution', fontsize=14, pad=10)
    ax1.set_xlabel('Cabin Number')
    ax1.set_ylabel('Count')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Survival percentage by cabin number bins (bins of size 100)
    df['NumBin'] = pd.cut(df['Num'], bins=bin_edges)
    num_survival = df.groupby('NumBin')['Transported'].agg(['count', 'sum']).reset_index()
    num_survival['survival_percentage'] = (num_survival['sum'] / num_survival['count'] * 100).round(1)
    
    # Create bar plot
    bars = ax2.bar(range(len(num_survival)), num_survival['survival_percentage'], 
                   color='mediumseagreen', alpha=0.7, edgecolor='black')
    ax2.set_title('Survival Percentage by Cabin Number Bin', fontsize=14, pad=10)
    ax2.set_xlabel('Cabin Number Bin')
    ax2.set_ylabel('Survival Percentage (%)')
    ax2.set_xticks(range(len(num_survival)))
    ax2.set_xticklabels([str(b) for b in num_survival['NumBin']], rotation=45, ha='right', fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, percentage, count in zip(bars, num_survival['survival_percentage'], num_survival['count']):
        height = bar.get_height()
        if not np.isnan(percentage):
            ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{percentage}%', ha='center', va='bottom', fontweight='bold', fontsize=8)
            ax2.text(bar.get_x() + bar.get_width()/2., height/2,
                    f'n={int(count)}', ha='center', va='center', fontweight='bold', color='white', fontsize=7)
    
    plt.tight_layout()
    os.makedirs('data_out/visualizations', exist_ok=True)
    plt.savefig('data_out/visualizations/cabin_num_survival_histogram.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Cabin number survival histogram saved to data_out/visualizations/cabin_num_survival_histogram.png")
    
    if stats:
        print("\nCabin Number Survival Statistics:")
        print(num_survival[['NumBin', 'count', 'survival_percentage']].sort_values('survival_percentage', ascending=False))

# Ensure the visualizations directory exists
os.makedirs('data_out/visualizations', exist_ok=True) 