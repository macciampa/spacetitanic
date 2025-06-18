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

def create_destination_homeplanet_heatmap(df):
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
    plt.savefig('data_out/destination_homeplanet_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Destination/Home Planet heatmap saved to data_out/destination_homeplanet_heatmap.png")
    
    # Print summary statistics
    print("\nDestination/Home Planet Transport Statistics:")
    print(pivot_data.sort_values('transport_percentage', ascending=False)) 