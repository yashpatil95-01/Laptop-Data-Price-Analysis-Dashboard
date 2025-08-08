```
laptop-pricing-analysis-dashboard/
│
├── dashboard_style_laptop_pricing_analysis.py
├── results/
│   ├── correlation_matrix.png
│   ├── scatter_plots.png
│   ├── boxplots.png
│   └── grouped_heatmap.png
│
├── requirements.txt
└── README.md
```

# Laptop Pricing Analysis Dashboard

This project explores and analyzes a laptop pricing dataset using Python, with a focus on visualization and statistical relationships between key features and price.

##  Key Tasks

- Clean and preprocess the dataset
- Map encoded values (e.g., GPU, OS, Category) to readable labels
- Visualize numeric and categorical feature relationships with price
- Compute and display Pearson correlation scores
- Create heatmaps and boxplots for feature comparison
- Save all plots for use in presentations or dashboards

##  Output Files

All analysis plots are saved in the `results/` directory:
- `correlation_matrix.png`: Heatmap of feature correlations
- `scatter_plots.png`: Regression plots for numeric features vs Price
- `boxplots.png`: Boxplots of price distribution by categorical features
- `grouped_heatmap.png`: Average price by GPU and CPU core count

##  Requirements

To run the script, install the following packages:

```bash
pip install -r requirements.txt
```

 Dataset
Dataset is loaded from the following URL:
```bash
https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101E
```

📂 File Descriptions
dashboard_style_laptop_pricing_analysis.py: Main script containing all data processing and plotting logic

requirements.txt: Python dependencies

results/: Folder storing all generated plots

 Notes
This is part of a hands-on data analysis project using real-world-style datasets. It's designed to help practice:

Exploratory Data Analysis (EDA)

Data visualization

Feature encoding and mapping

Correlation analysis

 Author
Yash Patil — Manufacturing Engineer pivoting into data and product analytics

---

##  `requirements.txt`

Include this so others can set up the environment easily:

```txt
numpy
pandas
matplotlib
seaborn
scipy
```

