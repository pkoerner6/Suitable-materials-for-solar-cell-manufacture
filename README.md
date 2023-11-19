# Suitable materials for solar cell manufacture

This project is centered around identifying promising materials for organic solar cells, with a specific focus on predicting the HOMO-LUMO gap. The HOMO-LUMO gap serves as a crucial indicator of molecular efficiency in utilizing solar energy—the larger the gap, the more efficient the molecule is for solar cells. Unlike traditional density functional theory, this project leverages machine learning techniques to predict the HOMO-LUMO gap from molecular descriptions.

The primary goal is to develop a machine learning model capable of predicting the HOMO-LUMO gap based on molecular features. This approach offers a more efficient alternative to conventional density functional theory methods.

## Data 
- Small Dataset: Consists of 100 molecules with associated HOMO-LUMO gaps.
- Large Dataset: Comprises 50,000 molecules with labeled LUMO energy levels.

## Transfer Learning
The project adopts a transfer learning approach, utilizing the large dataset with LUMO energy levels as a valuable resource. Although the labels are not the direct prediction target, there's a presumed correlation between features predictive of LUMO energy and the HOMO-LUMO gap. Features learned from the LUMO energy task are applied to enhance predictions for the HOMO-LUMO gap.
