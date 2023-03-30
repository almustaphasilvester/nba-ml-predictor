# NBA ML Predictor

The NBA ML Predictor is a machine learning model that uses player and team data to predict the outcomes of NBA games. This repository contains all the code and data needed to train and use the model.

# Getting Started

To use the NBA ML Predictor, you will need to have Python 3 and several Python libraries installed on your computer. These libraries include:

- pandas
- numpy
- scikit-learn
- matplotlib
- tensorflow

You can install these libraries using pip, the Python package manager. Here's an example command to install pandas:

```python
pip install pandas
```

Once you have all the required libraries installed, you can clone this repository to your computer and navigate to the project directory:

```bash
git clone https://github.com/almustaphasilvester/nba-ml-predictor.git
cd nba-ml-predictor
```
# Data

The data for this project comes from the official NBA stats website. The data directory contains several parquet files with player and team statistics for multiple NBA season.

# Model

The NBA ML Predictor uses a Multi-Input Tensorflow classifier to predict the outcomes of NBA games. The model is trained using the [NBA_Model.ipynb](https://github.com/almustaphasilvester/nba-ml-predictor/blob/main/NBA_Model.ipynb) script, which reads in the player and team data then processes and trains the player data.

# Usage

To use the NBA ML Predictor to make predictions on NBA games, run the predict.py script and provide the team names and date of the game as command-line arguments. For example:

```bash
python predict.py "Golden State Warriors" "Cleveland Cavaliers" "2018-06-03"
This will output the predicted winner of the game, along with the probability of that team winning.
```

# Conclusion

The NBA ML Predictor is a fun project that demonstrates how machine learning can be used to predict the outcomes of sporting events. While the model is not perfect, it is a good starting point for anyone interested in exploring the intersection of sports and data science. If you have any questions or feedback, feel free to contact the author of this repository.
